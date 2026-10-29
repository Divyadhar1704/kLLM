"""Plant photo and video diagnosis with a vision LLM, constrained and calibrated.

    photo -> validation + privacy cleaning + quality check
    video -> validation + best frames (app.vision.video), all sent in ONE model call
          -> (poor quality: ask for a retake, no AI call)
          -> Gemini vision, structured JSON restricted to the fixed crop/condition list
          -> calibration rules -> confident | uncertain | not_a_plant | retake_photo

A general vision model's self-reported confidence is not trusted as is: it is lowered when the
top two candidates are close, when the condition belongs to a different crop than the one seen,
and when the photo quality is poor. A dedicated classifier trained on field images is planned as
a second opinion (see project roadmap); this module is the first stage.
"""

from __future__ import annotations

import asyncio
import functools
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from google.genai import types
from pydantic import BaseModel, Field, ValidationError, field_validator

from app.core import metrics
from app.core.config import Settings
from app.core.logging import get_logger
from app.services.google_client import create_genai_client
from app.services.llm.errors import LLMError, LLMResponseError
from app.services.llm.gemini import translate_error
from app.vision.images import ImageQuality, PreparedImage, prepare_image
from app.vision.taxonomy import (
    CONDITION_IDS,
    CONDITIONS,
    CROP_IDS,
    CROPS,
    UNKNOWN,
    Condition,
    condition,
)
from app.vision.video import PreparedVideo, VideoInfo, prepare_video

logger = get_logger(__name__)

Status = Literal["confident", "uncertain", "not_a_plant", "retake_photo"]
Severity = Literal["none", "mild", "moderate", "severe", "unknown"]
PlantPart = Literal["leaf", "fruit", "stem", "flower", "root", "whole_plant", "other"]

_PLANT_PARTS: tuple[str, ...] = ("leaf", "fruit", "stem", "flower", "root", "whole_plant", "other")
_SEVERITIES: tuple[str, ...] = ("none", "mild", "moderate", "severe", "unknown")
_QUALITIES: tuple[str, ...] = ("good", "acceptable", "poor")

# Calibration rules (applied to the model's own confidence).
CLOSE_CALL_MARGIN = 0.15
CLOSE_CALL_CAP = 0.55
CROP_MISMATCH_FACTOR = 0.6
POOR_PHOTO_CAP = 0.5
QUALITY_WARNING_FACTOR = 0.9

SYSTEM_PROMPT = """You are an expert plant pathologist helping farmers in India.
Examine the photo and identify the crop and the most likely condition.

Rules:
- Choose ids ONLY from the allowed lists. Use "unknown" if the photo is not a plant, the crop or
  problem cannot be seen, or you are unsure. Never guess a crop that is not visible.
- Give up to 3 candidate conditions, best first. Confidence = your honest probability (0 to 1)
  that the condition is correct; the confidences of all candidates should sum to at most 1.
- Conditions must belong to the crop you see (or be "nutrient_deficiency", "water_stress",
  "unknown"). Choose "<crop>_healthy" only if no symptom is visible.
- visible_symptoms: short English phrases describing what you actually see (max 5).
- The farmer's note, if any, may name the crop or describe the problem; use it as a hint, but
  the photo decides."""

GENERIC_CONDITIONS: tuple[str, ...] = tuple(c.id for c in CONDITIONS.values() if c.crop == "any")


def allowed_ids(crop: str | None) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """(crop ids, condition ids) the model may return. A known crop narrows both: the most
    common error is misidentifying the crop (ginger read as banana), so when the farmer or the
    app knows the crop, only that crop's conditions are allowed."""
    if crop in CROPS:
        own = tuple(c.id for c in CONDITIONS.values() if c.crop == crop)
        return (crop, UNKNOWN), (*own, *GENERIC_CONDITIONS)
    return CROP_IDS, CONDITION_IDS


def allowed_list_prompt(crop: str | None) -> str:
    crops, conditions = allowed_ids(crop)
    lines = [f"Allowed crops: {', '.join(crops)}", f"Allowed conditions: {', '.join(conditions)}"]
    if crop in CROPS:
        lines.insert(0, f"The farmer says this is {CROPS[crop]}. Diagnose it as that crop.")
    return "\n".join(lines)


@functools.cache
def response_schema(crop: str | None = None) -> types.Schema:
    crop_ids, condition_ids = allowed_ids(crop)
    string = types.Type.STRING
    return types.Schema(
        type=types.Type.OBJECT,
        properties={
            "is_plant": types.Schema(type=types.Type.BOOLEAN),
            "crop": types.Schema(type=string, enum=list(crop_ids)),
            "plant_part": types.Schema(type=string, enum=list(_PLANT_PARTS)),
            "image_quality": types.Schema(type=string, enum=list(_QUALITIES)),
            "candidates": types.Schema(
                type=types.Type.ARRAY,
                max_items=3,
                min_items=1,
                items=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "condition_id": types.Schema(type=string, enum=list(condition_ids)),
                        "confidence": types.Schema(type=types.Type.NUMBER),
                    },
                    required=["condition_id", "confidence"],
                ),
            ),
            "severity": types.Schema(type=string, enum=list(_SEVERITIES)),
            "visible_symptoms": types.Schema(
                type=types.Type.ARRAY, max_items=5, items=types.Schema(type=string)
            ),
        },
        required=[
            "is_plant",
            "crop",
            "plant_part",
            "image_quality",
            "candidates",
            "severity",
            "visible_symptoms",
        ],
        property_ordering=[
            "is_plant",
            "crop",
            "plant_part",
            "image_quality",
            "visible_symptoms",
            "candidates",
            "severity",
        ],
    )


class VisionCandidate(BaseModel):
    condition_id: str
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("condition_id")
    @classmethod
    def _known(cls, value: str) -> str:
        return value if value in CONDITIONS else UNKNOWN

    @field_validator("confidence", mode="before")
    @classmethod
    def _clamp(cls, value: Any) -> float:
        return min(1.0, max(0.0, float(value)))


class VisionOutput(BaseModel):
    is_plant: bool
    crop: str
    plant_part: str = "other"
    image_quality: str = "acceptable"
    candidates: list[VisionCandidate] = Field(min_length=1)
    severity: str = "unknown"
    visible_symptoms: list[str] = []

    @field_validator("crop")
    @classmethod
    def _known_crop(cls, value: str) -> str:
        return value if value in CROPS else UNKNOWN


@dataclass(frozen=True, slots=True)
class Alternative:
    condition: Condition
    confidence: float


@dataclass(slots=True)
class Diagnosis:
    status: Status
    crop: str
    condition: Condition
    confidence: float  # after calibration rules
    model_confidence: float  # as reported by the model
    severity: str
    plant_part: str
    visible_symptoms: list[str]
    alternatives: list[Alternative]
    quality: ImageQuality
    reasons: list[str] = field(default_factory=list)  # why confidence was lowered
    model: str | None = None
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    video: VideoInfo | None = None  # set for video diagnoses


def decide(
    output: VisionOutput,
    quality: ImageQuality,
    *,
    min_confidence: float,
    crop_confirmed: bool,
) -> tuple[Status, Condition, float, list[str]]:
    """Calibration rules. Returns (status, condition, calibrated confidence, reasons).

    Measured on field photos (evals, Oct 2026): without a confirmed crop the model's diagnosis
    was right <= 53% of the time at ANY confidence threshold (it misidentifies the crop), so it
    is never reported as confident. With the crop confirmed, confidence >= 0.8 was right 77%.
    """
    candidates = sorted(output.candidates, key=lambda c: c.confidence, reverse=True)
    top = candidates[0]
    chosen = condition(top.condition_id)
    if not output.is_plant:
        return "not_a_plant", condition(UNKNOWN), 0.0, ["no plant visible"]

    confidence = top.confidence
    reasons: list[str] = []
    second = candidates[1].confidence if len(candidates) > 1 else 0.0
    if top.confidence - second < CLOSE_CALL_MARGIN and len(candidates) > 1:
        confidence = min(confidence, CLOSE_CALL_CAP)
        reasons.append("top candidates are close")
    if chosen.crop not in ("any", output.crop):
        confidence *= CROP_MISMATCH_FACTOR
        reasons.append(f"condition is for {chosen.crop}, photo looks like {output.crop}")
    if output.image_quality == "poor":
        confidence = min(confidence, POOR_PHOTO_CAP)
        reasons.append("model rated the photo poor")
    if quality.issues:
        confidence *= QUALITY_WARNING_FACTOR
        reasons.append("photo quality: " + ", ".join(quality.issues))
    confidence = round(confidence, 3)

    if chosen.id == UNKNOWN or output.crop == UNKNOWN:
        return "uncertain", chosen, confidence, [*reasons, "model could not identify"]
    if not crop_confirmed:
        return "uncertain", chosen, confidence, [*reasons, "crop not confirmed by the farmer"]
    status: Status = "confident" if confidence >= min_confidence else "uncertain"
    return status, chosen, confidence, reasons


class VisionDiagnoser:
    def __init__(
        self,
        *,
        client: Any,
        models: Sequence[str],
        timeout_seconds: float,
        max_bytes: int,
        model_max_side: int,
        min_confidence: float,
        reject_poor_quality: bool,
        video_max_bytes: int = 50 * 1024 * 1024,
        video_max_seconds: float = 60,
        video_frames: int = 4,
        video_candidate_frames: int = 16,
        video_max_concurrent: int = 2,
    ) -> None:
        self._client = client
        self._models = [m for m in models if m]
        self._timeout = timeout_seconds
        self._max_bytes = max_bytes
        self._max_side = model_max_side
        self._min_confidence = min_confidence
        self._reject_poor_quality = reject_poor_quality
        self._video_max_bytes = video_max_bytes
        self._video_max_seconds = video_max_seconds
        self._video_frames = video_frames
        self._video_candidate_frames = video_candidate_frames
        # Video decoding is CPU-heavy: bound how many uploads decode at once per instance.
        self._video_slots = asyncio.Semaphore(video_max_concurrent)

    @classmethod
    def from_settings(cls, settings: Settings) -> VisionDiagnoser:
        models = [settings.vision_model]
        if (
            settings.vision_fallback_model
            and settings.vision_fallback_model != settings.vision_model
        ):
            models.append(settings.vision_fallback_model)
        return cls(
            client=create_genai_client(settings, timeout_seconds=settings.vision_timeout_seconds),
            models=models,
            timeout_seconds=settings.vision_timeout_seconds,
            max_bytes=settings.max_image_bytes,
            model_max_side=settings.image_model_max_side,
            min_confidence=settings.image_min_confidence,
            reject_poor_quality=settings.image_reject_poor_quality,
            video_max_bytes=settings.max_video_bytes,
            video_max_seconds=settings.max_video_seconds,
            video_frames=settings.video_model_frames,
            video_candidate_frames=settings.video_candidate_frames,
            video_max_concurrent=settings.video_max_concurrent,
        )

    @property
    def model(self) -> str:
        return self._models[0]

    async def prepare(self, data: bytes) -> PreparedImage:
        return await asyncio.to_thread(
            prepare_image, data, max_bytes=self._max_bytes, max_side=self._max_side
        )

    async def diagnose(
        self, data: bytes, *, note: str | None = None, crop: str | None = None
    ) -> Diagnosis:
        image = await self.prepare(data)
        return await self.diagnose_prepared(image, note=note, crop=crop)

    async def prepare_video(self, data: bytes) -> PreparedVideo:
        async with self._video_slots:
            return await asyncio.to_thread(
                prepare_video,
                data,
                max_bytes=self._video_max_bytes,
                max_seconds=self._video_max_seconds,
                frames=self._video_frames,
                candidate_frames=self._video_candidate_frames,
                max_side=self._max_side,
            )

    async def diagnose_video(
        self, data: bytes, *, note: str | None = None, crop: str | None = None
    ) -> Diagnosis:
        video = await self.prepare_video(data)
        return await self.diagnose_prepared_video(video, note=note, crop=crop)

    async def diagnose_prepared(
        self, image: PreparedImage, *, note: str | None = None, crop: str | None = None
    ) -> Diagnosis:
        if self._reject_poor_quality and image.quality.severe:
            return self._retake(image.quality, source="image")
        return await self._diagnose(
            [image], image.quality, note=note, crop=crop, original_bytes=image.original_bytes
        )

    async def diagnose_prepared_video(
        self, video: PreparedVideo, *, note: str | None = None, crop: str | None = None
    ) -> Diagnosis:
        if not video.frames:  # every frame too dark / blurry / small
            return self._retake(video.quality, source="video", video=video.info)
        return await self._diagnose(
            video.images,
            video.quality,
            note=note,
            crop=crop,
            original_bytes=video.original_bytes,
            video=video.info,
        )

    def _retake(
        self, quality: ImageQuality, *, source: str, video: VideoInfo | None = None
    ) -> Diagnosis:
        metrics.diagnoses.labels("video" if source == "video" else "photo", "retake_photo").inc()
        logger.info("image_rejected_quality", source=source, issues=quality.issues)
        return Diagnosis(
            status="retake_photo",
            crop=UNKNOWN,
            condition=condition(UNKNOWN),
            confidence=0.0,
            model_confidence=0.0,
            severity="unknown",
            plant_part="other",
            visible_symptoms=[],
            alternatives=[],
            quality=quality,
            reasons=["photo quality: " + ", ".join(quality.issues)]
            if quality.issues
            else ["no usable frame"],
            video=video,
        )

    async def _diagnose(
        self,
        images: Sequence[PreparedImage],
        quality: ImageQuality,
        *,
        note: str | None,
        crop: str | None,
        original_bytes: int,
        video: VideoInfo | None = None,
    ) -> Diagnosis:
        frame_times = video.frame_times if video is not None else ()
        output, model, latency_ms, usage = await self._call_with_fallback(
            images, note, crop, frame_times
        )
        status, chosen, confidence, reasons = decide(
            output,
            quality,
            min_confidence=self._min_confidence,
            crop_confirmed=crop in CROPS,
        )
        candidates = sorted(output.candidates, key=lambda c: c.confidence, reverse=True)
        diagnosis = Diagnosis(
            status=status,
            crop=output.crop,
            condition=chosen,
            confidence=confidence,
            model_confidence=round(candidates[0].confidence, 3),
            severity=output.severity if output.severity in _SEVERITIES else "unknown",
            plant_part=output.plant_part if output.plant_part in _PLANT_PARTS else "other",
            visible_symptoms=[s[:120] for s in output.visible_symptoms[:5]],
            alternatives=[
                Alternative(condition(c.condition_id), round(c.confidence, 3))
                for c in candidates[1:3]
            ],
            quality=quality,
            reasons=reasons,
            model=model,
            latency_ms=latency_ms,
            input_tokens=usage[0],
            output_tokens=usage[1],
            video=video,
        )
        metrics.diagnoses.labels("video" if video is not None else "photo", status).inc()
        logger.info(
            "image_diagnosed",
            source="video" if video is not None else "image",
            status=status,
            crop=diagnosis.crop,
            condition=chosen.id,
            confidence=confidence,
            model_confidence=diagnosis.model_confidence,
            reasons=reasons or None,
            model=model,
            latency_ms=latency_ms,
            frames=len(images),
            upload_bytes=original_bytes,
            quality_issues=quality.issues or None,
        )
        return diagnosis

    async def _call_with_fallback(
        self,
        images: Sequence[PreparedImage],
        note: str | None,
        crop: str | None,
        frame_times: Sequence[float] = (),
    ) -> tuple[VisionOutput, str, float, tuple[int, int]]:
        last_error: LLMError | None = None
        for model in self._models:
            for attempt in (1, 2):
                try:
                    return await self._call(model, images, note, crop, frame_times)
                except LLMError as exc:
                    last_error = exc
                    logger.warning(
                        "vision_call_failed", model=model, attempt=attempt, error_code=exc.code
                    )
                    if not exc.retryable or getattr(exc, "retry_after_seconds", None):
                        break  # long cooldown or non-retryable: try the next model
                    await asyncio.sleep(0.5 * attempt)
            if last_error is not None and not last_error.fallback_eligible:
                break
        assert last_error is not None  # noqa: S101 - at least one model is always configured
        raise last_error

    async def _call(
        self,
        model: str,
        images: Sequence[PreparedImage],
        note: str | None,
        crop: str | None,
        frame_times: Sequence[float] = (),
    ) -> tuple[VisionOutput, str, float, tuple[int, int]]:
        prompt = allowed_list_prompt(crop) + (
            f"\n\nFarmer's note: {note.strip()[:500]}" if note else ""
        )
        if len(images) > 1:
            times = ", ".join(f"{t:.1f}" for t in frame_times) or "different moments"
            prompt = (
                f"The {len(images)} images are frames from ONE short video of the same plant "
                f"(at {times} seconds). Combine what you see in all frames: a symptom may be "
                f"clear in only one of them. Answer once for the whole video.\n\n{prompt}"
            )
        parts = [types.Part.from_bytes(data=i.data, mime_type=i.mime_type) for i in images]
        contents = [types.Content(role="user", parts=[*parts, types.Part.from_text(text=prompt)])]
        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.1,
            max_output_tokens=800,
            response_mime_type="application/json",
            response_schema=response_schema(crop if crop in CROPS else None),
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        start = time.perf_counter()
        try:
            async with asyncio.timeout(self._timeout):
                response = await self._client.aio.models.generate_content(
                    model=model, contents=contents, config=config
                )
        except Exception as exc:
            error = translate_error(exc)
            metrics.llm_calls.labels("vision", model, error.code).inc()
            raise error from exc
        latency_ms = round((time.perf_counter() - start) * 1000, 1)
        meta = response.usage_metadata
        usage = (meta.prompt_token_count or 0, meta.candidates_token_count or 0) if meta else (0, 0)
        metrics.llm_duration.labels("vision", model).observe(latency_ms / 1000)
        metrics.llm_tokens.labels("vision", model, "input").inc(usage[0])
        metrics.llm_tokens.labels("vision", model, "output").inc(usage[1])
        try:
            output = VisionOutput.model_validate(json.loads(response.text or ""))
        except (ValueError, ValidationError) as exc:
            metrics.llm_calls.labels("vision", model, "invalid_output").inc()
            raise LLMResponseError("The image model returned an invalid result.") from exc
        metrics.llm_calls.labels("vision", model, "ok").inc()
        return output, model, latency_ms, usage

    async def warmup(self) -> None:
        async with asyncio.timeout(self._timeout):
            await self._client.aio.models.get(model=self.model)

    async def aclose(self) -> None:
        aclose = getattr(self._client.aio, "aclose", None)
        if aclose is not None:
            await aclose()
