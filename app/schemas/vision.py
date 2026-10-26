"""Image diagnosis response schema."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.vision.diagnosis import Diagnosis
from app.vision.taxonomy import Condition, crop_name
from app.vision.video import VideoInfo


class ConditionOut(BaseModel):
    id: str = Field(examples=["tomato_early_blight"])
    name: str = Field(examples=["Early blight"])
    kind: str = Field(
        description="disease | pest | virus | deficiency | abiotic | healthy | unknown"
    )
    crop: str

    @classmethod
    def of(cls, condition: Condition) -> ConditionOut:
        return cls(id=condition.id, name=condition.name, kind=condition.kind, crop=condition.crop)


class AlternativeOut(BaseModel):
    condition: ConditionOut
    confidence: float


class ImageQualityOut(BaseModel):
    width: int
    height: int
    brightness: float
    sharpness: float
    issues: list[str]


class VideoInfoOut(BaseModel):
    duration_seconds: float
    width: int
    height: int
    container: str = Field(examples=["mp4"])
    codec: str = Field(examples=["h264"])
    has_audio: bool = Field(description="The video has a sound track. Audio is never processed.")
    frames_examined: int = Field(description="Frames checked for sharpness and exposure")
    frames_used: list[float] = Field(description="Seconds into the video of the frames diagnosed")

    @classmethod
    def of(cls, info: VideoInfo) -> VideoInfoOut:
        return cls(
            duration_seconds=info.duration_seconds,
            width=info.width,
            height=info.height,
            container=info.container,
            codec=info.codec,
            has_audio=info.has_audio,
            frames_examined=info.frames_examined,
            frames_used=list(info.frame_times),
        )


class DiagnosisResponse(BaseModel):
    status: Literal["confident", "uncertain", "not_a_plant", "retake_photo"] = Field(
        description=(
            "confident: show the diagnosis. uncertain: show as a possibility and suggest a "
            "clearer photo or the KVK. not_a_plant / retake_photo: ask for a new photo."
        )
    )
    crop: str
    crop_name: str
    condition: ConditionOut
    confidence: float = Field(description="Calibrated confidence, 0 to 1")
    model_confidence: float = Field(description="Confidence reported by the model, 0 to 1")
    severity: str
    plant_part: str
    visible_symptoms: list[str]
    alternatives: list[AlternativeOut]
    quality: ImageQualityOut
    reasons: list[str] = Field(description="Why the confidence was lowered, if it was")
    model: str | None
    processing_time: float = Field(description="Server processing time in seconds")
    video: VideoInfoOut | None = Field(default=None, description="Set when a video was diagnosed")

    @classmethod
    def of(cls, diagnosis: Diagnosis, processing_time: float) -> DiagnosisResponse:
        quality = diagnosis.quality
        return cls(
            status=diagnosis.status,
            crop=diagnosis.crop,
            crop_name=crop_name(diagnosis.crop),
            condition=ConditionOut.of(diagnosis.condition),
            confidence=diagnosis.confidence,
            model_confidence=diagnosis.model_confidence,
            severity=diagnosis.severity,
            plant_part=diagnosis.plant_part,
            visible_symptoms=diagnosis.visible_symptoms,
            alternatives=[
                AlternativeOut(condition=ConditionOut.of(a.condition), confidence=a.confidence)
                for a in diagnosis.alternatives
            ],
            quality=ImageQualityOut(
                width=quality.width,
                height=quality.height,
                brightness=quality.brightness,
                sharpness=quality.sharpness,
                issues=list(quality.issues),
            ),
            reasons=diagnosis.reasons,
            model=diagnosis.model,
            processing_time=processing_time,
            video=VideoInfoOut.of(diagnosis.video) if diagnosis.video is not None else None,
        )
