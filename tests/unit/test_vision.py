"""Image validation, privacy, quality gate, calibration rules, diagnoser and API (no AI calls)."""

from __future__ import annotations

import io
import json
import os
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient
from google.genai import errors as genai_errors
from PIL import Image, ImageDraw, ImageFilter

from app.core.config import Settings
from app.core.errors import PayloadTooLargeError, UnsupportedMediaTypeError
from app.main import create_app
from app.services.llm.errors import LLMResponseError
from app.vision.diagnosis import VisionDiagnoser, VisionOutput, decide, response_schema
from app.vision.images import ImageQuality, InvalidImageError, assess_quality, prepare_image
from app.vision.taxonomy import CONDITIONS, CROPS, UNKNOWN

MB = 1024 * 1024

# ---------------------------------------------------------------- synthetic photos


def leaf_image(size: tuple[int, int] = (640, 480), *, brightness: int = 150) -> Image.Image:
    """A textured 'leaf': enough edges to count as sharp, adjustable brightness."""
    image = Image.new("RGB", size, (30, 60, 30))
    draw = ImageDraw.Draw(image)
    for x in range(0, size[0], 16):
        for y in range(0, size[1], 16):
            shade = (x * 7 + y * 13) % 120
            draw.rectangle((x, y, x + 8, y + 8), fill=(20, min(255, brightness + shade // 3), 20))
    draw.ellipse((200, 150, 260, 210), fill=(110, 70, 30))  # a "lesion"
    return image


def encode(image: Image.Image, fmt: str = "JPEG", **kwargs: Any) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format=fmt, **kwargs)
    return buffer.getvalue()


def with_gps_exif(image: Image.Image) -> bytes:
    exif = Image.Exif()
    exif[0x0112] = 6  # orientation: rotate 90 degrees
    exif[0x8825] = {1: "N", 2: (17.0, 23.0, 0.0), 3: "E", 4: (78.0, 29.0, 0.0)}  # GPS IFD
    return encode(image, "JPEG", exif=exif)


def prepare(data: bytes, **kwargs: Any) -> Any:
    return prepare_image(data, max_bytes=kwargs.get("max_bytes", 10 * MB), max_side=1024)


# ---------------------------------------------------------------- validation and privacy


@pytest.mark.parametrize("fmt", ["JPEG", "PNG", "WEBP"])
def test_accepts_supported_formats(fmt: str) -> None:
    prepared = prepare(encode(leaf_image(), fmt))
    assert prepared.original_format == fmt
    assert prepared.mime_type == "image/jpeg"
    assert prepared.quality.issues == ()


def test_accepts_phone_mpo_photos() -> None:
    # Multi-picture JPEG, as saved by many phone cameras under a ".jpg" name.
    main, depth = leaf_image(), leaf_image((320, 240))
    buffer = io.BytesIO()
    main.save(buffer, format="MPO", save_all=True, append_images=[depth])
    prepared = prepare(buffer.getvalue())
    assert prepared.original_format == "MPO"
    assert Image.open(io.BytesIO(prepared.data)).size == (640, 480)  # the main frame


def test_rejects_unsupported_format() -> None:
    with pytest.raises(UnsupportedMediaTypeError):
        prepare(encode(leaf_image(), "GIF"))


@pytest.mark.parametrize("data", [b"", b"not an image at all", b"\xff\xd8\xff" + b"\x00" * 50])
def test_rejects_corrupt_or_empty(data: bytes) -> None:
    with pytest.raises(InvalidImageError):
        prepare(data)


def test_rejects_too_large_file() -> None:
    with pytest.raises(PayloadTooLargeError):
        prepare(encode(leaf_image()), max_bytes=1000)


def test_rejects_decompression_bomb() -> None:
    bomb = encode(Image.new("L", (9000, 9000), 0), "PNG")  # 81 MP, tiny file
    with pytest.raises(InvalidImageError):
        prepare(bomb)


def test_strips_exif_gps_and_applies_rotation() -> None:
    original = leaf_image((640, 480))
    prepared = prepare(with_gps_exif(original))
    cleaned = Image.open(io.BytesIO(prepared.data))
    assert cleaned.size == (480, 640)  # rotated upright per EXIF orientation
    assert not cleaned.getexif()  # no metadata at all, including GPS


def test_resizes_for_the_model() -> None:
    prepared = prepare(encode(leaf_image((3000, 2000))))
    assert max(prepared.model_size) == 1024
    assert prepared.quality.width == 3000  # quality measured on the original


def test_transparent_png_flattened() -> None:
    image = leaf_image().convert("RGBA")
    image.putalpha(128)
    prepared = prepare(encode(image, "PNG"))
    assert Image.open(io.BytesIO(prepared.data)).mode == "RGB"


# ---------------------------------------------------------------- quality gate


def test_quality_issues() -> None:
    assert "too_small" in assess_quality(leaf_image((150, 150))).issues
    assert "too_dark" in assess_quality(leaf_image(brightness=10).point(lambda v: v // 4)).issues
    assert "overexposed" in assess_quality(Image.new("RGB", (640, 480), (250, 250, 250))).issues
    mild = assess_quality(leaf_image().filter(ImageFilter.GaussianBlur(6))).issues
    assert "blurry" in mild or "very_blurry" in mild
    heavy = assess_quality(leaf_image().filter(ImageFilter.GaussianBlur(30))).issues
    assert "very_blurry" in heavy


def test_dark_background_with_lit_leaf_is_ok() -> None:
    image = Image.new("RGB", (640, 480), (5, 5, 5))
    image.paste(leaf_image((300, 300), brightness=170), (170, 90))
    assert "too_dark" not in assess_quality(image).issues


# ---------------------------------------------------------------- calibration rules

GOOD = ImageQuality(width=800, height=600, brightness=120, sharpness=500)


def output(**overrides: Any) -> VisionOutput:
    data: dict[str, Any] = {
        "is_plant": True,
        "crop": "tomato",
        "plant_part": "leaf",
        "image_quality": "good",
        "candidates": [{"condition_id": "tomato_early_blight", "confidence": 0.85}],
        "severity": "moderate",
        "visible_symptoms": ["concentric rings"],
    }
    data.update(overrides)
    return VisionOutput.model_validate(data)


def test_confident_diagnosis() -> None:
    status, chosen, confidence, reasons = decide(
        output(), GOOD, min_confidence=0.6, crop_confirmed=True
    )
    assert (status, chosen.id, confidence, reasons) == (
        "confident",
        "tomato_early_blight",
        0.85,
        [],
    )


def test_never_confident_without_confirmed_crop() -> None:
    sure = output(candidates=[{"condition_id": "tomato_early_blight", "confidence": 0.98}])
    status, chosen, confidence, reasons = decide(
        sure, GOOD, min_confidence=0.6, crop_confirmed=False
    )
    assert status == "uncertain"
    assert chosen.id == "tomato_early_blight" and confidence == 0.98  # still reported
    assert "crop not confirmed by the farmer" in reasons


def test_close_candidates_become_uncertain() -> None:
    candidates = [
        {"condition_id": "tomato_early_blight", "confidence": 0.45},
        {"condition_id": "tomato_target_spot", "confidence": 0.40},
    ]
    status, _, confidence, reasons = decide(
        output(candidates=candidates), GOOD, min_confidence=0.6, crop_confirmed=True
    )
    assert status == "uncertain" and confidence <= 0.55
    assert "top candidates are close" in reasons


def test_crop_mismatch_lowers_confidence() -> None:
    candidates = [{"condition_id": "potato_late_blight", "confidence": 0.9}]
    status, _, confidence, reasons = decide(
        output(candidates=candidates), GOOD, min_confidence=0.6, crop_confirmed=True
    )
    assert confidence == pytest.approx(0.54)
    assert status == "uncertain"
    assert any("condition is for potato" in r for r in reasons)


def test_poor_photo_caps_confidence() -> None:
    status, _, confidence, _ = decide(
        output(image_quality="poor"), GOOD, min_confidence=0.6, crop_confirmed=True
    )
    assert confidence == 0.5 and status == "uncertain"


def test_not_a_plant() -> None:
    status, chosen, _, _ = decide(
        output(is_plant=False), GOOD, min_confidence=0.6, crop_confirmed=True
    )
    assert status == "not_a_plant" and chosen.id == UNKNOWN


def test_unknown_ids_are_mapped_to_unknown() -> None:
    parsed = output(crop="dragonfruit", candidates=[{"condition_id": "made_up", "confidence": 0.9}])
    assert parsed.crop == UNKNOWN and parsed.candidates[0].condition_id == UNKNOWN
    status, _, _, _ = decide(parsed, GOOD, min_confidence=0.6, crop_confirmed=True)
    assert status == "uncertain"


def test_generic_conditions_fit_any_crop() -> None:
    candidates = [{"condition_id": "nutrient_deficiency", "confidence": 0.8}]
    status, _, confidence, _ = decide(
        output(candidates=candidates), GOOD, min_confidence=0.6, crop_confirmed=True
    )
    assert status == "confident" and confidence == 0.8


def test_taxonomy_integrity() -> None:
    for condition in CONDITIONS.values():
        assert condition.crop in (*CROPS, "any")
        if condition.kind == "healthy":
            assert condition.id == f"{condition.crop}_healthy"
    for crop in CROPS:
        assert f"{crop}_healthy" in CONDITIONS, f"missing healthy class for {crop}"
    schema = response_schema()
    assert schema.properties is not None
    assert UNKNOWN in (schema.properties["crop"].enum or [])


# ---------------------------------------------------------------- diagnoser (fake client)


class FakeModels:
    def __init__(self, outcomes: list[Any]) -> None:
        self.outcomes = outcomes
        self.calls: list[dict[str, Any]] = []

    async def generate_content(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return SimpleNamespace(
            text=outcome if isinstance(outcome, str) else json.dumps(outcome),
            usage_metadata=SimpleNamespace(prompt_token_count=1300, candidates_token_count=90),
        )

    async def get(self, *, model: str) -> object:
        return object()


GOOD_JSON = {
    "is_plant": True,
    "crop": "tomato",
    "plant_part": "leaf",
    "image_quality": "good",
    "candidates": [
        {"condition_id": "tomato_early_blight", "confidence": 0.8},
        {"condition_id": "tomato_target_spot", "confidence": 0.1},
    ],
    "severity": "moderate",
    "visible_symptoms": ["concentric rings", "yellow halo"],
}


def diagnoser(
    outcomes: list[Any], models: tuple[str, ...] = ("primary", "fallback")
) -> tuple[VisionDiagnoser, FakeModels]:
    fake = FakeModels(outcomes)
    return (
        VisionDiagnoser(
            client=SimpleNamespace(aio=SimpleNamespace(models=fake)),
            models=models,
            timeout_seconds=5,
            max_bytes=10 * MB,
            model_max_side=1024,
            min_confidence=0.6,
            reject_poor_quality=True,
        ),
        fake,
    )


async def test_diagnose_sends_image_and_constrained_schema() -> None:
    service, fake = diagnoser([GOOD_JSON])
    result = await service.diagnose(
        encode(leaf_image()), note="tomato leaves have spots", crop="tomato"
    )
    assert result.status == "confident"
    assert result.condition.id == "tomato_early_blight"
    assert result.alternatives[0].condition.id == "tomato_target_spot"
    assert result.model == "primary" and result.input_tokens == 1300
    call = fake.calls[0]
    parts = call["contents"][0].parts
    assert parts[0].inline_data.mime_type == "image/jpeg"
    assert "Farmer's note: tomato leaves have spots" in parts[1].text
    config = call["config"]
    assert config.response_mime_type == "application/json"
    assert (
        "tomato_early_blight"
        in config.response_schema.properties["candidates"].items.properties["condition_id"].enum
    )


async def test_poor_photo_skips_the_ai_call() -> None:
    service, fake = diagnoser([GOOD_JSON])
    dark = leaf_image().point(lambda v: v // 8)
    result = await service.diagnose(encode(dark))
    assert result.status == "retake_photo"
    assert fake.calls == []


async def test_invalid_json_retried_then_fallback_model() -> None:
    overloaded = genai_errors.ServerError(503, {"error": {"code": 503, "message": "busy"}})
    service, fake = diagnoser(["not json", overloaded, GOOD_JSON])
    result = await service.diagnose(encode(leaf_image()), crop="tomato")
    assert result.status == "confident"
    assert result.model == "fallback"
    assert [c["model"] for c in fake.calls] == ["primary", "primary", "fallback"]


async def test_all_models_fail_raises() -> None:
    service, _ = diagnoser(["bad", "bad", "bad", "bad"])
    with pytest.raises(LLMResponseError):
        await service.diagnose(encode(leaf_image()))


# ---------------------------------------------------------------- API


def client_with(service: VisionDiagnoser | None, **overrides: Any) -> TestClient:
    app = create_app(Settings(_env_file=None, **overrides), diagnoser=service)  # type: ignore[arg-type]
    return TestClient(app)


def upload(
    client: TestClient,
    data: bytes,
    name: str = "leaf.jpg",
    note: str | None = None,
    crop: str | None = None,
) -> Any:
    form = {k: v for k, v in (("note", note), ("crop", crop)) if v} or None
    return client.post(
        "/api/v1/image/diagnose", files={"file": (name, data, "image/jpeg")}, data=form
    )


def test_api_diagnose() -> None:
    service, _ = diagnoser([GOOD_JSON])
    response = upload(client_with(service), encode(leaf_image()), note="spots", crop="tomato")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "confident"
    assert body["crop"] == "tomato" and body["crop_name"] == "Tomato"
    assert body["condition"] == {
        "id": "tomato_early_blight",
        "name": "Early blight",
        "kind": "disease",
        "crop": "tomato",
    }
    assert body["alternatives"][0]["condition"]["id"] == "tomato_target_spot"
    assert body["quality"]["issues"] == []
    assert body["processing_time"] >= 0


def test_api_crop_hint_narrows_the_schema() -> None:
    service, fake = diagnoser([GOOD_JSON])
    upload(client_with(service), encode(leaf_image()), crop="tomato")
    config = fake.calls[0]["config"]
    conditions = config.response_schema.properties["candidates"].items.properties["condition_id"]
    assert "tomato_early_blight" in conditions.enum
    assert "potato_late_blight" not in conditions.enum  # other crops excluded
    assert "nutrient_deficiency" in conditions.enum  # generic conditions kept
    assert "The farmer says this is Tomato" in fake.calls[0]["contents"][0].parts[1].text


def test_api_without_crop_is_never_confident() -> None:
    service, _ = diagnoser([GOOD_JSON])
    body = upload(client_with(service), encode(leaf_image())).json()
    assert body["status"] == "uncertain"
    assert "crop not confirmed by the farmer" in body["reasons"]


def test_api_unknown_crop_rejected() -> None:
    service, fake = diagnoser([GOOD_JSON])
    response = upload(client_with(service), encode(leaf_image()), crop="dragonfruit")
    assert response.status_code == 400
    assert fake.calls == []


def test_api_file_type_checked_by_content_not_name() -> None:
    service, _ = diagnoser([GOOD_JSON])
    response = upload(client_with(service), encode(leaf_image(), "GIF"), name="photo.jpg")
    assert response.status_code == 415
    assert response.json()["error"]["code"] == "unsupported_media_type"


def test_api_corrupt_image() -> None:
    service, _ = diagnoser([GOOD_JSON])
    response = upload(client_with(service), b"garbage bytes")
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_image"


def test_api_missing_file() -> None:
    service, _ = diagnoser([GOOD_JSON])
    response = client_with(service).post("/api/v1/image/diagnose", data={"note": "x"})
    assert response.status_code == 422


def test_api_body_size_limit() -> None:
    service, _ = diagnoser([GOOD_JSON])
    client = client_with(service, max_image_bytes=50_000)
    noise = Image.frombytes("RGB", (1000, 1000), os.urandom(1000 * 1000 * 3))
    big = encode(noise, "PNG")  # random pixels do not compress: ~3 MB
    assert len(big) > 50_000 + 512 * 1024
    response = upload(client, big)
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "payload_too_large"


def test_api_vision_disabled() -> None:
    response = upload(client_with(None), encode(leaf_image()))
    assert response.status_code == 503
