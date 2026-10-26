"""Plant photo diagnosis endpoint (multipart upload)."""

from __future__ import annotations

import time
from typing import Annotated

from fastapi import APIRouter, File, Form, UploadFile

from app.api.deps import DiagnoserDep, SettingsDep
from app.core.errors import BadRequestError, PayloadTooLargeError
from app.schemas.errors import ErrorResponse
from app.schemas.vision import DiagnosisResponse
from app.vision.taxonomy import CROPS

router = APIRouter(prefix="/image", tags=["image"])


@router.post(
    "/diagnose",
    response_model=DiagnosisResponse,
    summary="Diagnose a plant photo",
    description=(
        "Upload a photo of a leaf, fruit or plant (JPEG, PNG or WEBP, max 10 MB). Returns the crop,"
        " the most likely condition from a fixed list, a calibrated confidence and alternatives. "
        "Photos that are too dark, blurry or small return `retake_photo` without an AI call. "
        "Location and other metadata are removed from the photo before processing."
    ),
    responses={
        413: {"model": ErrorResponse, "description": "Photo too large"},
        415: {"model": ErrorResponse, "description": "Not a JPEG/PNG/WEBP photo"},
        422: {"model": ErrorResponse, "description": "Unreadable photo or invalid request"},
        429: {"model": ErrorResponse, "description": "AI service rate limit reached"},
        503: {"model": ErrorResponse, "description": "AI service unavailable"},
    },
)
async def diagnose(
    diagnoser: DiagnoserDep,
    settings: SettingsDep,
    file: Annotated[UploadFile, File(description="Photo of the affected plant part")],
    note: Annotated[
        str | None, Form(max_length=500, description="Optional: crop name or problem description")
    ] = None,
    crop: Annotated[
        str | None,
        Form(
            description=(
                "Optional crop id if known (e.g. tomato, chilli, ginger). Strongly recommended: "
                "diagnosis is then limited to that crop's conditions, which avoids the most "
                "common error (misidentified crop)."
            )
        ),
    ] = None,
) -> DiagnosisResponse:
    start = time.perf_counter()
    if crop is not None and crop not in CROPS:
        raise BadRequestError(f"Unknown crop '{crop[:40]}'. Supported: {', '.join(CROPS)}.")
    data = await file.read(settings.max_image_bytes + 1)
    if len(data) > settings.max_image_bytes:
        raise PayloadTooLargeError(
            f"Photo is larger than {settings.max_image_bytes // (1024 * 1024)} MB."
        )
    diagnosis = await diagnoser.diagnose(data, note=note, crop=crop)
    return DiagnosisResponse.of(diagnosis, round(time.perf_counter() - start, 3))
