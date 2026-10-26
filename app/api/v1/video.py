"""Plant video diagnosis endpoint (multipart upload)."""

from __future__ import annotations

import time
from typing import Annotated

from fastapi import APIRouter, File, Form, UploadFile

from app.api.deps import DiagnoserDep, SettingsDep
from app.core.errors import BadRequestError, PayloadTooLargeError
from app.schemas.errors import ErrorResponse
from app.schemas.vision import DiagnosisResponse
from app.vision.taxonomy import CROPS

router = APIRouter(prefix="/video", tags=["video"])


@router.post(
    "/diagnose",
    response_model=DiagnosisResponse,
    summary="Diagnose a short plant video",
    description=(
        "Upload a short video of the affected plant (MP4/MOV/3GP or WEBM, max 50 MB, max 60 s). "
        "The sharpest, well-lit frames from different moments are diagnosed together, with the "
        "same crop/condition list and confidence rules as photos. The sound track is never "
        "processed. Videos with no usable frame return `retake_photo` without an AI call. "
        "Location and other metadata never leave the server."
    ),
    responses={
        413: {"model": ErrorResponse, "description": "Video too large"},
        415: {"model": ErrorResponse, "description": "Not an MP4/MOV/3GP/WEBM video"},
        422: {"model": ErrorResponse, "description": "Unreadable or too long video"},
        429: {"model": ErrorResponse, "description": "AI service rate limit reached"},
        503: {"model": ErrorResponse, "description": "AI service unavailable"},
    },
)
async def diagnose(
    diagnoser: DiagnoserDep,
    settings: SettingsDep,
    file: Annotated[UploadFile, File(description="Short video of the affected plant")],
    note: Annotated[
        str | None, Form(max_length=500, description="Optional: crop name or problem description")
    ] = None,
    crop: Annotated[
        str | None,
        Form(
            description=(
                "Optional crop id if known (e.g. tomato, chilli, ginger). Strongly recommended: "
                "without it the result is never reported as confident."
            )
        ),
    ] = None,
) -> DiagnosisResponse:
    start = time.perf_counter()
    if crop is not None and crop not in CROPS:
        raise BadRequestError(f"Unknown crop '{crop[:40]}'. Supported: {', '.join(CROPS)}.")
    data = await file.read(settings.max_video_bytes + 1)
    if len(data) > settings.max_video_bytes:
        raise PayloadTooLargeError(
            f"Video is larger than {settings.max_video_bytes // (1024 * 1024)} MB."
        )
    diagnosis = await diagnoser.diagnose_video(data, note=note, crop=crop)
    return DiagnosisResponse.of(diagnosis, round(time.perf_counter() - start, 3))
