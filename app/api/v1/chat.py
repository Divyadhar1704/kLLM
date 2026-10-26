"""Text chat endpoints: blocking JSON and Server-Sent Events streaming."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Annotated, Any

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import StreamingResponse

from app.api.deps import ChatServiceDep, ImageChatServiceDep, SettingsDep
from app.core.errors import (
    INTERNAL_ERROR_MESSAGE,
    AppError,
    PayloadTooLargeError,
    error_payload,
    get_request_id,
)
from app.core.logging import get_logger
from app.schemas.chat import ChatRequest, ChatResponse, ConversationId
from app.schemas.errors import ErrorResponse
from app.services.language import LanguageCode

logger = get_logger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])

_ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    400: {"model": ErrorResponse, "description": "Message too long"},
    422: {"model": ErrorResponse, "description": "Invalid request or content blocked"},
    429: {"model": ErrorResponse, "description": "AI service rate limit reached"},
    502: {"model": ErrorResponse, "description": "AI service error"},
    503: {"model": ErrorResponse, "description": "AI service overloaded or not configured"},
    504: {"model": ErrorResponse, "description": "AI service timeout"},
}

_STREAM_DESCRIPTION = """Streams the answer as Server-Sent Events (`text/event-stream`).

Events, in order:

| event | data |
|---|---|
| `meta` | `{conversation_id, language, language_detected, grounded, sources[]}` |
| `delta` | `{text}`: append to the answer as it arrives (repeated) |
| `done` | the complete `ChatResponse` (same as `POST /chat`) |
| `error` | `{error: {code, message, request_id}}` if the stream fails after it started |

Errors before the first text (validation, rate limit, AI service unavailable) are returned as
normal HTTP errors with the standard error body, exactly like `POST /chat`.
"""


@router.post(
    "", response_model=ChatResponse, summary="Ask a text question", responses=_ERROR_RESPONSES
)
async def chat(body: ChatRequest, service: ChatServiceDep) -> ChatResponse:
    return await service.answer(body)


@router.post(
    "/image",
    response_model=ChatResponse,
    summary="Ask about a plant photo",
    description=(
        "Multipart upload: a photo (JPEG/PNG/WEBP, max 10 MB) with an optional question. The "
        "photo is diagnosed, the official documents are searched for that condition, and the "
        "answer is written in the farmer's language. `diagnosis` holds the photo result. "
        "Send `crop` whenever it is known: without a confirmed crop the diagnosis is always "
        "reported as uncertain. Unclear photos get a 'please retake' message without an AI call."
    ),
    responses={
        **_ERROR_RESPONSES,
        413: {"model": ErrorResponse, "description": "Photo too large"},
        415: {"model": ErrorResponse, "description": "Not a JPEG/PNG/WEBP photo"},
    },
)
async def chat_image(
    service: ImageChatServiceDep,
    settings: SettingsDep,
    file: Annotated[UploadFile, File(description="Photo of the affected leaf, fruit or plant")],
    message: Annotated[str | None, Form(max_length=10000)] = None,
    language: Annotated[LanguageCode | None, Form()] = None,
    crop: Annotated[str | None, Form(description="Crop id if known, e.g. tomato")] = None,
    conversation_id: Annotated[ConversationId | None, Form()] = None,
    debug: Annotated[bool, Form()] = False,
) -> ChatResponse:
    data = await file.read(settings.max_image_bytes + 1)
    if len(data) > settings.max_image_bytes:
        raise PayloadTooLargeError(
            f"Photo is larger than {settings.max_image_bytes // (1024 * 1024)} MB."
        )
    return await service.answer(
        image=data,
        message=message,
        language=language,
        crop=crop,
        conversation_id=conversation_id,
        debug=debug,
    )


@router.post(
    "/video",
    response_model=ChatResponse,
    summary="Ask about a short plant video",
    description=(
        "Multipart upload: a short video (MP4/MOV/3GP or WEBM, max 50 MB, max 60 s) with an "
        "optional question. The sharpest frames are diagnosed together, the official documents "
        "are searched for that condition, and the answer is written in the farmer's language. "
        "`diagnosis.video` describes the frames used. The sound track is never processed. Send "
        "`crop` whenever it is known: without a confirmed crop the diagnosis is always reported "
        "as uncertain. Unclear videos get a 'please record again' message without an AI call."
    ),
    responses={
        **_ERROR_RESPONSES,
        413: {"model": ErrorResponse, "description": "Video too large"},
        415: {"model": ErrorResponse, "description": "Not an MP4/MOV/3GP/WEBM video"},
    },
)
async def chat_video(
    service: ImageChatServiceDep,
    settings: SettingsDep,
    file: Annotated[UploadFile, File(description="Short video of the affected plant")],
    message: Annotated[str | None, Form(max_length=10000)] = None,
    language: Annotated[LanguageCode | None, Form()] = None,
    crop: Annotated[str | None, Form(description="Crop id if known, e.g. tomato")] = None,
    conversation_id: Annotated[ConversationId | None, Form()] = None,
    debug: Annotated[bool, Form()] = False,
) -> ChatResponse:
    data = await file.read(settings.max_video_bytes + 1)
    if len(data) > settings.max_video_bytes:
        raise PayloadTooLargeError(
            f"Video is larger than {settings.max_video_bytes // (1024 * 1024)} MB."
        )
    return await service.answer_video(
        video=data,
        message=message,
        language=language,
        crop=crop,
        conversation_id=conversation_id,
        debug=debug,
    )


def sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post(
    "/stream",
    summary="Ask a text question (streaming)",
    description=_STREAM_DESCRIPTION,
    response_class=StreamingResponse,
    responses={200: {"content": {"text/event-stream": {}}}, **_ERROR_RESPONSES},
)
async def chat_stream(
    body: ChatRequest, service: ChatServiceDep, request: Request
) -> StreamingResponse:
    events = service.stream(body)
    # Hold the response until the first text arrives: anything that fails before that
    # (validation, retrieval, LLM rate limit / overload) raises here and becomes a normal HTTP
    # error response with the right status code.
    buffered: list[str] = []
    async for name, data in events:
        buffered.append(sse(name, data))
        if name in ("delta", "done"):
            break

    request_id = get_request_id(request)

    async def body_iterator() -> AsyncIterator[str]:
        for chunk in buffered:
            yield chunk
        try:
            async for name, data in events:
                yield sse(name, data)
        except AppError as exc:
            logger.warning("chat_stream_failed", code=exc.code, error_message=exc.message)
            yield sse(
                "error", error_payload(code=exc.code, message=exc.message, request_id=request_id)
            )
        except Exception as exc:
            logger.error("chat_stream_unhandled", exc_info=exc)
            yield sse(
                "error",
                error_payload(
                    code="internal_error", message=INTERNAL_ERROR_MESSAGE, request_id=request_id
                ),
            )
        finally:
            await events.aclose()

    return StreamingResponse(
        body_iterator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-store",
            "X-Accel-Buffering": "no",  # disable proxy buffering so tokens arrive immediately
        },
    )
