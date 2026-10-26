"""Aggregates all /api/v1 routers. Every route requires a caller (app.api.security): the AI
routes count against the AI rate limits, feedback against its own."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends

from app.api.security import ai_rate_limit, feedback_rate_limit
from app.api.v1 import chat, feedback, image, video
from app.schemas.errors import ErrorResponse

_AUTH_RESPONSES: dict[int | str, dict[str, Any]] = {
    401: {"model": ErrorResponse, "description": "Missing or invalid API key / token"},
    403: {"model": ErrorResponse, "description": "The API key may not act for users"},
    429: {"model": ErrorResponse, "description": "Rate limit exceeded (see Retry-After)"},
}

api_router = APIRouter(responses=_AUTH_RESPONSES)
for router in (chat.router, image.router, video.router):
    api_router.include_router(router, dependencies=[Depends(ai_rate_limit)])
api_router.include_router(feedback.router, dependencies=[Depends(feedback_rate_limit)])
