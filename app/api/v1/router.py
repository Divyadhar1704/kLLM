"""Aggregates all /api/v1 routers."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.v1 import chat, feedback, image, video

api_router = APIRouter()
api_router.include_router(chat.router)
api_router.include_router(image.router)
api_router.include_router(video.router)
api_router.include_router(feedback.router)
