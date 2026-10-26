"""Error response schema shared by every endpoint."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ErrorBody(BaseModel):
    code: str = Field(description="Stable machine-readable error code", examples=["not_found"])
    message: str = Field(description="Human-readable message, safe to show to developers")
    request_id: str | None = Field(default=None, description="Correlates with server logs")
    details: list[dict[str, Any]] | None = Field(
        default=None, description="Field-level details (validation errors)"
    )


class ErrorResponse(BaseModel):
    error: ErrorBody
