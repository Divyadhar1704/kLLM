"""Health and service-info response schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class ServiceInfo(BaseModel):
    service: str
    version: str
    environment: str
    api_base: str = Field(description="Base path of the versioned API")
    docs: str | None = Field(description="OpenAPI docs path, null when disabled")
    supported_languages: list[str]


class LivenessResponse(BaseModel):
    status: Literal["ok"] = "ok"


class CheckResultSchema(BaseModel):
    name: str
    status: Literal["ok", "fail", "timeout"]
    critical: bool
    latency_ms: float
    error: str | None = None


class ReadinessResponse(BaseModel):
    status: Literal["ok", "degraded", "unavailable"]
    version: str
    uptime_seconds: float
    checked_at: datetime
    checks: list[CheckResultSchema]
