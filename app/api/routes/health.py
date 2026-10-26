"""Service info and health endpoints (unversioned infrastructure routes).

Cloud Run probes:
    liveness  -> GET /health        (process is alive; never touches dependencies)
    startup   -> GET /health/ready  (dependencies reachable)
"""

from __future__ import annotations

import time

from fastapi import APIRouter, Request, Response, status

from app.api.deps import HealthRegistryDep, SettingsDep
from app.schemas.errors import ErrorResponse
from app.schemas.health import (
    CheckResultSchema,
    LivenessResponse,
    ReadinessResponse,
    ServiceInfo,
)

router = APIRouter(tags=["health"])


@router.get("/", response_model=ServiceInfo, summary="Service information")
async def service_info(settings: SettingsDep) -> ServiceInfo:
    return ServiceInfo(
        service=settings.app_name,
        version=settings.app_version,
        environment=settings.app_env,
        api_base=settings.api_v1_prefix,
        docs="/docs" if settings.enable_docs else None,
        supported_languages=list(settings.supported_languages),
    )


@router.get("/health", response_model=LivenessResponse, summary="Liveness probe")
async def liveness() -> LivenessResponse:
    return LivenessResponse()


@router.get(
    "/health/ready",
    response_model=ReadinessResponse,
    summary="Readiness probe",
    responses={
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "model": ReadinessResponse,
            "description": "A critical dependency is unavailable",
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": ErrorResponse},
    },
)
async def readiness(
    request: Request,
    response: Response,
    settings: SettingsDep,
    registry: HealthRegistryDep,
) -> ReadinessResponse:
    report = await registry.check()
    if not report.is_available:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    started_at: float = request.app.state.started_at
    return ReadinessResponse(
        status=report.status,
        version=settings.app_version,
        uptime_seconds=round(time.monotonic() - started_at, 1),
        checked_at=report.checked_at,
        checks=[
            CheckResultSchema(
                name=r.name,
                status=r.status,
                critical=r.critical,
                latency_ms=r.latency_ms,
                error=r.error,
            )
            for r in report.checks
        ],
    )
