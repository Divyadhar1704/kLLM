"""GET /metrics (Prometheus text format). Registered only when not deployed: deployed instances
serve metrics on 127.0.0.1:METRICS_PORT to the monitoring sidecar, never on the public port."""

from __future__ import annotations

from fastapi import APIRouter, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

router = APIRouter(include_in_schema=False)


@router.get("/metrics")
def metrics_endpoint() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
