from __future__ import annotations

import asyncio
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app
from app.services.health import HealthRegistry
from tests.fakes import FakeLLM

# ---------------------------------------------------------------- helpers


async def passing() -> None:
    return None


async def failing() -> None:
    raise ConnectionError("could not connect to postgres://admin:hunter2@10.0.0.5")


async def hanging() -> None:
    await asyncio.sleep(10)


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def registry(cache_seconds: float = 0, clock: FakeClock | None = None) -> HealthRegistry:
    return HealthRegistry(
        timeout_seconds=0.05, cache_seconds=cache_seconds, clock=clock or FakeClock()
    )


def build_app(**overrides: Any) -> FastAPI:
    params: dict[str, Any] = {"health_cache_seconds": 0}
    params.update(overrides)
    return create_app(Settings(_env_file=None, **params), llm=FakeLLM())  # type: ignore[arg-type]


def check_named(body: dict[str, Any], name: str) -> dict[str, Any]:
    result: dict[str, Any] = next(c for c in body["checks"] if c["name"] == name)
    return result


# ---------------------------------------------------------------- registry


async def test_no_checks_is_ok() -> None:
    report = await registry().check()
    assert report.status == "ok"
    assert report.checks == ()


async def test_all_passing() -> None:
    reg = registry()
    reg.register("postgres", passing)
    reg.register("redis", passing)
    report = await reg.check()
    assert report.status == "ok"
    assert [c.name for c in report.checks] == ["postgres", "redis"]
    assert all(c.status == "ok" and c.error is None for c in report.checks)


async def test_critical_failure_is_unavailable() -> None:
    reg = registry()
    reg.register("postgres", failing)
    reg.register("redis", passing)
    report = await reg.check()
    assert report.status == "unavailable"
    assert not report.is_available
    pg = report.checks[0]
    assert pg.status == "fail"
    assert pg.error == "ConnectionError"


async def test_non_critical_failure_is_degraded() -> None:
    reg = registry()
    reg.register("postgres", passing)
    reg.register("llm", failing, critical=False)
    report = await reg.check()
    assert report.status == "degraded"
    assert report.is_available


async def test_timeout() -> None:
    reg = registry()
    reg.register("slow", hanging)
    report = await reg.check()
    assert report.status == "unavailable"
    assert report.checks[0].status == "timeout"
    assert report.checks[0].latency_ms < 1000


async def test_checks_run_concurrently() -> None:
    reg = registry()
    for i in range(5):
        reg.register(f"slow-{i}", hanging, critical=False)
    loop = asyncio.get_running_loop()
    start = loop.time()
    await reg.check()
    # Five 50 ms timeouts in parallel, not 250 ms in sequence.
    assert loop.time() - start < 0.2


async def test_duplicate_name_rejected() -> None:
    reg = registry()
    reg.register("postgres", passing)
    with pytest.raises(ValueError, match="already registered"):
        reg.register("postgres", passing)


async def test_result_cached_within_ttl() -> None:
    calls = 0

    async def counting() -> None:
        nonlocal calls
        calls += 1

    clock = FakeClock()
    reg = registry(cache_seconds=5, clock=clock)
    reg.register("counting", counting)
    await reg.check()
    await reg.check()
    assert calls == 1
    clock.now += 5.1
    await reg.check()
    assert calls == 2


async def test_concurrent_probes_share_one_run() -> None:
    calls = 0

    async def slow_counting() -> None:
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.01)

    reg = registry(cache_seconds=5)
    reg.register("counting", slow_counting)
    await asyncio.gather(*(reg.check() for _ in range(10)))
    assert calls == 1


async def test_register_invalidates_cache() -> None:
    reg = registry(cache_seconds=60)
    await reg.check()
    reg.register("postgres", failing)
    assert (await reg.check()).status == "unavailable"


# ---------------------------------------------------------------- endpoints


def test_service_info() -> None:
    response = TestClient(build_app()).get("/")
    assert response.status_code == 200
    body = response.json()
    assert body["service"] == "KrishiMitra API"
    assert body["version"] == "0.1.0"
    assert body["environment"] == "test"
    assert body["api_base"] == "/api/v1"
    assert body["docs"] == "/docs"
    assert body["supported_languages"] == ["en", "te", "kn", "hi"]


def test_service_info_hides_docs_when_disabled() -> None:
    response = TestClient(build_app(enable_docs=False)).get("/")
    assert response.json()["docs"] is None


def test_liveness() -> None:
    response = TestClient(build_app()).get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert response.headers["Cache-Control"] == "no-store"


def test_liveness_does_not_run_dependency_checks() -> None:
    app = build_app()
    app.state.health_registry.register("postgres", failing)
    assert TestClient(app).get("/health").status_code == 200


def test_readiness_ok() -> None:
    app = build_app()
    app.state.health_registry.register("postgres", passing)
    response = TestClient(app).get("/health/ready")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["version"] == "0.1.0"
    assert body["uptime_seconds"] >= 0
    assert check_named(body, "postgres")["status"] == "ok"
    assert check_named(body, "llm")["status"] == "ok"


def test_readiness_unavailable_returns_503_without_leaking() -> None:
    app = build_app()
    app.state.health_registry.register("postgres", failing)
    response = TestClient(app).get("/health/ready")
    assert response.status_code == 503
    assert response.json()["status"] == "unavailable"
    assert check_named(response.json(), "postgres")["error"] == "ConnectionError"
    assert "hunter2" not in response.text
    assert "10.0.0.5" not in response.text


def test_readiness_degraded_returns_200() -> None:
    app = build_app()
    app.state.health_registry.register("optional-service", failing, critical=False)
    response = TestClient(app).get("/health/ready")
    assert response.status_code == 200
    assert response.json()["status"] == "degraded"


def test_openapi_lists_health_routes() -> None:
    paths = TestClient(build_app()).get("/openapi.json").json()["paths"]
    assert {"/", "/health", "/health/ready"} <= set(paths)
