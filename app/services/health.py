"""Readiness checks for /health/ready.

Dependencies (Postgres, Redis, LLM, storage) register an async check as they are added.
A check passes by returning and fails by raising. Checks run concurrently, each with a timeout.

Overall status:
    ok           every check passed
    degraded     only non-critical checks failed   -> HTTP 200 (keep serving traffic)
    unavailable  at least one critical check failed -> HTTP 503 (take instance out of rotation)
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

from app.core.logging import get_logger

logger = get_logger(__name__)

CheckFunc = Callable[[], Awaitable[None]]
CheckStatus = Literal["ok", "fail", "timeout"]
OverallStatus = Literal["ok", "degraded", "unavailable"]


@dataclass(frozen=True, slots=True)
class HealthCheck:
    name: str
    func: CheckFunc
    critical: bool = True


@dataclass(frozen=True, slots=True)
class CheckResult:
    name: str
    status: CheckStatus
    critical: bool
    latency_ms: float
    error: str | None = None


@dataclass(frozen=True, slots=True)
class ReadinessReport:
    status: OverallStatus
    checks: tuple[CheckResult, ...]
    checked_at: datetime

    @property
    def is_available(self) -> bool:
        return self.status != "unavailable"


class HealthRegistry:
    def __init__(
        self,
        *,
        timeout_seconds: float,
        cache_seconds: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._timeout_seconds = timeout_seconds
        self._cache_seconds = cache_seconds
        self._clock = clock
        self._checks: dict[str, HealthCheck] = {}
        self._cached: ReadinessReport | None = None
        self._cached_at = 0.0
        self._lock = asyncio.Lock()

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._checks)

    def register(self, name: str, func: CheckFunc, *, critical: bool = True) -> None:
        if name in self._checks:
            raise ValueError(f"Health check already registered: {name}")
        self._checks[name] = HealthCheck(name=name, func=func, critical=critical)
        self._cached = None

    def unregister(self, name: str) -> None:
        self._checks.pop(name, None)
        self._cached = None

    async def check(self) -> ReadinessReport:
        if self._is_fresh():
            assert self._cached is not None  # noqa: S101 - guaranteed by _is_fresh
            return self._cached
        # One probe at a time runs the checks; concurrent probes reuse its result.
        async with self._lock:
            if self._is_fresh():
                assert self._cached is not None  # noqa: S101 - guaranteed by _is_fresh
                return self._cached
            report = await self._run_all()
            self._cached, self._cached_at = report, self._clock()
            return report

    def _is_fresh(self) -> bool:
        return (
            self._cached is not None
            and self._cache_seconds > 0
            and self._clock() - self._cached_at < self._cache_seconds
        )

    async def _run_all(self) -> ReadinessReport:
        results = await asyncio.gather(*(self._run_one(c) for c in self._checks.values()))
        if any(r.status != "ok" and r.critical for r in results):
            status: OverallStatus = "unavailable"
        elif any(r.status != "ok" for r in results):
            status = "degraded"
        else:
            status = "ok"
        if status != "ok":
            logger.warning(
                "readiness_not_ok",
                status=status,
                failing=[r.name for r in results if r.status != "ok"],
            )
        return ReadinessReport(status=status, checks=tuple(results), checked_at=datetime.now(UTC))

    async def _run_one(self, check: HealthCheck) -> CheckResult:
        start = time.perf_counter()
        status: CheckStatus = "ok"
        error: str | None = None
        try:
            async with asyncio.timeout(self._timeout_seconds):
                await check.func()
        except TimeoutError:
            status, error = "timeout", f"exceeded {self._timeout_seconds}s"
            logger.warning("health_check_timeout", check=check.name)
        except Exception as exc:
            # Only the exception type is exposed; the message may contain hosts or credentials.
            status, error = "fail", type(exc).__name__
            logger.warning("health_check_failed", check=check.name, exc_info=exc)
        latency_ms = round((time.perf_counter() - start) * 1000, 1)
        return CheckResult(
            name=check.name,
            status=status,
            critical=check.critical,
            latency_ms=latency_ms,
            error=error,
        )
