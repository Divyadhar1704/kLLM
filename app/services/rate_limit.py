"""Request rate limits (fixed windows): per farmer per minute and per day, per client app.

- `memory`: counters inside this process. Correct for one instance (local, single Cloud Run
  instance); with N instances each one allows the full limit.
- `redis`: counters shared by all instances (INCR + EXPIRE, one round trip).
If Redis is unreachable the request is allowed and a warning is logged: rate limiting protects
the AI quota, it must not take the service down.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Protocol

from app.core import metrics
from app.core.config import Settings
from app.core.errors import RateLimitedError
from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class Limit:
    key: str  # what is counted, e.g. "user:krishindra-backend:u123"
    limit: int
    window_seconds: int
    name: str = "limit"  # metric label: user_minute, user_day, client_minute, feedback_minute


class RateLimiter(Protocol):
    async def hit(self, key: str, window_seconds: int) -> tuple[int, int]:
        """Count one request. Returns (requests in the current window, seconds until reset)."""
        ...

    async def check(self) -> None: ...

    async def aclose(self) -> None: ...


def _window(window_seconds: int) -> tuple[int, int]:
    now = int(time.time())
    window_id = now // window_seconds
    return window_id, (window_id + 1) * window_seconds - now


class MemoryRateLimiter:
    def __init__(self, *, max_keys: int = 100_000) -> None:
        self._counts: dict[str, tuple[int, int]] = {}
        self._max_keys = max_keys

    async def hit(self, key: str, window_seconds: int) -> tuple[int, int]:
        window_id, reset = _window(window_seconds)
        slot = f"{key}:{window_seconds}"
        current_window, count = self._counts.get(slot, (window_id, 0))
        count = count + 1 if current_window == window_id else 1
        if len(self._counts) >= self._max_keys and slot not in self._counts:
            self._prune()
        self._counts[slot] = (window_id, count)
        return count, reset

    def _prune(self) -> None:
        live = {
            slot: value
            for slot, value in self._counts.items()
            if value[0] == int(time.time()) // int(slot.rsplit(":", 1)[1])
        }
        self._counts = live if len(live) < self._max_keys else {}

    async def check(self) -> None:
        return None

    async def aclose(self) -> None:
        return None


class RedisRateLimiter:
    def __init__(self, client: Any, *, prefix: str = "krishimitra:rl") -> None:
        self._redis = client
        self._prefix = prefix

    @classmethod
    def from_settings(cls, settings: Settings) -> RedisRateLimiter:
        from redis.asyncio import Redis

        client = Redis.from_url(
            settings.redis_url.get_secret_value(),
            socket_timeout=0.5,
            socket_connect_timeout=1.0,
            health_check_interval=30,
        )
        return cls(client)

    async def hit(self, key: str, window_seconds: int) -> tuple[int, int]:
        window_id, reset = _window(window_seconds)
        name = f"{self._prefix}:{key}:{window_seconds}:{window_id}"
        async with self._redis.pipeline(transaction=False) as pipe:
            pipe.incr(name)
            pipe.expire(name, window_seconds + 5, nx=True)
            count, _ = await pipe.execute()
        return int(count), reset

    async def check(self) -> None:
        await self._redis.ping()

    async def aclose(self) -> None:
        await self._redis.aclose()


def build_rate_limiter(settings: Settings) -> RateLimiter:
    if settings.rate_limit_backend == "redis":
        return RedisRateLimiter.from_settings(settings)
    return MemoryRateLimiter()


async def enforce(limiter: RateLimiter, limits: list[Limit]) -> None:
    """Count the request against every limit; raise 429 if any is exceeded."""
    exceeded: list[tuple[Limit, int]] = []
    for limit in limits:
        try:
            count, reset = await limiter.hit(limit.key, limit.window_seconds)
        except Exception as exc:  # backend down: allow (fail open), but say so in the logs
            logger.warning("rate_limit_unavailable", error=type(exc).__name__)
            return
        if count > limit.limit:
            exceeded.append((limit, reset))
    if exceeded:
        limit, reset = max(exceeded, key=lambda item: item[1])  # the longest wait wins
        metrics.rate_limited.labels(limit.name).inc()
        logger.info("rate_limited", limit=limit.limit, window_seconds=limit.window_seconds)
        per = "minute" if limit.window_seconds == 60 else "day"
        raise RateLimitedError(
            f"Too many requests: the limit is {limit.limit} per {per}. Try again in {reset} s.",
            headers={"Retry-After": str(max(1, reset))},
        )
