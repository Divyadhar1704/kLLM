"""Client-side sliding-window rate limiter (units per 60 s).

Keeps bulk embedding just under the provider quota instead of hammering it into 429s.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from collections.abc import Awaitable, Callable

WINDOW_SECONDS = 60.0


class SlidingWindowLimiter:
    def __init__(
        self,
        max_per_minute: int,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        if max_per_minute <= 0:
            raise ValueError("max_per_minute must be positive")
        self._max = max_per_minute
        self._clock = clock
        self._sleep = sleep
        self._events: deque[tuple[float, int]] = deque()
        self._used = 0
        self._lock = asyncio.Lock()

    def _expire(self, now: float) -> None:
        while self._events and now - self._events[0][0] >= WINDOW_SECONDS:
            _, units = self._events.popleft()
            self._used -= units

    async def acquire(self, units: int = 1) -> float:
        """Wait until `units` fit in the window. Returns seconds waited."""
        if units > self._max:
            raise ValueError(f"cannot acquire {units} units; limit is {self._max} per minute")
        waited = 0.0
        async with self._lock:
            while True:
                now = self._clock()
                self._expire(now)
                if self._used + units <= self._max:
                    self._events.append((now, units))
                    self._used += units
                    return waited
                wait = WINDOW_SECONDS - (now - self._events[0][0]) + 0.05
                waited += wait
                await self._sleep(wait)
