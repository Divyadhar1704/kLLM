"""Retry and fallback wrappers. Both implement LLMProvider, so they compose:

    FallbackLLM(RetryingLLM(primary), RetryingLLM(fallback))

Streams are retried / switched to the fallback only before the first text chunk is sent;
after that a failure is surfaced, because the client has already shown partial text.
"""

from __future__ import annotations

import asyncio
import dataclasses
import random
from collections.abc import AsyncIterator, Awaitable, Callable

from app.core.logging import get_logger
from app.services.llm.base import (
    LLMProvider,
    LLMRequest,
    LLMResponse,
    StreamDelta,
    StreamEnd,
    StreamEvent,
)
from app.services.llm.errors import LLMError

logger = get_logger(__name__)

Sleep = Callable[[float], Awaitable[None]]


class RetryingLLM:
    def __init__(
        self,
        inner: LLMProvider,
        *,
        max_retries: int,
        base_delay_seconds: float = 0.5,
        max_delay_seconds: float = 4.0,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        self._inner = inner
        self._max_retries = max_retries
        self._base_delay = base_delay_seconds
        self._max_delay = max_delay_seconds
        self._sleep = sleep

    @property
    def provider_name(self) -> str:
        return self._inner.provider_name

    @property
    def model(self) -> str:
        return self._inner.model

    def _delay(self, attempt: int) -> float:
        backoff: float = min(self._max_delay, self._base_delay * 2.0**attempt)
        jitter: float = random.uniform(0, backoff / 4)  # noqa: S311 - jitter, not crypto
        return backoff + jitter

    async def _before_retry(self, exc: LLMError, attempt: int) -> bool:
        if not exc.retryable or attempt >= self._max_retries:
            return False
        retry_after = getattr(exc, "retry_after_seconds", None)
        if retry_after is not None and retry_after > self._max_delay:
            # A farmer is waiting: don't sleep for the provider's long cooldown; fail over instead.
            return False
        delay = self._delay(attempt)
        logger.warning(
            "llm_retry",
            model=self.model,
            error_code=exc.code,
            attempt=attempt + 1,
            delay_seconds=round(delay, 2),
        )
        await self._sleep(delay)
        return True

    async def generate(self, request: LLMRequest) -> LLMResponse:
        attempt = 0
        while True:
            try:
                response = await self._inner.generate(request)
                return dataclasses.replace(response, attempts=attempt + 1)
            except LLMError as exc:
                if not await self._before_retry(exc, attempt):
                    raise
                attempt += 1

    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamEvent]:
        attempt = 0
        while True:
            sent_text = False
            try:
                async for event in self._inner.stream(request):
                    if isinstance(event, StreamDelta):
                        sent_text = True
                    elif isinstance(event, StreamEnd):
                        event = StreamEnd(dataclasses.replace(event.response, attempts=attempt + 1))
                    yield event
                return
            except LLMError as exc:
                if sent_text or not await self._before_retry(exc, attempt):
                    raise
                attempt += 1

    async def check(self) -> None:
        await self._inner.check()

    async def aclose(self) -> None:
        await self._inner.aclose()


class FallbackLLM:
    def __init__(self, primary: LLMProvider, fallback: LLMProvider) -> None:
        self._primary = primary
        self._fallback = fallback

    @property
    def provider_name(self) -> str:
        return self._primary.provider_name

    @property
    def model(self) -> str:
        return self._primary.model

    def _log_fallback(self, exc: LLMError) -> None:
        logger.warning(
            "llm_fallback",
            from_model=self._primary.model,
            to_model=self._fallback.model,
            error_code=exc.code,
        )

    async def generate(self, request: LLMRequest) -> LLMResponse:
        try:
            return await self._primary.generate(request)
        except LLMError as exc:
            if not exc.fallback_eligible:
                raise
            self._log_fallback(exc)
            return await self._fallback.generate(request)

    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamEvent]:
        sent_text = False
        try:
            async for event in self._primary.stream(request):
                if isinstance(event, StreamDelta):
                    sent_text = True
                yield event
            return
        except LLMError as exc:
            if sent_text or not exc.fallback_eligible:
                raise
            self._log_fallback(exc)
        async for event in self._fallback.stream(request):
            yield event

    async def check(self) -> None:
        # Readiness reflects the primary model; fallback problems show up in llm_fallback logs.
        await self._primary.check()

    async def aclose(self) -> None:
        await asyncio.gather(self._primary.aclose(), self._fallback.aclose())
