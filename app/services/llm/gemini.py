"""Gemini adapter (Google AI Studio API key for development, Vertex AI for production).

One call = one attempt. Retries and fallback live in resilience.py.
"""

from __future__ import annotations

import asyncio
import enum
import re
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx
from google.genai import errors as genai_errors
from google.genai import types

from app.core import metrics
from app.core.config import Settings
from app.core.logging import get_logger
from app.services.google_client import create_genai_client
from app.services.llm.base import (
    LLMRequest,
    LLMResponse,
    LLMUsage,
    StreamDelta,
    StreamEnd,
    StreamEvent,
)
from app.services.llm.errors import (
    LLMBlockedError,
    LLMConfigurationError,
    LLMError,
    LLMQuotaExhaustedError,
    LLMRateLimitedError,
    LLMRequestError,
    LLMResponseError,
    LLMTimeoutError,
    LLMUnavailableError,
)

logger = get_logger(__name__)

_BLOCKED_FINISH_REASONS = frozenset(
    {"SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII", "RECITATION", "IMAGE_SAFETY"}
)
_RETRY_DELAY = re.compile(r"^(\d+(?:\.\d+)?)s$")


def rate_limit_info(exc: genai_errors.APIError) -> tuple[float | None, bool]:
    """(retry delay in seconds, is a daily quota) from a 429's google.rpc error details."""
    details = exc.details if isinstance(exc.details, dict) else {}
    error = details.get("error", details)
    entries = error.get("details", []) if isinstance(error, dict) else []
    delay: float | None = None
    daily = False
    for entry in entries if isinstance(entries, list) else []:
        if not isinstance(entry, dict):
            continue
        if (match := _RETRY_DELAY.match(str(entry.get("retryDelay", "")))) is not None:
            delay = float(match.group(1))
        for violation in entry.get("violations", []) or []:
            if isinstance(violation, dict) and "PerDay" in str(violation.get("quotaId", "")):
                daily = True
    return delay, daily


def translate_error(exc: BaseException) -> LLMError:
    """Map SDK/transport exceptions to LLMError. Unknown exceptions become LLMResponseError."""
    if isinstance(exc, LLMError):
        return exc
    if isinstance(exc, TimeoutError | httpx.TimeoutException):
        return LLMTimeoutError()
    if isinstance(exc, httpx.TransportError):
        return LLMUnavailableError()
    if isinstance(exc, genai_errors.APIError):
        code = exc.code
        if code == 429:
            delay, daily = rate_limit_info(exc)
            if daily:
                return LLMQuotaExhaustedError()
            return LLMRateLimitedError(retry_after_seconds=delay)
        if code == 504:
            return LLMTimeoutError()
        if code >= 500:
            return LLMUnavailableError()
        if code in (401, 403, 404):
            return LLMConfigurationError()
        return LLMRequestError()
    return LLMResponseError()


def _enum_value(value: Any) -> str:
    if value is None:
        return "UNSPECIFIED"
    if isinstance(value, enum.Enum):
        return str(value.value)
    return str(value)


def _extract_text(response: types.GenerateContentResponse) -> str:
    if not response.candidates:
        return ""
    content = response.candidates[0].content
    if content is None or not content.parts:
        return ""
    # Thought parts are internal reasoning and never shown to farmers.
    return "".join(part.text for part in content.parts if part.text and not part.thought)


def _finish_reason(response: types.GenerateContentResponse) -> str | None:
    if not response.candidates:
        return None
    reason = response.candidates[0].finish_reason
    return _enum_value(reason) if reason is not None else None


def _raise_if_blocked(response: types.GenerateContentResponse) -> None:
    feedback = response.prompt_feedback
    if feedback is not None and feedback.block_reason is not None:
        raise LLMBlockedError()
    if _finish_reason(response) in _BLOCKED_FINISH_REASONS:
        raise LLMBlockedError()


def _usage(response: types.GenerateContentResponse | None) -> LLMUsage:
    meta = response.usage_metadata if response is not None else None
    if meta is None:
        return LLMUsage()
    return LLMUsage(
        input_tokens=meta.prompt_token_count or 0,
        output_tokens=meta.candidates_token_count or 0,
        thinking_tokens=meta.thoughts_token_count or 0,
    )


class GeminiProvider:
    def __init__(
        self,
        *,
        client: Any,
        model: str,
        timeout_seconds: float,
        max_output_tokens: int,
        temperature: float,
        thinking_level: str | None = None,
        input_price_per_million: float = 0.0,
        output_price_per_million: float = 0.0,
    ) -> None:
        self._client = client
        self._model = model
        self._timeout_seconds = timeout_seconds
        self._max_output_tokens = max_output_tokens
        self._temperature = temperature
        self._thinking_level = thinking_level
        self._input_price = input_price_per_million
        self._output_price = output_price_per_million

    @classmethod
    def from_settings(cls, settings: Settings, *, model: str | None = None) -> GeminiProvider:
        return cls(
            client=create_genai_client(settings, timeout_seconds=settings.llm_timeout_seconds),
            model=model or settings.llm_model,
            timeout_seconds=settings.llm_timeout_seconds,
            max_output_tokens=settings.llm_max_output_tokens,
            temperature=settings.llm_temperature,
            thinking_level=settings.llm_thinking_level,
            input_price_per_million=settings.llm_input_price_per_million,
            output_price_per_million=settings.llm_output_price_per_million,
        )

    @property
    def provider_name(self) -> str:
        return "gemini"

    @property
    def model(self) -> str:
        return self._model

    # ------------------------------------------------------------------ request building

    def _config(self, request: LLMRequest) -> types.GenerateContentConfig:
        config = types.GenerateContentConfig(
            system_instruction=request.system_instruction,
            max_output_tokens=request.max_output_tokens or self._max_output_tokens,
            temperature=(
                request.temperature if request.temperature is not None else self._temperature
            ),
            # No tools are used: skip the SDK's automatic function-calling layer.
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        if self._thinking_level is not None:
            config.thinking_config = types.ThinkingConfig(
                thinking_level=types.ThinkingLevel(self._thinking_level.upper())
            )
        return config

    @staticmethod
    def _contents(request: LLMRequest) -> list[types.Content]:
        return [
            types.Content(role=turn.role, parts=[types.Part.from_text(text=turn.text)])
            for turn in request.messages
        ]

    def _cost(self, usage: LLMUsage) -> float:
        billable_output = usage.output_tokens + usage.thinking_tokens
        cost = usage.input_tokens * self._input_price + billable_output * self._output_price
        return round(cost / 1_000_000, 8)

    def _build_response(
        self, text: str, last: types.GenerateContentResponse | None, start: float
    ) -> LLMResponse:
        finish = (_finish_reason(last) if last is not None else None) or "UNSPECIFIED"
        if not text.strip():
            raise LLMResponseError("The AI service returned an empty response.")
        usage = _usage(last)
        response = LLMResponse(
            text=text,
            provider=self.provider_name,
            model=self._model,
            usage=usage,
            latency_ms=round((time.perf_counter() - start) * 1000, 1),
            finish_reason=finish,
            truncated=finish == "MAX_TOKENS",
            cost_usd=self._cost(usage),
        )
        metrics.llm_calls.labels("chat", self._model, "ok").inc()
        metrics.llm_duration.labels("chat", self._model).observe(response.latency_ms / 1000)
        for kind, count in (
            ("input", usage.input_tokens),
            ("output", usage.output_tokens),
            ("thinking", usage.thinking_tokens),
        ):
            metrics.llm_tokens.labels("chat", self._model, kind).inc(count)
        metrics.llm_cost.labels(self._model).inc(response.cost_usd)
        logger.info(
            "llm_call",
            provider=response.provider,
            model=response.model,
            latency_ms=response.latency_ms,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            thinking_tokens=usage.thinking_tokens,
            cost_usd=response.cost_usd,
            finish_reason=finish,
        )
        return response

    def _log_failure(self, error: LLMError, exc: BaseException, start: float) -> None:
        metrics.llm_calls.labels("chat", self._model, error.code).inc()
        logger.warning(
            "llm_call_failed",
            provider=self.provider_name,
            model=self._model,
            error_code=error.code,
            latency_ms=round((time.perf_counter() - start) * 1000, 1),
            upstream_error=f"{type(exc).__name__}: {str(exc)[:300]}",
        )

    # ------------------------------------------------------------------ calls

    async def generate(self, request: LLMRequest) -> LLMResponse:
        start = time.perf_counter()
        try:
            async with asyncio.timeout(self._timeout_seconds):
                response = await self._client.aio.models.generate_content(
                    model=self._model,
                    contents=self._contents(request),
                    config=self._config(request),
                )
            _raise_if_blocked(response)
            return self._build_response(_extract_text(response), response, start)
        except Exception as exc:
            error = translate_error(exc)
            self._log_failure(error, exc, start)
            raise error from exc

    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamEvent]:
        start = time.perf_counter()
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._timeout_seconds
        chunks: list[str] = []
        last: types.GenerateContentResponse | None = None
        try:
            iterator = await asyncio.wait_for(
                self._client.aio.models.generate_content_stream(
                    model=self._model,
                    contents=self._contents(request),
                    config=self._config(request),
                ),
                timeout=self._timeout_seconds,
            )
            while True:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    raise TimeoutError
                try:
                    chunk = await asyncio.wait_for(anext(iterator), timeout=remaining)
                except StopAsyncIteration:
                    break
                _raise_if_blocked(chunk)
                if chunk.usage_metadata is not None or chunk.candidates:
                    last = chunk
                text = _extract_text(chunk)
                if text:
                    chunks.append(text)
                    yield StreamDelta(text=text)
            final = self._build_response("".join(chunks), last, start)
        except Exception as exc:
            error = translate_error(exc)
            self._log_failure(error, exc, start)
            raise error from exc
        yield StreamEnd(response=final)

    async def check(self) -> None:
        # Model metadata lookup: verifies key/credentials and model name without generating tokens.
        async with asyncio.timeout(self._timeout_seconds):
            await self._client.aio.models.get(model=self._model)

    async def aclose(self) -> None:
        aclose = getattr(self._client.aio, "aclose", None)
        if aclose is not None:
            await aclose()
