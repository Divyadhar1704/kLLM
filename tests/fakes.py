"""Test doubles. Unit tests never call real AI services."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from app.services.llm.base import (
    LLMRequest,
    LLMResponse,
    LLMUsage,
    StreamDelta,
    StreamEnd,
    StreamEvent,
)
from app.services.llm.errors import LLMError


@dataclass
class FakeLLM:
    """Scripted LLM: each call consumes the next outcome (a reply string or an LLMError)."""

    outcomes: list[str | LLMError] = field(default_factory=lambda: ["Fake answer."])
    model_name: str = "fake-model"
    requests: list[LLMRequest] = field(default_factory=list)
    check_error: Exception | None = None
    closed: bool = False

    @property
    def provider_name(self) -> str:
        return "fake"

    @property
    def model(self) -> str:
        return self.model_name

    def _next(self, request: LLMRequest) -> str:
        self.requests.append(request)
        outcome = self.outcomes.pop(0) if len(self.outcomes) > 1 else self.outcomes[0]
        if isinstance(outcome, LLMError):
            raise outcome
        return outcome

    def _response(self, text: str) -> LLMResponse:
        return LLMResponse(
            text=text,
            provider="fake",
            model=self.model_name,
            usage=LLMUsage(input_tokens=10, output_tokens=len(text.split()), thinking_tokens=0),
            latency_ms=1.0,
            finish_reason="STOP",
        )

    async def generate(self, request: LLMRequest) -> LLMResponse:
        return self._response(self._next(request))

    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamEvent]:
        text = self._next(request)
        # Pieces join back to exactly `text`, like the real provider's stream.
        words = text.split(" ")
        for index, word in enumerate(words):
            yield StreamDelta(text=word if index == len(words) - 1 else word + " ")
        yield StreamEnd(response=self._response(text))

    async def check(self) -> None:
        if self.check_error is not None:
            raise self.check_error

    async def aclose(self) -> None:
        self.closed = True
