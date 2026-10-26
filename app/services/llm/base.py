"""Provider-independent LLM types. Services depend only on this module, never on a vendor SDK."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Literal, Protocol

Role = Literal["user", "model"]


@dataclass(frozen=True, slots=True)
class ChatTurn:
    role: Role
    text: str


@dataclass(frozen=True, slots=True)
class LLMRequest:
    messages: tuple[ChatTurn, ...]
    system_instruction: str | None = None
    max_output_tokens: int | None = None
    temperature: float | None = None


@dataclass(frozen=True, slots=True)
class LLMUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens + self.thinking_tokens


@dataclass(frozen=True, slots=True)
class LLMResponse:
    text: str
    provider: str
    model: str
    usage: LLMUsage
    latency_ms: float
    finish_reason: str
    truncated: bool = False
    cost_usd: float = 0.0
    attempts: int = 1


@dataclass(frozen=True, slots=True)
class StreamDelta:
    text: str


@dataclass(frozen=True, slots=True)
class StreamEnd:
    """Final stream event: full text, usage and metadata."""

    response: LLMResponse


StreamEvent = StreamDelta | StreamEnd


class LLMProvider(Protocol):
    @property
    def provider_name(self) -> str: ...

    @property
    def model(self) -> str: ...

    async def generate(self, request: LLMRequest) -> LLMResponse: ...

    def stream(self, request: LLMRequest) -> AsyncIterator[StreamEvent]: ...

    async def check(self) -> None:
        """Cheap reachability/config check for readiness probes. Must not generate tokens."""
        ...

    async def aclose(self) -> None: ...
