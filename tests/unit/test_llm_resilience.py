from __future__ import annotations

import pytest

from app.core.config import Settings
from app.services.llm import ChatTurn, LLMRequest, StreamDelta, StreamEnd, build_llm
from app.services.llm.errors import (
    LLMBlockedError,
    LLMRateLimitedError,
    LLMTimeoutError,
    LLMUnavailableError,
)
from app.services.llm.gemini import GeminiProvider
from app.services.llm.resilience import FallbackLLM, RetryingLLM
from tests.fakes import FakeLLM

REQUEST = LLMRequest(messages=(ChatTurn(role="user", text="hello"),))


class RecordingSleep:
    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, delay: float) -> None:
        self.delays.append(delay)


def retrying(fake: FakeLLM, max_retries: int = 2) -> tuple[RetryingLLM, RecordingSleep]:
    sleep = RecordingSleep()
    return RetryingLLM(fake, max_retries=max_retries, sleep=sleep), sleep


# ---------------------------------------------------------------- retry


async def test_retry_then_success() -> None:
    fake = FakeLLM(outcomes=[LLMUnavailableError(), LLMRateLimitedError(), "ok"])
    llm, sleep = retrying(fake)
    response = await llm.generate(REQUEST)
    assert response.text == "ok"
    assert response.attempts == 3
    assert len(sleep.delays) == 2
    assert sleep.delays[1] > sleep.delays[0] * 1.5  # exponential backoff


async def test_retry_gives_up_after_max() -> None:
    fake = FakeLLM(outcomes=[LLMUnavailableError()])
    llm, sleep = retrying(fake, max_retries=2)
    with pytest.raises(LLMUnavailableError):
        await llm.generate(REQUEST)
    assert len(fake.requests) == 3
    assert len(sleep.delays) == 2


@pytest.mark.parametrize("error", [LLMBlockedError(), LLMTimeoutError()])
async def test_no_retry_for_non_retryable(error: Exception) -> None:
    fake = FakeLLM(outcomes=[error, "ok"])  # type: ignore[list-item]
    llm, sleep = retrying(fake)
    with pytest.raises(type(error)):
        await llm.generate(REQUEST)
    assert len(fake.requests) == 1
    assert sleep.delays == []


async def test_stream_retry_before_first_delta() -> None:
    fake = FakeLLM(outcomes=[LLMUnavailableError(), "hello world"])
    llm, _ = retrying(fake)
    events = [e async for e in llm.stream(REQUEST)]
    assert next(e for e in events if isinstance(e, StreamDelta)).text == "hello "
    end = events[-1]
    assert isinstance(end, StreamEnd)
    assert end.response.attempts == 2


class FailsMidStream(FakeLLM):
    async def stream(self, request: LLMRequest):  # type: ignore[no-untyped-def]
        self.requests.append(request)
        yield StreamDelta(text="partial ")
        raise LLMUnavailableError()


async def test_stream_no_retry_after_text_sent() -> None:
    fake = FailsMidStream()
    llm, sleep = retrying(fake)
    with pytest.raises(LLMUnavailableError):
        async for _ in llm.stream(REQUEST):
            pass
    assert len(fake.requests) == 1
    assert sleep.delays == []


# ---------------------------------------------------------------- fallback


async def test_fallback_on_eligible_error() -> None:
    primary = FakeLLM(outcomes=[LLMTimeoutError()], model_name="primary")
    fallback = FakeLLM(outcomes=["from fallback"], model_name="fallback")
    response = await FallbackLLM(primary, fallback).generate(REQUEST)
    assert response.text == "from fallback"
    assert response.model == "fallback"


async def test_no_fallback_when_blocked() -> None:
    primary = FakeLLM(outcomes=[LLMBlockedError()])
    fallback = FakeLLM(outcomes=["should not be used"])
    with pytest.raises(LLMBlockedError):
        await FallbackLLM(primary, fallback).generate(REQUEST)
    assert fallback.requests == []


async def test_primary_success_skips_fallback() -> None:
    primary = FakeLLM(outcomes=["primary answer"])
    fallback = FakeLLM()
    response = await FallbackLLM(primary, fallback).generate(REQUEST)
    assert response.text == "primary answer"
    assert fallback.requests == []


async def test_stream_fallback_before_first_delta() -> None:
    primary = FakeLLM(outcomes=[LLMUnavailableError()])
    fallback = FakeLLM(outcomes=["fallback stream"], model_name="fallback")
    events = [e async for e in FallbackLLM(primary, fallback).stream(REQUEST)]
    end = events[-1]
    assert isinstance(end, StreamEnd)
    assert end.response.model == "fallback"


async def test_stream_no_fallback_after_text_sent() -> None:
    fallback = FakeLLM()
    with pytest.raises(LLMUnavailableError):
        async for _ in FallbackLLM(FailsMidStream(), fallback).stream(REQUEST):
            pass
    assert fallback.requests == []


async def test_fallback_check_and_close() -> None:
    primary, fallback = FakeLLM(), FakeLLM()
    llm = FallbackLLM(primary, fallback)
    await llm.check()
    await llm.aclose()
    assert primary.closed and fallback.closed


# ---------------------------------------------------------------- factory


def settings(**overrides: object) -> Settings:
    params: dict[str, object] = {"google_genai_use_vertexai": False, "google_api_key": "k"}
    params.update(overrides)
    return Settings(_env_file=None, **params)  # type: ignore[arg-type]


def test_factory_primary_only() -> None:
    llm = build_llm(settings())
    assert isinstance(llm, RetryingLLM)
    assert isinstance(llm._inner, GeminiProvider)
    assert llm.model == "gemini-3.5-flash-lite"


def test_factory_with_fallback() -> None:
    llm = build_llm(settings(llm_fallback_provider="gemini", llm_fallback_model="gemini-3.5-flash"))
    assert isinstance(llm, FallbackLLM)
    assert llm._fallback.model == "gemini-3.5-flash"


def test_factory_rejects_unimplemented_provider() -> None:
    with pytest.raises(ValueError, match="not implemented"):
        build_llm(settings(llm_provider="anthropic"))
