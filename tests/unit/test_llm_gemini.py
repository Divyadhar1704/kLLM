"""GeminiProvider against a fake SDK client: parsing, error mapping, timeouts, streaming."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from google.genai import errors as genai_errors
from google.genai import types

from app.core.config import Settings
from app.services.llm import ChatTurn, LLMRequest, StreamDelta, StreamEnd
from app.services.llm.errors import (
    LLMBlockedError,
    LLMConfigurationError,
    LLMRateLimitedError,
    LLMRequestError,
    LLMResponseError,
    LLMTimeoutError,
    LLMUnavailableError,
)
from app.services.llm.gemini import GeminiProvider

REQUEST = LLMRequest(
    messages=(ChatTurn(role="user", text="What is early blight?"),),
    system_instruction="You are KrishiMitra.",
)


def gemini_response(
    text: str | None = "Early blight is a fungal disease.",
    *,
    finish_reason: str = "STOP",
    thought: str | None = None,
    block_reason: str | None = None,
    usage: tuple[int, int, int | None] | None = (12, 8, None),
) -> types.GenerateContentResponse:
    parts: list[dict[str, Any]] = []
    if thought:
        parts.append({"text": thought, "thought": True})
    if text is not None:
        parts.append({"text": text})
    data: dict[str, Any] = {}
    if block_reason:
        data["prompt_feedback"] = {"block_reason": block_reason}
    else:
        data["candidates"] = [
            {"content": {"role": "model", "parts": parts}, "finish_reason": finish_reason}
        ]
    if usage:
        data["usage_metadata"] = {
            "prompt_token_count": usage[0],
            "candidates_token_count": usage[1],
            "thoughts_token_count": usage[2],
        }
    return types.GenerateContentResponse.model_validate(data)


def api_error(code: int) -> genai_errors.APIError:
    payload = {"error": {"code": code, "message": f"upstream {code}", "status": "X"}}
    cls = genai_errors.ServerError if code >= 500 else genai_errors.ClientError
    return cls(code, payload)


class FakeModels:
    def __init__(self) -> None:
        self.result: Any = gemini_response()
        self.stream_chunks: list[Any] = []
        self.delay = 0.0
        self.calls: list[dict[str, Any]] = []
        self.get_calls: list[str] = []

    async def generate_content(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.delay:
            await asyncio.sleep(self.delay)
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result

    async def generate_content_stream(self, **kwargs: Any) -> AsyncIterator[Any]:
        self.calls.append(kwargs)
        chunks = self.stream_chunks
        delay = self.delay

        async def iterate() -> AsyncIterator[Any]:
            for chunk in chunks:
                if delay:
                    await asyncio.sleep(delay)
                if isinstance(chunk, BaseException):
                    raise chunk
                yield chunk

        return iterate()

    async def get(self, *, model: str) -> object:
        self.get_calls.append(model)
        return object()


def provider(
    models: FakeModels, *, timeout: float = 1.0, thinking: str | None = None
) -> GeminiProvider:
    client = SimpleNamespace(aio=SimpleNamespace(models=models))
    return GeminiProvider(
        client=client,
        model="gemini-test",
        timeout_seconds=timeout,
        max_output_tokens=512,
        temperature=0.2,
        thinking_level=thinking,
        input_price_per_million=0.10,
        output_price_per_million=0.40,
    )


# ---------------------------------------------------------------- generate


async def test_generate_success_parses_text_usage_and_cost() -> None:
    models = FakeModels()
    response = await provider(models).generate(REQUEST)
    assert response.text == "Early blight is a fungal disease."
    assert response.provider == "gemini"
    assert response.model == "gemini-test"
    assert response.usage.input_tokens == 12
    assert response.usage.output_tokens == 8
    assert response.finish_reason == "STOP"
    assert not response.truncated
    assert response.cost_usd == pytest.approx((12 * 0.10 + 8 * 0.40) / 1_000_000)


async def test_request_is_built_correctly() -> None:
    models = FakeModels()
    await provider(models).generate(REQUEST)
    call = models.calls[0]
    assert call["model"] == "gemini-test"
    contents = call["contents"]
    assert contents[0].role == "user"
    assert contents[0].parts[0].text == "What is early blight?"
    config = call["config"]
    assert config.system_instruction == "You are KrishiMitra."
    assert config.max_output_tokens == 512
    assert config.temperature == 0.2
    assert config.thinking_config is None
    assert config.automatic_function_calling.disable is True


async def test_thinking_level_applied() -> None:
    models = FakeModels()
    await provider(models, thinking="low").generate(REQUEST)
    assert models.calls[0]["config"].thinking_config.thinking_level == types.ThinkingLevel.LOW


async def test_thought_parts_excluded_and_billed() -> None:
    models = FakeModels()
    models.result = gemini_response("Answer.", thought="secret reasoning", usage=(10, 5, 20))
    response = await provider(models).generate(REQUEST)
    assert response.text == "Answer."
    assert response.usage.thinking_tokens == 20
    assert response.cost_usd == pytest.approx((10 * 0.10 + 25 * 0.40) / 1_000_000)


async def test_max_tokens_marks_truncated() -> None:
    models = FakeModels()
    models.result = gemini_response("Partial answer", finish_reason="MAX_TOKENS")
    response = await provider(models).generate(REQUEST)
    assert response.truncated


@pytest.mark.parametrize("reason", ["SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST"])
async def test_blocked_finish_reason(reason: str) -> None:
    models = FakeModels()
    models.result = gemini_response(None, finish_reason=reason)
    with pytest.raises(LLMBlockedError):
        await provider(models).generate(REQUEST)


async def test_blocked_prompt() -> None:
    models = FakeModels()
    models.result = gemini_response(block_reason="SAFETY")
    with pytest.raises(LLMBlockedError):
        await provider(models).generate(REQUEST)


async def test_empty_response() -> None:
    models = FakeModels()
    models.result = gemini_response("   ")
    with pytest.raises(LLMResponseError):
        await provider(models).generate(REQUEST)


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (api_error(429), LLMRateLimitedError),
        (api_error(500), LLMUnavailableError),
        (api_error(503), LLMUnavailableError),
        (api_error(504), LLMTimeoutError),
        (api_error(400), LLMRequestError),
        (api_error(401), LLMConfigurationError),
        (api_error(403), LLMConfigurationError),
        (api_error(404), LLMConfigurationError),
        (httpx.ConnectError("refused"), LLMUnavailableError),
        (httpx.ReadTimeout("slow"), LLMTimeoutError),
        (ValueError("weird SDK failure"), LLMResponseError),
    ],
)
async def test_error_mapping(exc: BaseException, expected: type[Exception]) -> None:
    models = FakeModels()
    models.result = exc
    with pytest.raises(expected):
        await provider(models).generate(REQUEST)


async def test_rate_limit_sets_retry_after() -> None:
    models = FakeModels()
    models.result = api_error(429)
    with pytest.raises(LLMRateLimitedError) as exc_info:
        await provider(models).generate(REQUEST)
    assert exc_info.value.headers == {"Retry-After": "30"}


async def test_timeout() -> None:
    models = FakeModels()
    models.delay = 0.5
    with pytest.raises(LLMTimeoutError):
        await provider(models, timeout=0.05).generate(REQUEST)


async def test_check_uses_metadata_lookup() -> None:
    models = FakeModels()
    await provider(models).check()
    assert models.get_calls == ["gemini-test"]
    assert models.calls == []  # no generation


# ---------------------------------------------------------------- stream


async def test_stream_yields_deltas_then_end() -> None:
    models = FakeModels()
    models.stream_chunks = [
        gemini_response("Early ", usage=None),
        gemini_response("blight.", usage=(12, 4, None)),
    ]
    events = [e async for e in provider(models).stream(REQUEST)]
    assert events[:2] == [StreamDelta(text="Early "), StreamDelta(text="blight.")]
    end = events[-1]
    assert isinstance(end, StreamEnd)
    assert end.response.text == "Early blight."
    assert end.response.usage.input_tokens == 12
    assert end.response.finish_reason == "STOP"


async def test_stream_error_mid_way_is_translated() -> None:
    models = FakeModels()
    models.stream_chunks = [gemini_response("Early ", usage=None), api_error(503)]
    events: list[object] = []
    with pytest.raises(LLMUnavailableError):
        async for event in provider(models).stream(REQUEST):
            events.append(event)
    assert events == [StreamDelta(text="Early ")]


async def test_stream_blocked() -> None:
    models = FakeModels()
    models.stream_chunks = [gemini_response(None, finish_reason="SAFETY")]
    with pytest.raises(LLMBlockedError):
        async for _ in provider(models).stream(REQUEST):
            pass


async def test_stream_empty() -> None:
    models = FakeModels()
    models.stream_chunks = []
    with pytest.raises(LLMResponseError):
        async for _ in provider(models).stream(REQUEST):
            pass


async def test_stream_timeout() -> None:
    models = FakeModels()
    models.delay = 0.5
    models.stream_chunks = [gemini_response("slow")]
    with pytest.raises(LLMTimeoutError):
        async for _ in provider(models, timeout=0.05).stream(REQUEST):
            pass


# ---------------------------------------------------------------- construction


def test_from_settings_api_key() -> None:
    settings = Settings(_env_file=None, google_genai_use_vertexai=False, google_api_key="k")  # type: ignore[call-arg]
    gemini = GeminiProvider.from_settings(settings)
    assert gemini.model == settings.llm_model


def test_from_settings_requires_api_key() -> None:
    settings = Settings(_env_file=None, google_genai_use_vertexai=False)  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="GOOGLE_API_KEY"):
        GeminiProvider.from_settings(settings)


def test_from_settings_vertex_requires_project() -> None:
    settings = Settings(_env_file=None, google_genai_use_vertexai=True)  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="GCP_PROJECT_ID"):
        GeminiProvider.from_settings(settings)
