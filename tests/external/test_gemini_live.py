"""Live Gemini calls using the real key in .env. Uses free-tier quota; skipped by default.

Run:
    $env:KRISHIMITRA_RUN_EXTERNAL = "1"; pytest tests/external -v
"""

from __future__ import annotations

import os

import pytest

from app.core.config import Settings
from app.services.language import detect_language
from app.services.llm import ChatTurn, LLMRequest, StreamDelta, StreamEnd, build_llm
from app.services.prompts import build_system_prompt

pytestmark = [
    pytest.mark.external,
    pytest.mark.skipif(
        os.environ.get("KRISHIMITRA_RUN_EXTERNAL") != "1",
        reason="set KRISHIMITRA_RUN_EXTERNAL=1 to call the real LLM",
    ),
]


def live_settings() -> Settings:
    return Settings(_env_file=".env")  # type: ignore[call-arg]


async def test_live_check() -> None:
    llm = build_llm(live_settings())
    try:
        await llm.check()
    finally:
        await llm.aclose()


async def test_live_generate_telugu() -> None:
    llm = build_llm(live_settings())
    try:
        response = await llm.generate(
            LLMRequest(
                messages=(ChatTurn(role="user", text="టమాటాలో ఎర్లీ బ్లైట్ లక్షణాలు ఏమిటి?"),),
                system_instruction=build_system_prompt("te"),
                max_output_tokens=400,
            )
        )
    finally:
        await llm.aclose()
    assert detect_language(response.text) == "te"
    assert response.usage.input_tokens > 0
    assert response.usage.output_tokens > 0


async def test_live_stream_english() -> None:
    llm = build_llm(live_settings())
    try:
        events = [
            e
            async for e in llm.stream(
                LLMRequest(
                    messages=(
                        ChatTurn(role="user", text="Name two signs of nitrogen deficiency."),
                    ),
                    system_instruction=build_system_prompt("en"),
                    max_output_tokens=200,
                )
            )
        ]
    finally:
        await llm.aclose()
    assert any(isinstance(e, StreamDelta) for e in events)
    end = events[-1]
    assert isinstance(end, StreamEnd)
    assert end.response.text
