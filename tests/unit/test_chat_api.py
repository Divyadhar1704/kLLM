from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app
from app.services.llm.errors import (
    LLMBlockedError,
    LLMRateLimitedError,
    LLMTimeoutError,
    LLMUnavailableError,
)
from tests.fakes import FakeLLM

URL = "/api/v1/chat"


def build(fake: FakeLLM | None = None, **overrides: Any) -> tuple[TestClient, FakeLLM]:
    fake = fake or FakeLLM(outcomes=["Early blight causes brown rings on older leaves."])
    app: FastAPI = create_app(Settings(_env_file=None, **overrides), llm=fake)  # type: ignore[arg-type]
    return TestClient(app), fake


def test_english_question() -> None:
    client, fake = build()
    response = client.post(
        URL, json={"message": "What are the symptoms of early blight in tomato?", "language": "en"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["response"] == "Early blight causes brown rings on older leaves."
    assert body["language"] == "en"
    assert body["language_detected"] is False
    assert body["model"] == "fake-model"
    assert len(body["conversation_id"]) == 32
    assert body["processing_time"] >= 0
    assert body["truncated"] is False
    assert body["usage"]["input_tokens"] == 10
    sent = fake.requests[0]
    assert sent.messages[0].text == (
        "DETECTED FROM THE QUESTION: crop: tomato. Answer for this crop and problem only; "
        "ignore source rows about other crops.\n\n"
        "What are the symptoms of early blight in tomato?"
    )
    assert "Reply ONLY in English" in (sent.system_instruction or "")


def test_telugu_auto_detected() -> None:
    client, fake = build()
    response = client.post(URL, json={"message": "టమాటాలో ఎర్లీ బ్లైట్ లక్షణాలు ఏమిటి?"})
    body = response.json()
    assert body["language"] == "te"
    assert body["language_detected"] is True
    assert "Telugu" in (fake.requests[0].system_instruction or "")


def test_explicit_language_overrides_detection() -> None:
    client, fake = build()
    response = client.post(URL, json={"message": "What is DAP?", "language": "kn"})
    assert response.json()["language"] == "kn"
    assert "Kannada" in (fake.requests[0].system_instruction or "")


def test_conversation_id_echoed() -> None:
    client, _ = build()
    response = client.post(URL, json={"message": "hi", "conversation_id": "conv_12345678"})
    assert response.json()["conversation_id"] == "conv_12345678"


def test_message_whitespace_stripped() -> None:
    client, fake = build()
    client.post(URL, json={"message": "   What is DAP?   "})
    assert fake.requests[0].messages[0].text == "What is DAP?"


def test_validation_errors() -> None:
    client, fake = build()
    assert client.post(URL, json={"message": "   "}).status_code == 422
    assert client.post(URL, json={}).status_code == 422
    assert client.post(URL, json={"message": "hi", "language": "ta"}).status_code == 422
    bad_id = client.post(URL, json={"message": "hi", "conversation_id": "../etc/passwd"})
    assert bad_id.status_code == 422
    assert fake.requests == []


def test_message_too_long() -> None:
    client, fake = build(max_message_chars=20)
    response = client.post(URL, json={"message": "x" * 21})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "bad_request"
    assert fake.requests == []


def test_llm_rate_limited() -> None:
    client, _ = build(FakeLLM(outcomes=[LLMRateLimitedError(headers={"Retry-After": "30"})]))
    response = client.post(URL, json={"message": "hi"})
    assert response.status_code == 429
    assert response.json()["error"]["code"] == "llm_rate_limited"
    assert response.headers["Retry-After"] == "30"


def test_llm_errors_map_to_status() -> None:
    cases = [
        (LLMUnavailableError(), 503, "llm_unavailable"),
        (LLMTimeoutError(), 504, "llm_timeout"),
        (LLMBlockedError(), 422, "content_blocked"),
    ]
    for error, status, code in cases:
        client, _ = build(FakeLLM(outcomes=[error]))
        response = client.post(URL, json={"message": "hi"})
        assert response.status_code == status
        assert response.json()["error"]["code"] == code


def test_llm_not_configured() -> None:
    app = create_app(Settings(_env_file=None))  # type: ignore[call-arg]
    response = TestClient(app).post(URL, json={"message": "hi"})
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "service_unavailable"


def test_readiness_reports_llm() -> None:
    client, _ = build(health_cache_seconds=0)
    body = client.get("/health/ready").json()
    llm = next(c for c in body["checks"] if c["name"] == "llm")
    assert llm["status"] == "ok"
    assert llm["critical"] is False


def test_readiness_degraded_when_llm_check_fails() -> None:
    client, _ = build(FakeLLM(check_error=ConnectionError("down")), health_cache_seconds=0)
    response = client.get("/health/ready")
    assert response.status_code == 200
    assert response.json()["status"] == "degraded"
