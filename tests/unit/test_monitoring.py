"""Metrics, Cloud Logging trace correlation and Error Reporting format (no cloud calls)."""

from __future__ import annotations

import json
import logging
import socket
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY

from app.core.config import Settings
from app.core.logging import error_reporting, get_logger, setup_logging
from app.main import create_app
from app.rag.retriever import RetrievalResult
from app.services.llm.errors import LLMRateLimitedError
from app.services.llm.resilience import FallbackLLM, RetryingLLM
from tests.fakes import FakeLLM
from tests.unit.test_chat_rag import FakeRetriever, FakeSafety, chunk
from tests.unit.test_dev_console import SECURE_PRODUCTION
from tests.unit.test_llm_gemini import REQUEST, FakeModels, api_error, provider
from tests.unit.test_vision import GOOD_JSON, diagnoser, encode, leaf_image


def value(metric: str, **labels: str) -> float:
    return REGISTRY.get_sample_value(metric, labels) or 0.0


def build(outcomes: list[str] | None = None, **overrides: Any) -> TestClient:
    app = create_app(
        Settings(_env_file=None, enable_dev_ui=True, **overrides),
        llm=FakeLLM(outcomes=outcomes or ["Spray mancozeb [1]."]),
    )
    app.state.retriever = FakeRetriever(
        RetrievalResult(chunks=[chunk("c1", "Spray mancozeb")], crops=["tomato"], total_ms=120.0)
    )
    app.state.safety = FakeSafety()
    return TestClient(app)


# ---------------------------------------------------------------- /metrics


def test_metrics_endpoint_local_only() -> None:
    response = build().get("/metrics")
    assert response.status_code == 200
    assert "krishimitra_http_requests_total" in response.text
    assert response.headers["content-type"].startswith("text/plain")
    deployed = create_app(Settings(_env_file=None, **SECURE_PRODUCTION), llm=FakeLLM())
    assert TestClient(deployed).get("/metrics").status_code == 404  # never on the public port


def test_metrics_port_serves_loopback_only() -> None:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    with build(metrics_port=port):  # lifespan starts and stops the metrics server
        response = httpx.get(f"http://127.0.0.1:{port}/metrics", timeout=5)
        assert response.status_code == 200 and "krishimitra_build_info" in response.text


def test_http_metrics_use_route_templates() -> None:
    client = build()
    before = value(
        "krishimitra_http_requests_total", method="POST", route="/api/v1/chat", status="200"
    )
    unmatched = value(
        "krishimitra_http_requests_total", method="GET", route="unmatched", status="404"
    )
    client.post("/api/v1/chat", json={"message": "spots on tomato", "language": "en"})
    client.get("/no/such/path/12345")
    assert (
        value("krishimitra_http_requests_total", method="POST", route="/api/v1/chat", status="200")
        == before + 1
    )
    assert (
        value("krishimitra_http_requests_total", method="GET", route="unmatched", status="404")
        == unmatched + 1
    )
    assert value("krishimitra_http_request_duration_seconds_count", route="/api/v1/chat") > 0


# ---------------------------------------------------------------- answers


def test_answer_metrics() -> None:
    client = build(["Apply phorate granules."], answer_cache_size=10)
    labels = {"kind": "text", "language": "en", "grounded": "true", "streamed": "false"}
    answers = value("krishimitra_answers_total", **labels)
    warned = value("krishimitra_safety_warnings_total", language="en")
    misses = value("krishimitra_answer_cache_total", result="miss")
    hits = value("krishimitra_answer_cache_total", result="hit")
    searches = value("krishimitra_retrieval_duration_seconds_count")
    body = {"message": "what to spray on tomato?", "language": "en"}
    client.post("/api/v1/chat", json=body)
    client.post("/api/v1/chat", json=body)  # same question: from the cache
    # Both are answers sent: the generated one and the one served from the cache.
    assert value("krishimitra_answers_total", **labels) == answers + 2
    assert value("krishimitra_safety_warnings_total", language="en") == warned + 1
    assert value("krishimitra_answer_cache_total", result="miss") == misses + 1
    assert value("krishimitra_answer_cache_total", result="hit") == hits + 1
    assert value("krishimitra_retrieval_duration_seconds_count") == searches + 1


def test_language_retry_metric() -> None:
    client = build(["Spray mancozeb now [1].", "మాంకోజెబ్ పిచికారీ చేయండి [1]."])
    before = value("krishimitra_language_retries_total", language="te")
    client.post("/api/v1/chat", json={"message": "ఏ మందు?", "language": "te"})
    assert value("krishimitra_language_retries_total", language="te") == before + 1


def test_rate_limit_and_dependency_metrics() -> None:
    client = build(rate_limit_per_minute=1)
    before = value("krishimitra_rate_limited_total", limit="user_minute")
    for _ in range(2):
        client.post("/api/v1/chat", json={"message": "spots", "language": "en"})
    assert value("krishimitra_rate_limited_total", limit="user_minute") == before + 1
    client.get("/health/ready")
    assert value("krishimitra_dependency_up", name="llm") == 1.0


# ---------------------------------------------------------------- AI calls


async def test_llm_call_metrics_from_the_provider() -> None:
    labels = {"purpose": "chat", "model": "gemini-test"}
    calls = value("krishimitra_llm_calls_total", outcome="ok", **labels)
    tokens = value("krishimitra_llm_tokens_total", kind="input", **labels)
    cost = value("krishimitra_llm_cost_usd_total", model="gemini-test")
    await provider(FakeModels()).generate(REQUEST)
    assert value("krishimitra_llm_calls_total", outcome="ok", **labels) == calls + 1
    assert value("krishimitra_llm_tokens_total", kind="input", **labels) == tokens + 12
    assert value("krishimitra_llm_cost_usd_total", model="gemini-test") == pytest.approx(
        cost + (12 * 0.10 + 8 * 0.40) / 1_000_000
    )
    failing = FakeModels()
    failing.result = api_error(429)
    failed = value("krishimitra_llm_calls_total", outcome="llm_rate_limited", **labels)
    with pytest.raises(LLMRateLimitedError):
        await provider(failing).generate(REQUEST)
    assert value("krishimitra_llm_calls_total", outcome="llm_rate_limited", **labels) == failed + 1


async def test_retry_and_fallback_metrics() -> None:
    error = LLMRateLimitedError("busy")
    primary = FakeLLM(outcomes=[error, error, error], model_name="primary-m")
    fallback = FakeLLM(outcomes=["ok"], model_name="fallback-m")

    async def no_sleep(_: float) -> None:
        return None

    llm = FallbackLLM(RetryingLLM(primary, max_retries=1, sleep=no_sleep), fallback)
    retries = value("krishimitra_llm_retries_total", model="primary-m", error="llm_rate_limited")
    switches = value(
        "krishimitra_llm_fallbacks_total", from_model="primary-m", error="llm_rate_limited"
    )
    await llm.generate(REQUEST)
    assert (
        value("krishimitra_llm_retries_total", model="primary-m", error="llm_rate_limited")
        == retries + 1
    )
    assert (
        value("krishimitra_llm_fallbacks_total", from_model="primary-m", error="llm_rate_limited")
        == switches + 1
    )


async def test_vision_metrics() -> None:
    service, _ = diagnoser([GOOD_JSON])
    diagnosed = value("krishimitra_diagnoses_total", source="photo", status="confident")
    calls = value("krishimitra_llm_calls_total", purpose="vision", model="primary", outcome="ok")
    retakes = value("krishimitra_diagnoses_total", source="photo", status="retake_photo")
    await service.diagnose(encode(leaf_image()), crop="tomato")
    await service.diagnose(encode(leaf_image().point(lambda v: v // 8)))  # too dark: no AI
    assert value("krishimitra_diagnoses_total", source="photo", status="confident") == diagnosed + 1
    assert (
        value("krishimitra_llm_calls_total", purpose="vision", model="primary", outcome="ok")
        == calls + 1
    )
    assert (
        value("krishimitra_diagnoses_total", source="photo", status="retake_photo") == retakes + 1
    )


# ---------------------------------------------------------------- Cloud Logging


def json_logs(capsys: pytest.CaptureFixture[str]) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith("{")
    ]


def test_logs_carry_the_cloud_trace(capsys: pytest.CaptureFixture[str]) -> None:
    client = build(log_json=True, gcp_project_id="krishimitra-prod")
    capsys.readouterr()
    client.get("/", headers={"X-Cloud-Trace-Context": "105445aa7843bc8bf206b12000100000/1;o=1"})
    access = next(e for e in json_logs(capsys) if e["event"] == "request_completed")
    assert access["logging.googleapis.com/trace"] == (
        "projects/krishimitra-prod/traces/105445aa7843bc8bf206b12000100000"
    )
    assert access["logging.googleapis.com/trace_sampled"] is True
    client.get("/")  # no trace header: no trace fields
    plain = next(e for e in json_logs(capsys) if e["event"] == "request_completed")
    assert "logging.googleapis.com/trace" not in plain


def test_errors_formatted_for_error_reporting(capsys: pytest.CaptureFixture[str]) -> None:
    setup_logging(Settings(_env_file=None, log_json=True))
    logger = get_logger("app.test_errors")
    capsys.readouterr()
    try:
        raise ValueError("boom")
    except ValueError:
        logger.error("something_failed", exc_info=True)
    logger.warning("just_a_warning")
    records = json_logs(capsys)
    error = next(r for r in records if r["event"] == "something_failed")
    assert error["@type"].endswith("ReportedErrorEvent")
    assert error["serviceContext"]["service"] == "krishimitra-api"
    assert "ValueError: boom" in error["stack_trace"]
    warning = next(r for r in records if r["event"] == "just_a_warning")
    assert "@type" not in warning
    logging.getLogger().handlers.clear()


def test_error_reporting_ignores_errors_without_exceptions() -> None:
    processor = error_reporting("svc", "1.0")
    event = processor(None, "error", {"level": "error", "event": "x"})  # type: ignore[arg-type]
    assert "@type" not in event
