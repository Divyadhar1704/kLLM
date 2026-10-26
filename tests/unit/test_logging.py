from __future__ import annotations

import json
import logging
from typing import Any

import pytest
import structlog

from app.core.config import Settings
from app.core.logging import REDACTED, get_logger, setup_logging


def json_settings() -> Settings:
    return Settings(_env_file=None, log_json=True, log_level="INFO")  # type: ignore[call-arg]


def last_json_line(captured: str) -> dict[str, Any]:
    lines = [line for line in captured.splitlines() if line.strip()]
    assert lines, "no log output captured"
    parsed: dict[str, Any] = json.loads(lines[-1])
    return parsed


def test_json_output_has_cloud_logging_fields(capsys: pytest.CaptureFixture[str]) -> None:
    setup_logging(json_settings())
    get_logger("test").info("chat_completed", latency_ms=1234)
    record = last_json_line(capsys.readouterr().out)
    assert record["event"] == "chat_completed"
    assert record["severity"] == "INFO"
    assert record["logger"] == "test"
    assert record["latency_ms"] == 1234
    assert "timestamp" in record


def test_sensitive_values_redacted(capsys: pytest.CaptureFixture[str]) -> None:
    setup_logging(json_settings())
    get_logger("test").info(
        "outbound_call",
        api_key="sk-live-123",
        headers={"Authorization": "Bearer abc", "X-API-Key": "k", "Accept": "json"},
        llm_password="x",
        input_tokens=120,
        output_tokens=480,
    )
    record = last_json_line(capsys.readouterr().out)
    assert record["api_key"] == REDACTED
    assert record["llm_password"] == REDACTED
    assert record["headers"] == {"Authorization": REDACTED, "X-API-Key": REDACTED, "Accept": "json"}
    assert record["input_tokens"] == 120
    assert record["output_tokens"] == 480


def test_indic_text_not_escaped(capsys: pytest.CaptureFixture[str]) -> None:
    setup_logging(json_settings())
    question = "టమాటాలో ఎర్లీ బ్లైట్ లక్షణాలు ఏమిటి?"
    get_logger("test").info("question_received", text=question)
    out = capsys.readouterr().out
    assert question in out
    assert last_json_line(out)["text"] == question


def test_context_vars_attached(capsys: pytest.CaptureFixture[str]) -> None:
    setup_logging(json_settings())
    structlog.contextvars.bind_contextvars(request_id="req-42")
    get_logger("test").info("with_context")
    assert last_json_line(capsys.readouterr().out)["request_id"] == "req-42"


def test_stdlib_loggers_use_same_pipeline(capsys: pytest.CaptureFixture[str]) -> None:
    setup_logging(json_settings())
    logging.getLogger("uvicorn.error").warning("server_warning")
    record = last_json_line(capsys.readouterr().out)
    assert record["event"] == "server_warning"
    assert record["severity"] == "WARNING"


def test_exceptions_rendered_structured(capsys: pytest.CaptureFixture[str]) -> None:
    setup_logging(json_settings())
    try:
        raise ValueError("boom")
    except ValueError:
        get_logger("test").exception("failed")
    record = last_json_line(capsys.readouterr().out)
    assert record["severity"] == "ERROR"
    assert "exception" in record
    assert "boom" in json.dumps(record["exception"])


def test_level_filtering(capsys: pytest.CaptureFixture[str]) -> None:
    setup_logging(Settings(_env_file=None, log_json=True, log_level="WARNING"))  # type: ignore[call-arg]
    get_logger("test").info("hidden")
    get_logger("test").warning("shown")
    out = capsys.readouterr().out
    assert "hidden" not in out
    assert "shown" in out


def test_console_mode(capsys: pytest.CaptureFixture[str]) -> None:
    setup_logging(Settings(_env_file=None, log_json=False))  # type: ignore[call-arg]
    get_logger("test").info("console_line", crop="tomato")
    out = capsys.readouterr().out
    assert "console_line" in out
    assert "tomato" in out
