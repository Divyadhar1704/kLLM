"""Structured logging.

- JSON output (one object per line) with a `severity` field for Google Cloud Logging.
- Human-readable console output for local development.
- Standard-library loggers (uvicorn, httpx, google libraries) go through the same pipeline.
- Sensitive keys are redacted, including inside nested dicts.
- Context bound via structlog.contextvars (for example request_id) is attached to every log line.
"""

from __future__ import annotations

import contextlib
import io
import logging
import sys
import traceback
from collections.abc import Mapping
from typing import Any

import structlog
from structlog.types import EventDict, Processor, WrappedLogger

from app.core.config import Settings

REDACTED = "[REDACTED]"

_SENSITIVE_KEYS = frozenset(
    {
        "password",
        "passwd",
        "secret",
        "token",
        "access_token",
        "refresh_token",
        "id_token",
        "api_key",
        "apikey",
        "x_api_key",
        "authorization",
        "proxy_authorization",
        "cookie",
        "set_cookie",
        "credentials",
        "private_key",
        "client_secret",
        "signed_url",
    }
)
_SENSITIVE_SUFFIXES = ("_password", "_secret", "_api_key", "_access_token", "_auth_token")

_NOISY_LOGGERS = ("httpx", "httpcore", "urllib3", "google", "grpc", "asyncio")


def _is_sensitive(key: str) -> bool:
    normalized = key.lower().replace("-", "_")
    return normalized in _SENSITIVE_KEYS or normalized.endswith(_SENSITIVE_SUFFIXES)


def _redact(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            k: REDACTED if isinstance(k, str) and _is_sensitive(k) else _redact(v)
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [_redact(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_redact(v) for v in value)
    return value


def redact_sensitive(_: WrappedLogger, __: str, event_dict: EventDict) -> EventDict:
    for key in list(event_dict):
        if _is_sensitive(key):
            event_dict[key] = REDACTED
        else:
            event_dict[key] = _redact(event_dict[key])
    return event_dict


def add_severity(_: WrappedLogger, __: str, event_dict: EventDict) -> EventDict:
    level = event_dict.get("level")
    if isinstance(level, str):
        event_dict["severity"] = level.upper()
    return event_dict


_ERROR_EVENT_TYPE = (
    "type.googleapis.com/google.devtools.clouderrorreporting.v1beta1.ReportedErrorEvent"
)


def error_reporting(service: str, version: str) -> Processor:
    """Marks ERROR logs that carry an exception for Cloud Error Reporting (groups new errors and
    can alert on them): the event type, the service, and the stack trace as text."""

    def processor(_: WrappedLogger, __: str, event_dict: EventDict) -> EventDict:
        exc_info = event_dict.get("exc_info")
        if not exc_info or event_dict.get("level") not in ("error", "critical"):
            return event_dict
        if isinstance(exc_info, BaseException):
            exc_info = (type(exc_info), exc_info, exc_info.__traceback__)
        elif not isinstance(exc_info, tuple):
            exc_info = sys.exc_info()
        if exc_info[0] is None:
            return event_dict
        event_dict["@type"] = _ERROR_EVENT_TYPE
        event_dict["serviceContext"] = {"service": service, "version": version}
        event_dict["stack_trace"] = "".join(traceback.format_exception(*exc_info))
        return event_dict

    return processor


def _ensure_utf8_stdout() -> None:
    # Telugu/Kannada/Hindi text must not crash logging on Windows consoles or redirected output.
    stream = sys.stdout
    if isinstance(stream, io.TextIOWrapper) and (stream.encoding or "").lower() != "utf-8":
        with contextlib.suppress(ValueError, io.UnsupportedOperation):
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")


def setup_logging(settings: Settings) -> None:
    """Configure structlog and the stdlib root logger. Safe to call more than once."""
    _ensure_utf8_stdout()

    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        redact_sensitive,
    ]

    final_processors: list[Processor] = [structlog.stdlib.ProcessorFormatter.remove_processors_meta]
    if settings.log_json:
        shared_processors.append(add_severity)
        final_processors += [
            error_reporting("krishimitra-api", settings.app_version),  # before exc_info is used
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer(ensure_ascii=False),
        ]
    else:
        final_processors.append(structlog.dev.ConsoleRenderer(colors=sys.stdout.isatty()))

    structlog.configure(
        processors=[*shared_processors, structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=final_processors,
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root_level = logging.getLevelNamesMapping()[settings.log_level]
    root.setLevel(root_level)

    # Uvicorn installs its own handlers; route everything through the root handler instead.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        uvicorn_logger = logging.getLogger(name)
        uvicorn_logger.handlers.clear()
        uvicorn_logger.propagate = True
    # Request logging is done by our own middleware (Phase 4).
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(max(logging.WARNING, root_level))


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    return structlog.stdlib.get_logger(name)
