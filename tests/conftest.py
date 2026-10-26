"""Shared test fixtures."""

from __future__ import annotations

import logging
from collections.abc import Iterator

import pytest
import structlog

from app.core.config import get_settings


@pytest.fixture(autouse=True)
def _isolate_settings(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Every test starts with a fresh settings cache, APP_ENV=test and no real database.

    Tests that exercise database wiring opt in with database_enabled=True (and fakes).
    """
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("DATABASE_ENABLED", "false")
    monkeypatch.setenv("WARMUP_ON_STARTUP", "false")  # warm-up would call real services
    monkeypatch.setenv("ANSWER_CACHE_SIZE", "0")  # tests opt in to caching explicitly
    monkeypatch.setenv("VISION_ENABLED", "false")  # tests inject a fake diagnoser
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
def _restore_logging() -> Iterator[None]:
    """Undo logging configuration done by a test (setup_logging / create_app)."""
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    yield
    root.handlers[:] = handlers
    root.setLevel(level)
    structlog.contextvars.clear_contextvars()
    structlog.reset_defaults()
