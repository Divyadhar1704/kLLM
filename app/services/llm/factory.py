"""Builds the configured LLM stack from settings."""

from __future__ import annotations

from app.core.config import Settings
from app.services.llm.base import LLMProvider
from app.services.llm.gemini import GeminiProvider
from app.services.llm.resilience import FallbackLLM, RetryingLLM


def _build_single(settings: Settings, provider: str, model: str) -> LLMProvider:
    if provider == "gemini":
        return GeminiProvider.from_settings(settings, model=model)
    raise ValueError(f"LLM provider {provider!r} is not implemented yet")


def build_llm(settings: Settings) -> LLMProvider:
    primary: LLMProvider = RetryingLLM(
        _build_single(settings, settings.llm_provider, settings.llm_model),
        max_retries=settings.llm_max_retries,
    )
    if settings.llm_fallback_provider is None or settings.llm_fallback_model is None:
        return primary
    fallback = RetryingLLM(
        _build_single(settings, settings.llm_fallback_provider, settings.llm_fallback_model),
        max_retries=min(settings.llm_max_retries, 1),
    )
    return FallbackLLM(primary, fallback)
