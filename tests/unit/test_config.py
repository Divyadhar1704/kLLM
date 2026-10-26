from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from app.core.config import SUPPORTED_LANGUAGES, Settings, get_settings

SECURE_DEPLOYED = {
    "postgres_password": "a-strong-generated-password",
    "gcp_project_id": "krishimitra-prod",
    "gcs_bucket": "krishimitra-media-prod",
    "log_json": True,
    "cors_allowed_origins": "https://admin.example.com",
}


def make(**overrides: object) -> Settings:
    return Settings(_env_file=None, **overrides)  # type: ignore[arg-type]


def test_defaults() -> None:
    s = make()
    assert s.app_env == "test"
    assert s.gcp_region == "asia-south1"
    assert s.supported_languages == SUPPORTED_LANGUAGES == ("en", "te", "kn", "hi")
    assert s.embedding_dimensions == 1536


def test_env_vars_override_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("RAG_TOP_K", "8")
    s = make()
    assert s.log_level == "DEBUG"
    assert s.rag_top_k == 8


def test_numeric_env_vars_coerced(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EMBEDDING_DIMENSIONS", "768")
    assert make().embedding_dimensions == 768
    monkeypatch.setenv("EMBEDDING_DIMENSIONS", "1000")
    with pytest.raises(ValidationError, match="embedding_dimensions"):
        make()


def test_env_example_is_valid() -> None:
    s = Settings(_env_file=".env.example")  # type: ignore[call-arg]
    assert s.embedding_dimensions == 1536
    assert s.llm_fallback_provider is None
    assert s.cors_allowed_origins == ["http://localhost:8010", "http://127.0.0.1:8010"]


def test_empty_env_var_falls_back_to_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_FALLBACK_PROVIDER", "")
    assert make().llm_fallback_provider is None


def test_cors_comma_separated_and_trailing_slash() -> None:
    s = make(cors_allowed_origins="http://localhost:8000/, https://a.example.com ,")
    assert s.cors_allowed_origins == ["http://localhost:8000", "https://a.example.com"]


def test_cors_rejects_invalid_origin() -> None:
    with pytest.raises(ValidationError, match="CORS origin"):
        make(cors_allowed_origins="localhost:8000")


def test_invalid_values_rejected() -> None:
    with pytest.raises(ValidationError):
        make(app_env="prod")
    with pytest.raises(ValidationError):
        make(rag_top_k=0)
    with pytest.raises(ValidationError):
        make(api_v1_prefix="api/v1/")


def test_fallback_provider_requires_model() -> None:
    with pytest.raises(ValidationError, match="LLM_FALLBACK_MODEL"):
        make(llm_fallback_provider="anthropic")
    s = make(llm_fallback_provider="anthropic", llm_fallback_model="claude-haiku-4-5")
    assert s.llm_fallback_model == "claude-haiku-4-5"


def test_production_rejects_insecure_defaults() -> None:
    with pytest.raises(ValidationError) as exc_info:
        make(app_env="production", log_json=False, enable_dev_ui=True, debug=True)
    message = str(exc_info.value)
    for expected in (
        "POSTGRES_PASSWORD",
        "GCP_PROJECT_ID",
        "GCS_BUCKET",
        "LOG_JSON",
        "DEBUG",
        "ENABLE_DEV_UI",
    ):
        assert expected in message


def test_production_rejects_wildcard_cors() -> None:
    with pytest.raises(ValidationError, match="must not contain"):
        make(app_env="production", **{**SECURE_DEPLOYED, "cors_allowed_origins": "*"})


def test_production_accepts_secure_config() -> None:
    s = make(app_env="production", **SECURE_DEPLOYED)
    assert s.is_production and s.is_deployed


def test_staging_allows_dev_ui() -> None:
    s = make(app_env="staging", enable_dev_ui=True, **SECURE_DEPLOYED)
    assert s.enable_dev_ui


def test_secrets_never_exposed() -> None:
    s = make(
        postgres_password="pg-secret-value",
        google_api_key="google-secret-value",
        redis_url="redis://:redis-secret-value@host:6379/0",
    )
    rendered = repr(s) + str(s) + json.dumps(s.public_summary())
    for secret in ("pg-secret-value", "google-secret-value", "redis-secret-value"):
        assert secret not in rendered


def test_database_url_escapes_password() -> None:
    s = make(postgres_password="p@ss:w/rd", postgres_host="db", postgres_port=5433)
    assert s.sqlalchemy_database_url == (
        "postgresql+asyncpg://krishimitra:p%40ss%3Aw%2Frd@db:5433/krishimitra"
    )


def test_get_settings_is_cached() -> None:
    assert get_settings() is get_settings()
