"""Application configuration loaded from environment variables and an optional .env file.

All settings are validated at startup. Deployed environments (staging/production) refuse
to start with insecure defaults.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Literal
from urllib.parse import quote_plus

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

from app import __version__

Environment = Literal["local", "test", "staging", "production"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
LLMProvider = Literal["gemini", "anthropic", "openai"]

SUPPORTED_LANGUAGES: tuple[str, ...] = ("en", "te", "kn", "hi")

_INSECURE_PASSWORDS = frozenset({"", "change-me-local-only", "postgres", "password", "admin"})

_ALLOWED_EMBEDDING_DIMENSIONS: tuple[int, ...] = (768, 1536, 3072)

_MB = 1024 * 1024


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        env_ignore_empty=True,
    )

    # --- Application ---
    app_name: str = "KrishiMitra API"
    app_env: Environment = "local"
    app_version: str = __version__
    debug: bool = False
    log_level: LogLevel = "INFO"
    log_json: bool = True

    # --- HTTP ---
    api_v1_prefix: str = "/api/v1"
    cors_allowed_origins: Annotated[list[str], NoDecode] = Field(default_factory=list)
    enable_docs: bool = True
    enable_dev_ui: bool = False
    request_timeout_seconds: float = Field(default=60.0, gt=0, le=300)

    # --- Health checks ---
    health_check_timeout_seconds: float = Field(default=2.0, gt=0, le=10)
    health_cache_seconds: float = Field(default=5.0, ge=0, le=60)

    # --- PostgreSQL ---
    postgres_host: str = "localhost"
    postgres_port: int = Field(default=5432, ge=1, le=65535)
    postgres_user: str = "krishimitra"
    postgres_password: SecretStr = SecretStr("change-me-local-only")
    postgres_db: str = "krishimitra"
    postgres_ssl: Literal["disable", "prefer", "require", "verify-full"] = "prefer"
    # Set 0 behind a transaction-mode pooler (PgBouncer / Supabase port 6543).
    postgres_statement_cache_size: int = Field(default=100, ge=0)
    postgres_schema: str = Field(default="krishimitra", pattern=r"^[a-z_][a-z0-9_]{0,62}$")
    database_enabled: bool = True
    # Supabase free session pooler allows ~15 connections per project: stay well below.
    db_pool_size: int = Field(default=5, ge=1, le=100)
    db_max_overflow: int = Field(default=5, ge=0, le=100)
    db_pool_timeout_seconds: float = Field(default=10.0, gt=0, le=60)
    # A pre-ping costs one extra round trip on every checkout (~120 ms to Mumbai). Off by
    # default: connections are recycled before poolers drop them, and read queries retry once
    # on a stale connection instead.
    db_pool_pre_ping: bool = False
    db_pool_recycle_seconds: int = Field(default=240, ge=30, le=3600)
    db_warmup_connections: int = Field(default=3, ge=0, le=20)
    db_command_timeout_seconds: float = Field(default=30.0, gt=0, le=300)

    # --- Redis ---
    redis_url: SecretStr = SecretStr("redis://localhost:6379/0")

    # --- Google Cloud ---
    gcp_project_id: str | None = None
    gcp_region: str = "asia-south1"
    gcs_bucket: str | None = None

    # --- Media storage (cleaned photos / video frames of chat questions) ---
    # none: nothing stored. local: development folder (refused when deployed). gcs: GCS_BUCKET.
    media_storage: Literal["none", "local", "gcs"] = "none"
    media_local_dir: Path = Path(".media")
    media_retention_days: int = Field(default=180, ge=1, le=3650)
    media_url_ttl_seconds: int = Field(default=900, ge=60, le=86400)  # signed download links

    # --- LLM ---
    llm_provider: LLMProvider = "gemini"
    llm_model: str = "gemini-3.5-flash-lite"
    llm_fallback_provider: LLMProvider | None = None
    llm_fallback_model: str | None = None
    llm_timeout_seconds: float = Field(default=30.0, gt=0, le=120)
    llm_max_output_tokens: int = Field(default=1024, ge=64, le=8192)
    llm_temperature: float = Field(default=0.2, ge=0.0, le=1.0)
    llm_thinking_level: Literal["minimal", "low", "medium", "high"] | None = None
    llm_max_retries: int = Field(default=2, ge=0, le=5)
    # USD per 1M tokens, for cost tracking. 0 on the free tier.
    llm_input_price_per_million: float = Field(default=0.0, ge=0)
    llm_output_price_per_million: float = Field(default=0.0, ge=0)
    google_genai_use_vertexai: bool = True
    google_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None

    # --- Embeddings ---
    embedding_model: str = "gemini-embedding-2"
    embedding_dimensions: int = 1536
    embedding_batch_size: int = Field(default=50, ge=1, le=100)
    # Each text counts as one request on the Gemini free tier (limit 100/min). 0 = no throttle.
    embedding_requests_per_minute: int = Field(default=90, ge=0)
    embedding_timeout_seconds: float = Field(default=60.0, gt=0, le=300)
    embedding_max_retries: int = Field(default=6, ge=0, le=20)
    embedding_query_cache_size: int = Field(default=2048, ge=0)

    # --- RAG ---
    rag_enabled: bool = True
    rag_top_k: int = Field(default=5, ge=1, le=20)
    rag_candidates: int = Field(default=30, ge=5, le=100)  # per search before fusion
    rag_min_similarity: float = Field(default=0.55, ge=0.0, le=1.0)
    rag_hybrid: bool = True  # vector + full-text (exact product / pest names)
    rag_max_context_chars: int = Field(default=6000, ge=500, le=30000)
    rag_timeout_seconds: float = Field(default=8.0, gt=0, le=60)

    # --- Answer cache (identical questions answered instantly) ---
    answer_cache_size: int = Field(default=1000, ge=0)  # 0 disables
    answer_cache_ttl_seconds: int = Field(default=6 * 3600, ge=0)

    # --- Startup ---
    warmup_on_startup: bool = True  # open DB connections and AI clients before traffic

    # --- Input limits ---
    max_message_chars: int = Field(default=2000, ge=1, le=10000)
    max_image_bytes: int = Field(default=10 * _MB, ge=1)
    max_video_bytes: int = Field(default=50 * _MB, ge=1)
    max_video_seconds: int = Field(default=60, ge=1, le=600)

    # --- Image diagnosis ---
    vision_enabled: bool = True
    vision_model: str = "gemini-3.5-flash-lite"
    vision_fallback_model: str | None = "gemini-3.5-flash"
    vision_timeout_seconds: float = Field(default=30.0, gt=0, le=120)
    image_model_max_side: int = Field(default=1024, ge=256, le=3072)
    # Diagnoses below this (after calibration rules) are reported as uncertain. 0.8 measured at
    # 77% precision on field photos with the crop confirmed (0.6 was 64%). See scripts/eval_vision.
    image_min_confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    # Skip the AI call for photos that are too dark / blurry / small; ask for a retake instead.
    image_reject_poor_quality: bool = True

    # --- Video diagnosis (uses the image model on the best frames; audio is never processed) ---
    video_model_frames: int = Field(default=4, ge=1, le=8)  # frames sent in the one model call
    video_candidate_frames: int = Field(default=16, ge=4, le=60)  # frames quality-checked
    video_max_concurrent: int = Field(default=2, ge=1, le=16)  # decodes at once per instance

    # --- Authentication ---
    # Required in staging/production (validated). Off locally so /dev and the evals work.
    auth_required: bool = False
    # User tokens (Authorization: Bearer) signed by the Krishindra backend: either a shared
    # HS256 secret or a JWKS URL with its public keys (RS256/ES256). Claim `sub` = user id.
    auth_jwt_secret: SecretStr | None = None
    auth_jwt_jwks_url: str | None = None
    auth_jwt_issuer: str | None = None
    auth_jwt_audience: str = "krishimitra-api"
    auth_jwt_max_lifetime_seconds: int = Field(default=24 * 3600, ge=60)  # longer tokens refused
    auth_jwt_client_id: str = Field(default="krishindra-app", pattern=r"^[a-z0-9-]{1,40}$")
    api_key_cache_seconds: int = Field(default=60, ge=0, le=3600)  # revocation delay

    # --- Rate limiting (fixed windows; memory = per instance, redis = shared) ---
    rate_limit_backend: Literal["memory", "redis"] = "memory"
    rate_limit_per_minute: int = Field(default=30, ge=1)  # AI requests per user
    rate_limit_per_day: int = Field(default=500, ge=1)  # AI requests per user
    rate_limit_client_per_minute: int = Field(default=600, ge=1)  # AI requests per client app
    rate_limit_feedback_per_minute: int = Field(default=60, ge=1)  # per user

    # ------------------------------------------------------------------ validators

    @field_validator("cors_allowed_origins", mode="before")
    @classmethod
    def _split_origins(cls, value: Any) -> Any:
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @field_validator("cors_allowed_origins")
    @classmethod
    def _validate_origins(cls, origins: list[str]) -> list[str]:
        cleaned: list[str] = []
        for origin in origins:
            if origin == "*":
                cleaned.append(origin)
                continue
            if not origin.startswith(("http://", "https://")):
                raise ValueError(f"CORS origin must start with http:// or https://: {origin!r}")
            cleaned.append(origin.rstrip("/"))
        return cleaned

    @field_validator("embedding_dimensions")
    @classmethod
    def _validate_embedding_dimensions(cls, dims: int) -> int:
        if dims not in _ALLOWED_EMBEDDING_DIMENSIONS:
            raise ValueError(f"embedding_dimensions must be one of {_ALLOWED_EMBEDDING_DIMENSIONS}")
        return dims

    @field_validator("api_v1_prefix")
    @classmethod
    def _validate_prefix(cls, prefix: str) -> str:
        if not prefix.startswith("/") or prefix.endswith("/"):
            raise ValueError("api_v1_prefix must start with '/' and not end with '/'")
        return prefix

    @model_validator(mode="after")
    def _validate_consistency(self) -> Settings:
        if self.llm_fallback_provider is not None and not self.llm_fallback_model:
            raise ValueError("LLM_FALLBACK_MODEL is required when LLM_FALLBACK_PROVIDER is set")
        if self.media_storage == "gcs" and not self.gcs_bucket:
            raise ValueError("GCS_BUCKET is required when MEDIA_STORAGE=gcs")
        if self.auth_jwt_secret is not None and self.auth_jwt_jwks_url:
            raise ValueError("Set AUTH_JWT_SECRET or AUTH_JWT_JWKS_URL, not both")
        if self.auth_jwt_secret is not None and len(self.auth_jwt_secret.get_secret_value()) < 32:
            raise ValueError("AUTH_JWT_SECRET must be at least 32 characters")
        if self.auth_jwt_jwks_url and not self.auth_jwt_jwks_url.startswith("https://"):
            raise ValueError("AUTH_JWT_JWKS_URL must use https")

        if self.is_deployed:
            problems = self._deployment_problems()
            if problems:
                raise ValueError(
                    f"Insecure configuration for APP_ENV={self.app_env}: " + "; ".join(problems)
                )
        return self

    def _deployment_problems(self) -> list[str]:
        problems: list[str] = []
        if self.postgres_password.get_secret_value() in _INSECURE_PASSWORDS:
            problems.append("POSTGRES_PASSWORD is a default/insecure value")
        if not self.gcp_project_id:
            problems.append("GCP_PROJECT_ID is required")
        if not self.gcs_bucket:
            problems.append("GCS_BUCKET is required")
        if self.media_storage == "local":
            problems.append("MEDIA_STORAGE=local is for development only (use gcs or none)")
        if not self.auth_required:
            problems.append("AUTH_REQUIRED must be true")
        if not self.log_json:
            problems.append("LOG_JSON must be true")
        if self.app_env == "production":
            if self.debug:
                problems.append("DEBUG must be false")
            if self.enable_dev_ui:
                problems.append("ENABLE_DEV_UI must be false")
            if "*" in self.cors_allowed_origins:
                problems.append("CORS_ALLOWED_ORIGINS must not contain '*'")
        return problems

    # ------------------------------------------------------------------ derived values

    @property
    def is_deployed(self) -> bool:
        return self.app_env in ("staging", "production")

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def supported_languages(self) -> tuple[str, ...]:
        return SUPPORTED_LANGUAGES

    @property
    def sqlalchemy_database_url(self) -> str:
        """Async SQLAlchemy URL. Contains the password: never log this value."""
        user = quote_plus(self.postgres_user)
        password = quote_plus(self.postgres_password.get_secret_value())
        return (
            f"postgresql+asyncpg://{user}:{password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    def public_summary(self) -> dict[str, Any]:
        """Settings safe to log or display: secret values are masked."""
        return self.model_dump(mode="json")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
