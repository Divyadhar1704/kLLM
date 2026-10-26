"""Shared google-genai client construction (AI Studio API key or Vertex AI)."""

from __future__ import annotations

from google import genai
from google.genai import types

from app.core.config import Settings


def create_genai_client(settings: Settings, *, timeout_seconds: float) -> genai.Client:
    """Build a client. The SDK timeout is a backstop; callers enforce their own asyncio timeouts."""
    http_options = types.HttpOptions(timeout=int((timeout_seconds + 5) * 1000))
    if settings.google_genai_use_vertexai:
        if not settings.gcp_project_id:
            raise ValueError("GCP_PROJECT_ID is required when GOOGLE_GENAI_USE_VERTEXAI=true")
        return genai.Client(
            vertexai=True,
            project=settings.gcp_project_id,
            location=settings.gcp_region,
            http_options=http_options,
        )
    if settings.google_api_key is None:
        raise ValueError("GOOGLE_API_KEY is required when GOOGLE_GENAI_USE_VERTEXAI=false")
    return genai.Client(
        api_key=settings.google_api_key.get_secret_value(), http_options=http_options
    )
