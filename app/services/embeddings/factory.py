"""Builds the configured embedder."""

from __future__ import annotations

from app.core.config import Settings
from app.services.embeddings.cache import CachedEmbedder, EmbeddingStore, MemoryEmbeddingStore
from app.services.embeddings.gemini import GeminiEmbedder


def build_embedder(settings: Settings, *, store: EmbeddingStore | None = None) -> CachedEmbedder:
    """Gemini embedder with a cache: in-memory LRU by default (runtime queries) or `store`."""
    return CachedEmbedder(
        GeminiEmbedder.from_settings(settings),
        store or MemoryEmbeddingStore(settings.embedding_query_cache_size),
    )
