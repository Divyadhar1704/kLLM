"""Text embeddings: Gemini embedder, rate limiting, caching."""

from app.services.embeddings.base import Embedder, EmbeddingTask
from app.services.embeddings.cache import (
    CachedEmbedder,
    MemoryEmbeddingStore,
    SqliteEmbeddingStore,
    embedding_key,
)
from app.services.embeddings.factory import build_embedder

__all__ = [
    "CachedEmbedder",
    "Embedder",
    "EmbeddingTask",
    "MemoryEmbeddingStore",
    "SqliteEmbeddingStore",
    "build_embedder",
    "embedding_key",
]
