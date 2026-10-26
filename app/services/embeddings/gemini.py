"""Gemini embedder.

gemini-embedding-2 is multimodal: a plain list of strings is treated as ONE multi-part input and
returns a single vector. Every text is therefore wrapped in its own Content, and the response
count is verified so a silent merge can never corrupt the index.
"""

from __future__ import annotations

import asyncio
import math
import random
import time
from collections.abc import Awaitable, Callable
from typing import Any

from google.genai import types

from app.core.config import Settings
from app.core.logging import get_logger
from app.services.embeddings.base import EmbeddingTask, l2_normalize
from app.services.embeddings.ratelimit import SlidingWindowLimiter
from app.services.google_client import create_genai_client
from app.services.llm.errors import LLMError, LLMRateLimitedError, LLMResponseError
from app.services.llm.gemini import translate_error

logger = get_logger(__name__)

_TASK_TYPES: dict[EmbeddingTask, str] = {
    "document": "RETRIEVAL_DOCUMENT",
    "query": "RETRIEVAL_QUERY",
}
_MAX_BACKOFF_SECONDS = 60.0


class GeminiEmbedder:
    def __init__(
        self,
        *,
        client: Any,
        model: str,
        dimensions: int,
        batch_size: int,
        timeout_seconds: float,
        max_retries: int,
        limiter: SlidingWindowLimiter | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._client = client
        self._model = model
        self._dimensions = dimensions
        self._batch_size = batch_size
        self._timeout = timeout_seconds
        self._max_retries = max_retries
        self._limiter = limiter
        self._sleep = sleep

    @staticmethod
    def paced_batch_size(batch_size: int, requests_per_minute: int) -> int:
        """Largest batch size <= batch_size that divides the per-minute budget evenly.

        With 90/min and batches of 50, only one batch fits per window (50/min). Batches of 45
        fit twice (90/min).
        """
        if requests_per_minute <= 0:
            return batch_size
        batches_per_minute = math.ceil(requests_per_minute / batch_size)
        return max(1, requests_per_minute // batches_per_minute)

    @classmethod
    def from_settings(cls, settings: Settings) -> GeminiEmbedder:
        rpm = settings.embedding_requests_per_minute
        return cls(
            client=create_genai_client(
                settings, timeout_seconds=settings.embedding_timeout_seconds
            ),
            model=settings.embedding_model,
            dimensions=settings.embedding_dimensions,
            batch_size=cls.paced_batch_size(settings.embedding_batch_size, rpm),
            timeout_seconds=settings.embedding_timeout_seconds,
            max_retries=settings.embedding_max_retries,
            limiter=SlidingWindowLimiter(rpm) if rpm else None,
        )

    @property
    def model(self) -> str:
        return self._model

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def batch_size(self) -> int:
        return self._batch_size

    async def embed(self, texts: list[str], task: EmbeddingTask) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            vectors.extend(await self._embed_batch(texts[start : start + self._batch_size], task))
        return vectors

    async def _embed_batch(self, texts: list[str], task: EmbeddingTask) -> list[list[float]]:
        if any(not t.strip() for t in texts):
            raise ValueError("cannot embed empty text")
        attempt = 0
        while True:
            if self._limiter is not None:
                await self._limiter.acquire(len(texts))
            start = time.perf_counter()
            try:
                return await self._call(texts, task)
            except Exception as exc:
                error = translate_error(exc)
                if not error.retryable or attempt >= self._max_retries:
                    logger.error(
                        "embedding_failed",
                        model=self._model,
                        error_code=error.code,
                        texts=len(texts),
                    )
                    raise error from exc
                delay = self._retry_delay(error, attempt)
                logger.warning(
                    "embedding_retry",
                    model=self._model,
                    error_code=error.code,
                    attempt=attempt + 1,
                    delay_seconds=round(delay, 1),
                    latency_ms=round((time.perf_counter() - start) * 1000, 1),
                )
                await self._sleep(delay)
                attempt += 1

    @staticmethod
    def _retry_delay(error: LLMError, attempt: int) -> float:
        if isinstance(error, LLMRateLimitedError) and error.retry_after_seconds is not None:
            return error.retry_after_seconds + 1.0  # the provider told us exactly how long
        backoff: float = min(_MAX_BACKOFF_SECONDS, 2.0 * 2.0**attempt)
        jitter: float = random.uniform(0, backoff / 4)  # noqa: S311 - jitter, not crypto
        return backoff + jitter

    async def _call(self, texts: list[str], task: EmbeddingTask) -> list[list[float]]:
        contents = [types.Content(parts=[types.Part.from_text(text=t)]) for t in texts]
        config = types.EmbedContentConfig(
            task_type=_TASK_TYPES[task], output_dimensionality=self._dimensions
        )
        async with asyncio.timeout(self._timeout):
            response = await self._client.aio.models.embed_content(
                model=self._model, contents=contents, config=config
            )
        embeddings = response.embeddings or []
        if len(embeddings) != len(texts):
            raise LLMResponseError(
                f"Embedding count mismatch: sent {len(texts)}, received {len(embeddings)}"
            )
        vectors: list[list[float]] = []
        for embedding in embeddings:
            values = list(embedding.values or [])
            if len(values) != self._dimensions:
                raise LLMResponseError(
                    f"Embedding has {len(values)} dimensions, expected {self._dimensions}"
                )
            # Truncated (Matryoshka) Gemini embeddings are not unit length: normalise.
            vectors.append(l2_normalize(values))
        return vectors

    async def warmup(self) -> None:
        """Open the HTTPS connection with a model metadata lookup (no embedding quota used)."""
        async with asyncio.timeout(self._timeout):
            await self._client.aio.models.get(model=self._model)

    async def aclose(self) -> None:
        aclose = getattr(self._client.aio, "aclose", None)
        if aclose is not None:
            await aclose()
