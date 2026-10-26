from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from google.genai import errors as genai_errors

from app.core.config import Settings
from app.services.embeddings import (
    CachedEmbedder,
    MemoryEmbeddingStore,
    SqliteEmbeddingStore,
    build_embedder,
    embedding_key,
)
from app.services.embeddings.base import EmbeddingTask, l2_normalize
from app.services.embeddings.gemini import GeminiEmbedder
from app.services.embeddings.ratelimit import SlidingWindowLimiter
from app.services.llm.errors import (
    LLMConfigurationError,
    LLMQuotaExhaustedError,
    LLMRateLimitedError,
    LLMResponseError,
)
from app.services.llm.gemini import translate_error

DIMS = 4

# ---------------------------------------------------------------- helpers


def quota_error(*, delay: str | None = "23s", per_day: bool = False) -> genai_errors.APIError:
    quota_id = "EmbedContentRequestsPerDay" if per_day else "EmbedContentRequestsPerMinute"
    details: list[dict[str, Any]] = [
        {
            "@type": "type.googleapis.com/google.rpc.QuotaFailure",
            "violations": [{"quotaId": f"{quota_id}PerUserPerProjectPerModel-FreeTier"}],
        }
    ]
    if delay:
        details.append({"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": delay})
    payload = {
        "error": {
            "code": 429,
            "message": "quota",
            "status": "RESOURCE_EXHAUSTED",
            "details": details,
        }
    }
    return genai_errors.ClientError(429, payload)


def server_error(code: int = 503) -> genai_errors.APIError:
    return genai_errors.ServerError(code, {"error": {"code": code, "message": "x", "status": "X"}})


class FakeEmbedModels:
    """Returns a deterministic vector per text; scripted failures come first."""

    def __init__(self, failures: list[BaseException] | None = None) -> None:
        self.failures = list(failures or [])
        self.calls: list[dict[str, Any]] = []
        self.merge = False  # simulate the multimodal "list of strings -> one vector" behaviour
        self.wrong_dims = False

    @staticmethod
    def vector_for(text: str) -> list[float]:
        return [float(len(text)), 1.0, 2.0, float(sum(map(ord, text)) % 7 + 1)]

    async def embed_content(self, *, model: str, contents: list[Any], config: Any) -> Any:
        self.calls.append({"model": model, "contents": contents, "config": config})
        if self.failures:
            raise self.failures.pop(0)
        texts = [c.parts[0].text for c in contents]
        vectors = [self.vector_for(t) for t in texts]
        if self.merge:
            vectors = vectors[:1]
        if self.wrong_dims:
            vectors = [[*v, 0.0] for v in vectors]
        return SimpleNamespace(embeddings=[SimpleNamespace(values=v) for v in vectors])


class RecordingSleep:
    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.delays.append(seconds)


def gemini(
    models: FakeEmbedModels, *, batch_size: int = 3, max_retries: int = 3
) -> tuple[GeminiEmbedder, RecordingSleep]:
    sleep = RecordingSleep()
    embedder = GeminiEmbedder(
        client=SimpleNamespace(aio=SimpleNamespace(models=models)),
        model="gemini-embedding-test",
        dimensions=DIMS,
        batch_size=batch_size,
        timeout_seconds=5,
        max_retries=max_retries,
        sleep=sleep,
    )
    return embedder, sleep


def is_unit(vector: list[float]) -> bool:
    return math.isclose(sum(x * x for x in vector), 1.0, rel_tol=1e-6)


# ---------------------------------------------------------------- 429 parsing


def test_translate_rate_limit_with_retry_delay() -> None:
    error = translate_error(quota_error(delay="23s"))
    assert isinstance(error, LLMRateLimitedError)
    assert error.retry_after_seconds == 23.0
    assert error.headers == {"Retry-After": "23"}


def test_translate_daily_quota() -> None:
    error = translate_error(quota_error(per_day=True))
    assert isinstance(error, LLMQuotaExhaustedError)
    assert not error.retryable
    assert error.fallback_eligible


def test_translate_rate_limit_without_details() -> None:
    plain = genai_errors.ClientError(429, {"error": {"code": 429, "message": "x", "status": "X"}})
    error = translate_error(plain)
    assert isinstance(error, LLMRateLimitedError)
    assert error.retry_after_seconds is None
    assert error.headers == {"Retry-After": "30"}


# ---------------------------------------------------------------- gemini embedder


async def test_each_text_wrapped_and_batched() -> None:
    models = FakeEmbedModels()
    embedder, _ = gemini(models, batch_size=3)
    texts = [f"text {i}" for i in range(7)]
    vectors = await embedder.embed(texts, "document")
    assert len(vectors) == 7
    assert [len(c["contents"]) for c in models.calls] == [3, 3, 1]
    first = models.calls[0]
    assert first["contents"][0].parts[0].text == "text 0"  # one Content per text
    assert first["config"].task_type == "RETRIEVAL_DOCUMENT"
    assert first["config"].output_dimensionality == DIMS
    assert all(is_unit(v) for v in vectors)


async def test_query_task_type() -> None:
    models = FakeEmbedModels()
    embedder, _ = gemini(models)
    await embedder.embed(["question?"], "query")
    assert models.calls[0]["config"].task_type == "RETRIEVAL_QUERY"


async def test_merged_response_is_rejected() -> None:
    models = FakeEmbedModels()
    models.merge = True
    embedder, _ = gemini(models, max_retries=0)
    with pytest.raises(LLMResponseError, match="count mismatch"):
        await embedder.embed(["a text", "b text"], "document")


async def test_wrong_dimensions_rejected() -> None:
    models = FakeEmbedModels()
    models.wrong_dims = True
    embedder, _ = gemini(models, max_retries=0)
    with pytest.raises(LLMResponseError, match="dimensions"):
        await embedder.embed(["a text"], "document")


async def test_rate_limit_waits_provider_delay_then_succeeds() -> None:
    models = FakeEmbedModels(failures=[quota_error(delay="23s")])
    embedder, sleep = gemini(models)
    vectors = await embedder.embed(["a text"], "document")
    assert len(vectors) == 1
    assert sleep.delays == [24.0]


async def test_server_error_backoff_then_succeeds() -> None:
    models = FakeEmbedModels(failures=[server_error(503), server_error(500)])
    embedder, sleep = gemini(models)
    await embedder.embed(["a text"], "document")
    assert len(sleep.delays) == 2
    assert sleep.delays[1] > sleep.delays[0]


async def test_daily_quota_stops_immediately() -> None:
    models = FakeEmbedModels(failures=[quota_error(per_day=True)])
    embedder, sleep = gemini(models)
    with pytest.raises(LLMQuotaExhaustedError):
        await embedder.embed(["a text"], "document")
    assert sleep.delays == []


async def test_config_error_not_retried() -> None:
    models = FakeEmbedModels(
        failures=[genai_errors.ClientError(404, {"error": {"code": 404, "message": "no model"}})]
    )
    embedder, sleep = gemini(models)
    with pytest.raises(LLMConfigurationError):
        await embedder.embed(["a text"], "document")
    assert sleep.delays == []


async def test_gives_up_after_max_retries() -> None:
    models = FakeEmbedModels(failures=[server_error()] * 10)
    embedder, sleep = gemini(models, max_retries=2)
    with pytest.raises(Exception, match="overloaded"):
        await embedder.embed(["a text"], "document")
    assert len(sleep.delays) == 2


async def test_empty_text_rejected() -> None:
    embedder, _ = gemini(FakeEmbedModels())
    with pytest.raises(ValueError, match="empty"):
        await embedder.embed(["ok", "  "], "document")


def test_l2_normalize() -> None:
    assert l2_normalize([3.0, 4.0]) == [0.6, 0.8]
    with pytest.raises(ValueError):
        l2_normalize([0.0, 0.0])


# ---------------------------------------------------------------- rate limiter


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


async def test_limiter_waits_for_window() -> None:
    clock = FakeClock()

    async def sleep(seconds: float) -> None:
        clock.now += seconds

    limiter = SlidingWindowLimiter(100, clock=clock, sleep=sleep)
    assert await limiter.acquire(60) == 0
    clock.now = 10
    assert await limiter.acquire(40) == 0
    waited = await limiter.acquire(30)  # 100 used: must wait until the first 60 expire at t=60
    assert waited == pytest.approx(50.05, abs=0.01)
    assert clock.now == pytest.approx(60.05, abs=0.01)


async def test_limiter_rejects_oversized_request() -> None:
    with pytest.raises(ValueError):
        await SlidingWindowLimiter(10).acquire(11)


# ---------------------------------------------------------------- caches


def test_embedding_key_depends_on_all_inputs() -> None:
    base = embedding_key("m", 4, "document", "text")
    assert base != embedding_key("m2", 4, "document", "text")
    assert base != embedding_key("m", 8, "document", "text")
    assert base != embedding_key("m", 4, "query", "text")
    assert base != embedding_key("m", 4, "document", "text ")
    assert base == embedding_key("m", 4, "document", "text")


def test_sqlite_store_roundtrip(tmp_path: Path) -> None:
    store = SqliteEmbeddingStore(tmp_path / "e.sqlite")
    store.put_many({"a": [0.1, 0.2, 0.3], "b": [1.0, 0.0, -1.0]})
    store.close()
    reopened = SqliteEmbeddingStore(tmp_path / "e.sqlite")
    found = reopened.get_many(["a", "b", "missing"])
    assert set(found) == {"a", "b"}
    assert found["a"] == pytest.approx([0.1, 0.2, 0.3], rel=1e-6)  # float32 precision
    assert reopened.count() == 2


def test_sqlite_store_many_keys(tmp_path: Path) -> None:
    store = SqliteEmbeddingStore(tmp_path / "e.sqlite")
    store.put_many({f"k{i}": [float(i)] for i in range(1200)})
    assert len(store.get_many([f"k{i}" for i in range(1200)])) == 1200


def test_memory_store_lru_eviction() -> None:
    store = MemoryEmbeddingStore(2)
    store.put_many({"a": [1.0], "b": [2.0]})
    store.get_many(["a"])  # a is now most recent
    store.put_many({"c": [3.0]})
    assert set(store.get_many(["a", "b", "c"])) == {"a", "c"}
    assert len(store) == 2


async def test_cached_embedder_hits_misses_and_dedup(tmp_path: Path) -> None:
    models = FakeEmbedModels()
    inner, _ = gemini(models, batch_size=2)
    cached = CachedEmbedder(inner, SqliteEmbeddingStore(tmp_path / "e.sqlite"))
    first = await cached.embed(["one", "two", "one", "three"], "document")
    # Three unique texts embedded; the duplicate "one" reuses the same API result.
    assert cached.misses == 3 and cached.hits == 0
    assert sum(len(c["contents"]) for c in models.calls) == 3
    assert first[0] == first[2]
    calls_before = len(models.calls)
    again = await cached.embed(["three", "two", "one"], "document")
    assert len(models.calls) == calls_before  # everything from cache
    assert again == [first[3], first[1], first[0]]
    # The same text as a query is a different cache entry.
    await cached.embed(["one"], "query")
    assert len(models.calls) == calls_before + 1


async def test_cached_embedder_resumes_after_failure(tmp_path: Path) -> None:
    store = SqliteEmbeddingStore(tmp_path / "e.sqlite")
    models = FakeEmbedModels()
    inner, _ = gemini(models, batch_size=2, max_retries=0)
    texts = [f"chunk {i}" for i in range(6)]

    original = models.embed_content
    calls = 0

    async def fail_on_third(**kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise quota_error(per_day=True)
        return await original(**kwargs)

    models.embed_content = fail_on_third  # type: ignore[method-assign]
    with pytest.raises(LLMQuotaExhaustedError):
        await CachedEmbedder(inner, store).embed(texts, "document")
    assert store.count() == 4  # the two completed batches were saved

    models.embed_content = original  # type: ignore[method-assign]
    resumed = CachedEmbedder(inner, store)
    progress: list[tuple[int, int]] = []
    await resumed.embed(texts, "document", on_progress=lambda d, t: progress.append((d, t)))
    assert resumed.misses == 2 and resumed.hits == 4
    assert progress[0] == (4, 6) and progress[-1] == (6, 6)


async def test_embed_query_helper() -> None:
    inner, _ = gemini(FakeEmbedModels())
    cached = CachedEmbedder(inner, MemoryEmbeddingStore(10))
    vector = await cached.embed_query("what is blast?")
    assert len(vector) == DIMS and is_unit(vector)


def test_build_embedder_from_settings() -> None:
    settings = Settings(  # type: ignore[call-arg]
        _env_file=None,
        google_genai_use_vertexai=False,
        google_api_key="k",
        embedding_batch_size=100,
        embedding_requests_per_minute=40,
    )
    embedder = build_embedder(settings)
    assert embedder.model == "gemini-embedding-2"
    assert embedder.dimensions == 1536
    assert embedder.batch_size == 40  # a batch can never exceed the per-minute limit


@pytest.mark.parametrize(
    ("batch", "rpm", "expected"),
    [
        (50, 90, 45),  # 2 x 45 = 90/min instead of 1 x 50
        (50, 100, 50),
        (100, 40, 40),
        (50, 0, 50),  # no throttle
        (100, 250, 83),
    ],
)
def test_paced_batch_size(batch: int, rpm: int, expected: int) -> None:
    assert GeminiEmbedder.paced_batch_size(batch, rpm) == expected


def task_values() -> list[EmbeddingTask]:
    return ["document", "query"]
