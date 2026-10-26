"""Embedding caches.

Key = sha256(model, dimensions, task, text): a vector is reused only for exactly the same input.

SqliteEmbeddingStore  persistent; bulk ingestion (resumable, never pays twice for a chunk)
MemoryEmbeddingStore  bounded LRU; runtime query embeddings (farmers repeat questions)
CachedEmbedder        wraps any Embedder; stores each batch as soon as it is embedded
"""

from __future__ import annotations

import hashlib
import sqlite3
from array import array
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from app.services.embeddings.base import Embedder, EmbeddingTask


def embedding_key(model: str, dimensions: int, task: EmbeddingTask, text: str) -> str:
    payload = f"{model}\x1f{dimensions}\x1f{task}\x1f{text}".encode()
    return hashlib.sha256(payload).hexdigest()


class EmbeddingStore(Protocol):
    def get_many(self, keys: list[str]) -> dict[str, list[float]]: ...

    def put_many(self, items: dict[str, list[float]]) -> None: ...


def _pack(vector: list[float]) -> bytes:
    return array("f", vector).tobytes()


def as_float32(vector: list[float]) -> list[float]:
    return array("f", vector).tolist()


def _unpack(blob: bytes) -> list[float]:
    values = array("f")
    values.frombytes(blob)
    return values.tolist()


class SqliteEmbeddingStore:
    """float32 vectors in a local SQLite file (1536 dims = 6 KB per vector)."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS embeddings (key TEXT PRIMARY KEY, vector BLOB NOT NULL)"
        )
        self._db.commit()

    def get_many(self, keys: list[str]) -> dict[str, list[float]]:
        found: dict[str, list[float]] = {}
        for start in range(0, len(keys), 500):  # SQLite parameter limit
            batch = keys[start : start + 500]
            placeholders = ",".join("?" * len(batch))
            rows = self._db.execute(
                f"SELECT key, vector FROM embeddings WHERE key IN ({placeholders})",  # noqa: S608 - placeholders only
                batch,
            )
            found.update({key: _unpack(blob) for key, blob in rows})
        return found

    def put_many(self, items: dict[str, list[float]]) -> None:
        self._db.executemany(
            "INSERT OR REPLACE INTO embeddings (key, vector) VALUES (?, ?)",
            [(key, _pack(vector)) for key, vector in items.items()],
        )
        self._db.commit()

    def count(self) -> int:
        row = self._db.execute("SELECT COUNT(*) FROM embeddings").fetchone()
        return int(row[0])

    def close(self) -> None:
        self._db.close()


class MemoryEmbeddingStore:
    def __init__(self, max_items: int) -> None:
        self._max = max_items
        self._items: OrderedDict[str, list[float]] = OrderedDict()

    def get_many(self, keys: list[str]) -> dict[str, list[float]]:
        found: dict[str, list[float]] = {}
        for key in keys:
            if key in self._items:
                self._items.move_to_end(key)
                found[key] = self._items[key]
        return found

    def put_many(self, items: dict[str, list[float]]) -> None:
        if self._max <= 0:
            return
        for key, vector in items.items():
            self._items[key] = vector
            self._items.move_to_end(key)
        while len(self._items) > self._max:
            self._items.popitem(last=False)

    def __len__(self) -> int:
        return len(self._items)


ProgressCallback = Callable[[int, int], None]


class CachedEmbedder:
    def __init__(self, inner: Embedder, store: EmbeddingStore) -> None:
        self._inner = inner
        self._store = store
        self.hits = 0
        self.misses = 0

    @property
    def model(self) -> str:
        return self._inner.model

    @property
    def dimensions(self) -> int:
        return self._inner.dimensions

    @property
    def batch_size(self) -> int:
        return self._inner.batch_size

    def key(self, text: str, task: EmbeddingTask) -> str:
        return embedding_key(self.model, self.dimensions, task, text)

    async def embed(
        self,
        texts: list[str],
        task: EmbeddingTask,
        *,
        on_progress: ProgressCallback | None = None,
    ) -> list[list[float]]:
        keys = [self.key(t, task) for t in texts]
        found = self._store.get_many(list(dict.fromkeys(keys)))
        missing: dict[str, str] = {}
        for key, text in zip(keys, texts, strict=True):
            if key not in found and key not in missing:
                missing[key] = text
        self.hits += len(texts) - sum(1 for k in keys if k in missing)
        self.misses += len(missing)

        pending = list(missing.items())
        done = len(texts) - len(pending)
        if on_progress:
            on_progress(done, len(texts))
        for start in range(0, len(pending), self._inner.batch_size):
            batch = pending[start : start + self._inner.batch_size]
            vectors = await self._inner.embed([text for _, text in batch], task)
            # Round to float32 (the stored precision) so a text always maps to the identical
            # vector whether it was just embedded or read back from the cache.
            new = {key: as_float32(vector) for (key, _), vector in zip(batch, vectors, strict=True)}
            self._store.put_many(new)  # persist immediately: a crash loses at most one batch
            found.update(new)
            done += len(batch)
            if on_progress:
                on_progress(done, len(texts))
        return [found[key] for key in keys]

    async def embed_query(self, text: str) -> list[float]:
        return (await self.embed([text], "query"))[0]

    async def warmup(self) -> None:
        warmup = getattr(self._inner, "warmup", None)
        if warmup is not None:
            await warmup()

    async def aclose(self) -> None:
        await self._inner.aclose()
