"""Embedding interface. Vectors are L2-normalised, so cosine similarity = dot product."""

from __future__ import annotations

import math
from typing import Literal, Protocol

EmbeddingTask = Literal["document", "query"]


class Embedder(Protocol):
    @property
    def model(self) -> str: ...

    @property
    def dimensions(self) -> int: ...

    @property
    def batch_size(self) -> int: ...

    async def embed(self, texts: list[str], task: EmbeddingTask) -> list[list[float]]: ...

    async def aclose(self) -> None: ...


def l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vector))
    if norm == 0:
        raise ValueError("cannot normalise a zero vector")
    return [x / norm for x in vector]
