"""Hybrid retrieval: vector search + full-text search, fused with Reciprocal Rank Fusion.

    question ─┬─ embed query ──────────── vector search (top N) ─┐
              ├─ Latin-script words ───── keyword search (top N) ─┼─ RRF fusion ─ top K
              └─ crops (te/kn/hi/en) ───────────── boost / penalty ┘

- The query embedding and the keyword search run concurrently.
- Keyword search matters for exact names (monocrotophos, Acetamiprid 20% SP), which embeddings
  blur. The corpus is English, so only Latin-script words of the question are used; farmers
  often type product names in English even inside Telugu/Kannada/Hindi text.
- Any failure (database, embedding quota, timeout) returns an empty result with an error code:
  the farmer still gets an answer, marked as not grounded.
"""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.core.errors import AppError
from app.core.logging import get_logger
from app.db import Database
from app.rag.crops import detect_query_crops
from app.rag.glossary import detect_terms
from app.rag.ingestion.metadata import CROP_ALIASES
from app.repositories.rag import DEFAULT_EF_SEARCH, ChunkHit, keyword_search, search_chunks

logger = get_logger(__name__)

RRF_K = 60
# RRF scores are ~1/61 (rank 1) to ~1/90 (rank 30); bonuses are on that scale.
CROP_MATCH_BONUS = 0.01
CROP_MISMATCH_PENALTY = 0.01
# A keyword-only hit (no vector similarity) is kept only if it ranks this high in keyword search.
KEYWORD_ONLY_MAX_RANK = 5
# A chunk about other crops than the question's needs a clearly higher similarity to be used
# (a brinjal page must not answer a rice question just because it is about "a fungal disease").
CROP_MISMATCH_MIN_SIMILARITY = 0.70
_MAX_KEYWORDS = 12
_LATIN_WORD = re.compile(r"[A-Za-z][A-Za-z0-9-]{2,}")


class QueryEmbedder(Protocol):
    @property
    def model(self) -> str: ...

    async def embed_query(self, text: str) -> list[float]: ...


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    hit: ChunkHit
    score: float
    vector_similarity: float | None
    vector_rank: int | None
    keyword_rank: int | None


@dataclass(slots=True)
class RetrievalResult:
    chunks: list[RetrievedChunk] = field(default_factory=list)
    crops: list[str] = field(default_factory=list)
    terms: list[str] = field(default_factory=list)  # pests/diseases detected from local names
    keyword_query: str | None = None
    embed_ms: float = 0.0
    vector_ms: float = 0.0
    keyword_ms: float = 0.0
    total_ms: float = 0.0
    error: str | None = None


# Pest names written both as one word and as two in Indian documents ("Fall Army Worm").
# Each spelling is expanded to the other so keyword search matches either.
SPELLING_VARIANTS: dict[str, str] = {
    "armyworm": "army worm",
    "bollworm": "boll worm",
    "whitefly": "white fly",
    "whiteflies": "white flies",
    "leafminer": "leaf miner",
    "leafhopper": "leaf hopper",
    "planthopper": "plant hopper",
    "stemborer": "stem borer",
    "fruitborer": "fruit borer",
    "podborer": "pod borer",
    "cutworm": "cut worm",
    "mealybug": "mealy bug",
    "diamondback": "diamond back",
}
_REVERSE_VARIANTS = {two: one for one, two in SPELLING_VARIANTS.items()}

# Words that appear on almost every page of an IPM package ("Cultural control", "Chemical
# control") or in almost every question. In an OR query they only add noise: the identifying
# words are pest, disease, crop and product names. (English stopwords are removed by Postgres.)
QUESTION_NOISE_WORDS = frozenset(
    {
        "control", "controlling", "manage", "management", "managing", "prevent", "prevention",
        "treat", "treatment", "cure", "remedy", "how", "what", "which", "when", "why", "can",
        "should", "use", "using", "used", "dose", "dosage", "spray", "spraying", "apply",
        "medicine", "chemical", "pesticide", "pesticides", "crop", "crops", "plant", "plants",
        "farmer", "field", "best", "good", "tell", "please", "give", "about", "need",
        "symptom", "symptoms", "sign", "signs", "problem", "disease", "diseases", "pest", "pests",
    }
)  # fmt: skip


def _phrase(term: str) -> str:
    return f'"{term}"' if " " in term else term  # quoted = phrase match


def keyword_query_for(text: str, extra_terms: Sequence[str] = ()) -> str | None:
    """Identifying words joined with " or " (websearch_to_tsquery: match any).

    Latin-script words of the question, plus `extra_terms`: English names of the crops and
    pests/diseases detected from local names, so Telugu/Kannada/Hindi questions also get exact
    keyword matching against the English documents.
    """
    words = [
        w
        for w in dict.fromkeys(w.lower() for w in _LATIN_WORD.findall(text))
        if w not in QUESTION_NOISE_WORDS
    ][:_MAX_KEYWORDS]
    terms = [*words, *(t.lower() for t in extra_terms)]
    if not terms:
        return None
    joined = " ".join(terms)
    expanded = [_phrase(t) for t in terms]
    for term in terms:
        if term in SPELLING_VARIANTS:
            expanded.append(_phrase(SPELLING_VARIANTS[term]))
    for two_words, one_word in _REVERSE_VARIANTS.items():
        if two_words in joined:
            expanded.append(one_word)
    return " or ".join(dict.fromkeys(expanded))


def crop_keywords(crops: Sequence[str]) -> list[str]:
    """English spellings used in the documents for each canonical crop (first three aliases)."""
    return [alias for crop in crops for alias in CROP_ALIASES.get(crop, (crop,))[:3]]


def _body_key(hit: ChunkHit) -> str:
    body = hit.content.split("\n", 1)[-1]
    return re.sub(r"\s+", " ", body).strip().lower()[:300]


def fuse(
    vector_hits: Sequence[ChunkHit],
    keyword_hits: Sequence[ChunkHit],
    *,
    query_crops: Sequence[str],
    top_k: int,
    min_similarity: float,
) -> list[RetrievedChunk]:
    hits: dict[str, ChunkHit] = {}
    vector_rank: dict[str, int] = {}
    similarity: dict[str, float] = {}
    keyword_rank: dict[str, int] = {}
    for rank, hit in enumerate(vector_hits):
        hits.setdefault(hit.chunk_id, hit)
        vector_rank[hit.chunk_id] = rank
        similarity[hit.chunk_id] = hit.similarity
    for rank, hit in enumerate(keyword_hits):
        hits.setdefault(hit.chunk_id, hit)
        keyword_rank[hit.chunk_id] = rank

    wanted = set(query_crops)
    scored: list[RetrievedChunk] = []
    for chunk_id, hit in hits.items():
        sim = similarity.get(chunk_id)
        k_rank = keyword_rank.get(chunk_id)
        other_crop = bool(wanted and hit.crops and not wanted & set(hit.crops))
        threshold = (
            max(min_similarity, CROP_MISMATCH_MIN_SIMILARITY) if other_crop else min_similarity
        )
        relevant_by_vector = sim is not None and sim >= threshold
        strong_keyword = k_rank is not None and k_rank < KEYWORD_ONLY_MAX_RANK and not other_crop
        if not (relevant_by_vector or strong_keyword):
            continue
        score = 0.0
        if chunk_id in vector_rank:
            score += 1.0 / (RRF_K + vector_rank[chunk_id] + 1)
        if k_rank is not None:
            score += 1.0 / (RRF_K + k_rank + 1)
        if wanted and hit.crops:
            score += CROP_MATCH_BONUS if wanted & set(hit.crops) else -CROP_MISMATCH_PENALTY
        scored.append(
            RetrievedChunk(
                hit=hit,
                score=score,
                vector_similarity=sim,
                vector_rank=vector_rank.get(chunk_id),
                keyword_rank=k_rank,
            )
        )

    scored.sort(key=lambda c: c.score, reverse=True)
    results: list[RetrievedChunk] = []
    seen: set[str] = set()
    for chunk in scored:
        key = _body_key(chunk.hit)
        if key in seen:  # identical text from duplicated pages / documents
            continue
        seen.add(key)
        results.append(chunk)
        if len(results) == top_k:
            break
    return results


def _error_code(exc: BaseException) -> str:
    if isinstance(exc, BaseExceptionGroup) and exc.exceptions:
        return _error_code(exc.exceptions[0])
    if isinstance(exc, AppError):
        return exc.code
    if isinstance(exc, TimeoutError):
        return "retrieval_timeout"
    return f"retrieval_failed:{type(exc).__name__}"


class Retriever:
    def __init__(self, *, db: Database, embedder: QueryEmbedder, settings: Settings) -> None:
        self._db = db
        self._embedder = embedder
        self._settings = settings

    async def retrieve(self, query: str, *, hybrid: bool | None = None) -> RetrievalResult:
        start = time.perf_counter()
        settings = self._settings
        use_hybrid = settings.rag_hybrid if hybrid is None else hybrid
        crops = detect_query_crops(query)
        terms = detect_terms(query)
        result = RetrievalResult(
            crops=crops,
            terms=terms,
            keyword_query=(
                keyword_query_for(query, [*terms, *crop_keywords(crops)]) if use_hybrid else None
            ),
        )
        try:
            async with asyncio.timeout(settings.rag_timeout_seconds):
                async with asyncio.TaskGroup() as group:
                    vector_task = group.create_task(self._embed(query, result))
                    keyword_task = group.create_task(self._keyword(result.keyword_query, result))
                vector_hits = await self._vector(vector_task.result(), result)
                keyword_hits = keyword_task.result()
        except Exception as exc:
            result.error = _error_code(exc)
            result.total_ms = round((time.perf_counter() - start) * 1000, 1)
            logger.warning("retrieval_failed", error=result.error, ms=result.total_ms, exc_info=exc)
            return result

        result.chunks = fuse(
            vector_hits,
            keyword_hits,
            query_crops=result.crops,
            top_k=settings.rag_top_k,
            min_similarity=settings.rag_min_similarity,
        )
        result.total_ms = round((time.perf_counter() - start) * 1000, 1)
        logger.info(
            "retrieval_done",
            crops=result.crops,
            hybrid=use_hybrid,
            vector_hits=len(vector_hits),
            keyword_hits=len(keyword_hits),
            returned=len(result.chunks),
            top_similarity=max((c.vector_similarity or 0.0 for c in result.chunks), default=None),
            embed_ms=result.embed_ms,
            vector_ms=result.vector_ms,
            keyword_ms=result.keyword_ms,
            total_ms=result.total_ms,
        )
        return result

    async def _embed(self, query: str, result: RetrievalResult) -> list[float]:
        start = time.perf_counter()
        vector = await self._embedder.embed_query(query)
        result.embed_ms = round((time.perf_counter() - start) * 1000, 1)
        return vector

    async def _vector(self, vector: list[float], result: RetrievalResult) -> list[ChunkHit]:
        """Unrestricted search, plus (in parallel) a search restricted to the question's crops.

        Cross-lingual similarity is weaker than same-language similarity, so for a Kannada
        question the right rice rows can fall outside the unrestricted top N; the crop-restricted
        search guarantees crop-relevant candidates.
        """
        start = time.perf_counter()
        async with asyncio.TaskGroup() as group:
            unrestricted = group.create_task(self._vector_search(vector, None))
            restricted = (
                group.create_task(self._vector_search(vector, result.crops))
                if result.crops
                else None
            )
        hits = {h.chunk_id: h for h in unrestricted.result()}
        if restricted is not None:
            for h in restricted.result():
                hits.setdefault(h.chunk_id, h)
        result.vector_ms = round((time.perf_counter() - start) * 1000, 1)
        return sorted(hits.values(), key=lambda h: h.similarity, reverse=True)

    async def _vector_search(self, vector: list[float], crops: list[str] | None) -> list[ChunkHit]:
        top_k = self._settings.rag_candidates

        async def query(session: AsyncSession) -> list[ChunkHit]:
            return await search_chunks(
                session, vector, embedding_model=self._embedder.model, top_k=top_k, crops=crops
            )

        if top_k > DEFAULT_EF_SEARCH:
            # Needs SET LOCAL hnsw.ef_search, which only works inside a transaction.
            async with self._db.session() as session:
                return await query(session)
        return await self._db.read(query)

    async def _keyword(self, query: str | None, result: RetrievalResult) -> list[ChunkHit]:
        if not query:
            return []
        start = time.perf_counter()
        top_k = self._settings.rag_candidates
        hits = await self._db.read(lambda session: keyword_search(session, query, top_k=top_k))
        result.keyword_ms = round((time.perf_counter() - start) * 1000, 1)
        return hits
