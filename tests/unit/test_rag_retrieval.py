from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import date
from typing import Any

import pytest
from sqlalchemy.dialects import postgresql

from app.core.config import Settings
from app.rag import retriever as retriever_module
from app.rag.crops import detect_query_crops
from app.rag.retriever import RetrievedChunk, Retriever, fuse, keyword_query_for
from app.repositories.rag import ChunkHit, build_keyword_statement
from app.services.llm.errors import LLMQuotaExhaustedError
from app.services.prompts import build_user_message

# ---------------------------------------------------------------- crops


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("టమాటాలో ఎర్లీ బ్లైట్ లక్షణాలు ఏమిటి?", ["tomato"]),
        ("ಟೊಮೆಟೊದಲ್ಲಿ ಅರ್ಲಿ ಬ್ಲೈಟ್", ["tomato"]),
        ("टमाटर में अगेती झुलसा", ["tomato"]),
        ("పత్తిలో తెల్లదోమ", ["cotton"]),
        ("ತೊಗರಿಯಲ್ಲಿ ಕಾಯಿ ಕೊರಕ", ["redgram"]),
        ("पत्ता गोभी में डायमंड बैक मोथ", ["cabbage"]),
        # Longest match wins: maize is not also sorghum, groundnut is not also chickpea.
        ("మొక్కజొన్నలో కత్తెర పురుగు", ["maize"]),
        ("వేరుశనగలో తిక్క ఆకుమచ్చ", ["groundnut"]),
        ("मूंगफली में टिक्का रोग", ["groundnut"]),
        # Mixed script and English names.
        ("Mancozeb ను వంకాయ మరియు tomato పై వాడవచ్చా?", ["brinjal", "tomato"]),
        ("How to control fall armyworm in maize?", ["maize"]),
        # Ambiguous common words are not treated as crops.
        ("ఆకులు పసుపు రంగులోకి మారుతున్నాయి", []),  # "yellow", not turmeric
        ("రాగి ఆక్సీక్లోరైడ్ మోతాదు", []),  # "copper oxychloride", not finger millet
        ("आम तौर पर कितना पानी दें?", []),  # "generally", not mango
        ("", []),
    ],
)
def test_detect_query_crops(question: str, expected: list[str]) -> None:
    assert detect_query_crops(question) == expected


# ---------------------------------------------------------------- keyword query


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("Is monocrotophos banned on vegetables?", "monocrotophos or banned or vegetables"),
        ("Mancozeb ను టమాటాపై ఎంత వాడాలి? mancozeb", "mancozeb"),
        ("టమాటాలో ఎర్లీ బ్లైట్", None),
        ("Acetamiprid 20% SP on cotton", "acetamiprid or cotton"),
        # Noise words (control, how, dose, spray...) are dropped; identifying words kept.
        ("How to control fall armyworm in maize?", 'fall or armyworm or maize or "army worm"'),
        ("What dose of spray to use?", None),
    ],
)
def test_keyword_query_for(question: str, expected: str | None) -> None:
    assert keyword_query_for(question) == expected


def test_keyword_query_expands_spelling_variants() -> None:
    query = keyword_query_for("How to control fall armyworm in maize?")
    assert query is not None
    assert '"army worm"' in query.split(" or ")
    reverse = keyword_query_for("white fly in cotton")
    assert reverse is not None and "whitefly" in reverse.split(" or ")


def test_keyword_statement_sql() -> None:
    sql = str(
        build_keyword_statement("monocrotophos or banned", top_k=10).compile(
            dialect=postgresql.dialect()
        )
    )
    assert "websearch_to_tsquery('english'::regconfig" in sql
    assert "rag_chunks.content_tsv @@ websearch_to_tsquery" in sql
    assert "ts_rank_cd(rag_chunks.content_tsv" in sql
    assert "ORDER BY" in sql and "DESC" in sql


# ---------------------------------------------------------------- fusion


def hit(
    chunk_id: str, similarity: float = 0.7, crops: list[str] | None = None, body: str | None = None
) -> ChunkHit:
    return ChunkHit(
        chunk_id=chunk_id,
        document_id="doc",
        title="Title",
        content=f"Title > Section\n{body or 'body of ' + chunk_id}",
        section="Section",
        page_start=1,
        page_end=1,
        crops=crops if crops is not None else [],
        doc_type="ipm_package",
        source_org=None,
        source_path="raw/x.pdf",
        as_of=date(2026, 3, 31),
        similarity=similarity,
    )


def test_fuse_combines_rankings() -> None:
    vector = [hit("a", 0.80), hit("b", 0.75), hit("c", 0.70)]
    keyword = [hit("c", 3.0), hit("d", 2.0)]
    results = fuse(vector, keyword, query_crops=[], top_k=5, min_similarity=0.5)
    ids = [r.hit.chunk_id for r in results]
    assert ids[0] == "c"  # found by both searches
    assert set(ids) == {"a", "b", "c", "d"}
    c = results[0]
    assert c.vector_rank == 2 and c.keyword_rank == 0
    assert c.vector_similarity == 0.70
    d = next(r for r in results if r.hit.chunk_id == "d")
    assert d.vector_similarity is None


def test_fuse_threshold_and_keyword_only_rank_limit() -> None:
    vector = [hit("weak", 0.40)]
    keyword = [hit(f"k{i}", 1.0) for i in range(8)]
    results = fuse(vector, keyword, query_crops=[], top_k=20, min_similarity=0.55)
    ids = {r.hit.chunk_id for r in results}
    assert "weak" not in ids  # below the similarity threshold, no keyword support
    assert ids == {"k0", "k1", "k2", "k3", "k4"}  # keyword-only hits beyond rank 5 dropped


def test_fuse_crop_boost_and_penalty() -> None:
    vector = [
        hit("potato", 0.80, crops=["potato"]),
        hit("generic", 0.79, crops=[]),
        hit("tomato", 0.78, crops=["tomato"]),
    ]
    results = fuse(vector, [], query_crops=["tomato"], top_k=3, min_similarity=0.5)
    assert [r.hit.chunk_id for r in results] == ["tomato", "generic", "potato"]


def test_fuse_other_crop_needs_higher_similarity() -> None:
    vector = [
        hit("brinjal-weak", 0.66, crops=["brinjal"]),
        hit("brinjal-strong", 0.74, crops=["brinjal"]),
        hit("rice", 0.60, crops=["rice"]),
        hit("generic", 0.60, crops=[]),
    ]
    keyword = [hit("ginger-kw", 5.0, crops=["ginger"])]
    ids = [
        r.hit.chunk_id
        for r in fuse(vector, keyword, query_crops=["rice"], top_k=5, min_similarity=0.55)
    ]
    assert "brinjal-weak" not in ids  # other crop, below 0.70
    assert "ginger-kw" not in ids  # keyword-only, other crop
    assert set(ids) == {"brinjal-strong", "rice", "generic"}
    assert ids[0] == "rice"


def test_fuse_deduplicates_identical_text() -> None:
    vector = [hit("a", 0.8, body="same text"), hit("b", 0.79, body="same  text"), hit("c", 0.7)]
    results = fuse(vector, [], query_crops=[], top_k=5, min_similarity=0.5)
    assert [r.hit.chunk_id for r in results] == ["a", "c"]


def test_fuse_top_k() -> None:
    vector = [hit(str(i), 0.9 - i / 100) for i in range(10)]
    assert len(fuse(vector, [], query_crops=[], top_k=3, min_similarity=0.5)) == 3


# ---------------------------------------------------------------- retriever


class FakeEmbedder:
    model = "fake-embed"

    def __init__(self, error: Exception | None = None, delay: float = 0.0) -> None:
        self.error = error
        self.delay = delay
        self.queries: list[str] = []

    async def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error:
            raise self.error
        return [1.0, 0.0]


class FakeDb:
    def __init__(self) -> None:
        self.reads = 0

    @asynccontextmanager
    async def session(self) -> AsyncIterator[object]:
        yield object()

    async def read(self, query: Any) -> Any:
        self.reads += 1
        return await query(object())


def make_retriever(
    monkeypatch: pytest.MonkeyPatch,
    *,
    embedder: FakeEmbedder | None = None,
    vector_hits: list[ChunkHit] | None = None,
    keyword_hits: list[ChunkHit] | None = None,
    **overrides: Any,
) -> tuple[Retriever, dict[str, Any]]:
    calls: dict[str, Any] = {}

    async def fake_search(session: object, vector: list[float], **kwargs: Any) -> list[ChunkHit]:
        calls.setdefault("vector_calls", []).append(kwargs)
        calls["vector"] = kwargs
        candidates = vector_hits or []
        if kwargs.get("crops"):
            return [h for h in candidates if set(h.crops) & set(kwargs["crops"])]
        return candidates

    async def fake_keyword(session: object, query: str, **kwargs: Any) -> list[ChunkHit]:
        calls["keyword"] = query
        return keyword_hits or []

    monkeypatch.setattr(retriever_module, "search_chunks", fake_search)
    monkeypatch.setattr(retriever_module, "keyword_search", fake_keyword)
    settings = Settings(_env_file=None, **overrides)  # type: ignore[arg-type]
    retriever = Retriever(db=FakeDb(), embedder=embedder or FakeEmbedder(), settings=settings)  # type: ignore[arg-type]
    return retriever, calls


async def test_retrieve_hybrid(monkeypatch: pytest.MonkeyPatch) -> None:
    retriever, calls = make_retriever(
        monkeypatch,
        vector_hits=[hit("t1", 0.8, crops=["tomato"]), hit("p1", 0.81, crops=["potato"])],
        keyword_hits=[hit("t1", 2.0, crops=["tomato"])],
    )
    result = await retriever.retrieve("Mancozeb dose for టమాటా early blight")
    assert result.error is None
    assert result.crops == ["tomato"]
    assert result.keyword_query is not None and "mancozeb" in result.keyword_query
    assert calls["vector"]["embedding_model"] == "fake-embed"
    assert calls["vector"]["top_k"] == 30
    # One unrestricted and one tomato-restricted vector search.
    assert sorted(str(c.get("crops")) for c in calls["vector_calls"]) == ["None", "['tomato']"]
    # The potato chunk (other crop, similarity 0.81 >= 0.70) is still allowed, ranked below.
    assert [c.hit.chunk_id for c in result.chunks] == ["t1", "p1"]
    assert result.total_ms >= 0


async def test_retrieve_single_vector_search_without_crops(monkeypatch: pytest.MonkeyPatch) -> None:
    retriever, calls = make_retriever(monkeypatch, vector_hits=[hit("a")])
    await retriever.retrieve("what is integrated pest management")
    assert len(calls["vector_calls"]) == 1


async def test_retrieve_vector_only_when_hybrid_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    retriever, calls = make_retriever(monkeypatch, vector_hits=[hit("a")], rag_hybrid=False)
    result = await retriever.retrieve("monocrotophos banned?")
    assert result.keyword_query is None
    assert "keyword" not in calls


async def test_indic_question_gets_keywords_from_crop_and_pest_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retriever, calls = make_retriever(monkeypatch, vector_hits=[hit("a")])
    result = await retriever.retrieve("పత్తిలో తెల్లదోమ నివారణ")
    assert result.crops == ["cotton"]
    assert result.terms == ["whitefly"]
    assert calls["keyword"] == 'whitefly or cotton or "white fly"'


async def test_retrieve_skips_keyword_without_identifying_words(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retriever, calls = make_retriever(monkeypatch, vector_hits=[hit("a")])
    await retriever.retrieve("ఇది ఎలా ఉంది?")  # "how is this?": no crop, pest or English word
    assert "keyword" not in calls


async def test_retrieve_degrades_on_embedding_quota(monkeypatch: pytest.MonkeyPatch) -> None:
    retriever, _ = make_retriever(
        monkeypatch, embedder=FakeEmbedder(error=LLMQuotaExhaustedError()), vector_hits=[hit("a")]
    )
    result = await retriever.retrieve("blight in tomato")
    assert result.chunks == []
    assert result.error == "llm_quota_exhausted"
    assert result.crops == ["tomato"]


async def test_retrieve_times_out(monkeypatch: pytest.MonkeyPatch) -> None:
    retriever, _ = make_retriever(
        monkeypatch, embedder=FakeEmbedder(delay=1.0), rag_timeout_seconds=0.05
    )
    result = await retriever.retrieve("blight")
    assert result.chunks == []
    assert result.error == "retrieval_timeout"


# ---------------------------------------------------------------- prompt


def test_user_message_without_sources_is_the_question() -> None:
    assert build_user_message("What is DAP?", [], max_chars=1000) == ("What is DAP?", 0)


def test_user_message_with_numbered_sources() -> None:
    sources = [
        RetrievedChunk(hit("a", body="Spray mancozeb 75% WP @ 600-800 g"), 0.1, 0.8, 0, None),
        RetrievedChunk(hit("b", body="Remove infected leaves"), 0.09, 0.7, 1, None),
    ]
    message, used = build_user_message("How to control early blight?", sources, max_chars=5000)
    assert used == 2
    assert message.startswith("SOURCES:\n\n[1] Title, p. 1, as of 2026-03-31 | Section\n")
    assert "Spray mancozeb 75% WP @ 600-800 g" in message
    assert "[2] Title" in message
    assert message.rstrip().endswith("How to control early blight?")
    assert "Title > Section" not in message  # raw header replaced by the label


def test_user_message_respects_budget() -> None:
    long_body = "word " * 400  # 2000 chars
    sources = [RetrievedChunk(hit(str(i), body=long_body), 0.1, 0.8, i, None) for i in range(5)]
    message, used = build_user_message("q?", sources, max_chars=3000)
    assert used == 2  # 2000 + truncated 1000, then the budget is exhausted
    assert " ..." in message
