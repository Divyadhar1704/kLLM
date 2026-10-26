"""Real Postgres + pgvector (Supabase or local). Skipped unless KRISHIMITRA_DB_TESTS=1.

Requires `alembic upgrade head` on the target database. Test rows use ids prefixed `zztest`
and are removed afterwards; the sync test runs inside a transaction that is rolled back,
so the real knowledge base is never modified.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator

import pytest
from sqlalchemy import delete

from app.core.config import Settings
from app.db import Database
from app.models import EMBEDDING_DIMENSIONS, RagDocument
from app.repositories.rag import search_chunks, sync_corpus

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("KRISHIMITRA_DB_TESTS") != "1",
        reason="set KRISHIMITRA_DB_TESTS=1 to run against the configured database",
    ),
]


def unit(index: int) -> list[float]:
    vector = [0.0] * EMBEDDING_DIMENSIONS
    vector[index] = 1.0
    return vector


@pytest.fixture
async def database() -> AsyncIterator[Database]:
    db = Database.from_settings(Settings(_env_file=".env"))  # type: ignore[call-arg]
    yield db
    async with db.session() as session:
        await session.execute(delete(RagDocument).where(RagDocument.id.like("zztest%")))
    await db.aclose()


async def test_connectivity(database: Database) -> None:
    await database.check()


async def test_upsert_search_filter_and_stale_cleanup(database: Database) -> None:
    documents = [
        {
            "id": "zztestdoc1",
            "sha256": "z" * 64,
            "title": "ZZ Test Tomato IPM",
            "doc_type": "ipm_package",
            "source_org": None,
            "source_path": "raw/test.pdf",
            "as_of": None,
            "crops": ["tomato"],
            "language": "en",
            "pages": 1,
            "chunk_count": 2,
        }
    ]

    def chunk_row(chunk_id: str, axis: int, crops: list[str]) -> dict[str, object]:
        return {
            "id": chunk_id,
            "document_id": "zztestdoc1",
            "chunk_index": axis,
            "content": f"test chunk {chunk_id}",
            "section": None,
            "page_start": 1,
            "page_end": 1,
            "crops": crops,
            "doc_type": "ipm_package",
            "language": "en",
            "as_of": None,
            "token_estimate": 3,
            "embedding_model": "zztest-model",
            "embedding": unit(axis),
        }

    # Never committed: closing the session rolls everything back, even if an assert fails.
    # (sync_corpus with only the test document would otherwise delete the real corpus.)
    async with database.sessionmaker() as session:
        await sync_corpus(
            session,
            documents,
            [chunk_row("zztestc0", 0, ["tomato"]), chunk_row("zztestc1", 1, ["potato"])],
            restrictions=None,
        )
        hits = await search_chunks(session, unit(0), embedding_model="zztest-model", top_k=2)
        assert [h.chunk_id for h in hits] == ["zztestc0", "zztestc1"]
        assert hits[0].similarity == pytest.approx(1.0, abs=1e-3)
        assert hits[0].title == "ZZ Test Tomato IPM"

        filtered = await search_chunks(
            session, unit(0), embedding_model="zztest-model", top_k=2, crops=["potato"]
        )
        assert [h.chunk_id for h in filtered] == ["zztestc1"]

        thresholded = await search_chunks(
            session, unit(0), embedding_model="zztest-model", top_k=2, min_similarity=0.5
        )
        assert [h.chunk_id for h in thresholded] == ["zztestc0"]

        # Re-sync without one chunk: the stale chunk is deleted.
        stats = await sync_corpus(
            session, documents, [chunk_row("zztestc0", 0, ["tomato"])], restrictions=None
        )
        assert stats.chunks_deleted == 1
        remaining = await search_chunks(session, unit(1), embedding_model="zztest-model", top_k=5)
        assert [h.chunk_id for h in remaining] == ["zztestc0"]
        await session.rollback()
