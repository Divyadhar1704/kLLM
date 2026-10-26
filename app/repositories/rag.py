"""Knowledge-base persistence and vector search (Postgres + pgvector)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any

from sqlalchemy import Text, any_, bindparam, delete, func, not_, select, text
from sqlalchemy.dialects.postgresql import ARRAY, insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import PesticideRestrictionRow, RagChunk, RagDocument

_UPSERT_BATCH = 200
# pgvector's default hnsw.ef_search. Raising it costs an extra round trip (SET LOCAL), so it is
# only done when more candidates than this are requested.
DEFAULT_EF_SEARCH = 40
_FTS_CONFIG = "english"


@dataclass(frozen=True, slots=True)
class ChunkHit:
    chunk_id: str
    document_id: str
    title: str
    content: str
    section: str | None
    page_start: int | None
    page_end: int | None
    crops: list[str]
    doc_type: str
    source_org: str | None
    source_path: str
    as_of: date | None
    # Score from the search that produced the hit: cosine similarity (vector search)
    # or ts_rank_cd (keyword search). Not comparable across the two.
    similarity: float


@dataclass(frozen=True, slots=True)
class SyncStats:
    documents_upserted: int
    chunks_upserted: int
    chunks_deleted: int
    documents_deleted: int
    restrictions_loaded: int


def _ids_param(name: str, ids: Sequence[str]) -> Any:
    # One array parameter instead of thousands of individual binds.
    return bindparam(name, list(ids), type_=ARRAY(Text))


async def sync_corpus(
    session: AsyncSession,
    documents: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    restrictions: list[dict[str, Any]] | None = None,
) -> SyncStats:
    """Make the knowledge base equal to the given corpus (in the caller's transaction)."""
    if documents:
        stmt = insert(RagDocument)
        await session.execute(
            stmt.on_conflict_do_update(
                index_elements=[RagDocument.id],
                set_={
                    **{c: stmt.excluded[c] for c in documents[0] if c != "id"},
                    "updated_at": func.now(),
                },
            ),
            documents,
        )

    for start in range(0, len(chunks), _UPSERT_BATCH):
        batch = chunks[start : start + _UPSERT_BATCH]
        stmt = insert(RagChunk)
        await session.execute(
            stmt.on_conflict_do_update(
                index_elements=[RagChunk.id],
                set_={
                    **{c: stmt.excluded[c] for c in batch[0] if c != "id"},
                    "updated_at": func.now(),
                },
            ),
            batch,
        )

    document_ids = [d["id"] for d in documents]
    chunk_ids = [c["id"] for c in chunks]
    stale_chunks = await session.execute(
        delete(RagChunk).where(
            RagChunk.document_id == any_(_ids_param("doc_ids", document_ids)),
            not_(RagChunk.id == any_(_ids_param("chunk_ids", chunk_ids))),
        )
    )
    stale_documents = await session.execute(
        delete(RagDocument).where(
            not_(RagDocument.id == any_(_ids_param("keep_ids", document_ids)))
        )
    )

    restrictions_loaded = 0
    if restrictions is not None:
        await session.execute(delete(PesticideRestrictionRow))
        if restrictions:
            await session.execute(insert(PesticideRestrictionRow), restrictions)
        restrictions_loaded = len(restrictions)

    return SyncStats(
        documents_upserted=len(documents),
        chunks_upserted=len(chunks),
        chunks_deleted=stale_chunks.rowcount or 0,  # type: ignore[attr-defined]
        documents_deleted=stale_documents.rowcount or 0,  # type: ignore[attr-defined]
        restrictions_loaded=restrictions_loaded,
    )


def build_search_statement(
    query_vector: list[float],
    *,
    embedding_model: str,
    top_k: int,
    crops: Sequence[str] | None = None,
    doc_types: Sequence[str] | None = None,
) -> Any:
    distance = RagChunk.embedding.cosine_distance(query_vector)
    stmt = (
        select(
            RagChunk.id,
            RagChunk.document_id,
            RagDocument.title,
            RagChunk.content,
            RagChunk.section,
            RagChunk.page_start,
            RagChunk.page_end,
            RagChunk.crops,
            RagChunk.doc_type,
            RagDocument.source_org,
            RagDocument.source_path,
            RagChunk.as_of,
            (1 - distance).label("similarity"),
        )
        .join(RagDocument, RagDocument.id == RagChunk.document_id)
        .where(RagChunk.embedding_model == embedding_model)
    )
    if crops:
        stmt = stmt.where(RagChunk.crops.overlap(list(crops)))
    if doc_types:
        stmt = stmt.where(RagChunk.doc_type == any_(_ids_param("doc_types", doc_types)))
    # Similarity thresholds are applied after the scan: a WHERE on distance disables the index.
    return stmt.order_by(distance).limit(top_k)


async def search_chunks(
    session: AsyncSession,
    query_vector: list[float],
    *,
    embedding_model: str,
    top_k: int,
    crops: Sequence[str] | None = None,
    doc_types: Sequence[str] | None = None,
    min_similarity: float = 0.0,
) -> list[ChunkHit]:
    if top_k > DEFAULT_EF_SEARCH:
        await session.execute(text(f"SET LOCAL hnsw.ef_search = {int(top_k)}"))
    stmt = build_search_statement(
        query_vector,
        embedding_model=embedding_model,
        top_k=top_k,
        crops=crops,
        doc_types=doc_types,
    )
    hits = [_hit(row) for row in (await session.execute(stmt)).all()]
    return [hit for hit in hits if hit.similarity >= min_similarity]


def _hit(row: Any) -> ChunkHit:
    return ChunkHit(
        chunk_id=row.id,
        document_id=row.document_id,
        title=row.title,
        content=row.content,
        section=row.section,
        page_start=row.page_start,
        page_end=row.page_end,
        crops=list(row.crops or []),
        doc_type=row.doc_type,
        source_org=row.source_org,
        source_path=row.source_path,
        as_of=row.as_of,
        similarity=float(row.similarity),
    )


def build_keyword_statement(query: str, *, top_k: int) -> Any:
    """Full-text search. `query` uses websearch syntax: words joined with " or " match any."""
    tsquery = func.websearch_to_tsquery(text(f"'{_FTS_CONFIG}'::regconfig"), query)
    rank = func.ts_rank_cd(RagChunk.content_tsv, tsquery)
    return (
        select(
            RagChunk.id,
            RagChunk.document_id,
            RagDocument.title,
            RagChunk.content,
            RagChunk.section,
            RagChunk.page_start,
            RagChunk.page_end,
            RagChunk.crops,
            RagChunk.doc_type,
            RagDocument.source_org,
            RagDocument.source_path,
            RagChunk.as_of,
            rank.label("similarity"),
        )
        .join(RagDocument, RagDocument.id == RagChunk.document_id)
        .where(RagChunk.content_tsv.op("@@")(tsquery))
        .order_by(rank.desc())
        .limit(top_k)
    )


async def keyword_search(session: AsyncSession, query: str, *, top_k: int) -> list[ChunkHit]:
    if not query.strip():
        return []
    rows = (await session.execute(build_keyword_statement(query, top_k=top_k))).all()
    return [_hit(row) for row in rows]


async def load_restrictions(session: AsyncSession) -> list[PesticideRestrictionRow]:
    return list((await session.execute(select(PesticideRestrictionRow))).scalars().all())


async def corpus_stats(session: AsyncSession) -> dict[str, Any]:
    documents = await session.scalar(select(func.count()).select_from(RagDocument))
    chunks = await session.scalar(select(func.count()).select_from(RagChunk))
    models = (await session.execute(select(RagChunk.embedding_model).distinct())).scalars().all()
    restrictions = await session.scalar(select(func.count()).select_from(PesticideRestrictionRow))
    return {
        "documents": documents or 0,
        "chunks": chunks or 0,
        "embedding_models": list(models),
        "pesticide_restrictions": restrictions or 0,
    }
