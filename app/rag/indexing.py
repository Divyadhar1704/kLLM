"""Plan a knowledge-base sync: ingested chunks + cached embeddings -> database rows.

Pure functions (no I/O) so the rules are unit-tested without a database:
- a chunk is indexed only if its embedding exists for the configured model/dimensions;
- a document is indexed only if at least one of its chunks is indexed;
- documents with some chunks still unembedded are reported as partial.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from app.models import EMBEDDING_DIMENSIONS
from app.rag.ingestion.models import Chunk, DocumentRecord
from app.rag.ingestion.pesticides import PesticideRestriction


@dataclass(slots=True)
class IndexPlan:
    documents: list[dict[str, Any]] = field(default_factory=list)
    chunks: list[dict[str, Any]] = field(default_factory=list)
    restrictions: list[dict[str, Any]] = field(default_factory=list)
    missing_vectors: dict[str, int] = field(default_factory=dict)  # title -> chunks not embedded
    total_chunks: int = 0

    @property
    def indexed_chunks(self) -> int:
        return len(self.chunks)


def _as_date(value: str | None) -> date | None:
    return date.fromisoformat(value) if value else None


def plan_index(
    records: list[DocumentRecord],
    chunks: list[Chunk],
    vectors: dict[str, list[float]],
    *,
    key_for: Callable[[Chunk], str],
    embedding_model: str,
    restrictions: list[PesticideRestriction] | None = None,
) -> IndexPlan:
    plan = IndexPlan(total_chunks=len(chunks))
    by_id = {r.doc_id: r for r in records}
    indexed_per_doc: Counter[str] = Counter()
    missing: Counter[str] = Counter()

    for chunk in chunks:
        vector = vectors.get(key_for(chunk))
        if vector is None:
            missing[chunk.doc_id] += 1
            continue
        if len(vector) != EMBEDDING_DIMENSIONS:
            raise ValueError(
                f"Embedding has {len(vector)} dimensions; the database column is "
                f"{EMBEDDING_DIMENSIONS}. Re-embed or migrate."
            )
        indexed_per_doc[chunk.doc_id] += 1
        plan.chunks.append(
            {
                "id": chunk.chunk_id,
                "document_id": chunk.doc_id,
                "chunk_index": chunk.chunk_index,
                "content": chunk.text,
                "section": chunk.section,
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
                "crops": chunk.crops,
                "doc_type": chunk.doc_type,
                "language": chunk.language,
                "as_of": _as_date(chunk.as_of),
                "token_estimate": chunk.token_estimate,
                "embedding_model": embedding_model,
                "embedding": vector,
            }
        )

    for doc_id in indexed_per_doc:
        record = by_id.get(doc_id)
        if record is None:
            raise ValueError(f"Chunk references unknown document {doc_id}: re-run ingestion")
        plan.documents.append(
            {
                "id": record.doc_id,
                "sha256": record.sha256,
                "title": record.title,
                "doc_type": record.doc_type,
                "source_org": record.source_org,
                "source_path": record.source_path,
                "as_of": _as_date(record.as_of),
                "crops": record.crops,
                "language": record.language,
                "pages": record.pages,
                "chunk_count": indexed_per_doc[doc_id],
            }
        )
    plan.missing_vectors = {
        (by_id[d].title if d in by_id else d): n for d, n in sorted(missing.items())
    }
    plan.restrictions = [
        {
            "name": r.name,
            "normalized": r.normalized,
            "aliases": r.aliases,
            "status": r.status,
            "details": r.details,
        }
        for r in restrictions or []
    ]
    return plan
