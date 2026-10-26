"""RAG knowledge-base tables."""

from __future__ import annotations

from datetime import date, datetime

from pgvector.sqlalchemy import HALFVEC
from sqlalchemy import (
    Computed,
    Date,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY, TSVECTOR
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base

# Fixed by the database column type. Changing it requires a migration and a full re-embed.
EMBEDDING_DIMENSIONS = 1536


class RagDocument(Base):
    __tablename__ = "rag_documents"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    sha256: Mapped[str] = mapped_column(String(64), unique=True)
    title: Mapped[str] = mapped_column(Text)
    doc_type: Mapped[str] = mapped_column(String(64))
    source_org: Mapped[str | None] = mapped_column(Text)
    source_path: Mapped[str] = mapped_column(Text)
    as_of: Mapped[date | None] = mapped_column(Date)
    crops: Mapped[list[str]] = mapped_column(ARRAY(Text), server_default="{}")
    language: Mapped[str] = mapped_column(String(8))
    pages: Mapped[int] = mapped_column(Integer)
    chunk_count: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class RagChunk(Base):
    __tablename__ = "rag_chunks"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    document_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("rag_documents.id", ondelete="CASCADE"), index=True
    )
    chunk_index: Mapped[int] = mapped_column(Integer)
    content: Mapped[str] = mapped_column(Text)
    section: Mapped[str | None] = mapped_column(Text)
    page_start: Mapped[int | None] = mapped_column(Integer)
    page_end: Mapped[int | None] = mapped_column(Integer)
    crops: Mapped[list[str]] = mapped_column(ARRAY(Text), server_default="{}")
    doc_type: Mapped[str] = mapped_column(String(64))
    language: Mapped[str] = mapped_column(String(8))
    as_of: Mapped[date | None] = mapped_column(Date)
    token_estimate: Mapped[int] = mapped_column(Integer)
    embedding_model: Mapped[str] = mapped_column(String(64))
    embedding: Mapped[list[float]] = mapped_column(HALFVEC(EMBEDDING_DIMENSIONS))
    # Hybrid search (Phase 10): the corpus is English, so English stemming.
    content_tsv: Mapped[str] = mapped_column(
        TSVECTOR, Computed("to_tsvector('english'::regconfig, content)", persisted=True)
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class PesticideRestrictionRow(Base):
    __tablename__ = "pesticide_restrictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(Text)
    normalized: Mapped[str] = mapped_column(Text, index=True)
    aliases: Mapped[list[str]] = mapped_column(ARRAY(Text), server_default="{}")
    status: Mapped[str] = mapped_column(String(32))
    details: Mapped[str | None] = mapped_column(Text)
