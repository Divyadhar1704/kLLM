"""RAG knowledge base: documents, chunks (halfvec + HNSW + full text), pesticide restrictions.

Revision ID: 0001
Revises:
Create Date: 2026-10-08
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import HALFVEC
from sqlalchemy.dialects import postgresql

from app.core.config import get_settings

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SCHEMA = get_settings().postgres_schema
EMBEDDING_DIMENSIONS = 1536
MIN_PGVECTOR = (0, 7)  # halfvec + halfvec_cosine_ops
TABLES = ("rag_documents", "rag_chunks", "pesticide_restrictions")


def _timestamps() -> list[sa.Column]:  # type: ignore[type-arg]
    return [
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    ]


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    if not op.get_context().as_sql:  # offline (--sql) mode has no connection to query
        version = op.get_bind().execute(
            sa.text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
        ).scalar_one()
        if tuple(int(p) for p in str(version).split(".")[:2]) < MIN_PGVECTOR:
            raise RuntimeError(f"pgvector >= 0.7 required for halfvec, found {version}")

    op.create_table(
        "rag_documents",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("sha256", sa.String(64), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("doc_type", sa.String(64), nullable=False),
        sa.Column("source_org", sa.Text()),
        sa.Column("source_path", sa.Text(), nullable=False),
        sa.Column("as_of", sa.Date()),
        sa.Column("crops", postgresql.ARRAY(sa.Text()), server_default="{}", nullable=False),
        sa.Column("language", sa.String(8), nullable=False),
        sa.Column("pages", sa.Integer(), nullable=False),
        sa.Column("chunk_count", sa.Integer(), nullable=False),
        *_timestamps(),
        sa.UniqueConstraint("sha256", name="uq_rag_documents_sha256"),
        schema=SCHEMA,
    )

    op.create_table(
        "rag_chunks",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column(
            "document_id",
            sa.String(32),
            sa.ForeignKey(f"{SCHEMA}.rag_documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("section", sa.Text()),
        sa.Column("page_start", sa.Integer()),
        sa.Column("page_end", sa.Integer()),
        sa.Column("crops", postgresql.ARRAY(sa.Text()), server_default="{}", nullable=False),
        sa.Column("doc_type", sa.String(64), nullable=False),
        sa.Column("language", sa.String(8), nullable=False),
        sa.Column("as_of", sa.Date()),
        sa.Column("token_estimate", sa.Integer(), nullable=False),
        sa.Column("embedding_model", sa.String(64), nullable=False),
        sa.Column("embedding", HALFVEC(EMBEDDING_DIMENSIONS), nullable=False),
        sa.Column(
            "content_tsv",
            postgresql.TSVECTOR(),
            sa.Computed("to_tsvector('english'::regconfig, content)", persisted=True),
        ),
        *_timestamps(),
        schema=SCHEMA,
    )
    op.create_index("ix_rag_chunks_document_id", "rag_chunks", ["document_id"], schema=SCHEMA)
    op.create_index("ix_rag_chunks_doc_type", "rag_chunks", ["doc_type"], schema=SCHEMA)
    op.create_index(
        "ix_rag_chunks_crops", "rag_chunks", ["crops"], schema=SCHEMA, postgresql_using="gin"
    )
    op.create_index(
        "ix_rag_chunks_content_tsv",
        "rag_chunks",
        ["content_tsv"],
        schema=SCHEMA,
        postgresql_using="gin",
    )
    op.execute(
        f'CREATE INDEX ix_rag_chunks_embedding_hnsw ON "{SCHEMA}".rag_chunks '
        "USING hnsw (embedding halfvec_cosine_ops) WITH (m = 16, ef_construction = 64)"
    )

    op.create_table(
        "pesticide_restrictions",
        sa.Column("id", sa.Integer(), sa.Identity(), primary_key=True),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("normalized", sa.Text(), nullable=False),
        sa.Column("aliases", postgresql.ARRAY(sa.Text()), server_default="{}", nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("details", sa.Text()),
        schema=SCHEMA,
    )
    op.create_index(
        "ix_pesticide_restrictions_normalized", "pesticide_restrictions", ["normalized"], schema=SCHEMA
    )

    # Defense in depth for Supabase: RLS on (no policies = no access for API roles), and the
    # API roles get no rights on this schema. The backend connects as the owner, which bypasses RLS.
    for table in TABLES:
        op.execute(f'ALTER TABLE "{SCHEMA}".{table} ENABLE ROW LEVEL SECURITY')
    op.execute(
        f"""
        DO $$
        BEGIN
            IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'anon') THEN
                EXECUTE 'REVOKE ALL ON SCHEMA "{SCHEMA}" FROM anon, authenticated';
                EXECUTE 'REVOKE ALL ON ALL TABLES IN SCHEMA "{SCHEMA}" FROM anon, authenticated';
            END IF;
        END $$;
        """
    )


def downgrade() -> None:
    op.drop_table("pesticide_restrictions", schema=SCHEMA)
    op.drop_table("rag_chunks", schema=SCHEMA)
    op.drop_table("rag_documents", schema=SCHEMA)
