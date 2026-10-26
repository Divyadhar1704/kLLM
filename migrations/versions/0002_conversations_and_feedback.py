"""Conversations, messages, media records and feedback.

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-15
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

from app.core.config import get_settings

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SCHEMA = get_settings().postgres_schema
TABLES = ("conversations", "conversation_messages", "media_records", "feedback")


def upgrade() -> None:
    op.create_table(
        "conversations",
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column("language", sa.String(length=8)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("last_message_at", sa.DateTime(timezone=True)),
        schema=SCHEMA,
    )
    op.create_table(
        "conversation_messages",
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column(
            "conversation_id",
            sa.String(length=64),
            sa.ForeignKey(f'{SCHEMA}.conversations.id', ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("role", sa.String(length=16), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("language", sa.String(length=8)),
        sa.Column("sources", sa.JSON(), nullable=True),
        sa.Column("diagnosis", sa.JSON(), nullable=True),
        sa.Column("latency_ms", sa.Float(), nullable=True),
        sa.Column("usage_input_tokens", sa.Integer(), nullable=True),
        sa.Column("usage_output_tokens", sa.Integer(), nullable=True),
        sa.Column("usage_thinking_tokens", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        schema=SCHEMA,
    )
    op.create_index(
        "ix_conversation_messages_conversation_id",
        "conversation_messages",
        ["conversation_id"],
        schema=SCHEMA,
    )

    op.create_table(
        "media_records",
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column(
            "conversation_id",
            sa.String(length=64),
            sa.ForeignKey(f'{SCHEMA}.conversations.id', ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "message_id",
            sa.String(length=64),
            sa.ForeignKey(f'{SCHEMA}.conversation_messages.id', ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("kind", sa.String(length=16), nullable=False),
        sa.Column("mime_type", sa.String(length=64), nullable=False),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        sa.Column("sha256", sa.String(length=64)),
        sa.Column("width", sa.Integer()),
        sa.Column("height", sa.Integer()),
        sa.Column("duration_seconds", sa.Float()),
        sa.Column("metadata", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        schema=SCHEMA,
    )
    op.create_index("ix_media_records_conversation_id", "media_records", ["conversation_id"], schema=SCHEMA)
    op.create_index("ix_media_records_message_id", "media_records", ["message_id"], schema=SCHEMA)

    op.create_table(
        "feedback",
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column(
            "conversation_id",
            sa.String(length=64),
            sa.ForeignKey(f'{SCHEMA}.conversations.id', ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "message_id",
            sa.String(length=64),
            sa.ForeignKey(f'{SCHEMA}.conversation_messages.id', ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("rating", sa.String(length=8), nullable=False),
        sa.Column("comment", sa.Text(), nullable=True),
        sa.Column("metadata", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        schema=SCHEMA,
    )
    op.create_index("ix_feedback_conversation_id", "feedback", ["conversation_id"], schema=SCHEMA)
    op.create_index("ix_feedback_message_id", "feedback", ["message_id"], schema=SCHEMA)

    for table in TABLES:
        op.execute(f'ALTER TABLE "{SCHEMA}".{table} ENABLE ROW LEVEL SECURITY')
    op.execute(
        f'''
        DO $$
        BEGIN
            IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'anon') THEN
                EXECUTE 'REVOKE ALL ON SCHEMA "{SCHEMA}" FROM anon, authenticated';
                EXECUTE 'REVOKE ALL ON ALL TABLES IN SCHEMA "{SCHEMA}" FROM anon, authenticated';
            END IF;
        END $$;
        '''
    )


def downgrade() -> None:
    op.drop_table("feedback", schema=SCHEMA)
    op.drop_table("media_records", schema=SCHEMA)
    op.drop_table("conversation_messages", schema=SCHEMA)
    op.drop_table("conversations", schema=SCHEMA)
