"""API clients (hashed keys) and conversation owners.

Revision ID: 0004
Revises: 0003
Create Date: 2026-10-26
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

from app.core.config import get_settings

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SCHEMA = get_settings().postgres_schema


def upgrade() -> None:
    op.create_table(
        "api_clients",
        sa.Column("id", sa.String(length=32), primary_key=True),
        sa.Column("name", sa.String(length=40), nullable=False, unique=True),
        sa.Column("key_hash", sa.String(length=64), nullable=False, unique=True),
        sa.Column("key_prefix", sa.String(length=12), nullable=False),
        sa.Column("trusted_backend", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("note", sa.String(length=200)),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column("revoked_at", sa.DateTime(timezone=True)),
        schema=SCHEMA,
    )
    op.execute(f'ALTER TABLE "{SCHEMA}".api_clients ENABLE ROW LEVEL SECURITY')
    op.execute(
        f"""
        DO $$
        BEGIN
            IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'anon') THEN
                EXECUTE 'REVOKE ALL ON "{SCHEMA}".api_clients FROM anon, authenticated';
            END IF;
        END $$;
        """
    )
    op.add_column(
        "conversations", sa.Column("owner_id", sa.String(length=180)), schema=SCHEMA
    )
    op.create_index(
        "ix_conversations_owner_id", "conversations", ["owner_id"], schema=SCHEMA
    )


def downgrade() -> None:
    op.drop_index("ix_conversations_owner_id", table_name="conversations", schema=SCHEMA)
    op.drop_column("conversations", "owner_id", schema=SCHEMA)
    op.drop_table("api_clients", schema=SCHEMA)
