"""Media records: where the cleaned file is stored, and an index for retention cleanup.

Revision ID: 0003
Revises: 0002
Create Date: 2026-10-26
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

from app.core.config import get_settings

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SCHEMA = get_settings().postgres_schema


def upgrade() -> None:
    # media_records has never held rows (nothing wrote to it before this phase), so the new
    # NOT NULL columns need no default.
    op.add_column(
        "media_records",
        sa.Column("storage_backend", sa.String(length=16), nullable=False),
        schema=SCHEMA,
    )
    op.add_column(
        "media_records",
        sa.Column("storage_key", sa.String(length=512), nullable=False),
        schema=SCHEMA,
    )
    op.create_unique_constraint(
        "uq_media_records_storage_key", "media_records", ["storage_backend", "storage_key"],
        schema=SCHEMA,
    )
    op.create_index(
        "ix_media_records_created_at", "media_records", ["created_at"], schema=SCHEMA
    )


def downgrade() -> None:
    op.drop_index("ix_media_records_created_at", table_name="media_records", schema=SCHEMA)
    op.drop_constraint("uq_media_records_storage_key", "media_records", schema=SCHEMA)
    op.drop_column("media_records", "storage_key", schema=SCHEMA)
    op.drop_column("media_records", "storage_backend", schema=SCHEMA)
