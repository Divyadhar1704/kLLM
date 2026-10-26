"""Alembic environment (async). Uses the application's settings and engine configuration."""

from __future__ import annotations

import asyncio

from alembic import context
from sqlalchemy import text
from sqlalchemy.engine import Connection

from app.core.config import get_settings
from app.core.logging import setup_logging
from app.db.session import create_engine
from app.models import Base

settings = get_settings()
setup_logging(settings)
SCHEMA = settings.postgres_schema
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    context.configure(
        url=settings.sqlalchemy_database_url,
        target_metadata=target_metadata,
        literal_binds=True,
        version_table_schema=SCHEMA,
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        version_table_schema=SCHEMA,
        include_schemas=True,
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    engine = create_engine(settings)
    try:
        async with engine.connect() as connection:
            # The version table lives in our schema, so the schema must exist first.
            await connection.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{SCHEMA}"'))
            await connection.commit()
            await connection.run_sync(do_run_migrations)
            await connection.commit()
    finally:
        await engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
