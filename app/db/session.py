"""Async engine + session factory.

- Connection pool sized for the Supabase session pooler (dev) and Cloud SQL (prod).
- Tables declared without a schema are mapped to settings.postgres_schema.
- Connections are recycled before poolers drop them; reads retry once on a stale connection
  (cheaper than a pre-ping round trip on every checkout).
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any, TypeVar

from sqlalchemy import text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import Settings


def connect_args(settings: Settings) -> dict[str, Any]:
    return {
        "ssl": settings.postgres_ssl,
        "statement_cache_size": settings.postgres_statement_cache_size,
        "command_timeout": settings.db_command_timeout_seconds,
        "timeout": settings.db_pool_timeout_seconds,  # connection establishment
        "server_settings": {"application_name": f"krishimitra-{settings.app_env}"},
    }


def create_engine(settings: Settings) -> AsyncEngine:
    return create_async_engine(
        settings.sqlalchemy_database_url,
        connect_args=connect_args(settings),
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
        pool_timeout=settings.db_pool_timeout_seconds,
        pool_pre_ping=settings.db_pool_pre_ping,
        pool_recycle=settings.db_pool_recycle_seconds,
        execution_options={"schema_translate_map": {None: settings.postgres_schema}},
    )


T = TypeVar("T")


class Database:
    def __init__(self, engine: AsyncEngine) -> None:
        self.engine = engine
        self.sessionmaker = async_sessionmaker(engine, expire_on_commit=False)
        # Reads run without BEGIN/COMMIT: saves two round trips per query.
        self.read_sessionmaker = async_sessionmaker(
            engine.execution_options(isolation_level="AUTOCOMMIT"), expire_on_commit=False
        )

    @classmethod
    def from_settings(cls, settings: Settings) -> Database:
        return cls(create_engine(settings))

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        """Session in a transaction: commits on success, rolls back on error."""
        async with self.sessionmaker() as session, session.begin():
            yield session

    async def read(self, query: Callable[[AsyncSession], Awaitable[T]]) -> T:
        """Run a read-only query in autocommit mode; retry once if the connection was stale.

        Statements that need a transaction (e.g. SET LOCAL) must use `session()` instead.
        """
        for attempt in (1, 2):
            try:
                async with self.read_sessionmaker() as session:
                    return await query(session)
            except DBAPIError as exc:
                if attempt == 2 or not exc.connection_invalidated:
                    raise
        raise AssertionError("unreachable")  # pragma: no cover

    async def warmup(self, connections: int) -> None:
        """Open connections up front so the first requests do not pay for TLS + auth."""

        async def ping() -> None:
            async with self.engine.connect() as connection:
                await connection.execute(text("SELECT 1"))

        await asyncio.gather(*(ping() for _ in range(connections)))

    async def check(self) -> None:
        async with self.engine.connect() as connection:
            await connection.execute(text("SELECT 1"))

    async def aclose(self) -> None:
        await self.engine.dispose()
