"""Check database connectivity: server version, SSL, pgvector, latency, migration state.

Usage:
    python -m scripts.db_check
"""

from __future__ import annotations

import asyncio
import sys
import time

from sqlalchemy import text

from app.core.config import get_settings
from app.core.logging import setup_logging
from app.db import Database


async def main() -> int:
    settings = get_settings()
    database = Database.from_settings(settings)
    print(f"target: {settings.postgres_host}:{settings.postgres_port}/{settings.postgres_db}")
    try:
        start = time.perf_counter()
        async with database.engine.connect() as connection:
            version = (await connection.execute(text("SELECT version()"))).scalar_one()
            ssl = (
                await connection.execute(
                    text("SELECT ssl FROM pg_stat_ssl WHERE pid = pg_backend_pid()")
                )
            ).scalar()
            vector = (
                await connection.execute(
                    text(
                        "SELECT default_version, installed_version FROM pg_available_extensions "
                        "WHERE name = 'vector'"
                    )
                )
            ).one_or_none()
            migrated = (
                await connection.execute(
                    text("SELECT to_regclass(:table)"),
                    {"table": f"{settings.postgres_schema}.alembic_version"},
                )
            ).scalar()
        print(f"connected in {(time.perf_counter() - start) * 1000:.0f} ms")
        print(f"server: {str(version)[:70]}")
        print(f"ssl: {ssl}")
        if vector is None:
            print("pgvector: NOT AVAILABLE on this server")
        else:
            installed = vector[1] or "no (migration enables it)"
            print(f"pgvector: available {vector[0]}, installed {installed}")
        print(f"migrations applied: {'yes' if migrated else 'no - run `alembic upgrade head`'}")

        start = time.perf_counter()
        for _ in range(5):
            await database.check()
        print(f"query round trip: {(time.perf_counter() - start) * 1000 / 5:.0f} ms (avg of 5)")
    except Exception as exc:
        print(f"FAILED: {type(exc).__name__}: {exc}")
        return 1
    finally:
        await database.aclose()
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    setup_logging(get_settings())
    raise SystemExit(asyncio.run(main()))
