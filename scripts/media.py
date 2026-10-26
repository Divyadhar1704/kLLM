"""Stored media (cleaned photos / video frames of chat questions): list, link, retention cleanup.

Staff tool. There is no public media endpoint: links printed here are signed and short-lived
(MEDIA_URL_TTL_SECONDS) on GCS, or local file paths in development.

Usage:
    python -m scripts.media list [--conversation ID] [--limit 20]
    python -m scripts.media url MEDIA_ID
    python -m scripts.media cleanup [--days N] [--yes]     # default: MEDIA_RETENTION_DAYS
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import UTC, datetime, timedelta

from sqlalchemy import func, select

from app.core.config import Settings, get_settings
from app.core.logging import setup_logging
from app.db import Database
from app.models.conversation import MediaRecord
from app.services.media_archive import MediaArchive
from app.services.storage import build_media_store


async def list_media(db: Database, conversation: str | None, limit: int) -> None:
    query = select(MediaRecord).order_by(MediaRecord.created_at.desc()).limit(limit)
    if conversation:
        query = query.where(MediaRecord.conversation_id == conversation)
    async with db.session() as session:
        records = (await session.execute(query)).scalars().all()
    if not records:
        print("no media stored")
    for r in records:
        diagnosis = (r.meta or {}).get("diagnosis", {})
        frame = (r.meta or {}).get("frame_time")
        print(
            f"{r.id}  {r.created_at:%Y-%m-%d %H:%M}  {r.kind:11} {r.size_bytes / 1024:6.0f} KB  "
            f"{r.width}x{r.height}{f' @{frame}s' if frame is not None else ''}  "
            f"{diagnosis.get('status', '?')}: {diagnosis.get('condition', '?')}  "
            f"conv={r.conversation_id}"
        )


async def cleanup(archive: MediaArchive, db: Database, days: int, confirmed: bool) -> None:
    cutoff = datetime.now(UTC) - timedelta(days=days)
    async with db.session() as session:
        due = (
            await session.execute(
                select(func.count())
                .select_from(MediaRecord)
                .where(MediaRecord.created_at < cutoff)
                .where(MediaRecord.storage_backend == archive.store.backend)
            )
        ).scalar_one()
    print(f"{due} media older than {days} days (before {cutoff:%Y-%m-%d})")
    if due and not confirmed:
        print("dry run: add --yes to delete them")
        return
    if due:
        print(f"deleted {await archive.cleanup(older_than_days=days)}")


async def main(args: argparse.Namespace, settings: Settings) -> int:
    store = build_media_store(settings)
    if store is None:
        print("MEDIA_STORAGE is none: nothing is stored")
        return 1
    db = Database.from_settings(settings)
    archive = MediaArchive(store, db, url_ttl_seconds=settings.media_url_ttl_seconds)
    try:
        if args.command == "list":
            await list_media(db, args.conversation, args.limit)
        elif args.command == "url":
            print(await archive.download_url(args.media_id))
        else:
            await cleanup(archive, db, args.days or settings.media_retention_days, args.yes)
    finally:
        await db.aclose()
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    listing = commands.add_parser("list", help="latest stored media")
    listing.add_argument("--conversation")
    listing.add_argument("--limit", type=int, default=20)
    link = commands.add_parser("url", help="short-lived download link")
    link.add_argument("media_id")
    clean = commands.add_parser("cleanup", help="delete media past the retention period")
    clean.add_argument("--days", type=int, help="default: MEDIA_RETENTION_DAYS")
    clean.add_argument("--yes", action="store_true", help="really delete (default: dry run)")
    settings = get_settings()
    setup_logging(settings)
    raise SystemExit(asyncio.run(main(parser.parse_args(), settings)))
