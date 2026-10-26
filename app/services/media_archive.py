"""Keeps the cleaned media of answered photo/video questions, linked to the answer.

    answer sent -> (background) store each cleaned JPEG -> one media_records row per file

- Only diagnosed media is kept (confident / uncertain). "Not a plant" photos may show people and
  "retake" photos are unusable: neither is ever stored.
- Background and best-effort: storage never delays or fails an answer. If the database write
  fails, the files just written are deleted again so no file exists without a record.
- `cleanup` enforces the retention period (files first, then records).
"""

from __future__ import annotations

import asyncio
import hashlib
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

from sqlalchemy import delete, select

from app.core.errors import NotFoundError
from app.core.logging import get_logger
from app.db.session import Database
from app.models.conversation import MediaRecord
from app.services.storage import MediaStore, media_key
from app.vision.diagnosis import Diagnosis
from app.vision.images import PreparedImage

logger = get_logger(__name__)

ARCHIVED_STATUSES = frozenset({"confident", "uncertain"})


@dataclass(frozen=True, slots=True)
class MediaItem:
    kind: Literal["photo", "video_frame"]
    image: PreparedImage  # cleaned JPEG, as sent to the model
    frame_time: float | None = None  # seconds into the video


class MediaArchive:
    def __init__(
        self,
        store: MediaStore,
        db: Database,
        *,
        url_ttl_seconds: int = 900,
        timeout_seconds: float = 60.0,
    ) -> None:
        self._store = store
        self._db = db
        self._url_ttl = url_ttl_seconds
        self._timeout = timeout_seconds
        self._tasks: set[asyncio.Task[list[str]]] = set()

    @property
    def store(self) -> MediaStore:
        return self._store

    def archive_later(
        self,
        *,
        conversation_id: str,
        message_id: str,
        items: list[MediaItem],
        diagnosis: Diagnosis,
    ) -> None:
        """Start archiving in the background (the answer is not delayed)."""
        if not items or diagnosis.status not in ARCHIVED_STATUSES:
            return
        task = asyncio.create_task(
            self.archive(
                conversation_id=conversation_id,
                message_id=message_id,
                items=items,
                diagnosis=diagnosis,
            )
        )
        self._tasks.add(task)  # keep a reference until done (asyncio holds only weak ones)
        task.add_done_callback(self._tasks.discard)

    async def archive(
        self,
        *,
        conversation_id: str,
        message_id: str,
        items: list[MediaItem],
        diagnosis: Diagnosis,
    ) -> list[str]:
        """Store the files and their records. Returns the media ids ([] on failure)."""
        now = datetime.now(UTC)
        records: list[MediaRecord] = []
        stored: list[str] = []
        try:
            async with asyncio.timeout(self._timeout):
                for item in items:
                    media_id = uuid.uuid4().hex
                    key = media_key(conversation_id=conversation_id, media_id=media_id, created=now)
                    await self._store.put(key, item.image.data, item.image.mime_type)
                    stored.append(key)
                    records.append(
                        self._record(
                            media_id, key, item, conversation_id, message_id, diagnosis, now
                        )
                    )
                async with self._db.session() as session:
                    session.add_all(records)
        except Exception as exc:
            logger.warning(
                "media_archive_failed",
                conversation_id=conversation_id,
                error=type(exc).__name__,
                stored_files=len(stored),
            )
            await self._delete_quietly(stored)
            return []
        logger.info(
            "media_archived",
            conversation_id=conversation_id,
            message_id=message_id,
            files=len(records),
            bytes=sum(r.size_bytes for r in records),
            backend=self._store.backend,
        )
        return [r.id for r in records]

    def _record(
        self,
        media_id: str,
        key: str,
        item: MediaItem,
        conversation_id: str,
        message_id: str,
        diagnosis: Diagnosis,
        now: datetime,
    ) -> MediaRecord:
        quality = item.image.quality
        meta: dict[str, Any] = {
            "frame_time": item.frame_time,
            "quality": {
                "brightness": quality.brightness,
                "sharpness": quality.sharpness,
                "issues": list(quality.issues),
            },
            "diagnosis": {
                "status": diagnosis.status,
                "crop": diagnosis.crop,
                "condition": diagnosis.condition.id,
                "confidence": diagnosis.confidence,
                "model": diagnosis.model,
            },
        }
        video = diagnosis.video
        return MediaRecord(
            id=media_id,
            conversation_id=conversation_id,
            message_id=message_id,
            kind=item.kind,
            storage_backend=self._store.backend,
            storage_key=key,
            mime_type=item.image.mime_type,
            size_bytes=len(item.image.data),
            sha256=hashlib.sha256(item.image.data).hexdigest(),
            width=item.image.model_size[0] or None,
            height=item.image.model_size[1] or None,
            duration_seconds=video.duration_seconds if video is not None else None,
            meta=meta,
            created_at=now,
        )

    async def _delete_quietly(self, keys: list[str]) -> None:
        for key in keys:
            try:
                await self._store.delete(key)
            except Exception as exc:  # the retention cleanup / lifecycle rule removes leftovers
                logger.warning("media_delete_failed", key=key, error=type(exc).__name__)

    async def drain(self, timeout_seconds: float = 10.0) -> None:
        """Wait for background archiving on shutdown."""
        if self._tasks:
            await asyncio.wait(set(self._tasks), timeout=timeout_seconds)

    async def download_url(self, media_id: str) -> str:
        """Short-lived link for staff review tools (no public endpoint)."""
        async with self._db.session() as session:
            record = await session.get(MediaRecord, media_id)
        if record is None or record.storage_backend != self._store.backend:
            raise NotFoundError("Unknown media id.")
        return await self._store.url(record.storage_key, self._url_ttl)

    async def cleanup(self, *, older_than_days: int, batch: int = 200) -> int:
        """Delete records (and their files) older than the retention period. Returns the count."""
        cutoff = datetime.now(UTC) - timedelta(days=older_than_days)
        removed = 0
        while True:
            async with self._db.session() as session:
                rows = (
                    await session.execute(
                        select(MediaRecord.id, MediaRecord.storage_key)
                        .where(MediaRecord.created_at < cutoff)
                        .where(MediaRecord.storage_backend == self._store.backend)
                        .limit(batch)
                    )
                ).all()
            if not rows:
                return removed
            for _, key in rows:
                await self._store.delete(key)  # files first: a record never points at nothing
            async with self._db.session() as session:
                await session.execute(
                    delete(MediaRecord).where(MediaRecord.id.in_([row.id for row in rows]))
                )
            removed += len(rows)
            logger.info("media_cleanup_batch", removed=len(rows), cutoff=cutoff.isoformat())
