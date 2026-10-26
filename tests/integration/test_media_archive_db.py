"""Media archive on the real database + a local folder. Skipped unless KRISHIMITRA_DB_TESTS=1.

Requires `alembic upgrade head` (migration 0003). Uses a `zztest-` conversation that is
removed afterwards; files go to pytest's temporary folder.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import delete, select, update

from app.core.config import Settings
from app.db import Database
from app.models.conversation import Conversation, MediaRecord
from app.services.conversations import ConversationStore
from app.services.media_archive import MediaArchive, MediaItem
from app.services.storage import LocalMediaStore
from tests.unit.test_image_chat import diagnosis, prepared_image
from tests.unit.test_media_storage import files_in

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("KRISHIMITRA_DB_TESTS") != "1",
        reason="set KRISHIMITRA_DB_TESTS=1 to run against the configured database",
    ),
]


@pytest.fixture
async def database() -> AsyncIterator[Database]:
    db = Database.from_settings(Settings(_env_file=".env"))  # type: ignore[call-arg]
    yield db
    async with db.session() as session:
        await session.execute(delete(Conversation).where(Conversation.id.like("zztest-%")))
    await db.aclose()


async def test_archive_link_and_retention(database: Database, tmp_path: Path) -> None:
    conversation = f"zztest-{uuid.uuid4().hex[:12]}"
    message_id = await ConversationStore(database, timeout_seconds=15).save_turn(
        conversation_id=conversation,
        owner="zz-app",
        language="en",
        question="q",
        answer="a",
        sources=[],
    )
    assert message_id is not None
    archive = MediaArchive(LocalMediaStore(tmp_path), database)
    ids = await archive.archive(
        conversation_id=conversation,
        message_id=message_id,
        items=[MediaItem("photo", prepared_image(b"real-db-photo"))],
        diagnosis=diagnosis(),
    )
    assert len(ids) == 1
    url = await archive.download_url(ids[0])
    assert url.startswith("file:") and url.endswith(".jpg")

    # Not yet due: cleanup keeps it. Backdated past the retention period: file and row go.
    assert await archive.cleanup(older_than_days=30) == 0
    async with database.session() as session:
        await session.execute(
            update(MediaRecord)
            .where(MediaRecord.id == ids[0])
            .values(created_at=datetime.now(UTC) - timedelta(days=31))
        )
    assert await archive.cleanup(older_than_days=30) == 1
    assert files_in(tmp_path) == []
    async with database.session() as session:
        assert (
            await session.execute(select(MediaRecord).where(MediaRecord.id == ids[0]))
        ).first() is None
