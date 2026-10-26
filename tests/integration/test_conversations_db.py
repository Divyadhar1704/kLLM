"""Conversation history and feedback on the real database. Skipped unless KRISHIMITRA_DB_TESTS=1.

Requires `alembic upgrade head` (migration 0002). Test conversations use ids prefixed
`zztest-` and are removed afterwards (messages and feedback cascade or are deleted first).
"""

from __future__ import annotations

import os
import uuid
from collections.abc import AsyncIterator

import pytest
from sqlalchemy import delete

from app.core.config import Settings
from app.core.errors import NotFoundError
from app.db import Database
from app.models.conversation import Conversation, Feedback
from app.services.conversations import ConversationStore

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("KRISHIMITRA_DB_TESTS") != "1",
        reason="set KRISHIMITRA_DB_TESTS=1 to run against the configured database",
    ),
]

OWNER = "zz-app:farmer-1"


@pytest.fixture
async def database() -> AsyncIterator[Database]:
    db = Database.from_settings(Settings(_env_file=".env"))  # type: ignore[call-arg]
    yield db
    async with db.session() as session:
        await session.execute(delete(Feedback).where(Feedback.conversation_id.like("zztest-%")))
        await session.execute(delete(Conversation).where(Conversation.id.like("zztest-%")))
    await db.aclose()


async def test_history_round_trip_in_order(database: Database) -> None:
    store = ConversationStore(database, timeout_seconds=15)
    conversation = f"zztest-{uuid.uuid4().hex[:12]}"
    first = await store.save_turn(
        conversation_id=conversation,
        owner=OWNER,
        language="te",
        question="q1",
        answer="a1",
        sources=[],
    )
    second = await store.save_turn(
        conversation_id=conversation,
        owner=OWNER,
        language="te",
        question="q2",
        answer="a2",
        sources=[{"index": 1, "title": "IPM"}],
        diagnosis={"status": "uncertain"},
    )
    assert first and second
    turns = await store.history(conversation, owner=OWNER)
    assert [(t.role, t.text) for t in turns] == [
        ("user", "q1"),
        ("model", "a1"),
        ("user", "q2"),
        ("model", "a2"),
    ]
    assert await store.history("zztest-never-used", owner=OWNER) == ()

    feedback_id = await store.save_feedback(
        owner=OWNER, conversation_id=None, message_id=second, rating="up", comment="good"
    )
    assert feedback_id
    with pytest.raises(NotFoundError):
        await store.save_feedback(
            owner=OWNER,
            conversation_id=None,
            message_id=uuid.uuid4().hex,
            rating="up",
            comment=None,
        )
