"""API keys and conversation ownership on the real database. Skipped unless KRISHIMITRA_DB_TESTS=1.

Requires `alembic upgrade head` (migration 0004). Test rows use `zztest-` names/ids and are
removed afterwards.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime

import pytest
from sqlalchemy import delete, update

from app.core.config import Settings
from app.core.errors import NotFoundError
from app.db import Database
from app.models.api_client import ApiClient
from app.models.conversation import Conversation
from app.services.auth import ApiKeyStore, generate_api_key, hash_api_key
from app.services.conversations import ConversationStore

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
        await session.execute(delete(ApiClient).where(ApiClient.name.like("zztest-%")))
        await session.execute(delete(Conversation).where(Conversation.id.like("zztest-%")))
    await db.aclose()


async def test_api_key_lookup_and_revocation(database: Database) -> None:
    key = generate_api_key()
    name = f"zztest-{uuid.uuid4().hex[:8]}"
    async with database.session() as session:
        session.add(
            ApiClient(
                id=uuid.uuid4().hex,
                name=name,
                key_hash=hash_api_key(key),
                key_prefix=key[:12],
                trusted_backend=True,
            )
        )
    client = await ApiKeyStore(database, cache_seconds=0).lookup(key)
    assert client is not None and client.name == name and client.trusted_backend
    assert await ApiKeyStore(database).lookup(generate_api_key()) is None
    async with database.session() as session:
        await session.execute(
            update(ApiClient).where(ApiClient.name == name).values(revoked_at=datetime.now(UTC))
        )
    assert await ApiKeyStore(database, cache_seconds=0).lookup(key) is None


async def test_conversation_ownership(database: Database) -> None:
    store = ConversationStore(database, timeout_seconds=15)
    conversation = f"zztest-{uuid.uuid4().hex[:12]}"
    message_id = await store.save_turn(
        conversation_id=conversation,
        owner="zz-app:farmer-1",
        language="en",
        question="q",
        answer="a",
        sources=[],
    )
    assert message_id is not None
    assert len(await store.history(conversation, owner="zz-app:farmer-1")) == 2
    with pytest.raises(NotFoundError):
        await store.history(conversation, owner="zz-app:farmer-2")
    with pytest.raises(NotFoundError):
        await store.check_owner(conversation, owner="zz-app:farmer-2")
    assert (
        await store.save_turn(
            conversation_id=conversation,
            owner="zz-app:farmer-2",
            language="en",
            question="intruder",
            answer="x",
            sources=[],
        )
        is None
    )
    with pytest.raises(NotFoundError):
        await store.save_feedback(
            owner="zz-app:farmer-2",
            conversation_id=None,
            message_id=message_id,
            rating="down",
            comment=None,
        )
    assert await store.save_feedback(
        owner="zz-app:farmer-1",
        conversation_id=None,
        message_id=message_id,
        rating="up",
        comment=None,
    )
    assert len(await store.history(conversation, owner="zz-app:farmer-1")) == 2  # unchanged
