"""Conversation history and feedback storage (Postgres).

Storing history is best-effort: a slow or failing database never fails or delays a chat answer
by more than `timeout_seconds`; the answer is returned without a `message_id` instead.

- Every conversation has an owner (`Principal.owner`). Another owner's conversation_id is
  answered exactly like an unknown one (404): its existence is not revealed.
- History is returned as chat turns (user / model), oldest first, each turn truncated, so the
  model sees a real multi-turn conversation instead of history pasted into the question.
- Timestamps are set here, not by the database: the user question and the answer are written
  in one transaction, where Postgres `now()` is identical for both and would make their order
  random.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.errors import NotFoundError
from app.core.logging import get_logger
from app.db.session import Database
from app.models.conversation import Conversation, ConversationMessage, Feedback
from app.services.llm import ChatTurn

logger = get_logger(__name__)

HISTORY_MESSAGES = 6  # last 3 question/answer pairs
MAX_TURN_CHARS = 1200  # long answers are cut: the gist is enough for follow-ups
LEGACY_OWNER = "anonymous"  # conversations stored before owners existed (development data)


def owner_of(conversation: Conversation) -> str:
    return conversation.owner_id or LEGACY_OWNER


def _not_found() -> NotFoundError:
    return NotFoundError("Unknown conversation_id.")


class ConversationStore:
    def __init__(self, db: Database, *, timeout_seconds: float = 3.0) -> None:
        self._db = db
        self._timeout = timeout_seconds

    async def history(
        self, conversation_id: str, *, owner: str, max_messages: int = HISTORY_MESSAGES
    ) -> tuple[ChatTurn, ...]:
        """Previous turns, oldest first. Empty for a new conversation or on a database problem.
        Raises NotFoundError if the conversation belongs to someone else."""

        async def query(session: AsyncSession) -> tuple[str | None, list[tuple[str, str]]]:
            conversation = await session.get(Conversation, conversation_id)
            if conversation is None:
                return None, []
            rows = await session.execute(
                select(ConversationMessage.role, ConversationMessage.content)
                .where(ConversationMessage.conversation_id == conversation_id)
                .order_by(desc(ConversationMessage.created_at))
                .limit(max_messages)
            )
            return owner_of(conversation), [(row.role, row.content) for row in rows]

        try:
            async with asyncio.timeout(self._timeout):
                stored_owner, rows = await self._db.read(query)
        except Exception as exc:  # degrade to no history, never fail the answer
            logger.warning("conversation_history_failed", error=type(exc).__name__)
            return ()
        if stored_owner is not None and stored_owner != owner:
            logger.warning("conversation_owner_mismatch", conversation_id=conversation_id)
            raise _not_found()
        turns = [
            ChatTurn(
                role="user" if role == "user" else "model",
                text=content if len(content) <= MAX_TURN_CHARS else content[:MAX_TURN_CHARS] + "…",
            )
            for role, content in reversed(rows)
        ]
        while turns and turns[0].role != "user":  # a conversation must start with the user
            turns.pop(0)
        return tuple(turns)

    async def check_owner(self, conversation_id: str, *, owner: str) -> None:
        """For requests that do not load history (photo / video): NotFoundError if the
        conversation belongs to someone else. A database problem is not an error here."""

        async def query(session: AsyncSession) -> str | None:
            conversation = await session.get(Conversation, conversation_id)
            return owner_of(conversation) if conversation is not None else None

        try:
            async with asyncio.timeout(self._timeout):
                stored_owner = await self._db.read(query)
        except Exception as exc:
            logger.warning("conversation_owner_check_failed", error=type(exc).__name__)
            return
        if stored_owner is not None and stored_owner != owner:
            logger.warning("conversation_owner_mismatch", conversation_id=conversation_id)
            raise _not_found()

    async def save_turn(
        self,
        *,
        conversation_id: str,
        owner: str,
        language: str,
        question: str,
        answer: str,
        sources: list[dict[str, Any]],
        diagnosis: dict[str, Any] | None = None,
        latency_ms: float | None = None,
        usage: tuple[int, int, int] = (0, 0, 0),
    ) -> str | None:
        """Store one question/answer pair. Returns the answer's message id, None on failure."""
        message_id = uuid.uuid4().hex
        asked_at = datetime.now(UTC)
        try:
            async with asyncio.timeout(self._timeout), self._db.session() as session:
                conversation = await session.get(Conversation, conversation_id)
                if conversation is None:
                    session.add(
                        Conversation(
                            id=conversation_id,
                            owner_id=owner,
                            language=language,
                            last_message_at=asked_at,
                        )
                    )
                elif owner_of(conversation) != owner:
                    # Checked before answering; reaching this means a race on a new id.
                    logger.warning("conversation_owner_mismatch", conversation_id=conversation_id)
                    return None
                else:
                    conversation.language = language
                    conversation.last_message_at = asked_at
                session.add(
                    ConversationMessage(
                        conversation_id=conversation_id,
                        role="user",
                        content=question,
                        language=language,
                        created_at=asked_at,
                    )
                )
                session.add(
                    ConversationMessage(
                        id=message_id,
                        conversation_id=conversation_id,
                        role="assistant",
                        content=answer,
                        language=language,
                        sources=sources,
                        diagnosis=diagnosis,
                        latency_ms=latency_ms,
                        usage_input_tokens=usage[0],
                        usage_output_tokens=usage[1],
                        usage_thinking_tokens=usage[2],
                        created_at=asked_at + timedelta(microseconds=1),
                    )
                )
        except Exception as exc:
            logger.warning(
                "conversation_save_failed",
                conversation_id=conversation_id,
                error=type(exc).__name__,
            )
            return None
        return message_id

    async def save_feedback(
        self,
        *,
        owner: str,
        conversation_id: str | None,
        message_id: str | None,
        rating: str,
        comment: str | None,
    ) -> str:
        """Store a rating for one of the caller's answers or conversations. NotFoundError for
        unknown ids and for other owners' ids alike."""
        async with self._db.session() as session:
            if message_id is not None:
                message = await session.get(ConversationMessage, message_id)
                if message is None or message.role != "assistant":
                    raise NotFoundError("Unknown message_id.")
                if conversation_id is not None and message.conversation_id != conversation_id:
                    raise NotFoundError("Unknown message_id.")
                conversation_id = message.conversation_id
            if conversation_id is None:
                raise _not_found()
            conversation = await session.get(Conversation, conversation_id)
            if conversation is None or owner_of(conversation) != owner:
                raise NotFoundError(
                    "Unknown message_id." if message_id else "Unknown conversation_id."
                )
            feedback = Feedback(
                id=uuid.uuid4().hex,
                conversation_id=conversation_id,
                message_id=message_id,
                rating=rating,
                comment=comment,
                meta={"source": "api"},
            )
            session.add(feedback)
        return feedback.id
