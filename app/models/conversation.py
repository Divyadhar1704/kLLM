"""Conversation, message, media and feedback tables for Phase 20."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    language: Mapped[str | None] = mapped_column(String(8), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
    last_message_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class ConversationMessage(Base):
    __tablename__ = "conversation_messages"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: uuid.uuid4().hex)
    conversation_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("conversations.id", ondelete="CASCADE"), index=True
    )
    role: Mapped[str] = mapped_column(String(16))
    content: Mapped[str] = mapped_column(Text)
    language: Mapped[str | None] = mapped_column(String(8), nullable=True)
    sources: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    diagnosis: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    usage_input_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    usage_output_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    usage_thinking_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class MediaRecord(Base):
    __tablename__ = "media_records"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: uuid.uuid4().hex)
    conversation_id: Mapped[str | None] = mapped_column(
        String(64), ForeignKey("conversations.id", ondelete="SET NULL"), nullable=True, index=True
    )
    message_id: Mapped[str | None] = mapped_column(
        String(64),
        ForeignKey("conversation_messages.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    kind: Mapped[str] = mapped_column(String(16))
    mime_type: Mapped[str] = mapped_column(String(64))
    size_bytes: Mapped[int] = mapped_column(Integer)
    sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    height: Mapped[int | None] = mapped_column(Integer, nullable=True)
    duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    meta: Mapped[dict[str, Any] | None] = mapped_column("metadata", JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Feedback(Base):
    __tablename__ = "feedback"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: uuid.uuid4().hex)
    conversation_id: Mapped[str | None] = mapped_column(
        String(64), ForeignKey("conversations.id", ondelete="SET NULL"), nullable=True, index=True
    )
    message_id: Mapped[str | None] = mapped_column(
        String(64),
        ForeignKey("conversation_messages.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    rating: Mapped[str] = mapped_column(String(8))
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    meta: Mapped[dict[str, Any] | None] = mapped_column("metadata", JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
