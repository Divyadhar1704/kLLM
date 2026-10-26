"""Client applications allowed to call the API (one row per API key)."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, String, func
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class ApiClient(Base):
    __tablename__ = "api_clients"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    # Stable slug used in rate-limit keys, conversation owners and logs: "krishindra-backend".
    name: Mapped[str] = mapped_column(String(40), unique=True)
    # Only the SHA-256 of the key is stored; the key itself is shown once, when created.
    key_hash: Mapped[str] = mapped_column(String(64), unique=True)
    key_prefix: Mapped[str] = mapped_column(String(12))  # first characters, to recognise a key
    # A trusted backend (e.g. Krishindra's server) may act for its users via X-User-Id.
    trusted_backend: Mapped[bool] = mapped_column(Boolean, default=False)
    note: Mapped[str | None] = mapped_column(String(200), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
