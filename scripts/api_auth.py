"""API keys for client applications, and user tokens for testing.

Keys are shown ONCE when created; only their SHA-256 is stored. Revocation takes effect within
API_KEY_CACHE_SECONDS. Scripts calling the API read the key from KRISHIMITRA_API_KEY.

Usage:
    python -m scripts.api_auth create NAME [--trusted-backend] [--note TEXT]
    python -m scripts.api_auth list
    python -m scripts.api_auth revoke NAME
    python -m scripts.api_auth token USER_ID [--minutes 60]   # needs AUTH_JWT_SECRET (testing)

NAME is a slug such as krishindra-backend. --trusted-backend lets the key act for farmers
(X-User-Id): only for a server you control, never for a key that ships inside an app.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
import uuid
from datetime import UTC, datetime

import jwt
from sqlalchemy import select

from app.core.config import Settings, get_settings
from app.core.logging import setup_logging
from app.db import Database
from app.models.api_client import ApiClient
from app.services.auth import CLIENT_NAME_PATTERN, generate_api_key, hash_api_key


def api_headers() -> dict[str, str]:
    """Headers for scripts calling the API: the key from KRISHIMITRA_API_KEY, when set."""
    key = os.environ.get("KRISHIMITRA_API_KEY")
    return {"X-API-Key": key} if key else {}


async def create(db: Database, name: str, trusted: bool, note: str | None) -> None:
    if not CLIENT_NAME_PATTERN.match(name):
        raise SystemExit("NAME must be lowercase letters, digits and dashes (max 40)")
    key = generate_api_key()
    async with db.session() as session:
        if (await session.execute(select(ApiClient).where(ApiClient.name == name))).first():
            raise SystemExit(f"a client named {name!r} already exists (revoke it first)")
        session.add(
            ApiClient(
                id=uuid.uuid4().hex,
                name=name,
                key_hash=hash_api_key(key),
                key_prefix=key[:12],
                trusted_backend=trusted,
                note=note,
            )
        )
    print(f"created {name} (trusted backend: {trusted})")
    print(f"API key (shown once, store it in a secret manager): {key}")


async def list_clients(db: Database) -> None:
    async with db.session() as session:
        clients = (
            await session.execute(select(ApiClient).order_by(ApiClient.created_at))
        ).scalars()
        for c in clients:
            state = f"REVOKED {c.revoked_at:%Y-%m-%d}" if c.revoked_at else "active"
            print(
                f"{c.name:24} {c.key_prefix}…  trusted={c.trusted_backend!s:5}  {state:18} "
                f"created {c.created_at:%Y-%m-%d}  {c.note or ''}"
            )


async def revoke(db: Database, name: str) -> None:
    async with db.session() as session:
        client = (
            await session.execute(select(ApiClient).where(ApiClient.name == name))
        ).scalar_one_or_none()
        if client is None:
            raise SystemExit(f"no client named {name!r}")
        client.revoked_at = datetime.now(UTC)
        # Free the name so a replacement key can be created under it.
        client.name = f"{name}-revoked-{int(time.time())}"[:40]
    print(f"revoked {name}")


def make_token(settings: Settings, user_id: str, minutes: int) -> str:
    if settings.auth_jwt_secret is None:
        raise SystemExit("AUTH_JWT_SECRET is not set (JWKS setups sign tokens elsewhere)")
    now = int(time.time())
    claims = {
        "sub": user_id,
        "aud": settings.auth_jwt_audience,
        "iat": now,
        "exp": now + minutes * 60,
    }
    if settings.auth_jwt_issuer:
        claims["iss"] = settings.auth_jwt_issuer
    return jwt.encode(claims, settings.auth_jwt_secret.get_secret_value(), "HS256")


async def main(args: argparse.Namespace, settings: Settings) -> int:
    if args.command == "token":
        print(make_token(settings, args.user_id, args.minutes))
        return 0
    db = Database.from_settings(settings)
    try:
        if args.command == "create":
            await create(db, args.name, args.trusted_backend, args.note)
        elif args.command == "list":
            await list_clients(db)
        else:
            await revoke(db, args.name)
    finally:
        await db.aclose()
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    new = commands.add_parser("create", help="create a client and print its key once")
    new.add_argument("name")
    new.add_argument("--trusted-backend", action="store_true")
    new.add_argument("--note")
    commands.add_parser("list", help="list clients (keys are never shown)")
    gone = commands.add_parser("revoke", help="revoke a client's key")
    gone.add_argument("name")
    tok = commands.add_parser("token", help="user token for testing (HS256 setups)")
    tok.add_argument("user_id")
    tok.add_argument("--minutes", type=int, default=60)
    settings = get_settings()
    setup_logging(settings)
    raise SystemExit(asyncio.run(main(parser.parse_args(), settings)))
