"""Who is calling: client apps (API keys) and farmers (user tokens).

- API key (`X-API-Key: km_...`): identifies a client application. Only the SHA-256 of a key is
  stored. A key marked `trusted_backend` (Krishindra's server) may pass the farmer's id in
  `X-User-Id`: server-to-server, the app on the phone never holds a key.
- User token (`Authorization: Bearer <JWT>`): short-lived, signed by the Krishindra backend for
  one farmer (`sub`). Lets the phone app call this API directly without a long-lived secret.
  Accepted algorithms are fixed by configuration (HS256 with a shared secret, or RS256/ES256
  with the backend's JWKS); "none" and algorithm switching are never accepted.

`Principal.owner` ("<client>:<user>") owns conversations and keys the per-user rate limits.
"""

from __future__ import annotations

import asyncio
import hashlib
import re
import secrets
import time
from dataclasses import dataclass
from typing import Any, Literal

import jwt
from sqlalchemy import select

from app.core.config import Settings
from app.core.errors import ServiceUnavailableError, UnauthorizedError
from app.core.logging import get_logger
from app.db.session import Database
from app.models.api_client import ApiClient

logger = get_logger(__name__)

KEY_PREFIX = "km_"
USER_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.:@-]{1,128}$")
CLIENT_NAME_PATTERN = re.compile(r"^[a-z0-9-]{1,40}$")
_NEGATIVE_CACHE_SECONDS = 10  # unknown keys: short, so a new key works almost at once


@dataclass(frozen=True, slots=True)
class Principal:
    client_id: str
    method: Literal["api_key", "jwt", "anonymous"]
    user_id: str | None = None

    @property
    def owner(self) -> str:
        return f"{self.client_id}:{self.user_id}" if self.user_id else self.client_id

    @property
    def subject_hash(self) -> str:
        """Pseudonymous id for logs: farmer ids are personal data and never logged as is."""
        return hashlib.sha256(self.owner.encode()).hexdigest()[:16]


ANONYMOUS = Principal(client_id="anonymous", method="anonymous")


def generate_api_key() -> str:
    return KEY_PREFIX + secrets.token_urlsafe(32)


def hash_api_key(key: str) -> str:
    # Keys are 256-bit random values: a fast hash is enough (no password stretching needed).
    return hashlib.sha256(key.encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class ApiClientInfo:
    id: str
    name: str
    trusted_backend: bool


class ApiKeyStore:
    """Looks up API keys (cached briefly: revoking a key takes effect within the cache time)."""

    def __init__(self, db: Database, *, cache_seconds: int = 60) -> None:
        self._db = db
        self._cache_seconds = cache_seconds
        self._cache: dict[str, tuple[ApiClientInfo | None, float]] = {}

    async def lookup(self, key: str) -> ApiClientInfo | None:
        if not key.startswith(KEY_PREFIX) or len(key) > 100:
            return None
        digest = hash_api_key(key)
        now = time.monotonic()
        cached = self._cache.get(digest)
        if cached is not None and cached[1] > now:
            return cached[0]

        async def query(session: Any) -> ApiClientInfo | None:
            row = (
                await session.execute(
                    select(ApiClient.id, ApiClient.name, ApiClient.trusted_backend)
                    .where(ApiClient.key_hash == digest)
                    .where(ApiClient.revoked_at.is_(None))
                )
            ).first()
            return ApiClientInfo(row.id, row.name, row.trusted_backend) if row else None

        try:
            client = await self._db.read(query)
        except Exception as exc:  # cannot verify: refuse rather than let anyone in
            logger.warning("api_key_lookup_failed", error=type(exc).__name__)
            raise ServiceUnavailableError("Authentication is temporarily unavailable.") from exc
        ttl = self._cache_seconds if client is not None else _NEGATIVE_CACHE_SECONDS
        if len(self._cache) > 10_000:
            self._cache.clear()
        self._cache[digest] = (client, now + ttl)
        return client


class TokenVerifier:
    """Verifies user tokens from the Krishindra backend. Returns the user id (`sub`)."""

    def __init__(self, settings: Settings) -> None:
        self._audience = settings.auth_jwt_audience
        self._issuer = settings.auth_jwt_issuer
        self._max_lifetime = settings.auth_jwt_max_lifetime_seconds
        self._secret = settings.auth_jwt_secret
        self._jwks: jwt.PyJWKClient | None = None
        if settings.auth_jwt_jwks_url:
            self._jwks = jwt.PyJWKClient(settings.auth_jwt_jwks_url, lifespan=3600, timeout=10)
            self._algorithms = ["RS256", "ES256"]
        else:
            self._algorithms = ["HS256"]

    @classmethod
    def from_settings(cls, settings: Settings) -> TokenVerifier | None:
        if settings.auth_jwt_secret is None and not settings.auth_jwt_jwks_url:
            return None
        return cls(settings)

    async def verify(self, token: str) -> str:
        try:
            key: Any
            if self._jwks is not None:
                # Fetching the JWKS is blocking HTTP (cached for an hour): run it in a thread.
                key = (await asyncio.to_thread(self._jwks.get_signing_key_from_jwt, token)).key
            else:
                assert self._secret is not None  # noqa: S101 - one of the two is configured
                key = self._secret.get_secret_value()
            claims = jwt.decode(
                token,
                key,
                algorithms=self._algorithms,
                audience=self._audience,
                issuer=self._issuer,
                options={"require": ["exp", "iat", "sub", "aud"]},
                leeway=30,
            )
        except jwt.PyJWTError as exc:
            logger.info("token_rejected", reason=type(exc).__name__)
            raise UnauthorizedError("Invalid or expired token.") from exc
        if int(claims["exp"]) - int(claims["iat"]) > self._max_lifetime:
            raise UnauthorizedError("Token lifetime is too long.")
        user_id = str(claims["sub"])
        if not USER_ID_PATTERN.match(user_id):
            raise UnauthorizedError("Invalid token subject.")
        return user_id
