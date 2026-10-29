"""Authentication and rate-limit dependencies for /api/v1 (see app.services.auth).

    X-API-Key            client app; a trusted backend may add X-User-Id (the farmer)
    Authorization: Bearer user token signed by the Krishindra backend (sub = the farmer)
    neither              anonymous, only when AUTH_REQUIRED is false (local development)

The identified caller is stored on `request.state.principal` for the access log.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request, Security
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer

from app.core.errors import (
    BadRequestError,
    ForbiddenError,
    ServiceUnavailableError,
    UnauthorizedError,
)
from app.services.auth import ANONYMOUS, USER_ID_PATTERN, ApiKeyStore, Principal, TokenVerifier
from app.services.rate_limit import Limit, RateLimiter, enforce

_CHALLENGE = {"WWW-Authenticate": 'Bearer realm="krishimitra"'}

api_key_header = APIKeyHeader(
    name="X-API-Key",
    auto_error=False,
    description="Client application key. Trusted backends may add X-User-Id for the farmer.",
)
bearer = HTTPBearer(auto_error=False, description="User token signed by the Krishindra backend.")


async def get_principal(
    request: Request,
    api_key: Annotated[str | None, Security(api_key_header)],
    credentials: Annotated[HTTPAuthorizationCredentials | None, Security(bearer)],
) -> Principal:
    settings = request.app.state.settings
    if api_key and credentials:
        raise BadRequestError("Send either X-API-Key or a Bearer token, not both.")
    user_header = request.headers.get("x-user-id")

    if api_key:
        keys: ApiKeyStore | None = request.app.state.api_keys
        if keys is None:
            raise ServiceUnavailableError("API keys cannot be checked (no database).")
        client = await keys.lookup(api_key)
        if client is None:
            raise UnauthorizedError("Invalid API key.", headers=_CHALLENGE)
        if user_header is not None:
            if not client.trusted_backend:
                raise ForbiddenError("This API key cannot act on behalf of users.")
            if not USER_ID_PATTERN.match(user_header):
                raise BadRequestError("Invalid X-User-Id.")
        principal = Principal(client.name, "api_key", user_header)
    elif credentials:
        if user_header is not None:
            raise BadRequestError("X-User-Id is only accepted with a trusted API key.")
        verifier: TokenVerifier | None = request.app.state.token_verifier
        if verifier is None:
            raise UnauthorizedError("User tokens are not accepted here.", headers=_CHALLENGE)
        user_id = await verifier.verify(credentials.credentials)
        principal = Principal(settings.auth_jwt_client_id, "jwt", user_id)
    elif settings.auth_required:
        raise UnauthorizedError(headers=_CHALLENGE)
    else:
        principal = ANONYMOUS

    request.state.principal = principal
    return principal


PrincipalDep = Annotated[Principal, Depends(get_principal)]


def _limiter(request: Request) -> RateLimiter:
    limiter: RateLimiter = request.app.state.rate_limiter
    return limiter


def _subject(request: Request, principal: Principal) -> str:
    # Anonymous callers (local development only) are told apart by address.
    if principal.method == "anonymous":
        client = request.client
        return f"anonymous:{client.host if client else 'unknown'}"
    return principal.owner


async def ai_rate_limit(request: Request, principal: PrincipalDep) -> None:
    """Endpoints that call the AI models (chat, photo, video)."""
    settings = request.app.state.settings
    subject = _subject(request, principal)
    await enforce(
        _limiter(request),
        [
            Limit(f"user:{subject}", settings.rate_limit_per_minute, 60, "user_minute"),
            Limit(f"user:{subject}", settings.rate_limit_per_day, 86400, "user_day"),
            Limit(
                f"client:{principal.client_id}",
                settings.rate_limit_client_per_minute,
                60,
                "client_minute",
            ),
        ],
    )


async def feedback_rate_limit(request: Request, principal: PrincipalDep) -> None:
    settings = request.app.state.settings
    subject = _subject(request, principal)
    await enforce(
        _limiter(request),
        [
            Limit(
                f"feedback:{subject}",
                settings.rate_limit_feedback_per_minute,
                60,
                "feedback_minute",
            )
        ],
    )
