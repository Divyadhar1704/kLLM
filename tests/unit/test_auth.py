"""Authentication, conversation ownership and rate limits (fakes; no database or AI calls)."""

from __future__ import annotations

import json
import time
from collections.abc import Awaitable, Callable
from types import SimpleNamespace
from typing import Any

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.core.errors import RateLimitedError, ServiceUnavailableError
from app.main import create_app
from app.rag.retriever import RetrievalResult
from app.services.auth import (
    KEY_PREFIX,
    ApiClientInfo,
    ApiKeyStore,
    Principal,
    TokenVerifier,
    generate_api_key,
    hash_api_key,
)
from app.services.rate_limit import Limit, MemoryRateLimiter, RedisRateLimiter, enforce
from tests.fakes import FakeLLM
from tests.unit.test_chat_rag import FakeRetriever, FakeSafety, chunk
from tests.unit.test_conversations import FakeStore
from tests.unit.test_dev_console import SECURE_PRODUCTION

SECRET = "s" * 40
URL = "/api/v1/chat"
APP_KEY = "km_app_key_for_tests_0000000000000000000000"
BACKEND_KEY = "km_backend_key_for_tests_00000000000000000000"


class FakeApiKeys:
    def __init__(self) -> None:
        self.clients = {
            APP_KEY: ApiClientInfo("1", "partner-app", trusted_backend=False),
            BACKEND_KEY: ApiClientInfo("2", "krishindra-backend", trusted_backend=True),
        }

    async def lookup(self, key: str) -> ApiClientInfo | None:
        return self.clients.get(key)


def token(sub: str = "farmer-42", **overrides: Any) -> str:
    now = int(time.time())
    claims: dict[str, Any] = {"sub": sub, "aud": "krishimitra-api", "iat": now, "exp": now + 600}
    claims.update(overrides)
    return jwt.encode({k: v for k, v in claims.items() if v is not None}, SECRET, "HS256")


def build(
    *, store: FakeStore | None = None, outcomes: list[str] | None = None, **overrides: Any
) -> tuple[TestClient, FakeLLM]:
    llm = FakeLLM(outcomes=outcomes or ["Early blight [1]."])
    params: dict[str, Any] = {"auth_required": True, "auth_jwt_secret": SECRET}
    params.update(overrides)
    app = create_app(Settings(_env_file=None, **params), llm=llm)
    app.state.retriever = FakeRetriever(RetrievalResult(chunks=[chunk("c1", "Spray")]))
    app.state.safety = FakeSafety()
    app.state.api_keys = FakeApiKeys()
    app.state.conversations = store
    return TestClient(app), llm


def ask(client: TestClient, headers: dict[str, str], **body: Any) -> Any:
    return client.post(URL, json={"message": "brown spots on tomato?", **body}, headers=headers)


def bearer(value: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {value}"}


# ---------------------------------------------------------------- who is calling


def test_anonymous_only_when_auth_is_off() -> None:
    open_client, _ = build(auth_required=False)
    assert ask(open_client, {}).status_code == 200
    client, llm = build()
    response = ask(client, {})
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "unauthorized"
    assert response.headers["WWW-Authenticate"].startswith("Bearer")
    assert llm.requests == []  # rejected before any AI work


def test_api_keys() -> None:
    client, _ = build()
    assert ask(client, {"X-API-Key": APP_KEY}).status_code == 200
    assert ask(client, {"X-API-Key": "km_unknown"}).status_code == 401
    # Only a trusted backend may act for a farmer.
    forbidden = ask(client, {"X-API-Key": APP_KEY, "X-User-Id": "farmer-1"})
    assert forbidden.status_code == 403
    assert ask(client, {"X-API-Key": BACKEND_KEY, "X-User-Id": "farmer-1"}).status_code == 200
    bad_user = ask(client, {"X-API-Key": BACKEND_KEY, "X-User-Id": "x" * 200})
    assert bad_user.status_code == 400
    both = ask(client, {"X-API-Key": APP_KEY, **bearer(token())})
    assert both.status_code == 400


def test_api_keys_need_the_database() -> None:
    client, _ = build()
    client.app.state.api_keys = None  # type: ignore[attr-defined]
    assert ask(client, {"X-API-Key": APP_KEY}).status_code == 503


def test_user_tokens() -> None:
    client, _ = build()
    assert ask(client, bearer(token())).status_code == 200
    rejected = {
        "expired": token(exp=int(time.time()) - 120, iat=int(time.time()) - 700),
        "wrong audience": token(aud="someone-else"),
        "no expiry": token(exp=None),
        "too long-lived": token(exp=int(time.time()) + 3 * 86400),
        "bad subject": token(sub="has spaces"),
        "wrong secret": jwt.encode(
            {
                "sub": "u",
                "aud": "krishimitra-api",
                "iat": int(time.time()),
                "exp": int(time.time()) + 60,
            },
            "x" * 40,
            "HS256",
        ),
        "alg none": jwt.encode({"sub": "u", "aud": "krishimitra-api"}, None, "none"),
        "garbage": "not-a-token",
    }
    for case, value in rejected.items():
        response = ask(client, bearer(value))
        assert response.status_code == 401, case
    with_user_header = ask(client, {**bearer(token()), "X-User-Id": "someone-else"})
    assert with_user_header.status_code == 400


def test_tokens_refused_when_not_configured() -> None:
    client, _ = build(auth_jwt_secret=None)
    assert ask(client, bearer(token())).status_code == 401


async def test_jwks_tokens_and_no_algorithm_switching() -> None:
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    settings = Settings(_env_file=None, auth_jwt_jwks_url="https://auth.example.com/jwks.json")
    verifier = TokenVerifier(settings)
    verifier._jwks = SimpleNamespace(  # type: ignore[assignment]
        get_signing_key_from_jwt=lambda _token: SimpleNamespace(key=private_key.public_key())
    )
    now = int(time.time())
    claims = {"sub": "farmer-7", "aud": "krishimitra-api", "iat": now, "exp": now + 300}
    assert await verifier.verify(jwt.encode(claims, private_key, "RS256")) == "farmer-7"
    # An HS256 token cannot be passed off against an RS256 key setup.
    with pytest.raises(Exception, match="Invalid or expired token"):
        await verifier.verify(jwt.encode(claims, SECRET, "HS256"))


# ---------------------------------------------------------------- API key store


class CountingDatabase:
    def __init__(self, result: ApiClientInfo | None, *, fail: bool = False) -> None:
        self.result, self.fail, self.queries = result, fail, 0

    async def read(self, query: Callable[[Any], Awaitable[Any]]) -> Any:
        self.queries += 1
        if self.fail:
            raise ConnectionError("db down")
        return self.result


async def test_api_key_store_caches_and_fails_closed() -> None:
    db = CountingDatabase(ApiClientInfo("1", "partner-app", False))
    store = ApiKeyStore(db, cache_seconds=60)  # type: ignore[arg-type]
    key = generate_api_key()
    assert key.startswith(KEY_PREFIX) and len(key) > 40
    assert (await store.lookup(key)) == db.result
    assert (await store.lookup(key)) == db.result
    assert db.queries == 1  # second lookup from the cache
    assert await store.lookup("not-a-km-key") is None and db.queries == 1  # no DB hit
    assert hash_api_key(key) != key and len(hash_api_key(key)) == 64
    failing = ApiKeyStore(CountingDatabase(None, fail=True))  # type: ignore[arg-type]
    with pytest.raises(ServiceUnavailableError):
        await failing.lookup(generate_api_key())


# ---------------------------------------------------------------- ownership


def test_conversations_are_private_to_their_owner() -> None:
    store = FakeStore()
    client, llm = build(store=store, outcomes=["A [1].", "B [1].", "C [1]."])
    farmer_1 = {"X-API-Key": BACKEND_KEY, "X-User-Id": "farmer-1"}
    farmer_2 = {"X-API-Key": BACKEND_KEY, "X-User-Id": "farmer-2"}
    first = ask(client, farmer_1, conversation_id="conv-shared-1").json()
    assert store.saved[0]["owner"] == "krishindra-backend:farmer-1"
    # Same conversation id, different farmer: indistinguishable from an unknown id.
    stolen = ask(client, farmer_2, conversation_id="conv-shared-1")
    assert stolen.status_code == 404
    assert len(llm.requests) == 1  # nothing generated, nothing leaked
    feedback = client.post(
        "/api/v1/feedback",
        json={"message_id": first["message_id"], "rating": "down"},
        headers=farmer_2,
    )
    assert feedback.status_code == 404 and store.feedback == []  # cannot rate others' answers
    own = client.post(
        "/api/v1/feedback",
        json={"message_id": first["message_id"], "rating": "up"},
        headers=farmer_1,
    )
    assert own.status_code == 201
    # The owner continues normally.
    assert ask(client, farmer_1, conversation_id="conv-shared-1").status_code == 200


# ---------------------------------------------------------------- rate limits


def test_ai_rate_limit_per_user() -> None:
    client, llm = build(rate_limit_per_minute=2)
    farmer = {"X-API-Key": BACKEND_KEY, "X-User-Id": "farmer-1"}
    assert ask(client, farmer).status_code == 200
    assert ask(client, farmer).status_code == 200
    limited = ask(client, farmer)
    assert limited.status_code == 429
    assert limited.json()["error"]["code"] == "rate_limited"
    assert 1 <= int(limited.headers["Retry-After"]) <= 60
    assert len(llm.requests) == 2  # the limited request never reached the model
    # Another farmer has their own allowance.
    assert ask(client, {"X-API-Key": BACKEND_KEY, "X-User-Id": "farmer-2"}).status_code == 200


def test_client_wide_rate_limit() -> None:
    client, _ = build(rate_limit_client_per_minute=2)
    for user in ("a", "b"):
        assert ask(client, {"X-API-Key": BACKEND_KEY, "X-User-Id": user}).status_code == 200
    assert ask(client, {"X-API-Key": BACKEND_KEY, "X-User-Id": "c"}).status_code == 429


def test_feedback_has_its_own_limit() -> None:
    client, _ = build(store=FakeStore(), rate_limit_feedback_per_minute=1)
    headers = {"X-API-Key": APP_KEY}
    body = {"message_id": "0" * 32, "rating": "up"}
    assert client.post("/api/v1/feedback", json=body, headers=headers).status_code == 404
    assert client.post("/api/v1/feedback", json=body, headers=headers).status_code == 429
    assert ask(client, headers).status_code == 200  # chat allowance untouched


async def test_memory_limiter_windows() -> None:
    limiter = MemoryRateLimiter()
    assert (await limiter.hit("k", 60))[0] == 1
    count, reset = await limiter.hit("k", 60)
    assert count == 2 and 1 <= reset <= 60
    assert (await limiter.hit("k", 86400))[0] == 1  # separate window, separate count


class FakePipeline:
    def __init__(self, redis: FakeRedis) -> None:
        self.redis, self.ops = redis, []

    async def __aenter__(self) -> FakePipeline:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    def incr(self, name: str) -> None:
        self.ops.append(("incr", name))

    def expire(self, name: str, seconds: int, nx: bool = False) -> None:
        self.ops.append(("expire", name, seconds, nx))

    async def execute(self) -> list[Any]:
        if self.redis.down:
            raise ConnectionError("redis down")
        name = self.ops[0][1]
        self.redis.counts[name] = self.redis.counts.get(name, 0) + 1
        self.redis.expires.append(self.ops[1])
        return [self.redis.counts[name], True]


class FakeRedis:
    def __init__(self, *, down: bool = False) -> None:
        self.down, self.counts, self.expires = down, {}, []

    def pipeline(self, transaction: bool = True) -> FakePipeline:
        assert transaction is False
        return FakePipeline(self)


async def test_redis_limiter_and_fail_open() -> None:
    redis = FakeRedis()
    limiter = RedisRateLimiter(redis)
    assert (await limiter.hit("user:x", 60))[0] == 1
    assert (await limiter.hit("user:x", 60))[0] == 2
    (_, name, seconds, nx) = redis.expires[0]
    assert name.startswith("krishimitra:rl:user:x:60:") and seconds == 65 and nx is True
    with pytest.raises(RateLimitedError):
        await enforce(limiter, [Limit("user:x", 2, 60)])  # third request
    # Redis down: requests are allowed (logged), the service stays up.
    await enforce(RedisRateLimiter(FakeRedis(down=True)), [Limit("user:y", 1, 60)])


# ---------------------------------------------------------------- settings and logs


def test_auth_settings_rules() -> None:
    with pytest.raises(ValueError, match="AUTH_REQUIRED must be true"):
        Settings(_env_file=None, **{**SECURE_PRODUCTION, "auth_required": False})
    with pytest.raises(ValueError, match="at least 32 characters"):
        Settings(_env_file=None, auth_jwt_secret="short")
    with pytest.raises(ValueError, match="not both"):
        Settings(_env_file=None, auth_jwt_secret=SECRET, auth_jwt_jwks_url="https://x/jwks")
    with pytest.raises(ValueError, match="https"):
        Settings(_env_file=None, auth_jwt_jwks_url="http://x/jwks")


def test_access_log_records_caller_without_raw_user_id(
    capsys: pytest.CaptureFixture[str],
) -> None:
    client, _ = build()
    capsys.readouterr()
    ask(client, {"X-API-Key": BACKEND_KEY, "X-User-Id": "farmer-secret-99"})
    output = capsys.readouterr().out
    access = next(json.loads(line) for line in output.splitlines() if '"request_completed"' in line)
    assert access["caller"] == "krishindra-backend" and access["auth"] == "api_key"
    expected = Principal("krishindra-backend", "api_key", "farmer-secret-99").subject_hash
    assert access["user"] == expected
    assert "farmer-secret-99" not in output  # the farmer id itself is never logged


def test_health_endpoints_stay_public() -> None:
    client, _ = build()
    assert client.get("/health").status_code == 200
