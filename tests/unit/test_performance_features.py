"""Phase 15: answer cache, autocommit reads with stale-connection retry, startup warm-up."""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import DBAPIError

from app.core.config import Settings
from app.db import Database
from app.main import create_app
from app.rag.retriever import RetrievalResult
from app.services.answer_cache import AnswerCache, normalize_question
from tests.fakes import FakeLLM
from tests.unit.test_chat_rag import FakeRetriever, chunk

URL = "/api/v1/chat"

# ---------------------------------------------------------------- cache unit


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("Symptoms of early blight in tomato?", "  symptoms of EARLY blight   in tomato "),
        ("टमाटर में झुलसा का इलाज।", "टमाटर में झुलसा का इलाज"),
        ("టమాటా తెగులు?", "టమాటా తెగులు"),
    ],
)
def test_normalize_question_equivalents(a: str, b: str) -> None:
    assert normalize_question(a) == normalize_question(b)


def test_cache_key_depends_on_language_model_and_prompt() -> None:
    base = AnswerCache.key("q", None, "m", "p1")
    assert base != AnswerCache.key("q", "te", "m", "p1")
    assert base != AnswerCache.key("q", None, "m2", "p1")
    assert base != AnswerCache.key("q", None, "m", "p2")


def test_cache_ttl_and_lru() -> None:
    clock = FakeClock()
    cache = AnswerCache(max_items=2, ttl_seconds=100, clock=clock)
    response: Any = object()
    cache.put("a", response)
    cache.put("b", response)
    assert cache.get("a") is response  # a is now most recent
    cache.put("c", response)
    assert cache.get("b") is None  # evicted (least recently used)
    clock.now = 101
    assert cache.get("a") is None  # expired
    assert cache.hits == 1 and cache.misses == 2


def test_disabled_cache() -> None:
    cache = AnswerCache(max_items=0, ttl_seconds=100)
    assert not cache.enabled
    cache.put("a", object())  # type: ignore[arg-type]
    assert cache.get("a") is None


# ---------------------------------------------------------------- cache in the API


def client_for(llm: FakeLLM, retrieval: RetrievalResult | None = None) -> TestClient:
    settings = Settings(_env_file=None, answer_cache_size=100, enable_dev_ui=True)  # type: ignore[call-arg]
    app = create_app(settings, llm=llm)
    app.state.retriever = FakeRetriever(retrieval or RetrievalResult(chunks=[chunk("c1", "text")]))
    return TestClient(app)


def test_repeat_question_served_from_cache() -> None:
    llm = FakeLLM(outcomes=["Answer one.", "Answer two."])
    client = client_for(llm)
    first = client.post(URL, json={"message": "Early blight in tomato?"}).json()
    second = client.post(URL, json={"message": "  early blight in TOMATO "}).json()
    assert first["cached"] is False
    assert second["cached"] is True
    assert second["response"] == "Answer one."
    assert second["sources"] == first["sources"]
    assert second["conversation_id"] != first["conversation_id"]  # each request gets its own
    assert len(llm.requests) == 1  # no second LLM call
    assert second["debug"] is None


def test_cache_respects_conversation_id_and_language() -> None:
    llm = FakeLLM(outcomes=["English answer.", "ಕನ್ನಡ ಉತ್ತರ."])
    client = client_for(llm)
    client.post(URL, json={"message": "blight"})
    other_language = client.post(URL, json={"message": "blight", "language": "kn"}).json()
    assert other_language["cached"] is False  # different requested language
    again = client.post(URL, json={"message": "blight", "conversation_id": "conv_12345678"}).json()
    assert again["cached"] is True
    assert again["conversation_id"] == "conv_12345678"


def test_degraded_answers_are_not_cached() -> None:
    llm = FakeLLM(outcomes=["Ungrounded answer."])
    client = client_for(llm, RetrievalResult(error="llm_quota_exhausted"))
    client.post(URL, json={"message": "blight"})
    second = client.post(URL, json={"message": "blight"}).json()
    assert second["cached"] is False
    assert len(llm.requests) == 2


def test_validation_applies_before_cache() -> None:
    client = client_for(FakeLLM())
    assert client.post(URL, json={"message": "x" * 2001}).status_code == 400


def test_stream_served_from_cache() -> None:
    llm = FakeLLM(outcomes=["Cached streaming answer."])
    client = client_for(llm)
    client.post(URL, json={"message": "blight"})
    events: list[tuple[str, Any]] = []
    with client.stream("POST", URL + "/stream", json={"message": "blight"}) as response:
        name = None
        for line in response.iter_lines():
            if line.startswith("event: "):
                name = line[7:]
            elif line.startswith("data: "):
                events.append((name, json.loads(line[6:])))
    assert [n for n, _ in events] == ["meta", "delta", "done"]
    assert events[1][1]["text"] == "Cached streaming answer."
    assert events[2][1]["cached"] is True
    assert len(llm.requests) == 1


# ---------------------------------------------------------------- database reads


class StaleThenOk:
    def __init__(self, failures: int, invalidated: bool = True) -> None:
        self.failures = failures
        self.invalidated = invalidated
        self.calls = 0

    async def __call__(self, session: Any) -> str:
        self.calls += 1
        if self.calls <= self.failures:
            raise DBAPIError(
                "SELECT 1", {}, ConnectionError("closed"), connection_invalidated=self.invalidated
            )
        return "rows"


def database() -> Database:
    from app.db import create_engine

    return Database(create_engine(Settings(_env_file=None)))  # type: ignore[call-arg]


async def test_read_retries_once_on_stale_connection(monkeypatch: pytest.MonkeyPatch) -> None:
    db = database()

    class NoopSession:
        async def __aenter__(self) -> object:
            return object()

        async def __aexit__(self, *args: object) -> None:
            return None

    monkeypatch.setattr(db, "read_sessionmaker", lambda: NoopSession())
    query = StaleThenOk(failures=1)
    assert await db.read(query) == "rows"
    assert query.calls == 2

    persistent = StaleThenOk(failures=5)
    with pytest.raises(DBAPIError):
        await db.read(persistent)
    assert persistent.calls == 2  # only one retry

    not_stale = StaleThenOk(failures=1, invalidated=False)
    with pytest.raises(DBAPIError):
        await db.read(not_stale)
    assert not_stale.calls == 1  # real query errors are not retried
    await db.aclose()


def test_read_sessions_use_autocommit() -> None:
    db = database()
    bind = db.read_sessionmaker.kw["bind"]
    assert bind.get_execution_options()["isolation_level"] == "AUTOCOMMIT"


def test_pre_ping_off_and_recycle_configured() -> None:
    db = database()
    assert db.engine.pool._pre_ping is False  # type: ignore[attr-defined]
    assert db.engine.pool._recycle == 240  # type: ignore[attr-defined]


# ---------------------------------------------------------------- warm-up


def test_warmup_runs_and_failures_are_not_fatal(capsys: pytest.CaptureFixture[str]) -> None:
    class FailingCheckLLM(FakeLLM):
        async def check(self) -> None:
            raise ConnectionError("no network")

    settings = Settings(_env_file=None, warmup_on_startup=True, log_json=True)  # type: ignore[call-arg]
    app = create_app(settings, llm=FailingCheckLLM())
    with TestClient(app):
        pass
    out = capsys.readouterr().out
    record = next(json.loads(line) for line in out.splitlines() if '"warmup_done"' in line)
    assert record["failed"] == {"llm": "ConnectionError"}
