"""Conversation history, answer storage and feedback (fakes; no database or AI calls)."""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any

from fastapi.testclient import TestClient

from app.core.config import Settings
from app.core.errors import NotFoundError
from app.main import create_app
from app.models.conversation import ConversationMessage
from app.rag.retriever import RetrievalResult
from app.schemas.chat import ChatResponse, TokenUsage
from app.services.chat import ChatService, retrieval_query
from app.services.conversations import MAX_TURN_CHARS, ConversationStore
from app.services.llm import ChatTurn
from tests.fakes import FakeLLM
from tests.unit.test_chat_rag import FakeRetriever, FakeSafety, chunk

URL = "/api/v1/chat"
CONV = "conv-aaaa1111"


class FakeStore:
    """In-memory ConversationStore."""

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.turns: dict[str, list[ChatTurn]] = {}
        self.saved: list[dict[str, Any]] = []
        self.feedback: list[dict[str, Any]] = []

    async def history(self, conversation_id: str) -> tuple[ChatTurn, ...]:
        return tuple(self.turns.get(conversation_id, [])[-6:])

    async def save_turn(self, **kwargs: Any) -> str | None:
        if self.fail:
            return None
        self.saved.append(kwargs)
        self.turns.setdefault(kwargs["conversation_id"], []).extend(
            [ChatTurn("user", kwargs["question"]), ChatTurn("model", kwargs["answer"])]
        )
        message_id = uuid.uuid4().hex
        kwargs["message_id"] = message_id
        return message_id

    async def save_feedback(self, **kwargs: Any) -> str:
        known = {s["message_id"] for s in self.saved}
        if kwargs["message_id"] is not None and kwargs["message_id"] not in known:
            raise NotFoundError("Unknown message_id.")
        self.feedback.append(kwargs)
        return uuid.uuid4().hex


def build(
    outcomes: list[str], store: FakeStore | None, **overrides: Any
) -> tuple[TestClient, FakeLLM]:
    llm = FakeLLM(outcomes=outcomes)
    app = create_app(
        Settings(_env_file=None, enable_dev_ui=True, **overrides),  # type: ignore[arg-type]
        llm=llm,
    )
    app.state.retriever = FakeRetriever(
        RetrievalResult(chunks=[chunk("c1", "Spray mancozeb 75% WP")], crops=["tomato"])
    )
    app.state.safety = FakeSafety()
    app.state.conversations = store
    return TestClient(app), llm


def ask(client: TestClient, message: str, conversation_id: str = CONV, **extra: Any) -> Any:
    body = {"message": message, "conversation_id": conversation_id, "language": "en", **extra}
    response = client.post(URL, json=body)
    assert response.status_code == 200, response.text
    return response.json()


# ---------------------------------------------------------------- chat flow


def test_follow_up_sends_previous_turns_to_the_model() -> None:
    store = FakeStore()
    client, llm = build(["Early blight [1].", "Spray every 10 days [1]."], store)
    first = ask(client, "What are the brown spots on my tomato leaves?")
    second = ask(client, "How often should I spray?")
    assert len(llm.requests[0].messages) == 1
    turns = llm.requests[1].messages
    assert [t.role for t in turns] == ["user", "model", "user"]
    assert turns[0].text == "What are the brown spots on my tomato leaves?"
    assert turns[1].text == "Early blight [1]."
    assert "How often should I spray?" in turns[2].text and "SOURCES" in turns[2].text
    assert (
        first["message_id"] and second["message_id"] and first["message_id"] != second["message_id"]
    )
    assert store.saved[1]["question"] == "How often should I spray?"  # the farmer's words only


def test_answer_still_returned_when_saving_fails() -> None:
    client, _ = build(["Early blight [1]."], FakeStore(fail=True))
    body = ask(client, "What are the brown spots?")
    assert body["response"] == "Early blight [1]."
    assert body["message_id"] is None


def test_without_storage_no_message_id() -> None:
    client, llm = build(["A [1].", "B [1]."], None)
    ask(client, "first question")
    body = ask(client, "second question")
    assert body["message_id"] is None
    assert len(llm.requests[1].messages) == 1  # no history without storage


def test_cache_is_never_used_for_follow_ups() -> None:
    store = FakeStore()
    client, llm = build(
        ["Answer one [1].", "Answer two [1].", "Answer three [1]."], store, answer_cache_size=100
    )
    ask(client, "What is early blight?", "conv-aaaa1111")
    cached = ask(client, "What is early blight?", "conv-bbbb2222")  # new conversation: cache hit
    assert cached["cached"] is True and len(llm.requests) == 1
    assert cached["message_id"]  # cached answers are stored in the conversation too
    # Same words as a follow-up: the answer depends on the conversation, so the model is asked.
    follow_up = ask(client, "What is early blight?", "conv-aaaa1111")
    assert follow_up["cached"] is False and len(llm.requests) == 2
    assert len(llm.requests[1].messages) == 3
    # A follow-up's answer is not put in the cache for other conversations.
    ask(client, "How often should I spray?", "conv-bbbb2222")
    fresh = ask(client, "How often should I spray?", "conv-cccc3333")
    assert fresh["cached"] is False and len(llm.requests) == 4


def test_streamed_answers_are_saved() -> None:
    store = FakeStore()
    client, llm = build(["Early blight [1].", "Every 10 days [1]."], store)
    with client.stream(
        "POST",
        URL + "/stream",
        json={"message": "spots?", "conversation_id": CONV, "language": "en"},
    ) as response:
        events = "".join(response.iter_text())
    assert '"message_id": "' in events.split("event: done")[1]
    assert store.saved[0]["answer"] == "Early blight [1]."
    ask(client, "how often?")
    assert len(llm.requests[1].messages) == 3


def test_language_retry_keeps_the_history() -> None:
    store = FakeStore()
    client, llm = build(
        ["Early blight [1].", "Spray mancozeb now [1].", "మాంకోజెబ్ పిచికారీ చేయండి [1]."], store
    )
    ask(client, "brown spots on tomato?")
    ask(client, "ఏ మందు వాడాలి?", language="te")
    retry = llm.requests[2].messages
    assert [t.role for t in retry] == ["user", "model", "user"]
    assert retry[0].text == "brown spots on tomato?"


async def test_diagnosis_is_stored_with_the_answer() -> None:
    store = FakeStore()
    service = ChatService(llm=FakeLLM(), settings=Settings(_env_file=None), conversations=store)  # type: ignore[arg-type]
    response = ChatResponse(
        response="May be early blight.",
        language="en",
        language_detected=False,
        conversation_id=CONV,
        model="fake",
        processing_time=1.5,
        truncated=False,
        grounded=True,
        usage=TokenUsage(input_tokens=5, output_tokens=6, thinking_tokens=0),
    )
    saved = await service.save(response, question="photo", diagnosis={"status": "uncertain"})
    assert saved.message_id is not None
    assert store.saved[0]["diagnosis"] == {"status": "uncertain"}
    assert store.saved[0]["latency_ms"] == 1500.0


# ---------------------------------------------------------------- feedback


def test_feedback_on_an_answer() -> None:
    store = FakeStore()
    client, _ = build(["Early blight [1]."], store)
    answer = ask(client, "What are the brown spots?")
    response = client.post(
        "/api/v1/feedback",
        json={"message_id": answer["message_id"], "rating": "down", "comment": " wrong dose "},
    )
    assert response.status_code == 201
    assert response.json()["rating"] == "down"
    assert store.feedback[0]["comment"] == "wrong dose"


def test_feedback_errors() -> None:
    client, _ = build(["x"], FakeStore())
    post = lambda body: client.post("/api/v1/feedback", json=body)  # noqa: E731
    assert post({"message_id": "0" * 32, "rating": "up"}).status_code == 404
    assert post({"rating": "up"}).status_code == 422  # neither message nor conversation
    assert post({"message_id": "not-an-id", "rating": "up"}).status_code == 422
    assert post({"message_id": "0" * 32, "rating": "meh"}).status_code == 422
    no_store, _ = build(["x"], None)
    response = no_store.post("/api/v1/feedback", json={"message_id": "0" * 32, "rating": "up"})
    assert response.status_code == 503


# ---------------------------------------------------------------- ConversationStore


class FakeSession:
    def __init__(self, rows: list[Any] | None = None, *, fail: bool = False) -> None:
        self.rows = rows or []
        self.fail = fail
        self.added: list[Any] = []

    async def execute(self, statement: Any) -> list[Any]:
        return self.rows

    async def get(self, model: Any, key: str) -> Any:
        return None

    def add(self, item: Any) -> None:
        if self.fail:
            raise ConnectionError("database down")
        self.added.append(item)


class FakeDatabase:
    def __init__(self, session: FakeSession, *, read_error: Exception | None = None) -> None:
        self._session = session
        self._read_error = read_error

    async def read(self, query: Callable[[Any], Awaitable[Any]]) -> Any:
        if self._read_error is not None:
            raise self._read_error
        return await query(self._session)

    @asynccontextmanager
    async def session(self) -> AsyncIterator[FakeSession]:
        yield self._session


def row(role: str, content: str) -> SimpleNamespace:
    return SimpleNamespace(role=role, content=content)


async def test_store_history_turns() -> None:
    newest_first = [
        row("assistant", "x" * (MAX_TURN_CHARS + 50)),
        row("user", "second question"),
        row("assistant", "first answer"),  # oldest kept row is an answer: dropped
    ]
    store = ConversationStore(FakeDatabase(FakeSession(newest_first)))  # type: ignore[arg-type]
    turns = await store.history(CONV)
    assert [t.role for t in turns] == ["user", "model"]
    assert turns[0].text == "second question"
    assert len(turns[1].text) == MAX_TURN_CHARS + 1 and turns[1].text.endswith("…")


async def test_store_history_failure_returns_no_history() -> None:
    db = FakeDatabase(FakeSession(), read_error=ConnectionError("down"))
    assert await ConversationStore(db).history(CONV) == ()  # type: ignore[arg-type]


async def test_store_save_turn_orders_question_before_answer() -> None:
    session = FakeSession()
    store = ConversationStore(FakeDatabase(session))  # type: ignore[arg-type]
    message_id = await store.save_turn(
        conversation_id=CONV, language="te", question="q", answer="a", sources=[]
    )
    conversation, question, answer = session.added
    assert conversation.id == CONV and conversation.language == "te"
    assert isinstance(question, ConversationMessage) and question.role == "user"
    assert answer.role == "assistant" and answer.id == message_id
    assert answer.created_at > question.created_at  # same transaction: order set explicitly


async def test_store_save_turn_failure_returns_none() -> None:
    store = ConversationStore(FakeDatabase(FakeSession(fail=True)))  # type: ignore[arg-type]
    result = await store.save_turn(
        conversation_id=CONV, language="en", question="q", answer="a", sources=[]
    )
    assert result is None


# ---------------------------------------------------------------- follow-up retrieval


def test_follow_up_retrieval_keeps_the_crop() -> None:
    store = FakeStore()
    client, _ = build(["Early blight [1].", "Spray mancozeb [1]."], store)
    ask(client, "brown spots with rings on my tomato leaves")
    ask(client, "which medicine and what dose?")
    queries = client.app.state.retriever.queries  # type: ignore[attr-defined]
    assert queries[0] == "brown spots with rings on my tomato leaves"
    # The earlier question (crop) + the follow-up + the disease named in the last answer.
    assert queries[1] == (
        "brown spots with rings on my tomato leaves\nwhich medicine and what dose?\nEarly blight"
    )


def test_retrieval_query_rules() -> None:
    history = (
        ChatTurn("user", "spots on leaves"),  # no crop
        ChatTurn("model", "This may be rust on your cowpea plants."),
    )
    # Crop only in the answer (e.g. after a photo): its crop id is added, not the long answer.
    assert retrieval_query("what dose?", history) == "cowpea\nwhat dose?\nRust"
    # A crop in the new question wins; no history means no change.
    assert retrieval_query("what dose for chilli?", history) == "what dose for chilli?"
    assert retrieval_query("what dose?", ()) == "what dose?"
    # The farmer's question wins over a later answer that mentions several crops.
    mixed = (
        ChatTurn("user", "brown spots on my tomato leaves"),
        ChatTurn("model", "Unlike wheat rust or cotton wilt, this is early blight."),
    )
    # Only the crop's own diseases are added: "wheat rust" is ignored for tomato.
    assert retrieval_query("what dose?", mixed) == (
        "brown spots on my tomato leaves\nwhat dose?\nEarly blight"
    )
    # An answer naming several crops is not used.
    many = (ChatTurn("user", "spots"), ChatTurn("model", "Seen in wheat and cotton."))
    assert retrieval_query("what dose?", many) == "what dose?"
    # No crop anywhere: unchanged.
    assert retrieval_query("what dose?", (ChatTurn("user", "spots"),)) == "what dose?"
