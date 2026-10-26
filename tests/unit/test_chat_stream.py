"""Streaming chat endpoint (Server-Sent Events)."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app
from app.rag.retriever import RetrievalResult
from app.services.llm.base import LLMRequest, StreamDelta, StreamEvent
from app.services.llm.errors import LLMRateLimitedError, LLMUnavailableError
from app.services.safety import PesticideSafetyChecker, Restriction
from tests.fakes import FakeLLM
from tests.unit.test_chat_rag import FakeRetriever, chunk

URL = "/api/v1/chat/stream"


class FakeSafety:
    async def ensure_loaded(self) -> PesticideSafetyChecker:
        return PesticideSafetyChecker([Restriction("Phorate", (), "banned", None)])


class FailsMidStream(FakeLLM):
    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamEvent]:
        self.requests.append(request)
        yield StreamDelta(text="Partial ")
        raise LLMUnavailableError()


def client_for(
    llm: FakeLLM, retrieval: RetrievalResult | None = None, **overrides: Any
) -> TestClient:
    params: dict[str, Any] = {"enable_dev_ui": True}
    params.update(overrides)
    app = create_app(Settings(_env_file=None, **params), llm=llm)  # type: ignore[arg-type]
    app.state.retriever = FakeRetriever(retrieval) if retrieval is not None else None
    app.state.safety = FakeSafety()
    return TestClient(app, raise_server_exceptions=False)


def read_events(client: TestClient, payload: dict[str, Any]) -> tuple[Any, list[tuple[str, Any]]]:
    events: list[tuple[str, Any]] = []
    with client.stream("POST", URL, json=payload) as response:
        if response.headers.get("content-type", "").startswith("text/event-stream"):
            name = None
            for line in response.iter_lines():
                if line.startswith("event: "):
                    name = line[len("event: ") :]
                elif line.startswith("data: ") and name is not None:
                    events.append((name, json.loads(line[len("data: ") :])))
                    name = None
        else:
            response.read()
    return response, events


def test_stream_event_sequence_and_headers() -> None:
    retrieval = RetrievalResult(chunks=[chunk("c1", "Spray mancozeb")], crops=["tomato"])
    llm = FakeLLM(outcomes=["Spray mancozeb as per label [1]."])
    response, events = read_events(
        client_for(llm, retrieval), {"message": "టమాటాలో బ్లైట్?", "debug": True}
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-accel-buffering"] == "no"
    assert "x-request-id" in response.headers

    names = [name for name, _ in events]
    assert names[0] == "meta"
    assert names[-1] == "done"
    assert set(names[1:-1]) == {"delta"}

    meta = events[0][1]
    assert meta["language"] == "te"
    assert meta["grounded"] is True
    assert meta["sources"][0]["title"] == "AESA based IPM - Tomato"

    streamed = "".join(data["text"] for name, data in events if name == "delta")
    done = events[-1][1]
    assert streamed == done["response"]  # deltas reassemble the final answer exactly
    assert done["response"].startswith("Spray mancozeb")
    assert done["conversation_id"] == meta["conversation_id"]
    assert done["debug"]["timings_ms"]["time_to_first_token"] >= 0


def test_stream_safety_warning_streamed_as_delta() -> None:
    llm = FakeLLM(outcomes=["Use phorate granules."])
    _, events = read_events(client_for(llm), {"message": "stem borer", "language": "en"})
    deltas = [data["text"] for name, data in events if name == "delta"]
    assert "Phorate is banned" in deltas[-1]
    done = events[-1][1]
    assert done["safety_warnings"] and "Phorate" in done["safety_warnings"][0]
    assert "".join(deltas) == done["response"]


def test_error_before_first_token_is_a_normal_http_error() -> None:
    llm = FakeLLM(outcomes=[LLMRateLimitedError(retry_after_seconds=20)])
    response, events = read_events(client_for(llm), {"message": "hi"})
    assert response.status_code == 429
    assert response.headers["retry-after"] == "20"
    assert response.json()["error"]["code"] == "llm_rate_limited"
    assert events == []


def test_validation_error_is_http_422() -> None:
    response, _ = read_events(client_for(FakeLLM()), {"message": "   "})
    assert response.status_code == 422


def test_message_too_long_is_http_400() -> None:
    response, _ = read_events(client_for(FakeLLM(), max_message_chars=5), {"message": "too long"})
    assert response.status_code == 400


def test_error_after_first_token_is_an_error_event() -> None:
    response, events = read_events(client_for(FailsMidStream()), {"message": "hi"})
    assert response.status_code == 200
    assert [name for name, _ in events] == ["meta", "delta", "error"]
    error = events[-1][1]["error"]
    assert error["code"] == "llm_unavailable"
    assert error["request_id"] == response.headers["x-request-id"]


def test_stream_without_rag_is_ungrounded() -> None:
    _, events = read_events(
        client_for(FakeLLM(outcomes=["ok answer"])), {"message": "What is DAP?"}
    )
    assert events[0][1]["grounded"] is False
    assert events[0][1]["sources"] == []
    assert events[-1][1]["response"] == "ok answer"


def test_blocking_and_streaming_return_same_answer() -> None:
    llm = FakeLLM(outcomes=["Same answer text."])
    client = client_for(llm)
    blocking = client.post("/api/v1/chat", json={"message": "q", "language": "en"}).json()
    _, events = read_events(client, {"message": "q", "language": "en"})
    streamed = events[-1][1]
    for key in ("response", "language", "grounded", "sources", "safety_warnings", "model"):
        assert blocking[key] == streamed[key]
