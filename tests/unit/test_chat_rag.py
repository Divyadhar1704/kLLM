"""Chat endpoint with retrieval and the safety check (fakes; no database or AI calls)."""

from __future__ import annotations

from datetime import date
from typing import Any

from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app
from app.rag.retriever import RetrievalResult, RetrievedChunk
from app.repositories.rag import ChunkHit
from app.services.safety import PesticideSafetyChecker, Restriction
from tests.fakes import FakeLLM

URL = "/api/v1/chat"


def chunk(chunk_id: str, body: str) -> RetrievedChunk:
    return RetrievedChunk(
        hit=ChunkHit(
            chunk_id=chunk_id,
            document_id="d1",
            title="AESA based IPM - Tomato",
            content=f"AESA based IPM - Tomato > Diseases\n{body}",
            section="Diseases",
            page_start=31,
            page_end=31,
            crops=["tomato"],
            doc_type="ipm_package",
            source_org="Govt. of India",
            source_path="raw/ipm_packages/Tomato.pdf",
            as_of=date(2026, 3, 31),
            similarity=0.74,
        ),
        score=0.03,
        vector_similarity=0.74,
        vector_rank=0,
        keyword_rank=None,
    )


class FakeRetriever:
    def __init__(self, result: RetrievalResult) -> None:
        self.result = result
        self.queries: list[str] = []

    async def retrieve(self, query: str, *, hybrid: bool | None = None) -> RetrievalResult:
        self.queries.append(query)
        return self.result


class FakeSafety:
    def __init__(self) -> None:
        self._checker = PesticideSafetyChecker([Restriction("Phorate", (), "banned", None)])

    async def ensure_loaded(self) -> PesticideSafetyChecker:
        return self._checker


def build(
    llm: FakeLLM, retrieval: RetrievalResult | None, **overrides: Any
) -> tuple[TestClient, FakeRetriever | None]:
    params: dict[str, Any] = {"enable_dev_ui": True}
    params.update(overrides)
    app = create_app(Settings(_env_file=None, **params), llm=llm)  # type: ignore[arg-type]
    retriever = FakeRetriever(retrieval) if retrieval is not None else None
    app.state.retriever = retriever
    app.state.safety = FakeSafety()
    return TestClient(app), retriever


def test_grounded_answer_with_sources() -> None:
    llm = FakeLLM(outcomes=["Spray mancozeb 75% WP @ 600-800 g [1]."])
    retrieval = RetrievalResult(
        chunks=[chunk("c1", "Spray mancozeb 75% WP @ 600-800 g in 300 l water/acre")],
        crops=["tomato"],
    )
    client, retriever = build(llm, retrieval)
    body = client.post(URL, json={"message": "టమాటాలో ఎర్లీ బ్లైట్ నివారణ?"}).json()

    assert body["grounded"] is True
    assert body["language"] == "te"
    assert body["sources"] == [
        {
            "index": 1,
            "title": "AESA based IPM - Tomato",
            "page_start": 31,
            "page_end": 31,
            "doc_type": "ipm_package",
            "source_org": "Govt. of India",
            "as_of": "2026-03-31",
        }
    ]
    assert body["safety_warnings"] == []
    assert body["debug"] is None  # not requested
    sent = llm.requests[0]
    assert sent.messages[0].text.startswith("SOURCES:\n\n[1] AESA based IPM - Tomato, p. 31")
    assert "600-800 g in 300 l water/acre" in sent.messages[0].text
    assert "Telugu" in (sent.system_instruction or "")
    assert retriever is not None and retriever.queries == ["టమాటాలో ఎర్లీ బ్లైట్ నివారణ?"]


def test_ungrounded_when_retrieval_fails() -> None:
    llm = FakeLLM(outcomes=["General advice."])
    retrieval = RetrievalResult(error="llm_quota_exhausted", crops=["tomato"])
    client, _ = build(llm, retrieval)
    body = client.post(URL, json={"message": "blight in tomato", "debug": True}).json()
    assert body["grounded"] is False
    assert body["sources"] == []
    sent = llm.requests[0].messages[0].text
    assert "SOURCES" not in sent  # no sources block
    assert sent.endswith("blight in tomato")
    assert "crop: tomato" in sent
    assert body["debug"]["retrieval_error"] == "llm_quota_exhausted"


def test_works_without_retriever() -> None:
    client, _ = build(FakeLLM(outcomes=["ok"]), None)
    body = client.post(URL, json={"message": "What is DAP?"}).json()
    assert body["grounded"] is False


def test_rag_disabled_ignores_retriever() -> None:
    retrieval = RetrievalResult(chunks=[chunk("c1", "text")])
    client, retriever = build(FakeLLM(outcomes=["ok"]), retrieval, rag_enabled=False)
    body = client.post(URL, json={"message": "blight"}).json()
    assert body["grounded"] is False
    assert retriever is not None and retriever.queries == []


def test_banned_pesticide_warning_appended_in_language() -> None:
    llm = FakeLLM(outcomes=["Apply phorate granules."])
    client, _ = build(llm, RetrievalResult())
    body = client.post(URL, json={"message": "पौधों में कीड़े", "language": "hi"}).json()
    assert len(body["safety_warnings"]) == 1
    assert "Phorate" in body["safety_warnings"][0]
    assert "प्रतिबंधित" in body["response"]
    assert body["response"].startswith("Apply phorate granules.")


def test_debug_details_in_development() -> None:
    retrieval = RetrievalResult(
        chunks=[chunk("c1", "text one")],
        crops=["tomato"],
        keyword_query="mancozeb",
        embed_ms=500,
        vector_ms=120,
        keyword_ms=110,
        total_ms=640,
    )
    client, _ = build(FakeLLM(outcomes=["ok"]), retrieval)
    debug = client.post(URL, json={"message": "mancozeb tomato", "debug": True}).json()["debug"]
    assert debug["crops_detected"] == ["tomato"]
    assert debug["keyword_query"] == "mancozeb"
    assert debug["timings_ms"]["embed"] == 500
    assert debug["timings_ms"]["llm"] >= 0
    assert debug["chunks"][0]["chunk_id"] == "c1"
    assert debug["chunks"][0]["vector_similarity"] == 0.74


def test_debug_refused_when_dev_ui_disabled() -> None:
    client, _ = build(FakeLLM(outcomes=["ok"]), RetrievalResult(), enable_dev_ui=False)
    body = client.post(URL, json={"message": "hi", "debug": True}).json()
    assert body["debug"] is None
