"""Database layer without a database: index planning, SQL generation, engine config, wiring."""

from __future__ import annotations

from datetime import date
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.dialects import postgresql

from app.core.config import Settings
from app.db import Database, create_engine
from app.db.session import connect_args
from app.main import create_app
from app.models import EMBEDDING_DIMENSIONS, Base
from app.rag.indexing import plan_index
from app.rag.ingestion.models import Chunk, DocumentRecord
from app.rag.ingestion.pesticides import PesticideRestriction
from app.repositories.rag import build_search_statement
from tests.fakes import FakeLLM


def settings(**overrides: Any) -> Settings:
    return Settings(_env_file=None, **overrides)  # type: ignore[arg-type]


# ---------------------------------------------------------------- planning


def record(doc_id: str, title: str) -> DocumentRecord:
    return DocumentRecord(
        doc_id=doc_id,
        sha256=doc_id * 4,
        source_path=f"raw/{doc_id}.pdf",
        title=title,
        doc_type="ipm_package",
        as_of="2026-03-31",
        crops=["tomato"],
        pages=10,
    )


def chunk(doc_id: str, index: int) -> Chunk:
    text = f"{doc_id} chunk {index}"
    return Chunk(
        chunk_id=f"{doc_id}-{index}",
        doc_id=doc_id,
        chunk_index=index,
        text=text,
        doc_type="ipm_package",
        title=doc_id,
        source_path=f"raw/{doc_id}.pdf",
        as_of="2026-03-31",
        crops=["tomato"],
        char_count=len(text),
        token_estimate=5,
    )


VECTOR = [0.1] * EMBEDDING_DIMENSIONS


def test_plan_includes_only_embedded_chunks_and_reports_missing() -> None:
    records = [record("docA", "Tomato IPM"), record("docB", "Potato IPM"), record("docC", "Chilli")]
    chunks = [chunk("docA", 0), chunk("docA", 1), chunk("docB", 0), chunk("docC", 0)]
    vectors = {"docA chunk 0": VECTOR, "docA chunk 1": VECTOR, "docB chunk 0": VECTOR}
    restriction = PesticideRestriction(
        name="Phorate", normalized="phorate", status="banned", aliases=[]
    )
    plan = plan_index(
        records,
        chunks,
        vectors,
        key_for=lambda c: c.text,
        embedding_model="gemini-embedding-2",
        restrictions=[restriction],
    )
    assert plan.total_chunks == 4
    assert plan.indexed_chunks == 3
    assert [d["id"] for d in plan.documents] == ["docA", "docB"]  # docC has nothing embedded
    assert plan.documents[0]["chunk_count"] == 2
    assert plan.documents[0]["as_of"] == date(2026, 3, 31)
    assert plan.missing_vectors == {"Chilli": 1}
    row = plan.chunks[0]
    assert row["embedding_model"] == "gemini-embedding-2"
    assert row["content"] == "docA chunk 0"
    assert row["embedding"] is VECTOR
    assert plan.restrictions == [
        {
            "name": "Phorate",
            "normalized": "phorate",
            "aliases": [],
            "status": "banned",
            "details": None,
        }
    ]


def test_plan_rejects_wrong_dimensions() -> None:
    with pytest.raises(ValueError, match="dimensions"):
        plan_index(
            [record("docA", "A")],
            [chunk("docA", 0)],
            {"docA chunk 0": [0.1] * 768},
            key_for=lambda c: c.text,
            embedding_model="m",
        )


def test_plan_rejects_unknown_document() -> None:
    with pytest.raises(ValueError, match="unknown document"):
        plan_index(
            [],
            [chunk("docX", 0)],
            {"docX chunk 0": VECTOR},
            key_for=lambda c: c.text,
            embedding_model="m",
        )


# ---------------------------------------------------------------- SQL


def compile_sql(stmt: Any) -> str:
    return str(stmt.compile(dialect=postgresql.dialect()))


def test_search_statement_uses_cosine_index_and_filters() -> None:
    sql = compile_sql(
        build_search_statement(
            VECTOR,
            embedding_model="gemini-embedding-2",
            top_k=5,
            crops=["tomato"],
            doc_types=["ipm_package"],
        )
    )
    assert "rag_chunks.embedding <=> " in sql  # cosine distance operator (HNSW-indexable)
    assert "ORDER BY rag_chunks.embedding <=> " in sql
    assert "rag_chunks.crops && " in sql
    assert "rag_chunks.doc_type = ANY (" in sql
    assert "rag_chunks.embedding_model = " in sql
    assert "LIMIT " in sql
    assert "JOIN rag_documents" in sql


def test_search_statement_without_filters() -> None:
    sql = compile_sql(build_search_statement(VECTOR, embedding_model="m", top_k=3))
    assert "&&" not in sql
    assert "doc_type = ANY" not in sql


def test_models_define_expected_tables() -> None:
    tables = Base.metadata.tables
    expected = {
        "rag_documents",
        "rag_chunks",
        "pesticide_restrictions",
        "conversations",
        "conversation_messages",
        "media_records",
        "feedback",
    }
    assert set(tables) == expected
    chunks = tables["rag_chunks"]
    assert chunks.c.embedding.type.dim == EMBEDDING_DIMENSIONS  # type: ignore[attr-defined]
    fk = next(iter(chunks.c.document_id.foreign_keys))
    assert fk.ondelete == "CASCADE"


# ---------------------------------------------------------------- engine / settings


def test_connect_args_for_supabase_session_pooler() -> None:
    args = connect_args(
        settings(postgres_ssl="require", postgres_statement_cache_size=0, app_env="test")
    )
    assert args["ssl"] == "require"
    assert args["statement_cache_size"] == 0
    assert args["server_settings"]["application_name"] == "krishimitra-test"


def test_engine_maps_tables_to_schema() -> None:
    engine = create_engine(settings(postgres_schema="kb_test"))
    assert engine.sync_engine.get_execution_options()["schema_translate_map"] == {None: "kb_test"}
    assert engine.pool.size() == 5  # type: ignore[attr-defined]


def test_invalid_schema_name_rejected() -> None:
    with pytest.raises(ValueError):
        settings(postgres_schema="bad-name; drop")


# ---------------------------------------------------------------- app wiring


class FakeDatabase(Database):
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.closed = False

    async def check(self) -> None:
        if self.error:
            raise self.error

    async def aclose(self) -> None:
        self.closed = True


def test_readiness_includes_postgres_when_db_injected() -> None:
    app = create_app(settings(health_cache_seconds=0), llm=FakeLLM(), db=FakeDatabase())
    body = TestClient(app).get("/health/ready").json()
    postgres = next(c for c in body["checks"] if c["name"] == "postgres")
    assert postgres["status"] == "ok" and postgres["critical"] is True


def test_readiness_unavailable_when_db_down() -> None:
    app = create_app(
        settings(health_cache_seconds=0), llm=FakeLLM(), db=FakeDatabase(ConnectionError("down"))
    )
    response = TestClient(app).get("/health/ready")
    assert response.status_code == 503
    assert response.json()["status"] == "unavailable"


def test_lifespan_creates_and_disposes_database_and_rag() -> None:
    app = create_app(
        settings(
            health_cache_seconds=0,
            database_enabled=True,
            # Nothing listens here: the safety-list load fails fast and is retried later.
            postgres_host="127.0.0.1",
            postgres_port=1,
            db_pool_timeout_seconds=1,
            google_genai_use_vertexai=False,
            google_api_key="test-key",
        ),
        llm=FakeLLM(),
    )
    with TestClient(app):
        assert isinstance(app.state.db, Database)
        assert "postgres" in app.state.health_registry.names
        assert app.state.retriever is not None
        assert app.state.safety is not None and app.state.safety.checker is None
    assert app.state.db is None
    assert app.state.retriever is None
    assert "postgres" not in app.state.health_registry.names


def test_database_disabled() -> None:
    app = create_app(settings(database_enabled=False), llm=FakeLLM())
    with TestClient(app):
        assert app.state.db is None
        assert "postgres" not in app.state.health_registry.names
