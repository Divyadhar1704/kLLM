from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field

from app.core.config import Settings
from app.core.errors import NotFoundError, RateLimitedError
from app.main import create_app

ALLOWED_ORIGIN = "http://localhost:8000"

SECURE_DEPLOYED: dict[str, Any] = {
    "app_env": "staging",
    "postgres_password": "a-strong-generated-password",
    "gcp_project_id": "krishimitra-staging",
    "gcs_bucket": "krishimitra-media-staging",
    "log_json": True,
    "auth_required": True,
}


class EchoBody(BaseModel):
    message: str = Field(min_length=1, max_length=10)


def build_app(**overrides: Any) -> FastAPI:
    params: dict[str, Any] = {"cors_allowed_origins": ALLOWED_ORIGIN, "log_json": True}
    params.update(overrides)
    app = create_app(Settings(_env_file=None, **params))  # type: ignore[arg-type]

    @app.get("/test/ok")
    async def ok() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/test/boom")
    async def boom() -> None:
        raise RuntimeError("database password is hunter2")

    @app.get("/test/missing")
    async def missing() -> None:
        raise NotFoundError("Conversation not found.")

    @app.get("/test/limited")
    async def limited() -> None:
        raise RateLimitedError(headers={"Retry-After": "30"})

    @app.post("/test/echo")
    async def echo(body: EchoBody) -> EchoBody:
        return body

    return app


@pytest.fixture
def client() -> TestClient:
    return TestClient(build_app())


def assert_error(response: Any, status: int, code: str) -> dict[str, Any]:
    assert response.status_code == status
    body: dict[str, Any] = response.json()
    assert set(body) == {"error"}
    assert body["error"]["code"] == code
    assert body["error"]["request_id"] == response.headers["X-Request-ID"]
    return body["error"]


def test_success_has_request_id_and_timing(client: TestClient) -> None:
    response = client.get("/test/ok")
    assert response.status_code == 200
    assert len(response.headers["X-Request-ID"]) == 32
    assert float(response.headers["X-Response-Time-Ms"]) >= 0


def test_valid_incoming_request_id_is_reused(client: TestClient) -> None:
    response = client.get("/test/ok", headers={"X-Request-ID": "krishindra-abc-123"})
    assert response.headers["X-Request-ID"] == "krishindra-abc-123"


def test_unsafe_incoming_request_id_is_replaced(client: TestClient) -> None:
    response = client.get("/test/ok", headers={"X-Request-ID": "bad id <script>"})
    assert response.headers["X-Request-ID"] != "bad id <script>"
    assert len(response.headers["X-Request-ID"]) == 32


def test_unknown_route_returns_error_format(client: TestClient) -> None:
    error = assert_error(client.get("/does-not-exist"), 404, "not_found")
    assert error["message"] == "Not Found"


def test_method_not_allowed(client: TestClient) -> None:
    assert_error(client.delete("/test/ok"), 405, "method_not_allowed")


def test_app_error(client: TestClient) -> None:
    error = assert_error(client.get("/test/missing"), 404, "not_found")
    assert error["message"] == "Conversation not found."


def test_app_error_headers_preserved(client: TestClient) -> None:
    response = client.get("/test/limited")
    assert_error(response, 429, "rate_limited")
    assert response.headers["Retry-After"] == "30"


def test_validation_error_does_not_echo_input(client: TestClient) -> None:
    secret_input = "sk-very-secret-input-value"
    response = client.post("/test/echo", json={"message": secret_input})
    error = assert_error(response, 422, "validation_error")
    assert error["details"][0]["loc"] == ["body", "message"]
    assert secret_input not in response.text


def test_unhandled_exception_returns_generic_500() -> None:
    client = TestClient(build_app(), raise_server_exceptions=False)
    response = client.get("/test/boom", headers={"Origin": ALLOWED_ORIGIN})
    error = assert_error(response, 500, "internal_error")
    assert "hunter2" not in response.text
    assert "Traceback" not in response.text
    assert error["message"].startswith("An unexpected error")
    # 500s still pass through CORS and security headers.
    assert response.headers["access-control-allow-origin"] == ALLOWED_ORIGIN
    assert response.headers["X-Content-Type-Options"] == "nosniff"


def test_security_headers_on_json(client: TestClient) -> None:
    response = client.get("/test/ok")
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"
    assert response.headers["Content-Security-Policy"].startswith("default-src 'none'")
    assert response.headers["Cache-Control"] == "no-store"
    assert "Strict-Transport-Security" not in response.headers


def test_hsts_when_deployed() -> None:
    client = TestClient(build_app(**SECURE_DEPLOYED))
    assert "max-age=" in client.get("/test/ok").headers["Strict-Transport-Security"]


def test_docs_page_not_given_api_csp(client: TestClient) -> None:
    response = client.get("/docs")
    assert response.status_code == 200
    assert "Content-Security-Policy" not in response.headers


def test_cors_preflight_allowed_origin(client: TestClient) -> None:
    response = client.options(
        "/test/echo",
        headers={"Origin": ALLOWED_ORIGIN, "Access-Control-Request-Method": "POST"},
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == ALLOWED_ORIGIN


def test_cors_rejects_unknown_origin(client: TestClient) -> None:
    response = client.get("/test/ok", headers={"Origin": "https://evil.example.com"})
    assert "access-control-allow-origin" not in response.headers


def test_docs_disabled() -> None:
    client = TestClient(build_app(enable_docs=False))
    assert client.get("/docs").status_code == 404
    assert client.get("/openapi.json").status_code == 404


def test_access_log(capsys: pytest.CaptureFixture[str]) -> None:
    client = TestClient(build_app())
    capsys.readouterr()
    response = client.get("/test/ok?phone=9999999999")
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.strip()]
    access = [r for r in records if r.get("event") == "request_completed"]
    assert len(access) == 1
    record = access[0]
    assert record["request_id"] == response.headers["X-Request-ID"]
    assert record["path"] == "/test/ok"
    assert record["status_code"] == 200
    assert record["method"] == "GET"
    assert "duration_ms" in record
    assert "9999999999" not in json.dumps(record)


def test_unhandled_exception_logged_once_with_traceback(
    capsys: pytest.CaptureFixture[str],
) -> None:
    client = TestClient(build_app(), raise_server_exceptions=False)
    capsys.readouterr()
    response = client.get("/test/boom")
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.strip()]
    errors = [r for r in records if r.get("event") == "unhandled_exception"]
    assert len(errors) == 1
    assert errors[0]["request_id"] == response.headers["X-Request-ID"]
    assert "RuntimeError" in json.dumps(errors[0]["exception"])
    access = [r for r in records if r.get("event") == "request_completed"]
    assert access[0]["status_code"] == 500
    assert access[0]["severity"] == "ERROR"


def test_lifespan_builds_and_closes_llm(capsys: pytest.CaptureFixture[str]) -> None:
    # A fake key: building the client makes no network call.
    app = build_app(google_genai_use_vertexai=False, google_api_key="test-key")
    assert app.state.llm is None
    with TestClient(app):
        assert app.state.llm is not None
        assert app.state.llm.model == "gemini-3.5-flash-lite"
    assert app.state.llm is None
    out = capsys.readouterr().out
    assert "application_startup" in out
    assert "application_shutdown" in out


def test_lifespan_fails_fast_without_credentials() -> None:
    app = build_app(google_genai_use_vertexai=False)
    with pytest.raises(ValueError, match="GOOGLE_API_KEY"), TestClient(app):
        pass


def test_injected_llm_not_closed_by_app() -> None:
    from tests.fakes import FakeLLM

    fake = FakeLLM()
    app = create_app(Settings(_env_file=None), llm=fake)  # type: ignore[call-arg]
    with TestClient(app):
        assert app.state.llm is fake
    assert fake.closed is False
