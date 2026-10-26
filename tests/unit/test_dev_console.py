from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app
from tests.fakes import FakeLLM

SECURE_PRODUCTION: dict[str, Any] = {
    "app_env": "production",
    "postgres_password": "a-strong-generated-password",
    "gcp_project_id": "p",
    "gcs_bucket": "b",
    "log_json": True,
}


def client(**overrides: Any) -> TestClient:
    return TestClient(create_app(Settings(_env_file=None, **overrides), llm=FakeLLM()))  # type: ignore[arg-type]


def test_dev_console_served_when_enabled() -> None:
    response = client(enable_dev_ui=True).get("/dev")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "KrishiMitra" in response.text
    assert "/api/v1/chat" in response.text
    csp = response.headers["content-security-policy"]
    assert "connect-src 'self'" in csp and "frame-ancestors 'none'" in csp
    assert response.headers["cache-control"] == "no-store"


def test_dev_console_absent_when_disabled() -> None:
    assert client(enable_dev_ui=False).get("/dev").status_code == 404


def test_dev_console_never_in_production() -> None:
    # Production config validation already forbids ENABLE_DEV_UI; also check the route itself.
    app = create_app(Settings(_env_file=None, **SECURE_PRODUCTION), llm=FakeLLM())  # type: ignore[arg-type]
    assert TestClient(app).get("/dev").status_code == 404


def test_dev_console_not_in_openapi() -> None:
    paths = client(enable_dev_ui=True).get("/openapi.json").json()["paths"]
    assert "/dev" not in paths


def test_dev_console_supports_photo_questions() -> None:
    response = client(enable_dev_ui=True).get("/dev")
    assert 'id="photo"' in response.text and 'id="crop"' in response.text
    assert '"/video" : "/image"' in response.text  # videos go to /chat/video
    # Previews use blob: URLs of the local file; nothing else is loosened.
    csp = response.headers["content-security-policy"]
    assert "img-src 'self' data: blob:;" in csp and "media-src blob:;" in csp


def test_dev_console_crop_options_match_taxonomy() -> None:
    from app.vision.taxonomy import CROPS

    html = client(enable_dev_ui=True).get("/dev").text
    for crop in CROPS:
        assert f'<option value="{crop}">' in html


def test_page_has_no_external_resources() -> None:
    html = client(enable_dev_ui=True).get("/dev").text
    assert "http://" not in html and "https://" not in html
