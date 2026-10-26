from __future__ import annotations

from app.core.config import Settings
from app.services.storage import StorageService


class FakeBlob:
    def __init__(self, name: str, *, bucket: str) -> None:
        self.name = name
        self.bucket = bucket

    def generate_signed_url(self, **kwargs: object) -> str:
        return (
            f"https://storage.googleapis.com/{self.bucket}/{self.name}?"
            f"sig=demo&expires={kwargs.get('expiration', 'now')}"
        )


class FakeBucket:
    def __init__(self, bucket_name: str) -> None:
        self.name = bucket_name

    def exists(self) -> bool:
        return True

    def blob(self, name: str) -> FakeBlob:
        return FakeBlob(name, bucket=self.name)


class FakeClient:
    def __init__(self, bucket_name: str) -> None:
        self._bucket_name = bucket_name

    def bucket(self, bucket_name: str) -> FakeBucket:
        assert bucket_name == self._bucket_name
        return FakeBucket(bucket_name)


def test_storage_builds_private_keys_for_media() -> None:
    settings = Settings(
        _env_file=None,
        gcs_bucket="krishimitra-media-local",
        gcp_project_id="demo-project",
    )
    storage = StorageService(settings, client=FakeClient("krishimitra-media-local"))

    key = storage.build_key(kind="photo", filename="leaf.jpg", conversation_id="conv-123")

    assert key.startswith("cleaned/photo/")
    assert "/conv-123/" in key
    assert key.endswith(".jpg")


def test_storage_generates_signed_urls() -> None:
    settings = Settings(
        _env_file=None,
        gcs_bucket="krishimitra-media-local",
        gcp_project_id="demo-project",
    )
    storage = StorageService(settings, client=FakeClient("krishimitra-media-local"))

    key = storage.build_key(kind="video", filename="clip.mp4", conversation_id="conv-456")
    download_url = storage.generate_download_url(key)

    assert download_url.startswith("https://storage.googleapis.com/krishimitra-media-local/")
    assert "sig=demo" in download_url
    assert not hasattr(storage, "generate_upload_url")  # clients never upload directly


def test_no_client_upload_endpoint() -> None:
    from fastapi.testclient import TestClient

    from app.main import create_app
    from tests.fakes import FakeLLM

    app = create_app(Settings(_env_file=None, gcs_bucket="b"), llm=FakeLLM())  # type: ignore[call-arg]
    response = TestClient(app).post("/api/v1/media/upload-url", json={"kind": "photo"})
    assert response.status_code == 404


def test_storage_requires_bucket_configuration() -> None:
    settings = Settings(_env_file=None)
    storage = StorageService(settings)
    assert storage.is_configured is False
    try:
        _ = storage.bucket_name
    except Exception as exc:  # pragma: no cover - exercised for coverage of the guard.
        assert "Object storage is not configured" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected misconfiguration error")
