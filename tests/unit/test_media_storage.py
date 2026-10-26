"""Media storage (local + GCS backends), the media archive and its chat hookup (no cloud calls)."""

from __future__ import annotations

import hashlib
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient
from google.api_core.exceptions import NotFound

from app.core.config import Settings
from app.main import create_app
from app.models.conversation import MediaRecord
from app.rag.retriever import RetrievalResult
from app.services.media_archive import MediaArchive, MediaItem
from app.services.storage import GCSMediaStore, LocalMediaStore, build_media_store, media_key
from tests.fakes import FakeLLM
from tests.unit.test_chat_rag import FakeRetriever, chunk
from tests.unit.test_dev_console import SECURE_PRODUCTION
from tests.unit.test_image_chat import (
    PHOTO,
    VIDEO_INFO,
    FakeDiagnoser,
    FakeSafety,
    diagnosis,
    prepared_image,
)

NOW = datetime(2026, 10, 26, 12, 0, tzinfo=UTC)


def files_in(folder: Path) -> list[Path]:
    return [p for p in folder.rglob("*") if p.is_file()]


# ---------------------------------------------------------------- keys and local backend


def test_media_key_layout_and_safety() -> None:
    key = media_key(conversation_id="conv-1234", media_id="a" * 32, created=NOW)
    assert key == f"media/2026/10/26/conv-1234/{'a' * 32}.jpg"
    for bad in ("../etc", "a/b", "", "x" * 65):
        with pytest.raises(ValueError, match="Unsafe"):
            media_key(conversation_id=bad, media_id="a" * 32, created=NOW)


async def test_local_store_round_trip(tmp_path: Path) -> None:
    store = LocalMediaStore(tmp_path / "media")
    await store.check()
    key = media_key(conversation_id="conv-1234", media_id="b" * 32, created=NOW)
    await store.put(key, b"jpeg", "image/jpeg")
    path = tmp_path / "media" / key
    assert path.read_bytes() == b"jpeg"
    assert await store.url(key, 60) == path.resolve().as_uri()
    with pytest.raises(FileExistsError):
        await store.put(key, b"other", "image/jpeg")  # never overwrites
    assert not list(path.parent.glob("*.part"))  # no temp files left behind
    await store.delete(key)
    await store.delete(key)  # deleting twice is fine
    assert not path.exists()


async def test_local_store_rejects_escaping_keys(tmp_path: Path) -> None:
    store = LocalMediaStore(tmp_path)
    with pytest.raises(ValueError, match="escapes"):
        await store.put("../outside.jpg", b"x", "image/jpeg")


# ---------------------------------------------------------------- GCS backend (fake client)


class FakeBlob:
    def __init__(self, bucket: FakeBucket, name: str) -> None:
        self.bucket, self.name = bucket, name

    def upload_from_string(self, data: bytes, **kwargs: Any) -> None:
        self.bucket.uploads.append((self.name, data, kwargs))

    def delete(self) -> None:
        raise NotFound("gone")

    def generate_signed_url(self, **kwargs: Any) -> str:
        self.bucket.signed.append(kwargs)
        return f"https://storage.googleapis.com/{self.bucket.name}/{self.name}?X-Goog-Signature=x"


class FakeBucket:
    def __init__(self, name: str) -> None:
        self.name = name
        self.uploads: list[tuple[str, bytes, dict[str, Any]]] = []
        self.signed: list[dict[str, Any]] = []

    def blob(self, name: str) -> FakeBlob:
        return FakeBlob(self, name)

    def exists(self) -> bool:
        return self.name == "krishimitra-media"


class FakeGCSClient:
    def __init__(self, bucket: str, credentials: Any = None) -> None:
        self._bucket = FakeBucket(bucket)
        self._credentials = credentials

    def bucket(self, name: str) -> FakeBucket:
        return self._bucket


async def test_gcs_store_writes_privately_and_signs_short_links() -> None:
    client = FakeGCSClient("krishimitra-media", credentials=SimpleNamespace(sign_bytes=print))
    store = GCSMediaStore("krishimitra-media", client=client)
    await store.check()
    await store.put("media/k.jpg", b"jpeg", "image/jpeg")
    name, data, kwargs = client._bucket.uploads[0]
    assert (name, data) == ("media/k.jpg", b"jpeg")
    assert kwargs == {"content_type": "image/jpeg", "if_generation_match": 0}  # no overwrite
    await store.delete("media/k.jpg")  # NotFound is ignored
    url = await store.url("media/k.jpg", 900)
    assert url.startswith("https://storage.googleapis.com/krishimitra-media/media/k.jpg?")
    signed = client._bucket.signed[0]
    assert signed["version"] == "v4" and signed["method"] == "GET"
    assert signed["expiration"].total_seconds() == 900
    assert "access_token" not in signed  # key-file credentials sign locally


async def test_gcs_signing_on_cloud_run_uses_iam() -> None:
    # Cloud Run credentials cannot sign locally: email + token are passed for IAM signing.
    credentials = SimpleNamespace(valid=True, service_account_email="api@p.iam", token="tok")
    client = FakeGCSClient("krishimitra-media", credentials=credentials)
    await GCSMediaStore("krishimitra-media", client=client).url("media/k.jpg", 60)
    signed = client._bucket.signed[0]
    assert signed["service_account_email"] == "api@p.iam" and signed["access_token"] == "tok"


async def test_gcs_check_fails_for_missing_bucket() -> None:
    store = GCSMediaStore("missing", client=FakeGCSClient("missing"))
    with pytest.raises(RuntimeError, match="missing or not accessible"):
        await store.check()


# ---------------------------------------------------------------- settings


def test_build_media_store_and_settings_rules(tmp_path: Path) -> None:
    assert build_media_store(Settings(_env_file=None)) is None  # default: nothing stored
    local = Settings(_env_file=None, media_storage="local", media_local_dir=tmp_path)
    assert isinstance(build_media_store(local), LocalMediaStore)
    gcs = Settings(_env_file=None, media_storage="gcs", gcs_bucket="b")
    assert isinstance(build_media_store(gcs), GCSMediaStore)
    with pytest.raises(ValueError, match="GCS_BUCKET is required when MEDIA_STORAGE=gcs"):
        Settings(_env_file=None, media_storage="gcs")
    with pytest.raises(ValueError, match="MEDIA_STORAGE=local is for development only"):
        Settings(_env_file=None, **SECURE_PRODUCTION, media_storage="local")


# ---------------------------------------------------------------- archive


class FakeSession:
    def __init__(self, sink: list[Any], fail: bool) -> None:
        self.sink, self.fail = sink, fail

    def add_all(self, items: list[Any]) -> None:
        if self.fail:
            raise ConnectionError("database down")
        self.sink.extend(items)


class FakeDatabase:
    def __init__(self, *, fail: bool = False) -> None:
        self.rows: list[Any] = []
        self.fail = fail

    @asynccontextmanager
    async def session(self) -> AsyncIterator[FakeSession]:
        yield FakeSession(self.rows, self.fail)


def archive_with(tmp_path: Path, *, fail: bool = False) -> tuple[MediaArchive, FakeDatabase]:
    db = FakeDatabase(fail=fail)
    return MediaArchive(LocalMediaStore(tmp_path), db), db  # type: ignore[arg-type]


async def test_archive_stores_files_and_records(tmp_path: Path) -> None:
    archive, db = archive_with(tmp_path)
    frames = [
        MediaItem("video_frame", prepared_image(b"frame-1"), 1.0),
        MediaItem("video_frame", prepared_image(b"frame-2"), 4.0),
    ]
    ids = await archive.archive(
        conversation_id="conv-1234",
        message_id="m" * 32,
        items=frames,
        diagnosis=diagnosis(video=VIDEO_INFO),
    )
    assert len(ids) == 2
    first: MediaRecord = db.rows[0]
    assert first.kind == "video_frame" and first.storage_backend == "local"
    assert first.message_id == "m" * 32 and first.conversation_id == "conv-1234"
    assert first.sha256 == hashlib.sha256(b"frame-1").hexdigest() and first.size_bytes == 7
    assert (first.width, first.height, first.duration_seconds) == (800, 600, 10.0)
    assert first.meta["frame_time"] == 1.0
    assert first.meta["diagnosis"]["condition"] == "tomato_early_blight"
    assert (tmp_path / first.storage_key).read_bytes() == b"frame-1"


async def test_archive_failure_removes_written_files(tmp_path: Path) -> None:
    archive, _ = archive_with(tmp_path, fail=True)
    ids = await archive.archive(
        conversation_id="conv-1234",
        message_id="m" * 32,
        items=[MediaItem("photo", prepared_image())],
        diagnosis=diagnosis(),
    )
    assert ids == []
    assert files_in(tmp_path) == []  # no file without a record


@pytest.mark.parametrize("status", ["retake_photo", "not_a_plant"])
async def test_unusable_media_is_never_kept(tmp_path: Path, status: str) -> None:
    archive, db = archive_with(tmp_path)
    archive.archive_later(
        conversation_id="conv-1234",
        message_id="m" * 32,
        items=[MediaItem("photo", prepared_image())],
        diagnosis=diagnosis(status, "unknown"),
    )
    await archive.drain()
    assert db.rows == [] and files_in(tmp_path) == []


# ---------------------------------------------------------------- chat hookup


class FakeArchive:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def archive_later(self, **kwargs: Any) -> None:
        self.calls.append(kwargs)


class FakeStore:
    async def save_turn(self, **kwargs: Any) -> str:
        return "a" * 32

    async def history(self, conversation_id: str) -> tuple[()]:
        return ()


def chat_client(result: Any, *, store: bool = True) -> tuple[TestClient, FakeArchive]:
    app = create_app(
        Settings(_env_file=None, enable_dev_ui=True),
        llm=FakeLLM(outcomes=["Early blight [1]."]),
        diagnoser=FakeDiagnoser(result),  # type: ignore[arg-type]
    )
    app.state.retriever = FakeRetriever(RetrievalResult(chunks=[chunk("c1", "Spray")]))
    app.state.safety = FakeSafety()
    app.state.conversations = FakeStore() if store else None
    archive = FakeArchive()
    app.state.media_archive = archive
    return TestClient(app), archive


def test_photo_answer_archives_the_cleaned_photo() -> None:
    client, archive = chat_client(diagnosis())
    body = client.post(
        "/api/v1/chat/image", files={"file": ("leaf.jpg", PHOTO, "image/jpeg")}
    ).json()
    call = archive.calls[0]
    assert call["message_id"] == body["message_id"] == "a" * 32
    assert call["conversation_id"] == body["conversation_id"]
    (item,) = call["items"]
    assert item.kind == "photo" and item.image.data == b"cleaned:" + PHOTO  # cleaned, not raw


def test_video_answer_archives_the_diagnosed_frames() -> None:
    client, archive = chat_client(diagnosis(video=VIDEO_INFO))
    client.post(
        "/api/v1/chat/video",
        files={"file": ("v.mp4", b"\x00\x00\x00\x18ftyp", "video/mp4")},
    )
    items = archive.calls[0]["items"]
    assert [(i.kind, i.frame_time) for i in items] == [("video_frame", 1.0), ("video_frame", 4.0)]


def test_nothing_archived_without_a_stored_answer() -> None:
    client, archive = chat_client(diagnosis(), store=False)  # no message_id to link to
    client.post("/api/v1/chat/image", files={"file": ("leaf.jpg", PHOTO, "image/jpeg")})
    assert archive.calls == []


def test_retake_photo_is_not_archived() -> None:
    client, archive = chat_client(diagnosis("retake_photo", "unknown"))
    client.post("/api/v1/chat/image", files={"file": ("leaf.jpg", PHOTO, "image/jpeg")})
    assert archive.calls == []


def test_no_client_upload_endpoint() -> None:
    app = create_app(Settings(_env_file=None), llm=FakeLLM())
    response = TestClient(app).post("/api/v1/media/upload-url", json={"kind": "photo"})
    assert response.status_code == 404
