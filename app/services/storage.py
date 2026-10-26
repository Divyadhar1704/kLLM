"""Private storage of cleaned media (diagnosed photos and video frames).

Only the server writes, and only media it has already cleaned (re-encoded JPEG, all metadata
such as GPS removed). Clients never get upload URLs: a direct upload would bypass the cleaning.
Objects are read through short-lived signed URLs (GCS) handed to staff tools, never publicly.

Backends:
- `local`: a folder on this machine, for development only (refused when deployed: Cloud Run
  disks are temporary).
- `gcs`: a private Google Cloud Storage bucket. Writes never overwrite an existing object.

The GCS client is synchronous: calls run in worker threads.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import re
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Protocol

from app.core.config import Settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_SAFE_PART = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class MediaStore(Protocol):
    @property
    def backend(self) -> str: ...

    async def put(self, key: str, data: bytes, content_type: str) -> None: ...

    async def delete(self, key: str) -> None: ...

    async def url(self, key: str, ttl_seconds: int) -> str: ...

    async def check(self) -> None: ...


def media_key(*, conversation_id: str, media_id: str, created: datetime) -> str:
    """`media/2026/10/26/<conversation>/<media id>.jpg`: by date for lifecycle rules and audits."""
    for part in (conversation_id, media_id):
        if not _SAFE_PART.match(part):
            raise ValueError(f"Unsafe storage key part: {part!r}")
    return f"media/{created:%Y/%m/%d}/{conversation_id}/{media_id}.jpg"


class LocalMediaStore:
    """Development backend: files under one root folder."""

    def __init__(self, root: Path) -> None:
        self._root = root.resolve()

    @property
    def backend(self) -> str:
        return "local"

    def _path(self, key: str) -> Path:
        path = (self._root / key).resolve()
        if self._root not in path.parents:
            raise ValueError(f"Storage key escapes the media folder: {key!r}")
        return path

    async def put(self, key: str, data: bytes, content_type: str) -> None:
        path = self._path(key)

        def write() -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                raise FileExistsError(key)
            # Write then rename: a crash never leaves a half-written file under the real name.
            handle, temp = tempfile.mkstemp(dir=path.parent, suffix=".part")
            try:
                with os.fdopen(handle, "wb") as file:
                    file.write(data)
                os.replace(temp, path)
            except BaseException:
                Path(temp).unlink(missing_ok=True)
                raise

        await asyncio.to_thread(write)

    async def delete(self, key: str) -> None:
        await asyncio.to_thread(self._path(key).unlink, missing_ok=True)

    async def url(self, key: str, ttl_seconds: int) -> str:
        return self._path(key).as_uri()  # local files: no signing, development only

    async def check(self) -> None:
        def probe() -> None:
            self._root.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryFile(dir=self._root):
                pass

        await asyncio.to_thread(probe)


class GCSMediaStore:
    """Private GCS bucket. Uniform bucket-level access and public access prevention are expected
    on the bucket; nothing here makes an object public."""

    def __init__(self, bucket: str, *, project: str | None = None, client: Any = None) -> None:
        self._bucket_name = bucket
        self._project = project
        self._client = client

    @property
    def backend(self) -> str:
        return "gcs"

    def _bucket(self) -> Any:
        if self._client is None:
            from google.cloud import storage  # type: ignore[import-untyped]  # lazy: gcs only

            self._client = storage.Client(project=self._project)
        return self._client.bucket(self._bucket_name)

    async def put(self, key: str, data: bytes, content_type: str) -> None:
        def upload() -> None:
            blob = self._bucket().blob(key)
            # if_generation_match=0: fail instead of overwriting an existing object.
            blob.upload_from_string(data, content_type=content_type, if_generation_match=0)

        await asyncio.to_thread(upload)

    async def delete(self, key: str) -> None:
        def remove() -> None:
            from google.api_core.exceptions import NotFound

            with contextlib.suppress(NotFound):  # already gone: nothing to do
                self._bucket().blob(key).delete()

        await asyncio.to_thread(remove)

    async def url(self, key: str, ttl_seconds: int) -> str:
        def sign() -> str:
            blob = self._bucket().blob(key)
            return str(
                blob.generate_signed_url(
                    version="v4",
                    expiration=timedelta(seconds=ttl_seconds),
                    method="GET",
                    **self._signing_credentials(),
                )
            )

        return await asyncio.to_thread(sign)

    def _signing_credentials(self) -> dict[str, Any]:
        """On Cloud Run the credentials have no private key: sign through the IAM API with the
        service account's email and an access token instead (the service account needs
        roles/iam.serviceAccountTokenCreator on itself)."""
        credentials = self._client._credentials if self._client is not None else None
        if credentials is None or hasattr(credentials, "sign_bytes"):
            return {}
        from google.auth.transport.requests import Request

        if not credentials.valid:
            credentials.refresh(Request())
        return {
            "service_account_email": credentials.service_account_email,
            "access_token": credentials.token,
        }

    async def check(self) -> None:
        def probe() -> None:
            if not self._bucket().exists():
                raise RuntimeError(f"Bucket {self._bucket_name!r} is missing or not accessible")

        await asyncio.to_thread(probe)


def build_media_store(settings: Settings) -> MediaStore | None:
    if settings.media_storage == "local":
        return LocalMediaStore(settings.media_local_dir)
    if settings.media_storage == "gcs":
        assert settings.gcs_bucket is not None  # noqa: S101 - enforced by settings validation
        return GCSMediaStore(settings.gcs_bucket, project=settings.gcp_project_id)
    return None
