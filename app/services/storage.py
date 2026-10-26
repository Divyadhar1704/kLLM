"""Private object storage for cleaned media (GCS).

Only the server writes to the bucket, and only media it has already cleaned (metadata such as
GPS removed, re-encoded). Clients never get upload URLs: a direct upload would bypass the
cleaning. Reads use short-lived signed download URLs. Server-side writes arrive in Phase 21.
"""

from __future__ import annotations

import re
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import PurePosixPath
from typing import Any, Literal, Protocol, cast

from app.core.config import Settings
from app.core.errors import ServiceUnavailableError
from app.core.logging import get_logger

try:  # pragma: no cover - library is optional until deployed.
    from google.cloud import storage as gcs  # type: ignore[import-untyped]
except ModuleNotFoundError:  # pragma: no cover
    gcs = None

logger = get_logger(__name__)

MediaKind = Literal["photo", "video"]


class BlobLike(Protocol):
    def generate_signed_url(self, *args: Any, **kwargs: Any) -> str: ...


class BucketLike(Protocol):
    def exists(self) -> bool: ...
    def blob(self, name: str) -> BlobLike: ...


class StorageClientLike(Protocol):
    def bucket(self, bucket_name: str) -> BucketLike: ...


class StorageService:
    """Thin, testable GCS wrapper that never exposes object metadata publicly."""

    def __init__(self, settings: Settings, client: StorageClientLike | None = None) -> None:
        self._settings = settings
        self._client = client if client is not None else self._build_client(settings)

    @staticmethod
    def _build_client(settings: Settings) -> StorageClientLike | None:
        if gcs is None or not settings.gcs_bucket:
            return None
        if settings.gcp_project_id:
            return cast(StorageClientLike, gcs.Client(project=settings.gcp_project_id))
        return cast(StorageClientLike, gcs.Client())

    @property
    def is_configured(self) -> bool:
        return bool(self._settings.gcs_bucket)

    @property
    def bucket_name(self) -> str:
        if not self._settings.gcs_bucket:
            raise ServiceUnavailableError("Object storage is not configured.")
        return self._settings.gcs_bucket

    def build_key(
        self,
        *,
        kind: MediaKind,
        filename: str | None = None,
        conversation_id: str | None = None,
        suffix: str | None = None,
    ) -> str:
        now = datetime.now(UTC)
        basename = filename or f"{kind}-{conversation_id or 'upload'}"
        clean_name = re.sub(r"[^A-Za-z0-9._-]+", "-", basename.strip().lower())
        clean_name = clean_name.strip(".-") or f"{kind}-media"
        path = PurePosixPath(clean_name)
        extension = path.suffix.lower() if path.suffix else (suffix or ".bin")
        if not extension.startswith("."):
            extension = f".{extension}"
        if len(extension) > 12:
            extension = ".bin"
        conversation = (conversation_id or "anonymous").strip() or "anonymous"
        object_name = f"cleaned/{kind}/{now:%Y/%m/%d}/{conversation}/{uuid.uuid4().hex}{extension}"
        logger.info(
            "storage_object_key_generated",
            kind=kind,
            conversation_id=conversation,
            key=object_name,
        )
        return object_name

    def generate_download_url(
        self,
        key: str,
        *,
        expires_in_seconds: int | None = None,
    ) -> str:
        if self._client is None:
            raise ServiceUnavailableError("Object storage client is not configured.")
        blob = self._client.bucket(self.bucket_name).blob(key)
        expiration = datetime.now(UTC) + timedelta(
            seconds=expires_in_seconds or self._settings.gcs_signed_url_ttl_seconds
        )
        try:
            return blob.generate_signed_url(
                expiration=expiration,
                method="GET",
                version="v4",
            )
        except TypeError:
            return blob.generate_signed_url(expiration, method="GET")

    async def check(self) -> None:
        if not self.is_configured:
            return
        if self._client is None:
            raise RuntimeError("Object storage client is not initialised")
        bucket = self._client.bucket(self.bucket_name)
        if not bucket.exists():
            raise RuntimeError(f"Bucket {self.bucket_name!r} is missing or inaccessible")

    def object_url(self, key: str) -> str:
        normalized = str(key).strip("/")
        return f"https://storage.googleapis.com/{self.bucket_name}/{normalized}"
