"""FastAPI application factory.

Run locally:
    uvicorn app.main:app --reload --port 8010
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Awaitable
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import start_http_server

from app.api.routes import dev, health
from app.api.routes import metrics as metrics_routes
from app.api.v1.router import api_router
from app.core import metrics
from app.core.config import Settings, get_settings
from app.core.errors import REQUEST_ID_HEADER, register_exception_handlers
from app.core.logging import get_logger, setup_logging
from app.core.middleware import (
    RESPONSE_TIME_HEADER,
    BodySizeLimitMiddleware,
    RequestContextMiddleware,
    SecurityHeadersMiddleware,
    UnhandledErrorMiddleware,
)
from app.db import Database
from app.rag.retriever import Retriever
from app.schemas.errors import ErrorResponse
from app.services.answer_cache import AnswerCache
from app.services.auth import ApiKeyStore, TokenVerifier
from app.services.conversations import ConversationStore
from app.services.embeddings import build_embedder
from app.services.health import HealthRegistry
from app.services.llm import LLMProvider, build_llm
from app.services.media_archive import MediaArchive
from app.services.rate_limit import build_rate_limiter
from app.services.safety import PesticideSafety
from app.services.storage import build_media_store
from app.vision.diagnosis import VisionDiagnoser

logger = get_logger(__name__)

_WARMUP_TIMEOUT_SECONDS = 15.0


async def _warmup(app: FastAPI, settings: Settings) -> None:
    """Open DB connections and the AI clients' HTTPS connections before the first request.

    Costs no quota (model metadata lookups only). Failures are logged, never fatal: readiness
    checks report real problems.
    """
    start = time.perf_counter()
    tasks: dict[str, Awaitable[None]] = {}
    if app.state.db is not None and settings.db_warmup_connections:
        tasks["database"] = app.state.db.warmup(settings.db_warmup_connections)
    if app.state.llm is not None:
        tasks["llm"] = app.state.llm.check()
    if app.state.embedder is not None:
        tasks["embedder"] = app.state.embedder.warmup()
    if app.state.diagnoser is not None:
        tasks["vision"] = app.state.diagnoser.warmup()
    try:
        async with asyncio.timeout(_WARMUP_TIMEOUT_SECONDS):
            results = await asyncio.gather(*tasks.values(), return_exceptions=True)
    except TimeoutError:
        logger.warning("warmup_timeout", seconds=_WARMUP_TIMEOUT_SECONDS)
        return
    failed = {
        name: type(result).__name__
        for name, result in zip(tasks, results, strict=True)
        if isinstance(result, BaseException)
    }
    logger.info(
        "warmup_done",
        ms=round((time.perf_counter() - start) * 1000, 1),
        warmed=[name for name in tasks if name not in failed],
        failed=failed or None,
    )


def create_app(
    settings: Settings | None = None,
    *,
    llm: LLMProvider | None = None,
    db: Database | None = None,
    diagnoser: VisionDiagnoser | None = None,
) -> FastAPI:
    """Build the app. Pass `llm` / `db` to inject them (tests); otherwise built at startup."""
    settings = settings or get_settings()
    setup_logging(settings)

    async def postgres_check() -> None:
        database: Database | None = app.state.db
        if database is None:
            raise RuntimeError("Database not initialised")
        await database.check()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        owns_llm = app.state.llm is None
        if owns_llm:
            # Fails fast on bad configuration (missing key / project) before serving traffic.
            app.state.llm = build_llm(settings)
        owns_db = app.state.db is None and settings.database_enabled
        if owns_db:
            # The engine connects lazily: startup does not fail if the database is down;
            # /health/ready reports it instead.
            app.state.db = Database.from_settings(settings)
            app.state.health_registry.register("postgres", postgres_check, critical=True)
        owns_embedder = False
        if settings.rag_enabled and app.state.db is not None and app.state.retriever is None:
            embedder = build_embedder(settings)  # in-memory LRU cache of query embeddings
            owns_embedder = True
            app.state.embedder = embedder
            app.state.retriever = Retriever(db=app.state.db, embedder=embedder, settings=settings)
        if app.state.safety is None and app.state.db is not None:
            app.state.safety = PesticideSafety(app.state.db)
            await app.state.safety.ensure_loaded()  # failure is logged and retried later
        if app.state.conversations is None and app.state.db is not None:
            app.state.conversations = ConversationStore(app.state.db)
        if app.state.api_keys is None and app.state.db is not None:
            app.state.api_keys = ApiKeyStore(
                app.state.db, cache_seconds=settings.api_key_cache_seconds
            )
        owns_archive = False
        if app.state.media_archive is None and app.state.db is not None:
            store = build_media_store(settings)
            if store is not None:
                owns_archive = True
                app.state.media_archive = MediaArchive(
                    store, app.state.db, url_ttl_seconds=settings.media_url_ttl_seconds
                )
                # Non-critical: without storage, answers still work (media is just not kept).
                app.state.health_registry.register("media_storage", store.check, critical=False)
        owns_diagnoser = settings.vision_enabled and app.state.diagnoser is None
        if owns_diagnoser:
            app.state.diagnoser = VisionDiagnoser.from_settings(settings)
        if settings.warmup_on_startup:
            await _warmup(app, settings)
        metrics_server = None
        if settings.metrics_port is not None:
            # Loopback only: reachable by the monitoring sidecar, not from the internet.
            metrics_server, _ = start_http_server(settings.metrics_port, addr="127.0.0.1")
        logger.info(
            "application_startup",
            app_env=settings.app_env,
            version=settings.app_version,
            llm_provider=settings.llm_provider,
            llm_model=settings.llm_model,
            llm_fallback_model=settings.llm_fallback_model,
            database=f"{settings.postgres_host}:{settings.postgres_port}/{settings.postgres_db}"
            if settings.database_enabled
            else "disabled",
            rag=app.state.retriever is not None,
            metrics_port=settings.metrics_port,
            media_storage=app.state.media_archive.store.backend
            if app.state.media_archive is not None
            else "none",
        )
        try:
            yield
        finally:
            if metrics_server is not None:
                metrics_server.shutdown()
            if app.state.media_archive is not None:
                await app.state.media_archive.drain()  # finish background saves first
            await app.state.rate_limiter.aclose()
            if owns_archive:
                app.state.media_archive = None
                app.state.health_registry.unregister("media_storage")
            if owns_diagnoser and app.state.diagnoser is not None:
                await app.state.diagnoser.aclose()
                app.state.diagnoser = None
            if owns_embedder:
                await app.state.embedder.aclose()
                app.state.embedder = None
                app.state.retriever = None
                app.state.safety = None
            if owns_llm and app.state.llm is not None:
                await app.state.llm.aclose()
                app.state.llm = None
            if owns_db and app.state.db is not None:
                await app.state.db.aclose()
                app.state.db = None
                app.state.conversations = None
                app.state.api_keys = None
                app.state.health_registry.unregister("postgres")
            logger.info("application_shutdown")

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="KrishiMitra agricultural AI assistant API.",
        docs_url="/docs" if settings.enable_docs else None,
        redoc_url=None,
        openapi_url="/openapi.json" if settings.enable_docs else None,
        lifespan=lifespan,
        responses={
            422: {"model": ErrorResponse, "description": "Validation error"},
            500: {"model": ErrorResponse, "description": "Internal error"},
        },
    )
    app.state.settings = settings
    app.state.started_at = time.monotonic()
    metrics.build_info.labels(settings.app_version, settings.app_env).set(1)
    # Dependencies register readiness checks here as they are added (DB, Redis, LLM, storage).
    app.state.health_registry = HealthRegistry(
        timeout_seconds=settings.health_check_timeout_seconds,
        cache_seconds=settings.health_cache_seconds,
    )
    app.state.llm = llm

    async def llm_check() -> None:
        provider: LLMProvider | None = app.state.llm
        if provider is None:
            raise RuntimeError("LLM provider not initialised")
        await provider.check()

    # Non-critical: an LLM outage degrades chat but the instance stays in rotation.
    app.state.health_registry.register("llm", llm_check, critical=False)

    app.state.db = db
    app.state.embedder = None
    app.state.retriever = None
    app.state.safety = None
    app.state.diagnoser = diagnoser
    app.state.conversations = None  # ConversationStore, set at startup when a database exists
    app.state.media_archive = None  # MediaArchive, set at startup when MEDIA_STORAGE is set
    app.state.api_keys = None  # ApiKeyStore, set at startup when a database exists
    app.state.token_verifier = TokenVerifier.from_settings(settings)
    app.state.rate_limiter = build_rate_limiter(settings)
    if settings.rate_limit_backend == "redis":
        # Non-critical: rate limiting fails open when Redis is down.
        app.state.health_registry.register(
            "rate_limit_redis", app.state.rate_limiter.check, critical=False
        )
    app.state.answer_cache = AnswerCache(
        max_items=settings.answer_cache_size, ttl_seconds=settings.answer_cache_ttl_seconds
    )
    if db is not None:
        app.state.health_registry.register("postgres", postgres_check, critical=True)

    register_exception_handlers(app)

    # add_middleware wraps: the last one added is the outermost.
    app.add_middleware(UnhandledErrorMiddleware)
    app.add_middleware(
        BodySizeLimitMiddleware,
        limits={
            # multipart overhead on top of the photo itself
            f"{settings.api_v1_prefix}/image": settings.max_image_bytes + 512 * 1024,
            f"{settings.api_v1_prefix}/chat/image": settings.max_image_bytes + 512 * 1024,
            f"{settings.api_v1_prefix}/video": settings.max_video_bytes + 512 * 1024,
            f"{settings.api_v1_prefix}/chat/video": settings.max_video_bytes + 512 * 1024,
            f"{settings.api_v1_prefix}/chat": 256 * 1024,
        },
        default_limit=1024 * 1024,
    )
    app.add_middleware(SecurityHeadersMiddleware, enable_hsts=settings.is_deployed)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allowed_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-API-Key", REQUEST_ID_HEADER],
        expose_headers=[REQUEST_ID_HEADER, RESPONSE_TIME_HEADER],
        max_age=600,
    )
    app.add_middleware(RequestContextMiddleware, gcp_project_id=settings.gcp_project_id)

    app.include_router(health.router)
    app.include_router(api_router, prefix=settings.api_v1_prefix)
    if not settings.is_deployed:
        app.include_router(metrics_routes.router)  # deployed: METRICS_PORT on loopback only
    if settings.enable_dev_ui and not settings.is_production:
        app.include_router(dev.router)

    return app


app = create_app()
