"""Pure ASGI middlewares (streaming-safe, no BaseHTTPMiddleware).

Order, outermost first (see app.main.create_app):
    RequestContextMiddleware  request id, timing, access log
    CORSMiddleware
    SecurityHeadersMiddleware
    UnhandledErrorMiddleware  converts unexpected exceptions into a clean 500 JSON response
"""

from __future__ import annotations

import re
import time
import uuid

import structlog
from fastapi.responses import JSONResponse
from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.core import metrics
from app.core.errors import INTERNAL_ERROR_MESSAGE, REQUEST_ID_HEADER, error_payload
from app.core.logging import get_logger

logger = get_logger("app.access")
error_logger = get_logger("app.errors")

RESPONSE_TIME_HEADER = "X-Response-Time-Ms"

# Accept a caller-supplied request id only if it is safe to log and echo.
_REQUEST_ID_PATTERN = re.compile(r"[A-Za-z0-9._-]{8,128}")

# Probe endpoints logged at DEBUG to keep logs clean.
QUIET_PATHS = frozenset({"/health", "/health/ready"})


def _incoming_request_id(scope: Scope) -> str:
    for name, value in scope.get("headers", []):
        if name == b"x-request-id":
            candidate: str = value.decode("latin-1")
            if _REQUEST_ID_PATTERN.fullmatch(candidate):
                return candidate
            break
    return uuid.uuid4().hex


def _scope_request_id(scope: Scope) -> str | None:
    state = scope.get("state")
    if isinstance(state, dict):
        request_id = state.get("request_id")
        if isinstance(request_id, str):
            return request_id
    return None


def _trace_context(scope: Scope, project_id: str | None) -> dict[str, str]:
    """Cloud Logging fields grouping all log lines of one request under its Cloud Trace id
    (from Google's load balancer: X-Cloud-Trace-Context "TRACE/SPAN;o=1", or W3C traceparent)."""
    if not project_id:
        return {}
    headers = dict(scope.get("headers", []))
    trace_id = span_id = None
    sampled = False
    if cloud := headers.get(b"x-cloud-trace-context"):
        match = _CLOUD_TRACE.match(cloud.decode("latin-1"))
        if match:
            trace_id, span_id, sampled = match.group(1), match.group(2), match.group(3) == "1"
    elif parent := headers.get(b"traceparent"):
        match = _TRACEPARENT.match(parent.decode("latin-1"))
        if match:
            trace_id, span_id = match.group(1), match.group(2)
            sampled = int(match.group(3), 16) & 1 == 1
    if trace_id is None:
        return {}
    context = {
        "logging.googleapis.com/trace": f"projects/{project_id}/traces/{trace_id}",
        "logging.googleapis.com/trace_sampled": sampled,
    }
    if span_id:
        context["logging.googleapis.com/spanId"] = span_id
    return context  # type: ignore[return-value]


def _route_template(scope: Scope) -> str:
    """ "/api/v1/items/{item_id}" for a matched route, "unmatched" otherwise. Built from the real
    path with the path parameters put back as placeholders: with routers included lazily, the
    route object only knows its path inside its own router (without the /api/v1 prefix)."""
    if scope.get("route") is None:
        return "unmatched"
    path: str = scope.get("path", "")
    for name, value in (scope.get("path_params") or {}).items():
        path = path.replace(f"/{value}", f"/{{{name}}}", 1)
    return path


_CLOUD_TRACE = re.compile(r"^([0-9a-f]{32})(?:/(\d+))?(?:;o=([01]))?", re.IGNORECASE)
_TRACEPARENT = re.compile(r"^00-([0-9a-f]{32})-([0-9a-f]{16})-([0-9a-f]{2})$", re.IGNORECASE)


class RequestContextMiddleware:
    def __init__(self, app: ASGIApp, *, gcp_project_id: str | None = None) -> None:
        self.app = app
        self.gcp_project_id = gcp_project_id

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = _incoming_request_id(scope)
        scope.setdefault("state", {})["request_id"] = request_id
        start = time.perf_counter()
        status_code: int | None = None

        async def send_wrapper(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
                headers = MutableHeaders(scope=message)
                headers[REQUEST_ID_HEADER] = request_id
                headers[RESPONSE_TIME_HEADER] = f"{(time.perf_counter() - start) * 1000:.1f}"
            await send(message)

        trace = _trace_context(scope, self.gcp_project_id)
        with structlog.contextvars.bound_contextvars(request_id=request_id, **trace):
            failed = False
            try:
                await self.app(scope, receive, send_wrapper)
            except Exception:
                failed = True
                raise
            finally:
                self._log(scope, status_code, start, failed)

    @staticmethod
    def _log(scope: Scope, status_code: int | None, start: float, failed: bool) -> None:
        duration_ms = round((time.perf_counter() - start) * 1000, 1)
        path: str = scope.get("path", "")
        client = scope.get("client")
        status = status_code if status_code is not None else 500

        if failed or status >= 500:
            log = logger.error
        elif status >= 400:
            log = logger.warning
        elif path in QUIET_PATHS:
            log = logger.debug
        else:
            log = logger.info

        # Route template ("/api/v1/chat"), never the raw path: unknown paths would otherwise
        # create a new metric series each.
        route = _route_template(scope)
        metrics.http_requests.labels(scope.get("method", ""), route, str(status)).inc()
        metrics.http_duration.labels(route).observe(duration_ms / 1000)

        # Who called (audit): set by app.api.security once the caller is identified. The
        # farmer id is logged only as a hash.
        principal = (scope.get("state") or {}).get("principal")
        log(
            "request_completed",
            method=scope.get("method"),
            path=path,
            status_code=status,
            duration_ms=duration_ms,
            client_ip=client[0] if client else None,
            caller=principal.client_id if principal is not None else None,
            auth=principal.method if principal is not None else None,
            user=principal.subject_hash if principal is not None and principal.user_id else None,
            failed_mid_response=failed and status_code is not None,
        )


class UnhandledErrorMiddleware:
    """Last line of defense: log the exception once and return a generic 500."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        response_started = False

        async def send_wrapper(message: Message) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception as exc:
            error_logger.error(
                "unhandled_exception",
                path=scope.get("path"),
                method=scope.get("method"),
                exc_info=exc,
            )
            if response_started:
                # Headers already sent (mid-stream): nothing safe to send, let the server abort.
                raise
            request_id = _scope_request_id(scope)
            response = JSONResponse(
                status_code=500,
                content=error_payload(
                    code="internal_error", message=INTERNAL_ERROR_MESSAGE, request_id=request_id
                ),
            )
            await response(scope, receive, send_wrapper)


_SECURITY_HEADERS: dict[str, str] = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Cross-Origin-Opener-Policy": "same-origin",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
}
_API_CSP = "default-src 'none'; frame-ancestors 'none'"
_HSTS = "max-age=63072000; includeSubDomains"


class SecurityHeadersMiddleware:
    def __init__(self, app: ASGIApp, *, enable_hsts: bool = False) -> None:
        self.app = app
        self.enable_hsts = enable_hsts

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_wrapper(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                for name, value in _SECURITY_HEADERS.items():
                    headers.setdefault(name, value)
                if self.enable_hsts:
                    headers.setdefault("Strict-Transport-Security", _HSTS)
                content_type = headers.get("content-type", "")
                # API responses: no content execution, no caching. HTML (docs, dev UI) is untouched.
                if content_type.startswith(("application/json", "text/event-stream")):
                    headers.setdefault("Content-Security-Policy", _API_CSP)
                    headers.setdefault("Cache-Control", "no-store")
            await send(message)

        await self.app(scope, receive, send_wrapper)


class _BodyTooLargeError(Exception):
    pass


class BodySizeLimitMiddleware:
    """Reject oversized request bodies before they are parsed or written to disk.

    Limits are per path prefix (longest match wins). A declared Content-Length over the limit is
    refused immediately; chunked bodies are counted as they stream in.
    """

    def __init__(self, app: ASGIApp, *, limits: dict[str, int], default_limit: int) -> None:
        self.app = app
        self.limits = sorted(limits.items(), key=lambda item: len(item[0]), reverse=True)
        self.default_limit = default_limit

    def _limit_for(self, path: str) -> int:
        return next(
            (limit for prefix, limit in self.limits if path.startswith(prefix)), self.default_limit
        )

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        limit = self._limit_for(scope.get("path", ""))
        for name, value in scope.get("headers", []):
            if name == b"content-length":
                try:
                    declared = int(value)
                except ValueError:
                    declared = 0
                if declared > limit:
                    await self._reject(scope, receive, send, limit)
                    return
                break

        received = 0
        response_started = False

        async def counting_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    raise _BodyTooLargeError
            return message

        async def tracking_send(message: Message) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, counting_receive, tracking_send)
        except _BodyTooLargeError:
            if not response_started:
                await self._reject(scope, receive, send, limit)

    @staticmethod
    async def _reject(scope: Scope, receive: Receive, send: Send, limit: int) -> None:
        response = JSONResponse(
            status_code=413,
            content=error_payload(
                code="payload_too_large",
                message=f"Request body is larger than the {limit // 1024} KB limit.",
                request_id=_scope_request_id(scope),
            ),
        )
        await response(scope, receive, send)
