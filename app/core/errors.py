"""Application exceptions and handlers producing a single error format:

    {"error": {"code": "...", "message": "...", "request_id": "...", "details": [...]}}

Internal details (exception messages, stack traces, user input) are never returned to clients.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.logging import get_logger
from app.schemas.errors import ErrorBody, ErrorResponse

logger = get_logger(__name__)

REQUEST_ID_HEADER = "X-Request-ID"

_STATUS_CODES: dict[int, str] = {
    400: "bad_request",
    401: "unauthorized",
    403: "forbidden",
    404: "not_found",
    405: "method_not_allowed",
    406: "not_acceptable",
    409: "conflict",
    413: "payload_too_large",
    415: "unsupported_media_type",
    422: "validation_error",
    429: "rate_limited",
    500: "internal_error",
    502: "upstream_error",
    503: "service_unavailable",
    504: "upstream_timeout",
}

INTERNAL_ERROR_MESSAGE = "An unexpected error occurred. Please try again later."


class AppError(Exception):
    """Base class for errors that are safe to report to clients."""

    status_code: ClassVar[int] = 500
    code: ClassVar[str] = "internal_error"
    default_message: ClassVar[str] = INTERNAL_ERROR_MESSAGE

    def __init__(
        self,
        message: str | None = None,
        *,
        details: list[dict[str, Any]] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        self.message = message or self.default_message
        self.details = details
        self.headers = dict(headers) if headers else None
        super().__init__(self.message)


class BadRequestError(AppError):
    status_code = 400
    code = "bad_request"
    default_message = "The request is invalid."


class UnauthorizedError(AppError):
    status_code = 401
    code = "unauthorized"
    default_message = "Authentication is required."


class ForbiddenError(AppError):
    status_code = 403
    code = "forbidden"
    default_message = "You do not have access to this resource."


class NotFoundError(AppError):
    status_code = 404
    code = "not_found"
    default_message = "The requested resource was not found."


class PayloadTooLargeError(AppError):
    status_code = 413
    code = "payload_too_large"
    default_message = "The uploaded content is too large."


class UnsupportedMediaTypeError(AppError):
    status_code = 415
    code = "unsupported_media_type"
    default_message = "The uploaded file type is not supported."


class RateLimitedError(AppError):
    status_code = 429
    code = "rate_limited"
    default_message = "Too many requests. Please slow down."


class UpstreamError(AppError):
    status_code = 502
    code = "upstream_error"
    default_message = "An AI service failed to respond correctly. Please try again."


class ServiceUnavailableError(AppError):
    status_code = 503
    code = "service_unavailable"
    default_message = "The service is temporarily unavailable. Please try again shortly."


class UpstreamTimeoutError(AppError):
    status_code = 504
    code = "upstream_timeout"
    default_message = "An AI service took too long to respond. Please try again."


def code_for_status(status_code: int) -> str:
    return _STATUS_CODES.get(status_code, "internal_error" if status_code >= 500 else "error")


def error_payload(
    *,
    code: str,
    message: str,
    request_id: str | None,
    details: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    body = ErrorResponse(
        error=ErrorBody(code=code, message=message, request_id=request_id, details=details)
    )
    return body.model_dump(mode="json", exclude_none=True)


def get_request_id(request: Request) -> str | None:
    request_id = getattr(request.state, "request_id", None)
    return request_id if isinstance(request_id, str) else None


def error_response(
    request: Request,
    *,
    status_code: int,
    code: str,
    message: str,
    details: list[dict[str, Any]] | None = None,
    headers: Mapping[str, str] | None = None,
) -> JSONResponse:
    request_id = get_request_id(request)
    response_headers = dict(headers or {})
    if request_id:
        response_headers[REQUEST_ID_HEADER] = request_id
    return JSONResponse(
        status_code=status_code,
        content=error_payload(code=code, message=message, request_id=request_id, details=details),
        headers=response_headers,
    )


async def app_error_handler(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, AppError)  # noqa: S101 - registered only for AppError
    log = logger.error if exc.status_code >= 500 else logger.info
    log("app_error", code=exc.code, status_code=exc.status_code, error_message=exc.message)
    return error_response(
        request,
        status_code=exc.status_code,
        code=exc.code,
        message=exc.message,
        details=exc.details,
        headers=exc.headers,
    )


async def http_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, StarletteHTTPException)  # noqa: S101 - registered only for HTTPException
    if exc.status_code >= 500 or not isinstance(exc.detail, str):
        message = INTERNAL_ERROR_MESSAGE if exc.status_code >= 500 else "Request failed."
    else:
        message = exc.detail
    return error_response(
        request,
        status_code=exc.status_code,
        code=code_for_status(exc.status_code),
        message=message,
        headers=exc.headers,
    )


async def validation_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, RequestValidationError)  # noqa: S101 - registered only for validation
    # Only location, message and type: user input is never echoed back.
    details = [
        {"loc": list(err.get("loc", ())), "msg": err.get("msg", ""), "type": err.get("type", "")}
        for err in exc.errors()
    ]
    return error_response(
        request,
        status_code=422,
        code="validation_error",
        message="The request contains invalid fields.",
        details=details,
    )


def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(AppError, app_error_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
