"""LLM errors. They are AppErrors, so they reach clients in the standard error format.

retryable          retry the same model (transient: overload, rate limit, empty output)
fallback_eligible  switching to the fallback model may help
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, ClassVar

from app.core.errors import AppError


class LLMError(AppError):
    retryable: ClassVar[bool] = False
    fallback_eligible: ClassVar[bool] = False


class LLMRateLimitedError(LLMError):
    """Short-term (per-minute) limit. `retry_after_seconds` comes from the provider when known."""

    status_code = 429
    code = "llm_rate_limited"
    default_message = "The AI service is receiving too many requests. Please try again in a minute."
    retryable = True
    fallback_eligible = True

    def __init__(
        self,
        message: str | None = None,
        *,
        retry_after_seconds: float | None = None,
        details: list[dict[str, Any]] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        self.retry_after_seconds = retry_after_seconds
        if headers is None:
            headers = {"Retry-After": str(math.ceil(retry_after_seconds or 30))}
        super().__init__(message, details=details, headers=headers)


class LLMQuotaExhaustedError(LLMError):
    """Daily (or longer) quota used up: retrying today is pointless."""

    status_code = 429
    code = "llm_quota_exhausted"
    default_message = "The AI service usage limit has been reached. Please try again later."
    retryable = False
    fallback_eligible = True  # another model has its own quota


class LLMUnavailableError(LLMError):
    status_code = 503
    code = "llm_unavailable"
    default_message = "The AI service is temporarily overloaded. Please try again shortly."
    retryable = True
    fallback_eligible = True


class LLMTimeoutError(LLMError):
    status_code = 504
    code = "llm_timeout"
    default_message = "The AI service took too long to respond. Please try again."
    retryable = False  # a retry would double the farmer's wait; use the fallback instead
    fallback_eligible = True


class LLMResponseError(LLMError):
    status_code = 502
    code = "llm_bad_response"
    default_message = "The AI service returned an invalid response. Please try again."
    retryable = True
    fallback_eligible = True


class LLMConfigurationError(LLMError):
    """Auth failure or unknown model: an operator problem, not the user's."""

    status_code = 502
    code = "llm_configuration_error"
    default_message = "The AI service is not available right now."
    retryable = False
    fallback_eligible = True


class LLMRequestError(LLMError):
    status_code = 502
    code = "llm_request_rejected"
    default_message = "The AI service could not process this request."
    retryable = False
    fallback_eligible = False


class LLMBlockedError(LLMError):
    status_code = 422
    code = "content_blocked"
    default_message = "This question could not be answered. Please rephrase it."
    retryable = False
    fallback_eligible = False
