"""Prometheus metrics: what the service does, how fast, how well, at what AI cost.

Exposed at /metrics locally, and in production only on METRICS_PORT bound to 127.0.0.1 (for the
Cloud Run monitoring sidecar): metrics are never public.

Labels are low-cardinality on purpose (route templates, model names, languages, statuses):
never user ids, conversation ids or raw paths.
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

_NS = "krishimitra"
_SLOW_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1, 2, 3, 5, 8, 13, 20, 30, 60)

# --- HTTP ---
http_requests = Counter(
    f"{_NS}_http_requests_total", "HTTP requests", ["method", "route", "status"]
)
http_duration = Histogram(
    f"{_NS}_http_request_duration_seconds",
    "HTTP request duration (to the end of the response)",
    ["route"],
    buckets=_SLOW_BUCKETS,
)

# --- AI models (purpose: chat | vision) ---
llm_calls = Counter(f"{_NS}_llm_calls_total", "AI model calls", ["purpose", "model", "outcome"])
llm_duration = Histogram(
    f"{_NS}_llm_call_duration_seconds",
    "AI model call duration",
    ["purpose", "model"],
    buckets=_SLOW_BUCKETS,
)
llm_tokens = Counter(f"{_NS}_llm_tokens_total", "AI model tokens", ["purpose", "model", "kind"])
llm_cost = Counter(
    f"{_NS}_llm_cost_usd_total", "Estimated AI cost in USD (configured prices)", ["model"]
)
llm_retries = Counter(f"{_NS}_llm_retries_total", "AI call retries", ["model", "error"])
llm_fallbacks = Counter(
    f"{_NS}_llm_fallbacks_total", "Switches to the fallback model", ["from_model", "error"]
)

# --- Answers ---
answers = Counter(
    f"{_NS}_answers_total",
    "Answers sent (kind: text | photo | video)",
    ["kind", "language", "grounded", "streamed"],
)
answer_cache = Counter(f"{_NS}_answer_cache_total", "Answer cache lookups", ["result"])
language_retries = Counter(
    f"{_NS}_language_retries_total", "Answers regenerated for the wrong language", ["language"]
)
safety_warnings = Counter(
    f"{_NS}_safety_warnings_total", "Answers with a pesticide safety warning", ["language"]
)
retrieval_duration = Histogram(
    f"{_NS}_retrieval_duration_seconds",
    "Document search duration",
    buckets=(0.05, 0.1, 0.2, 0.4, 0.8, 1.5, 3, 6, 10),
)
retrieval_errors = Counter(
    f"{_NS}_retrieval_errors_total", "Document searches that failed (answer not grounded)"
)

# --- Photos and videos ---
diagnoses = Counter(f"{_NS}_diagnoses_total", "Photo / video diagnoses", ["source", "status"])
media_archived = Counter(f"{_NS}_media_archive_total", "Media archive attempts", ["outcome"])

# --- Users ---
feedback = Counter(f"{_NS}_feedback_total", "Answer ratings", ["rating"])
rate_limited = Counter(f"{_NS}_rate_limited_total", "Requests refused by rate limits", ["limit"])

# --- Dependencies ---
dependency_up = Gauge(
    f"{_NS}_dependency_up", "Last readiness check result (1 ok, 0 failing)", ["name"]
)
build_info = Gauge(f"{_NS}_build_info", "Version and environment", ["version", "env"])
