"""Live summary of a running server's metrics (since it started).

Usage:
    python -m scripts.stats                               # http://127.0.0.1:8010/metrics
    python -m scripts.stats --url http://127.0.0.1:9464/metrics
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from collections.abc import Iterable

import httpx
from prometheus_client.parser import text_string_to_metric_families
from prometheus_client.samples import Sample

Samples = dict[str, list[Sample]]


def load(url: str) -> Samples:
    response = httpx.get(url, timeout=10)
    response.raise_for_status()
    samples: Samples = defaultdict(list)
    for family in text_string_to_metric_families(response.text):
        for sample in family.samples:
            samples[sample.name].append(sample)
    return samples


def total(samples: Iterable[Sample], **match: str) -> float:
    return sum(s.value for s in samples if all(s.labels.get(k) == v for k, v in match.items()))


def grouped(samples: Iterable[Sample], *keys: str) -> dict[tuple[str, ...], float]:
    out: dict[tuple[str, ...], float] = defaultdict(float)
    for s in samples:
        out[tuple(s.labels.get(k, "") for k in keys)] += s.value
    return dict(sorted(out.items()))


def quantile(buckets: list[Sample], q: float) -> float | None:
    """Approximate quantile from cumulative histogram buckets (upper bound of the bucket)."""
    by_le: dict[float, float] = defaultdict(float)
    for b in buckets:
        by_le[float(b.labels["le"])] += b.value
    if not by_le or by_le[max(by_le)] == 0:
        return None
    target = by_le[max(by_le)] * q
    for le in sorted(by_le):
        if by_le[le] >= target:
            return le
    return None


def pct(part: float, whole: float) -> str:
    return f"{100 * part / whole:.0f}%" if whole else "-"


def main(url: str) -> int:
    m = load(url)
    print(f"source: {url}\n")

    print("=== Requests by route ===")
    requests = m.get("krishimitra_http_requests_total", [])
    buckets = m.get("krishimitra_http_request_duration_seconds_bucket", [])
    for (route,), count in grouped(requests, "route").items():
        errors_5xx = sum(
            s.value
            for s in requests
            if s.labels["route"] == route and s.labels["status"].startswith("5")
        )
        errors_4xx = sum(
            s.value
            for s in requests
            if s.labels["route"] == route and s.labels["status"].startswith("4")
        )
        route_buckets = [b for b in buckets if b.labels.get("route") == route]
        p50, p95 = quantile(route_buckets, 0.5), quantile(route_buckets, 0.95)
        print(
            f"  {route:28} {count:6.0f}  4xx {errors_4xx:4.0f}  5xx {errors_5xx:4.0f}  "
            f"p50 <= {p50 or '-'} s  p95 <= {p95 or '-'} s"
        )

    print("\n=== Answers ===")
    answers = m.get("krishimitra_answers_total", [])
    count = total(answers)
    print(
        f"  total {count:.0f}  grounded in documents {pct(total(answers, grounded='true'), count)}"
    )
    for (kind, language), n in grouped(answers, "kind", "language").items():
        print(f"    {kind:6} {language}: {n:.0f}")
    cache = m.get("krishimitra_answer_cache_total", [])
    hits = total(cache, result="hit")
    print(f"  cache hits {hits:.0f} of {total(cache):.0f} lookups ({pct(hits, total(cache))})")
    print(
        f"  language retries {total(m.get('krishimitra_language_retries_total', [])):.0f}  "
        f"safety warnings {total(m.get('krishimitra_safety_warnings_total', [])):.0f}  "
        f"search errors {total(m.get('krishimitra_retrieval_errors_total', [])):.0f}"
    )

    print("\n=== AI calls ===")
    calls = m.get("krishimitra_llm_calls_total", [])
    for (purpose, model, outcome), n in grouped(calls, "purpose", "model", "outcome").items():
        print(f"  {purpose:6} {model:28} {outcome:22} {n:.0f}")
    tokens = m.get("krishimitra_llm_tokens_total", [])
    for (purpose, kind), n in grouped(tokens, "purpose", "kind").items():
        print(f"  tokens {purpose:6} {kind:9} {n:,.0f}")
    print(
        f"  estimated cost (chat) ${total(m.get('krishimitra_llm_cost_usd_total', [])):.4f}"
        f"  retries {total(m.get('krishimitra_llm_retries_total', [])):.0f}"
        f"  fallbacks {total(m.get('krishimitra_llm_fallbacks_total', [])):.0f}"
    )

    print("\n=== Photos / videos ===")
    for (source, status), n in grouped(
        m.get("krishimitra_diagnoses_total", []), "source", "status"
    ).items():
        print(f"  {source:5} {status:13} {n:.0f}")
    archived = m.get("krishimitra_media_archive_total", [])
    print(
        f"  media kept {total(archived, outcome='stored'):.0f}  "
        f"failed {total(archived, outcome='failed'):.0f}"
    )

    print("\n=== Users ===")
    feedback = m.get("krishimitra_feedback_total", [])
    up, down = total(feedback, rating="up"), total(feedback, rating="down")
    print(f"  feedback up {up:.0f}  down {down:.0f}  ({pct(up, up + down)} positive)")
    for (limit,), n in grouped(m.get("krishimitra_rate_limited_total", []), "limit").items():
        print(f"  rate limited ({limit}): {n:.0f}")

    print("\n=== Dependencies (last readiness check) ===")
    for (name,), n in grouped(m.get("krishimitra_dependency_up", []), "name").items():
        print(f"  {name:18} {'ok' if n >= 1 else 'FAILING'}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--url", default="http://127.0.0.1:8010/metrics")
    raise SystemExit(main(parser.parse_args().url))
