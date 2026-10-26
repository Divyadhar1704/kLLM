"""End-to-end answer-quality evaluation against a running API (evals/answers_v1.jsonl).

Deterministic checks per question (no LLM judge, so results are reproducible and free):
    status 200 · answer language · answer mostly in the target script · grounded
    · expected source document · crop detected · key facts (English answers)
    · safety warning when expected · polite decline for off-topic / prompt injection

Usage (server must be running, ENABLE_DEV_UI=true for debug details):
    python -m scripts.eval_answers
    python -m scripts.eval_answers --only te-   # ids starting with te-
    python -m scripts.eval_answers --delay 4    # seconds between questions (free tier RPM)

A JSON report is written to evals/reports/.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parent.parent
DECLINE_WORDS = (
    "agricultur", "farm", "crop", "వ్యవసాయ", "పంట", "कृषि", "खेती", "फसल", "ಕೃಷಿ", "ಬೆಳೆ",
)  # fmt: skip
DECLINE_MAX_CHARS = 700


def check_case(
    case: dict[str, Any], status: int, body: dict[str, Any]
) -> list[tuple[str, bool, str]]:
    """Returns (check name, passed, detail) for every check that applies to this case."""
    checks: list[tuple[str, bool, str]] = [("http 200", status == 200, f"status {status}")]
    if status != 200:
        return checks
    answer = body["response"]
    lower = answer.lower()
    debug = body.get("debug") or {}

    if "expect_language" in case:
        checks.append(
            ("language", body["language"] == case["expect_language"], f"got {body['language']}")
        )
        language_check = debug.get("language_check")
        if language_check and case["expect_language"] != "en" and not case.get("expect_decline"):
            share = language_check["answer_script_share"]
            checks.append(("answer in target script", share >= 0.4, f"{share:.0%} in script"))
    if "expect_grounded" in case:
        checks.append(
            (
                "grounded",
                body["grounded"] == case["expect_grounded"],
                f"grounded={body['grounded']}",
            )
        )
    if "source_title_any" in case:
        titles = [s["title"] for s in body["sources"]]
        hit = any(w.lower() in t.lower() for t in titles for w in case["source_title_any"])
        checks.append(
            ("expected source", hit, "; ".join(t[:40] for t in titles[:3]) or "no sources")
        )
    if "crops_include" in case and debug:
        detected = debug.get("crops_detected", [])
        ok = all(c in detected for c in case["crops_include"])
        checks.append(("crop detected", ok, f"detected {detected}"))
    for group in case.get("must_include_any", []):
        ok = any(term.lower() in lower for term in group)
        checks.append((f"mentions one of {group[:3]}", ok, ""))
    for term in case.get("must_not_include", []):
        checks.append((f"does not mention {term}", term.lower() not in lower, ""))
    if case.get("expect_safety"):
        warnings = body.get("safety_warnings", [])
        checks.append(("safety warning", bool(warnings), f"{len(warnings)} warning(s)"))
    if case.get("expect_decline"):
        ok = len(answer) <= DECLINE_MAX_CHARS and any(w in lower for w in DECLINE_WORDS)
        checks.append(("polite decline", ok, f"{len(answer)} chars"))
    return checks


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--url", default="http://127.0.0.1:8010")
    parser.add_argument("--eval", type=Path, default=ROOT / "evals" / "answers_v1.jsonl")
    parser.add_argument("--only", default="", help="run only ids starting with this prefix")
    parser.add_argument("--delay", type=float, default=4.0, help="seconds between questions")
    args = parser.parse_args()

    lines = args.eval.read_text(encoding="utf-8").splitlines()
    cases = [json.loads(line) for line in lines if line.strip()]
    cases = [c for c in cases if c["id"].startswith(args.only)]

    results: list[dict[str, Any]] = []
    latencies: list[float] = []
    with httpx.Client(base_url=args.url, timeout=120) as client:
        for index, case in enumerate(cases):
            if index:
                time.sleep(args.delay)
            start = time.perf_counter()
            response = client.post(
                "/api/v1/chat", json={"message": case["question"], "debug": True}
            )
            seconds = time.perf_counter() - start
            body = response.json()
            checks = check_case(case, response.status_code, body)
            passed = all(ok for _, ok, _ in checks)
            if response.status_code == 200:
                latencies.append(seconds)
            print(f"{'PASS' if passed else 'FAIL'}  {case['id']:30} {seconds:5.1f}s")
            for name, ok, detail in checks:
                if not ok:
                    print(f"        x {name}: {detail}")
            if not passed and response.status_code == 200:
                print(f"        answer: {body['response'][:160].replace(chr(10), ' ')}")
            if response.status_code != 200:
                print(f"        error: {body.get('error', body)}")
            results.append(
                {
                    "id": case["id"],
                    "passed": passed,
                    "seconds": round(seconds, 2),
                    "checks": [{"name": n, "ok": ok, "detail": d} for n, ok, d in checks],
                    "response": body.get("response"),
                    "sources": [s["title"] for s in body.get("sources", [])],
                }
            )

    total = len(results)
    passed_count = sum(r["passed"] for r in results)
    all_checks = [c for r in results for c in r["checks"]]
    print("-" * 70)
    print(
        f"questions passed {passed_count}/{total}   checks passed "
        f"{sum(c['ok'] for c in all_checks)}/{len(all_checks)}"
    )
    if latencies:
        print(
            f"latency p50 {statistics.median(latencies):.1f}s  "
            f"max {max(latencies):.1f}s  (n={len(latencies)})"
        )

    reports = ROOT / "evals" / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = reports / f"answers_{stamp}.json"
    path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"report: {path}")
    return 0 if passed_count == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
