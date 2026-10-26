"""Performance test against a running API (debug timings require ENABLE_DEV_UI=true).

1. Stage breakdown: distinct questions one at a time -> p50/p95 per stage
   (query embedding, vector search, keyword search, retrieval total, LLM, server total).
2. Repeat question: the same question twice -> cost of a repeat.
3. Load: N concurrent users (levels), each asking one question -> latency, errors, throughput.

Keep it small on the free tier (each question is one LLM call + one embedding call).

Usage:
    python -m scripts.perf_test                        # default: 8 questions, levels 1,3,6
    python -m scripts.perf_test --levels 1,3 --label baseline
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from scripts.api_auth import api_headers

ROOT = Path(__file__).resolve().parent.parent

QUESTIONS = [
    "Symptoms of early blight in tomato?",
    "టమాటాలో ఎర్లీ బ్లైట్ నివారణ ఎలా?",
    "ಭತ್ತದಲ್ಲಿ ಬೆಂಕಿ ರೋಗ ನಿಯಂತ್ರಣ ಹೇಗೆ?",
    "गेहूं में पीला रतुआ का इलाज",
    "What is the waiting period for acetamiprid on cotton?",
    "मिर्च में थ्रिप्स का नियंत्रण कैसे करें?",
    "pattilo tella doma nivaranaku e mandu vadali",
    "ತೊಗರಿಯಲ್ಲಿ ಕಾಯಿ ಕೊರಕ ಹುಳುವಿನ ನಿಯಂತ್ರಣ",
    "How to control fall armyworm in maize?",
    "आलू में पछेती झुलसा रोग का उपचार",
    "బంగాళాదుంపలో లేట్ బ్లైట్ లక్షణాలు",
    "Which fungicides are approved for potato late blight?",
]


def pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return ordered[index]


async def ask(client: httpx.AsyncClient, message: str) -> tuple[int, float, dict[str, Any]]:
    start = time.perf_counter()
    response = await client.post("/api/v1/chat", json={"message": message, "debug": True})
    wall = time.perf_counter() - start
    try:
        body = response.json()
    except ValueError:
        body = {}
    return response.status_code, wall, body


async def stage_breakdown(
    client: httpx.AsyncClient, questions: list[str], delay: float
) -> dict[str, Any]:
    stages: dict[str, list[float]] = defaultdict(list)
    for i, question in enumerate(questions):
        if i:
            await asyncio.sleep(delay)
        status, wall, body = await ask(client, question)
        if status != 200:
            print(f"  ! {status} {body.get('error', {}).get('code')}  {question[:40]}")
            continue
        timings = (body.get("debug") or {}).get("timings_ms", {})
        for name, value in timings.items():
            stages[name].append(value)
        stages["server_total"].append(body["processing_time"] * 1000)
        stages["wall"].append(wall * 1000)
        print(f"  {wall:5.2f}s  {question[:50]}")
    summary = {
        name: {"p50": round(statistics.median(v)), "p95": round(pct(v, 0.95)), "n": len(v)}
        for name, v in stages.items()
    }
    print(f"\n  {'stage':22} {'p50 ms':>8} {'p95 ms':>8}")
    for name in (
        "embed",
        "keyword_search",
        "vector_search",
        "retrieval_total",
        "llm",
        "server_total",
        "wall",
    ):
        if name in summary:
            s = summary[name]
            print(f"  {name:22} {s['p50']:8} {s['p95']:8}")
    return summary


async def repeat_question(client: httpx.AsyncClient) -> dict[str, float]:
    question = "Symptoms of early blight in tomato?"
    _, first, _ = await ask(client, question)
    await asyncio.sleep(1)
    _, second, body = await ask(client, question)
    print(f"  first {first:.2f}s, repeat {second:.2f}s (model={body.get('model')})")
    return {"first_s": round(first, 2), "repeat_s": round(second, 2)}


async def load(client: httpx.AsyncClient, level: int, offset: int) -> dict[str, Any]:
    questions = [QUESTIONS[(offset + i) % len(QUESTIONS)] for i in range(level)]
    start = time.perf_counter()
    results = await asyncio.gather(*(ask(client, q) for q in questions))
    elapsed = time.perf_counter() - start
    codes = Counter(
        status if status == 200 else f"{status}:{body.get('error', {}).get('code')}"
        for status, _, body in results
    )
    walls = [wall for status, wall, _ in results if status == 200]
    summary = {
        "users": level,
        "ok": codes.get(200, 0),
        "errors": {str(k): v for k, v in codes.items() if k != 200},
        "p50_s": round(statistics.median(walls), 2) if walls else None,
        "max_s": round(max(walls), 2) if walls else None,
        "throughput_per_min": round(len(walls) / elapsed * 60, 1),
    }
    print(
        f"  {level:2} users: ok {summary['ok']}/{level}  p50 {summary['p50_s']}s  "
        f"max {summary['max_s']}s  errors {summary['errors'] or '-'}"
    )
    return summary


async def main(url: str, n_questions: int, levels: list[int], delay: float, label: str) -> int:
    async with httpx.AsyncClient(base_url=url, timeout=120, headers=api_headers()) as client:
        print("=== stage breakdown (one at a time) ===")
        breakdown = await stage_breakdown(client, QUESTIONS[:n_questions], delay)
        print("\n=== repeat question ===")
        repeat = await repeat_question(client)
        print("\n=== load (concurrent users) ===")
        loads = []
        for i, level in enumerate(levels):
            await asyncio.sleep(delay * 2)
            loads.append(await load(client, level, offset=i * 3))

    reports = ROOT / "evals" / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = reports / f"perf_{label}_{stamp}.json"
    path.write_text(
        json.dumps({"breakdown": breakdown, "repeat": repeat, "load": loads}, indent=2),
        encoding="utf-8",
    )
    print(f"\nreport: {path}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--url", default="http://127.0.0.1:8010")
    parser.add_argument("--questions", type=int, default=8)
    parser.add_argument("--levels", default="1,3,6")
    parser.add_argument("--delay", type=float, default=3.0)
    parser.add_argument("--label", default="run")
    args = parser.parse_args()
    levels = [int(x) for x in args.levels.split(",") if x]
    raise SystemExit(asyncio.run(main(args.url, args.questions, levels, args.delay, args.label)))
