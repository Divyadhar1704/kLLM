"""System tests against a running API: edge cases, error handling, security, concurrency.

Most checks make no AI calls. The few that do are marked [AI] (about 8 requests in total).

Usage (server running):
    python -m scripts.system_test
    python -m scripts.system_test --no-ai          # skip checks that call the AI model
    python -m scripts.system_test --concurrency 5  # concurrent real chat requests
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from collections import Counter
from collections.abc import Awaitable, Callable

import httpx

Check = tuple[str, bool, str]


async def run_checks(client: httpx.AsyncClient, use_ai: bool) -> list[Check]:
    results: list[Check] = []

    async def check(name: str, func: Callable[[], Awaitable[tuple[bool, str]]]) -> None:
        try:
            ok, detail = await func()
        except Exception as exc:
            ok, detail = False, f"{type(exc).__name__}: {exc}"
        results.append((name, ok, detail))
        print(f"{'PASS' if ok else 'FAIL'}  {name:58} {detail[:70]}")

    async def status(method: str, path: str, expect: int, **kwargs: object) -> tuple[bool, str]:
        r = await client.request(method, path, **kwargs)  # type: ignore[arg-type]
        return r.status_code == expect, f"HTTP {r.status_code}"

    async def error_code(payload: object, expect_status: int, expect_code: str) -> tuple[bool, str]:
        r = await client.post("/api/v1/chat", json=payload)
        body = r.json()
        code = body.get("error", {}).get("code")
        has_id = bool(body.get("error", {}).get("request_id"))
        ok = r.status_code == expect_status and code == expect_code and has_id
        return ok, f"HTTP {r.status_code} {code} request_id={'yes' if has_id else 'no'}"

    # ---------------------------------------------------------------- service
    await check("GET / service info", lambda: status("GET", "/", 200))
    await check("GET /health liveness", lambda: status("GET", "/health", 200))

    async def ready() -> tuple[bool, str]:
        r = await client.get("/health/ready")
        body = r.json()
        return r.status_code == 200 and body["status"] == "ok", ", ".join(
            f"{c['name']}={c['status']}" for c in body["checks"]
        )

    await check("GET /health/ready all dependencies ok", ready)

    # ---------------------------------------------------------------- validation
    await check(
        "empty message -> 422", lambda: error_code({"message": "   "}, 422, "validation_error")
    )
    await check("missing message -> 422", lambda: error_code({}, 422, "validation_error"))
    await check(
        "unsupported language 'ta' -> 422",
        lambda: error_code({"message": "hi", "language": "ta"}, 422, "validation_error"),
    )
    await check(
        "message over limit -> 400",
        lambda: error_code({"message": "x" * 2001}, 400, "bad_request"),
    )
    await check(
        "message over hard cap (10k) -> 422",
        lambda: error_code({"message": "x" * 10001}, 422, "validation_error"),
    )
    await check(
        "path-like conversation_id -> 422",
        lambda: error_code(
            {"message": "hi", "conversation_id": "../../etc"}, 422, "validation_error"
        ),
    )

    async def bad_json() -> tuple[bool, str]:
        r = await client.post(
            "/api/v1/chat", content=b"{not json", headers={"Content-Type": "application/json"}
        )
        return r.status_code == 422, f"HTTP {r.status_code}"

    await check("malformed JSON body -> 422", bad_json)
    await check("GET on POST endpoint -> 405", lambda: status("GET", "/api/v1/chat", 405))
    await check("unknown path -> 404", lambda: status("GET", "/api/v1/nope", 404))

    async def validation_echo() -> tuple[bool, str]:
        secret = "sk-test-secret-value-123"  # noqa: S105 - fake value to detect echoing
        r = await client.post("/api/v1/chat", json={"message": "hi", "language": secret})
        return secret not in r.text, "input not echoed" if secret not in r.text else "ECHOED"

    await check("validation errors do not echo input", validation_echo)

    # ---------------------------------------------------------------- security
    async def headers() -> tuple[bool, str]:
        r = await client.get("/health")
        needed = {
            "x-content-type-options": "nosniff",
            "x-frame-options": "DENY",
            "cache-control": "no-store",
        }
        missing = [k for k, v in needed.items() if r.headers.get(k) != v]
        missing += [] if "x-request-id" in r.headers else ["x-request-id"]
        missing += [] if "content-security-policy" in r.headers else ["content-security-policy"]
        return not missing, "all present" if not missing else f"missing {missing}"

    await check("security headers on API responses", headers)

    async def cors_blocked() -> tuple[bool, str]:
        r = await client.get("/health", headers={"Origin": "https://evil.example.com"})
        allowed = r.headers.get("access-control-allow-origin")
        return allowed is None, f"allow-origin={allowed}"

    await check("CORS: unknown origin not allowed", cors_blocked)

    async def docs_page() -> tuple[bool, str]:
        r = await client.get("/dev")
        return r.status_code == 200 and "KrishiMitra" in r.text, f"HTTP {r.status_code}"

    await check("dev console served (development)", docs_page)

    if not use_ai:
        return results

    # ---------------------------------------------------------------- AI-backed edge cases
    async def chat_ok(
        message: str, check_body: Callable[[dict], tuple[bool, str]]
    ) -> tuple[bool, str]:  # type: ignore[type-arg]
        r = await client.post("/api/v1/chat", json={"message": message})
        if r.status_code != 200:
            return False, f"HTTP {r.status_code} {r.text[:80]}"
        return check_body(r.json())

    await check(
        "[AI] HTML/script in message handled as text",
        lambda: chat_ok(
            "<script>alert(1)</script> tomato early blight symptoms?",
            lambda b: ("<script>" not in b["response"], f"{len(b['response'])} chars"),
        ),
    )
    await check(
        "[AI] SQL-like text is harmless",
        lambda: chat_ok(
            "tomato'; DROP TABLE rag_chunks; -- early blight",
            lambda b: (bool(b["response"]), f"grounded={b['grounded']}"),
        ),
    )
    await check(
        "[AI] emoji-only message",
        lambda: chat_ok("🌾🍅🐛", lambda b: (bool(b["response"]), f"lang={b['language']}")),
    )

    async def stream_format() -> tuple[bool, str]:
        events: list[str] = []
        deltas: list[str] = []
        done: dict | None = None  # type: ignore[type-arg]
        async with client.stream(
            "POST", "/api/v1/chat/stream", json={"message": "What is IPM?", "language": "en"}
        ) as r:
            name = None
            async for line in r.aiter_lines():
                if line.startswith("event: "):
                    name = line[7:]
                elif line.startswith("data: ") and name:
                    events.append(name)
                    data = json.loads(line[6:])
                    if name == "delta":
                        deltas.append(data["text"])
                    elif name == "done":
                        done = data
        ok = (
            bool(events)
            and events[0] == "meta"
            and events[-1] == "done"
            and done is not None
            and "".join(deltas) == done["response"]
        )
        matches = done is not None and "".join(deltas) == done["response"]
        return ok, f"{len(events)} events, deltas==done: {matches}"

    await check("[AI] streaming: meta -> delta* -> done, deltas == done", stream_format)
    return results


async def concurrency(client: httpx.AsyncClient, use_ai: bool, ai_requests: int) -> list[Check]:
    results: list[Check] = []

    async def burst(
        name: str, count: int, make: Callable[[int], Awaitable[httpx.Response]], expect: int
    ) -> None:
        start = time.perf_counter()
        responses = await asyncio.gather(*(make(i) for i in range(count)), return_exceptions=True)
        seconds = time.perf_counter() - start
        codes = Counter(
            r.status_code if isinstance(r, httpx.Response) else type(r).__name__ for r in responses
        )
        ok = codes.get(expect, 0) == count
        detail = f"{dict(codes)} in {seconds:.1f}s"
        results.append((name, ok, detail))
        print(f"{'PASS' if ok else 'FAIL'}  {name:58} {detail}")

    await burst("50 concurrent /health/ready", 50, lambda i: client.get("/health/ready"), 200)
    await burst(
        "100 concurrent invalid requests (no AI)",
        100,
        lambda i: client.post("/api/v1/chat", json={"message": ""}),
        422,
    )
    if use_ai and ai_requests:
        questions = [
            "Symptoms of early blight in tomato?",
            "టమాటాలో ఎర్లీ బ్లైట్ లక్షణాలు?",
            "ಭತ್ತದಲ್ಲಿ ಬೆಂಕಿ ರೋಗ ನಿಯಂತ್ರಣ",
            "गेहूं में पीला रतुआ का इलाज",
            "Waiting period of acetamiprid on cotton?",
            "मिर्च में थ्रिप्स का नियंत्रण",
            "Whitefly control in cotton?",
            "ತೊಗರಿಯಲ್ಲಿ ಕಾಯಿ ಕೊರಕ",
        ]
        latencies: list[float] = []

        async def timed(i: int) -> httpx.Response:
            start = time.perf_counter()
            r = await client.post("/api/v1/chat", json={"message": questions[i % len(questions)]})
            latencies.append(time.perf_counter() - start)
            return r

        await burst(f"[AI] {ai_requests} concurrent chat questions", ai_requests, timed, 200)
        if latencies:
            print(
                f"        latency under concurrency: p50 {statistics.median(latencies):.1f}s "
                f"max {max(latencies):.1f}s"
            )
    return results


async def main(url: str, use_ai: bool, ai_requests: int) -> int:
    limits = httpx.Limits(max_connections=200, max_keepalive_connections=50)
    async with httpx.AsyncClient(base_url=url, timeout=120, limits=limits) as client:
        print("=== edge cases, errors, security ===")
        results = await run_checks(client, use_ai)
        print("\n=== concurrency ===")
        results += await concurrency(client, use_ai, ai_requests)
    passed = sum(ok for _, ok, _ in results)
    print("-" * 70)
    print(f"system checks passed {passed}/{len(results)}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--url", default="http://127.0.0.1:8010")
    parser.add_argument("--no-ai", action="store_true")
    parser.add_argument("--concurrency", type=int, default=5, help="concurrent real chat requests")
    args = parser.parse_args()
    raise SystemExit(asyncio.run(main(args.url, not args.no_ai, args.concurrency)))
