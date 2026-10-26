"""Send questions to a running KrishiMitra API and print the answers.

Usage:
    python -m scripts.ask "What are the symptoms of early blight in tomato?"
    python -m scripts.ask --lang te "టమాటాలో ఎర్లీ బ్లైట్ లక్షణాలు ఏమిటి?"
    python -m scripts.ask --demo          # one question in each language
    python -m scripts.ask --demo --stream # same, streamed (shows time to first text)
    python -m scripts.ask --url http://127.0.0.1:8010 "..."
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import httpx

from scripts.api_auth import api_headers

DEMO_QUESTIONS: list[tuple[str | None, str]] = [
    (None, "What are the symptoms of early blight in tomato?"),
    (None, "టమాటాలో ఎర్లీ బ్లైట్ లక్షణాలు ఏమిటి?"),
    (None, "ಭತ್ತದಲ್ಲಿ ಬೆಂಕಿ ರೋಗ (ಬ್ಲಾಸ್ಟ್) ತಡೆಯುವುದು ಹೇಗೆ?"),
    (None, "गेहूं में पीला रतुआ रोग का नियंत्रण कैसे करें?"),
]


def ask(client: httpx.Client, url: str, message: str, language: str | None) -> bool:
    payload: dict[str, object] = {"message": message, "debug": True}
    if language:
        payload["language"] = language
    start = time.perf_counter()
    response = client.post(f"{url}/api/v1/chat", json=payload)
    wall = time.perf_counter() - start
    print("=" * 80)
    print(f"Q: {message}")
    if response.status_code != 200:
        print(f"ERROR {response.status_code}: {response.text}")
        return False
    body = response.json()
    usage = body["usage"]
    print(
        f"[lang={body['language']} detected={body['language_detected']} model={body['model']} "
        f"grounded={body['grounded']} server={body['processing_time']}s wall={wall:.2f}s "
        f"tokens in={usage['input_tokens']} out={usage['output_tokens']}]"
    )
    debug = body.get("debug")
    if debug:
        timings = " ".join(f"{k}={v:.0f}" for k, v in debug["timings_ms"].items())
        print(f"[crops={debug['crops_detected']} keywords={debug['keyword_query']!r} {timings} ms]")
        if debug["retrieval_error"]:
            print(f"[retrieval error: {debug['retrieval_error']}]")
    print()
    print(body["response"])
    for source in body["sources"]:
        page = f", p. {source['page_start']}" if source["page_start"] else ""
        print(f"  [{source['index']}] {source['title']}{page}")
    for warning in body["safety_warnings"]:
        print(f"  SAFETY: {warning[:80]}...")
    return True


def ask_stream(client: httpx.Client, url: str, message: str, language: str | None) -> bool:
    payload: dict[str, object] = {"message": message, "debug": True}
    if language:
        payload["language"] = language
    print("=" * 80)
    print(f"Q: {message}  (streaming)\n")
    start = time.perf_counter()
    first_token: float | None = None
    event: str | None = None
    with client.stream("POST", f"{url}/api/v1/chat/stream", json=payload) as response:
        if response.status_code != 200:
            print(f"ERROR {response.status_code}: {response.read().decode()}")
            return False
        for line in response.iter_lines():
            if line.startswith("event: "):
                event = line[7:]
                continue
            if not line.startswith("data: "):
                continue
            data = json.loads(line[6:])
            if event == "delta":
                if first_token is None:
                    first_token = time.perf_counter() - start
                print(data["text"], end="", flush=True)
            elif event == "done":
                total = time.perf_counter() - start
                ttft = f"{first_token:.2f}s" if first_token is not None else "-"
                print(
                    f"\n\n[first text after {ttft}, complete after {total:.2f}s, "
                    f"grounded={data['grounded']}, sources={len(data['sources'])}, "
                    f"model={data['model']}]"
                )
            elif event == "error":
                print(f"\nSTREAM ERROR: {data['error']}")
                return False
    return True


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("message", nargs="?")
    parser.add_argument("--lang", choices=["en", "te", "kn", "hi"])
    parser.add_argument("--url", default="http://127.0.0.1:8010")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--stream", action="store_true", help="use /chat/stream (SSE)")
    args = parser.parse_args()

    questions = DEMO_QUESTIONS if args.demo else [(args.lang, args.message)]
    if not args.demo and not args.message:
        parser.error("message is required unless --demo is used")

    send = ask_stream if args.stream else ask
    ok = True
    with httpx.Client(timeout=90, headers=api_headers()) as client:
        for language, message in questions:
            ok = send(client, args.url, message, language) and ok
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
