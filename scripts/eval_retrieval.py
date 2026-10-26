"""Retrieval accuracy on a golden question set (evals/retrieval_v1.jsonl), against the live DB.

A question counts as a hit at rank r if the r-th retrieved chunk contains any of its `expect`
strings (case-insensitive). Reports hit@1, hit@K, MRR and latency, per mode and per language.

Usage:
    python -m scripts.eval_retrieval                  # hybrid vs vector-only
    python -m scripts.eval_retrieval --show-misses    # print what was retrieved for misses
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from app.core.config import get_settings
from app.core.logging import setup_logging
from app.db import Database
from app.rag.retriever import RetrievalResult, Retriever
from app.services.embeddings import build_embedder

ROOT = Path(__file__).resolve().parent.parent


def first_hit_rank(result: RetrievalResult, expect: list[str]) -> int | None:
    for rank, chunk in enumerate(result.chunks, start=1):
        text = chunk.hit.content.lower()
        if any(e.lower() in text for e in expect):
            return rank
    return None


def load_cases(eval_path: Path) -> list[dict[str, object]]:
    lines = eval_path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


async def main(cases: list[dict[str, object]], show_misses: bool) -> int:
    settings = get_settings()
    database = Database.from_settings(settings)
    embedder = build_embedder(settings)  # query embeddings are cached across the two modes
    retriever = Retriever(db=database, embedder=embedder, settings=settings)
    k = settings.rag_top_k
    try:
        for mode, hybrid in (("hybrid", True), ("vector-only", False)):
            ranks: list[int | None] = []
            latencies: list[float] = []
            by_language: dict[str, list[bool]] = defaultdict(list)
            misses: list[tuple[dict[str, object], RetrievalResult]] = []
            for case in cases:
                result = await retriever.retrieve(str(case["question"]), hybrid=hybrid)
                if result.error:
                    print(f"  {case['id']}: retrieval error {result.error}")
                rank = first_hit_rank(result, list(case["expect"]))  # type: ignore[arg-type]
                ranks.append(rank)
                latencies.append(result.total_ms - result.embed_ms)  # DB time; embedding cached
                by_language[str(case["language"])].append(rank is not None)
                if rank is None:
                    misses.append((case, result))

            n = len(cases)
            hit1 = sum(1 for r in ranks if r == 1) / n
            hitk = sum(1 for r in ranks if r is not None) / n
            mrr = sum(1 / r for r in ranks if r) / n
            print(f"\n=== {mode}: hit@1 {hit1:.0%}  hit@{k} {hitk:.0%}  MRR {mrr:.2f}  "
                  f"search p50 {statistics.median(latencies):.0f} ms")  # fmt: skip
            print("    " + "  ".join(
                f"{lang} {sum(v)}/{len(v)}" for lang, v in sorted(by_language.items())
            ))  # fmt: skip
            for case, result in misses:
                print(f"    MISS {case['id']}: {case['question']}  crops={result.crops}")
                if show_misses:
                    for chunk in result.chunks:
                        body = chunk.hit.content.split("\n", 1)[-1].replace("\n", " ")[:110]
                        print(f"         {chunk.score:.4f} {chunk.hit.title[:35]:35} {body}")
    finally:
        await embedder.aclose()
        await database.aclose()
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--eval", type=Path, default=ROOT / "evals" / "retrieval_v1.jsonl")
    parser.add_argument("--show-misses", action="store_true")
    args = parser.parse_args()
    setup_logging(get_settings())
    raise SystemExit(asyncio.run(main(load_cases(args.eval), args.show_misses)))
