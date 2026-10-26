"""Embed all ingested chunks (documents/processed/chunks.jsonl) into the local embedding cache.

Resumable: vectors are stored per batch in documents/processed/embeddings.sqlite, so a re-run
only embeds new or changed chunks. Phase 9 loads chunks + cached vectors into the vector database.

Usage:
    python -m scripts.embed                  # embed everything not yet embedded
    python -m scripts.embed --limit 100      # embed only the first 100 chunks (testing)
    python -m scripts.embed --check          # retrieval sanity check: 4-language questions
                                             # against all embedded chunks (brute force)
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

from app.core.config import get_settings
from app.core.logging import setup_logging
from app.rag.ingestion.models import Chunk
from app.rag.ingestion.pipeline import load_chunks
from app.services.embeddings import SqliteEmbeddingStore, build_embedder
from app.services.llm.errors import LLMError, LLMQuotaExhaustedError

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "documents" / "processed"

CHECK_QUESTIONS: list[tuple[str, str, str]] = [
    # (language, question, words expected in a relevant chunk)
    ("en", "What is the waiting period for acetamiprid on cotton?", "acetamiprid"),
    ("te", "టమాటాలో ఎర్లీ బ్లైట్ నివారణకు ఏ మందు పిచికారీ చేయాలి?", "blight"),
    ("te", "పత్తిలో తెల్లదోమ నివారణ ఎలా?", "whitefl"),
    ("kn", "ಆಲೂಗಡ್ಡೆಯಲ್ಲಿ ಲೇಟ್ ಬ್ಲೈಟ್ ರೋಗ ನಿಯಂತ್ರಣ ಹೇಗೆ?", "late blight"),
    ("kn", "ತೊಗರಿಯಲ್ಲಿ ಕಾಯಿ ಕೊರಕ ಹುಳುವಿನ ನಿರ್ವಹಣೆ", "pod borer"),
    ("hi", "क्या मोनोक्रोटोफॉस सब्जियों पर प्रतिबंधित है?", "monocrotophos"),
    ("hi", "मिर्च में थ्रिप्स का नियंत्रण कैसे करें?", "thrips"),
    ("hi", "अदरक में प्रकंद सड़न रोग का उपचार", "rot"),
]


def progress_printer(start: float):  # type: ignore[no-untyped-def]
    def show(done: int, total: int) -> None:
        elapsed = time.perf_counter() - start
        rate = done / elapsed if elapsed > 0 and done else 0.0
        eta = (total - done) / rate if rate > 0 else 0.0
        print(f"\r  {done}/{total} chunks  {rate * 60:5.0f}/min  ETA {eta / 60:4.1f} min", end="")
        sys.stdout.flush()

    return show


async def run_embed(chunks: list[Chunk]) -> int:
    settings = get_settings()
    store = SqliteEmbeddingStore(PROCESSED / "embeddings.sqlite")
    embedder = build_embedder(settings, store=store)
    print(
        f"model={embedder.model} dims={embedder.dimensions} batch={embedder.batch_size} "
        f"rate_limit={settings.embedding_requests_per_minute or 'none'}/min chunks={len(chunks)}"
    )
    start = time.perf_counter()
    try:
        await embedder.embed(
            [c.text for c in chunks], "document", on_progress=progress_printer(start)
        )
    except LLMQuotaExhaustedError:
        print(f"\nDaily quota reached. Progress saved ({store.count()} vectors); re-run tomorrow.")
        return 2
    except LLMError as exc:
        print(f"\nEmbedding failed: {exc.code}: {exc.message}. Progress saved; re-run to resume.")
        return 1
    finally:
        await embedder.aclose()
    seconds = time.perf_counter() - start
    print(
        f"\nDone: {embedder.misses} embedded, {embedder.hits} from cache, "
        f"{store.count()} vectors stored, {seconds:.0f}s"
    )
    store.close()
    return 0


async def run_check(chunks: list[Chunk]) -> int:
    settings = get_settings()
    store = SqliteEmbeddingStore(PROCESSED / "embeddings.sqlite")
    embedder = build_embedder(settings, store=store)
    keys = [embedder.key(c.text, "document") for c in chunks]
    vectors = store.get_many(keys)
    indexed = [(c, vectors[k]) for c, k in zip(chunks, keys, strict=True) if k in vectors]
    print(f"searching {len(indexed)} embedded chunks (of {len(chunks)})\n")
    if not indexed:
        print("Nothing embedded yet: run `python -m scripts.embed` first.")
        return 1

    query_vectors = await embedder.embed([q for _, q, _ in CHECK_QUESTIONS], "query")
    await embedder.aclose()
    hits = 0
    for (language, question, expected), qv in zip(CHECK_QUESTIONS, query_vectors, strict=True):
        scored = sorted(
            ((sum(a * b for a, b in zip(qv, dv, strict=True)), c) for c, dv in indexed),
            key=lambda item: item[0],
            reverse=True,
        )[:3]
        relevant = any(expected in c.text.lower() for _, c in scored)
        hits += relevant
        print(f"[{language}] {question}  ->  {'RELEVANT' if relevant else 'MISS'}")
        for score, chunk in scored:
            line = chunk.text.split("\n", 1)[-1].replace("\n", " ")[:110]
            print(f"   {score:.3f}  {chunk.title[:40]:40} p{chunk.page_start}  {line}")
        print()
    print(f"top-3 contains a relevant chunk: {hits}/{len(CHECK_QUESTIONS)}")
    store.close()
    return 0


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--chunks", type=Path, default=PROCESSED / "chunks.jsonl")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    setup_logging(get_settings())
    if not args.chunks.exists():
        print(f"{args.chunks} not found: run `python -m scripts.ingest` first.", file=sys.stderr)
        return 1
    chunks = load_chunks(args.chunks)[: args.limit]
    return asyncio.run(run_check(chunks) if args.check else run_embed(chunks))


if __name__ == "__main__":
    raise SystemExit(main())
