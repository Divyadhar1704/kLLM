"""Ingest agricultural documents into chunks for RAG.

Usage:
    python -m scripts.ingest                                  # documents/raw -> documents/processed
    python -m scripts.ingest --source "D:/more docs" --source documents/raw
    python -m scripts.ingest --force                          # ignore cache, re-process everything
    python -m scripts.ingest --show 3                         # print 3 sample chunks per doc type
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

from app.core.config import get_settings
from app.core.logging import setup_logging
from app.rag.ingestion.pipeline import load_chunks, run_ingestion

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--source", action="append", type=Path, help="file or folder (repeatable)")
    parser.add_argument("--out", type=Path, default=ROOT / "documents" / "processed")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--show", type=int, default=0, help="sample chunks to print per doc type")
    args = parser.parse_args()

    setup_logging(get_settings())
    sources = args.source or [ROOT / "documents" / "raw"]
    missing = [s for s in sources if not s.exists()]
    if missing:
        print(f"Source not found: {', '.join(map(str, missing))}", file=sys.stderr)
        return 1

    report = run_ingestion(sources, args.out, workers=args.workers, force=args.force)

    print("\n" + "=" * 100)
    print(f"{'document':58} {'type':22} {'pages':>5} {'chunks':>6} {'as_of':>10}")
    print("-" * 100)
    for doc in report.documents:
        print(
            f"{doc.title[:58]:58} {doc.doc_type[:22]:22} {doc.pages:5} {doc.chunks:6} "
            f"{doc.as_of or '-':>10}"
        )
    print("-" * 100)
    print(
        f"documents={len(report.documents)} (processed={report.processed}, cached={report.cached}) "
        f"chunks={report.total_chunks} pesticide_restrictions={report.restrictions} "
        f"time={report.seconds}s"
    )
    if report.empty_pages:
        print(f"pages with no text (scanned/images): {report.empty_pages}")
    if report.skipped:
        print(f"skipped: {dict(report.skipped)}")
    if report.failed:
        print("FAILED:")
        for path, error in report.failed.items():
            print(f"  {path}: {error}")
    print(f"output: {args.out}")

    if args.show:
        by_type: dict[str, list[str]] = defaultdict(list)
        for chunk in load_chunks(args.out / "chunks.jsonl"):
            by_type[chunk.doc_type].append(chunk.text)
        rng = random.Random(7)  # noqa: S311 - sampling for display only
        for doc_type, texts in sorted(by_type.items()):
            for text in rng.sample(texts, min(args.show, len(texts))):
                print(f"\n----- [{doc_type}] -----\n{text[:900]}")
    return 1 if report.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
