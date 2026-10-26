"""Load the knowledge base into Postgres/pgvector and search it.

Prerequisites: `alembic upgrade head`, `python -m scripts.ingest`, `python -m scripts.embed`.

Usage:
    python -m scripts.index                     # sync DB = chunks.jsonl + cached embeddings
    python -m scripts.index --dry-run           # show what would be loaded
    python -m scripts.index --stats             # what is in the database
    python -m scripts.index --search "టమాటాలో ఎర్లీ బ్లైట్ నివారణ" [--crop tomato] [--top 5]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

from app.core.config import get_settings
from app.core.logging import setup_logging
from app.db import Database
from app.rag.indexing import plan_index
from app.rag.ingestion.models import DocumentRecord
from app.rag.ingestion.pesticides import PesticideRestriction
from app.rag.ingestion.pipeline import load_chunks
from app.repositories.rag import corpus_stats, search_chunks, sync_corpus
from app.services.embeddings import SqliteEmbeddingStore, build_embedder

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "documents" / "processed"


async def run_sync(dry_run: bool) -> int:
    settings = get_settings()
    manifest = json.loads((PROCESSED / "manifest.json").read_text(encoding="utf-8"))
    records = [DocumentRecord.model_validate(d) for d in manifest["documents"]]
    chunks = load_chunks(PROCESSED / "chunks.jsonl")
    restrictions_path = PROCESSED / "pesticide_restrictions.json"
    restrictions = [
        PesticideRestriction.model_validate(r)
        for r in json.loads(restrictions_path.read_text(encoding="utf-8"))
    ]

    store = SqliteEmbeddingStore(PROCESSED / "embeddings.sqlite")
    embedder = build_embedder(settings, store=store)
    vectors = store.get_many([embedder.key(c.text, "document") for c in chunks])
    plan = plan_index(
        records,
        chunks,
        vectors,
        key_for=lambda c: embedder.key(c.text, "document"),
        embedding_model=embedder.model,
        restrictions=restrictions,
    )
    await embedder.aclose()
    store.close()

    print(
        f"plan: {len(plan.documents)} documents, {plan.indexed_chunks}/{plan.total_chunks} chunks "
        f"embedded, {len(plan.restrictions)} pesticide restrictions"
    )
    for title, count in plan.missing_vectors.items():
        print(f"  not yet embedded: {count:4} chunks  {title}")
    if not plan.chunks:
        print("Nothing to load: run `python -m scripts.embed` first.")
        return 1
    if dry_run:
        return 0

    database = Database.from_settings(settings)
    start = time.perf_counter()
    try:
        async with database.session() as session:
            stats = await sync_corpus(session, plan.documents, plan.chunks, plan.restrictions)
    finally:
        await database.aclose()
    print(
        f"synced in {time.perf_counter() - start:.1f}s: {stats.documents_upserted} documents, "
        f"{stats.chunks_upserted} chunks upserted; removed {stats.chunks_deleted} stale chunks, "
        f"{stats.documents_deleted} stale documents; {stats.restrictions_loaded} restrictions"
    )
    return 0


async def run_stats() -> int:
    database = Database.from_settings(get_settings())
    try:
        async with database.session() as session:
            print(json.dumps(await corpus_stats(session), indent=2))
    finally:
        await database.aclose()
    return 0


async def run_search(query: str, crops: list[str] | None, top: int) -> int:
    settings = get_settings()
    embedder = build_embedder(settings)
    database = Database.from_settings(settings)
    try:
        start = time.perf_counter()
        vector = await embedder.embed_query(query)
        embed_ms = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        async with database.session() as session:
            hits = await search_chunks(
                session, vector, embedding_model=embedder.model, top_k=top, crops=crops
            )
        search_ms = (time.perf_counter() - start) * 1000
    finally:
        await embedder.aclose()
        await database.aclose()
    print(f"query embed {embed_ms:.0f} ms, vector search {search_ms:.0f} ms, {len(hits)} hits\n")
    for hit in hits:
        body = hit.content.split("\n", 1)[-1].replace("\n", " ")
        print(f"{hit.similarity:.3f}  {hit.title}  p{hit.page_start}  crops={hit.crops[:5]}")
        print(f"       {body[:220]}\n")
    return 0


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stats", action="store_true")
    parser.add_argument("--search", type=str)
    parser.add_argument("--crop", action="append", dest="crops")
    parser.add_argument("--top", type=int, default=5)
    args = parser.parse_args()

    setup_logging(get_settings())
    if args.stats:
        return asyncio.run(run_stats())
    if args.search:
        return asyncio.run(run_search(args.search, args.crops, args.top))
    for name in ("manifest.json", "chunks.jsonl", "pesticide_restrictions.json"):
        if not (PROCESSED / name).exists():
            print(f"{PROCESSED / name} missing: run `python -m scripts.ingest`.", file=sys.stderr)
            return 1
    return asyncio.run(run_sync(args.dry_run))


if __name__ == "__main__":
    raise SystemExit(main())
