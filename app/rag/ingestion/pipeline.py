"""Ingestion pipeline: discover -> extract -> clean -> classify -> chunk -> write.

Outputs (in the output directory):
    chunks.jsonl               one Chunk per line, input to embedding (Phase 8)
    manifest.json              per-document records, skipped files, run statistics
    pesticide_restrictions.json  structured banned/restricted list (safety checks)
    cache/<sha>-v<N>.json      per-document results; unchanged files are not re-processed
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections import Counter
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import UTC, datetime
from fnmatch import fnmatch
from pathlib import Path

from pydantic import BaseModel

from app.core.logging import get_logger
from app.rag.ingestion.chunker import ChunkerConfig, chunk_document
from app.rag.ingestion.extractors import EXTRACTORS, extract
from app.rag.ingestion.metadata import (
    classify_doc_type,
    detect_as_of,
    detect_crops,
    make_title,
    source_org_for,
)
from app.rag.ingestion.models import Block, Chunk, DocumentRecord, ExtractedDocument, Page
from app.rag.ingestion.pesticides import PesticideRestriction, parse_banned_pdf
from app.rag.ingestion.text import clean_page_texts, normalize_text
from app.services.language import detect_language

logger = get_logger(__name__)

# Bump to force re-processing for reasons outside the code (e.g. a library upgrade).
PIPELINE_VERSION = 2


def _code_fingerprint() -> str:
    """Hash of the ingestion package source: any code change invalidates the cache automatically."""
    digest = hashlib.sha256()
    for module in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(module.name.encode())
        digest.update(module.read_bytes())
    return digest.hexdigest()[:10]


CACHE_KEY = f"v{PIPELINE_VERSION}-{_code_fingerprint()}"

DEFAULT_EXCLUDES: tuple[str, ...] = (
    "*/labels/*",  # image-dataset annotation files
    "*/.svn/*",
    "*/__MACOSX/*",
    "*/.*",
)


class CachedDocument(BaseModel):
    record: DocumentRecord
    chunks: list[Chunk]
    restrictions: list[PesticideRestriction] = []
    empty_pages: int = 0


@dataclass(slots=True)
class IngestionReport:
    documents: list[DocumentRecord] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)
    skipped: Counter[str] = field(default_factory=Counter)
    cached: int = 0
    processed: int = 0
    total_chunks: int = 0
    restrictions: int = 0
    empty_pages: dict[str, int] = field(default_factory=dict)
    seconds: float = 0.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def discover(
    sources: Iterable[Path], excludes: Iterable[str] = DEFAULT_EXCLUDES
) -> tuple[list[tuple[Path, Path]], Counter[str]]:
    """Return (file, source_root) pairs for supported files, and counts of skipped extensions."""
    patterns = tuple(excludes)
    found: list[tuple[Path, Path]] = []
    skipped: Counter[str] = Counter()
    for root in sources:
        files = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file())
        base = root.parent if root.is_file() else root
        for path in files:
            posix = "/" + path.as_posix()
            if any(fnmatch(posix, pattern) for pattern in patterns):
                skipped["excluded"] += 1
                continue
            suffix = path.suffix.lower()
            if suffix in EXTRACTORS:
                found.append((path, base))
            else:
                skipped[suffix or "(no extension)"] += 1
    return found, skipped


def _clean(document: ExtractedDocument) -> tuple[list[Page], int, str]:
    """Clean prose blocks per page; returns pages, number of empty pages, detected language."""
    sample = " ".join(
        b.text for page in document.pages[:5] for b in page.blocks if b.kind == "text"
    )[:20000]
    language = detect_language(sample) if sample.strip() else "en"
    texts = ["\n\n".join(b.text for b in p.blocks if b.kind == "text") for p in document.pages]
    cleaned_texts = clean_page_texts(texts, drop_devanagari_lines=language == "en")

    pages: list[Page] = []
    empty = 0
    for page, text in zip(document.pages, cleaned_texts, strict=True):
        blocks = [Block("text", text)] if text else []
        blocks += [b for b in page.blocks if b.kind == "table_row"]
        if not blocks:
            empty += 1
        pages.append(Page(number=page.number, blocks=blocks))
    return pages, empty, language


def process_document(path: Path, root: Path, sha256: str, config: ChunkerConfig) -> CachedDocument:
    document = extract(path)
    pages, empty_pages, language = _clean(document)

    head = normalize_text(
        " ".join(b.text for p in pages[:3] for b in p.blocks if b.kind == "text")
    )[:8000]
    doc_type = classify_doc_type(path.name, head)
    record = DocumentRecord(
        doc_id=sha256[:16],
        sha256=sha256,
        source_path=path.relative_to(root.parent).as_posix(),
        title=make_title(path, doc_type, document.title_hint),
        doc_type=doc_type,
        source_org=source_org_for(doc_type),
        as_of=detect_as_of(path.name, head),
        # Document-level crops come from the file name only: catalogue documents mention
        # dozens of crops, but a "Tomato" IPM package is about tomato on every page.
        crops=detect_crops(path.stem.replace("_", " ")),
        language=language,
        pages=len(pages),
        extractor=document.extractor,
    )
    chunks = chunk_document(pages, record, config)
    record.chunks = len(chunks)
    record.chars = sum(c.char_count for c in chunks)

    restrictions: list[PesticideRestriction] = []
    if doc_type == "pesticide_banned_list" and path.suffix.lower() == ".pdf":
        restrictions = parse_banned_pdf(path)
    return CachedDocument(
        record=record, chunks=chunks, restrictions=restrictions, empty_pages=empty_pages
    )


def _cache_path(cache_dir: Path, sha256: str) -> Path:
    return cache_dir / f"{sha256}-{CACHE_KEY}.json"


def _prune_cache(cache_dir: Path, keep: set[Path]) -> int:
    removed = 0
    for path in cache_dir.glob("*.json"):
        if path not in keep:
            path.unlink(missing_ok=True)
            removed += 1
    return removed


def run_ingestion(
    sources: list[Path],
    output_dir: Path,
    *,
    workers: int | None = None,
    force: bool = False,
    config: ChunkerConfig | None = None,
) -> IngestionReport:
    start = time.perf_counter()
    config = config or ChunkerConfig()
    cache_dir = output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    files, skipped = discover(sources)
    report = IngestionReport(skipped=skipped)
    results: dict[str, CachedDocument] = {}
    pending: list[tuple[Path, Path, str]] = []

    for path, root in files:
        sha = sha256_file(path)
        if sha in results or any(p[2] == sha for p in pending):
            report.skipped["duplicate"] += 1
            continue
        cached = _cache_path(cache_dir, sha)
        if not force and cached.exists():
            results[sha] = CachedDocument.model_validate_json(cached.read_text(encoding="utf-8"))
            report.cached += 1
        else:
            pending.append((path, root, sha))

    max_workers = workers or min(4, os.cpu_count() or 1)
    if pending:
        logger.info("ingestion_processing", documents=len(pending), workers=max_workers)
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(process_document, path, root, sha, config): (path, sha)
                for path, root, sha in pending
            }
            for future in as_completed(futures):
                path, sha = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    report.failed[str(path)] = f"{type(exc).__name__}: {exc}"[:300]
                    logger.error("ingestion_failed", path=str(path), error=report.failed[str(path)])
                    continue
                _cache_path(cache_dir, sha).write_text(result.model_dump_json(), encoding="utf-8")
                results[sha] = result
                report.processed += 1
                logger.info(
                    "ingestion_document_done",
                    title=result.record.title,
                    pages=result.record.pages,
                    chunks=result.record.chunks,
                )

    ordered = sorted(results.values(), key=lambda r: r.record.source_path)
    _write_outputs(output_dir, ordered, report)
    # Drop cache entries from older code versions and from files no longer in the sources.
    _prune_cache(cache_dir, {_cache_path(cache_dir, sha) for sha in results})
    report.seconds = round(time.perf_counter() - start, 1)
    return report


def _write_outputs(
    output_dir: Path, results: list[CachedDocument], report: IngestionReport
) -> None:
    chunks_path = output_dir / "chunks.jsonl"
    tmp = chunks_path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for result in results:
            for chunk in result.chunks:
                handle.write(chunk.model_dump_json() + "\n")
                report.total_chunks += 1
    tmp.replace(chunks_path)  # atomic: readers never see a half-written file

    restrictions = [r for result in results for r in result.restrictions]
    report.restrictions = len(restrictions)
    (output_dir / "pesticide_restrictions.json").write_text(
        json.dumps([r.model_dump() for r in restrictions], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    report.documents = [r.record for r in results]
    report.empty_pages = {r.record.source_path: r.empty_pages for r in results if r.empty_pages}
    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "pipeline_version": CACHE_KEY,
        "documents": [r.model_dump() for r in report.documents],
        "total_chunks": report.total_chunks,
        "pesticide_restrictions": report.restrictions,
        "empty_pages": report.empty_pages,
        "failed": report.failed,
        "skipped": dict(report.skipped),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def load_chunks(path: Path) -> list[Chunk]:
    with path.open(encoding="utf-8") as handle:
        return [Chunk.model_validate_json(line) for line in handle if line.strip()]
