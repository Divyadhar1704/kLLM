"""Section-aware chunking.

Units: prose paragraphs (wrapped lines rejoined, bullets kept on their own lines) and table rows.
Units are packed into chunks of ~target_chars; a unit is never split unless it alone exceeds
max_chars. Prose chunks overlap slightly so a sentence near a boundary is not lost. Each chunk
starts with a context header (title > section) because the LLM and the embedding both need it.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from app.rag.ingestion.metadata import detect_crops
from app.rag.ingestion.models import BlockKind, Chunk, DocumentRecord, Page
from app.rag.ingestion.text import is_page_number

_NUMBERED_HEADING = re.compile(r"^(?:\d+(?:\.\d+)*\.?|[IVX]+\.|[A-H]\.)\s+\S")
_SENTENCE_END = re.compile(r"(?<=[.!?;])\s+")
_CHARS_PER_TOKEN = 4  # English approximation, good enough for sizing


@dataclass(frozen=True, slots=True)
class ChunkerConfig:
    target_chars: int = 1500
    max_chars: int = 2200
    overlap_chars: int = 250
    min_chars: int = 80


@dataclass(frozen=True, slots=True)
class Unit:
    kind: BlockKind
    text: str
    page: int | None
    section: str | None


def is_heading(line: str) -> bool:
    line = line.strip()
    if not line or len(line) > 90 or line.endswith((".", ",", ";", ":")) or line.startswith("•"):
        return False
    if "http" in line.lower() or "www." in line.lower():
        return False
    if is_page_number(line):
        return False
    words = line.split()
    if _NUMBERED_HEADING.match(line) and len(words) <= 12:
        return True
    letters = [c for c in line if c.isalpha()]
    return len(letters) >= 4 and sum(c.isupper() for c in letters) / len(letters) > 0.8


def _split_long(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    pieces: list[str] = []
    current = ""
    for sentence in _SENTENCE_END.split(text):
        while len(sentence) > max_chars:  # one enormous "sentence": hard split at a space
            cut = sentence.rfind(" ", 0, max_chars)
            cut = cut if cut > max_chars // 2 else max_chars
            pieces.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()
        if current and len(current) + len(sentence) + 1 > max_chars:
            pieces.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        pieces.append(current)
    return pieces


def _paragraph_units(
    text: str, page: int | None, section: str | None
) -> tuple[list[Unit], str | None]:
    units: list[Unit] = []
    lines: list[str] = []

    def flush() -> None:
        if lines:
            paragraph = "\n".join(lines)
            units.append(Unit("text", paragraph, page, section))
            lines.clear()

    for raw in text.split("\n"):
        line = raw.strip()
        if not line:
            flush()
        elif is_heading(line):
            flush()
            section = line
        elif line.startswith("•") or not lines:
            lines.append(line)
        else:
            lines[-1] = f"{lines[-1]} {line}"  # rejoin a wrapped line
    flush()
    return units, section


def build_units(pages: list[Page], max_chars: int) -> list[Unit]:
    units: list[Unit] = []
    section: str | None = None
    for page in pages:
        for block in page.blocks:
            if block.kind == "table_row":
                # Rows carry their own context; a prose heading (often a table-of-contents
                # line) would mislabel them.
                units.extend(
                    Unit("table_row", t, page.number, None)
                    for t in _split_long(block.text, max_chars)
                )
                continue
            paragraph_units, section = _paragraph_units(block.text, page.number, section)
            for unit in paragraph_units:
                units.extend(
                    Unit(unit.kind, piece, unit.page, unit.section)
                    for piece in _split_long(unit.text, max_chars)
                )
    return units


def _chunk_id(doc_sha: str, index: int) -> str:
    return hashlib.sha256(f"{doc_sha}:{index}".encode()).hexdigest()[:24]


def chunk_document(
    pages: list[Page], record: DocumentRecord, config: ChunkerConfig | None = None
) -> list[Chunk]:
    config = config or ChunkerConfig()
    units = build_units(pages, config.max_chars)
    groups: list[list[Unit]] = []
    current: list[Unit] = []
    size = 0

    for unit in units:
        section_changed = (
            bool(current) and unit.section != current[-1].section and size >= config.min_chars * 4
        )
        if current and (size + len(unit.text) > config.target_chars or section_changed):
            groups.append(current)
            overlap: list[Unit] = []
            if not section_changed:
                overlap_size = 0
                for previous in reversed(current):
                    if (
                        previous.kind != "text"
                        or overlap_size + len(previous.text) > config.overlap_chars
                    ):
                        break
                    overlap.insert(0, previous)
                    overlap_size += len(previous.text)
            current, size = overlap, sum(len(u.text) for u in overlap)
        current.append(unit)
        size += len(unit.text) + 1
    if current:
        groups.append(current)

    chunks: list[Chunk] = []
    for group in groups:
        body = "\n".join(u.text for u in group).strip()
        if len(body) < config.min_chars:
            continue
        section = next((u.section for u in reversed(group) if u.section), None)
        header = record.title + (f" > {section}" if section else "")
        text = f"{header}\n{body}"
        pages_in_group = [u.page for u in group if u.page is not None]
        index = len(chunks)
        chunks.append(
            Chunk(
                chunk_id=_chunk_id(record.sha256, index),
                doc_id=record.doc_id,
                chunk_index=index,
                text=text,
                section=section,
                page_start=min(pages_in_group) if pages_in_group else None,
                page_end=max(pages_in_group) if pages_in_group else None,
                crops=sorted(set(record.crops) | set(detect_crops(body))),
                doc_type=record.doc_type,
                title=record.title,
                source_org=record.source_org,
                source_path=record.source_path,
                as_of=record.as_of,
                language=record.language,
                char_count=len(text),
                token_estimate=len(text) // _CHARS_PER_TOKEN + 1,
            )
        )
    return chunks
