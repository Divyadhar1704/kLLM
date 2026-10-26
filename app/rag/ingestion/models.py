"""Ingestion data structures.

Extraction produces pages of blocks. A block is either prose ("text") or one self-contained table
row rendered as text ("table_row"). Chunks and document records are pydantic models because they
are persisted as JSON.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

BlockKind = Literal["text", "table_row"]


@dataclass(frozen=True, slots=True)
class Block:
    kind: BlockKind
    text: str


@dataclass(slots=True)
class Page:
    number: int | None  # None for formats without pages (docx, txt)
    blocks: list[Block] = field(default_factory=list)


@dataclass(slots=True)
class ExtractedDocument:
    path: Path
    pages: list[Page]
    title_hint: str | None = None
    extractor: str = ""


class DocumentRecord(BaseModel):
    doc_id: str
    sha256: str
    source_path: str
    title: str
    doc_type: str
    source_org: str | None = None
    as_of: str | None = None
    crops: list[str] = []
    language: str = "en"
    pages: int = 0
    chunks: int = 0
    chars: int = 0
    extractor: str = ""


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    chunk_index: int
    text: str  # context header + body: this is what gets embedded and shown to the LLM
    section: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    crops: list[str] = []
    doc_type: str
    title: str
    source_org: str | None = None
    source_path: str
    as_of: str | None = None
    language: str = "en"
    char_count: int
    token_estimate: int
