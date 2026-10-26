"""Format-specific extractors. Each returns an ExtractedDocument of pages and blocks.

PDF:  pdfplumber (tables become table_row blocks, text outside tables becomes a text block);
      pypdf is the fallback if pdfplumber fails on a file.
DOCX: parsed directly from the OOXML (no extra dependency); tables become table_row blocks.
TXT/MD: read as one page.
Scanned (image-only) PDFs produce empty pages; they are reported and need OCR (not yet needed).
"""

from __future__ import annotations

import logging
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import pdfplumber
from pypdf import PdfReader

from app.core.logging import get_logger
from app.rag.ingestion.models import Block, ExtractedDocument, Page
from app.rag.ingestion.tables import TableState, table_to_rows

logger = get_logger(__name__)

# pdfminer/pypdf log a warning for every malformed object in government PDFs.
for _noisy in ("pdfminer", "pypdf"):
    logging.getLogger(_noisy).setLevel(logging.ERROR)

Extractor = Callable[[Path], ExtractedDocument]

_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


class ExtractionError(Exception):
    pass


# ---------------------------------------------------------------- PDF


def _center_inside(obj: dict[str, Any], boxes: list[tuple[float, float, float, float]]) -> bool:
    if "x0" not in obj or "top" not in obj:
        return False
    cx = (obj["x0"] + obj["x1"]) / 2
    cy = (obj["top"] + obj["bottom"]) / 2
    return any(x0 <= cx <= x1 and top <= cy <= bottom for x0, top, x1, bottom in boxes)


def extract_pdf(path: Path) -> ExtractedDocument:
    try:
        return _extract_pdf_plumber(path)
    except Exception as exc:
        logger.warning("pdfplumber_failed_fallback_pypdf", path=str(path), error=repr(exc)[:200])
        return _extract_pdf_pypdf(path)


def _extract_pdf_plumber(path: Path) -> ExtractedDocument:
    pages: list[Page] = []
    state: TableState | None = None
    with pdfplumber.open(path) as pdf:
        title = (pdf.metadata or {}).get("Title")
        for number, page in enumerate(pdf.pages, start=1):
            tables = page.find_tables()
            boxes = [t.bbox for t in tables]
            if boxes:

                def outside_tables(
                    obj: dict[str, Any], b: list[tuple[float, float, float, float]] = boxes
                ) -> bool:
                    return not _center_inside(obj, b)

                text = page.filter(outside_tables).extract_text() or ""
            else:
                text = page.extract_text() or ""
            blocks = [Block("text", text)] if text.strip() else []
            for table in tables:
                rows, state = table_to_rows(table.extract(), state)
                blocks.extend(Block("table_row", row) for row in rows)
            pages.append(Page(number=number, blocks=blocks))
            page.flush_cache()
    return ExtractedDocument(
        path=path,
        pages=pages,
        title_hint=title if isinstance(title, str) else None,
        extractor="pdfplumber",
    )


def _extract_pdf_pypdf(path: Path) -> ExtractedDocument:
    try:
        reader = PdfReader(path)
    except Exception as exc:
        raise ExtractionError(f"Cannot open PDF: {exc}") from exc
    pages = []
    for number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append(Page(number=number, blocks=[Block("text", text)] if text.strip() else []))
    title = reader.metadata.title if reader.metadata else None
    return ExtractedDocument(path=path, pages=pages, title_hint=title, extractor="pypdf")


# ---------------------------------------------------------------- DOCX


def _paragraph_text(element: ElementTree.Element) -> str:
    return "".join(node.text or "" for node in element.iter(f"{_W}t"))


def extract_docx(path: Path) -> ExtractedDocument:
    try:
        with zipfile.ZipFile(path) as archive:
            xml = archive.read("word/document.xml")
    except (zipfile.BadZipFile, KeyError) as exc:
        raise ExtractionError(f"Not a valid .docx file: {exc}") from exc

    body = ElementTree.fromstring(xml).find(f"{_W}body")  # noqa: S314 - local trusted files
    blocks: list[Block] = []
    paragraphs: list[str] = []
    if body is not None:
        for element in body:
            if element.tag == f"{_W}p":
                paragraphs.append(_paragraph_text(element))
            elif element.tag == f"{_W}tbl":
                if any(p.strip() for p in paragraphs):
                    blocks.append(Block("text", "\n".join(paragraphs)))
                paragraphs = []
                table: list[list[str | None]] = [
                    [
                        "\n".join(_paragraph_text(p) for p in cell.iter(f"{_W}p"))
                        for cell in row.iter(f"{_W}tc")
                    ]
                    for row in element.iter(f"{_W}tr")
                ]
                rows, _ = table_to_rows(table)
                blocks.extend(Block("table_row", r) for r in rows)
    if any(p.strip() for p in paragraphs):
        blocks.append(Block("text", "\n".join(paragraphs)))
    return ExtractedDocument(path=path, pages=[Page(number=None, blocks=blocks)], extractor="docx")


# ---------------------------------------------------------------- plain text


def extract_text_file(path: Path) -> ExtractedDocument:
    text = path.read_text(encoding="utf-8", errors="replace")
    blocks = [Block("text", text)] if text.strip() else []
    return ExtractedDocument(path=path, pages=[Page(number=None, blocks=blocks)], extractor="text")


EXTRACTORS: dict[str, Extractor] = {
    ".pdf": extract_pdf,
    ".docx": extract_docx,
    ".txt": extract_text_file,
    ".md": extract_text_file,
}


def extract(path: Path) -> ExtractedDocument:
    extractor = EXTRACTORS.get(path.suffix.lower())
    if extractor is None:
        raise ExtractionError(f"Unsupported file type: {path.suffix}")
    return extractor(path)
