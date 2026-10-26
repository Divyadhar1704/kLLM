"""Convert extracted tables into self-contained text rows.

A table row on its own ("Potato | Late blight | 100 | 500 | 19") is useless for retrieval. Each row
is rendered with its column headers and the labels of the section/product it belongs to:

    1. Fungicides single product formulations use > Amisulbrom 20% SC | Crop: Potato; ...

Government tables span many pages and repeat the header only on the first one, so state (headers by
column count, section and product labels) carries across tables of the same document.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.rag.ingestion.text import normalize_text

_PARENS = re.compile(r"\([^)]*\)")
_SECTION_LABEL = re.compile(r"^(?:\d+|[IVX]+|[A-Z])\.\s")
_ENUMERATOR = re.compile(r"^(?:\d{1,3}|[A-Za-z]|[ivxIVX]{1,4})[.)]$")
_HEADER_WORDS = re.compile(
    r"\b(?:name|crop|crops|pest|pests|disease|dose|dosage|period|formulation|dilution|"
    r"commodity|s\.\s*no|sl\.\s*no|application|purpose|weed|weeds|stage|a\.\s*i)\b",
    re.IGNORECASE,
)
_MAX_CONTEXT_CHARS = 90
_MAX_HEADER_CELL_CHARS = 70
_FILL_DOWN_COLUMNS = 2


@dataclass(slots=True)
class TableState:
    ncols: int = 0
    headers: list[str] | None = None
    headers_by_ncols: dict[int, list[str]] = field(default_factory=dict)
    section: str = ""
    label: str = ""
    previous: list[str] = field(default_factory=list)


def clean_cell(cell: str | None) -> str:
    if not cell:
        return ""
    text = normalize_text(cell)
    text = text.replace("\n•", " •").replace("\n", " ")
    return re.sub(r"\s{2,}", " ", text).strip()


def _looks_like_header(row: list[str], min_keyword_cells: int) -> bool:
    cells = [c for c in row if c]
    if len(cells) < 2:
        return False
    for cell in cells:
        if len(cell) > _MAX_HEADER_CELL_CHARS or cell.endswith("."):
            return False
        if re.search(r"\d", _PARENS.sub("", cell)):
            return False
    return sum(1 for c in cells if _HEADER_WORDS.search(c)) >= min_keyword_cells


def _remember_headers(state: TableState, headers: list[str]) -> None:
    state.headers = headers
    state.headers_by_ncols[len(headers)] = headers
    compact = [h for h in headers if h]
    # Later pages often drop the empty spacer columns of merged header cells.
    state.headers_by_ncols.setdefault(len(compact), compact)


def _render(cells: list[str], headers: list[str] | None, context: str) -> str:
    parts: list[str] = []
    for index, cell in enumerate(cells):
        if not cell or _ENUMERATOR.match(cell):
            continue
        header = headers[index] if headers and index < len(headers) else ""
        parts.append(f"{header}: {cell}" if header else cell)
    body = "; ".join(parts)
    return f"{context} | {body}" if context and body else body or context


def _context(state: TableState) -> str:
    return " > ".join(p for p in (state.section, state.label) if p)


def table_to_rows(
    table: list[list[str | None]], state: TableState | None = None
) -> tuple[list[str], TableState]:
    rows = [[clean_cell(c) for c in row] for row in table]
    rows = [row for row in rows if any(row)]
    state = state or TableState()
    if not rows:
        return [], state

    ncols = max(len(row) for row in rows)
    rows = [row + [""] * (ncols - len(row)) for row in rows]
    if ncols == 1:
        return [row[0] for row in rows], state

    if state.ncols != ncols:
        # New table shape: reuse headers seen earlier with this many columns; keep the labels,
        # because a product's rows often continue from the previous page.
        state.ncols = ncols
        state.headers = state.headers_by_ncols.get(ncols)
        state.previous = []

    output: list[str] = []
    header_continuation = False
    for row in rows:
        filled_indexes = [i for i, c in enumerate(row) if c]

        if len(filled_indexes) == 1:
            index = filled_indexes[0]
            cell = row[index]
            if index == 0 and ncols >= 3 and len(cell) <= _MAX_CONTEXT_CHARS:
                # Section or product label spanning the row.
                if _SECTION_LABEL.match(cell):
                    state.section, state.label = cell, ""
                else:
                    state.label = cell
                state.previous = []
                header_continuation = False
                continue
            if index > 0 and output and state.previous:
                # Text wrapped into its own row: continue the cell of the row above.
                state.previous[index] = f"{state.previous[index]} {cell}".strip()
                output[-1] = _render(state.previous, state.headers, _context(state))
                continue

        min_keywords = 1 if not output else 2
        if header_continuation and _looks_like_header(row, 0):
            assert state.headers is not None  # noqa: S101 - set on the previous header row
            merged = [
                " ".join(p for p in (h, c) if p) for h, c in zip(state.headers, row, strict=True)
            ]
            _remember_headers(state, merged)
            header_continuation = False
            continue
        if _looks_like_header(row, min_keywords):
            _remember_headers(state, row)
            header_continuation = True
            continue
        header_continuation = False

        # Merged cells: empty leading label columns inherit the row above.
        filled = list(row)
        for col in range(min(_FILL_DOWN_COLUMNS, ncols - 1)):
            if (
                not filled[col]
                and state.previous
                and state.previous[col]
                and any(filled[col + 1 :])
            ):
                filled[col] = state.previous[col]
        state.previous = filled
        rendered = _render(filled, state.headers, _context(state))
        if rendered:
            output.append(rendered)

    return output, state
