"""Text normalization and page-level cleaning."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter

# Private-use bullet glyphs from Word/Symbol fonts plus common bullet characters.
_BULLETS = re.compile(
    r"[\uf0b7\uf0a7\uf076\uf0d8\uf0fc\uf06e\uf0a8\u2022\u25cf\u25aa\u25a0\u2023\u2043\u27a2]"
)
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\u200b\ufeff]")
_SPACES = re.compile(r"[ \t\u00a0\u2002-\u200a]+")
# Only rejoin words split across lines ("applica-\ntion"). Never join digits: "600-\n800 g"
# is a dose range, and joining it would turn 600-800 g into 6,00,800 g.
_HYPHENATED_BREAK = re.compile(r"([^\W\d_]{2})-\n([a-z])")
_MANY_NEWLINES = re.compile(r"\n{3,}")
_PAGE_NUMBER = re.compile(
    r"^(?:page\s*)?[-\u2013]?\s*\d{1,4}\s*[-\u2013]?(?:\s*(?:of|/)\s*\d{1,4})?$", re.IGNORECASE
)

_DEVANAGARI = (0x0900, 0x097F)
# A run of Devanagari words plus an optional "/" separator, as in "भारत सरकार/Government of India".
_DEVANAGARI_RUN = re.compile(r"[\u0900-\u097f]+(?:[\s\u0900-\u097f]*[\u0900-\u097f])?\s*/?")


def normalize_text(text: str) -> str:
    """Unicode NFKC, unified bullets, no control characters, single spaces, stripped lines."""
    text = unicodedata.normalize("NFKC", text)
    text = _BULLETS.sub("•", text)
    text = _CONTROL.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [_SPACES.sub(" ", line).strip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = _HYPHENATED_BREAK.sub(r"\1\2", text)
    return _MANY_NEWLINES.sub("\n\n", text).strip()


def is_page_number(line: str) -> bool:
    return bool(_PAGE_NUMBER.match(line.strip()))


def script_share(text: str, block: tuple[int, int]) -> float:
    letters = [c for c in text if c.isalpha() or block[0] <= ord(c) <= block[1]]
    if not letters:
        return 0.0
    return sum(1 for c in letters if block[0] <= ord(c) <= block[1]) / len(letters)


def repeated_lines(
    pages: list[list[str]], *, min_pages: int = 4, threshold: float = 0.5
) -> set[str]:
    """Lines that appear on at least `threshold` of pages: running headers and footers."""
    if len(pages) < min_pages:
        return set()
    counts: Counter[str] = Counter()
    for lines in pages:
        counts.update({line for line in lines if line and len(line) <= 120})
    limit = threshold * len(pages)
    return {line for line, count in counts.items() if count >= limit}


def clean_page_texts(texts: list[str], *, drop_devanagari_lines: bool) -> list[str]:
    """Clean the prose of every page of one document.

    Removes running headers/footers, page numbers and, for English documents, the Devanagari
    half of bilingual government headers (PDF font encoding garbles it on extraction).
    """
    pages = [normalize_text(t).split("\n") for t in texts]
    boilerplate = repeated_lines(pages)
    cleaned: list[str] = []
    for lines in pages:
        kept: list[str] = []
        for line in lines:
            if line in boilerplate or is_page_number(line):
                continue
            if drop_devanagari_lines:
                line = _DEVANAGARI_RUN.sub(" ", line)
                line = _SPACES.sub(" ", line).strip(" /")
                if not line:
                    continue
            kept.append(line)
        cleaned.append(_MANY_NEWLINES.sub("\n\n", "\n".join(kept)).strip())
    return cleaned
