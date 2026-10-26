"""Parse the CIB&RC "banned, refused registration and restricted" list into structured records.

Used later as a hard safety check on generated answers (never recommend a banned product), which
is more reliable than relying on retrieval or the prompt alone.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel
from pypdf import PdfReader

logging.getLogger("pypdf").setLevel(logging.ERROR)

Status = Literal[
    "banned",
    "banned_except_export",
    "withdrawn",
    "refused_registration",
    "restricted",
]

# Spelling errors in the official list and common alternate names.
KNOWN_ALIASES: dict[str, tuple[str, ...]] = {
    "endosulfron": ("Endosulfan",),
    "dichlorovos": ("Dichlorvos", "DDVP"),
    "dichlorvos": ("DDVP",),
    "dicohlro diphenyl trichloroethane": ("Dichloro Diphenyl Trichloroethane", "DDT"),
    "benzene hexachloride": ("BHC", "HCH"),
    "nicotin sulfate": ("Nicotine Sulfate", "Nicotine Sulphate"),
}


class PesticideRestriction(BaseModel):
    name: str
    normalized: str
    aliases: list[str] = []
    status: Status
    details: str | None = None


_MAJOR = re.compile(r"^(I{1,3}|IV)\.\s+(.*)$")
_SUB = re.compile(r"^([A-D])\.\s*(.*)$")
_ITEM = re.compile(r"^(\d{1,3})\.\s*(.*)$")
_NAME_END = re.compile(
    r"\s*\(\s*(?:S\.\s*O|ad\s*-?\s*interim|banned)|\s+(?:The|All|Use of|i\.)\s", re.IGNORECASE
)
_PAREN_ALIAS = re.compile(r"\(([^()]{2,30})\)")


def normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()


_SUBSECTION_STATUS: dict[str, Status] = {
    "A": "banned",
    "B": "banned_except_export",
    "C": "withdrawn",
}


def _status_for(major: str, sub: str | None) -> Status | None:
    if major == "I":
        return _SUBSECTION_STATUS.get(sub or "")
    if major == "II":
        return "refused_registration"
    if major == "III":
        return "restricted"
    return None


def _split_name(item: str) -> tuple[str, str | None]:
    """Split "<name> <details>" at the earliest name boundary."""
    cuts: list[int] = []
    # Restricted entries repeat the name at the start of the details: "Malathion Malathion is ..."
    words = item.split(" ")
    for k in range(1, min(5, len(words))):
        if words[k] and words[k].lower() == words[0].lower():
            cuts.append(len(" ".join(words[:k])))
            break
    if match := _NAME_END.search(item):
        cuts.append(match.start())
    if not cuts:
        return item.strip(), None
    cut = min(cuts)
    return item[:cut].strip(), item[cut:].strip() or None


def _make_record(item: str, status: Status) -> PesticideRestriction | None:
    item = re.sub(r"\s+", " ", item).strip().rstrip("*").strip()
    if not item:
        return None
    name, details = _split_name(item)
    aliases = [a.strip() for a in _PAREN_ALIAS.findall(name) if "S.O" not in a]
    base = _PAREN_ALIAS.sub("", name).strip()
    normalized = normalize_name(base)
    aliases += list(KNOWN_ALIASES.get(normalized, ()))
    return PesticideRestriction(
        name=base,
        normalized=normalized,
        aliases=sorted(set(aliases)),
        status=status,
        details=details if status == "restricted" else None,
    )


def parse_banned_list(text: str) -> list[PesticideRestriction]:
    records: list[PesticideRestriction] = []
    major: str | None = None
    sub: str | None = None
    current: list[str] | None = None
    current_status: Status | None = None
    expected = 1

    def flush() -> None:
        nonlocal current
        if current is not None and current_status is not None:
            record = _make_record(" ".join(current), current_status)
            if record is not None:
                records.append(record)
        current = None

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("*"):
            continue
        if match := _MAJOR.match(line):
            flush()
            major, sub, expected = match.group(1), None, 1
            continue
        if major == "I" and (match := _SUB.match(line)) and len(match.group(2)) < 120:
            flush()
            sub, expected = match.group(1), 1
            continue
        if (match := _ITEM.match(line)) and int(match.group(1)) == expected:
            flush()
            current_status = _status_for(major or "", sub)
            current = [match.group(2)] if match.group(2) else []
            expected += 1
            continue
        if current is not None:
            current.append(line)
    flush()
    return records


def parse_banned_pdf(path: Path) -> list[PesticideRestriction]:
    reader = PdfReader(path)
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    return parse_banned_list(text)
