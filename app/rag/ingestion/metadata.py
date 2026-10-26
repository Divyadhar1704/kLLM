"""Document and chunk metadata inference: document type, source, date, crops, title."""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path

# Canonical crop name -> spellings and local names found in Indian agricultural documents.
CROP_ALIASES: dict[str, tuple[str, ...]] = {
    "tomato": ("tomato", "tomatoes"),
    "brinjal": ("brinjal", "eggplant", "egg plant", "aubergine"),
    "chilli": ("chilli", "chillies", "chili", "chilies", "capsicum"),
    "potato": ("potato", "potatoes"),
    "ginger": ("ginger",),
    "turmeric": ("turmeric",),
    "redgram": ("redgram", "red gram", "pigeonpea", "pigeon pea", "arhar"),
    "cowpea": ("cowpea",),
    "blackgram": ("blackgram", "black gram", "urdbean"),
    "greengram": ("greengram", "green gram", "moong", "mungbean"),
    "bengalgram": ("bengal gram", "chickpea", "chick pea"),
    "cucumber": ("cucumber",),
    "pumpkin": ("pumpkin",),
    "gourds": (
        "bottle gourd",
        "bitter gourd",
        "sponge gourd",
        "snake gourd",
        "ash gourd",
        "ridge gourd",
        "cucurbits",
        "cucurbitaceous",
    ),
    "okra": ("okra", "bhindi", "lady's finger", "ladies finger"),
    "onion": ("onion",),
    "cabbage": ("cabbage",),
    "cauliflower": ("cauliflower",),
    "rice": ("rice", "paddy"),
    "wheat": ("wheat",),
    "maize": ("maize",),
    "sorghum": ("sorghum", "jowar"),
    "pearl_millet": ("pearl millet", "bajra"),
    "finger_millet": ("finger millet", "ragi"),
    "cotton": ("cotton",),
    "groundnut": ("groundnut", "peanut"),
    "soybean": ("soybean", "soyabean"),
    "mustard": ("mustard", "rapeseed"),
    "sugarcane": ("sugarcane",),
    "mango": ("mango",),
    "grape": ("grape", "grapes"),
    "banana": ("banana",),
    "citrus": ("citrus", "lemon", "orange", "mandarin"),
    "pomegranate": ("pomegranate",),
    "coconut": ("coconut",),
    "tea": ("tea",),
    "coffee": ("coffee",),
    "cumin": ("cumin",),
    "apple": ("apple",),
}

_CROP_PATTERNS: dict[str, re.Pattern[str]] = {
    crop: re.compile(r"\b(?:" + "|".join(re.escape(a) for a in aliases) + r")\b", re.IGNORECASE)
    for crop, aliases in CROP_ALIASES.items()
}

_DATE_AFTER_PHRASE = re.compile(
    r"(?:as\s+on|updated\s+on|as\s+of)\s*[:\-]?\s*(\d{1,2})\s*[./-]\s*(\d{1,2})\s*[./-]\s*(\d{4})",
    re.IGNORECASE,
)
_DATE_ANYWHERE = re.compile(r"\b(\d{1,2})\.(\d{1,2})\.(20\d{2})\b")

GOVT_PESTICIDE_SOURCE = (
    "Directorate of Plant Protection, Quarantine & Storage (CIB&RC), "
    "Ministry of Agriculture & Farmers Welfare, Govt. of India"
)
GOVT_IPM_SOURCE = (
    "Ministry of Agriculture & Farmers Welfare (AESA-based IPM package), Govt. of India"
)

DOC_TYPE_LABELS: dict[str, str] = {
    "pesticide_banned_list": "Banned / restricted pesticides list",
    "pesticide_use_guide": "Major uses of pesticides (approved crop, pest and dose)",
    "pesticide_registry": "Registered pesticides list",
    "ipm_package": "Integrated Pest Management package",
    "advisory": "Advisory",
    "general": "Document",
}


def detect_crops(text: str) -> list[str]:
    return sorted(crop for crop, pattern in _CROP_PATTERNS.items() if pattern.search(text))


def _filename_words(filename: str) -> str:
    return re.sub(r"[_\s]+", " ", filename).strip()


def classify_doc_type(filename: str, first_pages_text: str) -> str:
    name = _filename_words(filename).lower()
    head = first_pages_text.lower()
    if "banned" in name:
        return "pesticide_banned_list"
    if "mup" in name or "major uses of pesticides" in head:
        return "pesticide_use_guide"
    if any(
        key in name
        for key in (
            "registered",
            "list of pesticides",
            "approved source",
            "formulation approved",
            "desert locust",
        )
    ):
        return "pesticide_registry"
    if "aesa based ipm" in head or "aesa-based ipm" in head:
        return "ipm_package"
    if "control measures" in name or "advisory" in name:
        return "advisory"
    return "general"


def source_org_for(doc_type: str) -> str | None:
    if doc_type.startswith("pesticide_"):
        return GOVT_PESTICIDE_SOURCE
    if doc_type == "ipm_package":
        return GOVT_IPM_SOURCE
    return None


def _valid_date(day: str, month: str, year: str) -> str | None:
    try:
        return date(int(year), int(month), int(day)).isoformat()
    except ValueError:
        return None


def detect_as_of(filename: str, first_pages_text: str) -> str | None:
    filename = _filename_words(filename)
    for source in (first_pages_text, filename):
        match = _DATE_AFTER_PHRASE.search(source)
        if match and (iso := _valid_date(*match.groups())):
            return iso
    match = _DATE_ANYWHERE.search(filename)
    if match:
        return _valid_date(*match.groups())
    return None


def _split_camel(text: str) -> str:
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)


def make_title(path: Path, doc_type: str, title_hint: str | None) -> str:
    stem = path.stem
    if doc_type == "ipm_package":
        return f"AESA based IPM - {_split_camel(stem).strip().title()}"
    hint = (title_hint or "").strip()
    if 8 <= len(hint) <= 150 and not hint.lower().startswith(("microsoft", "untitled", "document")):
        return hint
    title = re.sub(r"^\d+\._", "", stem)  # list numbering prefix: "2._chemical_mup..."
    title = title.replace("_", " ")
    title = re.sub(r"\bmup\b", "Major Uses of Pesticides", title, flags=re.IGNORECASE)
    title = re.sub(r"\s+(?:\d|[a-z])$", "", title.strip())  # trailing version suffix: "_0", "_c"
    title = _split_camel(re.sub(r"\s{2,}", " ", title)).strip()
    return title[:1].upper() + title[1:] if title else path.name
