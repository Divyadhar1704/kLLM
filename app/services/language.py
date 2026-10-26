"""Language handling for en / te / kn / hi.

Detection, in order:
1. Script: Telugu, Kannada and Devanagari have their own Unicode blocks (instant, reliable).
2. Romanized: Latin-script text is checked for common Telugu / Kannada / Hindi function words
   ("tomato lo early blight ki mandu enti", "tamatar mein jhulsa ka ilaj kya hai").
3. Otherwise English.

Answers are always written in the language's native script.

`language_quality` measures whether an answer is really in the requested language, and counts
characters from unrelated Indic scripts (e.g. a Tamil word inside a Telugu answer).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal, get_args

LanguageCode = Literal["en", "te", "kn", "hi"]
DetectionMethod = Literal["script", "romanized", "default"]

SUPPORTED: tuple[LanguageCode, ...] = get_args(LanguageCode)

LANGUAGE_NAMES: dict[LanguageCode, str] = {
    "en": "English",
    "te": "Telugu (తెలుగు)",
    "kn": "Kannada (ಕನ್ನಡ)",
    "hi": "Hindi (हिन्दी)",
}

SCRIPT_BLOCKS: dict[LanguageCode, tuple[int, int]] = {
    "te": (0x0C00, 0x0C7F),
    "kn": (0x0C80, 0x0CFF),
    "hi": (0x0900, 0x097F),  # Devanagari
}

# Other Indic scripts that should never appear in a te/kn/hi/en answer.
_OTHER_INDIC_BLOCKS: tuple[tuple[int, int], ...] = (
    (0x0980, 0x09FF),  # Bengali
    (0x0A00, 0x0A7F),  # Gurmukhi
    (0x0A80, 0x0AFF),  # Gujarati
    (0x0B00, 0x0B7F),  # Odia
    (0x0B80, 0x0BFF),  # Tamil
    (0x0D00, 0x0D7F),  # Malayalam
)

# Share of letters that must be in an Indic script to classify the text as that language.
# Lower than 0.5 because farmers mix in English crop and product names.
_MIN_SCRIPT_SHARE = 0.3

# Frequent romanized function words / farming words. Words shared with English or with another
# of these languages are left out ("ki" is Hindi and Telugu; "me" and "in" are English).
ROMANIZED_MARKERS: dict[LanguageCode, frozenset[str]] = {
    "te": frozenset(
        {
            "lo",
            "emiti",
            "enti",
            "ela",
            "elaa",
            "undi",
            "unnayi",
            "ledu",
            "leda",
            "cheyali",
            "cheyyali",
            "vadali",
            "vaadali",
            "mandu",
            "mandulu",
            "enduku",
            "evi",
            "emi",
            "yemi",
            "kosam",
            "ante",
            "chala",
            "chaala",
            "ekkuva",
            "takkuva",
            "vastundi",
            "vachindi",
            "avutundi",
            "ayindi",
            "nivarana",
            "rogam",
            "tegulu",
            "purugu",
            "purugulu",
            "aakulu",
            "aaku",
            "panta",
            "pantalo",
            "meeru",
            "nenu",
            "koncham",
            "baaga",
            "bagundi",
            "cheppandi",
            "cheyandi",
            "pichikari",
            "vesthe",
            "veyali",
            "kavali",
            "unte",
        }
    ),
    "kn": frozenset(
        {
            "alli",
            "enu",
            "yenu",
            "hege",
            "hegey",
            "beku",
            "maadabeku",
            "madabeku",
            "maadi",
            "madi",
            "ide",
            "ive",
            "illa",
            "yaava",
            "yavudu",
            "bele",
            "beleya",
            "roga",
            "rogakke",
            "hulu",
            "keeta",
            "aushadhi",
            "oushadhi",
            "bagge",
            "mattu",
            "nimma",
            "nanna",
            "eshtu",
            "agide",
            "agutte",
            "bandide",
            "simpadane",
            "sinchane",
            "tilisi",
            "yaake",
            "yake",
            "kodi",
            "helabeku",
            "elegalu",
            "ele",
        }
    ),
    "hi": frozenset(
        {
            "mein",
            "ka",
            "ke",
            "hai",
            "hain",
            "kya",
            "kaise",
            "kyon",
            "kyu",
            "karna",
            "karein",
            "kare",
            "karen",
            "dawa",
            "dawai",
            "dava",
            "rog",
            "fasal",
            "kheti",
            "kitna",
            "kitni",
            "kab",
            "nahi",
            "nahin",
            "aur",
            "liye",
            "lagta",
            "laga",
            "gaya",
            "patti",
            "pattiyan",
            "keet",
            "keede",
            "kide",
            "upay",
            "ilaj",
            "ilaaj",
            "bimari",
            "beemari",
            "chhidkav",
            "chidkav",
            "khad",
            "paani",
            "batao",
            "bataiye",
            "kijiye",
            "konsi",
            "kaun",
            "jhulsa",
        }
    ),
}
_LATIN_WORD = re.compile(r"[A-Za-z]+")


def _script_counts(text: str) -> tuple[dict[LanguageCode, int], int]:
    counts: dict[LanguageCode, int] = dict.fromkeys(SUPPORTED, 0)
    letters = 0
    for char in text:
        code_point = ord(char)
        in_indic = False
        for language, (start, end) in SCRIPT_BLOCKS.items():
            if start <= code_point <= end:
                counts[language] += 1
                in_indic = True
                break
        if in_indic:
            letters += 1
        elif char.isalpha():
            letters += 1
            if char.isascii():
                counts["en"] += 1
    return counts, letters


def detect_romanized(text: str) -> LanguageCode | None:
    words = [w.lower() for w in _LATIN_WORD.findall(text)]
    if not words:
        return None
    scores = {
        lang: sum(1 for w in words if w in markers) for lang, markers in ROMANIZED_MARKERS.items()
    }
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    (best, best_score), (_, second_score) = ranked[0], ranked[1]
    if best_score == 0 or best_score == second_score:
        return None
    # Two markers, or one marker in a very short message ("tomato lo blight").
    if best_score >= 2 or len(words) <= 4:
        return best
    return None


def detect_language_with_method(text: str) -> tuple[LanguageCode, DetectionMethod]:
    counts, letters = _script_counts(text)
    if letters:
        indic: tuple[LanguageCode, ...] = ("te", "kn", "hi")
        best = max(indic, key=lambda lang: counts[lang])
        if counts[best] / letters >= _MIN_SCRIPT_SHARE:
            return best, "script"
    romanized = detect_romanized(text)
    if romanized is not None:
        return romanized, "romanized"
    return "en", "default"


def detect_language(text: str) -> LanguageCode:
    return detect_language_with_method(text)[0]


@dataclass(frozen=True, slots=True)
class LanguageQuality:
    target_share: float  # share of letters written in the target language's script
    foreign_script_chars: int  # characters from unrelated Indic scripts (e.g. Tamil)
    ok: bool


def language_quality(text: str, target: LanguageCode) -> LanguageQuality:
    counts, letters = _script_counts(text)
    foreign = sum(
        1 for char in text if any(start <= ord(char) <= end for start, end in _OTHER_INDIC_BLOCKS)
    )
    if target != "en":
        # Another of te/kn/hi in the answer is also foreign.
        foreign += sum(counts[lang] for lang in ("te", "kn", "hi") if lang != target)
    share = counts[target] / letters if letters else 0.0
    # Indic answers legitimately contain English product names; English answers should not
    # contain much Indic text.
    threshold = 0.6 if target == "en" else 0.4
    # Letters from another script inside a word ("క్లోరోథాలനില്‍") make it unreadable for the
    # farmer, even when the rest of the answer is fine: fail so the answer is regenerated.
    return LanguageQuality(
        target_share=round(share, 3),
        foreign_script_chars=foreign,
        ok=share >= threshold and foreign == 0,
    )
