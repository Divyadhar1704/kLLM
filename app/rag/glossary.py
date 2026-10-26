"""Local names of pests and diseases -> the English terms used in the official documents.

Farmers say "తెల్ల దోమ", "tella doma", "सफेद मक्खी" or "ಬಿಳಿ ನೊಣ"; the CIB&RC and IPM documents say
"whitefly". Detected terms are used to:
- add English keywords to the full-text search, so questions in Telugu / Kannada / Hindi also
  get exact-name matching (otherwise keyword search only sees Latin-script words);
- tell the model which pest/disease the farmer means, so it does not guess
  (e.g. "tella doma" is whitefly, not brown planthopper).

Native-script names are matched as substrings (longest first, then blanked), romanized names as
whole phrases. Only names with one clear agricultural meaning are listed.
"""

from __future__ import annotations

import re

# English term (as written in the documents) -> local names (te / kn / hi, native and romanized).
PEST_DISEASE_NAMES: dict[str, tuple[str, ...]] = {
    "whitefly": (
        "తెల్లదోమ", "తెల్ల దోమ", "tella doma", "telladoma",
        "ಬಿಳಿ ನೊಣ", "ಬಿಳಿನೊಣ", "bili nona",
        "सफेद मक्खी", "सफ़ेद मक्खी", "safed makhi", "safed makkhi",
    ),
    "aphid": ("పేనుబంక", "penubanka", "ಸಸ್ಯ ಹೇನು", "माहू", "mahu", "चेपा"),
    "thrips": ("తామర పురుగు", "tamara purugu", "थ्रिप्स", "ಥ್ರಿಪ್ಸ್"),
    # "ಕಾಯಿ" / "కాయ" mean both pod and fruit: pod borer in pulses, fruit borer in vegetables.
    "pod borer": (
        "కాయ తొలుచు పురుగు", "kaya tolichu purugu", "ಕಾಯಿ ಕೊರಕ", "kaayi koraka", "kayi koraka",
        "फली छेदक", "phali chhedak",
    ),
    "fruit borer": (
        "కాయ తొలుచు పురుగు", "kaya tolichu purugu", "ಕಾಯಿ ಕೊರಕ", "kaayi koraka", "kayi koraka",
        "फल छेदक", "phal chhedak",
    ),
    "shoot and fruit borer": (
        "కొమ్మ మరియు కాయ తొలుచు పురుగు", "కొమ్మ కాయ తొలుచు పురుగు",
        "ಕಾಂಡ ಮತ್ತು ಕಾಯಿ ಕೊರಕ", "ಕಾಂಡ ಮತ್ತು ಹಣ್ಣು ಕೊರಕ",
        "तना एवं फल छेदक", "तना और फल छेदक", "tana aur phal chhedak",
    ),
    "stem borer": (
        "కాండం తొలుచు పురుగు", "kandam tolichu purugu", "ಕಾಂಡ ಕೊರಕ", "kanda koraka",
        "तना छेदक", "tana chhedak",
    ),
    "fall army worm": ("కత్తెర పురుగు", "kattera purugu", "फॉल आर्मीवर्म"),
    "brown plant hopper": ("సుడిదోమ", "sudidoma", "भूरा फुदका", "bhura phudka"),
    "mealy bug": ("పిండి నల్లి", "pindi nalli", "मिलीबग"),
    "leaf curl": ("ఆకు ముడత", "aaku mudata", "पत्ती मरोड़", "ಎಲೆ ಮುದುಡು"),
    "blast": ("అగ్గి తెగులు", "aggi tegulu", "ಬೆಂಕಿ ರೋಗ", "benki roga", "झोंका"),
    "late blight": ("पछेती झुलसा", "pacheti jhulsa", "पिछेती झुलसा"),
    "early blight": ("अगेती झुलसा", "ageti jhulsa"),
    "yellow rust": ("पीला रतुआ", "peela ratua", "pila ratua"),
    "powdery mildew": ("బూడిద తెగులు", "boodida tegulu", "ಬೂದಿ ರೋಗ", "चूर्णिल आसिता"),
    "downy mildew": ("బూజు తెగులు", "booju tegulu", "मृदुरोमिल आसिता"),
    "wilt": ("ఎండు తెగులు", "endu tegulu", "ಸೊರಗು ರೋಗ", "उकठा", "ukatha", "मुरझान"),
    "leaf spot": ("ఆకుమచ్చ", "ఆకు మచ్చ", "aaku machha", "पत्ती धब्बा", "ಎಲೆ ಚುಕ್ಕೆ"),
    "tikka": ("తిక్క", "टिक्का"),
    "rhizome rot": ("దుంప కుళ్ళు", "ಗಡ್ಡೆ ಕೊಳೆ", "प्रकंद सड़न"),
}  # fmt: skip


def _name_to_terms(native: bool) -> dict[str, set[str]]:
    mapping: dict[str, set[str]] = {}
    for term, names in PEST_DISEASE_NAMES.items():
        for name in names:
            if name.isascii() != native:
                mapping.setdefault(name.lower(), set()).add(term)
    return mapping


# Native names: longest first, so a full phrase ("ಕಾಂಡ ಮತ್ತು ಕಾಯಿ ಕೊರಕ") wins over its parts.
_NATIVE_NAMES: list[tuple[str, set[str]]] = sorted(
    _name_to_terms(native=True).items(), key=lambda item: len(item[0]), reverse=True
)
_ROMANIZED_NAMES: list[tuple[re.Pattern[str], set[str]]] = [
    (re.compile(rf"\b{re.escape(name)}\b", re.IGNORECASE), terms)
    for name, terms in sorted(
        _name_to_terms(native=False).items(), key=lambda item: len(item[0]), reverse=True
    )
]


def detect_terms(text: str) -> list[str]:
    """English pest/disease terms for local names found in the text (sorted, unique)."""
    found: set[str] = set()
    remaining = text
    for name, terms in _NATIVE_NAMES:
        if name in remaining:
            found |= terms
            remaining = remaining.replace(name, " ")
    for pattern, terms in _ROMANIZED_NAMES:
        if pattern.search(remaining):
            found |= terms
            remaining = pattern.sub(" ", remaining)
    return sorted(found)
