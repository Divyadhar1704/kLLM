"""Detect crops mentioned in a farmer's question, in English, Telugu, Kannada or Hindi.

English uses word-boundary matching (app.rag.ingestion.metadata.CROP_ALIASES). Indic scripts
are agglutinative ("టమాటాలో" = "in tomato"), so names are matched as substrings, longest first,
and each match is blanked out so a longer name is never re-read as a shorter one
("మొక్కజొన్న" maize must not also count as "జొన్న" sorghum).

Deliberately omitted because they are common words with another meaning:
    Telugu పసుపు (also "yellow"), రాగి (also "copper"), టీ; Kannada ಹೆಸರು (also "name");
    Hindi आम (also "common/general").
"""

from __future__ import annotations

import re

from app.rag.ingestion.metadata import detect_crops

INDIC_CROP_NAMES: dict[str, tuple[str, ...]] = {
    "tomato": ("టమాటా", "టమోటా", "టొమాటో", "ಟೊಮೆಟೊ", "ಟೊಮ್ಯಾಟೊ", "ಟೊಮಾಟೊ", "ಟೊಮೇಟೊ", "टमाटर"),
    "brinjal": ("వంకాయ", "వంగ", "ಬದನೆ", "बैंगन"),
    "chilli": ("మిరప", "మిర్చి", "ಮೆಣಸಿನಕಾಯಿ", "ಮೆಣಸಿನ", "मिर्च"),
    "potato": ("బంగాళాదుంప", "బంగాళదుంప", "ఆలుగడ్డ", "ಆಲೂಗಡ್ಡೆ", "आलू"),
    "ginger": ("అల్లం", "ಶುಂಠಿ", "अदरक"),
    "turmeric": ("ಅರಿಶಿನ", "ಅರಿಶಿಣ", "हल्दी"),
    "redgram": ("కంది", "ತೊಗರಿ", "अरहर", "तुअर", "तूर"),
    "cowpea": ("అలసంద", "బొబ్బర్లు", "ಅಲಸಂದೆ", "लोबिया"),
    "blackgram": ("మినుము", "మినుములు", "ಉದ್ದು", "उड़द", "उड़द"),
    "greengram": ("పెసర", "పెసలు", "मूंग"),
    "bengalgram": ("శనగ", "సెనగ", "ಕಡಲೆ", "चना"),
    "groundnut": ("వేరుశనగ", "వేరుసెనగ", "పల్లీ", "ಶೇಂಗಾ", "ಕಡಲೆಕಾಯಿ", "मूंगफली"),
    "cucumber": ("దోసకాయ", "ಸೌತೆಕಾಯಿ", "ಸೌತೆ", "खीरा"),
    "pumpkin": ("గుమ్మడి", "ಕುಂಬಳ", "कद्दू"),
    "gourds": (
        "కాకరకాయ",
        "సొరకాయ",
        "బీరకాయ",
        "పొట్లకాయ",
        "ಹಾಗಲಕಾಯಿ",
        "ಸೋರೆಕಾಯಿ",
        "ಹೀರೆಕಾಯಿ",
        "लौकी",
        "करेला",
        "तोरई",
    ),
    "okra": ("బెండకాయ", "బెండ", "ಬೆಂಡೆಕಾಯಿ", "ಬೆಂಡೆ", "भिंडी"),
    "onion": ("ఉల్లిపాయ", "ఉల్లి", "ಈರುಳ್ಳಿ", "प्याज"),
    "cabbage": ("క్యాబేజీ", "ಎಲೆಕೋಸು", "पत्ता गोभी", "पत्तागोभी", "बंदगोभी", "बंद गोभी"),
    "cauliflower": ("కాలీఫ్లవర్", "కాలిఫ్లవర్", "ಹೂಕೋಸು", "फूलगोभी", "फूल गोभी"),
    "rice": ("వరి", "ಭತ್ತ", "धान", "चावल"),
    "wheat": ("గోధుమ", "ಗೋಧಿ", "गेहूं", "गेहूँ"),
    "maize": ("మొక్కజొన్న", "ಮೆಕ್ಕೆಜೋಳ", "मक्का"),
    "sorghum": ("జొన్న", "ಜೋಳ", "ज्वार"),
    "pearl_millet": ("సజ్జ", "ಸಜ್ಜೆ", "बाजरा"),
    "finger_millet": ("ರಾಗಿ", "रागी", "मंडुआ"),
    "cotton": ("పత్తి", "ಹತ್ತಿ", "कपास"),
    "soybean": ("సోయాబీన్", "సోయా", "ಸೋಯಾಬೀನ್", "ಸೋಯಾ", "सोयाबीन"),
    "mustard": ("ఆవాలు", "ಸಾಸಿವೆ", "सरसों"),
    "sugarcane": ("చెరకు", "ಕಬ್ಬು", "गन्ना"),
    "mango": ("మామిడి", "ಮಾವು"),
    "grape": ("ద్రాక్ష", "ದ್ರಾಕ್ಷಿ", "अंगूर"),
    "banana": ("అరటి", "ಬಾಳೆ", "केला"),
    "citrus": ("నిమ్మ", "బత్తాయి", "ನಿಂಬೆ", "नींबू", "संतरा"),
    "pomegranate": ("దానిమ్మ", "ದಾಳಿಂಬೆ", "अनार"),
    "coconut": ("కొబ్బరి", "ತೆಂಗು", "नारियल"),
    "tea": ("ಚಹಾ", "चाय"),
    "coffee": ("కాఫీ", "ಕಾಫಿ", "कॉफी"),
    "cumin": ("జీలకర్ర", "ಜೀರಿಗೆ", "जीरा"),
    "apple": ("ఆపిల్", "యాపిల్", "ಸೇಬು", "सेब"),
}

# Local crop names typed in English letters ("tamatar mein jhulsa", "vankaya lo purugu").
# Omitted: "vari" (Telugu also "their"), "patti" (Hindi "leaf"), "palli" (also "village").
ROMANIZED_CROP_NAMES: dict[str, tuple[str, ...]] = {
    "tomato": ("tamatar", "tamata", "tamaata", "tomota"),
    "brinjal": ("vankaya", "vankai", "baingan", "badane"),
    "chilli": ("mirapa", "mirapakaya", "mirchi", "menasinakai", "menasina"),
    "potato": ("aloo", "aalu", "bangaladumpa", "alugadde", "aloogadde"),
    # Plain "patti" is Hindi for "leaf"; with a Telugu case suffix it is cotton ("pattilo" = in
    # cotton).
    "cotton": (
        "patthi",
        "paththi",
        "kapas",
        "hatti",
        "pattilo",
        "pattiki",
        "pattini",
        "pattipanta",
    ),
    "rice": ("dhan", "dhaan", "bhatta", "bhatha"),
    "wheat": ("gehu", "gehun", "gehoon", "godhuma", "godhi"),
    "maize": ("makka", "mokkajonna", "makkajonna", "mekkejola"),
    "sorghum": ("jonna", "jola"),
    "redgram": ("kandi", "kandulu", "togari", "toor", "arhar"),
    "groundnut": ("verusenaga", "verusanaga", "shenga", "moongphali", "mungfali", "mungphali"),
    "bengalgram": ("senaga", "senagalu", "chana", "kadale"),
    "okra": ("bendakaya", "benda", "bende", "bhindi"),
    "onion": ("ullipaya", "ulli", "pyaz", "pyaj", "eerulli", "erulli"),
    "ginger": ("allam", "adrak", "shunti"),
    "sugarcane": ("cheraku", "ganna", "kabbu"),
    "mango": ("mamidi", "maavu"),
    "banana": ("arati", "kela", "baale"),
    "cucumber": ("dosakaya", "kheera", "southekai", "sautekai"),
    "cabbage": ("pattagobhi", "bandgobhi"),
    "cauliflower": ("phoolgobhi", "hookosu"),
}


def _romanized_pattern(names: tuple[str, ...]) -> re.Pattern[str]:
    # Names of 5+ letters may carry a suffix ("vankayalo" = in brinjal). Short names must be
    # whole words: "dhan" (paddy) must not match "dhanyavad" (thank you).
    alternatives = [rf"{n}\w*" if len(n) >= 5 else rf"{n}\b" for n in names]
    return re.compile(r"\b(?:" + "|".join(alternatives) + ")", re.IGNORECASE)


_ROMANIZED_PATTERNS: dict[str, re.Pattern[str]] = {
    crop: _romanized_pattern(names) for crop, names in ROMANIZED_CROP_NAMES.items()
}

# (name, crop) pairs, longest name first.
_INDIC_TERMS: list[tuple[str, str]] = sorted(
    ((name, crop) for crop, names in INDIC_CROP_NAMES.items() for name in names),
    key=lambda item: len(item[0]),
    reverse=True,
)


# Crop names used as ordinary words: Telugu "గోధుమ రంగు" (literally "wheat colour") is how
# farmers say BROWN ("brown spots"), not wheat. Removed before the crop names are matched.
_NOT_A_CROP = re.compile(r"గోధుమ\s*(?:రంగు|వర్ణ)|\bgodhuma\s*rang\w*", re.IGNORECASE)


def detect_query_crops(text: str) -> list[str]:
    text = _NOT_A_CROP.sub(" ", text)
    crops = set(detect_crops(text))
    crops.update(crop for crop, pattern in _ROMANIZED_PATTERNS.items() if pattern.search(text))
    remaining = text
    for name, crop in _INDIC_TERMS:
        if name in remaining:
            crops.add(crop)
            remaining = remaining.replace(name, " ")
    return sorted(crops)
