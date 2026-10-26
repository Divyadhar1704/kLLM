"""Pesticide safety check on generated answers, using the official CIB&RC restriction list.

Two kinds of warnings, appended to the answer in the farmer's language:
- banned:   banned / withdrawn / refused registration / banned-except-export products, always.
- crop:     products banned on specific crops ("Malathion is banned for use on ... Tomato"),
            only when the question is about one of those crops.

This is a deterministic safety net under the prompt rules: the model is told never to recommend
banned products, and this catches it if it does anyway (or if a farmer asks about one).
"""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import Iterable
from dataclasses import dataclass

from app.core.logging import get_logger
from app.db import Database
from app.rag.ingestion.metadata import detect_crops
from app.repositories.rag import load_restrictions
from app.services.language import LanguageCode

logger = get_logger(__name__)

BANNED_STATUSES = frozenset({"banned", "banned_except_export", "withdrawn", "refused_registration"})

VEGETABLES = frozenset(
    {"tomato", "brinjal", "chilli", "okra", "cabbage", "cauliflower", "onion", "potato",
     "cucumber", "gourds", "pumpkin", "cowpea"}
)  # fmt: skip
FRUITS = frozenset({"mango", "grape", "banana", "citrus", "pomegranate", "apple"})

# Native-script spellings (te / kn / hi) of banned or crop-restricted products that farmers
# actually encounter. Answers in Indian languages often write product names in that script only
# ("फोरेट" instead of "Phorate"); without these the check would miss them. Keyed by the lowercase
# name as it appears in the official list.
INDIC_PESTICIDE_NAMES: dict[str, tuple[str, ...]] = {
    "phorate": ("ఫోరేట్", "ಫೋರೇಟ್", "फोरेट", "फ़ोरेट"),
    "monocrotophos": ("మోనోక్రోటోఫాస్", "ಮೋನೋಕ್ರೋಟೋಫಾಸ್", "मोनोक्रोटोफॉस", "मोनोक्रोटोफास"),
    "endosulfron": ("ఎండోసల్ఫాన్", "ಎಂಡೋಸಲ್ಫಾನ್", "एंडोसल्फान", "एण्डोसल्फान"),
    "methyl parathion": ("మిథైల్ పారాథియాన్", "ಮಿಥೈಲ್ ಪ್ಯಾರಾಥಿಯಾನ್", "मिथाइल पैराथियान", "मिथाइल पैराथियॉन"),
    "dichlorovos": ("డైక్లోర్వాస్", "ಡೈಕ್ಲೋರ್ವಾಸ್", "डाइक्लोरवोस", "डीडीवीपी"),
    "triazophos": ("ట్రైజోఫాస్", "ಟ್ರೈಜೋಫಾಸ್", "ट्राइजोफॉस", "ट्रायजोफॉस"),
    "carbaryl": ("కార్బరిల్", "ಕಾರ್ಬರಿಲ್", "कार्बेरिल", "कार्बारिल"),
    "lindane": ("లిండేన్", "ಲಿಂಡೇನ್", "लिंडेन"),
    "phosphamidon": ("ఫాస్ఫామిడాన్", "ಫಾಸ್ಫಾಮಿಡಾನ್", "फॉस्फामिडॉन"),
    "methomyl": ("మెథోమిల్", "ಮೆಥೋಮಿಲ್", "मेथोमिल"),
    "dicofol": ("డైకోఫాల్", "ಡೈಕೋಫಾಲ್", "डाइकोफॉल"),
    "malathion": ("మలాథియాన్", "ಮ್ಯಾಲಥಿಯಾನ್", "ಮಲಾಥಿಯಾನ್", "मैलाथियान", "मेलाथियान"),
    "chlorpyriphos": (
        "క్లోరిపైరిఫాస్", "క్లోర్‌పైరిఫాస్", "ಕ್ಲೋರೋಪೈರಿಫಾಸ್", "ಕ್ಲೋರ್ಪೈರಿಫಾಸ್",
        "क्लोरपाइरीफॉस", "क्लोरपायरीफॉस",
    ),
    "dimethoate": ("డైమిథోయేట్", "డైమెథోయేట్", "ಡೈಮಿಥೋಯೇಟ್", "डाइमेथोएट", "डायमिथोएट"),
    "carbofuran": ("కార్బోఫ్యూరాన్", "ಕಾರ್ಬೋಫ್ಯುರಾನ್", "कार्बोफ्यूरान"),
}  # fmt: skip

_CROP_BAN_CLAUSE = re.compile(
    r"(?:banned|not permitted)\s+(?:for\s+use\s+)?(?:on|in)\s+([^.()]+)", re.IGNORECASE
)
_RELOAD_AFTER_FAILURE_SECONDS = 300.0

WARNINGS: dict[str, dict[LanguageCode, str]] = {
    "banned": {
        "en": "⚠️ Safety note: {names} is banned in India. Do not use it. Ask your local KVK "
        "or agriculture officer for an approved alternative.",
        "te": "⚠️ భద్రతా సూచన: {names} భారతదేశంలో నిషేధించబడింది. దీన్ని వాడకండి. ఆమోదించిన "
        "ప్రత్యామ్నాయం కోసం మీ దగ్గరలోని కృషి విజ్ఞాన కేంద్రం (KVK) లేదా వ్యవసాయ అధికారిని "
        "సంప్రదించండి.",
        "kn": "⚠️ ಸುರಕ್ಷತಾ ಸೂಚನೆ: {names} ಅನ್ನು ಭಾರತದಲ್ಲಿ ನಿಷೇಧಿಸಲಾಗಿದೆ. ಇದನ್ನು ಬಳಸಬೇಡಿ. "
        "ಅನುಮೋದಿತ ಪರ್ಯಾಯಕ್ಕಾಗಿ ನಿಮ್ಮ ಹತ್ತಿರದ ಕೃಷಿ ವಿಜ್ಞಾನ ಕೇಂದ್ರ (KVK) ಅಥವಾ ಕೃಷಿ "
        "ಅಧಿಕಾರಿಯನ್ನು ಸಂಪರ್ಕಿಸಿ.",
        "hi": "⚠️ सुरक्षा सूचना: {names} भारत में प्रतिबंधित है। इसका उपयोग न करें। अनुमोदित "
        "विकल्प के लिए अपने नज़दीकी कृषि विज्ञान केंद्र (KVK) या कृषि अधिकारी से संपर्क करें।",
    },
    "crop": {
        "en": "⚠️ Safety note: {names} is not permitted on this crop in India. Do not use it on "
        "this crop; ask your local KVK or agriculture officer for an approved alternative.",
        "te": "⚠️ భద్రతా సూచన: భారతదేశంలో ఈ పంటపై {names} వాడటానికి అనుమతి లేదు. ఈ పంటపై "
        "దీన్ని వాడకండి; ఆమోదించిన ప్రత్యామ్నాయం కోసం మీ దగ్గరలోని కృషి విజ్ఞాన కేంద్రం (KVK) "
        "లేదా వ్యవసాయ అధికారిని సంప్రదించండి.",
        "kn": "⚠️ ಸುರಕ್ಷತಾ ಸೂಚನೆ: ಭಾರತದಲ್ಲಿ ಈ ಬೆಳೆಯ ಮೇಲೆ {names} ಬಳಕೆಗೆ ಅನುಮತಿ ಇಲ್ಲ. ಈ "
        "ಬೆಳೆಗೆ ಇದನ್ನು ಬಳಸಬೇಡಿ; ಅನುಮೋದಿತ ಪರ್ಯಾಯಕ್ಕಾಗಿ ನಿಮ್ಮ ಹತ್ತಿರದ ಕೃಷಿ ವಿಜ್ಞಾನ ಕೇಂದ್ರ "
        "(KVK) ಅಥವಾ ಕೃಷಿ ಅಧಿಕಾರಿಯನ್ನು ಸಂಪರ್ಕಿಸಿ.",
        "hi": "⚠️ सुरक्षा सूचना: भारत में इस फसल पर {names} के उपयोग की अनुमति नहीं है। इस "
        "फसल पर इसका उपयोग न करें; अनुमोदित विकल्प के लिए अपने नज़दीकी कृषि विज्ञान केंद्र "
        "(KVK) या कृषि अधिकारी से संपर्क करें।",
    },
}


@dataclass(frozen=True, slots=True)
class Restriction:
    name: str
    aliases: tuple[str, ...]
    status: str
    details: str | None


@dataclass(frozen=True, slots=True)
class SafetyFinding:
    kind: str  # "banned" | "crop"
    name: str


def _term_pattern(term: str) -> re.Pattern[str]:
    escaped = re.escape(term.strip())
    escaped = re.sub(r"(?:\\ )+", r"[\\s-]*", escaped)  # flexible spaces / hyphens
    if len(term.strip()) <= 4:  # short codes (DDT, EDB, BHC): exact case, whole word
        return re.compile(rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])")
    return re.compile(rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])", re.IGNORECASE)


def _banned_crops(details: str | None) -> frozenset[str]:
    if not details:
        return frozenset()
    crops: set[str] = set()
    for clause in _CROP_BAN_CLAUSE.findall(details):
        crops.update(detect_crops(clause))
        lowered = clause.lower()
        if "vegetable" in lowered:
            crops |= VEGETABLES
        if "fruit" in lowered:
            crops |= FRUITS
    return frozenset(crops)


class PesticideSafetyChecker:
    def __init__(self, restrictions: Iterable[Restriction]) -> None:
        self._banned: list[tuple[str, list[re.Pattern[str]]]] = []
        self._crop_bans: list[tuple[str, list[re.Pattern[str]], frozenset[str]]] = []
        for r in restrictions:
            # "Thiodemeton / Disulfoton" lists two names in one entry.
            terms = [
                part.strip()
                for entry in (r.name, *r.aliases)
                for part in entry.split("/")
                if part.strip()
            ]
            terms += INDIC_PESTICIDE_NAMES.get(r.name.strip().lower(), ())
            patterns = [_term_pattern(t) for t in terms]
            if r.status in BANNED_STATUSES:
                self._banned.append((r.name, patterns))
            elif r.status == "restricted" and (crops := _banned_crops(r.details)):
                self._crop_bans.append((r.name, patterns, crops))

    @property
    def size(self) -> int:
        return len(self._banned) + len(self._crop_bans)

    def find(self, answer: str, question_crops: Iterable[str]) -> list[SafetyFinding]:
        findings: list[SafetyFinding] = []
        seen: set[str] = set()
        for name, patterns in self._banned:
            if name not in seen and any(p.search(answer) for p in patterns):
                findings.append(SafetyFinding("banned", name))
                seen.add(name)
        crops = set(question_crops)
        if crops:
            for name, patterns, banned_on in self._crop_bans:
                if (
                    name not in seen
                    and crops & banned_on
                    and any(p.search(answer) for p in patterns)
                ):
                    findings.append(SafetyFinding("crop", name))
                    seen.add(name)
        return findings

    @staticmethod
    def warnings(findings: list[SafetyFinding], language: LanguageCode) -> list[str]:
        messages: list[str] = []
        for kind in ("banned", "crop"):
            names = [f.name for f in findings if f.kind == kind]
            if names:
                messages.append(WARNINGS[kind][language].format(names=", ".join(names)))
        return messages


class PesticideSafety:
    """Loads the restriction list from the database once; retries later if loading failed."""

    def __init__(self, db: Database | None) -> None:
        self._db = db
        self._checker: PesticideSafetyChecker | None = None
        self._failed_at: float | None = None
        self._lock = asyncio.Lock()

    @property
    def checker(self) -> PesticideSafetyChecker | None:
        return self._checker

    async def ensure_loaded(self) -> PesticideSafetyChecker | None:
        if self._checker is not None or self._db is None:
            return self._checker
        if self._failed_at and time.monotonic() - self._failed_at < _RELOAD_AFTER_FAILURE_SECONDS:
            return None
        async with self._lock:
            if self._checker is not None:
                return self._checker
            try:
                async with asyncio.timeout(10), self._db.session() as session:
                    rows = await load_restrictions(session)
            except Exception as exc:
                self._failed_at = time.monotonic()
                logger.warning("pesticide_safety_load_failed", exc_info=exc)
                return None
            self._checker = PesticideSafetyChecker(
                Restriction(r.name, tuple(r.aliases or ()), r.status, r.details) for r in rows
            )
            logger.info("pesticide_safety_loaded", rules=self._checker.size)
            return self._checker
