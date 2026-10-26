"""Prompts. The system prompt is static per language (cacheable); retrieved sources go into the
user turn. Kept short: every prompt token adds latency and cost on every request."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

from app.rag.retriever import RetrievedChunk
from app.services.language import LANGUAGE_NAMES, LanguageCode

_SYSTEM_PROMPT = """You are KrishiMitra, an agricultural assistant for farmers in India.

Rules:
- Reply ONLY in {language_name}, in its native script, even if the question is typed in English \
letters. Use simple words a farmer understands. Keep product and chemical names recognizable.
- Be practical and brief: short paragraphs or numbered steps, under 200 words unless the question \
truly needs more. No greetings or self-introductions; start directly with the answer.
- If SOURCES are provided, they are official Government of India documents. Take product names, \
doses, dilution and waiting periods ONLY from them and cite them like [1] (always digits 0-9 \
inside the brackets). Only numbered sources are cited: never put anything else in brackets, such \
as the photo analysis or the conversation. If the sources do not \
cover the question, say so in one sentence, then give cautious general advice.
- Doses are crop-specific. Use a product or dose from a source ONLY for the crop that source \
names. Never apply one crop's dose to another crop; if no source covers the farmer's crop, say \
that clearly and advise the KVK instead of borrowing a dose.
- If a PHOTO ANALYSIS is given (of a photo, or of frames from the farmer's video), it was made \
automatically and can be wrong. Result "confident": name the problem directly. Result \
"uncertain": say it MAY be the most likely problem, mention the other possibilities, and suggest \
a clearer photo or video, confirming the crop, or showing the plant to the KVK. Never claim \
certainty from a photo or video alone. Healthy: say no disease is visible.
- Diseases and pests: give symptoms, cause, prevention and treatment. Prefer cultural, biological \
and integrated pest management (IPM) methods first.
- Chemicals: name only active ingredients approved for use in India. Never invent doses; if no \
source gives a dose, tell the farmer to follow the product label. Always remind the farmer to \
follow the label and wear protective gear. Never recommend banned pesticides.
- If you are unsure, or the problem needs a field visit or lab test, say so and advise contacting \
the local Krishi Vigyan Kendra (KVK) or agriculture officer.
- Answer only agriculture-related questions (crops, soil, water, fertilizer, pests, diseases, \
livestock, weather impact, markets, farmer schemes). Politely decline other topics.
- Never invent statistics, scheme amounts, prices or phone numbers."""

_MIN_SOURCE_CHARS = 200

# Changes whenever the system prompt changes; part of the answer-cache key.
PROMPT_VERSION = hashlib.sha256(_SYSTEM_PROMPT.encode()).hexdigest()[:12]


def build_system_prompt(language: LanguageCode) -> str:
    return _SYSTEM_PROMPT.format(language_name=LANGUAGE_NAMES[language])


def language_reminder(language: LanguageCode) -> str:
    """Appended to the question when a first answer came back in the wrong language."""
    return (
        f"\n\nIMPORTANT: Your previous answer was not in {LANGUAGE_NAMES[language]}. "
        f"Write the entire answer in {LANGUAGE_NAMES[language]} script only."
    )


def source_label(index: int, chunk: RetrievedChunk) -> str:
    hit = chunk.hit
    parts = [hit.title]
    if hit.page_start:
        pages = (
            f"p. {hit.page_start}"
            if hit.page_end in (None, hit.page_start)
            else f"pp. {hit.page_start}-{hit.page_end}"
        )
        parts.append(pages)
    if hit.as_of:
        parts.append(f"as of {hit.as_of.isoformat()}")
    return f"[{index}] " + ", ".join(parts)


def detected_context(crops: Sequence[str], terms: Sequence[str]) -> str:
    """What the farmer is asking about, from local names in the question (may be empty)."""
    parts = []
    if crops:
        parts.append("crop: " + ", ".join(c.replace("_", " ") for c in crops))
    if terms:
        parts.append("pest/disease: " + ", ".join(terms))
    if not parts:
        return ""
    return (
        "DETECTED FROM THE QUESTION: " + "; ".join(parts) + ". Answer for this crop and "
        "problem only; ignore source rows about other crops."
    )


def build_user_message(
    question: str,
    sources: Sequence[RetrievedChunk],
    *,
    max_chars: int,
    crops: Sequence[str] = (),
    terms: Sequence[str] = (),
    extra_context: str = "",
) -> tuple[str, int]:
    """Question with numbered sources. Returns (message, number of sources included).

    `extra_context` (e.g. the photo analysis) is placed between the sources and the question.
    """
    context = "\n\n".join(p for p in (extra_context, detected_context(crops, terms)) if p)
    if not sources:
        return (f"{context}\n\n{question}" if context else question), 0
    blocks: list[str] = []
    budget = max_chars
    for chunk in sources:
        if budget < _MIN_SOURCE_CHARS:
            break
        # The stored text starts with a "title > section" header line; the label replaces it.
        header, _, body = chunk.hit.content.partition("\n")
        section = header.split(" > ", 1)[1] if " > " in header else None
        label = source_label(len(blocks) + 1, chunk) + (f" | {section}" if section else "")
        body = body.strip() or header
        if len(body) > budget:
            body = body[:budget].rsplit(" ", 1)[0] + " ..."
        blocks.append(f"{label}\n{body}")
        budget -= len(body)
    message = (
        "SOURCES:\n\n"
        + "\n\n".join(blocks)
        + (f"\n\n{context}" if context else "")
        + f"\n\nQUESTION (answer in the required language):\n{question}"
    )
    return message, len(blocks)
