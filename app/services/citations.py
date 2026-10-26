"""Citation markers: ASCII digits only, and only numbered sources.

- Models writing Kannada/Telugu/Hindi sometimes localise the digits in citation markers
  ("[೫]" / "[౩]" / "[२, ४]"), which breaks linking a citation to `sources[n]` in the app:
  they become "[5]" / "[3]" / "[2, 4]". Only digits inside short bracketed citations are
  changed; numbers in the answer text (doses, days) are left as written.
- Models also invent labels that are not sources, e.g. "[ಫೋಟೋ ವಿಶ್ಲೇಷಣೆ]" / "[PHOTO ANALYSIS]"
  for the photo result; the prompt forbids it but the small model does not always obey. Short
  bracketed text without any digit is removed. Bracketed text with a number ("[2 g/l]") stays.

`CitationStreamNormalizer` gives exactly the same result for streamed text, even when a marker
is split across chunks ("[" in one delta, "೫]" in the next).
"""

from __future__ import annotations

import re

_INDIC_DIGITS = "೦೧೨೩೪೫೬೭೮೯౦౧౨౩౪౫౬౭౮౯०१२३४५६७८९"
_ASCII_DIGITS = "0123456789" * 3
_TRANSLATE = str.maketrans(_INDIC_DIGITS, _ASCII_DIGITS)

_DIGIT = f"[0-9{_INDIC_DIGITS}]"
_CITATION = re.compile(rf"\[({_DIGIT}(?:{_DIGIT}|[,\s]){{0,10}})\]")
_MAX_LABEL = 40
_FAKE_CITATION = re.compile(rf"[ \t]?\[[^\[\]\n0-9{_INDIC_DIGITS}]{{1,{_MAX_LABEL}}}\]")
# Longest marker that may still be completed: optional space + "[" + label + "]".
_MAX_MARKER_LENGTH = _MAX_LABEL + 3


def normalize_citations(text: str) -> str:
    text = _FAKE_CITATION.sub("", text)
    return _CITATION.sub(lambda m: "[" + m.group(1).translate(_TRANSLATE) + "]", text)


class CitationStreamNormalizer:
    """Feed text chunks; emits normalized text, holding back a possibly unfinished marker."""

    def __init__(self) -> None:
        self._pending = ""

    def feed(self, chunk: str) -> str:
        text = self._pending + chunk
        self._pending = ""
        start = text.rfind("[")
        if start != -1 and "]" not in text[start:] and len(text) - start < _MAX_MARKER_LENGTH:
            # Possibly the start of a marker: wait for the closing bracket. The space before
            # it is held back too, because a removed label takes its leading space with it.
            if start > 0 and text[start - 1] in " \t":
                start -= 1
            text, self._pending = text[:start], text[start:]
        elif text.endswith((" ", "\t")):
            # A label may start in the next chunk and must remove this space as well.
            text, self._pending = text[:-1], text[-1]
        return normalize_citations(text)

    def flush(self) -> str:
        text, self._pending = self._pending, ""
        return normalize_citations(text)
