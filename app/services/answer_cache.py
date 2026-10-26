"""Answer cache: identical questions get the stored answer instantly (no embedding, database or
LLM call, no quota used). Farmers in one season ask the same questions many times.

- Key: normalised question (Unicode NFKC, case, spacing, trailing punctuation) + requested
  language + model + prompt fingerprint. Changing the model or the prompt invalidates entries.
- Only clean answers are stored: retrieval succeeded and the answer was not truncated.
- Entries expire after a TTL so knowledge-base updates show up.
- In-memory, per server instance. Multi-instance deployments move this to Redis (same interface).
"""

from __future__ import annotations

import hashlib
import re
import time
import unicodedata
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass

from app.schemas.chat import ChatResponse

_SPACES = re.compile(r"\s+")
_TRAILING_PUNCTUATION = re.compile(r"[\s?!.,;:।॥]+$")


def normalize_question(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    text = _SPACES.sub(" ", text).strip()
    return _TRAILING_PUNCTUATION.sub("", text)


@dataclass(slots=True)
class _Entry:
    response: ChatResponse
    stored_at: float


class AnswerCache:
    def __init__(
        self,
        *,
        max_items: int,
        ttl_seconds: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._max = max_items
        self._ttl = ttl_seconds
        self._clock = clock
        self._items: OrderedDict[str, _Entry] = OrderedDict()
        self.hits = 0
        self.misses = 0

    @property
    def enabled(self) -> bool:
        return self._max > 0 and self._ttl > 0

    @staticmethod
    def key(message: str, language: str | None, model: str, prompt_version: str) -> str:
        raw = "\x1f".join((normalize_question(message), language or "auto", model, prompt_version))
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, key: str) -> ChatResponse | None:
        if not self.enabled:
            return None
        entry = self._items.get(key)
        if entry is None or self._clock() - entry.stored_at > self._ttl:
            if entry is not None:
                del self._items[key]
            self.misses += 1
            return None
        self._items.move_to_end(key)
        self.hits += 1
        return entry.response

    def put(self, key: str, response: ChatResponse) -> None:
        if not self.enabled:
            return
        self._items[key] = _Entry(response=response, stored_at=self._clock())
        self._items.move_to_end(key)
        while len(self._items) > self._max:
            self._items.popitem(last=False)

    def clear(self) -> None:
        self._items.clear()

    def __len__(self) -> int:
        return len(self._items)
