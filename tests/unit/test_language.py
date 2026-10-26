from __future__ import annotations

import pytest

from app.services.language import LANGUAGE_NAMES, SUPPORTED, detect_language
from app.services.prompts import build_system_prompt


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("What are the symptoms of early blight in tomato?", "en"),
        ("టమాటాలో ఎర్లీ బ్లైట్ లక్షణాలు ఏమిటి?", "te"),
        ("ಟೊಮೆಟೊದಲ್ಲಿ ಅರ್ಲಿ ಬ್ಲೈಟ್ ರೋಗದ ಲಕ್ಷಣಗಳು ಯಾವುವು?", "kn"),
        ("टमाटर में अगेती झुलसा रोग के लक्षण क्या हैं?", "hi"),
        # Mixed: native script with English product names stays native.
        ("Mancozeb ఎంత మోతాదులో పిచికారీ చేయాలి?", "te"),
        ("DAP और यूरिया कब डालें?", "hi"),
        # No letters at all.
        ("12345 ???", "en"),
        ("", "en"),
        # Romanized (Phase 12).
        ("tomato lo early blight lakshanalu emiti", "te"),
    ],
)
def test_detect_language(text: str, expected: str) -> None:
    assert detect_language(text) == expected


def test_supported_languages() -> None:
    assert SUPPORTED == ("en", "te", "kn", "hi")
    assert set(LANGUAGE_NAMES) == set(SUPPORTED)


@pytest.mark.parametrize("language", SUPPORTED)
def test_system_prompt_names_language(language: str) -> None:
    prompt = build_system_prompt(language)  # type: ignore[arg-type]
    assert f"Reply ONLY in {LANGUAGE_NAMES[language]}" in prompt  # type: ignore[index]
    assert "KVK" in prompt
    assert "Never apply one crop's dose to another crop" in prompt
