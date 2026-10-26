from __future__ import annotations

import pytest

from app.rag.glossary import detect_terms
from app.rag.retriever import crop_keywords, keyword_query_for
from app.services.prompts import detected_context


@pytest.mark.parametrize(
    ("text", "terms"),
    [
        ("పత్తిలో తెల్లదోమ నివారణ ఎలా?", ["whitefly"]),
        ("pattilo tella doma nivaranaku e mandu vadali", ["whitefly"]),
        ("कपास में सफेद मक्खी का इलाज", ["whitefly"]),
        ("safed makkhi kaise roke", ["whitefly"]),
        ("ಹತ್ತಿಯಲ್ಲಿ ಬಿಳಿ ನೊಣ", ["whitefly"]),
        ("మిరపలో తామర పురుగు", ["thrips"]),
        ("ತೊಗರಿಯಲ್ಲಿ ಕಾಯಿ ಕೊರಕ ಹುಳು", ["fruit borer", "pod borer"]),  # ambiguous: both
        ("ಬದನೆಯಲ್ಲಿ ಕಾಂಡ ಮತ್ತು ಕಾಯಿ ಕೊರಕದ ನಿರ್ವಹಣೆ", ["shoot and fruit borer"]),  # phrase wins
        ("बैंगन में तना एवं फल छेदक", ["shoot and fruit borer"]),
        ("मक्का में फॉल आर्मीवर्म", ["fall army worm"]),
        ("ಭತ್ತದಲ್ಲಿ ಬೆಂಕಿ ರೋಗ", ["blast"]),
        ("आलू में पछेती झुलसा", ["late blight"]),
        ("gehu mein peela ratua", ["yellow rust"]),
        ("వేరుశనగలో తిక్క ఆకుమచ్చ తెగులు", ["leaf spot", "tikka"]),
        ("What fertilizer for rice?", []),
        ("", []),
    ],
)
def test_detect_terms(text: str, terms: list[str]) -> None:
    assert detect_terms(text) == terms


def test_keyword_query_with_extra_terms() -> None:
    query = keyword_query_for("పత్తిలో తెల్లదోమ", ["whitefly", *crop_keywords(["cotton"])])
    assert query == 'whitefly or cotton or "white fly"'
    assert crop_keywords(["redgram"]) == ["redgram", "red gram", "pigeonpea"]


def test_detected_context() -> None:
    assert detected_context([], []) == ""
    context = detected_context(["cotton"], ["whitefly"])
    assert context.startswith("DETECTED FROM THE QUESTION: crop: cotton; pest/disease: whitefly.")
    assert "other crops" in context
