"""Phase 12: romanized detection, romanized crops, citation digits, answer-language check."""

from __future__ import annotations

import json
import random

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app
from app.rag.crops import detect_query_crops
from app.services.citations import CitationStreamNormalizer, normalize_citations
from app.services.language import detect_language_with_method, language_quality
from tests.fakes import FakeLLM

# ---------------------------------------------------------------- detection


@pytest.mark.parametrize(
    ("text", "language", "method"),
    [
        ("tomato lo early blight ki mandu enti", "te", "romanized"),
        ("pattilo tella doma ela nivarana cheyali", "te", "romanized"),
        ("tamatar mein jhulsa rog ka ilaj kya hai", "hi", "romanized"),
        ("gehu ki fasal mein khad kab dalna hai", "hi", "romanized"),
        ("tomato alli early blight roga ide, enu maadabeku", "kn", "romanized"),
        ("bele ge yaava aushadhi beku", "kn", "romanized"),
        ("tomato lo blight", "te", "romanized"),  # one marker in a short message
        ("What is the dose of mancozeb for tomato?", "en", "default"),
        ("How to control fall armyworm in maize?", "en", "default"),
        ("Is it OK to spray in the evening?", "en", "default"),
        ("టమాటాలో ఎర్లీ బ్లైట్", "te", "script"),  # native script always wins
        ("मिर्च में thrips control", "hi", "script"),
    ],
)
def test_detect_language_with_method(text: str, language: str, method: str) -> None:
    assert detect_language_with_method(text) == (language, method)


def test_ambiguous_romanized_falls_back_to_english() -> None:
    # One Hindi and one Telugu marker in a long sentence: no clear winner.
    text = "please tell me about the mandu for this disease and fasal management today"
    assert detect_language_with_method(text) == ("en", "default")


@pytest.mark.parametrize(
    ("text", "crops"),
    [
        ("vankaya lo purugu ela", ["brinjal"]),
        ("vankayalo kaya tolichu purugu", ["brinjal"]),  # suffix on a long name
        ("tamatar aur baingan mein keet", ["brinjal", "tomato"]),
        ("kapas mein safed makkhi", ["cotton"]),
        ("dhan mein bhura phudka", ["rice"]),
        ("dhanyavad sir", []),  # short name "dhan" must not match inside "dhanyavad"
        ("aloo ki patti peeli", ["potato"]),  # "patti" (leaf) is not cotton
        ("pattilo tella doma nivaranaku e mandu vadali", ["cotton"]),  # Telugu suffix = cotton
        ("mirchi lo tamara purugu", ["chilli"]),
    ],
)
def test_romanized_crops(text: str, crops: list[str]) -> None:
    assert detect_query_crops(text) == crops


@pytest.mark.parametrize(
    ("text", "crops"),
    [
        # "గోధుమ రంగు" = brown (literally "wheat colour"): not the wheat crop.
        ("నా టమాటా ఆకులపై గోధుమ రంగు మచ్చలు", ["tomato"]),
        ("ఆకులపై గోధుమరంగు మచ్చలు", []),
        ("tomato aakulapai godhuma rangu machalu", ["tomato"]),
        # The crop itself is still detected.
        ("గోధుమ పంటలో తుప్పు తెగులు", ["wheat"]),
        ("గోధుమ పంటపై గోధుమ రంగు మచ్చలు", ["wheat"]),
        ("godhuma panta", ["wheat"]),
    ],
)
def test_brown_is_not_wheat(text: str, crops: list[str]) -> None:
    assert detect_query_crops(text) == crops


# ---------------------------------------------------------------- citations


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("ಸಿಂಪಡಿಸಿ [೫].", "ಸಿಂಪಡಿಸಿ [5]."),
        ("వాడండి [౩]", "వాడండి [3]"),
        ("छिड़काव करें [२, ४]।", "छिड़काव करें [2, 4]।"),
        ("[1, 2, 4]", "[1, 2, 4]"),
        ("೫೦೦ ಗ್ರಾಂ [೧]", "೫೦೦ ಗ್ರಾಂ [1]"),  # dose digits outside brackets untouched
        # Labels without a number are not sources: removed with their leading space.
        ("see [note] and [a1]", "see and [a1]"),
        ("ತುಕ್ಕು ರೋಗ ಆಗಿರಬಹುದು [ಫೋಟೋ ವಿಶ್ಲೇಷಣೆ].", "ತುಕ್ಕು ರೋಗ ಆಗಿರಬಹುದು."),
        ("ಹಳದಿ [ಕೋಷ್ಟಕ: PHOTO ANALYSIS] [1].", "ಹಳದಿ [1]."),
        ("dose [2 g/l] here", "dose [2 g/l] here"),  # has a number: kept
    ],
)
def test_normalize_citations(raw: str, expected: str) -> None:
    assert normalize_citations(raw) == expected


def stream_normalize(chunks: list[str]) -> str:
    normalizer = CitationStreamNormalizer()
    return "".join(normalizer.feed(c) for c in chunks) + normalizer.flush()


def test_stream_normalizer_handles_split_citations() -> None:
    assert stream_normalize(["ಸಿಂಪಡಿಸಿ [", "೫", "]. ಮತ್ತೆ"]) == "ಸಿಂಪಡಿಸಿ [5]. ಮತ್ತೆ"
    assert stream_normalize(["text [೧, ", "೨] end"]) == "text [1, 2] end"
    assert stream_normalize(["unclosed [೩"]) == "unclosed [೩"  # never closed: left as is
    assert stream_normalize(["ರೋಗ ", "[ಫೋಟೋ ", "ವಿಶ್ಲೇಷಣೆ]", " ಇದೆ"]) == "ರೋಗ ಇದೆ"


def test_stream_normalizer_equals_full_text_for_any_chunking() -> None:
    text = (
        "ರೋಗ ನಿಯಂತ್ರಣ [೧]. Propiconazole 25% EC [२, ४] — 500 ml [3] [note] "
        "ವಿವರ [೫] ಮತ್ತು [ ] [೧೦] [ಫೋಟೋ ವಿಶ್ಲೇಷಣೆ] [PHOTO ANALYSIS]  two  spaces\t[tab] end [x" * 3
    )
    expected = normalize_citations(text)
    rng = random.Random(42)  # noqa: S311 - deterministic test data
    for _ in range(200):
        cuts = sorted(rng.sample(range(1, len(text)), rng.randint(1, 30)))
        chunks = [text[a:b] for a, b in zip([0, *cuts], [*cuts, len(text)], strict=True)]
        assert stream_normalize(chunks) == expected


# ---------------------------------------------------------------- answer language


def test_language_quality() -> None:
    telugu = "టమాటాలో ఎర్లీ బ్లైట్ నివారణకు Mancozeb 75% WP వాడండి [1]."
    assert language_quality(telugu, "te").ok
    english = "Spray mancozeb 75% WP at 2 g per litre."
    assert not language_quality(english, "te").ok
    assert language_quality(english, "en").ok
    mixed = "టమాటాలో ముதிர்ந்த ఆకులపై మచ్చలు"  # contains a Tamil word
    quality = language_quality(mixed, "te")
    assert not quality.ok  # a foreign-script word is unreadable: regenerate
    assert quality.foreign_script_chars > 0
    # Seen live: Malayalam letters inside a Telugu pesticide name.
    leaked = "లేదా క్లోరోథాలనില്‍ 75% డబ్ల్యూపీ: 350-500 గ్రాములు పిచికారీ చేయాలి [4]."
    assert not language_quality(leaked, "te").ok
    hindi_in_kannada = "ಎಲೆಗಳ ಮೇಲೆ ಕಲೆಗಳು ಕಾಣಿಸುತ್ತವೆ, मैंकोजेब ಸಿಂಪಡಿಸಿ"
    assert not language_quality(hindi_in_kannada, "kn").ok


# ---------------------------------------------------------------- chat integration


def client_for(llm: FakeLLM) -> TestClient:
    app = create_app(Settings(_env_file=None, enable_dev_ui=True), llm=llm)  # type: ignore[call-arg]
    return TestClient(app)


def test_wrong_language_answer_is_retried_once() -> None:
    llm = FakeLLM(outcomes=["Spray mancozeb as per label.", "లేబుల్ ప్రకారం మాంకోజెబ్ పిచికారీ చేయండి."])
    body = (
        client_for(llm)
        .post("/api/v1/chat", json={"message": "tomato lo blight ki mandu enti", "debug": True})
        .json()
    )
    assert body["language"] == "te"
    assert body["response"].startswith("లేబుల్")
    assert len(llm.requests) == 2
    assert "IMPORTANT" in llm.requests[1].messages[0].text
    check = body["debug"]["language_check"]
    assert check["detection_method"] == "romanized"
    assert check["retried"] is True
    assert check["answer_script_share"] > 0.9


def test_retry_kept_only_if_better() -> None:
    llm = FakeLLM(outcomes=["English answer one.", "English answer two."])
    body = client_for(llm).post("/api/v1/chat", json={"message": "q", "language": "hi"}).json()
    assert body["response"] == "English answer one."  # retry was not better
    assert len(llm.requests) == 2


def test_correct_language_is_not_retried() -> None:
    llm = FakeLLM(outcomes=["Use resistant varieties."])
    body = (
        client_for(llm)
        .post("/api/v1/chat", json={"message": "How to control blight?", "debug": True})
        .json()
    )
    assert len(llm.requests) == 1
    assert body["debug"]["language_check"]["retried"] is False
    assert body["debug"]["language_check"]["detection_method"] == "default"


def test_explicit_language_method() -> None:
    llm = FakeLLM(outcomes=["ಉತ್ತರ"])
    body = (
        client_for(llm)
        .post("/api/v1/chat", json={"message": "hello", "language": "kn", "debug": True})
        .json()
    )
    assert body["debug"]["language_check"]["detection_method"] == "explicit"


def test_citation_digits_normalized_in_blocking_and_stream() -> None:
    answer = "ಬೆಂಕಿ ರೋಗಕ್ಕೆ ಸಿಂಪಡಿಸಿ [೫]."
    client = client_for(FakeLLM(outcomes=[answer]))
    blocking = client.post("/api/v1/chat", json={"message": "q", "language": "kn"}).json()
    assert blocking["response"] == "ಬೆಂಕಿ ರೋಗಕ್ಕೆ ಸಿಂಪಡಿಸಿ [5]."

    deltas: list[str] = []
    done = None
    with client.stream("POST", "/api/v1/chat/stream", json={"message": "q", "language": "kn"}) as r:
        event = None
        for line in r.iter_lines():
            if line.startswith("event: "):
                event = line[7:]
            elif line.startswith("data: "):
                data = json.loads(line[6:])
                if event == "delta":
                    deltas.append(data["text"])
                elif event == "done":
                    done = data
    assert done is not None
    assert "".join(deltas) == done["response"] == "ಬೆಂಕಿ ರೋಗಕ್ಕೆ ಸಿಂಪಡಿಸಿ [5]."
