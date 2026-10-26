"""Photo and video chat: diagnosis + retrieval + answer (fakes; no AI or database calls)."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app
from app.rag.retriever import RetrievalResult
from app.services.image_chat import crop_from_text, photo_context, retrieval_query
from app.services.safety import PesticideSafetyChecker, Restriction
from app.vision.diagnosis import Alternative, Diagnosis
from app.vision.images import ImageQuality
from app.vision.taxonomy import condition
from app.vision.video import VideoInfo
from tests.fakes import FakeLLM
from tests.unit.test_chat_rag import FakeRetriever, chunk

URL = "/api/v1/chat/image"
PHOTO = b"\xff\xd8 fake jpeg bytes"
QUALITY = ImageQuality(width=800, height=600, brightness=120, sharpness=500)


VIDEO_INFO = VideoInfo(
    duration_seconds=10.0,
    width=1920,
    height=1080,
    container="mp4",
    codec="h264",
    has_audio=True,
    frames_examined=10,
    frame_times=(1.0, 4.0, 6.0, 9.0),
)


def diagnosis(
    status: str = "confident",
    condition_id: str = "tomato_early_blight",
    video: VideoInfo | None = None,
) -> Diagnosis:
    return Diagnosis(
        status=status,  # type: ignore[arg-type]
        crop=condition(condition_id).crop if condition_id != "unknown" else "unknown",
        condition=condition(condition_id),
        confidence=0.82 if status == "confident" else 0.5,
        model_confidence=0.82,
        severity="moderate",
        plant_part="leaf",
        visible_symptoms=["concentric rings on older leaves"],
        alternatives=[Alternative(condition("tomato_target_spot"), 0.1)],
        quality=QUALITY,
        model="fake-vision",
        video=video,
    )


class FakeDiagnoser:
    def __init__(self, result: Diagnosis) -> None:
        self.result = result
        self.calls: list[dict[str, Any]] = []

    async def diagnose(
        self, data: bytes, *, note: str | None = None, crop: str | None = None
    ) -> Diagnosis:
        self.calls.append({"kind": "image", "bytes": len(data), "note": note, "crop": crop})
        return self.result

    async def diagnose_video(
        self, data: bytes, *, note: str | None = None, crop: str | None = None
    ) -> Diagnosis:
        self.calls.append({"kind": "video", "bytes": len(data), "note": note, "crop": crop})
        return self.result


class FakeSafety:
    async def ensure_loaded(self) -> PesticideSafetyChecker:
        return PesticideSafetyChecker([Restriction("Phorate", (), "banned", None)])


def build(
    result: Diagnosis,
    llm: FakeLLM | None = None,
    retrieval: RetrievalResult | None = None,
    **settings: object,
) -> tuple[TestClient, FakeLLM, FakeDiagnoser, FakeRetriever]:
    llm = llm or FakeLLM(outcomes=["Early blight. Spray mancozeb as per label [1]."])
    fake_diagnoser = FakeDiagnoser(result)
    app = create_app(
        Settings(_env_file=None, enable_dev_ui=True, **settings),  # type: ignore[call-arg, arg-type]
        llm=llm,
        diagnoser=fake_diagnoser,  # type: ignore[arg-type]
    )
    retriever = FakeRetriever(retrieval or RetrievalResult(chunks=[chunk("c1", "Spray mancozeb")]))
    app.state.retriever = retriever
    app.state.safety = FakeSafety()
    return TestClient(app), llm, fake_diagnoser, retriever


def post(client: TestClient, **form: str) -> Any:
    return client.post(URL, files={"file": ("leaf.jpg", PHOTO, "image/jpeg")}, data=form or None)


# ---------------------------------------------------------------- helpers


@pytest.mark.parametrize(
    ("text", "crop"),
    [
        ("నా టమాటా ఆకులపై మచ్చలు", "tomato"),
        ("tamatar ke patte", "tomato"),
        ("what is this?", None),
        ("tomato and potato leaves", None),  # two crops: no guess
    ],
)
def test_crop_from_text(text: str, crop: str | None) -> None:
    assert crop_from_text(text) == crop


def test_retrieval_query_includes_condition_terms() -> None:
    query = retrieval_query(diagnosis(), "spots on leaves")
    assert query.startswith("Tomato Early blight early blight alternaria management")
    assert query.endswith("spots on leaves")
    uncertain = retrieval_query(diagnosis("uncertain"), "")
    assert "Target spot" in uncertain  # runner-up searched too when uncertain


def test_photo_context() -> None:
    text = photo_context(diagnosis(), crop_confirmed=True)
    assert "PHOTO ANALYSIS (automatic, can be wrong)" in text
    assert "crop: Tomato (confirmed by the farmer)" in text
    assert "result: confident" in text
    assert "most likely: Early blight (confidence 0.82)" in text
    assert "other possibilities: Target spot (0.10)" in text


# ---------------------------------------------------------------- endpoint


def test_confident_photo_answer() -> None:
    client, llm, fake_diagnoser, retriever = build(diagnosis())
    response = post(client, message="నా టమాటా ఆకులపై మచ్చలు ఏమిటి?", debug="true")
    assert response.status_code == 200
    body = response.json()
    assert body["language"] == "te"
    assert body["grounded"] is True
    assert body["diagnosis"]["status"] == "confident"
    assert body["diagnosis"]["condition"]["id"] == "tomato_early_blight"
    # Crop named in the question is passed to the diagnoser as a hint.
    assert fake_diagnoser.calls[0]["crop"] == "tomato"
    assert fake_diagnoser.calls[0]["note"] == "నా టమాటా ఆకులపై మచ్చలు ఏమిటి?"
    # Retrieval searched for the diagnosed condition.
    assert "Early blight" in retriever.queries[0]
    sent = llm.requests[0].messages[0].text
    assert "PHOTO ANALYSIS" in sent and "result: confident" in sent
    assert "SOURCES" in sent
    assert sent.rstrip().endswith("నా టమాటా ఆకులపై మచ్చలు ఏమిటి?")
    assert "Telugu" in (llm.requests[0].system_instruction or "")


def test_explicit_crop_and_language_fields() -> None:
    client, _, fake_diagnoser, _ = build(diagnosis())
    body = post(client, crop="tomato", language="kn").json()
    assert fake_diagnoser.calls[0]["crop"] == "tomato"
    assert body["language"] == "kn"


def test_photo_only_uses_default_question() -> None:
    client, llm, _, _ = build(diagnosis())
    post(client, language="hi")
    assert "इस फोटो में मेरे पौधे को क्या हुआ है" in llm.requests[0].messages[0].text


def test_uncertain_diagnosis_told_to_model() -> None:
    client, llm, _, _ = build(diagnosis("uncertain"))
    body = post(client, message="what is wrong?").json()
    assert body["diagnosis"]["status"] == "uncertain"
    assert "result: uncertain" in llm.requests[0].messages[0].text


@pytest.mark.parametrize(
    ("status", "expected_start"),
    [
        ("retake_photo", "ఈ ఫోటోను సరిగ్గా పరిశీలించలేకపోయాను"),
        ("not_a_plant", "ఈ ఫోటోలో మొక్క కనిపించలేదు"),
    ],
)
def test_unusable_photo_gets_fixed_message_without_ai(status: str, expected_start: str) -> None:
    client, llm, _, retriever = build(diagnosis(status, "unknown"))
    body = post(client, language="te").json()
    assert body["response"].startswith(expected_start)
    assert body["diagnosis"]["status"] == status
    assert body["grounded"] is False and body["model"] == "none"
    assert llm.requests == [] and retriever.queries == []


def test_safety_check_applies_to_photo_answers() -> None:
    llm = FakeLLM(outcomes=["Apply phorate granules."])
    client, _, _, _ = build(diagnosis(), llm=llm)
    body = post(client, message="what to spray?", language="en").json()
    assert body["safety_warnings"] and "Phorate" in body["safety_warnings"][0]


def test_unknown_crop_rejected() -> None:
    client, _, fake_diagnoser, _ = build(diagnosis())
    response = post(client, crop="dragonfruit")
    assert response.status_code == 400
    assert fake_diagnoser.calls == []


def test_invalid_language_rejected() -> None:
    client, _, _, _ = build(diagnosis())
    assert post(client, language="ta").status_code == 422


def test_photo_upload_not_blocked_by_chat_body_limit() -> None:
    client, _, _, _ = build(diagnosis())
    big_photo = b"\xff\xd8" + b"x" * (600 * 1024)  # > 256 KB text-chat limit, < photo limit
    response = client.post(URL, files={"file": ("leaf.jpg", big_photo, "image/jpeg")})
    assert response.status_code == 200


# ---------------------------------------------------------------- video


VIDEO_URL = "/api/v1/chat/video"
VIDEO = b"\x00\x00\x00\x18ftypmp42 fake video bytes"


def post_video(client: TestClient, data: bytes = VIDEO, **form: str) -> Any:
    return client.post(
        VIDEO_URL, files={"file": ("plant.mp4", data, "video/mp4")}, data=form or None
    )


def test_photo_context_mentions_video_frames() -> None:
    text = photo_context(diagnosis(video=VIDEO_INFO), crop_confirmed=True)
    assert "- source: 4 frames from the farmer's 10-second video" in text
    assert "source:" not in photo_context(diagnosis(), crop_confirmed=True)


def test_video_answer() -> None:
    client, llm, fake_diagnoser, retriever = build(diagnosis(video=VIDEO_INFO))
    response = post_video(client, message="నా టమాటా మొక్కకు ఏమైంది?")
    assert response.status_code == 200
    body = response.json()
    assert body["language"] == "te" and body["grounded"] is True
    assert body["diagnosis"]["video"]["frames_used"] == [1.0, 4.0, 6.0, 9.0]
    assert body["diagnosis"]["video"]["has_audio"] is True
    assert fake_diagnoser.calls[0]["kind"] == "video"
    assert fake_diagnoser.calls[0]["crop"] == "tomato"
    assert "Early blight" in retriever.queries[0]
    sent = llm.requests[0].messages[0].text
    assert "PHOTO ANALYSIS" in sent and "frames from the farmer's 10-second video" in sent


def test_video_only_uses_video_question() -> None:
    client, llm, _, _ = build(diagnosis(video=VIDEO_INFO))
    post_video(client, language="hi")
    assert "इस वीडियो में मेरे पौधे को क्या हुआ है" in llm.requests[0].messages[0].text


@pytest.mark.parametrize(
    ("status", "expected_start"),
    [
        ("retake_photo", "ಈ ವೀಡಿಯೊದಲ್ಲಿ ಗಿಡ ಸ್ಪಷ್ಟವಾಗಿ ಕಾಣಿಸಲಿಲ್ಲ"),
        ("not_a_plant", "ಈ ವೀಡಿಯೊದಲ್ಲಿ ಗಿಡ ಕಾಣಿಸಲಿಲ್ಲ"),
    ],
)
def test_unusable_video_gets_video_message_without_ai(status: str, expected_start: str) -> None:
    client, llm, _, retriever = build(diagnosis(status, "unknown", video=VIDEO_INFO))
    body = post_video(client, language="kn").json()
    assert body["response"].startswith(expected_start)
    assert body["model"] == "none"
    assert llm.requests == [] and retriever.queries == []


def test_video_validation() -> None:
    client, _, fake_diagnoser, _ = build(diagnosis(video=VIDEO_INFO))
    assert post_video(client, crop="dragonfruit").status_code == 400
    assert post_video(client, language="ta").status_code == 422
    assert fake_diagnoser.calls == []


def test_video_upload_limits() -> None:
    client, _, _, _ = build(diagnosis(video=VIDEO_INFO))
    # > 256 KB text-chat limit and > photo limit's overhead: video uploads use the video limit.
    assert post_video(client, VIDEO + b"x" * (600 * 1024)).status_code == 200
    small, _, _, _ = build(diagnosis(video=VIDEO_INFO), max_video_bytes=10_000)
    response = post_video(small, VIDEO + b"x" * (600 * 1024))
    assert response.status_code == 413
