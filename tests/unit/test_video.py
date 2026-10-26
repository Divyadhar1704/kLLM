"""Video validation, best-frame selection, privacy, diagnoser and API (no AI calls).

Videos are encoded on the fly with PyAV from synthetic leaf frames.
"""

from __future__ import annotations

import io
from fractions import Fraction
from itertools import pairwise
from types import SimpleNamespace
from typing import Any

import av
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageFilter

from app.core.errors import PayloadTooLargeError, UnsupportedMediaTypeError
from app.vision.diagnosis import VisionDiagnoser
from app.vision.video import (
    InvalidVideoError,
    VideoTooLongError,
    detect_container,
    prepare_video,
    upright,
)
from tests.unit.test_vision import (
    GOOD_JSON,
    MB,
    FakeModels,
    client_with,
    diagnoser,
    encode,
    leaf_image,
)

SIZE = (320, 240)


def sharp_frame() -> Image.Image:
    return leaf_image(SIZE)


def blurry_frame() -> Image.Image:
    return leaf_image(SIZE).filter(ImageFilter.GaussianBlur(6))


def dark_frame() -> Image.Image:
    return leaf_image(SIZE).point(lambda v: v // 8)


def make_video(
    frames: list[Image.Image],
    *,
    fps: int = 10,
    fmt: str = "mp4",
    codec: str = "mpeg4",
    audio: bool = False,
    gop: int | None = None,
) -> bytes:
    buffer = io.BytesIO()
    with av.open(buffer, mode="w", format=fmt) as out:
        stream = out.add_stream(codec, rate=fps)
        stream.width, stream.height = frames[0].size
        stream.pix_fmt = "yuv420p"
        stream.bit_rate = 4_000_000
        if gop is not None:
            stream.codec_context.gop_size = gop
        sound = out.add_stream("aac", rate=16000, layout="mono") if audio else None
        for index, image in enumerate(frames):
            frame = av.VideoFrame.from_image(image)
            frame.pts = index
            frame.time_base = Fraction(1, fps)
            for packet in stream.encode(frame):
                out.mux(packet)
        if sound is not None:
            samples = sound.codec_context.frame_size or 1024
            for index in range(int(len(frames) / fps * 16000 / samples)):
                chunk = av.AudioFrame(format="fltp", layout="mono", samples=samples)
                chunk.planes[0].update(bytes(samples * 4))
                chunk.sample_rate = 16000
                chunk.pts = index * samples
                for packet in sound.encode(chunk):
                    out.mux(packet)
            for packet in sound.encode(None):
                out.mux(packet)
        for packet in stream.encode(None):
            out.mux(packet)
    return buffer.getvalue()


def prepare(data: bytes, **overrides: Any) -> Any:
    options: dict[str, Any] = {
        "max_bytes": 50 * MB,
        "max_seconds": 60,
        "frames": 4,
        "candidate_frames": 16,
        "max_side": 1024,
    }
    return prepare_video(data, **{**options, **overrides})


# ---------------------------------------------------------------- validation


def test_container_identified_by_content() -> None:
    assert detect_container(make_video([sharp_frame()] * 5)) == "mov"
    webm = make_video([sharp_frame()] * 5, fmt="webm", codec="libvpx")
    assert detect_container(webm) == "matroska"
    for not_video in (encode(leaf_image()), encode(leaf_image(), "GIF"), b"#EXTM3U\nhttp://x/"):
        with pytest.raises(UnsupportedMediaTypeError):
            detect_container(not_video)


@pytest.mark.parametrize(
    "data",
    [b"", b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 200, b"\x1a\x45\xdf\xa3" + b"junk" * 50],
)
def test_corrupt_or_empty_video(data: bytes) -> None:
    with pytest.raises(InvalidVideoError):
        prepare(data)


def test_truncated_video() -> None:
    data = make_video([sharp_frame()] * 20)
    with pytest.raises(InvalidVideoError):
        prepare(data[: len(data) // 3])


def test_too_large_file() -> None:
    with pytest.raises(PayloadTooLargeError):
        prepare(make_video([sharp_frame()] * 5), max_bytes=1000)


def test_too_long_video() -> None:
    data = make_video([sharp_frame()] * 30, fps=10)  # 3 s
    with pytest.raises(VideoTooLongError):
        prepare(data, max_seconds=2)


# ---------------------------------------------------------------- frame selection


def test_picks_sharp_frames_over_blurry_ones() -> None:
    frames = [blurry_frame()] * 20 + [sharp_frame()] * 20  # 2 s blurry, then 2 s sharp
    video = prepare(make_video(frames))
    assert len(video.frames) == 4
    assert all(t >= 2.0 for t in video.info.frame_times)
    assert len(set(video.info.frame_times)) == 4  # different moments
    assert all(not f.image.quality.issues for f in video.frames)
    assert video.info.duration_seconds == pytest.approx(4.0, abs=0.2)
    assert video.info.frames_examined >= 8


def test_frames_spread_over_the_video() -> None:
    video = prepare(make_video([sharp_frame()] * 80))  # 8 s, all sharp
    times = video.info.frame_times
    assert times[0] < 2.0 and times[-1] >= 5.0
    assert min(b - a for a, b in pairwise(times)) >= 1.0


def test_all_dark_video_has_no_usable_frame() -> None:
    video = prepare(make_video([dark_frame()] * 20))
    assert video.frames == ()
    assert video.quality.severe


def test_single_keyframe_video_is_fully_decoded() -> None:
    video = prepare(make_video([sharp_frame()] * 40, gop=1000))
    assert video.info.frames_examined >= 8
    assert len(video.frames) == 4


def test_webm_and_audio_track() -> None:
    webm = prepare(make_video([sharp_frame()] * 20, fmt="webm", codec="libvpx"))
    assert webm.info.container == "webm" and webm.info.codec == "vp8"
    with_sound = prepare(make_video([sharp_frame()] * 20, audio=True))
    assert with_sound.info.has_audio is True  # reported; the audio is never decoded
    assert with_sound.info.container == "mp4" and len(with_sound.frames) == 4


def test_frames_are_metadata_free_jpegs() -> None:
    video = prepare(make_video([sharp_frame()] * 20))
    image = Image.open(io.BytesIO(video.frames[0].image.data))
    assert image.format == "JPEG"
    assert len(image.getexif()) == 0
    assert video.frames[0].image.mime_type == "image/jpeg"


def test_upright_rotation() -> None:
    image = Image.new("RGB", (40, 20), "white")
    image.putpixel((0, 0), (255, 0, 0))  # red at top-left
    turned = upright(image, 90)  # 90 degrees counterclockwise
    assert turned.size == (20, 40)
    assert turned.getpixel((0, 39)) == (255, 0, 0)  # top-left moved to bottom-left
    assert upright(image, -90).getpixel((19, 0)) == (255, 0, 0)  # clockwise: top-right
    assert upright(image, 180).getpixel((39, 19)) == (255, 0, 0)
    assert upright(image, 0) is image


# ---------------------------------------------------------------- diagnoser


async def test_video_frames_sent_in_one_call() -> None:
    service, fake = diagnoser([GOOD_JSON])
    result = await service.diagnose_video(make_video([sharp_frame()] * 40), crop="tomato")
    assert result.status == "confident"
    assert result.video is not None and len(result.video.frame_times) == 4
    assert len(fake.calls) == 1
    parts = fake.calls[0]["contents"][0].parts
    assert [p.inline_data.mime_type for p in parts[:4]] == ["image/jpeg"] * 4
    assert "frames from ONE short video" in parts[4].text
    assert "The farmer says this is Tomato" in parts[4].text


async def test_unusable_video_skips_the_ai_call() -> None:
    service, fake = diagnoser([GOOD_JSON])
    result = await service.diagnose_video(make_video([dark_frame()] * 20))
    assert result.status == "retake_photo"
    assert result.video is not None and result.video.frame_times == ()
    assert fake.calls == []


# ---------------------------------------------------------------- API


def upload(client: TestClient, data: bytes, **form: str) -> Any:
    return client.post(
        "/api/v1/video/diagnose",
        files={"file": ("plant.mp4", data, "video/mp4")},
        data=form or None,
    )


def test_api_video_diagnose() -> None:
    service, _ = diagnoser([GOOD_JSON])
    response = upload(client_with(service), make_video([sharp_frame()] * 30), crop="tomato")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "confident"
    assert body["condition"]["id"] == "tomato_early_blight"
    video = body["video"]
    assert video["container"] == "mp4" and video["codec"] == "mpeg4"
    assert video["width"] == 320 and video["height"] == 240
    assert len(video["frames_used"]) == 4 and video["has_audio"] is False


def test_api_image_response_has_no_video_block() -> None:
    service, _ = diagnoser([GOOD_JSON])
    response = client_with(service).post(
        "/api/v1/image/diagnose", files={"file": ("leaf.jpg", encode(leaf_image()), "image/jpeg")}
    )
    assert response.json()["video"] is None


def test_api_errors() -> None:
    fake = FakeModels([GOOD_JSON])
    service = VisionDiagnoser(
        client=SimpleNamespace(aio=SimpleNamespace(models=fake)),
        models=("primary",),
        timeout_seconds=5,
        max_bytes=10 * MB,
        model_max_side=1024,
        min_confidence=0.6,
        reject_poor_quality=True,
        video_max_seconds=2,
    )
    client = client_with(service)
    gif = upload(client, encode(leaf_image(), "GIF"))
    assert gif.status_code == 415 and gif.json()["error"]["code"] == "unsupported_media_type"
    too_long = upload(client, make_video([sharp_frame()] * 30))
    assert too_long.status_code == 422 and too_long.json()["error"]["code"] == "video_too_long"
    corrupt = upload(client, b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 200)
    assert corrupt.status_code == 422 and corrupt.json()["error"]["code"] == "invalid_video"
    assert upload(client, make_video([sharp_frame()] * 5), crop="dragonfruit").status_code == 400
    assert fake.calls == []


def test_api_body_size_limit() -> None:
    service, _ = diagnoser([GOOD_JSON])
    client = client_with(service, max_video_bytes=10_000)
    response = upload(client, b"\x00\x00\x00\x18ftyp" + b"x" * (600 * 1024))
    assert response.status_code == 413


def test_api_vision_disabled() -> None:
    assert upload(client_with(None), make_video([sharp_frame()] * 5)).status_code == 503
