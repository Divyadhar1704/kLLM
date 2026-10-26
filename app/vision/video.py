"""Uploaded video validation and best-frame extraction for diagnosis.

A farmer's video is turned into a few sharp, well-exposed still frames; the frames go through the
same constrained, calibrated photo diagnosis. The model never receives the video itself:

- Container is identified from the file's magic bytes (MP4/MOV/3GP or WebM/MKV) and the demuxer
  is chosen explicitly, never by probing: probing would also accept playlist and concat formats
  that make FFmpeg open other files or URLs.
- Only the video track is decoded. Audio (farmers' voices) is never decoded or sent anywhere.
- Limits: file size, duration, pixel size, decoded-frame count and decode wall time.
- Keyframes are decoded first (fast); short or sparse-keyframe videos are fully decoded.
- Candidate frames are scored with the photo quality check; the sharpest usable frame of each
  time segment is kept, so the frames show different parts of the plant.
- Phone rotation is applied and frames are re-encoded as JPEG: container metadata (GPS location
  atoms, device names) never leaves this module.
CPU-bound: callers run `prepare_video` in a worker thread.
"""

from __future__ import annotations

import io
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import av
from PIL import Image

from app.core.errors import AppError, PayloadTooLargeError, UnsupportedMediaTypeError
from app.vision.images import ImageQuality, PreparedImage, assess_quality

# ISO base media (MP4 / MOV / 3GP): the first box type at bytes 4..8.
_ISO_BOXES = (b"ftyp", b"moov", b"mdat", b"free", b"skip", b"wide")
_EBML_MAGIC = b"\x1a\x45\xdf\xa3"  # Matroska / WebM
ALLOWED_CODECS = frozenset({"h264", "hevc", "vp8", "vp9", "av1", "mpeg4", "h263"})
MAX_VIDEO_PIXELS = 4096 * 2304  # 4K phone video is fine; larger is refused
MAX_DECODE_SECONDS = 20.0  # wall time for decoding one upload
_MAX_FPS_ASSUMED = 120  # decoded-frame cap = max_seconds * this
_SPARSE_KEYFRAMES = 0.5  # fewer keyframe samples than this share of the target: decode all frames
_SEVERE_PENALTY = 0.0
_WARNING_PENALTY = 0.5
_DEFAULT_INTERVAL = 0.5  # seconds between candidates when the duration is unknown


class InvalidVideoError(AppError):
    status_code = 422
    code = "invalid_video"
    default_message = "The file could not be read as a video. Please upload an MP4 or WEBM video."


class VideoTooLongError(AppError):
    status_code = 422
    code = "video_too_long"
    default_message = "The video is too long."


@dataclass(frozen=True, slots=True)
class VideoInfo:
    duration_seconds: float
    width: int  # upright (after rotation)
    height: int
    container: str  # mp4 | webm
    codec: str
    has_audio: bool  # reported only: audio is never decoded
    frames_examined: int
    frame_times: tuple[float, ...]  # seconds, of the frames sent to the model


@dataclass(frozen=True, slots=True)
class VideoFrameSample:
    time_seconds: float
    image: PreparedImage


@dataclass(frozen=True, slots=True)
class PreparedVideo:
    frames: tuple[VideoFrameSample, ...]  # empty = no usable frame (ask for a retake)
    quality: ImageQuality  # of the best frame
    info: VideoInfo
    original_bytes: int

    @property
    def images(self) -> list[PreparedImage]:
        return [frame.image for frame in self.frames]


@dataclass(slots=True)
class _Candidate:
    time: float
    image: Image.Image
    quality: ImageQuality

    @property
    def usable(self) -> bool:
        return not self.quality.severe

    @property
    def score(self) -> float:
        if self.quality.severe:
            return self.quality.sharpness * _SEVERE_PENALTY
        return self.quality.sharpness * (_WARNING_PENALTY if self.quality.issues else 1.0)


def detect_container(data: bytes) -> str:
    """'mov' (MP4/MOV/3GP) or 'matroska' (WebM/MKV), from the magic bytes."""
    if data[:4] == _EBML_MAGIC:
        return "matroska"
    if len(data) >= 8 and data[4:8] in _ISO_BOXES:
        return "mov"
    raise UnsupportedMediaTypeError("Unsupported video type. Please upload an MP4 or WEBM video.")


def upright(image: Image.Image, rotation: int) -> Image.Image:
    """Apply the display rotation (degrees counterclockwise, as FFmpeg reports it)."""
    turns = round(rotation / 90) % 4
    if turns == 1:
        return image.transpose(Image.Transpose.ROTATE_90)
    if turns == 2:
        return image.transpose(Image.Transpose.ROTATE_180)
    if turns == 3:
        return image.transpose(Image.Transpose.ROTATE_270)
    return image


def _frame_time(frame: Any, stream: Any, index: int) -> float:
    if frame.time is not None:
        return float(frame.time)
    rate = float(stream.average_rate or 30)
    return index / rate


def _check_deadline(deadline: float) -> None:
    if time.monotonic() > deadline:
        raise InvalidVideoError("The video took too long to process. Please send a shorter clip.")


def _packet_duration(container: Any, stream: Any, *, max_packets: int, deadline: float) -> float:
    """Duration from packet timestamps, without decoding (browser WebM has none in its header)."""
    last = 0.0
    for index, packet in enumerate(container.demux(stream)):
        if index >= max_packets:
            raise InvalidVideoError("The video has too many frames.")
        if index % 256 == 0:
            _check_deadline(deadline)
        if packet.pts is not None and packet.time_base is not None:
            last = max(last, float(packet.pts * packet.time_base))
    return last


def _sampled_frames(
    data: bytes,
    demuxer: str,
    *,
    keyframes_only: bool,
    interval: float,
    max_seconds: float,
    max_frames: int,
    max_side: int,
    deadline: float,
) -> Iterator[tuple[float, Image.Image]]:
    """Upright, downscaled RGB frames, at most one per `interval` seconds."""
    with av.open(io.BytesIO(data), format=demuxer, mode="r") as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        if keyframes_only:
            stream.codec_context.skip_frame = "NONKEY"
        next_time = 0.0
        for index, frame in enumerate(container.decode(stream)):  # audio is never decoded
            if index >= max_frames:
                raise InvalidVideoError("The video has too many frames.")
            _check_deadline(deadline)
            moment = _frame_time(frame, stream, index)
            if moment > max_seconds + 0.5:
                raise VideoTooLongError(f"The video is longer than {int(max_seconds)} seconds.")
            if moment + 1e-6 >= next_time:
                next_time = moment + interval
                picture: Image.Image = frame.to_image()  # type: ignore[no-untyped-call]
                image = upright(picture, frame.rotation or 0).convert("RGB")
                image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
                yield moment, image


def _thin[T](items: list[T], limit: int) -> list[T]:
    """`limit` items spread evenly over the list."""
    if len(items) <= limit:
        return items
    step = len(items) / limit
    return [items[int(i * step)] for i in range(limit)]


def select_frames(candidates: list[_Candidate], count: int) -> list[_Candidate]:
    """Up to `count` frames, in time order: the sharpest clean frame of each of `count` equal
    time segments (different views of the plant), topped up with the next sharpest clean frames,
    then with frames that have only minor issues (slightly dark / blurry). Frames with severe
    issues are never used."""
    clean = [c for c in candidates if not c.quality.issues]
    minor = [c for c in candidates if c.usable and c.quality.issues]
    if not clean and not minor:
        return []
    start, end = candidates[0].time, candidates[-1].time
    span = max(end - start, 1e-6)
    best: dict[int, _Candidate] = {}
    for candidate in clean:
        segment = min(count - 1, int((candidate.time - start) / span * count))
        if segment not in best or candidate.score > best[segment].score:
            best[segment] = candidate
    chosen = list(best.values())
    for pool in (clean, minor):
        if len(chosen) >= count:
            break
        rest = sorted((c for c in pool if c not in chosen), key=lambda c: c.score, reverse=True)
        chosen.extend(rest[: count - len(chosen)])
    return sorted(chosen, key=lambda c: c.time)


def _encode(candidate: _Candidate, original_bytes: int) -> PreparedImage:
    buffer = io.BytesIO()
    candidate.image.save(buffer, format="JPEG", quality=88, optimize=True)  # metadata-free
    return PreparedImage(
        data=buffer.getvalue(),
        mime_type="image/jpeg",
        original_format="video",
        original_bytes=original_bytes,
        quality=candidate.quality,
        model_size=candidate.image.size,
    )


def prepare_video(
    data: bytes,
    *,
    max_bytes: int,
    max_seconds: float,
    frames: int,
    candidate_frames: int,
    max_side: int,
) -> PreparedVideo:
    if len(data) > max_bytes:
        raise PayloadTooLargeError(f"Video is larger than {max_bytes // (1024 * 1024)} MB.")
    if not data:
        raise InvalidVideoError("The uploaded file is empty.")
    demuxer = detect_container(data)
    deadline = time.monotonic() + MAX_DECODE_SECONDS
    max_frames = int(max_seconds * _MAX_FPS_ASSUMED)
    try:
        with av.open(io.BytesIO(data), format=demuxer, mode="r") as container:
            if not container.streams.video:
                raise InvalidVideoError("The file has no video track.")
            stream = container.streams.video[0]
            codec = stream.codec_context.name
            width, height = stream.codec_context.width, stream.codec_context.height
            has_audio = bool(container.streams.audio)
            if codec not in ALLOWED_CODECS:
                raise UnsupportedMediaTypeError(
                    "Unsupported video format. Please send the video as recorded by the phone "
                    "camera."
                )
            if not width or not height:
                raise InvalidVideoError()
            if width * height > MAX_VIDEO_PIXELS:
                raise InvalidVideoError("The video resolution is too large.")
            duration = (
                container.duration / av.time_base
                if container.duration
                else float(stream.duration * stream.time_base)
                if stream.duration and stream.time_base
                else 0.0
            )
            if duration <= 0:
                duration = _packet_duration(
                    container, stream, max_packets=max_frames, deadline=deadline
                )
        if duration > max_seconds + 0.5:
            raise VideoTooLongError(f"The video is longer than {int(max_seconds)} seconds.")

        decode: dict[str, Any] = {
            "interval": duration / candidate_frames if duration > 0 else _DEFAULT_INTERVAL,
            "max_seconds": max_seconds,
            "max_frames": max_frames,
            "max_side": max_side,
            "deadline": deadline,
        }
        sampled = list(_sampled_frames(data, demuxer, keyframes_only=True, **decode))
        if len(sampled) < max(2, candidate_frames * _SPARSE_KEYFRAMES):
            sampled = list(_sampled_frames(data, demuxer, keyframes_only=False, **decode))
    except AppError:
        raise
    except Exception as exc:  # FFmpeg errors: corrupt, truncated or unsupported data
        raise InvalidVideoError() from exc
    if not sampled:
        raise InvalidVideoError("No frames could be read from the video.")

    candidates = [
        _Candidate(round(moment, 2), image, assess_quality(image))
        for moment, image in _thin(sampled, candidate_frames)
    ]
    chosen = select_frames(candidates, frames)
    best = max(chosen or candidates, key=lambda c: (c.score, c.quality.sharpness))
    samples = tuple(VideoFrameSample(c.time, _encode(c, len(data))) for c in chosen)
    upright_size = candidates[0].image.size
    scale = max(width, height) / max(upright_size)
    return PreparedVideo(
        frames=samples,
        quality=best.quality,
        info=VideoInfo(
            duration_seconds=round(duration or candidates[-1].time, 2),
            width=round(upright_size[0] * scale),
            height=round(upright_size[1] * scale),
            container="webm" if demuxer == "matroska" else "mp4",
            codec=codec,
            has_audio=has_audio,
            frames_examined=len(candidates),
            frame_times=tuple(c.time for c in chosen),
        ),
        original_bytes=len(data),
    )
