"""Uploaded photo validation, privacy cleaning, quality check and preparation for the model.

- Format is identified from the decoded file, never from the name or the client's content type.
- Decompression bombs are refused (pixel limit) and the file is fully decoded (corrupt -> error).
- Phone rotation (EXIF orientation) is applied, then ALL metadata is dropped by re-encoding:
  farmers' photos carry GPS coordinates.
- Quality check (cheap, before any AI call): too small, too dark, overexposed, blurry.
- The model receives a resized RGB JPEG (longest side `max_side`): cost and latency scale with size.
CPU-bound: callers run `prepare_image` in a worker thread.
"""

from __future__ import annotations

import io
import warnings
from dataclasses import dataclass, field
from typing import Literal

from PIL import Image, ImageFilter, ImageOps, ImageStat

from app.core.errors import AppError, PayloadTooLargeError, UnsupportedMediaTypeError

# MPO = multi-picture JPEG written by many phone cameras under a plain ".jpg" name (about 1 in 10
# field photos in our datasets); the first frame is the normal photo.
ALLOWED_FORMATS = {
    "JPEG": "image/jpeg",
    "MPO": "image/jpeg",
    "PNG": "image/png",
    "WEBP": "image/webp",
}
MAX_PIXELS = 40_000_000  # ~ 7000 x 5700; larger is refused (decompression-bomb guard)

# Quality thresholds, calibrated on field and lab leaf photos (see tests and eval script).
MIN_SIDE = 200
# Exposure is judged on percentiles, not the mean: a well-lit leaf on a dark background has a low
# mean brightness but is perfectly usable. Too dark = even the bright 10% of the photo is dark;
# overexposed = even the darkest 10% is near white.
TOO_DARK = 50.0  # 90th-percentile luminance below this
DARK = 80.0
OVEREXPOSED = 235.0  # 10th-percentile luminance above this
VERY_BLURRY = 15.0
BLURRY = 45.0
_BLUR_ANALYSIS_SIDE = 512
_LAPLACIAN = ImageFilter.Kernel((3, 3), (0, 1, 0, 1, -4, 1, 0, 1, 0), scale=1, offset=128)

QualityIssue = Literal["too_small", "too_dark", "dark", "overexposed", "very_blurry", "blurry"]
SEVERE_ISSUES: frozenset[QualityIssue] = frozenset(
    {"too_small", "too_dark", "overexposed", "very_blurry"}
)


class InvalidImageError(AppError):
    status_code = 422
    code = "invalid_image"
    default_message = (
        "The file could not be read as a photo. Please upload a JPEG, PNG or WEBP image."
    )


@dataclass(frozen=True, slots=True)
class ImageQuality:
    width: int
    height: int
    brightness: float
    sharpness: float  # variance of the Laplacian: low = blurry
    issues: tuple[QualityIssue, ...] = ()

    @property
    def severe(self) -> bool:
        return any(issue in SEVERE_ISSUES for issue in self.issues)


@dataclass(frozen=True, slots=True)
class PreparedImage:
    data: bytes  # JPEG for the model, metadata-free
    mime_type: str
    original_format: str
    original_bytes: int
    quality: ImageQuality
    model_size: tuple[int, int] = field(default=(0, 0))


def _decode(data: bytes) -> Image.Image:
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        try:
            Image.MAX_IMAGE_PIXELS = MAX_PIXELS
            probe = Image.open(io.BytesIO(data))
            image_format = probe.format or ""
            if image_format not in ALLOWED_FORMATS:
                raise UnsupportedMediaTypeError(
                    "Unsupported image type. Please upload a JPEG, PNG or WEBP photo."
                )
            probe.verify()  # structural check; the image must be reopened afterwards
            image = Image.open(io.BytesIO(data))
            image.load()
        except AppError:
            raise
        except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
            raise InvalidImageError("The photo is too large in pixels.") from exc
        except Exception as exc:
            raise InvalidImageError() from exc
    return image


def _percentile(histogram: list[int], fraction: float) -> int:
    target = sum(histogram) * fraction
    running = 0
    for level, count in enumerate(histogram):
        running += count
        if running >= target:
            return level
    return 255


def assess_quality(image: Image.Image) -> ImageQuality:
    width, height = image.size
    gray = ImageOps.grayscale(image)
    histogram = gray.histogram()
    brightness = float(ImageStat.Stat(gray).mean[0])
    bright_regions = _percentile(histogram, 0.9)
    dark_regions = _percentile(histogram, 0.1)
    small = gray.copy()
    small.thumbnail((_BLUR_ANALYSIS_SIDE, _BLUR_ANALYSIS_SIDE))
    edges = small.filter(_LAPLACIAN)
    # Pillow leaves the 1-pixel border unfiltered (raw brightness instead of the 128 offset),
    # which would add a large fake "sharpness": measure the interior only.
    edges = edges.crop((2, 2, max(3, edges.width - 2), max(3, edges.height - 2)))
    sharpness = float(ImageStat.Stat(edges).var[0])

    issues: list[QualityIssue] = []
    if min(width, height) < MIN_SIDE:
        issues.append("too_small")
    if bright_regions < TOO_DARK:
        issues.append("too_dark")
    elif bright_regions < DARK:
        issues.append("dark")
    if dark_regions > OVEREXPOSED:
        issues.append("overexposed")
    if sharpness < VERY_BLURRY:
        issues.append("very_blurry")
    elif sharpness < BLURRY:
        issues.append("blurry")
    return ImageQuality(
        width=width,
        height=height,
        brightness=round(brightness, 1),
        sharpness=round(sharpness, 1),
        issues=tuple(issues),
    )


def prepare_image(data: bytes, *, max_bytes: int, max_side: int) -> PreparedImage:
    if len(data) > max_bytes:
        raise PayloadTooLargeError(f"Photo is larger than {max_bytes // (1024 * 1024)} MB.")
    if not data:
        raise InvalidImageError("The uploaded file is empty.")
    image = _decode(data)
    original_format = image.format or ""
    image = ImageOps.exif_transpose(image)  # apply phone rotation before metadata is dropped
    if image.mode not in ("RGB", "L"):
        background = Image.new("RGB", image.size, (255, 255, 255))
        rgba = image.convert("RGBA")
        background.paste(rgba, mask=rgba.split()[-1])
        image = background
    image = image.convert("RGB")
    quality = assess_quality(image)

    resized = image.copy()
    resized.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    resized.save(buffer, format="JPEG", quality=88, optimize=True)  # no exif= : metadata dropped
    return PreparedImage(
        data=buffer.getvalue(),
        mime_type="image/jpeg",
        original_format=original_format,
        original_bytes=len(data),
        quality=quality,
        model_size=resized.size,
    )
