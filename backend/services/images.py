"""Photo handling: validate by content, fix orientation, strip metadata, crop for public display."""

import io
import warnings
from dataclasses import dataclass

from PIL import Image, ImageOps, UnidentifiedImageError

MAX_BYTES = 10 * 1024 * 1024
MAX_SIDE = 2048
MAX_PIXELS = 25_000_000  # ~25 MP: above any phone camera after our client-side resize
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP", "MPO"}  # MPO = multi-picture JPEG some phones produce
JPEG_QUALITY = 85

Image.MAX_IMAGE_PIXELS = MAX_PIXELS


class UnsupportedImage(ValueError):
    """Not a decodable JPEG/PNG/WebP."""


class ImageTooLarge(ValueError):
    pass


@dataclass(frozen=True)
class ProcessedImage:
    full_jpeg: bytes
    width: int
    height: int


def _encode(img: Image.Image, quality: int = JPEG_QUALITY) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality, optimize=True)  # no exif= argument, so metadata is dropped
    return buf.getvalue()


def process_upload(raw: bytes) -> ProcessedImage:
    if len(raw) > MAX_BYTES:
        raise ImageTooLarge("image exceeds 10 MB")
    try:
        img = Image.open(io.BytesIO(raw))
        if img.format not in ALLOWED_FORMATS:
            raise UnsupportedImage(f"unsupported format {img.format}")
        if img.width * img.height > MAX_PIXELS:  # check the header before allocating anything
            raise ImageTooLarge("image has too many pixels")
        if img.format in ("JPEG", "MPO"):
            img.draft("RGB", (MAX_SIDE, MAX_SIDE))  # decode JPEGs at reduced size directly
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            img.load()
    except ImageTooLarge:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as e:
        raise ImageTooLarge("image has too many pixels") from e
    except (UnidentifiedImageError, OSError, SyntaxError) as e:
        raise UnsupportedImage(str(e)) from e
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")
    if max(img.size) > MAX_SIDE:
        img.thumbnail((MAX_SIDE, MAX_SIDE), Image.Resampling.LANCZOS)
    return ProcessedImage(_encode(img), img.width, img.height)


def make_public_crop(full_jpeg: bytes, boxes: list[tuple[int, int, int, int]], margin: float = 0.35,
                     min_side: int = 320) -> bytes:
    """Crop to the union of detected boxes plus a margin, so bystanders and plates are mostly left out."""
    img = Image.open(io.BytesIO(full_jpeg))
    w, h = img.size
    if boxes:
        x1, y1 = min(b[0] for b in boxes), min(b[1] for b in boxes)
        x2, y2 = max(b[2] for b in boxes), max(b[3] for b in boxes)
        bw, bh = x2 - x1, y2 - y1
        x1, x2 = x1 - bw * margin, x2 + bw * margin
        y1, y2 = y1 - bh * margin, y2 + bh * margin
    else:  # no detection: keep the central 60%
        x1, y1, x2, y2 = w * 0.2, h * 0.2, w * 0.8, h * 0.8
    # grow to the minimum size around the centre, then clamp to the image
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half_w = max((x2 - x1) / 2, min(min_side, w) / 2)
    half_h = max((y2 - y1) / 2, min(min_side, h) / 2)
    left, right = cx - half_w, cx + half_w
    top, bottom = cy - half_h, cy + half_h
    if left < 0:
        left, right = 0, right - left
    if right > w:
        left, right = max(0, left - (right - w)), w
    if top < 0:
        top, bottom = 0, bottom - top
    if bottom > h:
        top, bottom = max(0, top - (bottom - h)), h
    return _encode(img.crop((round(left), round(top), round(right), round(bottom))))


def make_thumb(jpeg: bytes, max_side: int = 256) -> bytes:
    img = Image.open(io.BytesIO(jpeg))
    img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return _encode(img, quality=80)
