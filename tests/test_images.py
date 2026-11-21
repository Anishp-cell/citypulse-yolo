import io

import pytest
from PIL import Image

from backend.services.images import (
    MAX_SIDE,
    ImageTooLarge,
    UnsupportedImage,
    make_public_crop,
    make_thumb,
    process_upload,
)


def jpeg_bytes(size=(800, 600), color=(200, 50, 50), exif=None, fmt="JPEG"):
    buf = io.BytesIO()
    img = Image.new("RGB", size, color)
    kwargs = {"exif": exif} if exif is not None else {}
    img.save(buf, fmt, **kwargs)
    return buf.getvalue()


def open_(b):
    return Image.open(io.BytesIO(b))


def test_png_is_reencoded_as_jpeg():
    p = process_upload(jpeg_bytes(fmt="PNG"))
    assert p.full_jpeg[:2] == b"\xff\xd8"
    assert (p.width, p.height) == (800, 600)


def test_rejects_non_image_bytes():
    with pytest.raises(UnsupportedImage):
        process_upload(b"definitely not an image" * 10)


def test_heic_named_jpeg_rejected_cleanly():
    fake_heic = b"\x00\x00\x00\x18ftypheic\x00\x00\x00\x00mif1heic" + b"\x00" * 200
    with pytest.raises(UnsupportedImage):
        process_upload(fake_heic)


def test_rejects_too_large():
    with pytest.raises(ImageTooLarge):
        process_upload(b"\xff\xd8" + b"\x00" * (10 * 1024 * 1024 + 1))


def test_strips_exif_gps():
    exif = Image.Exif()
    exif[0x010F] = "PhoneMaker"  # Make
    exif[0x8825] = {1: "N", 2: (18.0, 31.0, 0.0)}  # GPSInfo
    p = process_upload(jpeg_bytes(exif=exif.tobytes()))
    assert len(open_(p.full_jpeg).getexif()) == 0


def test_applies_exif_orientation():
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 CW
    p = process_upload(jpeg_bytes(size=(800, 600), exif=exif.tobytes()))
    assert (p.width, p.height) == (600, 800)


def test_downscales_very_large_dimensions():
    p = process_upload(jpeg_bytes(size=(5000, 2500)))
    assert max(p.width, p.height) == MAX_SIDE


def test_crop_contains_box():
    full = process_upload(jpeg_bytes(size=(1000, 800))).full_jpeg
    crop = open_(make_public_crop(full, [(400, 300, 600, 500)]))
    # box is 200x200, +35% margin each side => 340x340
    assert 330 <= crop.width <= 350 and 330 <= crop.height <= 350


def test_crop_respects_min_side_and_image_bounds():
    full = process_upload(jpeg_bytes(size=(1000, 800))).full_jpeg
    crop = open_(make_public_crop(full, [(0, 0, 20, 20)]))
    assert crop.width >= 320 and crop.height >= 320


def test_crop_without_boxes_uses_centre():
    full = process_upload(jpeg_bytes(size=(1000, 800))).full_jpeg
    crop = open_(make_public_crop(full, []))
    assert (crop.width, crop.height) == (600, 480)


def test_thumb_max_side():
    full = process_upload(jpeg_bytes(size=(1000, 800))).full_jpeg
    t = open_(make_thumb(full, 256))
    assert max(t.size) == 256


def test_decompression_bomb_rejected_before_decoding():
    # Review C2: a ~10 KB palette PNG that decodes to 78 megapixels must not be loaded at all.
    buf = io.BytesIO()
    Image.new("P", (7800, 10000)).save(buf, "PNG", optimize=True)
    assert len(buf.getvalue()) < 200_000
    with pytest.raises(ImageTooLarge):
        process_upload(buf.getvalue())


def test_large_but_reasonable_photo_still_accepted():
    p = process_upload(jpeg_bytes(size=(4000, 3000)))  # 12 MP phone photo
    assert max(p.width, p.height) == MAX_SIDE
