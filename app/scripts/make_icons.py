"""Generates the PWA icons in public/icons (run from app/: python scripts/make_icons.py)."""

from pathlib import Path

from PIL import Image, ImageDraw

BLUE = (29, 78, 216, 255)
WHITE = (255, 255, 255, 255)
OUT = Path(__file__).resolve().parent.parent / "public" / "icons"


def pin(draw, cx, top, s, fg, hole_color):
    r = s * 0.32
    cy = top + s * 0.08 + r
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fg)
    draw.polygon([(cx - r * 0.93, cy + r * 0.35), (cx + r * 0.93, cy + r * 0.35), (cx, top + s * 0.95)], fill=fg)
    hole = r * 0.48
    draw.ellipse([cx - hole, cy - hole * 0.6, cx + hole, cy + hole * 0.6], fill=hole_color)


def icon(size, maskable=False):
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    if maskable:  # full-bleed background, glyph inside the 80% safe zone
        d.rectangle([0, 0, size, size], fill=BLUE)
        scale = 0.6
    else:
        d.rounded_rectangle([0, 0, size - 1, size - 1], radius=int(size * 0.22), fill=BLUE)
        scale = 0.75
    s = size * scale
    pin(d, size / 2, (size - s) / 2, s, WHITE, BLUE)
    return img


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    icon(192).save(OUT / "icon-192.png")
    icon(512).save(OUT / "icon-512.png")
    icon(512, maskable=True).save(OUT / "icon-maskable-512.png")
    icon(180, maskable=True).convert("RGB").save(OUT / "apple-touch-icon.png")
    badge = Image.new("RGBA", (72, 72), (0, 0, 0, 0))
    pin(ImageDraw.Draw(badge), 36, 4, 66, WHITE, (0, 0, 0, 0))
    badge.save(OUT / "badge-72.png")
    print("icons written to", OUT)


if __name__ == "__main__":
    main()
