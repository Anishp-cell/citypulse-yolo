"""Serves the built PWA (app/dist): static files, SPA fallback, and Open Graph tags for /h/{id} share links."""

import html
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from sqlalchemy.orm import Session

from backend.db import get_db
from backend.domain.status import PUBLIC_STATUSES
from backend.models import Hazard
from backend.services.present import days_open

router = APIRouter(include_in_schema=False)

OG_MARKER = "<!-- OG_TAGS -->"
TYPE_LABEL = {"pothole": "Pothole", "open_manhole": "Open manhole", "waterlogging": "Waterlogging",
              "debris": "Road obstruction", "other": "Road hazard"}
RESERVED_PREFIXES = ("api/", "media/", "health")


def _dist(request: Request) -> Path:
    return Path(request.app.state.settings.frontend_dist).resolve()


def _index_html(request: Request) -> str:
    index = _dist(request) / "index.html"
    if not index.is_file():
        raise HTTPException(status_code=404, detail="frontend not built")
    return index.read_text(encoding="utf-8")


def _html(content: str) -> HTMLResponse:
    return HTMLResponse(content, headers={"Cache-Control": "no-cache"})


def _og_tags(request: Request, h: Hazard) -> str:
    settings = request.app.state.settings
    place = h.road or h.ward or h.city or "the road"
    title = f"{TYPE_LABEL.get(h.type, 'Road hazard')} on {place}"
    days = days_open(h)
    desc = (f"Fixed after {days} days." if h.status.value == "fixed"
            else f"Open for {days} days · reported by {h.report_count} "
                 f"{'person' if h.report_count == 1 else 'people'}. Help get it fixed on CityPulse.")
    tags = {"og:title": title, "og:description": desc, "og:type": "website",
            "og:url": f"{settings.app_url.rstrip('/')}/h/{h.id}"}
    if h.cover_image_key:
        img = request.app.state.storage.url(h.cover_image_key, public=True)
        tags["og:image"] = img if img.startswith("http") else settings.app_url.rstrip("/") + img
    meta = "".join(f'<meta property="{k}" content="{html.escape(v, quote=True)}">' for k, v in tags.items())
    return meta + '<meta name="twitter:card" content="summary_large_image">'


@router.get("/h/{hazard_id}")
def hazard_page(hazard_id: int, request: Request, db: Session = Depends(get_db)):
    page = _index_html(request)
    h = db.get(Hazard, hazard_id)
    if h is not None and h.status in PUBLIC_STATUSES:
        tags = _og_tags(request, h)
        page = page.replace(OG_MARKER, tags) if OG_MARKER in page else page.replace("</head>", tags + "</head>", 1)
    return _html(page)


@router.get("/{path:path}")
def spa(path: str, request: Request):
    if path.startswith(RESERVED_PREFIXES):
        raise HTTPException(status_code=404)
    dist = _dist(request)
    if path:
        candidate = (dist / path).resolve()
        if candidate.is_relative_to(dist) and candidate.is_file():
            immutable = path.startswith("assets/")
            cache = "public, max-age=31536000, immutable" if immutable else "no-cache"
            return FileResponse(candidate, headers={"Cache-Control": cache})
        if "." in Path(path).name:  # a missing static file, not an app route
            raise HTTPException(status_code=404)
    return _html(_index_html(request))
