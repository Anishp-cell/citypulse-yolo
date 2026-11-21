"""Serves media from LocalStorage (dev/test). In production media comes straight from R2."""

import mimetypes

from fastapi import APIRouter, HTTPException, Request, Response

router = APIRouter(include_in_schema=False)


def _local(request: Request):
    from backend.services.storage import LocalStorage

    storage = request.app.state.storage
    if not isinstance(storage, LocalStorage):
        raise HTTPException(status_code=404)
    return storage


def _respond(data: bytes | None, key: str, cache: str) -> Response:
    if data is None:
        raise HTTPException(status_code=404)
    media_type = mimetypes.guess_type(key)[0] or "application/octet-stream"
    return Response(data, media_type=media_type, headers={"Cache-Control": cache})


@router.get("/media/public/{key:path}")
def public_media(key: str, request: Request):
    return _respond(_local(request).read(key, public=True), key, "public, max-age=86400")


@router.get("/media/private/{key:path}")
def private_media(key: str, exp: str, sig: str, request: Request):
    storage = _local(request)
    if not storage.verify_signature(key, exp, sig):
        raise HTTPException(status_code=403)
    return _respond(storage.read(key, public=False), key, "private, max-age=3600")
