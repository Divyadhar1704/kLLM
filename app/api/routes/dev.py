"""Developer test console (GET /dev). Registered only when ENABLE_DEV_UI is on and not in
production: it is a testing tool, not the farmer-facing frontend."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter(include_in_schema=False)

_PAGE = Path(__file__).resolve().parents[2] / "web" / "index.html"

# The page has no external resources; scripts and styles are inline.
_CSP = (
    "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; "
    # blob: previews the user's own photo / video in the page (the local file, never re-fetched)
    "img-src 'self' data: blob:; media-src blob:; connect-src 'self'; frame-ancestors 'none'; "
    "base-uri 'none'"
)


@lru_cache(maxsize=1)
def _page() -> str:
    return _PAGE.read_text(encoding="utf-8")


@router.get("/dev", response_class=HTMLResponse)
async def dev_console() -> HTMLResponse:
    return HTMLResponse(
        _page(), headers={"Content-Security-Policy": _CSP, "Cache-Control": "no-store"}
    )
