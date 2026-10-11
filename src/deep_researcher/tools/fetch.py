# Copyright (c) 2026 C10 Hackathon G1
"""Full-text fetcher: PDFs via pypdf, HTML via a tag-stripping parser. Size- and page-capped."""
from __future__ import annotations

import io
import logging
import os
import re
from html.parser import HTMLParser

import httpx

from deep_researcher.tools.base import clip, shared_client

log = logging.getLogger(__name__)

MAX_BYTES = int(os.getenv("RETRIEVAL_FETCH_MAX_BYTES") or 15_000_000)
MAX_PAGES = int(os.getenv("RETRIEVAL_FETCH_MAX_PAGES") or 30)
MAX_CHARS = int(os.getenv("RETRIEVAL_FETCH_MAX_CHARS") or 60_000)


class _TextExtractor(HTMLParser):
    _SKIP = {"script", "style", "noscript", "svg", "nav", "footer", "header", "form"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip_depth = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP:
            self._skip_depth += 1
        elif tag in {"p", "br", "div", "li", "h1", "h2", "h3", "h4", "tr"}:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self._SKIP and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data):
        if not self._skip_depth:
            self.parts.append(data)


def html_to_text(html: str) -> str:
    p = _TextExtractor()
    p.feed(html)
    text = "".join(p.parts).replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def pdf_to_text(data: bytes, max_pages: int = MAX_PAGES) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    pages = [page.extract_text() or "" for page in reader.pages[:max_pages]]
    return "\n\n".join(pages).strip()


async def fetch_text(url: str, *, client: httpx.AsyncClient | None = None, max_bytes: int = MAX_BYTES,
                     max_chars: int = MAX_CHARS) -> tuple[str, str] | None:
    """Return ``(text, kind)`` with kind in {'pdf', 'html'}, or None when the page is unusable.

    Never raises: fetching is best-effort enrichment.
    """
    client = client or shared_client()
    try:
        async with client.stream("GET", url) as resp:
            if resp.status_code != 200:
                return None
            ctype = (resp.headers.get("content-type") or "").lower()
            buf = bytearray()
            async for chunk in resp.aiter_bytes():
                buf.extend(chunk)
                if len(buf) > max_bytes:
                    log.info("fetch_text: %s exceeds %d bytes; truncated", url, max_bytes)
                    break
        data = bytes(buf)
        is_pdf = "pdf" in ctype or url.lower().split("?")[0].endswith(".pdf") or data[:5] == b"%PDF-"
        if is_pdf:
            return clip(pdf_to_text(data), max_chars), "pdf"
        if "html" in ctype or "xml" in ctype or b"<html" in data[:2000].lower():
            html = data.decode(resp.encoding or "utf-8", errors="replace")
            return clip(html_to_text(html), max_chars), "html"
        return None
    except Exception as exc:  # noqa: BLE001 - enrichment must never break retrieval
        log.info("fetch_text: %s failed: %s", url, exc)
        return None
