"""Wikipedia via the MediaWiki action API (no key): search + intro extracts in one call."""
from __future__ import annotations

import os
from typing import Any

from deep_researcher.tools.base import RetrievalTool, SearchResult, clip, iso_date


class WikipediaTool(RetrievalTool):
    name = "wikipedia"
    source_type = "wikipedia"
    min_interval_s = 0.5

    @property
    def endpoint(self) -> str:
        lang = os.getenv("WIKIPEDIA_LANG", "en")
        return f"https://{lang}.wikipedia.org/w/api.php"

    async def _search(self, query: str, limit: int) -> list[SearchResult]:
        resp = await self._get(self.endpoint, params={
            "action": "query", "generator": "search", "gsrsearch": query, "gsrlimit": limit,
            "prop": "extracts|info", "inprop": "url", "explaintext": 1, "exintro": 1, "exlimit": "max",
            "format": "json", "formatversion": 2,
        })
        return parse_wikipedia(resp.json())


def parse_wikipedia(payload: dict[str, Any]) -> list[SearchResult]:
    pages = (payload.get("query") or {}).get("pages") or []
    if isinstance(pages, dict):                      # formatversion=1 shape
        pages = list(pages.values())
    out: list[SearchResult] = []
    for p in sorted(pages, key=lambda x: x.get("index", 0)):
        extract = p.get("extract") or ""
        if not extract:
            continue
        out.append(SearchResult(
            title=p.get("title") or "", url=p.get("fullurl") or p.get("canonicalurl") or "",
            text=clip(extract), source_type="wikipedia", published_at=iso_date(p.get("touched")),
            extra={"pageid": p.get("pageid"), "length": p.get("length")},
        ))
    return out
