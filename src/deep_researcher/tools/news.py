"""News: NewsAPI when NEWSAPI_KEY is set (recent articles with descriptions), else GDELT (keyless).

GDELT's DOC 2.0 API returns titles and URLs only (no body) and asks for one request per 5 s.
"""
from __future__ import annotations

import os
import re
from typing import Any

from deep_researcher.tools.base import RetrievalTool, SearchResult, clip, iso_date


class NewsApiTool(RetrievalTool):
    name = "newsapi"
    source_type = "news"
    requires_env = ("NEWSAPI_KEY",)
    endpoint = "https://newsapi.org/v2/everything"

    async def _search(self, query: str, limit: int) -> list[SearchResult]:
        resp = await self._get(self.endpoint, params={"q": query, "pageSize": limit, "sortBy": "relevancy",
                                                      "language": "en"},
                               headers={"X-Api-Key": os.environ["NEWSAPI_KEY"]})
        return parse_newsapi(resp.json())


class GdeltTool(RetrievalTool):
    name = "gdelt"
    source_type = "news"
    min_interval_s = 5.5
    endpoint = "https://api.gdeltproject.org/api/v2/doc/doc"

    async def _search(self, query: str, limit: int) -> list[SearchResult]:
        query = gdelt_query(query)
        if not query:
            return []
        resp = await self._get(self.endpoint, params={"query": query, "mode": "artlist", "format": "json",
                                                      "maxrecords": max(limit, 5), "sort": "hybridrel",
                                                      "sourcelang": "english"})
        if "json" not in (resp.headers.get("content-type") or "") and not resp.text.lstrip().startswith("{"):
            raise ValueError(f"GDELT returned non-JSON: {resp.text[:120]!r}")
        return parse_gdelt(resp.json())


_GDELT_STOP = {"or", "and", "not", "the", "a", "an", "of", "in", "on", "to", "for", "do", "does", "did",
               "what", "which", "how", "where", "when", "why", "is", "are", "was", "were", "be", "by", "with"}


def gdelt_query(query: str, max_words: int = 8) -> str:
    """GDELT treats OR/AND/NOT and parentheses as operators and rejects long sentences; keep plain
    keywords (each must be >= 3 chars for GDELT) and drop boolean words."""
    tokens = re.sub(r"[^\w\s-]", " ", query).split()
    words = [w for w in tokens if len(w) >= 3 and w.lower() not in _GDELT_STOP]
    return " ".join(words[:max_words])


def parse_newsapi(payload: dict[str, Any]) -> list[SearchResult]:
    out: list[SearchResult] = []
    for a in payload.get("articles") or []:
        text = " ".join(x for x in (a.get("description"), a.get("content")) if x)
        out.append(SearchResult(
            title=a.get("title") or "", url=a.get("url") or "", text=clip(text), source_type="news",
            published_at=iso_date(a.get("publishedAt")),
            extra={"source": (a.get("source") or {}).get("name"), "author": a.get("author")},
        ))
    return out


def parse_gdelt(payload: dict[str, Any]) -> list[SearchResult]:
    out: list[SearchResult] = []
    for a in payload.get("articles") or []:
        title = a.get("title") or ""
        out.append(SearchResult(
            # GDELT gives no body: the title is the only text until the fetcher enriches it.
            title=title, url=a.get("url") or "", text=clip(title), source_type="news",
            published_at=iso_date(a.get("seendate")),
            extra={"domain": a.get("domain"), "country": a.get("sourcecountry"), "needs_fetch": True},
        ))
    return out
