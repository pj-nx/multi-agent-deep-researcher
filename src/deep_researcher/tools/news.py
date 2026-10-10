"""News: newsdata.io when NEWSDATA_API_KEY is set, else GDELT (keyless).

newsdata.io ``/api/1/latest`` covers recent news (roughly the last 48 hours; older articles need the
paid ``archive`` endpoint). Free plan: up to 10 articles per request, 30 credits per 15 minutes,
``description`` but no full ``content``. GDELT's DOC 2.0 API returns titles and URLs only (no body)
and asks for one request per 5 s.
"""
from __future__ import annotations

import os
import re
from typing import Any

from deep_researcher.tools.base import RetrievalTool, SearchResult, clip, iso_date


class NewsDataTool(RetrievalTool):
    name = "newsdata"
    source_type = "news"
    requires_env = ("NEWSDATA_API_KEY",)
    min_interval_s = 0.5
    endpoint = "https://newsdata.io/api/1/latest"
    max_size = int(os.getenv("NEWSDATA_MAX_SIZE") or 10)      # 10 on the free plan, 50 on paid

    async def _search(self, query: str, limit: int) -> list[SearchResult]:
        params: dict[str, Any] = {"q": newsdata_query(query),
                                  "language": os.getenv("NEWSDATA_LANGUAGE", "en"),
                                  "size": max(1, min(limit, self.max_size)), "removeduplicate": 1}
        if country := os.getenv("NEWSDATA_COUNTRY"):
            params["country"] = country
        # The key travels in a header, never in the URL (query strings end up in logs).
        resp = await self._get(self.endpoint, params=params,
                               headers={"X-ACCESS-KEY": os.environ["NEWSDATA_API_KEY"]})
        payload = resp.json()
        if payload.get("status") != "success":
            err = payload.get("results") or payload.get("message") or payload
            raise ValueError(f"newsdata.io error: {str(err)[:200]}")
        return parse_newsdata(payload)


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


_PAID_ONLY = "ONLY AVAILABLE IN"


def newsdata_query(query: str, max_chars: int = 512) -> str:
    """newsdata.io accepts plain keywords (AND/OR/NOT are operators); keep it simple and <= 512 chars."""
    words = [w for w in re.sub(r"[^\w\s-]", " ", query).split() if w.upper() not in {"AND", "OR", "NOT"}]
    return " ".join(words)[:max_chars].strip()


def parse_newsdata(payload: dict[str, Any]) -> list[SearchResult]:
    out: list[SearchResult] = []
    for a in payload.get("results") or []:
        content = a.get("content") or ""
        if _PAID_ONLY in content.upper():
            content = ""                                  # free plan placeholder, not article text
        text = " ".join(x for x in (a.get("description"), content) if x)
        out.append(SearchResult(
            title=a.get("title") or "", url=a.get("link") or "", text=clip(text), source_type="news",
            published_at=iso_date(a.get("pubDate")),
            extra={"source": a.get("source_name") or a.get("source_id"),
                   "source_priority": a.get("source_priority"), "country": a.get("country"),
                   "category": a.get("category"), "creator": a.get("creator"),
                   "needs_fetch": not content},
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
