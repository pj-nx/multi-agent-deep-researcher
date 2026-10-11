# Copyright (c) 2026 C10 Hackathon G1
"""Semantic Scholar Graph API. Keyless at a low shared rate limit; SEMANTIC_SCHOLAR_API_KEY raises it."""
from __future__ import annotations

import os
from typing import Any

from deep_researcher.tools.base import RetrievalTool, SearchResult, clip, iso_date

_FIELDS = "title,abstract,url,year,publicationDate,citationCount,openAccessPdf,externalIds,venue,authors"


class SemanticScholarTool(RetrievalTool):
    name = "semantic_scholar"
    source_type = "papers"
    min_interval_s = 1.1     # unauthenticated: ~1 request / s shared pool; 429s are common, hence the retry
    max_retries = 2
    endpoint = "https://api.semanticscholar.org/graph/v1/paper/search"

    async def _search(self, query: str, limit: int) -> list[SearchResult]:
        headers = {}
        if key := os.getenv("SEMANTIC_SCHOLAR_API_KEY"):
            headers["x-api-key"] = key
        resp = await self._get(self.endpoint, params={"query": query, "limit": limit, "fields": _FIELDS},
                               headers=headers)
        return parse_s2(resp.json())


def parse_s2(payload: dict[str, Any]) -> list[SearchResult]:
    out: list[SearchResult] = []
    for p in payload.get("data") or []:
        abstract = p.get("abstract") or ""
        if not abstract:
            continue                      # no text to ground a claim on
        ext = p.get("externalIds") or {}
        oa = p.get("openAccessPdf") or {}
        url = p.get("url") or (f"https://doi.org/{ext['DOI']}" if ext.get("DOI") else "")
        out.append(SearchResult(
            title=p.get("title") or "", url=url, text=clip(abstract), source_type="papers",
            published_at=iso_date(p.get("publicationDate") or (str(p["year"]) if p.get("year") else None)),
            extra={"pdf_url": oa.get("url"), "citation_count": p.get("citationCount"),
                   "venue": p.get("venue"), "doi": ext.get("DOI"), "arxiv": ext.get("ArXiv"),
                   "authors": [a.get("name") for a in (p.get("authors") or [])[:10]]},
        ))
    return out
