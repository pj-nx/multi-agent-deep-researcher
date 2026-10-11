# Copyright (c) 2026 C10 Hackathon G1
"""Tavily web search (TAVILY_API_KEY). Also used for 'reports' with a query hint towards PDFs/reports."""
from __future__ import annotations

import os
from typing import Any, ClassVar

from deep_researcher.schemas import SourceType
from deep_researcher.tools.base import RetrievalTool, SearchResult, clip, iso_date


class TavilyTool(RetrievalTool):
    name = "tavily"
    source_type: ClassVar[SourceType] = "web"
    requires_env = ("TAVILY_API_KEY",)
    endpoint = "https://api.tavily.com/search"
    search_depth = "advanced"

    async def _search(self, query: str, limit: int) -> list[SearchResult]:
        body: dict[str, Any] = {
            "api_key": os.environ["TAVILY_API_KEY"], "query": query, "max_results": limit,
            "search_depth": self.search_depth, "include_answer": False, "include_raw_content": False,
        }
        resp = await self._request("POST", self.endpoint, json=body)
        return parse_tavily(resp.json(), self.source_type)


class TavilyReportsTool(TavilyTool):
    """Same API, steered towards long-form reports and PDFs; results tagged as 'reports'."""

    name = "tavily_reports"
    source_type: ClassVar[SourceType] = "reports"

    async def _search(self, query: str, limit: int) -> list[SearchResult]:
        return await super()._search(f"{query} report OR whitepaper OR filetype:pdf", limit)


def parse_tavily(payload: dict[str, Any], source_type: SourceType = "web") -> list[SearchResult]:
    out: list[SearchResult] = []
    for r in payload.get("results") or []:
        text = r.get("raw_content") or r.get("content") or ""
        out.append(SearchResult(
            title=r.get("title") or "", url=r.get("url") or "", text=clip(text), source_type=source_type,
            published_at=iso_date(r.get("published_date")),
            extra={"score": r.get("score")},
        ))
    return out
