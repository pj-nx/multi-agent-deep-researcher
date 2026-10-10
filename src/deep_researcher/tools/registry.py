"""Which tools serve which source type, and which of them are enabled by the environment.

Policy per source type:
- ``all``   every enabled tool runs (papers: arXiv + Semantic Scholar complement each other)
- ``first`` the first enabled tool runs (news: NewsAPI when a key exists, else GDELT)

Keyless tools (arXiv, Semantic Scholar, Wikipedia, GDELT) are always enabled. ``api`` has no
built-in tool: register domain APIs with :func:`register` (e.g. FRED, Polygon) when a project
needs structured data.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import httpx

from deep_researcher.schemas import SourceType
from deep_researcher.tools.arxiv import ArxivTool
from deep_researcher.tools.base import RetrievalTool
from deep_researcher.tools.news import GdeltTool, NewsApiTool
from deep_researcher.tools.semantic_scholar import SemanticScholarTool
from deep_researcher.tools.web_search import TavilyReportsTool, TavilyTool
from deep_researcher.tools.wikipedia import WikipediaTool


@dataclass
class SourcePolicy:
    tools: list[type[RetrievalTool]]
    mode: Literal["all", "first"] = "all"


POLICIES: dict[SourceType, SourcePolicy] = {
    "papers": SourcePolicy([ArxivTool, SemanticScholarTool], "all"),
    "web": SourcePolicy([TavilyTool], "all"),
    "news": SourcePolicy([NewsApiTool, GdeltTool], "first"),
    "wikipedia": SourcePolicy([WikipediaTool], "all"),
    "reports": SourcePolicy([TavilyReportsTool], "all"),
    "api": SourcePolicy([], "all"),
}


def register(source_type: SourceType, tool: type[RetrievalTool], *, first: bool = False) -> None:
    """Add a tool class for a source type (e.g. a domain API under 'api')."""
    policy = POLICIES.setdefault(source_type, SourcePolicy([], "all"))
    if tool not in policy.tools:
        policy.tools.insert(0, tool) if first else policy.tools.append(tool)


def tools_for(source_types: list[SourceType] | tuple[SourceType, ...],
              client: httpx.AsyncClient | None = None) -> list[RetrievalTool]:
    """Instantiate the enabled tools for the requested source types, de-duplicated by name."""
    chosen: dict[str, RetrievalTool] = {}
    for st in source_types:
        policy = POLICIES.get(st)
        if not policy:
            continue
        enabled = [t for t in policy.tools if t.enabled()]
        if policy.mode == "first":
            enabled = enabled[:1]
        for t in enabled:
            chosen.setdefault(t.name, t(client))
    return list(chosen.values())


def info() -> dict[str, Any]:
    """Health payload: per source type, each tool with its enabled flag and required env vars."""
    return {
        st: {"mode": p.mode, "tools": [t.describe() for t in p.tools],
             "active": [t.name for t in (p.tools if p.mode == "all" else p.tools[:1]) if t.enabled()]
             if p.mode == "all" else [t.name for t in p.tools if t.enabled()][:1]}
        for st, p in POLICIES.items()
    }
