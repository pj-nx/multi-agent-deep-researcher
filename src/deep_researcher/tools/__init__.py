# Copyright (c) 2026 C10 Hackathon G1
"""Retrieval tools: papers (arXiv, Semantic Scholar), web/reports (Tavily), news (newsdata.io/GDELT),
Wikipedia."""

from deep_researcher.tools.base import RetrievalTool, SearchResult
from deep_researcher.tools.fetch import fetch_text
from deep_researcher.tools.registry import POLICIES, info, register, tools_for

__all__ = ["POLICIES", "RetrievalTool", "SearchResult", "fetch_text", "info", "register", "tools_for"]
