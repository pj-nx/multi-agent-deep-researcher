"""Retrieval tools: papers (arXiv, Semantic Scholar), web/reports (Tavily), news (NewsAPI/GDELT),
Wikipedia."""

from deep_researcher.tools.base import RetrievalTool, SearchResult
from deep_researcher.tools.fetch import fetch_text
from deep_researcher.tools.registry import POLICIES, info, register, tools_for

__all__ = ["POLICIES", "RetrievalTool", "SearchResult", "fetch_text", "info", "register", "tools_for"]
