# Copyright (c) 2026 C10 Hackathon G1
"""arXiv Atom API (no key). https://info.arxiv.org/help/api/user-manual.html"""
from __future__ import annotations

import xml.etree.ElementTree as ET

from deep_researcher.tools.base import RetrievalTool, SearchResult, clip, iso_date

_NS = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}


class ArxivTool(RetrievalTool):
    name = "arxiv"
    source_type = "papers"
    min_interval_s = 3.0     # arXiv asks for ~1 request / 3 s
    endpoint = "https://export.arxiv.org/api/query"

    async def _search(self, query: str, limit: int) -> list[SearchResult]:
        resp = await self._get(self.endpoint, params={"search_query": f"all:{query}", "max_results": limit,
                                                      "sortBy": "relevance"})
        return parse_arxiv_atom(resp.text)


def _primary_category(entry: ET.Element) -> str | None:
    node = entry.find("arxiv:primary_category", _NS)
    return node.get("term") if node is not None else None


def parse_arxiv_atom(xml_text: str) -> list[SearchResult]:
    root = ET.fromstring(xml_text)
    out: list[SearchResult] = []
    for e in root.findall("a:entry", _NS):
        title = " ".join((e.findtext("a:title", default="", namespaces=_NS) or "").split())
        summary = clip(e.findtext("a:summary", default="", namespaces=_NS) or "")
        abs_url = (e.findtext("a:id", default="", namespaces=_NS) or "").replace("http://", "https://")
        pdf_url = None
        for link in e.findall("a:link", _NS):
            if link.get("type") == "application/pdf" or link.get("title") == "pdf":
                pdf_url = link.get("href")
            elif link.get("rel") == "alternate" and link.get("href"):
                abs_url = link.get("href") or abs_url
        authors = [a.findtext("a:name", default="", namespaces=_NS) for a in e.findall("a:author", _NS)]
        out.append(SearchResult(
            title=title, url=abs_url, text=summary, source_type="papers",
            published_at=iso_date(e.findtext("a:published", namespaces=_NS)),
            extra={"pdf_url": pdf_url, "authors": authors[:10],
                   "doi": e.findtext("arxiv:doi", namespaces=_NS),
                   "primary_category": _primary_category(e)},
        ))
    return out
