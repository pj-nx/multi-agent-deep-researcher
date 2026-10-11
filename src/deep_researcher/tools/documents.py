"""``documents`` source: similarity search over the sections of one uploaded document set.

Bound to a set per run (``documents.bind_documents``), never registered as a class-level tool, so a
run without uploads never sees the source type. Results are shaped like any other tool's so the
retriever's language gate, relevance gate, evidence ids, chunking and validation apply unchanged.
"""

from __future__ import annotations

import asyncio
from typing import Any, ClassVar

from deep_researcher.schemas import SourceType
from deep_researcher.tools.base import RetrievalTool, SearchResult


class DocumentsTool(RetrievalTool):
    name: ClassVar[str] = "documents"
    source_type: ClassVar[SourceType] = "documents"

    def __init__(self, set_id: str, store: Any = None, client: Any = None) -> None:
        super().__init__(client)
        self.set_id = set_id
        self._store = store

    @property
    def store(self) -> Any:
        if self._store is None:
            from deep_researcher.store.uploads import get_upload_store

            self._store = get_upload_store()
        return self._store

    @classmethod
    def enabled(cls) -> bool:
        return True

    async def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        """Local and fast, so it bypasses the shared result cache (whose key has no set id)."""
        return (await self._search_checked(query, limit))[0]

    async def _search(self, query: str, limit: int) -> list[SearchResult]:
        hits = await asyncio.to_thread(self.store.search, self.set_id, query, k=limit)
        return [
            SearchResult(
                title=h.title,
                url=h.url,
                text=h.text,
                source_type="documents",
                extra={
                    "doc_id": h.doc_id,
                    "page_start": h.page_start,
                    "page_end": h.page_end,
                    "filename": h.filename,
                    "score": round(h.score, 3),
                    "set_id": self.set_id,
                },
            )
            for h in hits
        ]
