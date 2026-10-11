"""Chroma collection for uploaded document sections (one record per section).

Same backend and embedding fingerprint rules as the evidence store; a separate collection so a
set's sections can be searched, listed and deleted on their own and never mix with run evidence.
Metadata per record: ``document_set_id``, ``doc_id``, ``filename``, ``kind``, ``page_start``,
``page_end``, ``section``, ``created_at`` plus the document-level manifest fields, so the manifest
can be rebuilt from the records alone (hosted deployments have no file system to keep).
"""

from __future__ import annotations

import functools
import hashlib
import logging
import os
import time
from dataclasses import dataclass
from typing import Any

from deep_researcher.documents import Section, document_url
from deep_researcher.schemas import DocumentInfo
from deep_researcher.store.evidence_store import EmbeddingMismatchError, _clean_meta

log = logging.getLogger(__name__)

DEFAULT_COLLECTION = "uploads"
_META_EMBEDDING_ID = "embedding_id"


@dataclass(frozen=True)
class SectionHit:
    set_id: str
    doc_id: str
    filename: str
    page_start: int
    page_end: int
    text: str
    distance: float

    @property
    def score(self) -> float:
        return max(0.0, 1.0 - self.distance)

    @property
    def pages_label(self) -> str:
        return (
            f"{self.page_start}" if self.page_start == self.page_end else f"{self.page_start}-{self.page_end}"
        )

    @property
    def url(self) -> str:
        return document_url(self.set_id, self.doc_id, self.page_start)

    @property
    def title(self) -> str:
        return f"{self.filename} p.{self.pages_label}"


def section_record_id(set_id: str, doc_id: str, index: int) -> str:
    return hashlib.sha1(f"{set_id}/{doc_id}#{index}".encode()).hexdigest()[:24]


class UploadStore:
    def __init__(
        self,
        embeddings: Any,
        *,
        embedding_id: str,
        client: Any = None,
        path: str | None = None,
        collection: str | None = None,
    ) -> None:
        import chromadb

        self.embeddings = embeddings
        self.embedding_id = embedding_id
        self.collection_name = collection or os.getenv("CHROMA_UPLOADS_COLLECTION") or DEFAULT_COLLECTION
        self.client = client or chromadb.PersistentClient(path=path or os.getenv("CHROMA_PATH") or ".chroma")
        existing = {c.name for c in self.client.list_collections()}
        if self.collection_name in existing:
            self.collection = self.client.get_collection(self.collection_name)
            stored = (self.collection.metadata or {}).get(_META_EMBEDDING_ID)
            if stored and stored != embedding_id:
                raise EmbeddingMismatchError(
                    f"collection {self.collection_name!r} was built with {stored!r} but the configured "
                    f"embeddings are {embedding_id!r}; use another CHROMA_UPLOADS_COLLECTION or re-upload"
                )
        else:
            self.collection = self.client.create_collection(
                self.collection_name,
                metadata={_META_EMBEDDING_ID: embedding_id},
                configuration={"hnsw": {"space": "cosine"}},
            )

    # ------------------------------------------------------------------ write
    def add_document(self, set_id: str, info: DocumentInfo, sections: list[Section]) -> int:
        """Embed and upsert a document's sections; a document with no text gets one placeholder record
        (``no_text`` = True) so it still appears in the manifest but never in search results."""
        now = time.time()
        base = {
            "document_set_id": set_id,
            "doc_id": info.doc_id,
            "filename": info.filename,
            "kind": info.kind,
            "pages": info.pages,
            "doc_sections": len(sections),
            "chars": info.chars,
            "no_text": info.no_text,
            "language": info.language,
            "created_at": now,
            "excerpt": info.excerpt[:300],
            "warnings": "; ".join(info.warnings),
        }
        if not sections:
            self.collection.upsert(
                ids=[section_record_id(set_id, info.doc_id, 0)],
                embeddings=[self.embeddings.embed_query(info.filename)],
                documents=[f"(no extractable text in {info.filename})"],
                metadatas=[
                    _clean_meta({**base, "no_text": True, "page_start": 0, "page_end": 0, "section": 0})
                ],
            )
            return 0
        vectors = self.embeddings.embed_documents([s.text for s in sections])
        self.collection.upsert(
            ids=[section_record_id(set_id, info.doc_id, s.index) for s in sections],
            embeddings=vectors,
            documents=[s.text for s in sections],
            metadatas=[
                _clean_meta({**base, "page_start": s.page_start, "page_end": s.page_end, "section": s.index})
                for s in sections
            ],
        )
        return len(sections)

    # ------------------------------------------------------------------ read
    def _records(self, set_id: str, *, include_text: bool = False) -> list[dict[str, Any]]:
        inc = ["metadatas", "documents"] if include_text else ["metadatas"]
        res = self.collection.get(where={"document_set_id": set_id}, include=inc)
        rows = []
        for i, meta in enumerate(res["metadatas"] or []):
            row = dict(meta)
            if include_text:
                row["text"] = res["documents"][i]
            rows.append(row)
        return rows

    def manifest(self, set_id: str) -> list[DocumentInfo]:
        docs: dict[str, DocumentInfo] = {}
        for meta in self._records(set_id):
            if meta["doc_id"] in docs:
                continue
            docs[meta["doc_id"]] = DocumentInfo(
                doc_id=meta["doc_id"],
                filename=meta["filename"],
                kind=meta["kind"],
                pages=int(meta.get("pages") or 0),
                sections=int(meta.get("doc_sections") or 0),
                chars=int(meta.get("chars") or 0),
                no_text=bool(meta.get("no_text")),
                warnings=[w for w in str(meta.get("warnings") or "").split("; ") if w],
                language=meta.get("language") or None,
                excerpt=str(meta.get("excerpt") or ""),
            )
        return sorted(docs.values(), key=lambda d: d.filename)

    def count(self, set_id: str | None = None) -> int:
        if set_id is None:
            return self.collection.count()
        return len(self._records(set_id))

    def search(self, set_id: str, query: str, *, k: int = 5) -> list[SectionHit]:
        n = self.collection.count()
        if n == 0:
            return []
        vec = self.embeddings.embed_query(query)
        res = self.collection.query(
            query_embeddings=[vec],
            n_results=min(max(k, 1), n),
            where={"$and": [{"document_set_id": set_id}, {"no_text": False}]},
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        for doc, meta, dist in zip(
            res["documents"][0], res["metadatas"][0], res["distances"][0], strict=True
        ):
            hits.append(
                SectionHit(
                    set_id=set_id,
                    doc_id=meta["doc_id"],
                    filename=meta["filename"],
                    page_start=int(meta["page_start"]),
                    page_end=int(meta["page_end"]),
                    text=doc,
                    distance=float(dist),
                )
            )
        return hits

    # ------------------------------------------------------------------ delete
    def delete_set(self, set_id: str) -> int:
        n = self.count(set_id)
        if n:
            self.collection.delete(where={"document_set_id": set_id})
        return n

    def expired_sets(self, days: float, *, now: float | None = None) -> list[str]:
        cutoff = (now or time.time()) - days * 86400
        res = self.collection.get(where={"created_at": {"$lt": cutoff}}, include=["metadatas"])
        return sorted({m["document_set_id"] for m in res["metadatas"] or []})

    def expire(self, days: float, *, now: float | None = None) -> list[str]:
        sets = self.expired_sets(days, now=now)
        for s in sets:
            self.delete_set(s)
        return sets


@functools.cache
def get_upload_store() -> UploadStore:
    """Process-wide uploads store on the configured Chroma backend and embeddings."""
    from deep_researcher import embeddings
    from deep_researcher.store.evidence_store import make_chroma_client

    return UploadStore(
        embeddings.get_embeddings(), embedding_id=embeddings.embedding_id(), client=make_chroma_client()
    )


def reset_upload_store_cache() -> None:
    get_upload_store.cache_clear()
