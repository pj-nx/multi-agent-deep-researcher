# Copyright (c) 2026 C10 Hackathon G1
"""Persistent vector store for retrieved evidence, backed by Chroma.

Why Chroma: embedded, zero-ops, persists to a directory, and the same code talks to Chroma
Cloud / a Chroma server by swapping the client. The store never embeds on its own; it takes a
LangChain ``Embeddings`` so the model choice (local, Ollama, OpenAI-compatible) lives in
``deep_researcher.embeddings``.

Invariants
- Every chunk id is deterministic (sha1 of url + chunk index), so re-ingesting a source is an
  idempotent upsert rather than a duplicate.
- A collection records the ``embedding_id`` that built it; opening it with a different model
  raises ``EmbeddingMismatchError`` instead of silently returning garbage similarities.
- Chroma is not multi-process safe on one path: one writer process per CHROMA_PATH.
"""
from __future__ import annotations

import functools
import hashlib
import logging
import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from deep_researcher.schemas import Evidence, EvidenceChunk
from deep_researcher.store.chunking import chunk_text

log = logging.getLogger(__name__)

DEFAULT_PATH = ".chroma"
DEFAULT_COLLECTION = "evidence"
_META_EMBEDDING_ID = "embedding_id"
_META_EMBEDDING_DIM = "embedding_dim"
EMBEDDING_BATCH_SIZE = 128


class EmbeddingMismatchError(RuntimeError):
    pass


@dataclass(frozen=True)
class Hit:
    chunk: EvidenceChunk
    distance: float

    @property
    def score(self) -> float:
        """Cosine similarity in [0, 1] (Chroma returns cosine *distance*)."""
        return max(0.0, 1.0 - self.distance)


_CHUNK_FIELDS = frozenset(EvidenceChunk.model_fields) - {"id", "text"}


def _chunk_fields(meta: dict[str, Any]) -> dict[str, Any]:
    """Metadata may carry validator extras (relevance, keep); keep only EvidenceChunk fields."""
    return {k: v for k, v in meta.items() if k in _CHUNK_FIELDS}


def chunk_id(url: str, index: int) -> str:
    return hashlib.sha1(f"{url}#{index}".encode()).hexdigest()[:24]


def _clean_meta(d: dict[str, Any]) -> dict[str, str | int | float | bool]:
    """Chroma metadata must be flat scalars; drop None and stringify the rest."""
    out: dict[str, str | int | float | bool] = {}
    for k, v in d.items():
        if v is None:
            continue
        out[k] = v if isinstance(v, str | int | float | bool) else str(v)
    return out


class EvidenceStore:
    def __init__(self, embeddings: Any, *, embedding_id: str, path: str | None = None,
                 collection: str | None = None, client: Any = None) -> None:
        import chromadb

        self.embeddings = embeddings
        self.embedding_id = embedding_id
        self.path = path or os.getenv("CHROMA_PATH") or DEFAULT_PATH
        self.collection_name = collection or os.getenv("CHROMA_COLLECTION") or DEFAULT_COLLECTION
        self.client = client or chromadb.PersistentClient(path=self.path)
        self.collection = self._open()

    # ------------------------------------------------------------------ setup
    def _open(self):
        existing = {c.name for c in self.client.list_collections()}
        if self.collection_name in existing:
            coll = self.client.get_collection(self.collection_name)
            stored = (coll.metadata or {}).get(_META_EMBEDDING_ID)
            if stored and stored != self.embedding_id:
                raise EmbeddingMismatchError(
                    f"collection {self.collection_name!r} at {self.path} was built with {stored!r} "
                    f"but the configured embeddings are {self.embedding_id!r}. Use a different "
                    "CHROMA_COLLECTION or re-ingest."
                )
            return coll
        return self.client.create_collection(
            self.collection_name,
            metadata={_META_EMBEDDING_ID: self.embedding_id},
            configuration={"hnsw": {"space": "cosine"}},
        )

    def _record_dim(self, dim: int) -> None:
        meta = dict(self.collection.metadata or {})
        if meta.get(_META_EMBEDDING_DIM) in (None, dim):
            if _META_EMBEDDING_DIM not in meta:
                meta[_META_EMBEDDING_DIM] = dim
                self.collection.modify(metadata=meta)
            return
        raise EmbeddingMismatchError(
            f"collection {self.collection_name!r} holds {meta[_META_EMBEDDING_DIM]}-d vectors; "
            f"the configured model produced {dim}-d"
        )

    # ------------------------------------------------------------------ write
    def add_evidence(self, items: Iterable[Evidence], *, chunk_size: int | None = None,
                     overlap: int | None = None, unit: str | None = None) -> int:
        """Chunk (CHUNK_SIZE / CHUNK_OVERLAP / CHUNK_UNIT defaults), embed and upsert.

        Returns the number of chunks written.
        """
        chunks: list[EvidenceChunk] = []
        for ev in items:
            for i, piece in enumerate(chunk_text(ev.text, size=chunk_size, overlap=overlap, unit=unit)):
                chunks.append(EvidenceChunk(
                    id=chunk_id(ev.url, i), evidence_id=ev.id, chunk_index=i, text=piece,
                    sub_question_id=ev.sub_question_id, source_type=ev.source_type, title=ev.title,
                    url=ev.url, published_at=ev.published_at, credibility=ev.credibility,
                    language=ev.language,
                ))
        return self.add_chunks(chunks)

    def add_chunks(self, chunks: Sequence[EvidenceChunk]) -> int:
        if not chunks:
            return 0
        # Embed every batch before writing any: a provider failure part-way leaves the collection untouched.
        batches: list[tuple[Sequence[EvidenceChunk], list[list[float]]]] = []
        for start in range(0, len(chunks), EMBEDDING_BATCH_SIZE):
            batch = chunks[start:start + EMBEDDING_BATCH_SIZE]
            vectors = self.embeddings.embed_documents([c.text for c in batch])
            if len(vectors) != len(batch) or not vectors:
                raise ValueError(
                    f"embedding provider returned {len(vectors)} vectors for {len(batch)} chunks"
                )
            batches.append((batch, vectors))
        dimensions = {len(vector) for _, vectors in batches for vector in vectors}
        if len(dimensions) != 1:
            raise ValueError(f"embedding provider returned inconsistent vector dimensions: {dimensions}")
        self._record_dim(dimensions.pop())
        for batch, vectors in batches:
            self.collection.upsert(
                ids=[c.id for c in batch],
                embeddings=vectors,
                documents=[c.text for c in batch],
                metadatas=[_clean_meta(c.model_dump(exclude={"id", "text"})) for c in batch],
            )
        return len(chunks)

    # ------------------------------------------------------------------ read
    def query(self, text: str, *, k: int = 10, sub_question_id: str | None = None,
              sub_question_ids: Sequence[str] | None = None, evidence_ids: Sequence[str] | None = None,
              where: dict[str, Any] | None = None) -> list[Hit]:
        count = self.count()
        if count == 0 or k <= 0:
            return []
        clauses: list[dict[str, Any]] = []
        if sub_question_id:
            clauses.append({"sub_question_id": sub_question_id})
        if sub_question_ids:
            clauses.append({"sub_question_id": {"$in": list(sub_question_ids)}})
        if evidence_ids is not None:
            if not evidence_ids:
                return []
            clauses.append({"evidence_id": {"$in": list(evidence_ids)}})
        if where:
            clauses.append(where)
        filt = None if not clauses else (clauses[0] if len(clauses) == 1 else {"$and": clauses})
        vec = self.embeddings.embed_query(text)
        res = self.collection.query(query_embeddings=[vec], n_results=min(k, count), where=filt,
                                    include=["documents", "metadatas", "distances"])
        hits: list[Hit] = []
        for cid, doc, meta, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0],
                                        res["distances"][0], strict=True):
            chunk = EvidenceChunk(id=cid, text=doc, **_chunk_fields(meta))
            hits.append(Hit(chunk=chunk, distance=float(dist)))
        return hits

    def get(self, ids: Sequence[str]) -> list[EvidenceChunk]:
        res = self.collection.get(ids=list(ids), include=["documents", "metadatas"])
        return [EvidenceChunk(id=cid, text=doc, **_chunk_fields(meta))
                for cid, doc, meta in zip(res["ids"], res["documents"], res["metadatas"], strict=True)]

    def update_assessments(self, assessments: dict[str, dict[str, Any]]) -> int:
        """Write validator verdicts (credibility, relevance, keep) onto every chunk of each evidence id."""
        if not assessments:
            return 0
        res = self.collection.get(where={"evidence_id": {"$in": list(assessments)}}, include=["metadatas"])
        ids, metas = [], []
        for cid, meta in zip(res["ids"], res["metadatas"], strict=True):
            a = assessments.get(meta.get("evidence_id"))
            if not a:
                continue
            new = dict(meta)
            for key in ("credibility", "relevance", "keep"):
                if key in a and a[key] is not None:
                    new[key] = a[key]
            ids.append(cid)
            metas.append(_clean_meta(new))
        if ids:
            self.collection.update(ids=ids, metadatas=metas)
        return len(ids)

    def has_url(self, url: str) -> bool:
        res = self.collection.get(where={"url": url}, limit=1, include=[])
        return bool(res["ids"])

    def count(self) -> int:
        return self.collection.count()

    def reset(self) -> None:
        """Drop and recreate the collection (keeps the embedding fingerprint)."""
        self.client.delete_collection(self.collection_name)
        self.collection = self._open()

    def info(self) -> dict[str, Any]:
        return {"backend": "chroma", "path": self.path, "collection": self.collection_name,
                "count": self.count(), "metadata": dict(self.collection.metadata or {})}


def chroma_mode() -> str:
    return (os.getenv("CHROMA_MODE") or "persistent").lower()


def make_chroma_client() -> Any:
    """Chroma client for CHROMA_MODE: persistent (CHROMA_PATH, embedded, single writer), http
    (CHROMA_HOST/CHROMA_PORT/CHROMA_SSL, optional CHROMA_API_KEY header) or cloud (CHROMA_API_KEY,
    CHROMA_TENANT, CHROMA_DATABASE). The same EvidenceStore runs on all three."""
    import chromadb

    mode = chroma_mode()
    if mode == "persistent":
        return chromadb.PersistentClient(path=os.getenv("CHROMA_PATH") or DEFAULT_PATH)
    if mode == "http":
        host = os.getenv("CHROMA_HOST")
        if not host:
            raise ValueError("CHROMA_MODE=http needs CHROMA_HOST (and CHROMA_PORT, default 8000)")
        headers = {"x-chroma-token": os.environ["CHROMA_API_KEY"]} if os.getenv("CHROMA_API_KEY") else None
        return chromadb.HttpClient(host=host, port=int(os.getenv("CHROMA_PORT") or 8000),
                                   ssl=(os.getenv("CHROMA_SSL") or "false").lower() in {"1", "true", "yes"},
                                   headers=headers)
    if mode == "cloud":
        missing = [k for k in ("CHROMA_API_KEY", "CHROMA_TENANT", "CHROMA_DATABASE") if not os.getenv(k)]
        if missing:
            raise ValueError(f"CHROMA_MODE=cloud needs {', '.join(missing)}")
        return chromadb.CloudClient(tenant=os.environ["CHROMA_TENANT"],
                                    database=os.environ["CHROMA_DATABASE"],
                                    api_key=os.environ["CHROMA_API_KEY"])
    raise ValueError(f"CHROMA_MODE={mode!r}; expected persistent, http or cloud")


def store_info() -> dict[str, Any]:
    mode = chroma_mode()
    out: dict[str, Any] = {"backend": "chroma", "mode": mode,
                           "collection": os.getenv("CHROMA_COLLECTION") or DEFAULT_COLLECTION}
    if mode == "persistent":
        out["path"] = os.getenv("CHROMA_PATH") or DEFAULT_PATH
    elif mode == "http":
        out["host"] = os.getenv("CHROMA_HOST")
        out["port"] = os.getenv("CHROMA_PORT") or "8000"
    else:
        out["tenant"] = os.getenv("CHROMA_TENANT")
        out["database"] = os.getenv("CHROMA_DATABASE")
    return out


@functools.cache
def get_store() -> EvidenceStore:
    """Process-wide store built from the configured embeddings and CHROMA_MODE."""
    from deep_researcher import embeddings

    return EvidenceStore(embeddings.get_embeddings(), embedding_id=embeddings.embedding_id(),
                         client=make_chroma_client())


def reset_store_cache() -> None:
    get_store.cache_clear()
