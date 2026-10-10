"""Persistent evidence store (vector DB) and chunking."""

from deep_researcher.store.chunking import chunk_text, count_tokens
from deep_researcher.store.evidence_store import (
    EmbeddingMismatchError,
    EvidenceStore,
    Hit,
    get_store,
    make_chroma_client,
    reset_store_cache,
    store_info,
)

__all__ = ["EmbeddingMismatchError", "EvidenceStore", "Hit", "chunk_text", "count_tokens", "get_store",
           "make_chroma_client", "reset_store_cache", "store_info"]
