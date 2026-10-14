# Copyright (c) 2026 C10 Hackathon G1
"""Persistent evidence store (vector DB) and chunking."""

from deep_researcher.store.chunking import chunk_text, count_tokens
from deep_researcher.store.evidence_store import (
    EMBEDDING_BATCH_SIZE,
    EmbeddingMismatchError,
    EvidenceStore,
    Hit,
    get_store,
    make_chroma_client,
    reset_store_cache,
    store_info,
)

__all__ = ["EMBEDDING_BATCH_SIZE", "EmbeddingMismatchError", "EvidenceStore", "Hit", "chunk_text",
           "count_tokens", "get_store",
           "make_chroma_client", "reset_store_cache", "store_info"]
