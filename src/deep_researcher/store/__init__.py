"""Persistent evidence store (vector DB) and chunking."""

from deep_researcher.store.chunking import chunk_text
from deep_researcher.store.evidence_store import EmbeddingMismatchError, EvidenceStore, Hit

__all__ = ["EmbeddingMismatchError", "EvidenceStore", "Hit", "chunk_text"]
