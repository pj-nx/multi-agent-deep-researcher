import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

pytest.importorskip("chromadb")
from langchain_core.embeddings import DeterministicFakeEmbedding

from deep_researcher.schemas import Evidence
from deep_researcher.store import EmbeddingMismatchError, EvidenceStore, chunk_text, count_tokens


def _ev(i: int, sq: str = "sq1", text: str | None = None) -> Evidence:
    return Evidence(id=f"E{i}", sub_question_id=sq, source_type="papers", title=f"Paper {i}",
                    url=f"https://example.org/p{i}", text=text or f"Remote work study number {i}. " * 20,
                    published_at="2025-01-0" + str(i % 9 + 1), credibility=0.8)


def _store(tmp_path, *, embedding_id="fake:32", collection="evidence", size=32):
    return EvidenceStore(DeterministicFakeEmbedding(size=size), embedding_id=embedding_id,
                         path=str(tmp_path / "chroma"), collection=collection)


def test_chunking_chars_overlap_and_boundaries():
    text = ("Sentence one is here. " * 30).strip()
    chunks = chunk_text(text, size=200, overlap=40, unit="chars")
    assert len(chunks) > 1 and all(len(c) <= 200 for c in chunks)
    assert "".join(chunks).count("Sentence one") >= 30          # nothing lost
    assert chunk_text("   ") == []
    with pytest.raises(ValueError):
        chunk_text("x", size=10, overlap=10)
    with pytest.raises(ValueError):
        chunk_text("x", unit="words")


def test_chunking_default_is_512_tokens(monkeypatch):
    for v in ("CHUNK_SIZE", "CHUNK_OVERLAP", "CHUNK_UNIT"):
        monkeypatch.delenv(v, raising=False)
    para = "Remote work studies report mixed productivity effects across sectors and seniority levels. "
    text = (para * 120).strip()                                  # ~ 2k tokens
    chunks = chunk_text(text)
    assert len(chunks) >= 3
    assert all(count_tokens(c) <= 512 for c in chunks)
    assert max(count_tokens(c) for c in chunks) > 300            # budget actually used, not tiny pieces
    assert all(c.endswith(".") for c in chunks[:-1])             # sentence boundaries respected
    assert chunk_text("short text") == ["short text"]            # fits: returned whole


def test_chunking_env_overrides(monkeypatch):
    monkeypatch.setenv("CHUNK_SIZE", "64")
    monkeypatch.setenv("CHUNK_OVERLAP", "8")
    text = ("Alpha beta gamma delta. " * 60).strip()
    chunks = chunk_text(text)
    assert len(chunks) > 3 and all(count_tokens(c) <= 64 for c in chunks)
    monkeypatch.setenv("CHUNK_UNIT", "chars")
    assert all(len(c) <= 64 for c in chunk_text(text))


def test_persists_across_reopen_and_upsert_is_idempotent(tmp_path):
    s1 = _store(tmp_path)
    n = s1.add_evidence([_ev(1), _ev(2)], chunk_size=200, overlap=20, unit="chars")
    assert n == s1.count() > 2
    assert s1.add_evidence([_ev(1)], chunk_size=200, overlap=20, unit="chars") > 0
    assert s1.count() == n                                       # re-ingest: no duplicates
    assert s1.has_url("https://example.org/p1") and not s1.has_url("https://example.org/nope")
    del s1

    s2 = _store(tmp_path)                                        # same path, fresh process-like open
    assert s2.count() == n
    assert s2.info()["metadata"]["embedding_id"] == "fake:32"
    assert s2.info()["metadata"]["embedding_dim"] == 32


def test_query_filters_by_sub_question_and_roundtrips_metadata(tmp_path):
    s = _store(tmp_path)
    s.add_evidence([_ev(1, "sq1", "Four-day week trials in the UK raised productivity."),
                    _ev(2, "sq2", "Chroma is a vector database for embeddings.")])
    hits = s.query("productivity trials", k=5, sub_question_id="sq1")
    assert hits and all(h.chunk.sub_question_id == "sq1" for h in hits)
    h = hits[0]
    assert h.chunk.url == "https://example.org/p1" and h.chunk.source_type == "papers"
    assert h.chunk.credibility == 0.8 and 0.0 <= h.score <= 1.0
    assert s.get([h.chunk.id])[0].text == h.chunk.text
    assert s.query("anything", sub_question_id="missing") == []


def test_refuses_different_embedding_model(tmp_path):
    _store(tmp_path).add_evidence([_ev(1)])
    with pytest.raises(EmbeddingMismatchError, match="fake:32"):
        _store(tmp_path, embedding_id="local:other-model")
    # a different collection name on the same path is fine
    other = _store(tmp_path, embedding_id="local:other-model", collection="evidence_v2")
    assert other.count() == 0


def test_refuses_dimension_drift(tmp_path):
    _store(tmp_path).add_evidence([_ev(1)])
    drifted = _store(tmp_path, size=64)          # same id claimed, different vector size
    with pytest.raises(EmbeddingMismatchError, match="32-d"):
        drifted.add_evidence([_ev(2)])


def test_reset_keeps_fingerprint(tmp_path):
    s = _store(tmp_path)
    s.add_evidence([_ev(1)])
    s.reset()
    assert s.count() == 0 and s.info()["metadata"]["embedding_id"] == "fake:32"
