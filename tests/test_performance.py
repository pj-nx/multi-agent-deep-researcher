# Copyright (c) 2026 C10 Hackathon G1
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import deep_researcher.llm as llm
from deep_researcher import reports
from deep_researcher.graph import select_gaps
from deep_researcher.schemas import EvidenceChunk
from deep_researcher.store import EvidenceStore


class _Embeddings:
    def __init__(self) -> None:
        self.document_batches: list[list[str]] = []
        self.query_calls = 0

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_batches.append(texts)
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        self.query_calls += 1
        return [1.0, 0.0]


class _Collection:
    def __init__(self) -> None:
        self.metadata: dict[str, Any] = {}
        self.upserts: list[dict[str, Any]] = []
        self.query_n_results: int | None = None

    def modify(self, *, metadata: dict[str, Any]) -> None:
        self.metadata = metadata

    def upsert(self, **kwargs: Any) -> None:
        self.upserts.append(kwargs)

    def query(self, *, n_results: int, **kwargs: Any) -> dict[str, Any]:
        self.query_n_results = n_results
        return {
            "ids": [["chunk-1"]],
            "documents": [["evidence text"]],
            "metadatas": [[{
                "evidence_id": "E-1",
                "chunk_index": 0,
                "sub_question_id": "sq1",
                "source_type": "papers",
                "title": "A paper",
                "url": "https://example.org/paper",
            }]],
            "distances": [[0.1]],
        }


def _chunk(index: int) -> EvidenceChunk:
    return EvidenceChunk(
        id=f"chunk-{index}",
        evidence_id=f"E-{index}",
        chunk_index=0,
        text=f"evidence {index}",
        sub_question_id="sq1",
        source_type="papers",
        title=f"Paper {index}",
        url=f"https://example.org/{index}",
    )


def _bare_store(embeddings: _Embeddings, collection: _Collection) -> EvidenceStore:
    store = object.__new__(EvidenceStore)
    store.embeddings = embeddings
    store.collection = collection
    store.embedding_id = "test-model"
    store.collection_name = "test"
    store.path = ":memory:"
    return store


def test_add_chunks_batches_embedding_requests_and_upserts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("deep_researcher.store.evidence_store.EMBEDDING_BATCH_SIZE", 2)
    embeddings = _Embeddings()
    collection = _Collection()
    store = _bare_store(embeddings, collection)

    written = store.add_chunks([_chunk(i) for i in range(5)])

    assert written == 5
    assert [len(batch) for batch in embeddings.document_batches] == [2, 2, 1]
    assert [len(batch["ids"]) for batch in collection.upserts] == [2, 2, 1]
    assert collection.metadata["embedding_dim"] == 2


def test_add_chunks_writes_nothing_when_a_later_batch_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("deep_researcher.store.evidence_store.EMBEDDING_BATCH_SIZE", 2)

    class _FailingEmbeddings(_Embeddings):
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            if len(self.document_batches) == 2:
                raise RuntimeError("rate limited")
            return super().embed_documents(texts)

    collection = _Collection()
    store = _bare_store(_FailingEmbeddings(), collection)

    with pytest.raises(RuntimeError, match="rate limited"):
        store.add_chunks([_chunk(i) for i in range(5)])

    assert collection.upserts == []
    assert collection.metadata == {}


def test_query_reads_collection_count_only_once() -> None:
    embeddings = _Embeddings()
    collection = _Collection()
    store = _bare_store(embeddings, collection)
    count_calls = 0

    def count() -> int:
        nonlocal count_calls
        count_calls += 1
        return 3

    store.count = count  # type: ignore[method-assign]

    hits = store.query("evidence", k=9)

    assert count_calls == 1
    assert collection.query_n_results == 3
    assert embeddings.query_calls == 1
    assert hits[0].chunk.id == "chunk-1"


def test_report_filenames_do_not_overwrite_same_second_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(reports.time, "strftime", lambda *_: "20260101-120000")

    first = reports.write_report("same question", "# first", {"title": "first"}, tmp_path, pdf=False)
    second = reports.write_report("same question", "# second", {"title": "second"}, tmp_path, pdf=False)

    assert first[0].name == "20260101-120000-same-question.md"
    assert second[0].name == "20260101-120000-same-question-2.md"
    assert first[0].read_text(encoding="utf-8") == "# first"
    assert second[0].read_text(encoding="utf-8") == "# second"
    assert first[1] is not None and second[1] is not None
    assert first[1].read_text(encoding="utf-8") != second[1].read_text(encoding="utf-8")


def test_zero_gap_limit_does_not_schedule_a_branch() -> None:
    gap = {"question": "Which evidence is missing?", "parent_sub_question_id": "sq1"}

    assert select_gaps([gap], {}, limit=0) == []
    assert select_gaps([gap], {}, limit=-1) == []


@pytest.mark.parametrize("value", ["0", "-1"])
def test_llm_token_budget_must_be_positive(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("OPENROUTER_MAX_TOKENS", value)

    with pytest.raises(llm.LLMConfigError, match="positive integer"):
        llm.max_tokens("planner")
