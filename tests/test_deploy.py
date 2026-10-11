# Copyright (c) 2026 C10 Hackathon G1
"""Hosted-deployment wiring: Postgres checkpointer, Chroma http/cloud clients, health payload.

Live checks run only when a local Postgres (DR_TEST_DATABASE_URL) or Chroma server
(DR_TEST_CHROMA_HOST/PORT) is reachable; otherwise they are skipped.
"""
import os
import socket
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from deep_researcher import graph as G
from deep_researcher.store import evidence_store as ES

pytestmark = pytest.mark.real_graph          # conftest must not substitute canned streams here


def _port_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


@pytest.fixture
def clean_env(monkeypatch):
    for k in ("CHECKPOINTER", "DATABASE_URL", "CHROMA_MODE", "CHROMA_HOST", "CHROMA_PORT", "CHROMA_SSL",
              "CHROMA_API_KEY", "CHROMA_TENANT", "CHROMA_DATABASE", "CHROMA_PATH"):
        monkeypatch.delenv(k, raising=False)


@pytest.mark.anyio
async def test_postgres_checkpointer_requires_url(clean_env, monkeypatch):
    monkeypatch.setenv("CHECKPOINTER", "postgres")
    with pytest.raises(ValueError, match="DATABASE_URL"):
        await G.get_checkpointer()


def test_chroma_client_modes_validate(clean_env, monkeypatch):
    monkeypatch.setenv("CHROMA_MODE", "http")
    with pytest.raises(ValueError, match="CHROMA_HOST"):
        ES.make_chroma_client()
    monkeypatch.setenv("CHROMA_MODE", "cloud")
    with pytest.raises(ValueError, match="CHROMA_API_KEY, CHROMA_TENANT, CHROMA_DATABASE"):
        ES.make_chroma_client()
    monkeypatch.setenv("CHROMA_MODE", "s3")
    with pytest.raises(ValueError, match="expected persistent, http or cloud"):
        ES.make_chroma_client()
    monkeypatch.setenv("CHROMA_MODE", "http")
    monkeypatch.setenv("CHROMA_HOST", "chroma.internal")
    monkeypatch.setenv("CHROMA_PORT", "9000")
    assert ES.store_info() == {"backend": "chroma", "mode": "http", "collection": "evidence",
                               "host": "chroma.internal", "port": "9000"}


def test_persistent_client_is_default(clean_env, tmp_path, monkeypatch):
    monkeypatch.setenv("CHROMA_PATH", str(tmp_path / "c"))
    client = ES.make_chroma_client()
    assert client.heartbeat() > 0 and ES.store_info()["mode"] == "persistent"


@pytest.mark.anyio
async def test_postgres_checkpointer_live(clean_env, monkeypatch, tmp_path):
    url = os.getenv("DR_TEST_DATABASE_URL")
    if not url:
        pytest.skip("set DR_TEST_DATABASE_URL to run against a Postgres")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from fakes import make_deps

    monkeypatch.setenv("CHECKPOINTER", "postgres")
    monkeypatch.setenv("DATABASE_URL", url)
    saver = await G.get_checkpointer()
    try:
        graph = G.build_graph(saver, make_deps(tmp_path))
        tid = "deploy-test-" + os.urandom(3).hex()
        plan_events = [e async for e in G.stream_until_interrupt(question="Does remote work raise productivity?",
                                                                 thread_id=tid, max_hops=0, graph=graph)]
        assert plan_events[-1][0] == "plan" and plan_events[-1][1]["status"] == "awaiting_approval"
        # the interrupt survives in Postgres: a second graph instance on the same DB resumes it
        saver2 = await G.get_checkpointer()
        try:
            graph2 = G.build_graph(saver2, make_deps(tmp_path / "b"))
            events = [e async for e in G.resume_with_plan(thread_id=tid, plan=True, graph=graph2)
                      if e[0] != "activity"]
            assert events[-1][0] == "report"
            state = await graph2.aget_state({"configurable": {"thread_id": tid}})
            assert state.values["approved"] is True and state.values["report"]
        finally:
            await saver2.conn.close()
    finally:
        await saver.conn.close()


def test_chroma_http_store_live(clean_env, monkeypatch):
    host, port = os.getenv("DR_TEST_CHROMA_HOST"), int(os.getenv("DR_TEST_CHROMA_PORT") or 8000)
    if not host or not _port_open(host, port):
        pytest.skip("set DR_TEST_CHROMA_HOST/PORT to run against a Chroma server")
    from langchain_core.embeddings import DeterministicFakeEmbedding

    from deep_researcher.schemas import Evidence

    monkeypatch.setenv("CHROMA_MODE", "http")
    monkeypatch.setenv("CHROMA_HOST", host)
    monkeypatch.setenv("CHROMA_PORT", str(port))
    client = ES.make_chroma_client()
    store = ES.EvidenceStore(DeterministicFakeEmbedding(size=32), embedding_id="fake:32", client=client,
                             collection="deploy_test")
    store.reset()
    store.add_evidence([Evidence(id="E1", sub_question_id="sq1", url="https://ex.org/a",
                                 text="Remote work trial raised productivity. " * 10)])
    assert store.count() > 0 and store.query("productivity", k=1)[0].chunk.url == "https://ex.org/a"
    store.reset()
