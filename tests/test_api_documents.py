"""Phase 8 (b): upload endpoints, document sets on research/resume, Gradio ingest helper."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

pytest.importorskip("chromadb")
from langchain_core.embeddings import DeterministicFakeEmbedding

from deep_researcher.store import uploads as U
from test_api import _client, _load_api
from test_documents import make_pdf


@pytest.fixture
def upload_store(tmp_path, monkeypatch):
    store = U.UploadStore(DeterministicFakeEmbedding(size=32), embedding_id="fake:32", path=str(tmp_path / "up"))
    monkeypatch.setattr(U, "get_upload_store", lambda: store)
    return store


@pytest.mark.anyio
async def test_upload_get_delete_round_trip(monkeypatch, upload_store):
    mod = _load_api(monkeypatch)
    async with _client(mod) as c:
        r = await c.post("/api/documents", files=[("files", ("trial.pdf", make_pdf(["Four-day week trial raised output."]), "application/pdf")),
                                                  ("files", ("notes.md", b"# Notes\n\nIceland pilot.", "text/markdown"))])
        assert r.status_code == 200, r.text
        body = r.json()
        set_id = body["document_set_id"]
        assert [d["filename"] for d in body["documents"]] == ["trial.pdf", "notes.md"]
        assert body["documents"][0]["kind"] == "pdf" and body["documents"][0]["pages"] == 1
        # append to the same set
        r = await c.post("/api/documents", data={"document_set_id": set_id},
                         files=[("files", ("more.txt", b"more text here", "text/plain"))])
        assert r.status_code == 200 and len(r.json()["documents"]) == 3
        r = await c.get(f"/api/documents/{set_id}")
        assert r.status_code == 200 and sorted(d["filename"] for d in r.json()["documents"]) == ["more.txt", "notes.md", "trial.pdf"]
        r = await c.delete(f"/api/documents/{set_id}")
        assert r.status_code == 200 and r.json()["deleted"] >= 3
        assert (await c.get(f"/api/documents/{set_id}")).status_code == 404
        h = (await c.get("/api/health")).json()
        assert h["uploads"]["formats"] == ["pdf", "txt", "md", "html", "docx"] and h["uploads"]["max_request_mb"] is None


@pytest.mark.anyio
async def test_upload_limits_and_errors(monkeypatch, upload_store):
    monkeypatch.setenv("UPLOAD_MAX_FILE_MB", "1")
    mod = _load_api(monkeypatch, API_AUTH_TOKEN="secret")
    hdr = {"authorization": "Bearer secret"}
    async with _client(mod) as c:
        assert (await c.post("/api/documents", files=[("files", ("a.txt", b"x", "text/plain"))])).status_code == 401
        r = await c.post("/api/documents", headers=hdr, files=[("files", ("big.txt", b"x" * 1_000_001, "text/plain"))])
        assert r.status_code == 413
        r = await c.post("/api/documents", headers=hdr, files=[("files", ("fake.pdf", b"not a pdf", "application/pdf"))])
        assert r.status_code == 422 and "not a PDF" in r.json()["detail"]
        r = await c.post("/api/research", headers=hdr, json={"question": "Does remote work help?", "document_set_id": "nope"})
        assert r.status_code == 422                                            # pattern: too short
        r = await c.post("/api/research", headers=hdr, json={"question": "Does remote work help?",
                                                             "document_set_id": "a" * 32})
        assert r.status_code == 404                                            # unknown set


@pytest.mark.anyio
async def test_research_and_resume_bind_the_document_set(monkeypatch, upload_store):
    import deep_researcher.graph as G
    from deep_researcher.tools.documents import DocumentsTool

    mod = _load_api(monkeypatch)
    seen = {}

    async def fake_graph_for(deps=None):
        seen["deps"] = deps
        return None

    monkeypatch.setattr(G, "graph_for", fake_graph_for)
    captured = {}
    real = G.stream_until_interrupt

    async def spy(**kw):
        captured.update(kw)
        async for ev in real(**kw):
            yield ev

    monkeypatch.setattr(G, "stream_until_interrupt", spy)
    async with _client(mod) as c:
        set_id = (await c.post("/api/documents", files=[("files", ("t.txt", b"remote work raises output", "text/plain"))])).json()["document_set_id"]
        r = await c.post("/api/research", json={"question": "Does remote work help?", "document_set_id": set_id})
        assert r.status_code == 200 and f'"document_set_id": "{set_id}"' in r.text
        assert captured["documents"]["set_id"] == set_id and captured["documents"]["documents"][0]["filename"] == "t.txt"
        tools = seen["deps"].tools_for(["documents", "papers"])
        assert any(isinstance(t, DocumentsTool) and t.set_id == set_id for t in tools)
        plan = {"question": "Does remote work help?", "sub_questions": [{"id": "sq1", "question": "What do the documents say?", "source_types": ["documents"]}]}
        r = await c.post("/api/research/00000000-0000-4000-8000-000000000000/resume",
                         json={"plan": plan, "document_set_id": set_id})
        assert r.status_code == 200 and "event: report" in r.text


def test_gradio_ingest_helper(tmp_path, upload_store):
    from deep_researcher import ui

    pdf = tmp_path / "trial.pdf"
    pdf.write_bytes(make_pdf(["Four-day week trial raised output."]))
    bad = tmp_path / "fake.pdf"
    bad.write_bytes(b"nope")
    docs, md = ui.ingest_uploads([str(pdf)])
    assert docs and docs["documents"][0]["filename"] == "trial.pdf" and "**trial.pdf**" in md and "1 page(s)" in md
    first = docs["set_id"]
    docs2, md2 = ui.ingest_uploads([str(bad)], docs)                     # replaces the set; rejected file
    assert docs2 is None and "Upload rejected" in md2 and upload_store.manifest(first) == []
    assert ui.ingest_uploads(None, None) == (None, "")
