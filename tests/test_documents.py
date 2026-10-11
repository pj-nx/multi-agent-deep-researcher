"""Phase 8: uploaded documents as a source (parsing, sectioning, uploads store, tool, binding, retrieval)."""
import sys
import zlib
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

pytest.importorskip("chromadb")
from langchain_core.embeddings import DeterministicFakeEmbedding

from deep_researcher import documents as D
from deep_researcher.schemas import DocumentInfo
from deep_researcher.store.uploads import UploadStore
from deep_researcher.tools.documents import DocumentsTool
from fakes import make_deps

pytestmark = pytest.mark.real_graph


# ----------------------------------------------------------------------------- fixtures

def make_pdf(pages: list[str]) -> bytes:
    """A minimal valid PDF with one text line per page (Helvetica, uncompressed content streams)."""
    objs: list[bytes] = []

    def add(body: bytes) -> int:
        objs.append(body)
        return len(objs)

    font = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    page_ids = []
    kids_placeholder = add(b"")                     # pages root, filled below
    for text in pages:
        safe = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream = f"BT /F1 12 Tf 72 720 Td ({safe}) Tj ET".encode()
        content = add(b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream")
        page_ids.append(add(f"<< /Type /Page /Parent {kids_placeholder} 0 R /MediaBox [0 0 612 792] "
                            f"/Resources << /Font << /F1 {font} 0 R >> >> /Contents {content} 0 R >>".encode()))
    objs[kids_placeholder - 1] = (f"<< /Type /Pages /Kids [{' '.join(f'{i} 0 R' for i in page_ids)}] "
                                  f"/Count {len(page_ids)} >>").encode()
    catalog = add(f"<< /Type /Catalog /Pages {kids_placeholder} 0 R >>".encode())
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for n, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += f"{n} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objs) + 1} /Root {catalog} 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)


@pytest.fixture
def store(tmp_path):
    return UploadStore(DeterministicFakeEmbedding(size=32), embedding_id="fake:32", path=str(tmp_path / "chroma"))


# ----------------------------------------------------------------------------- parsing

def test_pdf_pages_are_extracted_and_capped(monkeypatch):
    pdf = make_pdf(["Four-day week trials raised productivity.", "Iceland pilot results were positive.",
                    "Limitations: self-selected firms."])
    parsed = D.parse_document("study.pdf", pdf)
    assert parsed.kind == "pdf" and len(parsed.pages) == 3 and "Iceland" in parsed.pages[1]
    monkeypatch.setenv("UPLOAD_MAX_PAGES", "2")
    parsed = D.parse_document("study.pdf", pdf)
    assert len(parsed.pages) == 2 and "first 2 of 3 pages" in parsed.warnings[0]


def test_text_markdown_html_and_sniffing():
    assert D.parse_document("notes.txt", b"plain notes\r\n\x00with control chars").pages == ["plain notes\n with control chars"]
    assert D.parse_document("readme.md", b"# Title\n\nbody").kind == "md"
    html = D.parse_document("page.html", b"<html><body><h1>Head</h1><p>Para</p><script>x()</script></body></html>")
    assert html.kind == "html" and "Head" in html.text and "x()" not in html.text
    assert D.sniff_kind("renamed.txt", b"<!DOCTYPE html><html></html>") == "html"      # bytes win over extension
    with pytest.raises(D.DocumentError, match="not a PDF"):
        D.parse_document("fake.pdf", b"just text")
    with pytest.raises(D.DocumentError, match="empty"):
        D.parse_document("empty.txt", b"")


def test_size_cap_and_scanned_pdf(monkeypatch):
    monkeypatch.setenv("UPLOAD_MAX_FILE_MB", "1")
    with pytest.raises(D.DocumentError, match="exceeds"):
        D.parse_document("big.txt", b"x" * 1_000_001)
    scanned = D.parse_document("scan.pdf", make_pdf(["", ""]))
    assert scanned.no_text and scanned.pages == ["", ""]


def test_sections_are_page_aligned_and_split_oversized_pages():
    pages = ["short page one", "", "short page three", "word " * 900]      # last page ~900 tokens
    secs = D.split_sections(pages, target_tokens=100)
    assert (secs[0].page_start, secs[0].page_end) == (1, 3) and "page three" in secs[0].text
    big = [s for s in secs if s.page_start == 4]
    assert len(big) >= 4 and all(s.page_end == 4 for s in big)             # oversized page split on its own
    assert [s.index for s in secs] == list(range(len(secs)))
    assert D.split_sections(["", ""]) == []


def test_urls_and_display():
    url = D.document_url("set1", "doc1", 12)
    assert url == "upload://set1/doc1#p12" and D.parse_upload_url(url) == ("set1", "doc1", 12)
    assert D.parse_upload_url("https://x") is None
    assert D.display_source("report.pdf p.12-13", url) == "Uploaded: report.pdf, p. 12-13"
    assert D.display_source("t", "https://x") == "https://x"


# ----------------------------------------------------------------------------- store + tool

def test_upload_store_manifest_search_and_delete(store):
    pdf = make_pdf(["Four-day week trials raised productivity in Iceland.", "Methods and limitations."])
    set_id, manifest = D.ingest([("study.pdf", pdf), ("notes.md", b"# Notes\n\nCongestion charges cut traffic.")],
                                store=store)
    assert [d.filename for d in manifest] == ["study.pdf", "notes.md"]
    assert manifest[0].kind == "pdf" and manifest[0].pages == 2 and manifest[0].sections >= 1
    again = store.manifest(set_id)
    assert [d.filename for d in again] == ["notes.md", "study.pdf"] and again[1].excerpt.startswith("Four-day")
    hits = store.search(set_id, "four-day week productivity", k=3)
    assert hits and hits[0].set_id == set_id and hits[0].url.startswith(f"upload://{set_id}/")
    assert store.search("other-set", "anything") == []                     # sets are isolated
    assert store.delete_set(set_id) >= 2 and store.manifest(set_id) == []


def test_scanned_document_is_listed_but_never_searched(store):
    set_id, manifest = D.ingest([("scan.pdf", make_pdf(["", ""]))], store=store)
    assert manifest[0].no_text and manifest[0].sections == 0
    assert store.manifest(set_id)[0].no_text and store.search(set_id, "anything") == []


def test_set_size_limit_and_expiry(store, monkeypatch):
    monkeypatch.setenv("UPLOAD_MAX_DOCS_PER_SET", "1")
    with pytest.raises(D.DocumentError, match="at most 1"):
        D.ingest([("a.txt", b"aaa"), ("b.txt", b"bbb")], store=store)
    set_id, _ = D.ingest([("a.txt", b"some text here")], store=store)
    assert store.expired_sets(days=1) == []
    assert store.expire(days=0, now=D.stamp() + 10) == [set_id] and store.count(set_id) == 0


@pytest.mark.anyio
async def test_documents_tool_returns_search_results(store):
    set_id, _ = D.ingest([("study.pdf", make_pdf(["Four-day week trials raised productivity."]))], store=store)
    tool = DocumentsTool(set_id, store=store)
    results = await tool.search("four-day week productivity", limit=3)
    assert results and results[0].source_type == "documents" and results[0].title == "study.pdf p.1"
    assert results[0].url == f"upload://{set_id}/{results[0].extra['doc_id']}#p1"
    assert await DocumentsTool("missing", store=store).search("x") == []


# ----------------------------------------------------------------------------- pipeline

@pytest.mark.anyio
async def test_bound_deps_feed_documents_through_retrieval_and_validation(store, tmp_path):
    from deep_researcher.agents.retriever import retrieve_node

    set_id, _ = D.ingest([("trial.pdf", make_pdf(["The four-day week trial RAISES productivity in knowledge work.",
                                                   "Manufacturing output LOWERS under compressed schedules."]))],
                         store=store)
    deps = D.bind_documents(make_deps(tmp_path), set_id, store=store)
    assert [t.name for t in deps.tools_for(["papers"])] == ["arxiv"]                  # unchanged elsewhere
    assert [t.name for t in deps.tools_for(["papers", "documents"])] == ["arxiv", "documents"]
    out = await retrieve_node({"sub_question": {"id": "sq1", "question": "Does the four-day week trial raise productivity?",
                                                "source_types": ["documents"], "search_queries": ["four-day week productivity"]},
                               "hop": 0, "deadline": 0}, deps)
    ev = out["evidence"]
    assert ev and all(e["source_type"] == "documents" and e["url"].startswith("upload://") for e in ev)
    assert out["retrieval_log"][0]["tools"] == {"documents": len(ev)}
    assert set(out["assessments"]) == {e["id"] for e in ev}                        # graded in the branch


@pytest.mark.anyio
async def test_full_run_cites_uploaded_sections_and_lists_them_as_uploaded(store, tmp_path):
    from langgraph.checkpoint.memory import InMemorySaver

    from deep_researcher import graph as G
    from deep_researcher.schemas import ResearchPlan, SubQuestion

    set_id, manifest = D.ingest([("trial.pdf", make_pdf(["Remote work trial RAISES productivity in knowledge work.",
                                                         "Remote work LOWERS output in call centres."]))], store=store)
    base = make_deps(tmp_path)                       # bind_documents returns a fresh Deps (no fake_* attrs)
    fake_llm = base.fake_llm
    fake_llm.gap_rounds = 0                          # no hop-1 follow-up: only the uploads are retrieved
    fake_llm.plan = ResearchPlan(question="Does remote work raise productivity?", sub_questions=[
        SubQuestion(id="a", question="What do the attached trial documents say about remote work productivity?",
                    source_types=["documents"], search_queries=["remote work productivity trial"])])
    deps = D.bind_documents(base, set_id, store=store)
    graph = G.build_graph(InMemorySaver(), deps)
    docs = {"set_id": set_id, "documents": [d.model_dump() for d in manifest]}
    events = [e async for e in G.stream_until_interrupt(question="Does remote work raise productivity?",
                                                        thread_id="d1", graph=graph, documents=docs)]
    planner_prompt = next(p for name, p in fake_llm.prompts if name == "ResearchPlan")
    assert "trial.pdf" in planner_prompt and "source type `documents`" in planner_prompt
    plan = events[-1][1]["plan"]
    events = [e async for e in G.resume_with_plan(thread_id="d1", plan=plan, graph=graph)]
    report = next(d for k, d in events if k == "report")
    assert report["stats"]["kept"] >= 1
    assert all(s["source_type"] == "documents" for s in report["report_json"]["sources"])
    assert "Uploaded: trial.pdf, p." in report["report"] and "upload://" not in report["report"].split("## Sources")[1]
    cited = {i for s in report["report_json"]["summaries"].values() for c in s["claims"] for i in c["evidence_ids"]}
    assert cited <= {s["id"] for s in report["report_json"]["sources"]}


def test_document_info_schema_bounds():
    DocumentInfo(doc_id="abc", filename="f.pdf", kind="pdf", pages=1, sections=1, chars=10)
    with pytest.raises(ValueError):
        DocumentInfo(doc_id="abc", filename="f.pdf", kind="exe", pages=1, sections=1, chars=10)


def test_pdf_fixture_is_a_real_pdf():
    data = make_pdf(["hello"])
    assert data.startswith(b"%PDF-1.4") and zlib.crc32(data)       # sanity: bytes, not a placeholder
