"""Gradio UI: pure helpers and the two streams, against the fake graph (no browser, no server)."""
import importlib.util
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

pytest.importorskip("chromadb")
from langgraph.checkpoint.memory import InMemorySaver

from deep_researcher import graph as G
from deep_researcher import ui
from fakes import make_deps

pytestmark = pytest.mark.real_graph


def test_plan_rows_roundtrip_and_validation():
    plan = {"question": "Does remote work raise productivity?", "sub_questions": [
        {"id": "sq1", "question": "What do RCTs find?", "source_types": ["papers", "news"],
         "search_queries": ["remote work RCT", "telework trial"], "depends_on": []},
        {"id": "sq2", "question": "Background?", "source_types": ["wikipedia"], "search_queries": [], "depends_on": ["sq1"]},
    ]}
    rows = ui.plan_to_rows(plan)
    assert rows[0] == ["sq1", "What do RCTs find?", "papers, news", "remote work RCT | telework trial", ""]
    assert rows[1][4] == "sq1"
    back = ui.rows_to_plan(rows, plan["question"])
    assert back["sub_questions"][0]["search_queries"] == ["remote work RCT", "telework trial"]
    assert back["sub_questions"][1]["depends_on"] == ["sq1"]
    # user edits: drop row 1 (blank question), add a row with an unknown source type and a dangling dependency
    edited = [["sq1", "", "papers", "", ""], ["sq2", "Background?", "wikipedia, tiktok", "x | y", "sq1"],
              ["", "New question here", "", "", ""]]
    back = ui.rows_to_plan(edited, plan["question"])
    assert [s["id"] for s in back["sub_questions"]] == ["sq1", "sq2"]           # renumbered
    assert back["sub_questions"][0]["source_types"] == ["wikipedia"]            # unknown type dropped
    assert back["sub_questions"][0]["depends_on"] == []                         # dangling dependency dropped
    assert back["sub_questions"][1]["source_types"] == ["web"]                  # default
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ui.rows_to_plan([], plan["question"])                                   # a plan needs >= 1 sub-question


def test_format_event_lines():
    assert "plan ready: 2 sub-questions" in ui.format_event("plan", {"status": "awaiting_approval",
                                                                     "plan": {"sub_questions": [{}, {}]}})
    line = ui.format_event("retrieve", {"sub_question_id": "sq1", "hop": 0, "sources": 3, "chunks_stored": 4,
                                        "tools": {"arxiv": 5}, "elapsed_s": 1.2})
    assert "retrieve sq1 (hop 0): 3 sources, 4 chunks [arxiv 5] in 1.2s" in line
    assert "skipped (budget exhausted)" in ui.format_event("retrieve", {"sub_question_id": "sq1", "skipped": "budget exhausted"})
    assert "kept 2 of 5" in ui.format_event("validate", {"assessed": 5, "kept": 2})
    assert "3 claims, 1 contradictions, 2 gaps across 2 sub-questions" in ui.format_event(
        "analyze", {"claims": 3, "contradictions": 1, "gaps": ["a", "b"], "sub_questions": ["sq1", "sq2"]})
    assert "pursuing gap1_1" in ui.format_event("route_gaps", {"next": "retrieve", "hop": 1, "new_sub_questions": ["gap1_1"]})
    ins = ui.format_event("insight", {"insights": ["H1", "H2"]})
    assert ins.startswith("[") and "insights: 2" in ins and "- H2" in ins
    assert "critic round 1: revise (2 issues)" in ui.format_event("critic", {"round": 1, "verdict": "revise", "issues": ["a", "b"]})
    assert "requesting: Q?" in ui.format_event("critic", {"round": 1, "verdict": "need_more_evidence", "follow_up": ["Q?"]})
    assert "report ready" in ui.format_event("report", {"stats": {"kept": 3}})
    a = ui.format_event("activity", {"kind": "llm_start", "role": "analysis", "label": "sq2 (9 chunks)",
                                     "model": "anthropic/claude-sonnet-5"})
    assert "analysis[sq2 (9 chunks)]: calling anthropic/claude-sonnet-5" in a
    a = ui.format_event("activity", {"kind": "llm_end", "role": "critic", "label": "round 1", "elapsed_s": 7.2,
                                     "tokens": 12345})
    assert "critic[round 1]: done in 7.2s, 12.3k tokens" in a
    assert "sq1: searching arxiv (2 queries)" in ui.format_event("activity", {"kind": "tool_start", "tool": "arxiv",
                                                                             "sub_question_id": "sq1", "queries": 2})
    assert "sq1: gdelt timed out after 45s (0 kept)" in ui.format_event(
        "activity", {"kind": "tool_timeout", "tool": "gdelt", "sub_question_id": "sq1", "elapsed_s": 45, "results": 0})
    assert "sq3: stored 12 chunks" in ui.format_event("activity", {"kind": "embed_end", "sub_question_id": "sq3",
                                                                   "chunks": 12})
    assert "ERROR: boom" in ui.format_event("error", {"message": "boom"})


@pytest.mark.anyio
async def test_plan_then_run_streams(tmp_path, monkeypatch):
    deps = make_deps(tmp_path)
    graph = G.build_graph(InMemorySaver(), deps)
    monkeypatch.setenv("LANGSMITH_TRACING", "false")

    plan_updates = [u async for u in ui.plan_stream("Does remote work raise productivity?", 1, 0, graph=graph)]
    log_text, rows, thread_id, pipe = plan_updates[-1]
    assert "dr-box done" in pipe and "3 sub-questions" in pipe and "awaiting approval" in pipe
    from deep_researcher.theme import CSS, DARK, PALETTE, THEME_INIT_JS, THEME_TOGGLE_JS, build_theme
    assert PALETTE["green"] in CSS and ".dark #dr-report" in CSS
    assert "localStorage" in THEME_TOGGLE_JS and "pref !== 'light'" in THEME_INIT_JS   # dark by default
    if importlib.util.find_spec("gradio"):                   # the theme object needs gradio (the `ui` extra)
        theme = build_theme()
        assert theme is not None and theme.body_background_fill_dark == DARK["bg"]
    assert "dr-box active" in plan_updates[0][3]                       # planner box pulses while planning
    assert thread_id and "plan ready: 3 sub-questions" in log_text
    assert [r[0] for r in rows] == ["sq1", "sq2", "sq3"]

    rows = rows[:2]                                          # user drops the third sub-question
    run_updates = [u async for u in ui.run_stream(thread_id, rows, "Does remote work raise productivity?", 0,
                                                  log_text, graph=graph, out_dir=tmp_path / "out")]
    log_text, report, md_path, json_path, pdf_path, pipe = run_updates[-1]
    assert pipe.count("dr-box done") == 7 and "dr-box active" not in pipe   # every stage finished
    assert "hop 1" in pipe                                               # the gap hop was shown
    assert "approved 2 sub-questions" in log_text and "report ready" in log_text and "saved" in log_text
    assert pdf_path and Path(pdf_path).read_bytes()[:5] == b"%PDF-"
    assert "calling fake" not in log_text and ": done in" in log_text and "searching arxiv" in log_text
    assert report.startswith("# Remote work and productivity") and "### gap1_1" in report
    assert Path(md_path).read_text() == report and Path(json_path).exists()
    assert "sq3" not in report
    # streaming: the log grew with each event and the report arrived before the file paths
    assert len(run_updates) > 5
    assert run_updates[-2][1] == report and run_updates[-2][2] is None and run_updates[-2][4] is None
    # mid-run, the Retrievers box was active while retrieving
    assert any("dr-box active" in u[5] and "Retrievers" in u[5] for u in run_updates[1:-1])


@pytest.mark.anyio
async def test_run_stream_reports_invalid_plan(tmp_path):
    updates = [u async for u in ui.run_stream("t", [], "q" * 10, 0, "", graph=None)]
    assert len(updates) == 1 and "plan invalid" in updates[0][0]


def test_pipeline_progress_state_machine():
    from deep_researcher.pipeline_view import PipelineProgress

    p = PipelineProgress()
    p.update("started", {})
    assert p.summary()["plan"] == "active"
    p.update("activity", {"kind": "llm_start", "role": "planner", "label": "plan", "model": "openai/gpt-5.1"})
    assert "calling gpt-5.1" in p.stages["plan"].note
    p.update("plan", {"status": "awaiting_approval", "plan": {"sub_questions": [{}, {}, {}]}})
    assert p.summary()["plan"] == "done" and p.summary()["retrieve"] == "waiting"

    p = PipelineProgress.after_plan(3)
    p.update("approve", {})
    p.update("activity", {"kind": "tool_start", "tool": "arxiv", "sub_question_id": "sq1", "queries": 2})
    assert p.summary()["retrieve"] == "active" and "searching arxiv" in p.stages["retrieve"].note
    for i in range(3):
        p.update("retrieve", {"sub_question_id": f"sq{i + 1}", "hop": 0, "sources": 2})
    assert p.stages["retrieve"].note == "3/3 sub-questions"
    p.update("validate", {"assessed": 9, "kept": 5})
    assert p.summary()["retrieve"] == "done" and p.stages["validate"].note == "kept 5/9"
    assert p.summary()["analyze"] == "active"
    p.update("analyze", {"claims": 4, "contradictions": 1, "gaps": ["g"]})
    p.update("route_gaps", {"next": "retrieve", "hop": 1, "new_sub_questions": ["gap1_1"]})
    s = p.summary()
    assert s["retrieve"] == "active" and s["validate"] == "pending" and s["analyze"] == "pending"
    assert p.stages["retrieve"].note == "hop 1: 0/1" and p.stages["retrieve"].hop == 1
    p.update("retrieve", {"sub_question_id": "gap1_1", "hop": 1, "sources": 1})
    p.update("validate", {"assessed": 1, "kept": 1})
    p.update("analyze", {"claims": 1, "contradictions": 0, "gaps": []})
    p.update("route_gaps", {"next": "insight", "hop": 1})
    assert p.summary()["insight"] == "active"
    p.update("insight", {"insights": ["a", "b"]})
    assert p.summary()["insight"] == "done" and p.summary()["critic"] == "active"
    p.update("critic", {"round": 1, "verdict": "revise", "issues": ["x"]})
    assert p.summary()["insight"] == "active" and p.summary()["critic"] == "pending"
    p.update("insight", {"insights": ["a"]})
    p.update("critic", {"round": 2, "verdict": "accept", "issues": []})
    assert p.summary()["report"] == "active"
    p.update("report", {"stats": {"kept": 6, "claims": 5}})
    assert set(p.summary().values()) == {"done"} and "6 sources, 5 claims" in p.stages["report"].note
    html = p.render()
    assert html.count("dr-box done") == 7 and "<style>" in html

    p2 = PipelineProgress.after_plan(2)
    p2.update("approve", {})
    p2.update("error", {"message": "boom"})
    assert p2.summary()["retrieve"] == "error" and "boom" in p2.render()


def test_build_app_constructs():
    pytest.importorskip("gradio")
    demo = ui.build_app(graph=None)
    assert demo is not None
