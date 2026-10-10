"""Gradio UI: ask a question, watch the agents work, edit and approve the plan, read and download the report.

Runs the graph in-process (no HTTP API in between). Launch with:

    uv run python -m deep_researcher.ui            # http://127.0.0.1:7860

The pure helpers (plan <-> table rows, event formatting) and the two async streams are kept free
of Gradio objects so they can be tested against the fake graph without a browser.
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from deep_researcher.reports import write_report
from deep_researcher.schemas import ResearchPlan, SourceType
from deep_researcher.state import ResearchState  # noqa: F401 - documents the state shape for readers

log = logging.getLogger(__name__)

PLAN_HEADERS = ["id", "question", "source_types", "search_queries", "depends_on"]
SOURCE_TYPES: tuple[str, ...] = tuple(SourceType.__args__)  # type: ignore[attr-defined]


# ----------------------------------------------------------------------------- plan <-> table

def plan_to_rows(plan: dict[str, Any]) -> list[list[str]]:
    return [[sq["id"], sq["question"], ", ".join(sq.get("source_types") or []),
             " | ".join(sq.get("search_queries") or []), ", ".join(sq.get("depends_on") or [])]
            for sq in plan["sub_questions"]]


def _split(value: Any, sep: str) -> list[str]:
    return [x.strip() for x in str(value or "").split(sep) if x and x.strip()]


def rows_to_plan(rows: list[list[Any]], question: str, notes: str | None = None) -> dict[str, Any]:
    """Rebuild a validated plan from edited table rows; blank rows are dropped, ids re-numbered."""
    subs = []
    for row in rows or []:
        cells = [*(row or []), "", "", "", "", ""][:5]
        q = str(cells[1] or "").strip()
        if not q:
            continue
        types = [t for t in _split(cells[2], ",") if t in SOURCE_TYPES] or ["web"]
        subs.append({"id": f"sq{len(subs) + 1}", "question": q, "source_types": types,
                     "search_queries": _split(cells[3], "|")[:5], "depends_on": _split(cells[4], ",")})
    valid = {s["id"] for s in subs}
    for s in subs:
        s["depends_on"] = [d for d in s["depends_on"] if d in valid and d != s["id"]]
    return ResearchPlan(question=question, sub_questions=subs, notes=notes).model_dump()


# ----------------------------------------------------------------------------- event log

def format_event(event: str, data: dict[str, Any]) -> str:
    """One human-readable line per stream event."""
    stamp = time.strftime("%H:%M:%S")
    if event == "plan":
        if data.get("status") == "awaiting_approval":
            n = len((data.get("plan") or {}).get("sub_questions", []))
            return f"[{stamp}] plan ready: {n} sub-questions - review, edit and approve below"
        return f"[{stamp}] planner: {'; '.join(data.get('notes', []))}"
    if event == "retrieve":
        if data.get("skipped"):
            return f"[{stamp}] retrieve {data.get('sub_question_id')}: skipped ({data['skipped']})"
        tools = ", ".join(f"{k} {v}" for k, v in (data.get("tools") or {}).items()) or "no tools"
        return (f"[{stamp}] retrieve {data.get('sub_question_id')} (hop {data.get('hop', 0)}): "
                f"{data.get('sources', 0)} sources, {data.get('chunks_stored', 0)} chunks [{tools}] "
                f"in {data.get('elapsed_s', '?')}s")
    if event == "validate":
        return f"[{stamp}] validator: kept {data.get('kept', 0)} of {data.get('assessed', 0)} sources"
    if event == "analyze":
        gaps = data.get("gaps") or []
        n_sq = len(data.get("sub_questions") or [])
        return (f"[{stamp}] analysis: {data.get('claims', 0)} claims, {data.get('contradictions', 0)} "
                f"contradictions, {len(gaps)} gaps across {n_sq} sub-questions")
    if event == "route_gaps":
        if data.get("next") == "retrieve":
            pursued = ", ".join(data.get("new_sub_questions") or [])
            return f"[{stamp}] hop {data.get('hop')}: pursuing {pursued}"
        return f"[{stamp}] gap router: {'; '.join(data.get('notes', []))}"
    if event == "insight":
        hyps = data.get("insights") or []
        lines = [f"[{stamp}] insights: {len(hyps)}"] + [f"    - {h[:140]}" for h in hyps]
        return "\n".join(lines)
    if event == "critic":
        issues = data.get("issues") or []
        line = f"[{stamp}] critic round {data.get('round')}: {data.get('verdict')} ({len(issues)} issues)"
        if data.get("follow_up"):
            line += " - requesting: " + "; ".join(data["follow_up"])[:200]
        return line
    if event == "report":
        return f"[{stamp}] report ready: {json.dumps(data.get('stats') or {}, default=str)}"
    if event == "error":
        return f"[{stamp}] ERROR: {data.get('message')}"
    if event in ("started", "done", "approve"):
        return f"[{stamp}] {event}" + (f" ({data.get('elapsed_s')}s)" if data.get("elapsed_s") else "")
    rest = json.dumps({k: v for k, v in data.items() if k != "notes"}, default=str)[:200]
    return f"[{stamp}] {event}: {rest}"


# ----------------------------------------------------------------------------- streams

async def plan_stream(question: str, max_hops: int, budget: int | None, *, graph: Any = None,
                      ) -> AsyncIterator[tuple[str, list[list[str]], str]]:
    """Yields (log_text, plan_rows, thread_id) until the planner pauses for approval."""
    from deep_researcher import graph as G
    from deep_researcher import tracing

    thread_id = str(uuid.uuid4())
    cfg = tracing.run_config(thread_id, question, tags=["ui"])
    lines = [f"[{time.strftime('%H:%M:%S')}] thread {thread_id[:8]}: planning..."]
    rows: list[list[str]] = []
    yield "\n".join(lines), rows, thread_id
    try:
        async for event, data in G.stream_until_interrupt(question=question, thread_id=thread_id,
                                                          max_hops=max_hops, budget_seconds=budget or None,
                                                          config=cfg, graph=graph):
            lines.append(format_event(event, data))
            if event == "plan" and data.get("status") == "awaiting_approval":
                rows = plan_to_rows(data["plan"])
            yield "\n".join(lines), rows, thread_id
    except Exception as exc:
        log.exception("plan stream failed")
        lines.append(f"[{time.strftime('%H:%M:%S')}] ERROR: {type(exc).__name__}: {str(exc)[:300]}")
        yield "\n".join(lines), rows, thread_id


async def run_stream(thread_id: str, rows: list[list[Any]], question: str, budget: int | None,
                     log_text: str, *, graph: Any = None, out_dir: Path | None = None,
                     ) -> AsyncIterator[tuple[str, str, str | None, str | None]]:
    """Yields (log_text, report_markdown, md_path, json_path) through to the finished report."""
    from deep_researcher import graph as G
    from deep_researcher import tracing

    lines = [log_text] if log_text else []
    try:
        plan = rows_to_plan(rows, question)
    except Exception as exc:  # noqa: BLE001 - validation errors must be shown, not raised
        lines.append(f"[{time.strftime('%H:%M:%S')}] plan invalid: {str(exc)[:300]}")
        yield "\n".join(lines), "", None, None
        return
    n = len(plan["sub_questions"])
    lines.append(f"[{time.strftime('%H:%M:%S')}] approved {n} sub-questions; running...")
    yield "\n".join(lines), "", None, None
    cfg = tracing.run_config(thread_id, question, tags=["ui", "research"])
    report_md, report_json = "", None
    try:
        async for event, data in G.resume_with_plan(thread_id=thread_id, plan=plan,
                                                    budget_seconds=budget or None, config=cfg, graph=graph):
            lines.append(format_event(event, data))
            if event == "report":
                report_md = data.get("report") or ""
                report_json = data.get("report_json")
            yield "\n".join(lines), report_md, None, None
    except Exception as exc:
        log.exception("run stream failed")
        lines.append(f"[{time.strftime('%H:%M:%S')}] ERROR: {type(exc).__name__}: {str(exc)[:300]}")
        yield "\n".join(lines), report_md, None, None
        return
    if report_md:
        md_path, json_path = write_report(question, report_md, report_json, out_dir)
        lines.append(f"[{time.strftime('%H:%M:%S')}] saved {md_path}")
        yield "\n".join(lines), report_md, str(md_path), str(json_path) if json_path else None
    tracing.flush()


# ----------------------------------------------------------------------------- Gradio app

def build_app(graph: Any = None):
    import gradio as gr

    with gr.Blocks(title="Multi-Agent Deep Researcher") as demo:
        gr.Markdown("# Multi-Agent Deep Researcher\n"
                    "Planner -> parallel retrievers -> validator -> critical analysis (multi-hop) -> "
                    "insights -> red-team critic -> report. Every claim cites stored evidence.")
        thread_state = gr.State("")
        question_state = gr.State("")
        with gr.Row():
            question = gr.Textbox(label="Research question", lines=2, scale=4,
                                  placeholder="Does the four-day work week raise productivity, "
                                              "and where do studies disagree?")
            with gr.Column(scale=1):
                max_hops = gr.Slider(0, 3, value=1, step=1, label="Max follow-up hops")
                budget = gr.Number(value=0, label="Budget (seconds, 0 = none)", precision=0)
        plan_btn = gr.Button("1. Plan", variant="primary")
        activity = gr.Textbox(label="Agent activity", lines=14, max_lines=30, interactive=False,
                              autoscroll=True)
        gr.Markdown("### 2. Review the plan (edit cells, clear a question to drop it), then approve")
        plan_table = gr.Dataframe(headers=PLAN_HEADERS, datatype=["str"] * 5, type="array", interactive=True,
                                  row_count=(0, "dynamic"), col_count=(5, "fixed"), wrap=True,
                                  column_widths=["7%", "43%", "15%", "25%", "10%"], label="Plan")
        with gr.Row():
            run_btn = gr.Button("3. Approve & run", variant="primary")
        report_md = gr.Markdown(label="Report", max_height=700)
        with gr.Row():
            dl_md = gr.DownloadButton("Download report (.md)", visible=False)
            dl_json = gr.DownloadButton("Download data (.json)", visible=False)

        async def on_plan(q, hops, bud):
            q = (q or "").strip()
            if len(q) < 5:
                yield "Enter a question first.", [], "", ""
                return
            async for log_text, rows, tid in plan_stream(q, int(hops), int(bud or 0), graph=graph):
                yield log_text, rows, tid, q

        async def on_run(tid, rows, q, bud, previous_log):
            hidden = gr.update(visible=False)
            if not tid:
                yield (previous_log or "") + "\nRun the planner first.", "", hidden, hidden
                return
            stream = run_stream(tid, rows, q, int(bud or 0), previous_log or "", graph=graph)
            async for log_text, md, md_path, json_path in stream:
                yield (log_text, md,
                       gr.update(value=md_path, visible=bool(md_path)),
                       gr.update(value=json_path, visible=bool(json_path)))

        plan_btn.click(on_plan, [question, max_hops, budget],
                       [activity, plan_table, thread_state, question_state])
        run_btn.click(on_run, [thread_state, plan_table, question_state, budget, activity],
                      [activity, report_md, dl_md, dl_json])
    return demo


def main() -> None:
    import os

    from dotenv import load_dotenv

    load_dotenv()
    from deep_researcher import tracing

    tracing.configure()
    demo = build_app()
    demo.queue().launch(server_name=os.getenv("GRADIO_HOST", "127.0.0.1"),
                        server_port=int(os.getenv("GRADIO_PORT") or 7860),
                        allowed_paths=[str(Path(os.getenv("REPORTS_DIR") or "reports").resolve())])


if __name__ == "__main__":
    main()
