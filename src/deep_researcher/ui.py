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
    if event == "activity":
        return format_activity(data, stamp)
    if event == "error":
        return f"[{stamp}] ERROR: {data.get('message')}"
    if event in ("started", "done", "approve"):
        return f"[{stamp}] {event}" + (f" ({data.get('elapsed_s')}s)" if data.get("elapsed_s") else "")
    rest = json.dumps({k: v for k, v in data.items() if k != "notes"}, default=str)[:200]
    return f"[{stamp}] {event}: {rest}"


def _tokens(n: Any) -> str:
    return f", {n / 1000:.1f}k tokens" if isinstance(n, int | float) and n else ""


def format_activity(data: dict[str, Any], stamp: str) -> str:
    """Indented line for a fine-grained event (model call, tool search, embedding)."""
    kind = data.get("kind")
    role, label, model = data.get("role"), data.get("label"), data.get("model")
    who = f"{role}[{label}]" if label else str(role)
    if kind == "llm_start":
        return f"[{stamp}]     {who}: calling {model} ..."
    if kind == "llm_end":
        return f"[{stamp}]     {who}: done in {data.get('elapsed_s')}s{_tokens(data.get('tokens'))}"
    if kind == "llm_error":
        return f"[{stamp}]     {who}: FAILED after {data.get('elapsed_s')}s ({data.get('error')})"
    sq = data.get("sub_question_id")
    if kind == "tool_start":
        return f"[{stamp}]     {sq}: searching {data.get('tool')} ({data.get('queries')} queries) ..."
    if kind == "tool_end":
        return (f"[{stamp}]     {sq}: {data.get('tool')} returned {data.get('results')} "
                f"in {data.get('elapsed_s')}s")
    if kind == "tool_timeout":
        return (f"[{stamp}]     {sq}: {data.get('tool')} timed out after {data.get('elapsed_s')}s "
                f"({data.get('results')} kept)")
    if kind == "embed_start":
        return f"[{stamp}]     {sq}: embedding {data.get('sources')} sources ..."
    if kind == "embed_end":
        return f"[{stamp}]     {sq}: stored {data.get('chunks')} chunks"
    if kind == "retrieval":
        return f"[{stamp}]     {label}"
    rest = {k: v for k, v in data.items() if k not in ("kind", "t")}
    return f"[{stamp}]     {kind}: {json.dumps(rest, default=str)[:160]}"


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


# ----------------------------------------------------------------------------- settings

DEFAULT_MODEL_CHOICES = (
    "anthropic/claude-opus-5.5", "anthropic/claude-sonnet-5", "anthropic/claude-haiku-4.5",
    "openai/gpt-5.1", "openai/gpt-5-mini", "google/gemini-3.1-pro-preview", "google/gemini-3-flash-preview",
    "deepseek/deepseek-v4-pro", "qwen/qwen3.5-plus-20260420",
)


def settings_summary_md() -> str:
    from deep_researcher import settings as S

    info = S.summary()
    models = info["llm"]["models"]
    by_model: dict[str, list[str]] = {}
    for role, model in models.items():
        by_model.setdefault(model, []).append(role)
    lines = [f"**Env file:** `{info['env_file']}`", "",
             "**Keys:** " + ", ".join(f"{k.replace('_API_KEY', '').replace('_KEY', '')}: {v}"
                                      for k, v in info["keys"].items()), "",
             "**Models:** " + "; ".join(f"`{m}` ({', '.join(r)})" for m, r in by_model.items()), "",
             f"**Embeddings:** `{info['embeddings'].get('embedding_id', '?')}` "
             f"via {info['embeddings'].get('provider')}",
             "", "**Active sources:** " + ", ".join(f"{k}: {', '.join(v) or '-'}"
                                                   for k, v in info["sources"].items()),
             "", f"**Tracing:** {'on' if info['tracing'].get('enabled') else 'off'}"]
    return "\n".join(lines)


def save_settings(form: dict[str, Any]) -> str:
    """Apply the form, persist to .env, return a status line (never echoes secrets)."""
    from deep_researcher import settings as S

    try:
        updates, path = S.save(form)
    except Exception as exc:  # noqa: BLE001 - show the error in the page
        return f"Save failed: {type(exc).__name__}: {str(exc)[:200]}"
    if not updates:
        return "Nothing changed."
    changed = ", ".join(f"{k} {'cleared' if v is None else 'set'}" for k, v in updates.items())
    note = ""
    if any(k.startswith("EMBEDDINGS_") for k in updates):
        note = (" Embedding settings changed: use a new CHROMA_COLLECTION "
                "(an existing collection is bound to its model).")
    return f"Saved to {path}: {changed}.{note}"


async def test_openrouter() -> str:
    from deep_researcher import llm

    if not llm.enabled():
        return "OPENROUTER_API_KEY is not set."
    try:
        from langchain_core.messages import HumanMessage

        model = llm.make_llm("grader", max_tokens=10)
        t = time.time()
        r = await model.ainvoke([HumanMessage("Reply with the single word: pong")])
        return f"OpenRouter OK: {model.model_name} answered {r.content!r} in {time.time() - t:.1f}s"
    except Exception as exc:  # noqa: BLE001
        return f"OpenRouter FAILED: {type(exc).__name__}: {str(exc)[:200]}"


async def fetch_model_choices() -> list[str]:
    from deep_researcher import llm

    try:
        rows = await llm.alist_models()
    except Exception as exc:  # noqa: BLE001
        log.warning("model list fetch failed: %s", exc)
        return list(DEFAULT_MODEL_CHOICES)
    ids = [r["id"] for r in rows if not r["id"].endswith(":batch")]
    return sorted(set(ids) | set(DEFAULT_MODEL_CHOICES))


def _build_settings_tab(gr: Any) -> None:
    from deep_researcher import settings as S

    gr.Markdown("### Settings\nValues are written to the project's `.env` (git-ignored) and applied "
                "immediately. Leave a key blank to keep it; type `none` to clear it. "
                "Secrets are shown masked.")
    inputs: dict[str, Any] = {}
    for group, title in (("keys", "API keys and tracing"), ("models", "Models (OpenRouter ids)"),
                         ("embeddings", "Embeddings and vector store"), ("run", "Run limits")):
        with gr.Accordion(title, open=(group != "run")):
            for f in [f for f in S.FIELDS if f.group == group]:
                if f.kind == "secret":
                    inputs[f.key] = gr.Textbox(label=f.label, value="", placeholder=S.display_value(f),
                                               type="password", info=f.help)
                elif f.kind == "bool":
                    on = S.current(f.key).lower() in {"1", "true", "yes"}
                    inputs[f.key] = gr.Checkbox(label=f.label, value=on, info=f.help)
                elif f.kind == "choice":
                    choices = list(f.choices) if f.choices else list(DEFAULT_MODEL_CHOICES)
                    cur = S.current(f.key)
                    if cur and cur not in choices:
                        choices.insert(0, cur)
                    inputs[f.key] = gr.Dropdown(label=f.label, choices=choices, value=cur or None,
                                                allow_custom_value=True, info=f.help)
                else:
                    inputs[f.key] = gr.Textbox(label=f.label, value=S.current(f.key), info=f.help)
    with gr.Row():
        save_btn = gr.Button("Save & apply", variant="primary")
        test_btn = gr.Button("Test OpenRouter key")
        models_btn = gr.Button("Load model list from OpenRouter")
    status = gr.Markdown()
    effective = gr.Markdown(value=settings_summary_md, label="Effective configuration")

    keys = list(inputs)

    def on_save(*values):
        msg = save_settings(dict(zip(keys, values, strict=True)))
        return msg, settings_summary_md()

    async def on_models():
        choices = await fetch_model_choices()
        return [gr.update(choices=choices) for _ in range(3)] + [f"Loaded {len(choices)} models."]

    save_btn.click(on_save, [inputs[k] for k in keys], [status, effective])
    test_btn.click(test_openrouter, None, status)
    models_btn.click(on_models, None, [inputs["OPENROUTER_MODEL_HEAVY"], inputs["OPENROUTER_MODEL_WORKER"],
                                       inputs["OPENROUTER_MODEL_CHEAP"], status])


# ----------------------------------------------------------------------------- Gradio app

def build_app(graph: Any = None):
    import gradio as gr

    with gr.Blocks(title="Multi-Agent Deep Researcher") as demo:
        gr.Markdown("# Multi-Agent Deep Researcher\n"
                    "Planner -> parallel retrievers -> validator -> critical analysis (multi-hop) -> "
                    "insights -> red-team critic -> report. Every claim cites stored evidence.")
        with gr.Tabs():
            with gr.Tab("Research"):
                _build_research_tab(gr, demo, graph)
            with gr.Tab("Settings"):
                _build_settings_tab(gr)
    return demo


def _build_research_tab(gr: Any, demo: Any, graph: Any) -> None:
    if True:
        thread_state = gr.State("")
        question_state = gr.State("")
        with gr.Row():
            question = gr.Textbox(label="Research question", lines=2, scale=4,
                                  placeholder="Does the four-day work week raise productivity, "
                                              "and where do studies disagree?")
            with gr.Column(scale=1):
                max_hops = gr.Slider(0, 3, value=1, step=1, label="Max follow-up hops")
                budget = gr.Number(value=0, label="Budget (seconds, 0 = none)", precision=0)
        with gr.Row():
            plan_btn = gr.Button("1. Plan", variant="primary", scale=4)
            busy = gr.Markdown("**Ready**", elem_id="busy")
        activity = gr.Textbox(label="Agent activity", lines=14, max_lines=30, interactive=False,
                              autoscroll=True)
        gr.Markdown("### 2. Review the plan (edit cells, clear a question to drop it), then approve")
        plan_table = gr.Dataframe(headers=PLAN_HEADERS, datatype=["str"] * 5, type="array", interactive=True,
                                  row_count=(0, "dynamic"), column_count=(5, "fixed"), wrap=True,
                                  column_widths=["7%", "43%", "15%", "25%", "10%"], label="Plan")
        with gr.Row():
            run_btn = gr.Button("3. Approve & run", variant="primary")
        report_md = gr.Markdown(label="Report", max_height=700)
        with gr.Row():
            dl_md = gr.DownloadButton("Download report (.md)", visible=False)
            dl_json = gr.DownloadButton("Download data (.json)", visible=False)

        BUSY = gr.update(interactive=False)
        FREE = gr.update(interactive=True)

        def _busy(text: str) -> str:
            return f"⏳ **{text}**"

        async def on_plan(q, hops, bud):
            q = (q or "").strip()
            if len(q) < 5:
                yield "Enter a question first.", [], "", "", "**Ready**", FREE, FREE
                return
            try:
                async for log_text, rows, tid in plan_stream(q, int(hops), int(bud or 0), graph=graph):
                    status = _busy("Planning...") if not rows else "**Plan ready: review and approve**"
                    yield log_text, rows, tid, q, status, BUSY, BUSY
            finally:
                yield gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), FREE, FREE

        async def on_run(tid, rows, q, bud, previous_log):
            hidden = gr.update(visible=False)
            if not tid:
                yield ((previous_log or "") + "\nRun the planner first.", "", hidden, hidden, "**Ready**",
                       FREE, FREE)
                return
            stream = run_stream(tid, rows, q, int(bud or 0), previous_log or "", graph=graph)
            try:
                async for log_text, md, md_path, json_path in stream:
                    status = _busy("Researching... (see activity)") if not md else "**Report ready**"
                    yield (log_text, md,
                           gr.update(value=md_path, visible=bool(md_path)),
                           gr.update(value=json_path, visible=bool(json_path)), status, BUSY, BUSY)
            finally:
                yield gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), FREE, FREE

        plan_btn.click(on_plan, [question, max_hops, budget],
                       [activity, plan_table, thread_state, question_state, busy, plan_btn, run_btn])
        run_btn.click(on_run, [thread_state, plan_table, question_state, budget, activity],
                      [activity, report_md, dl_md, dl_json, busy, plan_btn, run_btn])


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
