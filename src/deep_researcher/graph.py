"""The research graph: planner → approval → parallel retrievers → analysis loop → synthesis → report.

Phase status: planner and retrievers are real; analysis, insight, critic and report are
placeholders that keep the state contract and emit an evidence digest, so the full pipeline runs
end to end today and each later phase replaces one node.

Public entry points used by the API and CLI:
    stream_until_interrupt(question, thread_id, ...)  -> async (event, data) until the plan needs approval
    resume_with_plan(thread_id, plan, ...)            -> async (event, data) through to the report
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import AsyncIterator
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Send, interrupt

from deep_researcher.agents.deps import Deps
from deep_researcher.agents.planner import plan_node
from deep_researcher.agents.retriever import retrieve_node
from deep_researcher.schemas import ResearchPlan, Summary
from deep_researcher.state import ResearchState, RetrieveInput

log = logging.getLogger(__name__)

DEFAULT_MAX_HOPS = 1


# ----------------------------------------------------------------------------- nodes

def _mk_plan(deps: Deps):
    async def plan(state: ResearchState) -> dict[str, Any]:
        return await plan_node(state, deps)
    return plan


def approve(state: ResearchState) -> dict[str, Any]:
    """Pause for human approval. Resume payload: {"plan": {...edited...}} or True to accept as-is."""
    decision = interrupt({"plan": state["plan"], "message": "Review the research plan; edit or approve."})
    plan = state["plan"]
    if isinstance(decision, dict) and decision.get("plan"):
        plan = ResearchPlan.model_validate(decision["plan"]).model_dump()
    return {"plan": plan, "approved": True, "hop": 0, "notes": ["plan approved"]}


def fan_out(state: ResearchState) -> list[Send]:
    plan = ResearchPlan.model_validate(state["plan"])
    hop = state.get("hop", 0)
    return [Send("retrieve", RetrieveInput(sub_question=sq.model_dump(), hop=hop,
                                           deadline=state.get("deadline", 0.0)))
            for sq in plan.sub_questions]


def _mk_retrieve(deps: Deps):
    async def retrieve(payload: RetrieveInput) -> dict[str, Any]:
        return await retrieve_node(payload, deps)
    return retrieve


def analyze(state: ResearchState) -> dict[str, Any]:
    """Placeholder for the Critical Analysis agent (Phase 3): per-sub-question evidence digest."""
    plan = ResearchPlan.model_validate(state["plan"])
    by_sq: dict[str, list[dict[str, Any]]] = {sq.id: [] for sq in plan.sub_questions}
    for ev in state.get("evidence", []):
        by_sq.setdefault(ev["sub_question_id"], []).append(ev)
    summaries = {}
    for sq in plan.sub_questions:
        evs = by_sq.get(sq.id, [])
        kinds = ", ".join(sorted({e["source_type"] for e in evs}))
        summaries[sq.id] = Summary(
            sub_question_id=sq.id,
            summary=f"{len(evs)} sources collected ({kinds or 'none'}). "
                    "Critical analysis (claims, contradictions, gaps) lands in Phase 3.",
            evidence_ids=[e["id"] for e in evs],
        ).model_dump()
    return {"summaries": summaries, "gaps": [], "notes": ["analysis placeholder: digest only"]}


def route_after_analysis(state: ResearchState) -> str | list[Send]:
    """Multi-hop: unresolved gaps become new retrieval branches while hops and budget remain."""
    gaps = state.get("gaps", [])
    hop = state.get("hop", 0)
    deadline = state.get("deadline", 0.0)
    out_of_budget = bool(deadline) and time.time() > deadline
    if gaps and hop < state.get("max_hops", DEFAULT_MAX_HOPS) and not out_of_budget:
        sends = []
        for i, g in enumerate(gaps, start=1):
            sq = {"id": f"gap{hop + 1}_{i}", "question": g["question"],
                  "source_types": g.get("source_types", ["web"])}
            sends.append(Send("retrieve", RetrieveInput(sub_question=sq, hop=hop + 1, deadline=deadline)))
        return sends
    return "insight"


def insight(state: ResearchState) -> dict[str, Any]:
    """Placeholder for Insight Generation (Phase 4)."""
    return {"insights": [], "notes": ["insight placeholder"]}


def critic(state: ResearchState) -> dict[str, Any]:
    """Placeholder for the Red-Team Critic (Phase 4): accepts."""
    return {"critique": {"verdict": "accept", "issues": [], "follow_up": []},
            "notes": ["critic placeholder: accept"]}


def report(state: ResearchState) -> dict[str, Any]:
    """Placeholder for the Report Builder (Phase 4): structured evidence digest with bibliography."""
    plan = ResearchPlan.model_validate(state["plan"])
    evidence = state.get("evidence", [])
    by_id = {e["id"]: e for e in evidence}
    lines = [f"# Research digest: {plan.question}", "",
             f"_{len(evidence)} sources across {len(plan.sub_questions)} sub-questions; "
             f"hops used: {state.get('hop', 0)}. Analysis, insights and critique arrive in Phases 3-4._", ""]
    for sq in plan.sub_questions:
        summ = state.get("summaries", {}).get(sq.id, {})
        lines += [f"## {sq.id}: {sq.question}", "", summ.get("summary", ""), ""]
        for eid in summ.get("evidence_ids", []):
            e = by_id.get(eid)
            if e:
                date = f" ({e['published_at']})" if e.get("published_at") else ""
                title = e["title"] or e["url"]
                lines.append(f"- [{eid}] **{title}**{date} | {e['source_type']} | {e['url']}")
        lines.append("")
    lines += ["## Retrieval log", ""]
    for row in state.get("retrieval_log", []):
        lines.append(f"- {row}")
    return {"report": "\n".join(lines), "notes": ["report digest written"]}


# ----------------------------------------------------------------------------- graph

def build_graph(checkpointer: Any = None, deps: Deps | None = None):
    deps = deps or Deps()
    g = StateGraph(ResearchState)
    g.add_node("plan", _mk_plan(deps))
    g.add_node("approve", approve)
    g.add_node("retrieve", _mk_retrieve(deps))
    g.add_node("analyze", analyze)
    g.add_node("insight", insight)
    g.add_node("critic", critic)
    g.add_node("report", report)

    g.add_edge(START, "plan")
    g.add_edge("plan", "approve")
    g.add_conditional_edges("approve", fan_out, ["retrieve"])
    g.add_edge("retrieve", "analyze")
    g.add_conditional_edges("analyze", route_after_analysis, ["retrieve", "insight"])
    g.add_edge("insight", "critic")
    g.add_edge("critic", "report")
    g.add_edge("report", END)
    return g.compile(checkpointer=checkpointer)


_graph: Any = None
_graph_lock = asyncio.Lock()


async def get_checkpointer():
    kind = (os.getenv("CHECKPOINTER") or "sqlite").lower()
    if kind == "memory":
        from langgraph.checkpoint.memory import InMemorySaver

        return InMemorySaver()
    if kind == "sqlite":
        import aiosqlite
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        conn = await aiosqlite.connect(os.getenv("CHECKPOINT_DB") or ".checkpoints.sqlite")
        saver = AsyncSqliteSaver(conn)
        await saver.setup()
        return saver
    raise ValueError(f"CHECKPOINTER={kind!r}; expected memory or sqlite (postgres arrives with Phase 6)")


async def get_graph(deps: Deps | None = None):
    """Process-wide compiled graph (built once; tests pass their own deps via build_graph)."""
    global _graph
    async with _graph_lock:
        if _graph is None:
            _graph = build_graph(await get_checkpointer(), deps)
        return _graph


def reset_graph() -> None:
    global _graph
    _graph = None


async def aclose() -> None:
    """Close the checkpointer's connection (aiosqlite runs a non-daemon thread that otherwise keeps
    the process alive at exit). Safe to call when nothing was built."""
    global _graph
    graph, _graph = _graph, None
    conn = getattr(getattr(graph, "checkpointer", None), "conn", None)
    if conn is not None and hasattr(conn, "close"):
        try:
            await conn.close()
        except Exception as exc:  # noqa: BLE001 - shutdown must not raise
            log.debug("checkpointer close failed: %s", exc)


# ----------------------------------------------------------------------------- streaming API

def _config(thread_id: str, config: dict[str, Any] | None) -> dict[str, Any]:
    cfg = dict(config or {})
    cfg["configurable"] = {**cfg.get("configurable", {}), "thread_id": thread_id}
    return cfg


def _event_for(node: str, update: dict[str, Any] | None) -> tuple[str, dict[str, Any]]:
    update = update or {}
    data: dict[str, Any] = {"node": node, "notes": update.get("notes", [])}
    if node == "retrieve":
        rows = update.get("retrieval_log") or [{}]
        keep = ("sub_question_id", "hop", "tools", "sources", "chunks_stored", "elapsed_s", "skipped",
                "warning", "queries", "filtered_irrelevant")
        data.update({k: v for k, v in rows[0].items() if k in keep})
    elif node == "analyze":
        data["sub_questions"] = list((update.get("summaries") or {}).keys())
        data["gaps"] = len(update.get("gaps") or [])
    elif node == "report":
        data["report"] = update.get("report")
    elif node == "plan":
        data["plan"] = update.get("plan")
    return node, data


async def _stream(graph: Any, inp: Any, cfg: dict[str, Any]) -> AsyncIterator[tuple[str, dict[str, Any]]]:
    async for chunk in graph.astream(inp, config=cfg, stream_mode="updates"):
        for node, update in chunk.items():
            if node == "__interrupt__":
                payload = update[0].value if update else {}
                yield "plan", {"status": "awaiting_approval", "plan": payload.get("plan"),
                               "message": payload.get("message")}
                return
            yield _event_for(node, update)


async def stream_until_interrupt(*, question: str, thread_id: str, max_hops: int = DEFAULT_MAX_HOPS,
                                 budget_seconds: int | None = None, config: dict[str, Any] | None = None,
                                 graph: Any = None) -> AsyncIterator[tuple[str, dict[str, Any]]]:
    graph = graph or await get_graph()
    deadline = time.time() + budget_seconds if budget_seconds else 0.0
    state: dict[str, Any] = {"question": question, "max_hops": max_hops, "deadline": deadline,
                             "evidence": [], "retrieval_log": [], "notes": []}
    async for ev in _stream(graph, state, _config(thread_id, config)):
        yield ev


async def resume_with_plan(*, thread_id: str, plan: dict[str, Any] | bool, budget_seconds: int | None = None,
                           config: dict[str, Any] | None = None, graph: Any = None,
                           ) -> AsyncIterator[tuple[str, dict[str, Any]]]:
    graph = graph or await get_graph()
    cfg = _config(thread_id, config)
    if budget_seconds:   # a fresh budget for the resumed leg
        await graph.aupdate_state(cfg, {"deadline": time.time() + budget_seconds})
    resume = {"plan": plan} if isinstance(plan, dict) else True
    async for ev in _stream(graph, Command(resume=resume), cfg):
        yield ev
