"""The research graph: planner → approval → parallel retrievers → analysis loop → synthesis → report.

All seven agents are real: planner, retrievers, source validator, critical analysis (with the
multi-hop gap loop), insight generation, red-team critic (accept / revise / need_more_evidence) and
the report builder with a citation check.

Public entry points used by the API and CLI:
    stream_until_interrupt(question, thread_id, ...)  -> async (event, data) until the plan needs approval
    resume_with_plan(thread_id, plan, ...)            -> async (event, data) through to the report
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from collections.abc import AsyncIterator
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Send, interrupt

from deep_researcher import skills
from deep_researcher.agents.analysis import analyze_node
from deep_researcher.agents.critic import critic_node
from deep_researcher.agents.deps import Deps
from deep_researcher.agents.insight import insight_node
from deep_researcher.agents.planner import plan_node
from deep_researcher.agents.report import report_node
from deep_researcher.agents.retriever import retrieve_node
from deep_researcher.agents.validator import validate_node
from deep_researcher.schemas import ResearchPlan, SubQuestion
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


def _mk_validate(deps: Deps):
    async def validate(state: ResearchState) -> dict[str, Any]:
        return await validate_node(state, deps)
    return validate


def _mk_analyze(deps: Deps):
    async def analyze(state: ResearchState) -> dict[str, Any]:
        return await analyze_node(state, deps)
    return analyze


_WORD = re.compile(r"[a-z][a-z0-9-]{3,}")
_GAP_STOP = {"what", "which", "where", "when", "does", "did", "have", "has", "been", "about", "with",
             "from", "that", "this", "these", "those", "their", "there", "into", "than", "then", "they",
             "them", "were", "exist", "report", "studies", "evidence", "across", "such", "beyond"}


def _gap_terms(question: str) -> set[str]:
    return {w for w in _WORD.findall(question.lower()) if w not in _GAP_STOP}


def select_gaps(gaps: list[dict[str, Any]], summaries: dict[str, dict[str, Any]],
                limit: int) -> list[dict[str, Any]]:
    """De-duplicate near-identical gaps (Jaccard >= 0.5 on content words) and keep the ``limit`` most
    valuable: gaps from the least-answered sub-questions first (fewest claims), then by order raised."""
    def need(g: dict[str, Any]) -> int:
        return len((summaries.get(g.get("parent_sub_question_id", "")) or {}).get("claims", []))

    chosen: list[tuple[dict[str, Any], set[str]]] = []
    for g in sorted(gaps, key=need):
        terms = _gap_terms(g["question"])
        dup = any(terms and t and len(terms & t) / len(terms | t) >= 0.5 for _, t in chosen)
        if not dup:
            chosen.append((g, terms))
        if len(chosen) >= limit:
            break
    return [g for g, _ in chosen]


def _mk_route_gaps(deps: Deps):
    def route(state: ResearchState) -> dict[str, Any]:
        return route_gaps(state, max_branches=deps.max_gap_branches)
    return route


def route_gaps(state: ResearchState, *, max_branches: int = 4) -> dict[str, Any]:
    """Decide whether pending gaps start another retrieval hop; register them as sub-questions if so."""
    raised = state.get("pending_gaps") or []
    hop = state.get("hop", 0)
    deadline = state.get("deadline", 0.0)
    out_of_budget = bool(deadline) and time.time() > deadline
    if not raised or hop >= state.get("max_hops", DEFAULT_MAX_HOPS) or out_of_budget:
        why = "no gaps" if not raised else ("hop limit reached" if not out_of_budget else "budget exhausted")
        # Critic-requested gaps we cannot pursue: insights are unchanged, go straight to the report.
        nxt = "report" if state.get("gap_source") == "critic" else "insight"
        return {"next_action": nxt, "pending_gaps": [], "hop": hop,
                "notes": [f"gap router: stop ({why}; {len(raised)} gaps left for the report)"
                          + (" -> report" if nxt == "report" else "")]}
    gaps = select_gaps(raised, state.get("summaries") or {}, max_branches)
    new_subs = []
    for i, g in enumerate(gaps, start=1):
        sq = SubQuestion(id=f"gap{hop + 1}_{i}", question=g["question"],
                         source_types=g.get("source_types") or ["web"],
                         search_queries=g.get("search_queries") or [], rationale=g.get("reason"))
        new_subs.append({**sq.model_dump(), "parent_sub_question_id": g.get("parent_sub_question_id")})
    return {"next_action": "retrieve", "extra_sub_questions": new_subs, "hop": hop + 1, "pending_gaps": [],
            "notes": [f"gap router: hop {hop + 1}, pursuing {len(new_subs)} of {len(raised)} gaps"]}


def after_route_gaps(state: ResearchState) -> str | list[Send]:
    nxt = state.get("next_action")
    if nxt == "report":
        return "report"
    if nxt != "retrieve":
        return "insight"
    hop = state.get("hop", 0)
    current = [sq for sq in state.get("extra_sub_questions", []) if sq["id"].startswith(f"gap{hop}_")]
    return [Send("retrieve", RetrieveInput(sub_question=sq, hop=hop, deadline=state.get("deadline", 0.0)))
            for sq in current]


def _mk_insight(deps: Deps):
    async def insight(state: ResearchState) -> dict[str, Any]:
        return await insight_node(state, deps)
    return insight


def _mk_critic(deps: Deps):
    async def critic(state: ResearchState) -> dict[str, Any]:
        return await critic_node(state, deps)
    return critic


def _mk_report(deps: Deps):
    async def report(state: ResearchState) -> dict[str, Any]:
        return await report_node(state, deps)
    return report


def after_critic(state: ResearchState) -> str:
    """accept → report; revise → insight; need_more_evidence → route_gaps (which may still stop)."""
    verdict = (state.get("critique") or {}).get("verdict", "accept")
    if verdict == "revise":
        return "insight"
    if verdict == "need_more_evidence" and state.get("pending_gaps"):
        return "route_gaps"
    return "report"


# ----------------------------------------------------------------------------- graph

def build_graph(checkpointer: Any = None, deps: Deps | None = None):
    deps = deps or Deps()
    for problem in skills.validate_skills():       # a broken skills.md weakens prompts; say so at build time
        log.error("skills.md: %s", problem)
    g = StateGraph(ResearchState)
    g.add_node("plan", _mk_plan(deps))
    g.add_node("approve", approve)
    g.add_node("retrieve", _mk_retrieve(deps))
    g.add_node("validate", _mk_validate(deps))
    g.add_node("analyze", _mk_analyze(deps))
    g.add_node("route_gaps", _mk_route_gaps(deps))
    g.add_node("insight", _mk_insight(deps))
    g.add_node("critic", _mk_critic(deps))
    g.add_node("report", _mk_report(deps))

    g.add_edge(START, "plan")
    g.add_edge("plan", "approve")
    g.add_conditional_edges("approve", fan_out, ["retrieve"])
    g.add_edge("retrieve", "validate")
    g.add_edge("validate", "analyze")
    g.add_edge("analyze", "route_gaps")
    g.add_conditional_edges("route_gaps", after_route_gaps, ["retrieve", "insight", "report"])
    g.add_edge("insight", "critic")
    g.add_conditional_edges("critic", after_critic, ["report", "insight", "route_gaps"])
    g.add_edge("report", END)
    return g.compile(checkpointer=checkpointer)


_graph: Any = None
_graph_lock = asyncio.Lock()


def checkpointer_kind() -> str:
    return (os.getenv("CHECKPOINTER") or "sqlite").lower()


_SQLITE_NOW = "strftime('%Y-%m-%dT%H:%M:%SZ', 'now')"


async def _sqlite_add_created_at(conn: Any) -> None:
    """Give LangGraph's ``checkpoints`` table a ``created_at`` date (UTC, set on insert by a trigger).

    SQLite cannot add a column with a non-constant default, hence nullable column + backfill + trigger.
    Rows that predate the column are dated "now", so their age counts from the first start after upgrade.
    """
    cols = [row[1] for row in await (await conn.execute("PRAGMA table_info(checkpoints)")).fetchall()]
    if "created_at" not in cols:
        await conn.execute("ALTER TABLE checkpoints ADD COLUMN created_at TEXT")
        await conn.execute(f"UPDATE checkpoints SET created_at = {_SQLITE_NOW}")
    await conn.execute(
        "CREATE TRIGGER IF NOT EXISTS checkpoints_created_at AFTER INSERT ON checkpoints "
        "WHEN NEW.created_at IS NULL BEGIN "
        f"UPDATE checkpoints SET created_at = {_SQLITE_NOW} WHERE thread_id = NEW.thread_id "
        "AND checkpoint_ns = NEW.checkpoint_ns AND checkpoint_id = NEW.checkpoint_id; END"
    )
    await conn.commit()


async def get_checkpointer():
    """memory (tests) | sqlite (local, CHECKPOINT_DB) | postgres (hosted, DATABASE_URL)."""
    kind = checkpointer_kind()
    if kind == "memory":
        from langgraph.checkpoint.memory import InMemorySaver

        return InMemorySaver()
    if kind == "sqlite":
        import aiosqlite
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        conn = await aiosqlite.connect(os.getenv("CHECKPOINT_DB") or ".checkpoints.sqlite")
        saver = AsyncSqliteSaver(conn)
        await saver.setup()
        await _sqlite_add_created_at(conn)
        return saver
    if kind == "postgres":
        url = os.getenv("DATABASE_URL")
        if not url:
            raise ValueError("CHECKPOINTER=postgres needs DATABASE_URL (postgresql://user:pass@host:5432/db)")
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        from psycopg import AsyncConnection
        from psycopg.rows import dict_row

        # Same connection settings the library uses in from_conn_string(), but long-lived: the
        # graph is built once per process and aclose() closes the connection on shutdown.
        conn = await AsyncConnection.connect(url, autocommit=True, prepare_threshold=0, row_factory=dict_row)
        saver = AsyncPostgresSaver(conn)
        await saver.setup()
        await conn.execute("ALTER TABLE checkpoints ADD COLUMN IF NOT EXISTS "
                           "created_at timestamptz NOT NULL DEFAULT now()")
        return saver
    raise ValueError(f"CHECKPOINTER={kind!r}; expected memory, sqlite or postgres")


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
                "warning", "queries", "filtered_irrelevant", "filtered_language", "languages_dropped")
        data.update({k: v for k, v in rows[0].items() if k in keep})
    elif node == "validate":
        a = update.get("assessments") or {}
        data.update({"assessed": len(a), "kept": sum(1 for x in a.values() if x.get("keep", True))})
    elif node == "analyze":
        summaries = update.get("summaries") or {}
        data["sub_questions"] = list(summaries)
        data["claims"] = sum(len(s.get("claims", [])) for s in summaries.values())
        data["contradictions"] = len(update.get("contradictions") or [])
        data["gaps"] = [g["question"] for g in (update.get("gaps") or [])]
    elif node == "route_gaps":
        data.update({"next": update.get("next_action"), "hop": update.get("hop"),
                     "new_sub_questions": [sq["id"] for sq in update.get("extra_sub_questions") or []]})
    elif node == "insight":
        data["insights"] = [i["hypothesis"] for i in update.get("insights") or []]
    elif node == "critic":
        c = update.get("critique") or {}
        data.update({"verdict": c.get("verdict"), "issues": c.get("issues", []),
                     "follow_up": [g["question"] for g in c.get("follow_up", [])],
                     "round": update.get("revisions")})
    elif node == "report":
        data["report"] = update.get("report")
        data["report_json"] = update.get("report_json")
        data["stats"] = (update.get("report_json") or {}).get("stats")
        data["degradations"] = (update.get("report_json") or {}).get("degradations") or []
    elif node == "plan":
        data["plan"] = update.get("plan")
    return node, data


async def _stream(graph: Any, inp: Any, cfg: dict[str, Any]) -> AsyncIterator[tuple[str, dict[str, Any]]]:
    """Node updates plus the agents' fine-grained ``activity`` events (model calls, tool searches)."""
    async for mode, chunk in graph.astream(inp, config=cfg, stream_mode=["updates", "custom"]):
        if mode == "custom":
            if isinstance(chunk, dict):
                yield "activity", chunk
            continue
        for node, update in chunk.items():
            if node == "__interrupt__":
                payload = update[0].value if update else {}
                yield "plan", {"status": "awaiting_approval", "plan": payload.get("plan"),
                               "message": payload.get("message")}
                return
            yield _event_for(node, update)


async def _await_checkpoint(graph: Any, cfg: dict[str, Any], key: str, timeout: float = 5.0) -> None:
    """Block until the thread's latest checkpoint contains ``key`` (or ``timeout`` passes).

    Checkpointers commit the final write of a run slightly after ``astream`` yields its last chunk.
    The API answers the plan request and the client may resume on another instance immediately, so
    the checkpoint must be durable *before* we return; otherwise the resume starts a fresh run.
    """
    deadline = time.monotonic() + timeout
    while True:
        try:
            state = await graph.aget_state(cfg)
            if state and state.values.get(key) is not None:
                return
        except Exception as exc:  # noqa: BLE001 - transient read errors while the write lands
            log.debug("checkpoint not readable yet: %s", exc)
        if time.monotonic() > deadline:
            log.warning("checkpoint with %r not visible after %.1fs (thread %s)", key, timeout,
                        cfg["configurable"].get("thread_id"))
            return
        await asyncio.sleep(0.05)


async def stream_until_interrupt(*, question: str, thread_id: str, max_hops: int = DEFAULT_MAX_HOPS,
                                 budget_seconds: int | None = None, config: dict[str, Any] | None = None,
                                 graph: Any = None) -> AsyncIterator[tuple[str, dict[str, Any]]]:
    graph = graph or await get_graph()
    deadline = time.time() + budget_seconds if budget_seconds else 0.0
    state: dict[str, Any] = {"question": question, "max_hops": max_hops, "deadline": deadline,
                             "evidence": [], "retrieval_log": [], "notes": []}
    cfg = _config(thread_id, config)
    async for ev in _stream(graph, state, cfg):
        yield ev
    await _await_checkpoint(graph, cfg, "plan")


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
    await _await_checkpoint(graph, cfg, "report")
