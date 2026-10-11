"""End-to-end graph run with fake LLM, fake tools and a real (fake-embedded) store."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

pytest.importorskip("chromadb")
from langgraph.checkpoint.memory import InMemorySaver

from deep_researcher import graph as G
from deep_researcher.agents.planner import normalise_plan
from deep_researcher.schemas import (
    AnalysisOutput,
    Claim,
    ContradictionOut,
    GapOut,
    Insight,
    InsightOutput,
    ReasoningStep,
    SubQuestion,
)
from deep_researcher.state import merge_dicts
from deep_researcher.tools.base import SearchResult
from fakes import FakeTool, _plan, make_deps

pytestmark = pytest.mark.real_graph


@pytest.fixture
def deps(tmp_path):
    return make_deps(tmp_path)


def test_relevance_gate_drops_offtopic_results():
    from deep_researcher.agents.retriever import keywords, relevance

    sq = SubQuestion(id="x", question="What did the Iceland four-day week trials find about productivity?",
                     search_queries=["Iceland four-day week trial productivity"])
    kws = keywords(sq)
    assert {"iceland", "four-day", "week", "trials", "productivity"} <= kws and "what" not in kws
    on = SearchResult(title="Four-day workweek", url="u1", source_type="wikipedia",
                      text="Iceland ran trials of a four-day week; productivity was maintained.")
    off = SearchResult(title="Tyler Joseph", url="u2", source_type="wikipedia",
                       text="American musician, lead vocalist of Twenty One Pilots.")
    assert relevance(on, kws) >= 0.6 and relevance(off, kws) == 0.0


def test_reducer_merge_dicts():
    assert merge_dicts({"a": 1, "b": 1}, {"b": 2, "c": 3}) == {"a": 1, "b": 2, "c": 3}
    assert merge_dicts(None, {"x": 1}) == {"x": 1}


def test_normalise_plan_reassigns_ids_and_fixes_dependencies():
    plan = normalise_plan(_plan(), "Does remote work raise productivity?")
    assert [sq.id for sq in plan.sub_questions] == ["sq1", "sq2", "sq3"]
    assert plan.sub_questions[2].depends_on == ["sq1"]


@pytest.mark.anyio
async def test_plan_interrupt_then_parallel_retrieval_and_report(deps):
    graph = G.build_graph(InMemorySaver(), deps)
    events = [e async for e in G.stream_until_interrupt(question="Does remote work raise productivity?",
                                                        thread_id="t1", graph=graph)]
    assert any(k == "activity" and d["kind"] == "llm_start" and d["role"] == "planner" for k, d in events)
    events = [e for e in events if e[0] != "activity"]
    kinds = [k for k, _ in events]
    assert kinds[-1] == "plan" and events[-1][1]["status"] == "awaiting_approval"
    plan = events[-1][1]["plan"]
    assert [sq["id"] for sq in plan["sub_questions"]] == ["sq1", "sq2", "sq3"]

    # user edits the plan before approving: drop the wikipedia sub-question
    plan["sub_questions"] = plan["sub_questions"][:2]
    events = [e async for e in G.resume_with_plan(thread_id="t1", plan=plan, graph=graph)]
    activity = [d for k, d in events if k == "activity"]
    assert {a["kind"] for a in activity} >= {"llm_start", "llm_end", "tool_start", "tool_end", "embed_start",
                                              "embed_end", "retrieval"}
    llm_ends = [a for a in activity if a["kind"] == "llm_end"]
    assert {a["role"] for a in llm_ends} == {"validator", "analysis", "insight", "critic", "report"}
    assert all(a["tokens"] == 1200 and a["elapsed_s"] >= 0 for a in llm_ends)
    assert any(a["kind"] == "tool_end" and a["tool"] == "arxiv" and a["sub_question_id"] == "sq1" for a in activity)
    events = [e for e in events if e[0] != "activity"]
    kinds = [k for k, _ in events]
    # hop 0: two Send branches; hop 1: one gap-derived branch (max_hops=1 default)
    assert kinds.count("retrieve") == 3 and kinds.count("validate") == 2 and kinds.count("analyze") == 2
    assert kinds[-1] == "report"
    retrieves = [d for k, d in events if k == "retrieve"]
    by_sq = {d["sub_question_id"]: d for d in retrieves}
    assert by_sq["sq1"]["tools"] == {"arxiv": 4} and by_sq["sq2"]["tools"] == {"gdelt": 2}
    assert by_sq["sq2"]["sources"] == 1 and by_sq["sq2"]["languages_dropped"] == {"fr": 1}   # French article gated out
    assert by_sq["sq1"]["queries"] == ["remote work productivity trial", "telework output RCT"]
    assert "gap1_1" in by_sq and by_sq["gap1_1"]["hop"] == 1 and by_sq["gap1_1"]["queries"] == ["long-term follow-up"]
    # planner's keyword queries are used verbatim; a sub-question without them falls back to its text
    assert deps.fake_tools["papers"].calls[:2] == ["remote work productivity trial", "telework output RCT"]
    assert deps.fake_tools["news"].calls == ["What have large employers announced about office returns"]
    assert deps.fake_tools["wikipedia"].calls == []           # edited-out sub-question never ran

    validates = [d for k, d in events if k == "validate"]
    assert validates[0]["assessed"] == 5 and validates[0]["kept"] == 5
    analyses = [d for k, d in events if k == "analyze"]
    assert set(analyses[0]["sub_questions"]) == {"sq1", "sq2"}
    assert analyses[0]["contradictions"] >= 1 and analyses[0]["gaps"] == ["What do long-term follow-ups show?"]
    # hop 1 is incremental: only the gap sub-question and its parent are re-analysed
    assert set(analyses[1]["sub_questions"]) == {"sq1", "gap1_1"} and analyses[1]["gaps"] == []
    routes = [d for k, d in events if k == "route_gaps"]
    assert routes[0]["next"] == "retrieve" and routes[0]["new_sub_questions"] == ["gap1_1"]
    assert routes[1]["next"] == "insight"

    report = events[-1][1]["report"]
    assert report.startswith("# Remote work and productivity")
    assert "### sq1" in report and "### sq2" in report and "### gap1_1" in report and "sq3:" not in report
    assert "## Where the evidence disagrees" in report and "productivity rises" in report
    assert "hallucinated" not in report                       # ungrounded claim dropped by analysis
    assert "E-deadbeef00" not in report                       # fake citations stripped from insights and report text
    assert "### Insight 1: Effects depend on sector" in report and "Ungrounded speculation" not in report
    assert "## Limitations" in report and "few sources" in report and "## Methodology" in report
    assert "Citation check: 1 citations verified" in report and "2 removed" in report
    assert "## Sources" in report and "credibility" in report
    assert "## Open questions" in report and "long-run effects?" in report
    insight_ev = next(d for k, d in events if k == "insight")
    assert insight_ev["insights"] == ["Effects depend on sector"]
    critic_ev = next(d for k, d in events if k == "critic")
    assert critic_ev["verdict"] == "accept" and critic_ev["round"] == 1
    rj = events[-1][1]["report_json"]
    assert rj["stats"]["citations_removed"] == 2 and rj["stats"]["kept"] == len(rj["sources"])
    assert rj["insights"][0]["reasoning"][1]["evidence_ids"] == []      # stripped, step kept as a logical link

    state = await graph.aget_state({"configurable": {"thread_id": "t1"}})
    assert state.values["approved"] is True and state.values["hop"] == 1
    assert set(state.values["summaries"]) == {"sq1", "sq2", "gap1_1"}
    assert all(a["keep"] for a in state.values["assessments"].values())
    assert state.values["contradictions"][0]["sub_question_id"] == "sq1"
    # validator verdicts were written back onto the store's chunk metadata
    ev_id = state.values["evidence"][0]["id"]
    assert deps.fake_store.query("remote work", k=3, evidence_ids=[ev_id])
    meta = deps.fake_store.collection.get(where={"evidence_id": ev_id}, include=["metadatas"])["metadatas"][0]
    assert "relevance" in meta and meta["keep"] is True


@pytest.mark.anyio
async def test_resume_with_true_accepts_plan_as_is(deps):
    graph = G.build_graph(InMemorySaver(), deps)
    _ = [e async for e in G.stream_until_interrupt(question="Does remote work raise productivity?", thread_id="t2",
                                                   graph=graph)]
    events = [e async for e in G.resume_with_plan(thread_id="t2", plan=True, graph=graph) if e[0] != "activity"]
    kinds = [k for k, _ in events]
    assert kinds.count("retrieve") == 4                       # 3 planned + 1 gap hop
    validates = [d for k, d in events if k == "validate"]
    assert validates[0]["assessed"] == 7 and validates[0]["kept"] == 6      # OFFTOPIC dropped; French never reached it
    rj = events[-1][1]["report_json"]
    assert any(e["title"] == "Tyler Joseph" for e in rj["dropped_sources"])
    assert "Tyler Joseph" not in events[-1][1]["report"]      # dropped sources stay out of the bibliography


@pytest.mark.anyio
async def test_hop_limit_zero_disables_multi_hop(deps):
    graph = G.build_graph(InMemorySaver(), deps)
    _ = [e async for e in G.stream_until_interrupt(question="Does remote work raise productivity?", thread_id="t2b",
                                                   max_hops=0, graph=graph)]
    events = [e async for e in G.resume_with_plan(thread_id="t2b", plan=True, graph=graph) if e[0] != "activity"]
    kinds = [k for k, _ in events]
    assert kinds.count("retrieve") == 3 and kinds.count("analyze") == 1
    route = next(d for k, d in events if k == "route_gaps")
    assert route["next"] == "insight" and "hop limit" in " ".join(route["notes"])
    report = events[-1][1]["report"]
    assert "not pursued" in report                            # the gap is still reported, just not pursued


@pytest.mark.anyio
async def test_gap_loop_stops_at_hop_limit_even_when_gaps_keep_coming(deps):
    deps.fake_llm.gap_rounds = 99
    graph = G.build_graph(InMemorySaver(), deps)
    _ = [e async for e in G.stream_until_interrupt(question="Does remote work raise productivity?", thread_id="t2c",
                                                   max_hops=2, graph=graph)]
    events = [e async for e in G.resume_with_plan(thread_id="t2c", plan=True, graph=graph) if e[0] != "activity"]
    kinds = [k for k, _ in events]
    assert kinds.count("analyze") == 3                        # hop 0, 1, 2
    state = await graph.aget_state({"configurable": {"thread_id": "t2c"}})
    assert state.values["hop"] == 2


@pytest.mark.anyio
async def test_budget_exhausted_skips_retrieval(deps):
    graph = G.build_graph(InMemorySaver(), deps)
    _ = [e async for e in G.stream_until_interrupt(question="Does remote work raise productivity?", thread_id="t3",
                                                   budget_seconds=-5, graph=graph)]
    events = [e async for e in G.resume_with_plan(thread_id="t3", plan=True, graph=graph) if e[0] != "activity"]
    skipped = [d for k, d in events if k == "retrieve"]
    assert skipped and all(d.get("skipped") == "budget exhausted" for d in skipped)
    assert deps.fake_store.count() == 0
    assert [k for k, _ in events][-1] == "report"             # no evidence: analysis degrades, report still written


@pytest.mark.anyio
async def test_no_enabled_tools_is_reported_not_fatal(deps):
    deps.tools_for = lambda source_types: []
    graph = G.build_graph(InMemorySaver(), deps)
    _ = [e async for e in G.stream_until_interrupt(question="Does remote work raise productivity?", thread_id="t4",
                                                   graph=graph)]
    events = [e async for e in G.resume_with_plan(thread_id="t4", plan=True, graph=graph) if e[0] != "activity"]
    warns = [d for k, d in events if k == "retrieve"]
    assert all("no enabled tools" in d.get("warning", "") for d in warns) and [k for k, _ in events][-1] == "report"


@pytest.mark.anyio
async def test_slow_tool_is_bounded_by_timeout(deps):
    import asyncio

    class SlowTool(FakeTool):
        async def search(self, query, limit=5):
            await asyncio.sleep(5)
            return []

    deps.fake_tools["papers"] = SlowTool("slow", "papers")
    deps.tool_timeout_s = 0.2
    graph = G.build_graph(InMemorySaver(), deps)
    _ = [e async for e in G.stream_until_interrupt(question="Does remote work raise productivity?", thread_id="t5",
                                                   graph=graph)]
    events = [e async for e in G.resume_with_plan(thread_id="t5", plan=True, graph=graph) if e[0] != "activity"]
    sq1 = next(d for k, d in events if k == "retrieve" and d["sub_question_id"] == "sq1")
    assert sq1["tools"] == {"slow": 0} and sq1["elapsed_s"] < 3


@pytest.mark.anyio
async def test_critic_revise_regenerates_insights_once(deps):
    deps.fake_llm.verdicts = ["revise", "accept"]
    graph = G.build_graph(InMemorySaver(), deps)
    _ = [e async for e in G.stream_until_interrupt(question="Does remote work raise productivity?", thread_id="c1",
                                                   max_hops=0, graph=graph)]
    events = [e async for e in G.resume_with_plan(thread_id="c1", plan=True, graph=graph) if e[0] != "activity"]
    kinds = [k for k, _ in events]
    assert kinds.count("insight") == 2 and kinds.count("critic") == 2 and kinds[-1] == "report"
    insights = [d for k, d in events if k == "insight"]
    assert insights[1]["insights"] == ["Revised: Effects depend on sector"]
    critics = [d for k, d in events if k == "critic"]
    assert [c["verdict"] for c in critics] == ["revise", "accept"] and critics[0]["issues"] == ["issue from round 1"]
    assert "round 1: revise" in events[-1][1]["report"] and "round 2: accept" in events[-1][1]["report"]


@pytest.mark.anyio
async def test_critic_need_more_evidence_triggers_a_retrieval_hop(deps):
    deps.fake_llm.verdicts = ["need_more_evidence", "accept"]
    deps.fake_llm.gap_rounds = 0                               # analysis raises no gaps: the critic drives the hop
    graph = G.build_graph(InMemorySaver(), deps)
    _ = [e async for e in G.stream_until_interrupt(question="Does remote work raise productivity?", thread_id="c2",
                                                   max_hops=1, graph=graph)]
    events = [e async for e in G.resume_with_plan(thread_id="c2", plan=True, graph=graph) if e[0] != "activity"]
    kinds = [k for k, _ in events]
    assert kinds.count("retrieve") == 4                        # 3 planned + 1 critic-requested
    gap = next(d for k, d in events if k == "retrieve" and d["sub_question_id"] == "gap1_1")
    assert gap["queries"] == ["manufacturing four-day week"]
    assert kinds.count("critic") == 2 and kinds.index("critic") < kinds.index("route_gaps", kinds.index("critic"))
    assert kinds[-1] == "report"
    critics = [d for k, d in events if k == "critic"]
    assert critics[0]["follow_up"] == ["What do manufacturing-sector trials show?"]
    assert "requested by critic" in events[-1][1]["report"] or "gap1_1" in events[-1][1]["report"]


@pytest.mark.anyio
async def test_critic_need_more_evidence_without_hops_left_proceeds_to_report(deps):
    deps.fake_llm.verdicts = ["need_more_evidence", "accept"]
    deps.fake_llm.gap_rounds = 0
    graph = G.build_graph(InMemorySaver(), deps)
    _ = [e async for e in G.stream_until_interrupt(question="Does remote work raise productivity?", thread_id="c3",
                                                   max_hops=0, graph=graph)]
    events = [e async for e in G.resume_with_plan(thread_id="c3", plan=True, graph=graph) if e[0] != "activity"]
    kinds = [k for k, _ in events]
    assert kinds.count("retrieve") == 3 and kinds.count("critic") == 1 and kinds.count("insight") == 1
    assert kinds[-1] == "report"
    route = [d for k, d in events if k == "route_gaps"][-1]
    assert route["next"] == "report"                           # critic's request cannot run: straight to report
    assert "hop limit" in " ".join(route["notes"])


@pytest.mark.anyio
async def test_revision_cap_forces_acceptance(deps):
    deps.fake_llm.verdicts = ["revise"]                        # always revise
    deps.max_revisions = 2
    graph = G.build_graph(InMemorySaver(), deps)
    _ = [e async for e in G.stream_until_interrupt(question="Does remote work raise productivity?", thread_id="c4",
                                                   max_hops=0, graph=graph)]
    events = [e async for e in G.resume_with_plan(thread_id="c4", plan=True, graph=graph) if e[0] != "activity"]
    kinds = [k for k, _ in events]
    assert kinds.count("critic") == 3 and kinds.count("insight") == 3 and kinds[-1] == "report"
    last = [d for k, d in events if k == "critic"][-1]
    assert last["verdict"] == "accept" and "revision cap" in " ".join(last["notes"])
    assert "(cap reached, proceeded)" in events[-1][1]["report"]


class FlakyLLM:
    """Raises for the given schemas; otherwise defers to the wrapped fake."""

    def __init__(self, inner, failing):
        self.inner, self.failing = inner, failing

    def with_structured_output(self, schema, **kwargs):
        if schema.__name__ in self.failing:
            class Boom:
                async def ainvoke(self, messages):
                    raise RuntimeError(f"{schema.__name__} exploded")
            return Boom()
        return self.inner.with_structured_output(schema, **kwargs)


@pytest.mark.anyio
async def test_synthesis_failures_degrade_instead_of_killing_the_run(deps):
    flaky = FlakyLLM(deps.fake_llm, {"InsightOutput", "CritiqueOut", "ReportOutput", "AnalysisOutput"})
    deps.llm = lambda role: flaky
    graph = G.build_graph(InMemorySaver(), deps)
    _ = [e async for e in G.stream_until_interrupt(question="Does remote work raise productivity?", thread_id="f1",
                                                   max_hops=0, graph=graph)]
    events = [e async for e in G.resume_with_plan(thread_id="f1", plan=True, graph=graph) if e[0] != "activity"]
    kinds = [k for k, _ in events]
    assert kinds[-1] == "report" and "error" not in kinds
    analyze = next(d for k, d in events if k == "analyze")
    assert analyze["claims"] == 0 and set(analyze["sub_questions"]) == {"sq1", "sq2", "sq3"}
    assert next(d for k, d in events if k == "insight")["insights"] == []
    critic = next(d for k, d in events if k == "critic")
    assert critic["verdict"] == "accept" and "critic unavailable" in critic["issues"][0]
    report = events[-1][1]["report"]
    assert report.startswith("# Research digest:") and "Analysis failed" in report
    assert "not adversarially reviewed" in report and "## Sources" in report


def test_ground_insights_strips_and_drops():
    from deep_researcher.agents.insight import ground_insights

    out = InsightOutput(insights=[
        Insight(hypothesis="h1", reasoning=[ReasoningStep(step="a", evidence_ids=["E-1111111111", "E-bogus00000"]),
                                            ReasoningStep(step="b", evidence_ids=[])], confidence=0.5),
        Insight(hypothesis="h2", reasoning=[ReasoningStep(step="c", evidence_ids=["E-bogus00000"])], confidence=0.5),
    ])
    kept, stripped, dropped = ground_insights(out, {"E-1111111111"})
    assert [i.hypothesis for i in kept] == ["h1"] and stripped == 2 and dropped == 1
    assert kept[0].reasoning[0].evidence_ids == ["E-1111111111"]


def test_strip_unknown_citations():
    from deep_researcher.agents.common import cited_ids, strip_unknown_citations

    text, removed = strip_unknown_citations("Gains [E-1111111111] and losses [E-2222222222] [E-badbadbad0].",
                                            {"E-1111111111", "E-2222222222"})
    assert removed == 1 and "badbad" not in text and text.endswith("[E-2222222222] .")
    assert cited_ids(text) == ["E-1111111111", "E-2222222222"]


def test_incremental_analysis_skips_parents_whose_gap_evidence_was_all_dropped():
    from deep_researcher.agents.analysis import _touched_this_hop

    base = {
        "hop": 1,
        "extra_sub_questions": [{"id": "gap1_1", "parent_sub_question_id": "sq1"},
                                {"id": "gap1_2", "parent_sub_question_id": "sq2"}],
        "retrieval_log": [{"sub_question_id": "gap1_1", "hop": 1, "sources": 2},
                          {"sub_question_id": "gap1_2", "hop": 1, "sources": 1},
                          {"sub_question_id": "sq1", "hop": 0, "sources": 3}],
        "evidence": [{"id": "E-a", "sub_question_id": "gap1_1"}, {"id": "E-b", "sub_question_id": "gap1_1"},
                     {"id": "E-c", "sub_question_id": "gap1_2"}, {"id": "E-d", "sub_question_id": "sq1"}],
    }
    assessments = {"E-a": {"keep": True}, "E-b": {"keep": False}, "E-c": {"keep": False}, "E-d": {"keep": True}}
    assert _touched_this_hop({**base, "assessments": assessments}) == {"gap1_1", "sq1"}   # gap1_2 all dropped
    assert _touched_this_hop({**base, "assessments": {k: {"keep": False} for k in assessments}}) == set()
    assert _touched_this_hop({**base, "hop": 0}) is None


def test_select_gaps_dedups_and_prioritises_thin_subquestions():
    from deep_researcher.graph import select_gaps

    gaps = [
        {"parent_sub_question_id": "sq1", "question": "How does productivity change in manufacturing sectors?"},
        {"parent_sub_question_id": "sq2", "question": "What do long-term follow-ups show about productivity?"},
        {"parent_sub_question_id": "sq3", "question": "Productivity change in manufacturing and retail sectors?"},
        {"parent_sub_question_id": "sq2", "question": "Which countries ran national pilots?"},
        {"parent_sub_question_id": "sq1", "question": "What about burnout?"},
    ]
    summaries = {"sq1": {"claims": [1, 2, 3]}, "sq2": {"claims": []}, "sq3": {"claims": [1]}}
    chosen = select_gaps(gaps, summaries, limit=3)
    qs = [g["question"] for g in chosen]
    assert len(qs) == 3
    assert qs[0].startswith("What do long-term") and qs[1].startswith("Which countries")   # sq2 has no claims
    # the two near-identical manufacturing gaps collapse to one
    assert sum("manufacturing" in q for q in qs) == 1


def test_ground_drops_unshown_citations():
    from deep_researcher.agents.analysis import ground

    out = AnalysisOutput(
        summary="s",
        claims=[Claim(text="ok", evidence_ids=["E-1111111111", "E-bogus00000"]),
                Claim(text="bad", evidence_ids=["E-bogus00000"])],
        contradictions=[ContradictionOut(claim_a="a", claim_b="b", evidence_ids_a=["E-1111111111"],
                                         evidence_ids_b=["E-bogus00000"])],
        gaps=[GapOut(question="What about X?", source_types=["news"])],
    )
    summary, contradictions, gaps, dropped = ground(out, {"E-1111111111", "E-2222222222"}, "sq9")
    assert [c.text for c in summary.claims] == ["ok"] and summary.claims[0].evidence_ids == ["E-1111111111"]
    assert contradictions == [] and dropped == 2
    assert gaps[0].parent_sub_question_id == "sq9" and summary.evidence_ids == ["E-1111111111"]


def test_rerank_weights_credibility():
    from deep_researcher.agents.analysis import rerank

    class H:
        def __init__(self, eid, score, cred):
            self.score = score
            self.chunk = type("C", (), {"evidence_id": eid, "credibility": cred})()

    hits = [H("low", 0.80, 0.2), H("high", 0.70, 0.9)]
    assert [h.chunk.evidence_id for h in rerank(hits, {})] == ["high", "low"]
    assert [h.chunk.evidence_id for h in rerank(hits, {"low": {"credibility": 1.0}})] == ["low", "high"]


@pytest.mark.anyio
async def test_validator_degrades_without_model(deps, monkeypatch):
    from deep_researcher.agents.validator import validate_node

    def no_llm(role):
        raise RuntimeError("no key")

    deps.llm = no_llm
    state = {"plan": {"sub_questions": [{"id": "sq1", "question": "q"}]}, "assessments": {},
             "evidence": [{"id": "E-aaaaaaaaaa", "sub_question_id": "sq1", "text": "t", "credibility": 0.8}]}
    out = await validate_node(state, deps)
    a = out["assessments"]["E-aaaaaaaaaa"]
    assert a["keep"] is True and a["credibility"] == 0.8 and a["issues"] == ["unvalidated"]
    assert await validate_node({**state, "assessments": out["assessments"]}, deps) == {
        "notes": ["validator: nothing new to assess"]}


@pytest.mark.anyio
async def test_aclose_is_safe_without_graph():
    G.reset_graph()
    await G.aclose()


@pytest.mark.anyio
async def test_memory_checkpointer_selectable(monkeypatch):
    monkeypatch.setenv("CHECKPOINTER", "memory")
    cp = await G.get_checkpointer()
    assert isinstance(cp, InMemorySaver)
    monkeypatch.setenv("CHECKPOINTER", "redis")
    with pytest.raises(ValueError):
        await G.get_checkpointer()


@pytest.mark.anyio
async def test_analysis_skipped_when_the_chunks_shown_are_unchanged(tmp_path):
    from types import SimpleNamespace

    from deep_researcher.agents.analysis import analyze_node

    deps = make_deps(tmp_path)
    deps.fake_llm.gap_rounds = 0
    chunks = [SimpleNamespace(evidence_id="E-0000000001", source_type="papers", credibility=0.8,
                              published_at="2024", title="Trial", text="Productivity rose.")]
    store = SimpleNamespace(query=lambda *a, **k: [SimpleNamespace(chunk=c, score=0.9) for c in chunks])
    deps.store = lambda: store
    state = {"plan": {"sub_questions": [{"id": "sq1", "question": "Does it work?", "source_types": ["papers"]}]},
             "hop": 0}

    first = await analyze_node(state, deps)
    assert deps.fake_llm.analysis_calls == 1 and "sq1" in first["analysis_inputs"]

    state = {**state, "summaries": first["summaries"], "analysis_inputs": first["analysis_inputs"]}
    again = await analyze_node(state, deps)
    assert deps.fake_llm.analysis_calls == 1                                 # unchanged -> no model call
    assert again["summaries"] == {} and again["contradictions"] == []        # earlier summary stays in state
    assert "1 unchanged, skipped" in again["notes"][0]

    chunks[0].text = "Productivity fell."                                    # evidence changed -> re-analysed
    await analyze_node(state, deps)
    assert deps.fake_llm.analysis_calls == 2
