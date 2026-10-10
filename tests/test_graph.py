"""End-to-end graph run with fake LLM, fake tools and a real (fake-embedded) store."""
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

pytest.importorskip("chromadb")
from langchain_core.embeddings import DeterministicFakeEmbedding
from langgraph.checkpoint.memory import InMemorySaver

from deep_researcher import graph as G
from deep_researcher.agents.deps import Deps
from deep_researcher.agents.planner import normalise_plan
from deep_researcher.schemas import (
    AnalysisOutput,
    Claim,
    ContradictionOut,
    GapOut,
    ResearchPlan,
    SourceAssessment,
    SourceAssessments,
    SubQuestion,
)
from deep_researcher.state import merge_dicts
from deep_researcher.store import EvidenceStore
from deep_researcher.tools.base import SearchResult

pytestmark = pytest.mark.real_graph


class FakeStructured:
    def __init__(self, fn):
        self.fn = fn

    async def ainvoke(self, messages):
        return self.fn(messages)


class FakeLLM:
    """Dispatches on the structured-output schema; records every prompt it saw."""

    def __init__(self, plan, *, gap_rounds=1, contradiction=True):
        self.plan, self.gap_rounds, self.contradiction = plan, gap_rounds, contradiction
        self.prompts = []
        self.analysis_calls = 0

    def with_structured_output(self, schema, **kwargs):
        assert kwargs.get("method") == "json_schema"
        return FakeStructured(lambda msgs: self._answer(schema, msgs))

    def _answer(self, schema, msgs):
        prompt = msgs[-1].content
        self.prompts.append((schema.__name__, prompt))
        if schema is ResearchPlan:
            return self.plan
        if schema is SourceAssessments:
            ids = re.findall(r'<source id="([^"]+)"', prompt)
            items = []
            for i, eid in enumerate(ids):
                off_topic = "OFFTOPIC" in prompt.split(f'id="{eid}"', 1)[1].split("</source>", 1)[0]
                items.append(SourceAssessment(evidence_id=eid, relevance=0.1 if off_topic else 0.9,
                                              credibility=0.85 if i == 0 else 0.5, keep=not off_topic,
                                              issues=["off-topic"] if off_topic else []))
            return SourceAssessments(items=items)
        if schema is AnalysisOutput:
            self.analysis_calls += 1
            ids = re.findall(r"\[(E-[0-9a-f]{10})\]", prompt)
            sq_id = re.search(r"Sub-question \((\w+)\)", prompt).group(1)
            claims = [Claim(text=f"finding from {ids[0]}", evidence_ids=[ids[0]], confidence=0.8),
                      Claim(text="hallucinated", evidence_ids=["E-deadbeef00"], confidence=0.9)]  # must be dropped
            contradictions = []
            if self.contradiction and "RAISES" in prompt and "LOWERS" in prompt:
                a = next(i for i in ids if i in prompt.split("RAISES")[0][-400:]) if False else ids[0]
                b = ids[1] if len(ids) > 1 else ids[0]
                contradictions.append(ContradictionOut(claim_a="productivity rises", claim_b="productivity falls",
                                                       evidence_ids_a=[a], evidence_ids_b=[b],
                                                       explanation="different populations"))
            gaps = []
            hop_marker = prompt.count("gap")  # gap-derived sub-questions never raise further gaps
            if sq_id == "sq1" and self.gap_rounds > 0 and not sq_id.startswith("gap"):
                self.gap_rounds -= 1
                n = self.analysis_calls
                q = "What do long-term follow-ups show?" if self.gap_rounds >= 0 and n <= 2 else f"Open question {n}?"
                gaps.append(GapOut(question=q, source_types=["papers"],
                                   search_queries=["long-term follow-up"], reason="only short trials"))
            _ = hop_marker
            return AnalysisOutput(summary=f"Summary for {sq_id} citing [{ids[0]}].", claims=claims,
                                  contradictions=contradictions, gaps=gaps, confidence=0.7)
        raise AssertionError(f"unexpected schema {schema}")


class FakeTool:
    def __init__(self, name, source_type, n=2):
        self.name, self.source_type, self.n = name, source_type, n
        self.calls = []

    def _text(self, i, query):
        if self.source_type == "papers":
            verdict = "RAISES" if i % 2 == 0 else "LOWERS"
            return f"Evidence {i} about {query}: the trial {verdict} productivity. " * 4
        return f"Evidence {i} about {query}. " * 5

    async def search(self, query, limit=5):
        self.calls.append(query)
        out = [SearchResult(title=f"{self.name} {i} on {query[:12]}",
                            url=f"https://{self.name}.test/{abs(hash(query)) % 997}/{i}",
                            text=self._text(i, query), source_type=self.source_type,
                            published_at=f"2025-01-0{i + 1}") for i in range(self.n)]
        if self.source_type == "wikipedia":
            out.append(SearchResult(title="Tyler Joseph", url=f"https://{self.name}.test/offtopic/{abs(hash(query)) % 97}",
                                    text=f"OFFTOPIC musician biography {query}", source_type="wikipedia"))
        return out


def _plan():
    return ResearchPlan(question="Does remote work raise productivity?", sub_questions=[
        SubQuestion(id="a", question="What do randomised trials find about remote work productivity?",
                    source_types=["papers"], search_queries=["remote work productivity trial", "telework output RCT"]),
        SubQuestion(id="b", question="What have large employers announced about office returns?", source_types=["news"]),
        SubQuestion(id="c", question="Background on remote work definitions", source_types=["wikipedia"], depends_on=["a"]),
    ])


@pytest.fixture
def deps(tmp_path):
    store = EvidenceStore(DeterministicFakeEmbedding(size=32), embedding_id="fake:32", path=str(tmp_path / "chroma"))
    tools = {"papers": FakeTool("arxiv", "papers"), "news": FakeTool("gdelt", "news", n=1),
             "wikipedia": FakeTool("wikipedia", "wikipedia", n=1)}

    def tools_for(source_types):
        return [tools[s] for s in source_types if s in tools]

    fake_llm = FakeLLM(_plan())
    d = Deps(llm=lambda role: fake_llm, store=lambda: store, tools_for=tools_for)
    d.fake_tools = tools
    d.fake_store = store
    d.fake_llm = fake_llm
    return d


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
    kinds = [k for k, _ in events]
    assert kinds[-1] == "plan" and events[-1][1]["status"] == "awaiting_approval"
    plan = events[-1][1]["plan"]
    assert [sq["id"] for sq in plan["sub_questions"]] == ["sq1", "sq2", "sq3"]

    # user edits the plan before approving: drop the wikipedia sub-question
    plan["sub_questions"] = plan["sub_questions"][:2]
    events = [e async for e in G.resume_with_plan(thread_id="t1", plan=plan, graph=graph)]
    kinds = [k for k, _ in events]
    # hop 0: two Send branches; hop 1: one gap-derived branch (max_hops=1 default)
    assert kinds.count("retrieve") == 3 and kinds.count("validate") == 2 and kinds.count("analyze") == 2
    assert kinds[-1] == "report"
    retrieves = [d for k, d in events if k == "retrieve"]
    by_sq = {d["sub_question_id"]: d for d in retrieves}
    assert by_sq["sq1"]["tools"] == {"arxiv": 4} and by_sq["sq2"]["tools"] == {"gdelt": 1}
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
    assert "## sq1" in report and "## sq2" in report and "## gap1_1" in report and "sq3" not in report
    assert "## Contradictions" in report and "productivity rises" in report
    assert "## Gaps raised" in report and "hallucinated" not in report      # ungrounded claim dropped

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
    events = [e async for e in G.resume_with_plan(thread_id="t2", plan=True, graph=graph)]
    kinds = [k for k, _ in events]
    assert kinds.count("retrieve") == 4                       # 3 planned + 1 gap hop
    validates = [d for k, d in events if k == "validate"]
    assert validates[0]["assessed"] == 7 and validates[0]["kept"] == 6      # the OFFTOPIC wikipedia hit was dropped
    report = events[-1][1]["report"]
    assert "sources dropped by the validator" in report and "Tyler Joseph" in report


@pytest.mark.anyio
async def test_hop_limit_zero_disables_multi_hop(deps):
    graph = G.build_graph(InMemorySaver(), deps)
    _ = [e async for e in G.stream_until_interrupt(question="Does remote work raise productivity?", thread_id="t2b",
                                                   max_hops=0, graph=graph)]
    events = [e async for e in G.resume_with_plan(thread_id="t2b", plan=True, graph=graph)]
    kinds = [k for k, _ in events]
    assert kinds.count("retrieve") == 3 and kinds.count("analyze") == 1
    route = next(d for k, d in events if k == "route_gaps")
    assert route["next"] == "insight" and "hop limit" in " ".join(route["notes"])
    report = events[-1][1]["report"]
    assert "## Gaps raised" in report                         # the gap is still reported, just not pursued


@pytest.mark.anyio
async def test_gap_loop_stops_at_hop_limit_even_when_gaps_keep_coming(deps):
    deps.fake_llm.gap_rounds = 99
    graph = G.build_graph(InMemorySaver(), deps)
    _ = [e async for e in G.stream_until_interrupt(question="Does remote work raise productivity?", thread_id="t2c",
                                                   max_hops=2, graph=graph)]
    events = [e async for e in G.resume_with_plan(thread_id="t2c", plan=True, graph=graph)]
    kinds = [k for k, _ in events]
    assert kinds.count("analyze") == 3                        # hop 0, 1, 2
    state = await graph.aget_state({"configurable": {"thread_id": "t2c"}})
    assert state.values["hop"] == 2


@pytest.mark.anyio
async def test_budget_exhausted_skips_retrieval(deps):
    graph = G.build_graph(InMemorySaver(), deps)
    _ = [e async for e in G.stream_until_interrupt(question="Does remote work raise productivity?", thread_id="t3",
                                                   budget_seconds=-5, graph=graph)]
    events = [e async for e in G.resume_with_plan(thread_id="t3", plan=True, graph=graph)]
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
    events = [e async for e in G.resume_with_plan(thread_id="t4", plan=True, graph=graph)]
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
    events = [e async for e in G.resume_with_plan(thread_id="t5", plan=True, graph=graph)]
    sq1 = next(d for k, d in events if k == "retrieve" and d["sub_question_id"] == "sq1")
    assert sq1["tools"] == {"slow": 0} and sq1["elapsed_s"] < 3


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
