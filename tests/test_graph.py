"""End-to-end graph run with fake LLM, fake tools and a real (fake-embedded) store."""
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
from deep_researcher.schemas import ResearchPlan, SubQuestion
from deep_researcher.state import merge_dicts
from deep_researcher.store import EvidenceStore
from deep_researcher.tools.base import SearchResult

pytestmark = pytest.mark.real_graph


class FakeStructured:
    def __init__(self, plan):
        self.plan = plan

    async def ainvoke(self, messages):
        return self.plan


class FakePlannerLLM:
    def __init__(self, plan):
        self.plan = plan

    def with_structured_output(self, schema, **kwargs):
        assert schema is ResearchPlan and kwargs.get("method") == "json_schema"
        return FakeStructured(self.plan)


class FakeTool:
    def __init__(self, name, source_type, n=2):
        self.name, self.source_type, self.n = name, source_type, n
        self.calls = []

    async def search(self, query, limit=5):
        self.calls.append(query)
        return [SearchResult(title=f"{self.name} {i} on {query[:12]}",
                             url=f"https://{self.name}.test/{abs(hash(query)) % 997}/{i}",
                             text=f"Evidence {i} about {query}. " * 5, source_type=self.source_type,
                             published_at=f"2025-01-0{i + 1}") for i in range(self.n)]


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

    d = Deps(llm=lambda role: FakePlannerLLM(_plan()), store=lambda: store, tools_for=tools_for)
    d.fake_tools = tools
    d.fake_store = store
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
    assert kinds.count("retrieve") == 2                       # one Send branch per remaining sub-question
    assert kinds[-1] == "report" and kinds.index("analyze") > max(i for i, k in enumerate(kinds) if k == "retrieve")
    retrieves = {d["sub_question_id"]: d for k, d in events if k == "retrieve"}
    assert retrieves["sq1"]["tools"] == {"arxiv": 4} and retrieves["sq2"]["tools"] == {"gdelt": 1}
    assert retrieves["sq1"]["queries"] == ["remote work productivity trial", "telework output RCT"]
    # planner's keyword queries are used verbatim; a sub-question without them falls back to its text
    assert deps.fake_tools["papers"].calls == ["remote work productivity trial", "telework output RCT"]
    assert deps.fake_tools["news"].calls == ["What have large employers announced about office returns"]
    assert deps.fake_tools["wikipedia"].calls == []           # edited-out sub-question never ran
    assert deps.fake_store.count() > 0
    report = events[-1][1]["report"]
    assert "## sq1" in report and "## sq2" in report and "sq3" not in report
    assert "E-" in report                                     # evidence ids cited in the digest

    state = await graph.aget_state({"configurable": {"thread_id": "t1"}})
    assert state.values["approved"] is True and len(state.values["evidence"]) == 5
    assert set(state.values["summaries"]) == {"sq1", "sq2"}


@pytest.mark.anyio
async def test_resume_with_true_accepts_plan_as_is(deps):
    graph = G.build_graph(InMemorySaver(), deps)
    _ = [e async for e in G.stream_until_interrupt(question="Does remote work raise productivity?", thread_id="t2",
                                                   graph=graph)]
    events = [e async for e in G.resume_with_plan(thread_id="t2", plan=True, graph=graph)]
    assert [k for k, _ in events].count("retrieve") == 3


@pytest.mark.anyio
async def test_budget_exhausted_skips_retrieval(deps):
    graph = G.build_graph(InMemorySaver(), deps)
    _ = [e async for e in G.stream_until_interrupt(question="Does remote work raise productivity?", thread_id="t3",
                                                   budget_seconds=-5, graph=graph)]
    events = [e async for e in G.resume_with_plan(thread_id="t3", plan=True, graph=graph)]
    skipped = [d for k, d in events if k == "retrieve"]
    assert skipped and all(d.get("skipped") == "budget exhausted" for d in skipped)
    assert deps.fake_store.count() == 0


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
