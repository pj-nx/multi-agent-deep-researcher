"""Run health: degrade() records, tool failures surfaced by the retriever, report "Run health" section."""
import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from deep_researcher import health
from deep_researcher.tools.base import RetrievalTool


def test_degrade_returns_record_and_logs(caplog):
    with caplog.at_level("WARNING"):
        rec = health.degrade("tool:x", "rate_limited", "HTTP 429", count=2, sub_question_id="sq1")
    assert rec["component"] == "tool:x" and rec["severity"] == "warn" and rec["count"] == 2
    assert "degraded [warn] tool:x/rate_limited x2" in caplog.text
    with pytest.raises(ValueError):
        health.degrade("x", "y", "z", severity="fatal")


def test_summarise_groups_and_orders_worst_first():
    recs = [
        health.degrade("tool:arxiv", "rate_limited", "429", sub_question_id="sq1"),
        health.degrade("tool:arxiv", "rate_limited", "429", count=2, sub_question_id="sq2"),
        health.degrade("critic", "model_error", "boom", "critical"),
    ]
    out = health.summarise(recs)
    assert [g["component"] for g in out] == ["critic", "tool:arxiv"]
    assert out[1]["count"] == 3 and out[1]["sub_questions"] == ["sq1", "sq2"]
    md = "\n".join(health.health_md(out))
    assert "## Run health" in md and "**critical** · `critic` model error" in md and "x3" in md
    assert health.health_md([]) == []


class _Boom(RetrievalTool):
    name = "boom"
    source_type = "web"
    max_retries = 0

    def __init__(self, status):
        super().__init__(httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(status))))

    async def _search(self, query, limit):
        resp = await self._get("https://example.test/search")
        return resp.json()


@pytest.mark.anyio
@pytest.mark.parametrize("status,kind", [(429, "rate_limited"), (503, "http_503")])
async def test_tool_failure_is_reported_not_swallowed(status, kind):
    tool = _Boom(status)
    errors = []
    token = health.tool_errors.set(errors)
    try:
        assert await tool.search("q") == []              # still never raises
    finally:
        health.tool_errors.reset(token)
    assert [e["kind"] for e in errors] == [kind]


@pytest.mark.anyio
async def test_retriever_turns_tool_errors_into_degradations(tmp_path):
    from deep_researcher.agents.retriever import retrieve_node
    from fakes import FakeTool, make_deps

    class FailingTool(FakeTool):
        async def search(self, query, limit=5):
            health.record_tool_error(self.name, "rate_limited", "HTTP 429")
            return []

    deps = make_deps(tmp_path)
    tool = FailingTool("flaky", "papers")
    deps.tools_for = lambda types: [tool]
    out = await retrieve_node({"sub_question": {"id": "sq1", "question": "Does it work?",
                                                "source_types": ["papers"], "search_queries": ["a", "b"]},
                               "hop": 0, "deadline": 0}, deps)
    (d,) = out["degradations"]
    assert (d["component"], d["kind"], d["count"], d["sub_question_id"]) == ("tool:flaky", "rate_limited", 2, "sq1")


@pytest.mark.anyio
async def test_critic_failure_is_a_critical_degradation(tmp_path):
    from deep_researcher.agents.critic import critic_node
    from fakes import make_deps

    deps = make_deps(tmp_path)

    def broken(role):
        raise RuntimeError("no key")

    deps.llm = broken
    state = {"question": "q", "plan": {"sub_questions": []}, "insights": [], "summaries": {}, "revisions": 0}
    out = await critic_node(state, deps)
    (d,) = out["degradations"]
    assert d["component"] == "critic" and d["severity"] == "critical"
    assert out["critique"]["verdict"] == "accept"


@pytest.mark.anyio
@pytest.mark.real_graph
async def test_degradations_reach_the_report_and_the_stream(tmp_path):
    pytest.importorskip("chromadb")
    from langgraph.checkpoint.memory import InMemorySaver

    from deep_researcher import graph as G
    from fakes import make_deps

    deps = make_deps(tmp_path)
    real_llm = deps.llm

    def llm_for(role):
        if role == "critic":
            raise RuntimeError("critic model offline")
        return real_llm(role)

    deps.llm = llm_for
    real_tools_for = deps.tools_for

    def tools_for(types):
        tools = real_tools_for(types)
        for t in tools:
            orig = t.search

            async def search(query, limit=5, _orig=orig, _name=t.name):
                health.record_tool_error(_name, "rate_limited", "HTTP 429 for 'x'")
                return await _orig(query, limit)
            t.search = search
        return tools

    deps.tools_for = tools_for
    graph = G.build_graph(InMemorySaver(), deps)
    events = [e async for e in G.stream_until_interrupt(question="Does remote work raise productivity?",
                                                        thread_id="h1", graph=graph)]
    plan = events[-1][1]["plan"]
    events = [e async for e in G.resume_with_plan(thread_id="h1", plan=plan, graph=graph)]

    degraded = [d for k, d in events if k == "activity" and d["kind"] == "degraded"]
    assert {d["component"] for d in degraded} >= {"critic"} and any(d["component"].startswith("tool:") for d in degraded)
    report = next(d for k, d in events if k == "report")
    components = {g["component"] for g in report["degradations"]}
    assert "critic" in components and any(c.startswith("tool:") for c in components)
    assert "## Run health" in report["report"] and "`critic` model error" in report["report"]
    assert report["stats"]["degradations"] == len(report["degradations"])
