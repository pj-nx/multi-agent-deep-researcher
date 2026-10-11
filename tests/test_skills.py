"""skills.md: definitions are valid, a broken file never crashes an agent, and every agent's system prompt
carries exactly its skills at the moment it runs (checked over a full fake pipeline run)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from deep_researcher import skills

REQUESTED = {"academic-paper-search", "heavy-document-retrieval", "citation-verification",
             "adaptive-research-planning", "content-safety-guardrails"}
FOCUS_AREAS = {"autonomous-decision-making", "search-quality", "context-window-management",
               "factual-correctness", "research-depth-completeness", "hallucination-prevention",
               "tool-call-efficiency"}
# structured-output schema -> the agent that makes that call
AGENT_OF_SCHEMA = {"ResearchPlan": "planner", "SourceAssessments": "validator", "AnalysisOutput": "analysis",
                   "InsightOutput": "insight", "CritiqueOut": "critic", "ReportOutput": "report",
                   "ScopeVerdict": "guardrails"}


def test_skills_file_is_valid_and_complete():
    loaded = skills.load_skills()
    assert set(loaded) == REQUESTED | FOCUS_AREAS
    assert all(len(t) > 200 for t in loaded.values())
    assert skills.validate_skills() == []           # every mapped skill exists, none unused, none empty
    for agent in skills.AGENT_SKILLS:
        assert skills.skill_prompt(agent), agent


def test_planner_carries_the_supervisor_prompt():
    text = skills.skill_prompt("planner")
    assert "lead research supervisor" in text and "Never launch additional work merely to look thorough" in text
    assert "citation-verification" not in text      # planner does not get the citation skill


def test_switch_off_and_unknown_agent(monkeypatch):
    assert skills.skill_prompt("unknown-agent") == ""
    monkeypatch.setenv("SKILLS", "off")
    assert skills.skill_prompt("planner") == ""


def test_missing_skill_degrades_instead_of_raising(monkeypatch, caplog):
    monkeypatch.setitem(skills.AGENT_SKILLS, "planner", ("adaptive-research-planning", "no-such-skill"))
    monkeypatch.setattr(skills, "_report_once", lambda agent, kind: None)
    text = skills.skill_prompt("planner")                # must not raise inside an agent call
    assert "adaptive-research-planning" in text and "no-such-skill" not in text
    assert "no-such-skill" in " ".join(skills.validate_skills())
    assert "skill_missing" in caplog.text


def test_unreadable_file_degrades_instead_of_raising(monkeypatch):
    def unreadable(path=None):
        raise FileNotFoundError("skills.md")

    monkeypatch.setattr(skills, "load_skills", unreadable)
    monkeypatch.setattr(skills, "_report_once", lambda agent, kind: None)
    assert skills.skill_prompt("critic") == ""
    assert skills.validate_skills()                      # the problem is still reported at build time


@pytest.mark.anyio
@pytest.mark.real_graph
async def test_every_agent_receives_exactly_its_skills_when_it_runs(tmp_path):
    pytest.importorskip("chromadb")
    from langgraph.checkpoint.memory import InMemorySaver

    from deep_researcher import graph as G
    from fakes import make_deps

    deps = make_deps(tmp_path)
    seen: dict[str, list[str]] = {}
    orig = deps.fake_llm.with_structured_output

    def spy(schema, **kw):
        runnable = orig(schema, **kw)
        inner = runnable.ainvoke

        async def ainvoke(msgs):
            seen.setdefault(schema.__name__, []).append(msgs[0].content)
            return await inner(msgs)
        runnable.ainvoke = ainvoke
        return runnable

    deps.fake_llm.with_structured_output = spy
    graph = G.build_graph(InMemorySaver(), deps)
    events = [e async for e in G.stream_until_interrupt(question="Does remote work raise productivity?",
                                                        thread_id="s1", graph=graph)]
    plan = events[-1][1]["plan"]
    [e async for e in G.resume_with_plan(thread_id="s1", plan=plan, graph=graph)]

    all_skills = set(skills.load_skills())
    assert set(seen) >= set(AGENT_OF_SCHEMA), f"agents that never ran: {set(AGENT_OF_SCHEMA) - set(seen)}"
    for schema, agent in AGENT_OF_SCHEMA.items():
        expected = set(skills.AGENT_SKILLS[agent])
        for system in seen[schema]:
            got = {n for n in all_skills if f"### Skill: {n}\n" in system}
            assert got == expected, f"{agent}: got {sorted(got)}, expected {sorted(expected)}"
