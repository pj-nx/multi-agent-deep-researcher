"""skills.md: every skill an agent names exists, injection can be switched off, prompts really carry it."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from deep_researcher import skills

REQUESTED = {"academic-paper-search", "heavy-document-retrieval", "citation-verification",
             "adaptive-research-planning", "content-safety-guardrails"}


def test_skills_file_has_the_five_skills_and_every_agent_reference_resolves():
    loaded = skills.load_skills()
    assert set(loaded) == REQUESTED and all(len(t) > 200 for t in loaded.values())
    for agent, names in skills.AGENT_SKILLS.items():
        assert skills.skill_prompt(agent), agent
        assert all(n in loaded for n in names), agent


def test_skill_prompt_content_and_switch(monkeypatch):
    text = skills.skill_prompt("planner")
    assert "### Skill: adaptive-research-planning" in text and "### Skill: academic-paper-search" in text
    assert "citation-verification" not in text            # planner does not get the citation skill
    assert skills.skill_prompt("unknown-agent") == ""
    monkeypatch.setenv("SKILLS", "off")
    assert skills.skill_prompt("planner") == ""


def test_unknown_skill_name_fails_loudly(monkeypatch):
    monkeypatch.setitem(skills.AGENT_SKILLS, "planner", ("no-such-skill",))
    with pytest.raises(KeyError):
        skills.skill_prompt("planner")


@pytest.mark.anyio
@pytest.mark.real_graph
async def test_agents_send_their_skills_to_the_model(tmp_path):
    pytest.importorskip("chromadb")
    from deep_researcher.agents.planner import plan_node
    from fakes import make_deps

    deps = make_deps(tmp_path)
    seen = []
    orig = deps.fake_llm.with_structured_output

    def spy(schema, **kw):
        runnable = orig(schema, **kw)
        inner = runnable.ainvoke

        async def ainvoke(msgs):
            seen.append((schema.__name__, msgs[0].content))
            return await inner(msgs)
        runnable.ainvoke = ainvoke
        return runnable

    deps.fake_llm.with_structured_output = spy
    await plan_node({"question": "Does remote work raise productivity?"}, deps)
    system = dict(seen)["ResearchPlan"]
    assert "### Skill: adaptive-research-planning" in system and "Skill: citation-verification" not in system
