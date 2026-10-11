"""Input guardrails: profanity, max categories, unrelated categories; fail-open when the classifier is down."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from deep_researcher.guardrails import GuardrailViolation, check_question, has_profanity
from fakes import FakeLLM, _plan


def llm_for(role):
    return FakeLLM(_plan())


def test_profanity_validator_flags_profane_and_passes_domain_vocabulary():
    assert has_profanity("what the shit is going on with inflation")
    assert has_profanity("What the f*ck is going on")
    assert not has_profanity("Do AI coding assistants raise developer productivity?")
    assert not has_profanity("Sexual health outcomes of sex education programs")
    assert not has_profanity("Is cocaine use rising among adults?")


@pytest.mark.anyio
async def test_clean_single_category_passes():
    assert await check_question("Do AI coding assistants raise productivity?", llm_for) == []


@pytest.mark.anyio
async def test_related_two_categories_pass():
    assert await check_question("ZZRELATED AI tools in clinical care", llm_for) == []


@pytest.mark.anyio
async def test_profanity_is_rejected_without_a_model_call():
    def boom(role):
        raise AssertionError("model must not be called")

    with pytest.raises(GuardrailViolation) as e:
        await check_question("what the shit is going on with inflation", boom)
    assert e.value.rule == "profanity"


@pytest.mark.anyio
async def test_more_than_two_categories_rejected():
    with pytest.raises(GuardrailViolation) as e:
        await check_question("ZZTHREE everything about AI, drugs and markets", llm_for)
    assert e.value.rule == "too_many_categories" and "3 categories" in str(e.value)


@pytest.mark.anyio
async def test_unrelated_categories_rejected():
    with pytest.raises(GuardrailViolation) as e:
        await check_question("ZZUNRELATED AI chips and diabetes drugs", llm_for)
    assert e.value.rule == "unrelated_categories"


@pytest.mark.anyio
async def test_classifier_outage_fails_open_but_is_recorded():
    def down(role):
        raise RuntimeError("no key")

    (d,) = await check_question("A fine question about solar power", down)
    assert d["component"] == "guardrails" and d["severity"] == "critical"


@pytest.mark.anyio
async def test_profanity_library_failure_fails_open_but_is_recorded(monkeypatch):
    from deep_researcher import guardrails

    def broken(text):
        raise ImportError("guardrails not installed")

    monkeypatch.setattr(guardrails, "has_profanity", broken)
    ds = await check_question("Do AI coding assistants raise developer productivity?", llm_for)
    assert [d["kind"] for d in ds] == ["profanity_check_unavailable"]


@pytest.mark.anyio
async def test_guardrails_can_be_switched_off(monkeypatch):
    monkeypatch.setenv("GUARDRAILS", "off")
    assert await check_question("ZZTHREE and shit", llm_for) == []


@pytest.mark.anyio
@pytest.mark.real_graph
async def test_planner_node_rejects_before_planning(tmp_path):
    pytest.importorskip("chromadb")
    from deep_researcher.agents.planner import plan_node
    from fakes import make_deps

    deps = make_deps(tmp_path)
    with pytest.raises(GuardrailViolation):
        await plan_node({"question": "ZZTHREE too broad a question"}, deps)
    assert not any(name == "ResearchPlan" for name, _ in deps.fake_llm.prompts)   # planner never ran
