# Copyright (c) 2026 C10 Hackathon G1
import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture
def llm(monkeypatch):
    import os
    for var in list(os.environ):
        if var.startswith("OPENROUTER_"):
            monkeypatch.delenv(var, raising=False)
    import deep_researcher.llm as mod
    mod = importlib.reload(mod)
    mod.reset_cache()
    return mod


def test_defaults_per_role(llm):
    assert llm.model_for("planner") == "anthropic/claude-opus-5.5"
    assert llm.model_for("retriever") == "anthropic/claude-sonnet-5"
    assert llm.model_for("validator") == "anthropic/claude-haiku-4.5"
    assert llm.model_for("grader") == "openai/gpt-5-mini"
    assert llm.model_for("grader") != llm.model_for("validator")  # judge independent of the pipeline


def test_judge_tier_override_leaves_validator_alone(llm, monkeypatch):
    monkeypatch.setenv("OPENROUTER_MODEL_JUDGE", "google/gemini-3-flash-preview")
    monkeypatch.setenv("OPENROUTER_MODEL_CHEAP", "openai/gpt-5-mini")
    assert llm.model_for("grader") == "google/gemini-3-flash-preview"
    assert llm.model_for("validator") == "openai/gpt-5-mini"
    assert set(llm.info()["models"]) == set(llm.ROLES)


def test_tier_and_role_overrides(llm, monkeypatch):
    monkeypatch.setenv("OPENROUTER_MODEL_HEAVY", "openai/gpt-5.1")
    monkeypatch.setenv("OPENROUTER_MODEL_INSIGHT", "google/gemini-3.1-pro-preview")
    assert llm.model_for("planner") == "openai/gpt-5.1"       # tier
    assert llm.model_for("insight") == "google/gemini-3.1-pro-preview"  # role beats tier
    assert llm.model_for("report") == "anthropic/claude-sonnet-5"       # untouched tier


def test_unknown_role(llm):
    with pytest.raises(llm.LLMConfigError):
        llm.model_for("janitor")


def test_missing_key_is_clear_error(llm):
    assert llm.enabled() is False
    assert llm.info()["configured"] is False
    with pytest.raises(llm.LLMConfigError, match="OPENROUTER_API_KEY"):
        llm.make_llm("planner")


def test_build_kwargs_routing(llm, monkeypatch):
    monkeypatch.setenv("OPENROUTER_FALLBACK_MODELS", "openai/gpt-5.1, anthropic/claude-sonnet-5")
    monkeypatch.setenv("OPENROUTER_PROVIDER_ORDER", "anthropic,google-vertex")
    monkeypatch.setenv("OPENROUTER_ALLOW_FALLBACKS", "false")
    monkeypatch.setenv("OPENROUTER_ZDR", "true")
    kw = llm.build_kwargs("retriever")
    assert kw["model"] == "anthropic/claude-sonnet-5"
    # primary first, duplicates removed; routing travels in the OpenRouter-specific request body
    assert kw["extra_body"]["models"] == ["anthropic/claude-sonnet-5", "openai/gpt-5.1"]
    assert kw["extra_body"]["provider"] == {"order": ["anthropic", "google-vertex"], "allow_fallbacks": False,
                                            "zdr": True}
    assert kw["temperature"] == 0.0 and "role:retriever" in kw["tags"]
    assert kw["default_headers"]["X-OpenRouter-Title"]


def test_make_llm_constructs_chat_openai_bound_to_openrouter(llm, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("OPENROUTER_APP_TITLE", "Unit Test")
    model = llm.make_llm("insight", temperature=0.9)
    from langchain_openai import ChatOpenAI
    assert isinstance(model, ChatOpenAI)
    assert model.model_name == "anthropic/claude-opus-5.5"
    assert model.openai_api_base == "https://openrouter.ai/api/v1"
    assert model.openai_api_key.get_secret_value() == "sk-or-test"
    assert model.default_headers["X-OpenRouter-Title"] == "Unit Test"
    assert model.temperature == 0.9
    assert model.metadata["role"] == "insight"


def test_structured_uses_json_schema_method(llm):
    captured = {}

    class Fake:
        def with_structured_output(self, schema, **kw):
            captured.update(schema=schema, **kw)
            return "bound"

    assert llm.structured(Fake(), dict) == "bound"
    assert captured == {"schema": dict, "method": "json_schema"}


def test_reasoning_effort_env(llm, monkeypatch):
    assert "reasoning" not in llm.build_kwargs("critic").get("extra_body", {})
    monkeypatch.setenv("OPENROUTER_REASONING_EFFORT", "low")
    assert llm.build_kwargs("critic")["extra_body"]["reasoning"] == {"effort": "low"}
    monkeypatch.setenv("OPENROUTER_REASONING_EFFORT", "none")
    assert llm.build_kwargs("critic")["extra_body"]["reasoning"] == {"enabled": False}
    monkeypatch.setenv("OPENROUTER_REASONING_EFFORT", "turbo")
    with pytest.raises(llm.LLMConfigError):
        llm.build_kwargs("critic")
    assert llm.ROLE_MAX_TOKENS["critic"] >= 8_000      # reasoning tokens count against the budget
    monkeypatch.delenv("OPENROUTER_REASONING_EFFORT")
    monkeypatch.setenv("OPENROUTER_MAX_TOKENS", "300")  # opt-in cap to fit low OpenRouter credit
    assert llm.build_kwargs("critic")["max_tokens"] == 300


def test_get_llm_caches_without_overrides(llm, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    a, b = llm.get_llm("grader"), llm.get_llm("grader")
    assert a is b
    assert llm.get_llm("grader", temperature=0.5) is not a


def test_reasoning_effort_per_role_override(llm, monkeypatch):
    monkeypatch.setenv("OPENROUTER_REASONING_EFFORT", "high")
    monkeypatch.setenv("OPENROUTER_REASONING_EFFORT_ANALYSIS", "low")
    assert llm.build_kwargs("analysis")["extra_body"]["reasoning"] == {"effort": "low"}
    assert llm.build_kwargs("critic")["extra_body"]["reasoning"] == {"effort": "high"}      # global kept
    monkeypatch.setenv("OPENROUTER_REASONING_EFFORT_CRITIC", "turbo")
    with pytest.raises(llm.LLMConfigError):
        llm.build_kwargs("critic")
