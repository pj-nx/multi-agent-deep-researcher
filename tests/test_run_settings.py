"""Per-run settings: validation, model resolution, Deps construction, budget capping and the API wiring."""
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from deep_researcher import run_settings as RS
from test_api import _client, _load_api


def test_validation_accepts_tiers_and_roles_and_rejects_junk():
    s = RS.RunSettings(models={"heavy": "openai/gpt-5.1", "validator": "google/gemini-3-flash"},
                       reasoning_effort="low", fallback_models=["anthropic/claude-sonnet-5"], budget_seconds=120)
    assert s.model_for("planner") == "openai/gpt-5.1"           # tier override
    assert s.model_for("validator") == "google/gemini-3-flash"  # role override beats the cheap tier
    assert s.model_for("report") == RS.model_for("report")      # untouched role falls back to the server
    assert set(s.resolved_models()) == set(RS.ROLES) and s.summary()["reasoning_effort"] == "low"
    with pytest.raises(ValidationError, match="unknown tier/role"):
        RS.RunSettings(models={"turbo": "openai/gpt-5.1"})
    with pytest.raises(ValidationError, match="not an OpenRouter model id"):
        RS.RunSettings(models={"heavy": "gpt-5.1"})
    with pytest.raises(ValidationError, match="not an OpenRouter model id"):
        RS.RunSettings(fallback_models=["rm -rf /"])
    with pytest.raises(ValidationError):
        RS.RunSettings(reasoning_effort="max")
    with pytest.raises(ValidationError):
        RS.RunSettings(budget_seconds=5000)


def test_empty_settings_mean_shared_graph():
    assert RS.RunSettings().is_empty() and RS.make_deps(RS.RunSettings()) is None
    assert RS.make_deps(None) is None
    assert RS.RunSettings(budget_seconds=30).is_empty()        # budget alone: no model overrides, shared graph
    assert RS.make_deps(RS.RunSettings(budget_seconds=30)) is None


def test_effective_budget_never_exceeds_the_server():
    assert RS.effective_budget(None, 240) == 240
    assert RS.effective_budget(RS.RunSettings(), 0) is None
    assert RS.effective_budget(RS.RunSettings(budget_seconds=60), 240) == 60
    assert RS.effective_budget(RS.RunSettings(budget_seconds=900), 240) == 240
    assert RS.effective_budget(RS.RunSettings(budget_seconds=900), 0) == 900
    assert RS.effective_budget(RS.RunSettings(budget_seconds=0), 240) == 240   # 0 = server default, not "no cap"
    assert RS.effective_budget(RS.RunSettings(budget_seconds=0), 0) is None


def test_allowlist_bounds_overrides(monkeypatch):
    monkeypatch.setenv("ALLOWED_MODELS", "anthropic/*, openai/gpt-5.1")
    assert RS.allowed_models() == ["anthropic/*", "openai/gpt-5.1"]
    assert RS.is_allowed("anthropic/claude-haiku-4.5") and RS.is_allowed("openai/gpt-5.1")
    assert not RS.is_allowed("openai/o1-pro") and not RS.is_allowed("anthropicx/foo")
    RS.RunSettings(models={"heavy": "anthropic/claude-sonnet-5"}, fallback_models=["openai/gpt-5.1"])
    with pytest.raises(ValidationError, match="not allowed on this server"):
        RS.RunSettings(models={"heavy": "openai/o1-pro"})
    with pytest.raises(ValidationError, match="not allowed on this server"):
        RS.RunSettings(fallback_models=["google/gemini-3-pro"])
    monkeypatch.delenv("ALLOWED_MODELS")
    assert RS.is_allowed("openai/o1-pro")                                    # empty allowlist = anything


def test_make_deps_builds_models_with_overrides(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    monkeypatch.delenv("OPENROUTER_FALLBACK_MODELS", raising=False)
    monkeypatch.delenv("OPENROUTER_REASONING_EFFORT", raising=False)
    s = RS.RunSettings(models={"heavy": "openai/gpt-5.1"}, reasoning_effort="none",
                       fallback_models=["anthropic/claude-sonnet-5", "openai/gpt-5.1"])
    deps = RS.make_deps(s)
    planner = deps.llm("planner")
    assert planner.model_name == "openai/gpt-5.1" and deps.llm("planner") is planner   # cached per run
    assert planner.extra_body["models"] == ["openai/gpt-5.1", "anthropic/claude-sonnet-5"]  # no duplicate
    assert planner.extra_body["reasoning"] == {"enabled": False}
    assert deps.llm("validator").model_name == RS.model_for("validator")
    assert "openai/gpt-5.1: planner, analysis, insight, critic" in deps.model_names()


@pytest.mark.anyio
@pytest.mark.real_graph                     # the conftest fakes graph_for for API tests
async def test_graph_for_shares_the_checkpointer(monkeypatch, tmp_path):
    pytest.importorskip("chromadb")
    monkeypatch.setenv("CHECKPOINTER", "memory")
    from deep_researcher import graph as G
    from fakes import make_deps

    G.reset_graph()
    shared = await G.graph_for(None)
    custom = await G.graph_for(make_deps(tmp_path))
    assert custom is not shared and custom.checkpointer is shared.checkpointer
    assert await G.graph_for(None) is shared
    G.reset_graph()


@pytest.mark.anyio
async def test_api_accepts_settings_and_rejects_bad_ones(monkeypatch):
    mod = _load_api(monkeypatch, RUN_BUDGET_SECONDS="240")
    seen = {}
    import deep_researcher.graph as G

    async def fake_graph_for(deps=None):
        seen["deps"] = deps
        return "graph"

    monkeypatch.setattr(G, "graph_for", fake_graph_for)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    body = {"question": "Does remote work help?", "max_hops": 0,
            "settings": {"models": {"heavy": "openai/gpt-5.1"}, "reasoning_effort": "low", "budget_seconds": 900}}
    async with _client(mod) as c:
        r = await c.post("/api/research", json=body)
        assert r.status_code == 200
        assert "event: settings" in r.text and '"openai/gpt-5.1"' in r.text and "event: plan" in r.text
        assert '"budget_seconds": 240' in r.text                       # capped by the server
        assert seen["deps"] is not None and seen["deps"].llm("planner").model_name == "openai/gpt-5.1"
        r = await c.post("/api/research", json={"question": "Does remote work help?",
                                                "settings": {"models": {"heavy": "nonsense"}}})
        assert r.status_code == 422
        h = (await c.get("/api/health")).json()
        assert h["run_settings"]["budget_cap_seconds"] == 240 and "models" in h["run_settings"]["overridable"]
        assert h["run_settings"]["allowed_models"] is None and "reasoning_effort" in h["llm"]


@pytest.mark.anyio
async def test_api_enforces_the_allowlist_and_filters_the_catalogue(monkeypatch):
    mod = _load_api(monkeypatch, ALLOWED_MODELS="anthropic/*")

    async def fake_list(prefix="", *, timeout=15.0):
        return [{"id": "openai/gpt-5.1", "context_length": 1, "prompt_price": "0", "completion_price": "0"},
                {"id": "anthropic/claude-sonnet-5", "context_length": 1, "prompt_price": "0", "completion_price": "0"}]

    monkeypatch.setattr(mod.llm, "alist_models", fake_list)
    async with _client(mod) as c:
        r = await c.post("/api/research", json={"question": "Does remote work help?",
                                                "settings": {"models": {"heavy": "openai/gpt-5.1"}}})
        assert r.status_code == 422 and "not allowed on this server" in r.text
        r = await c.get("/api/models")
        assert [m["id"] for m in r.json()["models"]] == ["anthropic/claude-sonnet-5"]
        assert r.json()["allowed_models"] == ["anthropic/*"]
        assert (await c.get("/api/health")).json()["run_settings"]["allowed_models"] == ["anthropic/*"]


def test_api_skips_dotenv_when_told(monkeypatch):
    import dotenv

    calls = []
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: calls.append(a) or False)
    _load_api(monkeypatch)                                   # conftest sets DR_LOAD_DOTENV=0
    assert calls == []
    _load_api(monkeypatch, DR_LOAD_DOTENV="1")
    assert len(calls) == 1 and str(calls[0][0]).endswith(".env")


@pytest.mark.anyio
async def test_models_endpoint_is_cached_and_guarded(monkeypatch):
    mod = _load_api(monkeypatch, API_AUTH_TOKEN="secret")
    calls = {"n": 0}

    async def fake_list(prefix="", *, timeout=15.0):
        calls["n"] += 1
        return [{"id": "openai/gpt-5.1", "context_length": 400000, "prompt_price": "0.000001",
                 "completion_price": "0.000004"},
                {"id": "anthropic/claude-sonnet-5", "context_length": 200000, "prompt_price": "0.000003",
                 "completion_price": "0.000015"}]

    monkeypatch.setattr(mod.llm, "alist_models", fake_list)
    async with _client(mod) as c:
        assert (await c.get("/api/models")).status_code == 401
        hdr = {"authorization": "Bearer secret"}
        r = await c.get("/api/models", headers=hdr)
        assert r.status_code == 200 and r.json()["count"] == 2 and "planner" in r.json()["defaults"]
        r = await c.get("/api/models", params={"prefix": "openai/"}, headers=hdr)
        assert [m["id"] for m in r.json()["models"]] == ["openai/gpt-5.1"]
    assert calls["n"] == 1
