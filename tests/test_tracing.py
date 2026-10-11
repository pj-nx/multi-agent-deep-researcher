# Copyright (c) 2026 C10 Hackathon G1
import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture
def tracing(monkeypatch):
    for var in ("LANGSMITH_TRACING", "LANGSMITH_API_KEY", "LANGSMITH_PROJECT", "VERCEL",
                "LANGCHAIN_CALLBACKS_BACKGROUND", "LANGCHAIN_TRACING_V2"):
        monkeypatch.delenv(var, raising=False)
    import deep_researcher.tracing as mod
    return importlib.reload(mod)


def test_disabled_without_env(tracing):
    assert tracing.configure() is False
    assert tracing.enabled() is False
    assert tracing.client() is None
    assert tracing.info() == {"enabled": False, "project": None, "endpoint": None}
    with tracing.span("x") as run:
        assert run is None
    assert tracing.run_url(None) is None
    tracing.flush()  # no-op, must not raise


def test_flag_without_key_is_disabled(tracing, monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    assert tracing.enabled() is False


def test_enabled_with_env(tracing, monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_test")
    monkeypatch.setenv("LANGSMITH_PROJECT", "unit")
    assert tracing.enabled() is True
    assert tracing.info()["project"] == "unit"
    assert tracing.info()["enabled"] is True


def test_vercel_forces_sync_callbacks(tracing, monkeypatch):
    monkeypatch.setenv("VERCEL", "1")
    tracing.configure()
    import os
    assert os.environ["LANGCHAIN_CALLBACKS_BACKGROUND"] == "false"


def test_traced_decorator_is_transparent(tracing):
    @tracing.traced("add", run_type="tool", tags=["t"])
    def add(a, b):
        return a + b

    assert add(2, 3) == 5


def test_run_config_shape(tracing):
    cfg = tracing.run_config("t-1", "why?", tags=["plan"], metadata={"hop": 0}, max_hops=2)
    assert cfg["configurable"] == {"thread_id": "t-1", "max_hops": 2}
    assert cfg["run_name"] == "deep_researcher"
    assert "deep-researcher" in cfg["tags"] and "plan" in cfg["tags"]
    assert cfg["metadata"] == {"thread_id": "t-1", "hop": 0, "question": "why?"}


@pytest.mark.anyio
async def test_api_health_reports_tracing(tracing, monkeypatch):
    import httpx

    from test_api import _load_api

    monkeypatch.setenv("LANGSMITH_TRACING", "false")   # the API loads .env, which may enable tracing
    app = _load_api(monkeypatch).app        # a fresh module: other tests may have loaded it with a token

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
        body = (await c.get("/api/health")).json()
        assert body["tracing"]["enabled"] is False
        async with c.stream("POST", "/api/research", json={"question": "Does remote work help?"}) as s:
            text = "".join([chunk async for chunk in s.aiter_text()])
    assert "event: started" in text and '"trace_url": null' in text and "event: done" in text
