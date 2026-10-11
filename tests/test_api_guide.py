"""The API user guide: one Markdown source for the hosted page, the Gradio tab and GET /api/guide."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from deep_researcher import api_guide
from test_api import _client, _load_api


def test_guide_covers_every_route_event_and_auth_variant():
    md = api_guide.guide_markdown("https://example.test/", auth_required=True, rate_limit_per_minute=6,
                                 max_concurrent_runs=3, max_hops=1, budget_cap_seconds=240,
                                 allowed_models=["anthropic/*"], models={"planner": "x/y"})
    for path in ("/api/health", "/api/models", "/api/guide", "/api/research", "/api/research/{thread_id}/resume",
                 "/api/report/pdf"):
        assert path in md
    for name, _ in api_guide.EVENTS:
        assert f"| `{name}` |" in md
    assert "https://example.test/api/research" in md and "https://example.test/" not in md.split("Base URL")[0]
    assert "requires a token" in md and 'Authorization: Bearer $TOKEN' in md
    assert "6 requests per minute" in md and "3 research streams" in md and "cap 240 s" in md
    assert "`anthropic/*`" in md and "| `planner` | `x/y` |" in md
    open_md = api_guide.guide_markdown("", auth_required=False)
    assert "http://localhost:8000" in open_md and "does **not** require a token" in open_md
    assert "Authorization: Bearer $TOKEN" not in open_md and "no limits are configured" in open_md


def test_guide_from_health_payload():
    health = {"auth_required": True, "rate_limit_per_minute": 2, "max_concurrent_runs": 1, "max_hops": 0,
              "run_budget_seconds": 120, "run_settings": {"budget_cap_seconds": 120, "allowed_models": None},
              "llm": {"models": {"planner": "a/b"}}}
    md = api_guide.guide_from_health(health, "https://h.test")
    assert "2 requests per minute" in md and "cap 120 s" in md and "| `planner` | `a/b` |" in md
    assert "restricted to" not in md


@pytest.mark.anyio
async def test_guide_route_is_public_and_uses_the_request_host(monkeypatch):
    mod = _load_api(monkeypatch, API_AUTH_TOKEN="secret", RUN_BUDGET_SECONDS="90")
    async with _client(mod) as c:
        r = await c.get("/api/guide")
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/markdown")
        assert "http://t/api/research" in r.text and "requires a token" in r.text and "cap 90 s" in r.text
        r = await c.get("/api/guide", headers={"x-forwarded-proto": "https", "x-forwarded-host": "app.example"})
        assert "https://app.example/api/research" in r.text
        assert (await c.get("/api/health")).json()["max_concurrent_runs"] == 3


def test_gradio_tab_text_mentions_the_local_api(monkeypatch):
    from deep_researcher import ui

    monkeypatch.delenv("API_AUTH_TOKEN", raising=False)
    md = ui.api_guide_md()
    assert "uv run uvicorn api.index:app" in md and "http://localhost:8000/api/research" in md
    assert "does **not** require a token" in md
