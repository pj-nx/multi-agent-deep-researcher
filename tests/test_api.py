import importlib
import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

ROOT = Path(__file__).resolve().parents[1]


def _load_api(monkeypatch, **env):
    for var in ("API_AUTH_TOKEN", "RATE_LIMIT_PER_MINUTE", "MAX_CONCURRENT_RUNS", "MAX_HOPS", "RUN_BUDGET_SECONDS",
                "TRUST_PROXY_HEADERS", "VERCEL"):
        monkeypatch.delenv(var, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    import api.index as mod
    mod = importlib.reload(mod)
    mod._graph_available.cache_clear()
    return mod


def _client(mod):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=mod.app), base_url="http://t")


@pytest.mark.anyio
async def test_env_int_tolerates_empty_and_garbage(monkeypatch):
    mod = _load_api(monkeypatch, MAX_HOPS="", RUN_BUDGET_SECONDS="abc")
    assert mod.MAX_HOPS == 1 and mod.RUN_BUDGET_SECONDS == 240


@pytest.mark.anyio
async def test_open_by_default_and_health_fields(monkeypatch):
    mod = _load_api(monkeypatch)
    async with _client(mod) as c:
        h = (await c.get("/api/health")).json()
        assert h["auth_required"] is False and h["graph_wired"] is False
        r = await c.post("/api/research", json={"question": "Does remote work help?"})
        assert r.status_code == 200
        body = r.text
    assert "event: started" in body and "event: plan" in body and "event: done" in body


@pytest.mark.anyio
async def test_auth_required_when_token_set(monkeypatch):
    mod = _load_api(monkeypatch, API_AUTH_TOKEN="s3cret")
    async with _client(mod) as c:
        assert (await c.get("/api/health")).json()["auth_required"] is True
        r = await c.post("/api/research", json={"question": "Does remote work help?"})
        assert r.status_code == 401 and r.headers["www-authenticate"] == "Bearer"
        r = await c.post("/api/research", json={"question": "Does remote work help?"},
                         headers={"authorization": "Bearer wrong"})
        assert r.status_code == 401
        r = await c.post("/api/research", json={"question": "Does remote work help?"},
                         headers={"x-api-key": "s3cret"})
        assert r.status_code == 200
        r = await c.post("/api/research", json={"question": "Does remote work help?"},
                         headers={"authorization": "Bearer s3cret"})
        assert r.status_code == 200


@pytest.mark.anyio
async def test_rate_limit(monkeypatch):
    mod = _load_api(monkeypatch, RATE_LIMIT_PER_MINUTE="2")
    async with _client(mod) as c:
        for _ in range(2):
            assert (await c.post("/api/research", json={"question": "Does remote work help?"})).status_code == 200
        r = await c.post("/api/research", json={"question": "Does remote work help?"})
        assert r.status_code == 429 and "retry-after" in r.headers
        # without a trusted proxy a forged X-Forwarded-For must NOT open a fresh window
        r = await c.post("/api/research", json={"question": "Does remote work help?"},
                         headers={"x-forwarded-for": "203.0.113.9, 10.0.0.1"})
        assert r.status_code == 429


@pytest.mark.anyio
async def test_proxy_headers_honoured_only_when_trusted(monkeypatch):
    mod = _load_api(monkeypatch, RATE_LIMIT_PER_MINUTE="1", VERCEL="1")   # Vercel ⇒ trusted
    assert mod.TRUST_PROXY_HEADERS is True
    async with _client(mod) as c:
        q = {"question": "Does remote work help?"}
        assert (await c.post("/api/research", json=q, headers={"x-real-ip": "198.51.100.1"})).status_code == 200
        assert (await c.post("/api/research", json=q, headers={"x-real-ip": "198.51.100.1"})).status_code == 429
        # a genuinely different client (as reported by the proxy) has its own window
        assert (await c.post("/api/research", json=q, headers={"x-real-ip": "198.51.100.2"})).status_code == 200
        # x-real-ip wins over x-forwarded-for when both are present
        assert (await c.post("/api/research", json=q, headers={
            "x-real-ip": "198.51.100.2", "x-forwarded-for": "203.0.113.77"})).status_code == 429
    mod = _load_api(monkeypatch, TRUST_PROXY_HEADERS="false", VERCEL="1")  # explicit override
    assert mod.TRUST_PROXY_HEADERS is False


def test_rate_limiter_memory_is_bounded(monkeypatch):
    mod = _load_api(monkeypatch)
    now = [1000.0]
    rl = mod.RateLimiter(2, sweep_at=5, clock=lambda: now[0])
    rl.check("a")
    assert len(rl) == 1
    now[0] += 61                      # window expired → key dropped on next touch of that key
    rl.check("a")
    assert len(rl) == 1               # re-added once, not accumulated
    for ip in "bcdef":                # exceed sweep_at with stale keys...
        rl.check(ip)
    now[0] += 61
    rl.check("z")                     # ...sweep evicts every expired key
    assert len(rl) == 1 and "z" in rl._hits


@pytest.mark.anyio
async def test_slot_released_even_if_body_never_streams(monkeypatch):
    mod = _load_api(monkeypatch, MAX_CONCURRENT_RUNS="1")
    slot = mod.run_slots.acquire()
    assert mod.run_slots.active == 1

    async def never_started():
        yield "unreachable"

    resp = mod.SlotStreamingResponse(never_started(), slot=slot)

    async def receive():
        return {"type": "http.disconnect"}

    async def send(message):
        raise OSError("client went away")      # spec 2.4 path: Starlette raises ClientDisconnect

    from starlette.requests import ClientDisconnect
    with pytest.raises(ClientDisconnect):
        await resp({"type": "http", "asgi": {"spec_version": "2.4"}}, receive, send)
    assert mod.run_slots.active == 0
    slot.release()                            # idempotent
    assert mod.run_slots.active == 0


@pytest.mark.anyio
async def test_aiter_accepts_sync_and_async_sources(monkeypatch):
    mod = _load_api(monkeypatch)

    def sync_events():
        yield "a", {"n": 1}
        yield "b", {"n": 2}

    async def async_events():
        yield "c", {"n": 3}

    assert [e async for e in mod._aiter(sync_events())] == [("a", {"n": 1}), ("b", {"n": 2})]
    assert [e async for e in mod._aiter(async_events())] == [("c", {"n": 3})]
    with pytest.raises(TypeError):
        _ = [e async for e in mod._aiter(42)]


@pytest.mark.anyio
async def test_slots_released_after_stream(monkeypatch):
    mod = _load_api(monkeypatch, MAX_CONCURRENT_RUNS="1")
    async with _client(mod) as c:
        for _ in range(3):  # would 503 on the second call if the slot leaked
            assert (await c.post("/api/research", json={"question": "Does remote work help?"})).status_code == 200
    assert mod.run_slots.active == 0


@pytest.mark.anyio
async def test_resume_validates_plan_and_thread_id(monkeypatch):
    mod = _load_api(monkeypatch)
    tid = "7a6264cf-9872-4d26-acf1-fbb144aff0c4"
    async with _client(mod) as c:
        r = await c.post(f"/api/research/{tid}/resume", json={"plan": {}})
        assert r.status_code == 422                       # schema, not a bare dict
        r = await c.post("/api/research/not-a-uuid/resume", json={"plan": {
            "question": "Does remote work help?",
            "sub_questions": [{"id": "sq1", "question": "What do RCTs say?"}]}})
        assert r.status_code == 422
        r = await c.post(f"/api/research/{tid}/resume", json={"plan": {
            "question": "Does remote work help?",
            "sub_questions": [{"id": "sq1", "question": "What do RCTs say?"}]}})
        assert r.status_code == 501                       # valid, but graph not wired yet


def _with_graph_module(monkeypatch, tmp_path, source: str):
    """Drop a temporary deep_researcher/graph.py onto the package path."""
    import importlib as _il

    import deep_researcher

    (tmp_path / "graph.py").write_text(source)
    monkeypatch.setattr(deep_researcher, "__path__", [str(tmp_path), *deep_researcher.__path__])
    monkeypatch.delitem(sys.modules, "deep_researcher.graph", raising=False)
    _il.invalidate_caches()


def test_graph_available_distinguishes_missing_from_broken(monkeypatch, tmp_path, caplog, request):
    mod = _load_api(monkeypatch)

    def _cleanup():  # don't leak a "wired" graph into later tests
        import deep_researcher

        mod._graph_available.cache_clear()
        sys.modules.pop("deep_researcher.graph", None)
        deep_researcher.__dict__.pop("graph", None)

    request.addfinalizer(_cleanup)
    assert mod._graph_available() is False          # genuinely missing: quiet
    assert "graph" not in caplog.text

    mod._graph_available.cache_clear()
    _with_graph_module(monkeypatch, tmp_path, "raise RuntimeError('bug in graph')\n")
    with caplog.at_level("ERROR"):
        assert mod._graph_available() is False      # broken: still degrades...
    assert "failed to import" in caplog.text        # ...but loudly

    mod._graph_available.cache_clear()
    caplog.clear()
    _with_graph_module(monkeypatch, tmp_path, "WIRED = True\n")
    assert mod._graph_available() is True


def test_requirements_cover_core_dependencies():
    """Vercel installs requirements.txt; keep it a superset of pyproject's core deps."""
    import re
    import tomllib

    core = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["dependencies"]
    names = {re.split(r"[<>=!~\[ ]", d, maxsplit=1)[0].lower() for d in core}
    pinned = {re.split(r"[<>=!~\[ ]", line.strip(), maxsplit=1)[0].lower()
              for line in (ROOT / "requirements.txt").read_text().splitlines()
              if line.strip() and not line.startswith("#")}
    assert names <= pinned, f"missing from requirements.txt: {names - pinned}"
