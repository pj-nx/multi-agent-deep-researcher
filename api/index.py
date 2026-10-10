"""FastAPI entrypoint for Vercel.

Vercel routes /api/* here (see vercel.json). Locally: `uvicorn api.index:app --reload`
or `vercel dev`. The research graph (src/deep_researcher/graph.py) is wired in during
Phase 6 of PLAN.md; until then /api/research streams a stub so the deployment path can
be verified end to end.

Every request is a LangSmith parent run (when LANGSMITH_TRACING=true); graph runs nest
under it and the trace URL is sent to the client in the first SSE event.

Protection (all configurable by env):
  API_AUTH_TOKEN          if set, /api/research* require `Authorization: Bearer <token>`
                          or `X-API-Key: <token>`; /api/health stays public
  RATE_LIMIT_PER_MINUTE   per-client sliding window (default 6; 0 disables)
  MAX_CONCURRENT_RUNS     simultaneous research streams per instance (default 3)
The limiter is in-process: on Vercel each instance counts separately, which still bounds
burst cost; move it to Upstash/Redis if the project outgrows that.
"""
from __future__ import annotations

import asyncio
import functools
import json
import logging
import os
import secrets
import sys
import time
import uuid
from collections import defaultdict, deque
from collections.abc import AsyncIterator
from pathlib import Path

# Vercel installs requirements.txt but not the local package; make `src/` importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from deep_researcher import llm, tracing
from deep_researcher.schemas import ResearchPlan

log = logging.getLogger("deep_researcher.api")
tracing.configure()

app = FastAPI(title="Deep Researcher API", version="0.1.0")


def _env_int(name: str, default: int) -> int:
    """int() of an env var, tolerating unset or empty values (e.g. `MAX_HOPS=` in .env)."""
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        log.warning("%s=%r is not an integer; using %d", name, raw, default)
        return default


MAX_HOPS = _env_int("MAX_HOPS", 1)
RUN_BUDGET_SECONDS = _env_int("RUN_BUDGET_SECONDS", 240)
RATE_LIMIT_PER_MINUTE = _env_int("RATE_LIMIT_PER_MINUTE", 6)
MAX_CONCURRENT_RUNS = _env_int("MAX_CONCURRENT_RUNS", 3)
API_AUTH_TOKEN = os.getenv("API_AUTH_TOKEN") or None
API_TAGS = ["api", "vercel" if os.getenv("VERCEL") else "local"]


class ResearchRequest(BaseModel):
    question: str = Field(min_length=5, max_length=2000)
    max_hops: int | None = Field(default=None, ge=0, le=5)


class ResumeRequest(BaseModel):
    plan: ResearchPlan


# --------------------------------------------------------------------------- protection

def _client_ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def require_auth(request: Request) -> None:
    if API_AUTH_TOKEN is None:
        return
    auth = request.headers.get("authorization", "")
    presented = auth[7:] if auth.lower().startswith("bearer ") else request.headers.get("x-api-key", "")
    if not presented or not secrets.compare_digest(presented, API_AUTH_TOKEN):
        raise HTTPException(status_code=401, detail="Missing or invalid API token.",
                            headers={"WWW-Authenticate": "Bearer"})


class RateLimiter:
    """Sliding-window limiter keyed by client IP. In-process; see module docstring."""

    def __init__(self, per_minute: int) -> None:
        self.per_minute = per_minute
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def check(self, key: str) -> None:
        if self.per_minute <= 0:
            return
        now = time.monotonic()
        window = self._hits[key]
        while window and now - window[0] > 60:
            window.popleft()
        if len(window) >= self.per_minute:
            retry = max(1, int(60 - (now - window[0])) + 1)
            raise HTTPException(status_code=429, detail="Rate limit exceeded.",
                                headers={"Retry-After": str(retry)})
        window.append(now)


rate_limiter = RateLimiter(RATE_LIMIT_PER_MINUTE)
run_slots = asyncio.Semaphore(MAX_CONCURRENT_RUNS)


def guard(request: Request) -> None:
    require_auth(request)
    rate_limiter.check(_client_ip(request))


def acquire_slot() -> None:
    """Non-blocking concurrency cap; release in the stream's `finally`."""
    if run_slots.locked():
        raise HTTPException(status_code=503, detail="Too many research runs in progress; retry shortly.",
                            headers={"Retry-After": "30"})
    # Semaphore.acquire() would await; emulate try-acquire since we checked locked() above.
    run_slots._value -= 1  # noqa: SLF001 - asyncio has no public try_acquire


# --------------------------------------------------------------------------- helpers

def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@functools.cache
def _graph_available() -> bool:
    """True once src/deep_researcher/graph.py exists and imports cleanly.

    Only a *missing* graph module is treated as "not wired"; any other import failure is a
    real bug and is logged loudly instead of silently falling back to the stub.
    """
    try:
        import deep_researcher.graph  # noqa: F401
    except ModuleNotFoundError as exc:
        if exc.name == "deep_researcher.graph":
            return False
        log.exception("deep_researcher.graph has a missing dependency")
        return False
    except Exception:  # any other import error is a real bug: log loudly, then degrade
        log.exception("deep_researcher.graph failed to import")
        return False
    return True


@app.get("/api/health")
async def health() -> dict:
    return {
        "status": "ok",
        "graph_wired": _graph_available(),
        "max_hops": MAX_HOPS,
        "run_budget_seconds": RUN_BUDGET_SECONDS,
        "auth_required": API_AUTH_TOKEN is not None,
        "rate_limit_per_minute": RATE_LIMIT_PER_MINUTE,
        "tracing": tracing.info(),
        "llm": llm.info(),
    }


async def _stub_events(question: str, thread_id: str) -> AsyncIterator[tuple[str, dict]]:
    """Placeholder until the LangGraph graph is wired in."""
    yield "plan", {
        "thread_id": thread_id,
        "status": "awaiting_approval",
        "note": "Graph not wired yet; this is a deployment smoke-test stub.",
        "sub_questions": [],
    }


async def _run_stream(name: str, thread_id: str, inputs: dict, stage: str,
                      events_factory) -> AsyncIterator[str]:
    """Shared SSE envelope: started → events → done, with error event, tracing and slot release."""
    started = time.monotonic()
    last_event: dict = {}
    try:
        with tracing.span(name, inputs=inputs, tags=API_TAGS,
                          metadata={"thread_id": thread_id, "stage": stage}) as run:
            yield _sse("started", {"thread_id": thread_id, **inputs.get("echo", {}),
                                   "trace_url": tracing.run_url(run)})
            try:
                async for event, data in events_factory():
                    last_event = {"event": event, **data}
                    yield _sse(event, data)
            except Exception:  # must reach the client as an event, not a dropped socket
                log.exception("research stream failed thread_id=%s", thread_id)
                yield _sse("error", {"thread_id": thread_id,
                                     "message": "Research run failed; see server logs / trace."})
                if run is not None:
                    run.end(error="stream failed")
                return
            elapsed = round(time.monotonic() - started, 1)
            if run is not None:
                run.end(outputs={"last_event": last_event, "elapsed_s": elapsed})
            yield _sse("done", {"thread_id": thread_id, "elapsed_s": elapsed})
    finally:
        run_slots.release()
        await tracing.aflush()


# --------------------------------------------------------------------------- routes

@app.post("/api/research", dependencies=[Depends(guard)])
async def research(req: ResearchRequest) -> StreamingResponse:
    acquire_slot()
    thread_id = str(uuid.uuid4())
    max_hops = req.max_hops if req.max_hops is not None else MAX_HOPS

    def events():
        if not _graph_available():
            return _stub_events(req.question, thread_id)
        from deep_researcher.graph import stream_until_interrupt  # type: ignore[import-not-found]

        return stream_until_interrupt(
            question=req.question, thread_id=thread_id, max_hops=max_hops,
            budget_seconds=RUN_BUDGET_SECONDS,
            config=tracing.run_config(thread_id, req.question, tags=["plan"]),
        )

    inputs = {"question": req.question, "max_hops": max_hops, "echo": {"question": req.question}}
    return StreamingResponse(_run_stream("api.research", thread_id, inputs, "plan", events),
                             media_type="text/event-stream")


@app.post("/api/research/{thread_id}/resume", dependencies=[Depends(guard)])
async def resume(thread_id: str, req: ResumeRequest) -> StreamingResponse:
    try:
        uuid.UUID(thread_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="thread_id must be a UUID.") from exc
    if not _graph_available():
        raise HTTPException(status_code=501, detail="Research graph not wired yet (see PLAN.md Phase 6).")
    acquire_slot()
    from deep_researcher.graph import resume_with_plan  # type: ignore[import-not-found]

    def events():
        return resume_with_plan(thread_id=thread_id, plan=req.plan.model_dump(),
                                budget_seconds=RUN_BUDGET_SECONDS,
                                config=tracing.run_config(thread_id, tags=["research"]))

    inputs = {"thread_id": thread_id, "plan": req.plan.model_dump()}
    return StreamingResponse(_run_stream("api.resume", thread_id, inputs, "research", events),
                             media_type="text/event-stream")
