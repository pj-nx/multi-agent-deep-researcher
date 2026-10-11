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
  TRUST_PROXY_HEADERS     true → client IP from x-real-ip / x-vercel-forwarded-for /
                          x-forwarded-for (default: on when VERCEL is set, else off, because
                          a client can forge those headers when no proxy is in front)
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
from collections.abc import AsyncIterator, Callable, Iterable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

# Vercel installs requirements.txt but not the local package; make `src/` importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Local runs (uvicorn, the container) read the git-ignored .env; hosted runs get real env vars and
# have no file, so this is a no-op there. Existing variables are never overridden. The test suite
# sets DR_LOAD_DOTENV=0 so a developer's .env never shapes test results.
from dotenv import load_dotenv

if (os.getenv("DR_LOAD_DOTENV") or "1").strip().lower() not in {"0", "false", "no", "off"}:
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel, Field
from starlette.concurrency import iterate_in_threadpool
from starlette.types import Receive, Scope, Send

from deep_researcher import api_guide, embeddings, llm, tracing
from deep_researcher import documents as docs_mod
from deep_researcher.agents.deps import Deps
from deep_researcher.guardrails import GuardrailViolation
from deep_researcher.reports import markdown_to_pdf
from deep_researcher.run_settings import RunSettings, allowed_models, effective_budget, is_allowed, make_deps
from deep_researcher.schemas import ResearchPlan
from deep_researcher.store import chunking, store_info
from deep_researcher.tools import language as tool_language
from deep_researcher.tools import registry as tool_registry

log = logging.getLogger("deep_researcher.api")
tracing.configure()

@asynccontextmanager
async def _lifespan(_: FastAPI):
    yield
    if _graph_available():
        from deep_researcher.graph import aclose

        await aclose()


app = FastAPI(title="Deep Researcher API", version="0.1.0", lifespan=_lifespan)


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
_TRUTHY = {"1", "true", "yes", "on"}
_trust_default = "true" if os.getenv("VERCEL") else "false"
TRUST_PROXY_HEADERS = (os.getenv("TRUST_PROXY_HEADERS") or _trust_default).lower() in _TRUTHY
API_TAGS = ["api", "vercel" if os.getenv("VERCEL") else "local"]


SET_ID = r"^[A-Za-z0-9_-]{8,64}$"


class ResearchRequest(BaseModel):
    question: str = Field(min_length=5, max_length=2000)
    max_hops: int | None = Field(default=None, ge=0, le=5)
    settings: RunSettings | None = Field(default=None, description="per-run models / effort / budget")
    document_set_id: str | None = Field(default=None, pattern=SET_ID,
                                        description="a set from POST /api/documents to use as a source")


class ResumeRequest(BaseModel):
    plan: ResearchPlan
    settings: RunSettings | None = Field(default=None, description="send the same settings as the plan call")
    document_set_id: str | None = Field(default=None, pattern=SET_ID, description="same set as the plan call")


class PdfRequest(BaseModel):
    markdown: str = Field(min_length=1, max_length=2_000_000)
    title: str | None = Field(default=None, max_length=200)


# --------------------------------------------------------------------------- protection

def _client_ip(request: Request) -> str:
    """Rate-limit key. Proxy headers are only honoured when TRUST_PROXY_HEADERS is on."""
    if TRUST_PROXY_HEADERS:
        for header in ("x-real-ip", "x-vercel-forwarded-for", "x-forwarded-for"):
            value = request.headers.get(header)
            if value:
                return value.split(",")[0].strip()
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
    """Sliding-window limiter keyed by client IP. In-process; see module docstring.

    Memory is bounded: a key is dropped as soon as its window empties, and once the table
    exceeds ``sweep_at`` keys every expired window is evicted before the next check.
    """

    def __init__(self, per_minute: int, *, window_s: float = 60.0, sweep_at: int = 10_000,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.per_minute = per_minute
        self.window_s = window_s
        self.sweep_at = sweep_at
        self._clock = clock
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def __len__(self) -> int:
        return len(self._hits)

    def _expire(self, window: deque[float], now: float) -> None:
        while window and now - window[0] > self.window_s:
            window.popleft()

    def _sweep(self, now: float) -> None:
        for key in [k for k, w in self._hits.items() if not w or now - w[-1] > self.window_s]:
            del self._hits[key]

    def check(self, key: str) -> None:
        if self.per_minute <= 0:
            return
        now = self._clock()
        if len(self._hits) >= self.sweep_at:
            self._sweep(now)
        window = self._hits[key]
        self._expire(window, now)
        if len(window) >= self.per_minute:
            retry = max(1, int(self.window_s - (now - window[0])) + 1)
            if not window:
                del self._hits[key]
            raise HTTPException(status_code=429, detail="Rate limit exceeded.",
                                headers={"Retry-After": str(retry)})
        window.append(now)


class RunSlots:
    """Non-blocking concurrency cap (single event loop, so a plain counter is race-free)."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.active = 0

    def acquire(self) -> Slot:
        if self.active >= self.limit:
            raise HTTPException(status_code=503, detail="Too many research runs in progress; retry shortly.",
                                headers={"Retry-After": "30"})
        self.active += 1
        return Slot(self)


class Slot:
    """Handle for one acquired slot; ``release()`` is idempotent."""

    def __init__(self, pool: RunSlots) -> None:
        self._pool: RunSlots | None = pool

    def release(self) -> None:
        if self._pool is not None:
            self._pool.active -= 1
            self._pool = None


class SlotStreamingResponse(StreamingResponse):
    """StreamingResponse that releases its slot on every exit path.

    Starlette may never start the body generator (client gone before the first byte) or may
    raise ClientDisconnect mid-stream and skip background tasks, so the generator's ``finally``
    alone is not enough. Wrapping ``__call__`` is the one place that always runs.
    """

    def __init__(self, *args: Any, slot: Slot, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.slot = slot

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            self.slot.release()


rate_limiter = RateLimiter(RATE_LIMIT_PER_MINUTE)
run_slots = RunSlots(MAX_CONCURRENT_RUNS)


def guard(request: Request) -> None:
    require_auth(request)
    rate_limiter.check(_client_ip(request))


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


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    """The hosted UI. Vercel serves ``public/`` statically before reaching the function; this route
    makes the same page available under uvicorn / the container."""
    page = Path(__file__).resolve().parents[1] / "public" / "index.html"
    if not page.exists():
        raise HTTPException(status_code=404, detail="public/index.html not found")
    return FileResponse(page, media_type="text/html")


@app.get("/api/guide", include_in_schema=False)
async def guide(request: Request) -> Response:
    """The API user guide as Markdown, filled in with this deployment's URL, auth and limits. Public, like
    health: it reveals nothing the health payload does not."""
    proto = request.headers.get("x-forwarded-proto") or request.url.scheme
    host = request.headers.get("x-forwarded-host") or request.headers.get("host") or request.url.netloc
    text = api_guide.guide_from_health(await health(), f"{proto}://{host}")
    return Response(content=text, media_type="text/markdown; charset=utf-8")


@app.get("/api/health")
async def health() -> dict:
    return {
        "status": "ok",
        "graph_wired": _graph_available(),
        "max_hops": MAX_HOPS,
        "run_budget_seconds": RUN_BUDGET_SECONDS,
        "auth_required": API_AUTH_TOKEN is not None,
        "rate_limit_per_minute": RATE_LIMIT_PER_MINUTE,
        "max_concurrent_runs": MAX_CONCURRENT_RUNS,
        "uploads": {**docs_mod.limits(),
                    "max_request_mb": 4 if os.getenv("VERCEL") else None},   # Vercel caps request bodies
        "run_settings": {"overridable": ["models", "reasoning_effort", "fallback_models", "budget_seconds"],
                         "budget_cap_seconds": RUN_BUDGET_SECONDS or None, "max_hops_limit": 5,
                         "allowed_models": allowed_models() or None},
        "tracing": tracing.info(),
        "llm": llm.info(),
        "embeddings": embeddings.info(),
        "vector_db": store_info(),
        "checkpointer": {"kind": (os.getenv("CHECKPOINTER") or "sqlite").lower(),
                         "database_url_present": bool(os.getenv("DATABASE_URL"))},
        "chunking": {"size": chunking.default_chunk_size(), "overlap": chunking.default_chunk_overlap(),
                     "unit": chunking.default_chunk_unit()},
        "sources": tool_registry.info(),
        "languages": tool_language.info(),
    }


async def _stub_events(question: str, thread_id: str) -> AsyncIterator[tuple[str, dict]]:
    """Placeholder until the LangGraph graph is wired in."""
    yield "plan", {
        "thread_id": thread_id,
        "status": "awaiting_approval",
        "note": "Graph not wired yet; this is a deployment smoke-test stub.",
        "sub_questions": [],
    }


async def _aiter(events: Any) -> AsyncIterator[tuple[str, dict]]:
    """Accept an async iterator *or* a sync iterable (e.g. LangGraph ``graph.stream``).

    Sync iterables are pulled in a worker thread so a blocking graph never stalls the loop.
    """
    if hasattr(events, "__aiter__"):
        async for item in events:
            yield item
    elif isinstance(events, Iterable):
        async for item in iterate_in_threadpool(iter(events)):
            yield item
    else:
        raise TypeError(f"events source must be iterable, got {type(events).__name__}")


async def _run_stream(name: str, thread_id: str, inputs: dict, stage: str, slot: Slot,
                      events_factory: Callable[[], Any]) -> AsyncIterator[str]:
    """Shared SSE envelope: started → events → done, with error event and tracing."""
    started = time.monotonic()
    last_event: dict = {}
    try:
        with tracing.span(name, inputs=inputs, tags=API_TAGS,
                          metadata={"thread_id": thread_id, "stage": stage}) as run:
            yield _sse("started", {"thread_id": thread_id, **inputs.get("echo", {}),
                                   "trace_url": tracing.run_url(run)})
            try:
                async for event, data in _aiter(events_factory()):
                    last_event = {"event": event, **data}
                    yield _sse(event, data)
            except GuardrailViolation as exc:      # a rejected question, not a failure
                yield _sse("error", {"thread_id": thread_id, "code": "guardrail", "rule": exc.rule,
                                     "message": str(exc)})
                if run is not None:
                    run.end(error=f"rejected: {exc.rule}")
                return
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
        slot.release()
        await tracing.aflush()


# --------------------------------------------------------------------------- routes

@app.post("/api/research", dependencies=[Depends(guard)])
async def research(req: ResearchRequest) -> StreamingResponse:
    slot = run_slots.acquire()
    thread_id = str(uuid.uuid4())
    max_hops = req.max_hops if req.max_hops is not None else MAX_HOPS
    budget = effective_budget(req.settings, RUN_BUDGET_SECONDS)

    async def events():
        if not _graph_available():
            async for ev in _stub_events(req.question, thread_id):
                yield ev
            return
        from deep_researcher.graph import graph_for, stream_until_interrupt  # type: ignore[import-not-found]

        deps = _deps_for(req.settings, req.document_set_id)
        graph = await graph_for(deps) if deps else None      # None = the shared graph
        if req.settings and not req.settings.is_empty():
            yield "settings", req.settings.summary()
        async for ev in stream_until_interrupt(
            question=req.question, thread_id=thread_id, max_hops=max_hops, budget_seconds=budget,
            config=tracing.run_config(thread_id, req.question, tags=["plan"]), graph=graph,
            documents=documents,
        ):
            yield ev

    documents = _documents_for(req.document_set_id)
    echo = {"question": req.question, "max_hops": max_hops, "budget_seconds": budget,
            "document_set_id": req.document_set_id}
    inputs = {"question": req.question, "max_hops": max_hops, "echo": echo}
    return SlotStreamingResponse(_run_stream("api.research", thread_id, inputs, "plan", slot, events),
                                 media_type="text/event-stream", slot=slot)


def _upload_store() -> Any:
    from deep_researcher.store.uploads import get_upload_store

    return get_upload_store()


def _documents_for(set_id: str | None) -> dict[str, Any] | None:
    """Manifest dict for the graph state, or 404 when the set is unknown (expired, wrong instance)."""
    if not set_id:
        return None
    manifest = _upload_store().manifest(set_id)
    if not manifest:
        raise HTTPException(status_code=404, detail="Unknown document set (expired, or uploaded to another "
                                                    "instance); upload the files again.")
    return {"set_id": set_id, "documents": [d.model_dump() for d in manifest]}


def _deps_for(settings: RunSettings | None, set_id: str | None) -> Deps | None:
    """Per-run dependencies: model overrides and/or the documents tool bound to this run's set."""
    deps = make_deps(settings)
    if set_id:
        deps = docs_mod.bind_documents(deps or Deps(), set_id)
    return deps


_FILES = File(...)
_SET_ID_FORM = Form(default=None, pattern=SET_ID)


@app.post("/api/documents", dependencies=[Depends(guard)])
async def upload_documents(files: list[UploadFile] = _FILES,
                           document_set_id: str | None = _SET_ID_FORM) -> dict:
    """Attach files as a research source. Returns the set id to send as ``document_set_id`` with
    ``/api/research`` and ``/resume``, plus the manifest. Add to an existing set by passing its id."""
    cap = docs_mod.max_file_bytes()
    payload: list[tuple[str, bytes]] = []
    for f in files:
        data = await f.read(cap + 1)
        if len(data) > cap:
            raise HTTPException(status_code=413, detail=f"{f.filename}: larger than {cap // 1_000_000} MB")
        payload.append((f.filename or "upload", data))
    if not payload:
        raise HTTPException(status_code=422, detail="No files received.")
    try:
        set_id, manifest = await asyncio.to_thread(docs_mod.ingest, payload, set_id=document_set_id)
    except docs_mod.DocumentError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"document_set_id": set_id, "documents": [d.model_dump() for d in manifest]}


@app.get("/api/documents/{set_id}", dependencies=[Depends(guard)])
async def get_documents(set_id: str) -> dict:
    docs = _documents_for(set_id) or {}
    return {"document_set_id": set_id, "documents": docs.get("documents", [])}


@app.delete("/api/documents/{set_id}", dependencies=[Depends(guard)])
async def delete_documents(set_id: str) -> dict:
    return {"document_set_id": set_id, "deleted": await asyncio.to_thread(_upload_store().delete_set, set_id)}


_models_cache: dict[str, Any] = {"at": 0.0, "rows": []}


@app.get("/api/models", dependencies=[Depends(guard)])
async def models(prefix: str = "") -> dict:
    """OpenRouter's live model catalogue (id, context length, prices), cached for ten minutes, so the
    hosted UI can offer the same model pickers as the Gradio Settings tab."""
    if time.monotonic() - _models_cache["at"] > 600 or not _models_cache["rows"]:
        try:
            _models_cache["rows"] = await llm.alist_models()
            _models_cache["at"] = time.monotonic()
        except Exception as exc:  # upstream catalogue down: answer with what we have
            log.warning("model catalogue fetch failed: %s", exc)
            if not _models_cache["rows"]:
                raise HTTPException(status_code=502,
                                    detail="Could not fetch the OpenRouter model list.") from exc
    allowed = allowed_models()
    rows = [r for r in _models_cache["rows"] if r["id"].startswith(prefix) and is_allowed(r["id"], allowed)]
    return {"models": rows, "count": len(rows), "defaults": llm.info()["models"],
            "allowed_models": allowed or None}


@app.post("/api/report/pdf", dependencies=[Depends(guard)])
async def report_pdf(req: PdfRequest) -> Response:
    """Render a report's Markdown (as streamed in the `report` event) to PDF. Stateless, so it works
    on serverless hosts where nothing is kept between requests."""
    try:
        data = await asyncio.to_thread(markdown_to_pdf, req.markdown, title=req.title)
    except Exception as exc:
        log.warning("pdf render failed: %s", exc)
        raise HTTPException(status_code=500, detail="PDF rendering failed.") from exc
    name = (req.title or "research-report")[:60].replace('"', "")
    return Response(content=data, media_type="application/pdf",
                    headers={"Content-Disposition": f'attachment; filename="{name}.pdf"'})


@app.post("/api/research/{thread_id}/resume", dependencies=[Depends(guard)])
async def resume(thread_id: str, req: ResumeRequest) -> StreamingResponse:
    try:
        uuid.UUID(thread_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="thread_id must be a UUID.") from exc
    if not _graph_available():
        raise HTTPException(status_code=501, detail="Research graph not wired yet (see PLAN.md Phase 6).")
    slot = run_slots.acquire()
    from deep_researcher.graph import graph_for, resume_with_plan  # type: ignore[import-not-found]

    _documents_for(req.document_set_id)                      # 404 before the stream starts

    async def events():
        deps = _deps_for(req.settings, req.document_set_id)
        graph = await graph_for(deps) if deps else None      # None = the shared graph
        if req.settings and not req.settings.is_empty():
            yield "settings", req.settings.summary()
        budget = effective_budget(req.settings, RUN_BUDGET_SECONDS)
        cfg = tracing.run_config(thread_id, tags=["research"])
        async for ev in resume_with_plan(thread_id=thread_id, plan=req.plan.model_dump(),
                                         budget_seconds=budget, config=cfg, graph=graph):
            yield ev

    inputs = {"thread_id": thread_id, "plan": req.plan.model_dump()}
    return SlotStreamingResponse(_run_stream("api.resume", thread_id, inputs, "research", slot, events),
                                 media_type="text/event-stream", slot=slot)
