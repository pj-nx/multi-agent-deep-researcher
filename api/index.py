"""FastAPI entrypoint for Vercel.

Vercel routes /api/* here (see vercel.json). Locally: `uvicorn api.index:app --reload`
or `vercel dev`. The research graph (src/deep_researcher/graph.py) is wired in during
Phase 6 of PLAN.md; until then /api/research streams a stub so the deployment path can
be verified end to end.

Every request is a LangSmith parent run (when LANGSMITH_TRACING=true); graph runs nest
under it and the trace URL is sent to the client in the first SSE event.
"""
from __future__ import annotations

import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import AsyncIterator

# Vercel installs requirements.txt but not the local package; make `src/` importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import StreamingResponse  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from deep_researcher import llm, tracing  # noqa: E402

tracing.configure()

app = FastAPI(title="Deep Researcher API", version="0.1.0")

MAX_HOPS = int(os.getenv("MAX_HOPS", "1"))
RUN_BUDGET_SECONDS = int(os.getenv("RUN_BUDGET_SECONDS", "240"))
API_TAGS = ["api", "vercel" if os.getenv("VERCEL") else "local"]


class ResearchRequest(BaseModel):
    question: str = Field(min_length=5, max_length=2000)
    max_hops: int | None = Field(default=None, ge=0, le=5)


class ResumeRequest(BaseModel):
    plan: dict


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _graph_available() -> bool:
    try:
        import deep_researcher.graph  # noqa: F401
        return True
    except Exception:
        return False


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "graph_wired": _graph_available(),
        "max_hops": MAX_HOPS,
        "run_budget_seconds": RUN_BUDGET_SECONDS,
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


@app.post("/api/research")
async def research(req: ResearchRequest) -> StreamingResponse:
    thread_id = str(uuid.uuid4())
    max_hops = req.max_hops if req.max_hops is not None else MAX_HOPS

    async def gen() -> AsyncIterator[str]:
        started = time.monotonic()
        last_event: dict = {}
        try:
            with tracing.span(
                "api.research",
                inputs={"question": req.question, "max_hops": max_hops},
                tags=API_TAGS,
                metadata={"thread_id": thread_id, "stage": "plan"},
            ) as run:
                yield _sse("started", {
                    "thread_id": thread_id,
                    "question": req.question,
                    "trace_url": tracing.run_url(run),
                })
                if _graph_available():
                    from deep_researcher.graph import stream_until_interrupt  # type: ignore

                    events = stream_until_interrupt(
                        question=req.question,
                        thread_id=thread_id,
                        max_hops=max_hops,
                        budget_seconds=RUN_BUDGET_SECONDS,
                        config=tracing.run_config(thread_id, req.question, tags=["plan"]),
                    )
                else:
                    events = _stub_events(req.question, thread_id)
                async for event, data in events:
                    last_event = {"event": event, **data}
                    yield _sse(event, data)
                elapsed = round(time.monotonic() - started, 1)
                if run is not None:
                    run.end(outputs={"last_event": last_event, "elapsed_s": elapsed})
                yield _sse("done", {"thread_id": thread_id, "elapsed_s": elapsed})
        finally:
            await tracing.aflush()

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.post("/api/research/{thread_id}/resume")
async def resume(thread_id: str, req: ResumeRequest) -> StreamingResponse:
    if not _graph_available():
        raise HTTPException(status_code=501, detail="Research graph not wired yet (see PLAN.md Phase 6).")

    from deep_researcher.graph import resume_with_plan  # type: ignore

    async def gen() -> AsyncIterator[str]:
        started = time.monotonic()
        try:
            with tracing.span(
                "api.resume",
                inputs={"thread_id": thread_id, "plan": req.plan},
                tags=API_TAGS,
                metadata={"thread_id": thread_id, "stage": "research"},
            ) as run:
                yield _sse("started", {"thread_id": thread_id, "trace_url": tracing.run_url(run)})
                async for event, data in resume_with_plan(
                    thread_id=thread_id,
                    plan=req.plan,
                    budget_seconds=RUN_BUDGET_SECONDS,
                    config=tracing.run_config(thread_id, tags=["research"]),
                ):
                    yield _sse(event, data)
                elapsed = round(time.monotonic() - started, 1)
                if run is not None:
                    run.end(outputs={"elapsed_s": elapsed})
                yield _sse("done", {"thread_id": thread_id, "elapsed_s": elapsed})
        finally:
            await tracing.aflush()

    return StreamingResponse(gen(), media_type="text/event-stream")
