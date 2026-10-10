"""FastAPI entrypoint for Vercel.

Vercel routes /api/* here (see vercel.json). Locally: `uvicorn api.index:app --reload`
or `vercel dev`. The research graph (src/deep_researcher/graph.py) is wired in during
Phase 6 of PLAN.md; until then /api/research streams a stub so the deployment path can
be verified end to end.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

app = FastAPI(title="Deep Researcher API", version="0.1.0")

MAX_HOPS = int(os.getenv("MAX_HOPS", "1"))
RUN_BUDGET_SECONDS = int(os.getenv("RUN_BUDGET_SECONDS", "240"))


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
    }


async def _stub_stream(question: str, thread_id: str) -> AsyncIterator[str]:
    """Placeholder stream until the LangGraph graph is wired in."""
    yield _sse("started", {"thread_id": thread_id, "question": question})
    yield _sse(
        "plan",
        {
            "thread_id": thread_id,
            "status": "awaiting_approval",
            "note": "Graph not wired yet; this is a deployment smoke-test stub.",
            "sub_questions": [],
        },
    )
    yield _sse("done", {"thread_id": thread_id, "elapsed_s": 0})


@app.post("/api/research")
async def research(req: ResearchRequest) -> StreamingResponse:
    thread_id = str(uuid.uuid4())
    started = time.monotonic()

    if not _graph_available():
        return StreamingResponse(_stub_stream(req.question, thread_id), media_type="text/event-stream")

    from deep_researcher.graph import stream_until_interrupt  # type: ignore

    async def gen() -> AsyncIterator[str]:
        yield _sse("started", {"thread_id": thread_id, "question": req.question})
        async for event, data in stream_until_interrupt(
            question=req.question,
            thread_id=thread_id,
            max_hops=req.max_hops if req.max_hops is not None else MAX_HOPS,
            budget_seconds=RUN_BUDGET_SECONDS,
        ):
            yield _sse(event, data)
        yield _sse("done", {"thread_id": thread_id, "elapsed_s": round(time.monotonic() - started, 1)})

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.post("/api/research/{thread_id}/resume")
async def resume(thread_id: str, req: ResumeRequest) -> StreamingResponse:
    if not _graph_available():
        raise HTTPException(status_code=501, detail="Research graph not wired yet (see PLAN.md Phase 6).")

    from deep_researcher.graph import resume_with_plan  # type: ignore

    async def gen() -> AsyncIterator[str]:
        async for event, data in resume_with_plan(thread_id=thread_id, plan=req.plan, budget_seconds=RUN_BUDGET_SECONDS):
            yield _sse(event, data)
        yield _sse("done", {"thread_id": thread_id})

    return StreamingResponse(gen(), media_type="text/event-stream")
