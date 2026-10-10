"""LangSmith tracing support.

Everything here degrades to a no-op when tracing is off or the `langsmith` package is
missing, so the API and graph run identically with or without observability.

Configuration is via the standard LangSmith environment variables:

    LANGSMITH_TRACING=true
    LANGSMITH_API_KEY=lsv2_...
    LANGSMITH_PROJECT=deep-researcher        # optional, defaults to "default"
    LANGSMITH_ENDPOINT=https://api.smith.langchain.com   # optional; EU: https://eu.api.smith.langchain.com

Usage:

    from deep_researcher import tracing

    tracing.configure()                      # once at startup
    with tracing.span("api.research", inputs={...}, tags=[...], metadata={...}) as run:
        url = tracing.run_url(run)           # share with the caller
        graph.stream(state, config=tracing.run_config(thread_id, question))
    tracing.flush()                          # before a serverless function exits
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import os
from collections.abc import Callable, Iterator
from typing import Any, TypeVar

log = logging.getLogger(__name__)

_TRUTHY = {"1", "true", "yes", "on"}
_configured = False
_client: Any = None

F = TypeVar("F", bound=Callable[..., Any])


def _langsmith():
    try:
        import langsmith

        return langsmith
    except ImportError:  # pragma: no cover - exercised only when the dep is absent
        return None


def enabled() -> bool:
    """True when tracing is switched on and an API key is present."""
    flag = os.getenv("LANGSMITH_TRACING", os.getenv("LANGCHAIN_TRACING_V2", "")).strip().lower()
    return flag in _TRUTHY and bool(os.getenv("LANGSMITH_API_KEY")) and _langsmith() is not None


def project() -> str:
    return os.getenv("LANGSMITH_PROJECT", os.getenv("LANGCHAIN_PROJECT", "default"))


def configure() -> bool:
    """Prepare the process for tracing. Idempotent. Returns whether tracing is enabled.

    In serverless runtimes (Vercel sets ``VERCEL=1``) LangChain callbacks are forced to run
    synchronously so traces are not lost when the function instance is frozen.
    """
    global _configured
    if _configured:
        return enabled()
    _configured = True
    if os.getenv("VERCEL") or os.getenv("AWS_LAMBDA_FUNCTION_NAME"):
        os.environ.setdefault("LANGCHAIN_CALLBACKS_BACKGROUND", "false")
    if enabled():
        log.info("LangSmith tracing enabled; project=%s", project())
    return enabled()


def client() -> Any | None:
    """Shared LangSmith client, or None when tracing is off."""
    global _client
    if not enabled():
        return None
    if _client is None:
        _client = _langsmith().Client()
    return _client


def info() -> dict[str, Any]:
    """Status payload for health endpoints (never includes the key)."""
    return {
        "enabled": enabled(),
        "project": project() if enabled() else None,
        "endpoint": os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com") if enabled() else None,
    }


def traced(name: str | None = None, *, run_type: str = "chain", tags: list[str] | None = None,
           metadata: dict[str, Any] | None = None) -> Callable[[F], F]:
    """``@traceable`` that becomes a no-op decorator when tracing is off.

    Use ``run_type="retriever"`` for retrieval tools, ``"tool"`` for other tools, ``"llm"``
    for raw model calls and ``"chain"`` (default) for agents and nodes.
    """
    ls = _langsmith()
    if ls is None:
        return lambda fn: fn
    return ls.traceable(name=name, run_type=run_type, tags=tags or [], metadata=metadata or {})


@contextlib.contextmanager
def span(name: str, *, run_type: str = "chain", inputs: dict[str, Any] | None = None,
         tags: list[str] | None = None, metadata: dict[str, Any] | None = None) -> Iterator[Any]:
    """Open a parent run; every LangChain/LangGraph call inside nests under it.

    Yields the ``RunTree`` (or ``None`` when tracing is off). Set outputs with
    ``run.end(outputs={...})`` or let the context manager close it. Exceptions are recorded.
    """
    if not enabled():
        yield None
        return
    ls = _langsmith()
    with ls.trace(name=name, run_type=run_type, inputs=inputs or {}, tags=tags or [],
                  metadata=metadata or {}, project_name=project(), client=client()) as run:
        yield run


def run_url(run: Any) -> str | None:
    """Best-effort shareable URL for a run; None when unavailable."""
    if run is None:
        return None
    try:
        return run.get_url()
    except Exception as exc:  # noqa: BLE001 - observability must never fail a request
        log.debug("could not resolve trace URL: %s", exc)
        return None


def run_config(thread_id: str, question: str | None = None, *, run_name: str = "deep_researcher",
               tags: list[str] | None = None, metadata: dict[str, Any] | None = None,
               **configurable: Any) -> dict[str, Any]:
    """Build the ``config`` for ``graph.invoke/stream`` with LangSmith naming attached.

    ``thread_id`` goes into ``configurable`` for the checkpointer and into metadata so runs
    can be filtered by conversation in LangSmith.
    """
    meta = {"thread_id": thread_id, **(metadata or {})}
    if question is not None:
        meta["question"] = question[:500]
    return {
        "run_name": run_name,
        "tags": ["deep-researcher", *(tags or [])],
        "metadata": meta,
        "configurable": {"thread_id": thread_id, **configurable},
    }


def flush(timeout: float | None = 10.0) -> None:
    """Block until queued traces are uploaded, or ``timeout`` seconds pass.

    Call before a serverless function returns. Never raises: losing a trace is acceptable,
    failing the user's request because of telemetry is not.
    """
    if not enabled():
        return
    try:
        c = client()
        if c is not None:
            c.flush(timeout=timeout)
    except Exception as exc:  # noqa: BLE001 - see docstring
        log.warning("LangSmith flush failed: %s", exc)
    try:
        from langchain_core.tracers.langchain import wait_for_all_tracers

        wait_for_all_tracers()
    except ImportError:
        pass
    except Exception as exc:  # noqa: BLE001 - see docstring
        log.warning("wait_for_all_tracers failed: %s", exc)


async def aflush(timeout: float | None = 10.0) -> None:
    """Async flush bounded by ``timeout``; gives up (and logs) rather than hanging the response."""
    if not enabled():
        return
    try:
        outer = None if timeout is None else timeout + 1
        await asyncio.wait_for(asyncio.to_thread(flush, timeout), timeout=outer)
    except TimeoutError:
        log.warning("LangSmith flush timed out after %ss; traces may be incomplete", timeout)
