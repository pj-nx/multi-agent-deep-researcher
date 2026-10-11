"""Run health: one place where "something degraded but the run carried on" is recorded.

Agents that fall back (a rate-limited source, a model error, a missing key) call :func:`degrade`.
It logs in a fixed format, emits a ``degraded`` activity event for the UI / API stream, and returns a
record the node puts in its state update (``degradations``, appended across the run and checkpointed).
The report turns the records into a "Run health" section via :func:`summarise` / :func:`health_md`.

Severity: ``critical`` = a stage's output is missing or unreviewed (critic, validator, insights, report
narrative); ``warn`` = partial loss (one source failed, some sources unvalidated); ``info`` = FYI.

Tools run outside the graph state, so they report through :data:`tool_errors`: the retriever sets a
list for the duration of one tool call and :meth:`Tool.search` appends to it instead of swallowing.
"""
from __future__ import annotations

import contextvars
import logging
from typing import Any

log = logging.getLogger(__name__)

SEVERITIES = ("info", "warn", "critical")
_LEVEL = {"info": logging.INFO, "warn": logging.WARNING, "critical": logging.ERROR}

tool_errors: contextvars.ContextVar[list[dict[str, Any]] | None] = contextvars.ContextVar(
    "tool_errors", default=None)


def record_tool_error(tool: str, kind: str, detail: str) -> None:
    """Called by tools when a search fails; a no-op unless the retriever is collecting."""
    sink = tool_errors.get()
    if sink is not None:
        sink.append({"tool": tool, "kind": kind, "detail": detail[:200]})


def _thread_id() -> str | None:
    try:
        from langgraph.config import get_config

        return (get_config().get("configurable") or {}).get("thread_id")
    except Exception:  # noqa: BLE001 - outside a graph run (unit tests, scripts)
        return None


def degrade(component: str, kind: str, detail: str, severity: str = "warn", *, count: int = 1,
            **extra: Any) -> dict[str, Any]:
    """Log + stream a degradation and return its record (add it to the node's ``degradations``)."""
    if severity not in SEVERITIES:
        raise ValueError(f"severity {severity!r}; expected one of {SEVERITIES}")
    record = {"component": component, "kind": kind, "detail": detail[:300], "severity": severity,
              "count": count, **extra}
    thread = _thread_id()
    log.log(_LEVEL[severity], "degraded [%s] %s/%s x%d: %s%s", severity, component, kind, count,
            record["detail"], f" (thread {thread})" if thread else "")
    from deep_researcher.agents.common import emit

    emit("degraded", component=component, error_kind=kind, severity=severity, count=count,
         detail=record["detail"], **{k: v for k, v in extra.items() if isinstance(v, str | int | float)})
    return record


def summarise(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group by (component, kind): total count, worst severity, an example detail, affected sub-questions.

    Sorted worst first, then by count.
    """
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    for r in records:
        g = groups.setdefault((r["component"], r["kind"]), {
            "component": r["component"], "kind": r["kind"], "severity": "info", "count": 0,
            "detail": r.get("detail", ""), "sub_questions": []})
        g["count"] += int(r.get("count", 1))
        if SEVERITIES.index(r["severity"]) > SEVERITIES.index(g["severity"]):
            g["severity"] = r["severity"]
        if sq := r.get("sub_question_id"):
            if sq not in g["sub_questions"]:
                g["sub_questions"].append(sq)
    return sorted(groups.values(), key=lambda g: (-SEVERITIES.index(g["severity"]), -g["count"]))


def health_md(summary: list[dict[str, Any]]) -> list[str]:
    """Markdown lines for the report's "Run health" section; empty when the run was clean."""
    if not summary:
        return []
    lines = ["## Run health", "", "_Parts of this run did not work as intended; the findings below "
             "may be weaker than they look._", ""]
    for g in summary:
        where = f" (sub-questions: {', '.join(g['sub_questions'])})" if g["sub_questions"] else ""
        times = f" x{g['count']}" if g["count"] > 1 else ""
        lines.append(f"- **{g['severity']}** · `{g['component']}` {g['kind'].replace('_', ' ')}{times}"
                     f"{where}: {g['detail']}")
    return [*lines, ""]
