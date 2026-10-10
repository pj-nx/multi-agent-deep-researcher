"""Helpers shared by the agents: activity events, structured model calls, prompt rendering."""

from __future__ import annotations

import logging
import re
import time
from typing import Any

log = logging.getLogger(__name__)

CITATION = re.compile(r"\[(E-[0-9a-f]{10})\]")


def emit(kind: str, **data: Any) -> None:
    """Publish a fine-grained activity event to the graph's custom stream (no-op outside a graph run)."""
    try:
        from langgraph.config import get_stream_writer

        writer = get_stream_writer()
    except Exception:  # noqa: BLE001 - not inside a LangGraph run (unit tests, scripts)
        return
    if writer is not None:
        writer({"kind": kind, "t": time.time(), **data})


async def call_structured(llm: Any, schema: Any, messages: list[Any], *, role: str, label: str,
                          method: str = "json_schema") -> Any:
    """``with_structured_output(...).ainvoke`` wrapped in llm_start / llm_end / llm_error events.

    Uses ``include_raw`` so token usage can be reported; tolerates fakes that return the parsed
    object directly.
    """
    model = getattr(llm, "model_name", None) or getattr(llm, "model", None)
    emit("llm_start", role=role, label=label, model=model, schema=schema.__name__)
    started = time.monotonic()
    runnable = llm.with_structured_output(schema, method=method, include_raw=True)
    try:
        out = await runnable.ainvoke(messages)
    except Exception as exc:
        emit("llm_error", role=role, label=label, model=model, elapsed_s=round(time.monotonic() - started, 1),
             error=f"{type(exc).__name__}: {str(exc)[:160]}")
        raise
    if isinstance(out, dict) and ("parsed" in out or "raw" in out):
        if out.get("parsing_error"):
            emit("llm_error", role=role, label=label, model=model,
                 elapsed_s=round(time.monotonic() - started, 1),
                 error=f"parse: {str(out['parsing_error'])[:160]}")
            raise ValueError(f"structured output could not be parsed: {out['parsing_error']}")
        parsed, raw = out.get("parsed"), out.get("raw")
    else:
        parsed, raw = out, None
    usage = getattr(raw, "usage_metadata", None) or {}
    emit("llm_end", role=role, label=label, model=model, elapsed_s=round(time.monotonic() - started, 1),
         tokens=usage.get("total_tokens"), input_tokens=usage.get("input_tokens"),
         output_tokens=usage.get("output_tokens"))
    return parsed


def kept_evidence(state: dict[str, Any]) -> list[dict[str, Any]]:
    """Evidence that survived validation (everything, when nothing was validated)."""
    assessments = state.get("assessments") or {}
    return [e for e in state.get("evidence", []) if (assessments.get(e["id"]) or {}).get("keep", True)]


def bibliography(state: dict[str, Any], max_items: int = 80) -> tuple[str, set[str]]:
    """Compact source list for prompts: ``[E-id] (type, date, cred) title`` and the set of allowed ids."""
    assessments = state.get("assessments") or {}
    rows = []
    for e in kept_evidence(state)[:max_items]:
        cred = (assessments.get(e["id"]) or {}).get("credibility", e.get("credibility")) or 0.5
        rows.append(
            f"[{e['id']}] ({e['source_type']}, {e.get('published_at') or 'n.d.'}, cred={cred:.2f}) "
            f"{(e.get('title') or e['url'])[:120]}"
        )
    return "\n".join(rows), {e["id"] for e in kept_evidence(state)}


def summaries_text(state: dict[str, Any], max_chars: int = 2500) -> str:
    parts = []
    questions = {sq["id"]: sq["question"] for sq in state["plan"]["sub_questions"]}
    questions.update({sq["id"]: sq["question"] for sq in state.get("extra_sub_questions", [])})
    for sq_id, summ in (state.get("summaries") or {}).items():
        claims = "\n".join(
            f"  - {c['text']} {' '.join(f'[{i}]' for i in c['evidence_ids'])} "
            f"(confidence {c['confidence']:.2f})"
            for c in summ.get("claims", [])
        )
        parts.append(
            f"### {sq_id}: {questions.get(sq_id, '')}\n{summ.get('summary', '')[:max_chars]}\n{claims}"
        )
    return "\n\n".join(parts)


def contradictions_text(state: dict[str, Any]) -> str:
    rows = []
    for c in state.get("contradictions", []):
        a = " ".join(f"[{i}]" for i in c["evidence_ids_a"])
        b = " ".join(f"[{i}]" for i in c["evidence_ids_b"])
        rows.append(
            f"- ({c['sub_question_id']}) A: {c['claim_a']} {a} | B: {c['claim_b']} {b} "
            f"| why: {c.get('explanation', '')}"
        )
    return "\n".join(rows) or "(none found)"


def strip_unknown_citations(text: str, allowed: set[str]) -> tuple[str, int]:
    """Remove ``[E-id]`` markers that do not refer to kept evidence; returns (text, removed_count)."""
    removed = 0

    def repl(m: re.Match[str]) -> str:
        nonlocal removed
        if m.group(1) in allowed:
            return m.group(0)
        removed += 1
        return ""

    out = CITATION.sub(repl, text)
    return re.sub(r"[ \t]{2,}", " ", out).strip(), removed


def cited_ids(text: str) -> list[str]:
    return list(dict.fromkeys(CITATION.findall(text)))
