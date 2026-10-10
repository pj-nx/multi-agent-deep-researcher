"""Helpers shared by the synthesis agents (insight, critic, report)."""

from __future__ import annotations

import re
from typing import Any

CITATION = re.compile(r"\[(E-[0-9a-f]{10})\]")


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
