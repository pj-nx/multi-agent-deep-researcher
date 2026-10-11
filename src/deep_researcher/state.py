# Copyright (c) 2026 C10 Hackathon G1
"""LangGraph state for the research graph.

Everything is stored as plain JSON-serialisable dicts (Pydantic models are dumped on write and
re-validated on read) so any checkpointer can persist it. Reducers make parallel ``Send`` branches
safe: lists append, dicts merge, scalars take the latest write.
"""
from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


def merge_dicts(left: dict[str, Any] | None, right: dict[str, Any] | None) -> dict[str, Any]:
    """Shallow merge; the newer write wins per key. Used for per-sub-question maps."""
    out = dict(left or {})
    out.update(right or {})
    return out


def merge_assessments(left: dict[str, Any] | None, right: dict[str, Any] | None) -> dict[str, Any]:
    """Merge validator verdicts per evidence id.

    The same source is often retrieved by several sub-questions (and again by gap follow-ups), and each
    retrieve branch grades it against its own question. A source that is relevant to *any* sub-question
    must stay in the bibliography, otherwise claims that cited it legitimately end up pointing at a
    dropped source. So on conflict ``keep`` is OR-ed, scores take the maximum and issues are unioned.
    """
    out = dict(left or {})
    for eid, new in (right or {}).items():
        old = out.get(eid)
        if not old:
            out[eid] = new
            continue
        merged = {**old, **new}
        merged["keep"] = bool(old.get("keep", True) or new.get("keep", True))
        for key in ("relevance", "credibility"):
            vals = [v for v in (old.get(key), new.get(key)) if isinstance(v, int | float)]
            if vals:
                merged[key] = max(vals)
        merged["issues"] = list(dict.fromkeys([*(old.get("issues") or []), *(new.get("issues") or [])]))
        out[eid] = merged
    return out


class ResearchState(TypedDict, total=False):
    # inputs
    question: str
    max_hops: int
    deadline: float                 # unix time after which retrieval loops stop (0 = none)
    # planning
    plan: dict[str, Any] | None     # ResearchPlan.model_dump()
    approved: bool
    # retrieval (fan-in)
    evidence: Annotated[list[dict[str, Any]], operator.add]        # Evidence.model_dump() per source
    retrieval_log: Annotated[list[dict[str, Any]], operator.add]   # one row per (sub-question, hop)
    # validation
    assessments: Annotated[dict[str, dict[str, Any]], merge_assessments]  # evidence_id -> verdict; keep ORs
    validation: dict[str, int]                                      # hop summary from the validate step
    # analysis
    extra_sub_questions: Annotated[list[dict[str, Any]], operator.add]  # gap-derived SubQuestion dumps
    summaries: Annotated[dict[str, dict[str, Any]], merge_dicts]   # sub_question_id -> Summary dump
    analysis_inputs: Annotated[dict[str, str], merge_dicts]        # sub_question_id -> prompt hash
    contradictions: Annotated[list[dict[str, Any]], operator.add]
    gaps: Annotated[list[dict[str, Any]], operator.add]            # history of every gap raised
    pending_gaps: list[dict[str, Any]]                             # gaps raised in the latest round
    gap_source: str                                                # "analysis" | "critic": who raised them
    next_action: str                                               # "retrieve" | "insight" | "report"
    hop: int
    # synthesis
    insights: list[dict[str, Any]]
    critique: dict[str, Any] | None
    critiques: Annotated[list[dict[str, Any]], operator.add]      # every verdict (methodology appendix)
    revisions: int                                                 # critic rounds completed
    report: str | None
    report_json: dict[str, Any] | None
    # bookkeeping
    degradations: Annotated[list[dict[str, Any]], operator.add]   # health.degrade() records, whole run
    notes: Annotated[list[str], operator.add]                      # human-readable progress lines


class RetrieveInput(TypedDict):
    """Payload of each ``Send("retrieve", ...)`` branch."""

    sub_question: dict[str, Any]    # SubQuestion.model_dump()
    hop: int
    deadline: float
