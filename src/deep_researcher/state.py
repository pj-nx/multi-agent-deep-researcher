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
    # analysis
    summaries: Annotated[dict[str, dict[str, Any]], merge_dicts]   # sub_question_id -> Summary dump
    contradictions: Annotated[list[dict[str, Any]], operator.add]
    gaps: Annotated[list[dict[str, Any]], operator.add]
    hop: int
    # synthesis
    insights: list[dict[str, Any]]
    critique: dict[str, Any] | None
    revisions: int
    report: str | None
    # bookkeeping
    notes: Annotated[list[str], operator.add]                      # human-readable progress lines


class RetrieveInput(TypedDict):
    """Payload of each ``Send("retrieve", ...)`` branch."""

    sub_question: dict[str, Any]    # SubQuestion.model_dump()
    hop: int
    deadline: float
