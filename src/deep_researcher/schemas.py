"""Pydantic models shared by the API and the graph.

Plan, evidence and the analysis/insight/critique records. Later phases fill them in; the shapes
are fixed here so the graph state and API contract do not churn (see PLAN.md §3). Bounds are
deliberate: these objects cross the HTTP boundary, so they cap payload size and shape before
anything reaches a graph node.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

SourceType = Literal["papers", "news", "web", "reports", "api", "wikipedia"]


class SubQuestion(BaseModel):
    id: str = Field(min_length=1, max_length=32, pattern=r"^[A-Za-z0-9_.-]+$")
    question: str = Field(min_length=5, max_length=1000)
    source_types: list[SourceType] = Field(default_factory=lambda: ["web"], max_length=6)
    search_queries: list[str] = Field(default_factory=list, max_length=5,
                                      description="2-3 short keyword queries (<= 8 words) for search APIs")
    depends_on: list[str] = Field(default_factory=list, max_length=10)
    rationale: str | None = Field(default=None, max_length=1000)


class Evidence(BaseModel):
    """One retrieved source (a paper, article, page or API response) before chunking."""

    id: str = Field(min_length=1, max_length=64)
    sub_question_id: str = Field(min_length=1, max_length=32)
    source_type: SourceType = "web"
    title: str = Field(default="", max_length=500)
    url: str = Field(min_length=1, max_length=2000)
    text: str = Field(min_length=1)
    published_at: str | None = Field(default=None, max_length=40)   # ISO-8601 date if known
    credibility: float | None = Field(default=None, ge=0.0, le=1.0)


class EvidenceChunk(BaseModel):
    """The unit stored in the vector DB; metadata is flat so it round-trips through Chroma."""

    id: str = Field(min_length=1, max_length=64)
    evidence_id: str = Field(min_length=1, max_length=64)
    chunk_index: int = Field(ge=0)
    text: str = Field(min_length=1)
    sub_question_id: str = Field(min_length=1, max_length=32)
    source_type: SourceType = "web"
    title: str = ""
    url: str = Field(min_length=1, max_length=2000)
    published_at: str | None = None
    credibility: float | None = Field(default=None, ge=0.0, le=1.0)


class ResearchPlan(BaseModel):
    question: str = Field(min_length=5, max_length=2000)
    sub_questions: list[SubQuestion] = Field(min_length=1, max_length=12)
    notes: str | None = Field(default=None, max_length=2000)

    @property
    def open_questions(self) -> list[SubQuestion]:
        return list(self.sub_questions)


class Claim(BaseModel):
    text: str = Field(min_length=1, max_length=1000)
    evidence_ids: list[str] = Field(min_length=1, max_length=20)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class Summary(BaseModel):
    """Critical-analysis output for one sub-question."""

    sub_question_id: str = Field(min_length=1, max_length=32)
    summary: str = Field(min_length=1, max_length=6000)
    claims: list[Claim] = Field(default_factory=list, max_length=40)
    evidence_ids: list[str] = Field(default_factory=list, max_length=200)


class Contradiction(BaseModel):
    sub_question_id: str = Field(min_length=1, max_length=32)
    claim_a: str = Field(min_length=1, max_length=1000)
    claim_b: str = Field(min_length=1, max_length=1000)
    evidence_ids_a: list[str] = Field(min_length=1, max_length=20)
    evidence_ids_b: list[str] = Field(min_length=1, max_length=20)
    explanation: str = Field(default="", max_length=2000)


class Gap(BaseModel):
    """A follow-up question the analysis could not answer from current evidence (drives multi-hop)."""

    parent_sub_question_id: str = Field(min_length=1, max_length=32)
    question: str = Field(min_length=5, max_length=1000)
    source_types: list[SourceType] = Field(default_factory=lambda: ["web"], max_length=6)
    search_queries: list[str] = Field(default_factory=list, max_length=3)
    reason: str = Field(default="", max_length=1000)


class ReasoningStep(BaseModel):
    step: str = Field(min_length=1, max_length=1000)
    evidence_ids: list[str] = Field(default_factory=list, max_length=20)


class Insight(BaseModel):
    hypothesis: str = Field(min_length=1, max_length=1500)
    reasoning: list[ReasoningStep] = Field(min_length=1, max_length=12)
    confidence: float = Field(ge=0.0, le=1.0)
    falsifier: str = Field(default="", max_length=1000)   # what evidence would refute it


class Critique(BaseModel):
    verdict: Literal["accept", "revise", "need_more_evidence"]
    issues: list[str] = Field(default_factory=list, max_length=20)
    follow_up: list[Gap] = Field(default_factory=list, max_length=6)


class SourceAssessment(BaseModel):
    """Source Validator verdict for one piece of evidence."""

    evidence_id: str = Field(min_length=1, max_length=64)
    relevance: float = Field(ge=0.0, le=1.0, description="how directly it bears on the sub-question")
    credibility: float = Field(ge=0.0, le=1.0, description="trustworthiness of the source itself")
    source_kind: Literal["primary", "secondary", "tertiary", "opinion", "unknown"] = "unknown"
    peer_reviewed: bool | None = None
    issues: list[str] = Field(default_factory=list, max_length=6)
    keep: bool = True


class SourceAssessments(BaseModel):
    items: list[SourceAssessment] = Field(default_factory=list, max_length=40)


class ContradictionOut(BaseModel):
    """Contradiction as the analysis model emits it (sub_question_id is filled in by the node)."""

    claim_a: str = Field(min_length=1, max_length=1000)
    claim_b: str = Field(min_length=1, max_length=1000)
    evidence_ids_a: list[str] = Field(min_length=1, max_length=20)
    evidence_ids_b: list[str] = Field(min_length=1, max_length=20)
    explanation: str = Field(default="", max_length=2000)


class GapOut(BaseModel):
    question: str = Field(min_length=5, max_length=1000)
    source_types: list[SourceType] = Field(default_factory=lambda: ["web"], max_length=6)
    search_queries: list[str] = Field(default_factory=list, max_length=3)
    reason: str = Field(default="", max_length=1000)


class AnalysisOutput(BaseModel):
    """Critical Analysis result for one sub-question."""

    summary: str = Field(min_length=1, max_length=6000)
    claims: list[Claim] = Field(default_factory=list, max_length=40)
    contradictions: list[ContradictionOut] = Field(default_factory=list, max_length=10)
    gaps: list[GapOut] = Field(default_factory=list, max_length=3)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="how well the evidence answers it")
