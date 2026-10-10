"""Pydantic models shared by the API and the graph.

Plan and evidence models exist so far; Summary, Contradiction, Insight and Critique
arrive with their phases (see PLAN.md §3). Bounds are deliberate: these objects cross the HTTP
boundary, so they cap payload size and shape before anything reaches a graph node.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

SourceType = Literal["papers", "news", "web", "reports", "api", "wikipedia"]


class SubQuestion(BaseModel):
    id: str = Field(min_length=1, max_length=32, pattern=r"^[A-Za-z0-9_.-]+$")
    question: str = Field(min_length=5, max_length=1000)
    source_types: list[SourceType] = Field(default_factory=lambda: ["web"], max_length=6)
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
