"""Pydantic models shared by the API and the graph.

Only the plan-side models exist so far; Evidence, Summary, Contradiction, Insight and Critique
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


class ResearchPlan(BaseModel):
    question: str = Field(min_length=5, max_length=2000)
    sub_questions: list[SubQuestion] = Field(min_length=1, max_length=12)
    notes: str | None = Field(default=None, max_length=2000)

    @property
    def open_questions(self) -> list[SubQuestion]:
        return list(self.sub_questions)
