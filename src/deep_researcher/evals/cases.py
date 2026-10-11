"""Seeded evaluation cases: ``evals/questions.jsonl`` at the repository root (or ``EVALS_QUESTIONS``)."""

from __future__ import annotations

import json
import os
from pathlib import Path

from pydantic import BaseModel, Field


class EvalCase(BaseModel):
    """One seeded question with what a good report must cover."""

    id: str = Field(min_length=1, max_length=64, pattern=r"^[a-z0-9][a-z0-9-]*$")
    question: str = Field(min_length=5, max_length=2000)
    expected_facets: list[str] = Field(
        min_length=1, max_length=12, description="aspects a complete answer must address"
    )
    expect_contradiction: bool = False
    contradiction_hint: str | None = Field(
        default=None, max_length=1000, description="the disagreement the pipeline should surface"
    )
    tags: list[str] = Field(default_factory=list, max_length=8)
    max_hops: int | None = Field(default=None, ge=0, le=5, description="per-case override of the hop limit")
    documents: list[str] = Field(default_factory=list, max_length=10,
                                 description="files to attach (paths relative to the questions file)")

    def document_paths(self, base: Path) -> list[Path]:
        return [(base / d).resolve() for d in self.documents]


def default_path() -> Path:
    env = os.getenv("EVALS_QUESTIONS")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[3] / "evals" / "questions.jsonl"


def load_cases(
    path: Path | str | None = None, *, ids: list[str] | None = None, limit: int | None = None
) -> list[EvalCase]:
    """Read the JSONL file; blank lines and ``#`` comments are skipped. Ids must be unique."""
    p = Path(path) if path else default_path()
    cases: list[EvalCase] = []
    for n, line in enumerate(p.read_text().splitlines(), start=1):
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        try:
            case = EvalCase.model_validate(json.loads(s))
        except Exception as exc:
            raise ValueError(f"{p}:{n}: {exc}") from exc
        missing = [d for d in case.document_paths(p.parent) if not d.exists()]
        if missing:
            raise ValueError(f"{p}:{n}: attached document(s) not found: {[str(m) for m in missing]}")
        cases.append(case)
    seen = [c.id for c in cases]
    dupes = sorted({i for i in seen if seen.count(i) > 1})
    if dupes:
        raise ValueError(f"{p}: duplicate case ids {dupes}")
    if ids:
        wanted = set(ids)
        missing = wanted - set(seen)
        if missing:
            raise ValueError(f"unknown case ids: {sorted(missing)}")
        cases = [c for c in cases if c.id in wanted]
    if limit is not None:
        cases = cases[:limit]
    return cases
