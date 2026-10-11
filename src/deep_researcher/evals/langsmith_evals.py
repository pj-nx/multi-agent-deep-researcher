# Copyright (c) 2026 C10 Hackathon G1
"""Optional LangSmith integration: the seeded cases as a dataset, the judges as evaluators.

Everything here needs ``LANGSMITH_TRACING=true`` and ``LANGSMITH_API_KEY``; :func:`available`
says whether it will work. The local runner (``runner.py``) is the primary path and needs neither.
"""

from __future__ import annotations

import logging
from typing import Any

from deep_researcher.evals.cases import EvalCase
from deep_researcher.evals.judges import (
    citation_integrity,
    contradiction_recall,
    coverage,
    groundedness,
)

log = logging.getLogger(__name__)

DATASET_NAME = "deep-researcher-evals"


def available() -> bool:
    from deep_researcher import tracing

    return tracing.enabled()


def _client():
    from deep_researcher import tracing

    c = tracing.client()
    if c is None:
        raise RuntimeError("LangSmith is not configured (set LANGSMITH_TRACING=true and LANGSMITH_API_KEY)")
    return c


def upload_dataset(cases: list[EvalCase], *, name: str = DATASET_NAME) -> dict[str, Any]:
    """Create the dataset if needed and add any case not yet present (matched on ``case_id`` metadata)."""
    client = _client()
    from langsmith.utils import LangSmithNotFoundError

    try:
        dataset = client.read_dataset(dataset_name=name)
    except LangSmithNotFoundError:     # auth / network errors must surface, not turn into a create attempt
        dataset = client.create_dataset(
            dataset_name=name, description="Seeded deep-research questions with expected facets"
        )
    existing = {(ex.metadata or {}).get("case_id") for ex in client.list_examples(dataset_id=dataset.id)}
    new = [c for c in cases if c.id not in existing]
    if new:
        client.create_examples(
            dataset_id=dataset.id,
            examples=[
                {
                    "inputs": {"question": c.question},
                    "outputs": {
                        "expected_facets": c.expected_facets,
                        "expect_contradiction": c.expect_contradiction,
                        "contradiction_hint": c.contradiction_hint,
                    },
                    "metadata": {"case_id": c.id, "tags": c.tags},
                }
                for c in new
            ],
        )
    return {
        "dataset": name,
        "id": str(dataset.id),
        "added": [c.id for c in new],
        "total": len(existing - {None}) + len(new),
    }


def _case_from(
    inputs: dict[str, Any], reference: dict[str, Any] | None, metadata: dict[str, Any] | None
) -> EvalCase:
    reference = reference or {}
    return EvalCase(
        id=(metadata or {}).get("case_id", "example"),
        question=inputs["question"],
        expected_facets=reference.get("expected_facets") or ["(none)"],
        expect_contradiction=bool(reference.get("expect_contradiction")),
        contradiction_hint=reference.get("contradiction_hint"),
    )


def make_evaluators(judge_llm: Any | None) -> list[Any]:
    """LangSmith evaluators (``inputs, outputs, reference_outputs`` signature) wrapping the judges."""

    def _report(outputs: dict[str, Any]) -> dict[str, Any]:
        return outputs.get("report_json") or {}

    def citation_eval(outputs: dict) -> dict:
        r = citation_integrity(_report(outputs))
        return {"key": r.name, "score": r.score, "comment": f"{r.details.get('citations')} citations"}

    async def groundedness_eval(outputs: dict) -> dict:
        r = await groundedness(_report(outputs), llm=judge_llm)
        return {"key": r.name, "score": r.score, "comment": f"{r.mode}; {r.details.get('claims')} claims"}

    async def coverage_eval(outputs: dict, reference_outputs: dict) -> dict:
        r = await coverage(
            _report(outputs), (reference_outputs or {}).get("expected_facets") or [], llm=judge_llm
        )
        return {"key": r.name, "score": r.score, "comment": r.mode}

    async def contradiction_eval(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
        case = _case_from(inputs, reference_outputs, None)
        r = await contradiction_recall(_report(outputs), case, llm=judge_llm)
        return {"key": r.name, "score": r.score, "comment": str(r.details.get("reason", ""))[:200]}

    return [citation_eval, groundedness_eval, coverage_eval, contradiction_eval]


async def run_experiment(
    *,
    dataset: str = DATASET_NAME,
    max_hops: int = 1,
    budget_seconds: int | None = None,
    max_concurrency: int = 1,
    experiment_prefix: str = "deep-researcher",
    judge_llm: Any | None = None,
) -> Any:
    """``langsmith.aevaluate`` over the dataset with the pipeline as target and the judges as evaluators."""
    from langsmith import aevaluate

    from deep_researcher.evals.runner import run_pipeline

    async def target(inputs: dict[str, Any]) -> dict[str, Any]:
        report_md, report_json = await run_pipeline(
            inputs["question"], max_hops=max_hops, budget_seconds=budget_seconds
        )
        return {"report": report_md, "report_json": report_json}

    return await aevaluate(
        target,
        data=dataset,
        evaluators=make_evaluators(judge_llm),
        experiment_prefix=experiment_prefix,
        max_concurrency=max_concurrency,
        metadata={"max_hops": max_hops, "budget_seconds": budget_seconds},
    )
