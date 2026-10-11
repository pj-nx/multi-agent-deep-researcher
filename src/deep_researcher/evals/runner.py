"""Run the seeded cases through the real pipeline, judge each report and write a results folder.

Per case the runner streams the plan, auto-approves it, streams the research leg to the report,
sums model tokens from the agents' ``llm_end`` activity events (so cost needs no tracing backend)
and runs the judges. Results land in ``<out>/<timestamp>/``:

    <case>.report.md / .report.json   the report as the Report Builder produced it
    <case>.eval.json                  timings, tokens, estimated cost, judge scores and details
    summary.json / summary.md         one row per case plus means

``judge_existing`` scores report JSON files written earlier (CLI, UI, API) without re-running
retrieval, matching them to cases by question text.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from deep_researcher.evals.cases import EvalCase
from deep_researcher.evals.judges import JUDGE_NAMES, JudgeResult, judge_report, mean_score

log = logging.getLogger(__name__)


@dataclass
class Usage:
    """Token totals per model and call time per role, summed from ``llm_end`` activity events.

    ``roles[role]["seconds"]`` is the sum of call durations, not wall time: parallel calls overlap, so it
    shows where model time is spent, and a role's wall contribution is at most this number.
    """

    models: dict[str, dict[str, int]] = field(default_factory=dict)
    roles: dict[str, dict[str, float]] = field(default_factory=dict)

    def add(self, event: dict[str, Any]) -> None:
        model = str(event.get("model") or "unknown")
        row = self.models.setdefault(model, {"calls": 0, "input_tokens": 0, "output_tokens": 0})
        row["calls"] += 1
        row["input_tokens"] += int(event.get("input_tokens") or 0)
        row["output_tokens"] += int(event.get("output_tokens") or 0)
        rrow = self.roles.setdefault(str(event.get("role") or "unknown"), {"calls": 0, "seconds": 0.0})
        rrow["calls"] += 1
        rrow["seconds"] = round(rrow["seconds"] + float(event.get("elapsed_s") or 0), 1)

    @property
    def total_tokens(self) -> int:
        return sum(r["input_tokens"] + r["output_tokens"] for r in self.models.values())

    @property
    def calls(self) -> int:
        return sum(r["calls"] for r in self.models.values())


def estimate_cost(usage: Usage, prices: dict[str, tuple[float, float]] | None) -> float | None:
    """USD from OpenRouter per-token prices; None when any used model has no price."""
    if not prices:
        return None
    total = 0.0
    for model, row in usage.models.items():
        p = prices.get(model)
        if p is None:
            return None
        total += row["input_tokens"] * p[0] + row["output_tokens"] * p[1]
    return round(total, 6)


async def fetch_prices() -> dict[str, tuple[float, float]]:
    """Live OpenRouter prices keyed by model id; empty on any failure (cost then shows as n/a)."""
    try:
        from deep_researcher.llm import alist_models

        rows = await alist_models()
    except Exception as exc:  # noqa: BLE001 - pricing is informational
        log.warning("could not fetch model prices: %s", exc)
        return {}
    out = {}
    for r in rows:
        try:
            out[r["id"]] = (float(r["prompt_price"] or 0), float(r["completion_price"] or 0))
        except (TypeError, ValueError):
            continue
    return out


@dataclass
class CaseRun:
    case_id: str
    question: str
    ok: bool = False
    error: str | None = None
    thread_id: str | None = None
    plan_seconds: float = 0.0
    run_seconds: float = 0.0
    judge_seconds: float = 0.0
    usage: Usage = field(default_factory=Usage)
    est_cost_usd: float | None = None
    stats: dict[str, Any] = field(default_factory=dict)
    sub_questions: int = 0
    report_md: str | None = None
    report_json: dict[str, Any] | None = None
    judges: dict[str, JudgeResult] = field(default_factory=dict)

    @property
    def total_seconds(self) -> float:
        return round(self.plan_seconds + self.run_seconds, 1)

    @property
    def mean(self) -> float | None:
        return mean_score(self.judges)

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("report_md", None)
        d.pop("report_json", None)
        d["total_seconds"] = self.total_seconds
        d["total_tokens"] = self.usage.total_tokens
        d["llm_calls"] = self.usage.calls
        d["mean_score"] = self.mean
        d["judges"] = {k: v.as_dict() for k, v in self.judges.items()}
        return d


async def run_pipeline(
    question: str,
    *,
    max_hops: int,
    budget_seconds: int | None,
    graph: Any = None,
    thread_id: str | None = None,
    run: CaseRun | None = None,
) -> tuple[str | None, dict[str, Any] | None]:
    """Plan → auto-approve → research; returns (report_md, report_json) and fills timings/usage on ``run``."""
    from deep_researcher import graph as G
    from deep_researcher import tracing

    run = run or CaseRun(case_id="adhoc", question=question)
    run.thread_id = thread_id = thread_id or str(uuid.uuid4())
    cfg = tracing.run_config(thread_id, question, tags=["eval"], metadata={"eval_case": run.case_id})
    plan = None
    t = time.monotonic()
    async for event, data in G.stream_until_interrupt(
        question=question,
        thread_id=thread_id,
        max_hops=max_hops,
        budget_seconds=budget_seconds,
        config=cfg,
        graph=graph,
    ):
        if event == "activity" and data.get("kind") == "llm_end":
            run.usage.add(data)
        elif event == "plan" and data.get("status") == "awaiting_approval":
            plan = data["plan"]
    run.plan_seconds = round(time.monotonic() - t, 1)
    if plan is None:
        raise RuntimeError("planner produced no plan")
    run.sub_questions = len(plan.get("sub_questions", []))
    report_md = report_json = None
    t = time.monotonic()
    async for event, data in G.resume_with_plan(
        thread_id=thread_id, plan=plan, budget_seconds=budget_seconds, config=cfg, graph=graph
    ):
        if event == "activity":
            if data.get("kind") == "llm_end":
                run.usage.add(data)
        elif event == "report":
            report_md, report_json = data.get("report"), data.get("report_json")
        elif event == "error":
            raise RuntimeError(str(data.get("error") or data))
    run.run_seconds = round(time.monotonic() - t, 1)
    if report_json is None:
        raise RuntimeError("pipeline ended without a report")
    run.stats = report_json.get("stats") or {}
    return report_md, report_json


async def run_case(
    case: EvalCase,
    *,
    max_hops: int,
    budget_seconds: int | None,
    graph: Any = None,
    judge_llm: Any | None = None,
    prices: dict[str, tuple[float, float]] | None = None,
    judge: bool = True,
) -> CaseRun:
    run = CaseRun(case_id=case.id, question=case.question)
    hops = case.max_hops if case.max_hops is not None else max_hops
    try:
        run.report_md, run.report_json = await run_pipeline(
            case.question, max_hops=hops, budget_seconds=budget_seconds, graph=graph, run=run
        )
        run.ok = True
    except Exception as exc:  # noqa: BLE001 - one failing case must not abort the suite
        log.error("case %s failed: %s", case.id, exc)
        run.error = f"{type(exc).__name__}: {str(exc)[:300]}"
    run.est_cost_usd = estimate_cost(run.usage, prices)
    if run.ok and judge and run.report_json is not None:
        t = time.monotonic()
        run.judges = await judge_report(run.report_json, case, llm=judge_llm)
        run.judge_seconds = round(time.monotonic() - t, 1)
    return run


def _fmt(x: float | None, pct: bool = False) -> str:
    if x is None:
        return "n/a"
    return f"{x * 100:.0f}%" if pct else f"{x:g}"


def summary_rows(runs: list[CaseRun]) -> list[dict[str, Any]]:
    rows = []
    for r in runs:
        row: dict[str, Any] = {
            "case": r.case_id,
            "ok": r.ok,
            "sub_questions": r.sub_questions,
            "sources_kept": r.stats.get("kept"),
            "degradations": r.stats.get("degradations"),
            "claims": r.stats.get("claims"),
            "contradictions": r.stats.get("contradictions"),
            "hops": r.stats.get("hops"),
            "seconds": r.total_seconds,
            "tokens": r.usage.total_tokens,
            "cost_usd": r.est_cost_usd,
            "mean": r.mean,
            "error": r.error,
        }
        for name in JUDGE_NAMES:
            j = r.judges.get(name)
            row[name] = j.score if j else None
        rows.append(row)
    return rows


def aggregate(runs: list[CaseRun]) -> dict[str, Any]:
    ok = [r for r in runs if r.ok]

    def mean_of(values: list[float | None]) -> float | None:
        vals = [v for v in values if v is not None]
        return round(sum(vals) / len(vals), 3) if vals else None

    agg: dict[str, Any] = {
        "cases": len(runs),
        "succeeded": len(ok),
        "mean_seconds": mean_of([r.total_seconds for r in ok]),
        "mean_tokens": mean_of([float(r.usage.total_tokens) for r in ok]),
        "mean_cost_usd": mean_of([r.est_cost_usd for r in ok]),
        "total_cost_usd": (
            round(sum(r.est_cost_usd for r in ok if r.est_cost_usd is not None), 4)
            if any(r.est_cost_usd is not None for r in ok)
            else None
        ),
        "mean_score": mean_of([r.mean for r in ok]),
        "mean_role_seconds": {
            role: mean_of([r.usage.roles.get(role, {}).get("seconds", 0.0) for r in ok])
            for role in sorted({role for r in ok for role in r.usage.roles})
        },
    }
    for name in JUDGE_NAMES:
        agg[name] = mean_of([r.judges[name].score for r in ok if name in r.judges])
    return agg


def summary_markdown(
    runs: list[CaseRun],
    agg: dict[str, Any],
    *,
    title: str = "Evaluation summary",
    settings: dict[str, Any] | None = None,
) -> str:
    head = [
        "Case",
        "Sub-Qs",
        "Kept",
        "Claims",
        "Contra.",
        "Time",
        "Tokens",
        "Cost",
        "Citations",
        "Grounded",
        "Coverage",
        "Contra. recall",
        "Mean",
    ]
    lines = [f"# {title}", ""]
    if settings:
        lines += ["Settings: " + ", ".join(f"{k}={v}" for k, v in settings.items()), ""]
    lines += ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in runs:
        if not r.ok:
            lines.append(f"| {r.case_id} | failed: {r.error} |" + " |" * (len(head) - 2))
            continue
        j = {n: r.judges.get(n) for n in JUDGE_NAMES}
        cost = f"${r.est_cost_usd:.3f}" if r.est_cost_usd is not None else "n/a"
        lines.append(
            "| "
            + " | ".join(
                [
                    r.case_id,
                    str(r.sub_questions),
                    str(r.stats.get("kept", "")),
                    str(r.stats.get("claims", "")),
                    str(r.stats.get("contradictions", "")),
                    f"{r.total_seconds:.0f}s",
                    f"{r.usage.total_tokens:,}",
                    cost,
                    *[_fmt(j[n].score if j[n] else None, pct=True) for n in JUDGE_NAMES],
                    _fmt(r.mean, pct=True),
                ]
            )
            + " |"
        )
    cost = f"${agg['mean_cost_usd']:.3f}" if agg.get("mean_cost_usd") is not None else "n/a"
    secs = f"{agg['mean_seconds']:.0f}s" if agg.get("mean_seconds") is not None else "n/a"
    toks = f"{agg['mean_tokens']:,.0f}" if agg.get("mean_tokens") is not None else "n/a"
    lines.append(
        "| "
        + " | ".join(
            [
                f"**mean ({agg['succeeded']}/{agg['cases']} ok)**",
                "",
                "",
                "",
                "",
                secs,
                toks,
                cost,
                *[_fmt(agg.get(n), pct=True) for n in JUDGE_NAMES],
                _fmt(agg.get("mean_score"), pct=True),
            ]
        )
        + " |"
    )
    lines += [
        "",
        "Scores: citations = cited ids that exist in the bibliography; grounded = claims whose cited "
        "text supports them (model-judged, partial = 0.5); coverage = expected facets addressed; "
        "contra. recall = seeded disagreement surfaced (1.0 matches the hint, 0.5 other contradiction "
        "only, n/a when none was seeded). Cost is estimated from OpenRouter per-token prices for the "
        "models used; embeddings are not included.",
        "",
    ]
    if agg.get("mean_role_seconds"):
        lines += [
            "Model call time by role (mean per case; summed call durations, so parallel calls overlap and "
            "the total can exceed wall time): "
            + ", ".join(f"{role} {sec:g}s" for role, sec in agg["mean_role_seconds"].items()),
            "",
        ]
    return "\n".join(lines)


def write_results(
    runs: list[CaseRun],
    out_dir: Path,
    *,
    settings: dict[str, Any] | None = None,
    title: str = "Evaluation summary",
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    for r in runs:
        if r.report_md:
            (out_dir / f"{r.case_id}.report.md").write_text(r.report_md)
        if r.report_json:
            (out_dir / f"{r.case_id}.report.json").write_text(
                json.dumps(r.report_json, indent=2, default=str)
            )
        (out_dir / f"{r.case_id}.eval.json").write_text(json.dumps(r.as_dict(), indent=2, default=str))
    agg = aggregate(runs)
    (out_dir / "summary.json").write_text(
        json.dumps(
            {"settings": settings or {}, "aggregate": agg, "rows": summary_rows(runs)}, indent=2, default=str
        )
    )
    path = out_dir / "summary.md"
    path.write_text(summary_markdown(runs, agg, title=title, settings=settings))
    return path


async def run_suite(
    cases: list[EvalCase],
    *,
    out_dir: Path,
    max_hops: int = 1,
    budget_seconds: int | None = None,
    concurrency: int = 1,
    graph: Any = None,
    judge_llm: Any | None = None,
    prices: dict[str, tuple[float, float]] | None = None,
    judge: bool = True,
    settings: dict[str, Any] | None = None,
    on_done: Any = None,
) -> list[CaseRun]:
    """Run every case (``concurrency`` at a time), writing results as each finishes."""
    sem = asyncio.Semaphore(max(1, concurrency))
    runs: list[CaseRun] = []

    async def one(case: EvalCase) -> CaseRun:
        async with sem:
            r = await run_case(
                case,
                max_hops=max_hops,
                budget_seconds=budget_seconds,
                graph=graph,
                judge_llm=judge_llm,
                prices=prices,
                judge=judge,
            )
        runs.append(r)
        write_results(
            sorted(runs, key=lambda x: x.case_id), out_dir, settings=settings
        )  # partial = still useful
        if on_done:
            on_done(r)
        return r

    await asyncio.gather(*(one(c) for c in cases))
    ordered = sorted(runs, key=lambda r: [c.id for c in cases].index(r.case_id))
    write_results(ordered, out_dir, settings=settings)
    return ordered


def _match_case(report_json: dict[str, Any], cases: list[EvalCase]) -> EvalCase | None:
    q = (report_json.get("question") or "").strip().lower().rstrip("?")
    for c in cases:
        if c.question.strip().lower().rstrip("?") == q:
            return c
    return None


async def judge_existing(
    paths: list[Path],
    cases: list[EvalCase],
    *,
    judge_llm: Any | None = None,
) -> list[CaseRun]:
    """Score already-written report JSON files; coverage/contradiction judges need a matching case."""
    runs = []
    for p in paths:
        data = json.loads(Path(p).read_text())
        case = _match_case(data, cases)
        run = CaseRun(
            case_id=case.id if case else Path(p).stem.replace(".report", ""),
            question=data.get("question", ""),
            ok=True,
            report_json=data,
            stats=data.get("stats") or {},
            sub_questions=len((data.get("plan") or {}).get("sub_questions", [])),
        )
        t = time.monotonic()
        run.judges = await judge_report(data, case, llm=judge_llm)
        run.judge_seconds = round(time.monotonic() - t, 1)
        runs.append(run)
    return runs
