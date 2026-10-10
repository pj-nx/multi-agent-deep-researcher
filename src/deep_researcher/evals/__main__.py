"""Evaluation CLI.

uv run python -m deep_researcher.evals list
uv run python -m deep_researcher.evals run [--ids a,b] [--limit N] [--max-hops 1] [--budget 420]
                                           [--concurrency 1] [--out evals/results] [--no-judge]
uv run python -m deep_researcher.evals judge reports/*.json [--out evals/results]
uv run python -m deep_researcher.evals langsmith upload | experiment [--max-hops 1] [--budget 420]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from deep_researcher.evals.cases import default_path, load_cases


def _settings(args: argparse.Namespace) -> dict:
    from deep_researcher.llm import model_for

    return {
        "max_hops": args.max_hops,
        "budget_seconds": args.budget,
        "heavy": model_for("planner"),
        "worker": model_for("report"),
        "cheap": model_for("validator"),
        "grader": model_for("grader"),
        "reasoning_effort": os.getenv("OPENROUTER_REASONING_EFFORT") or "default",
        "embeddings": os.getenv("EMBEDDINGS_PROVIDER", "local"),
    }


async def _run(args: argparse.Namespace) -> int:
    from deep_researcher import graph as G
    from deep_researcher import tracing
    from deep_researcher.evals.judges import grader_llm
    from deep_researcher.evals.runner import fetch_prices, run_suite

    tracing.configure()
    cases = load_cases(args.questions, ids=args.ids.split(",") if args.ids else None, limit=args.limit)
    out = Path(args.out) / time.strftime("%Y%m%d-%H%M%S")
    prices = await fetch_prices()
    judge_llm = None if args.no_judge else grader_llm()
    settings = _settings(args)
    print(f"{len(cases)} case(s) -> {out}  [{', '.join(f'{k}={v}' for k, v in settings.items())}]")

    def on_done(r):
        status = "ok" if r.ok else f"FAILED {r.error}"
        scores = " ".join(f"{k}={v.score if v.score is not None else 'n/a'}" for k, v in r.judges.items())
        cost = f"${r.est_cost_usd:.3f}" if r.est_cost_usd is not None else "n/a"
        print(
            f"[{r.case_id}] {status} {r.total_seconds}s {r.usage.total_tokens:,} tok {cost} {scores}",
            flush=True,
        )

    try:
        runs = await run_suite(
            cases,
            out_dir=out,
            max_hops=args.max_hops,
            budget_seconds=args.budget,
            concurrency=args.concurrency,
            judge_llm=judge_llm,
            prices=prices,
            judge=not args.no_judge,
            settings=settings,
            on_done=on_done,
        )
    finally:
        tracing.flush()
        await G.aclose()
    print(f"\n{(out / 'summary.md').read_text()}")
    return 0 if all(r.ok for r in runs) else 1


async def _judge(args: argparse.Namespace) -> int:
    from deep_researcher.evals.judges import grader_llm
    from deep_researcher.evals.runner import judge_existing, write_results

    cases = load_cases(args.questions)
    runs = await judge_existing(
        [Path(p) for p in args.paths], cases, judge_llm=None if args.no_judge else grader_llm()
    )
    out = Path(args.out) / f"{time.strftime('%Y%m%d-%H%M%S')}-judge"
    path = write_results(runs, out, title="Judge-only evaluation")
    print(path.read_text())
    return 0


async def _langsmith(args: argparse.Namespace) -> int:
    from deep_researcher import graph as G
    from deep_researcher import tracing
    from deep_researcher.evals import langsmith_evals as LS
    from deep_researcher.evals.judges import grader_llm

    tracing.configure()
    if not LS.available():
        print(
            "LangSmith is not configured: set LANGSMITH_TRACING=true and LANGSMITH_API_KEY", file=sys.stderr
        )
        return 2
    if args.action == "upload":
        print(json.dumps(LS.upload_dataset(load_cases(args.questions), name=args.dataset), indent=2))
        return 0
    try:
        results = await LS.run_experiment(
            dataset=args.dataset,
            max_hops=args.max_hops,
            budget_seconds=args.budget,
            max_concurrency=args.concurrency,
            judge_llm=grader_llm(),
        )
    finally:
        tracing.flush()
        await G.aclose()
    print(f"experiment: {getattr(results, 'experiment_name', results)}")
    return 0


def main() -> None:
    load_dotenv()
    p = argparse.ArgumentParser(
        prog="deep_researcher.evals",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--questions", default=str(default_path()))
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="print the seeded cases")

    r = sub.add_parser("run", help="run the pipeline on the cases and judge the reports")
    r.add_argument("--ids", help="comma-separated case ids")
    r.add_argument("--limit", type=int)
    r.add_argument("--max-hops", type=int, default=int(os.getenv("MAX_HOPS", "1")))
    r.add_argument(
        "--budget",
        type=int,
        default=int(os.getenv("RUN_BUDGET_SECONDS", "0")) or None,
        help="seconds per leg before retrieval loops stop",
    )
    r.add_argument("--concurrency", type=int, default=1)
    r.add_argument("--out", default="evals/results")
    r.add_argument("--no-judge", action="store_true", help="run the pipeline only")

    j = sub.add_parser("judge", help="judge existing report JSON files")
    j.add_argument("paths", nargs="+")
    j.add_argument("--out", default="evals/results")
    j.add_argument("--no-judge", action="store_true", help="deterministic judges only (no model)")

    ls = sub.add_parser("langsmith", help="dataset upload / experiment on LangSmith")
    ls.add_argument("action", choices=["upload", "experiment"])
    ls.add_argument("--dataset", default="deep-researcher-evals")
    ls.add_argument("--max-hops", type=int, default=int(os.getenv("MAX_HOPS", "1")))
    ls.add_argument("--budget", type=int, default=int(os.getenv("RUN_BUDGET_SECONDS", "0")) or None)
    ls.add_argument("--concurrency", type=int, default=1)

    args = p.parse_args()
    if args.cmd == "list":
        for c in load_cases(args.questions):
            flag = " [contradiction]" if c.expect_contradiction else ""
            print(
                f"{c.id}: {c.question}{flag}\n    facets: {len(c.expected_facets)}; tags: {', '.join(c.tags)}"
            )
        sys.exit(0)
    fn = {"run": _run, "judge": _judge, "langsmith": _langsmith}[args.cmd]
    sys.exit(asyncio.run(fn(args)))


if __name__ == "__main__":
    main()
