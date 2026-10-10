"""CLI: `uv run python -m deep_researcher "question" [--max-hops N] [--auto-approve] [--out reports/]`"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv

from deep_researcher.reports import write_report


async def _main(args: argparse.Namespace) -> int:
    from deep_researcher import graph as G
    from deep_researcher import tracing

    tracing.configure()
    thread_id = str(uuid.uuid4())
    cfg = tracing.run_config(thread_id, args.question, tags=["cli"])
    plan = None
    async for event, data in G.stream_until_interrupt(question=args.question, thread_id=thread_id,
                                                      max_hops=args.max_hops, budget_seconds=args.budget,
                                                      config=cfg):
        _print(event, data)
        if event == "plan" and data.get("status") == "awaiting_approval":
            plan = data["plan"]
    if plan is None:
        print("no plan produced", file=sys.stderr)
        await G.aclose()
        return 1
    if not args.auto_approve:
        print("\nProposed plan:")
        for sq in plan["sub_questions"]:
            print(f"  {sq['id']}: {sq['question']}  [{', '.join(sq['source_types'])}]")
        if input("\nApprove? [y/N] ").strip().lower() not in {"y", "yes"}:
            print("aborted")
            await G.aclose()
            return 2
    report = None
    report_json = None
    async for event, data in G.resume_with_plan(thread_id=thread_id, plan=plan, budget_seconds=args.budget,
                                                config=cfg):
        _print(event, data)
        if event == "report":
            report = data.get("report")
            report_json = data.get("report_json")
    tracing.flush()
    await G.aclose()
    if report:
        path, json_path = write_report(args.question, report, report_json, Path(args.out))
        print(f"\nreport written to {path}" + (" (+ .json)" if json_path else ""))
    return 0


def _print(event: str, data: dict) -> None:
    if event == "report":
        print(f"[report] ready {json.dumps(data.get('stats'), default=str)}")
        return
    short = {k: v for k, v in data.items() if k not in ("plan", "report", "report_json")}
    print(f"[{event}] {json.dumps(short, default=str)[:300]}")


def main() -> None:
    load_dotenv()
    p = argparse.ArgumentParser(prog="deep_researcher")
    p.add_argument("question")
    p.add_argument("--max-hops", type=int, default=1)
    p.add_argument("--budget", type=int, default=None, help="seconds before retrieval loops stop")
    p.add_argument("--auto-approve", action="store_true")
    p.add_argument("--out", default="reports")
    sys.exit(asyncio.run(_main(p.parse_args())))


if __name__ == "__main__":
    main()
