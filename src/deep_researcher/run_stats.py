# Copyright (c) 2026 C10 Hackathon G1
"""Per-run statistics for the UIs: tokens and calls per model, call time per role, wall time per leg,
the report's own counters, run-health issues and an estimated cost.

Fed with the same ``(event, data)`` stream the UIs already consume (``activity`` events carry
``llm_end`` token usage; ``report`` carries ``stats``), so it needs no tracing backend. Cost uses
OpenRouter's per-token prices when the catalogue can be fetched; otherwise it is shown as unknown.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)

_prices: dict[str, Any] = {"at": 0.0, "rows": {}}


async def get_prices(timeout: float = 5.0) -> dict[str, tuple[float, float]]:
    """OpenRouter prices keyed by model id, cached for ten minutes; empty when unavailable."""
    if _prices["rows"] and time.monotonic() - _prices["at"] < 600:
        return _prices["rows"]
    try:
        from deep_researcher.llm import alist_models

        rows = await asyncio.wait_for(alist_models(timeout=timeout), timeout=timeout + 1)
        _prices["rows"] = {
            r["id"]: (float(r["prompt_price"] or 0), float(r["completion_price"] or 0)) for r in rows
        }
        _prices["at"] = time.monotonic()
    except Exception as exc:  # noqa: BLE001 - pricing is informational
        log.debug("model prices unavailable: %s", exc)
    return _prices["rows"]


@dataclass
class RunStats:
    question: str = ""
    thread_id: str = ""
    started: float = field(default_factory=time.time)
    plan_seconds: float = 0.0
    run_seconds: float = 0.0
    sub_questions: int = 0
    models: dict[str, dict[str, int]] = field(default_factory=dict)  # model -> calls/in/out
    roles: dict[str, dict[str, float]] = field(default_factory=dict)  # role -> calls/seconds
    tools: dict[str, int] = field(default_factory=dict)  # tool -> results
    stats: dict[str, Any] = field(default_factory=dict)  # the report's counters
    degradations: list[dict[str, Any]] = field(default_factory=list)
    critic_rounds: int = 0
    error: str | None = None
    est_cost_usd: float | None = None

    # ------------------------------------------------------------------ feed
    def add_event(self, event: str, data: dict[str, Any]) -> None:
        if event == "activity":
            kind = data.get("kind")
            if kind == "llm_end":
                m = self.models.setdefault(
                    str(data.get("model") or "unknown"), {"calls": 0, "input_tokens": 0, "output_tokens": 0}
                )
                m["calls"] += 1
                m["input_tokens"] += int(data.get("input_tokens") or 0)
                m["output_tokens"] += int(data.get("output_tokens") or 0)
                r = self.roles.setdefault(str(data.get("role") or "unknown"), {"calls": 0, "seconds": 0.0})
                r["calls"] += 1
                r["seconds"] = round(r["seconds"] + float(data.get("elapsed_s") or 0), 1)
            elif kind == "tool_end":
                t = str(data.get("tool") or "tool")
                self.tools[t] = self.tools.get(t, 0) + int(data.get("results") or 0)
            elif kind == "degraded":
                self.degradations.append(data)
            return
        if event == "plan" and data.get("status") == "awaiting_approval":
            self.sub_questions = len((data.get("plan") or {}).get("sub_questions", []))
        elif event == "critic":
            self.critic_rounds = max(self.critic_rounds, int(data.get("round") or 0))
        elif event == "report":
            self.stats = dict(data.get("stats") or {})
            if data.get("degradations") and not self.degradations:
                self.degradations = list(data["degradations"])
        elif event == "error":
            self.error = str(data.get("message") or "failed")

    # ------------------------------------------------------------------ derived
    @property
    def total_tokens(self) -> int:
        return sum(m["input_tokens"] + m["output_tokens"] for m in self.models.values())

    @property
    def calls(self) -> int:
        return sum(m["calls"] for m in self.models.values())

    @property
    def total_seconds(self) -> float:
        return round(self.plan_seconds + self.run_seconds, 1)

    def estimate_cost(self, prices: dict[str, tuple[float, float]] | None) -> float | None:
        if not prices or not self.models:
            return None
        total = 0.0
        for model, m in self.models.items():
            p = prices.get(model)
            if p is None:
                return None
            total += m["input_tokens"] * p[0] + m["output_tokens"] * p[1]
        self.est_cost_usd = round(total, 6)
        return self.est_cost_usd

    # ------------------------------------------------------------------ render
    def summary_markdown(self) -> str:
        s = self.stats
        cost = f"${self.est_cost_usd:.3f}" if self.est_cost_usd is not None else "n/a"
        run_s, hops, removed = self.run_seconds, s.get("hops", 0), s.get("citations_removed", 0)
        rows = [
            ("Time", f"{self.total_seconds:.0f} s (plan {self.plan_seconds:.0f} s, research {run_s:.0f} s)"),
            ("Model calls", f"{self.calls} · {self.total_tokens:,} tokens · est. {cost}"),
            (
                "Sources",
                f"{s.get('retrieved', 0)} retrieved, {s.get('kept', 0)} kept, {s.get('dropped', 0)} dropped"
                + (
                    f" · {sum(self.tools.values())} results from {len(self.tools)} tools"
                    if self.tools
                    else ""
                ),
            ),
            (
                "Analysis",
                f"{s.get('claims', 0)} claims · {s.get('contradictions', 0)} contradictions · "
                f"{s.get('gaps', 0)} gaps raised, {s.get('gaps_pursued', 0)} pursued · {hops} hop(s)",
            ),
            (
                "Synthesis",
                f"{self.sub_questions} sub-questions · critic rounds {self.critic_rounds} · "
                f"{s.get('citations_verified', 0)} citations verified, {removed} removed",
            ),
            (
                "Run health",
                (
                    "no issues"
                    if not self.degradations
                    else f"{len(self.degradations)} issue(s): "
                    + ", ".join(
                        dict.fromkeys(
                            f"{d.get('component')}/{d.get('error_kind') or d.get('kind')}"
                            for d in self.degradations
                        )
                    )
                ),
            ),
        ]
        if self.models:
            rows.append(
                (
                    "Models",
                    ", ".join(
                        f"{m.split('/')[-1]} x{v['calls']} ({v['input_tokens'] + v['output_tokens']:,} tok)"
                        for m, v in sorted(self.models.items(), key=lambda kv: -kv[1]["calls"])
                    ),
                )
            )
        if self.roles:
            rows.append(
                (
                    "Model time by role",
                    ", ".join(
                        f"{r} {v['seconds']:.0f}s"
                        for r, v in sorted(self.roles.items(), key=lambda kv: -kv[1]["seconds"])
                    ),
                )
            )
        if self.error:
            rows.append(("Error", self.error))
        return "| | |\n|---|---|\n" + "\n".join(f"| **{k}** | {v} |" for k, v in rows)

    def row(self) -> list[Any]:
        """One line for the session table."""
        s = self.stats
        cost = f"${self.est_cost_usd:.3f}" if self.est_cost_usd is not None else "n/a"
        return [
            time.strftime("%H:%M", time.localtime(self.started)),
            self.question[:60],
            f"{self.total_seconds:.0f} s",
            f"{self.total_tokens:,}",
            cost,
            str(s.get("kept", 0)),
            str(s.get("claims", 0)),
            str(s.get("contradictions", 0)),
            str(s.get("hops", 0)),
            str(len(self.degradations)),
            "error" if self.error else "ok",
        ]


SESSION_HEADERS = [
    "Time",
    "Question",
    "Duration",
    "Tokens",
    "Cost",
    "Kept",
    "Claims",
    "Contra.",
    "Hops",
    "Issues",
    "Status",
]


def session_markdown(rows: list[list[Any]]) -> str:
    if not rows:
        return ""
    head = "| " + " | ".join(SESSION_HEADERS) + " |\n|" + "---|" * len(SESSION_HEADERS)
    body = "\n".join("| " + " | ".join(str(c).replace("|", "/") for c in r) + " |" for r in reversed(rows))
    return f"{head}\n{body}"
