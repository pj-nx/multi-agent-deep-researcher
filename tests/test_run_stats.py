"""Per-run statistics shown in the UIs: fed from the event stream, cost from prices, Markdown rendering."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from deep_researcher import run_stats as RS


def _fed() -> RS.RunStats:
    s = RS.RunStats(question="Does remote work raise productivity?", thread_id="t1")
    s.add_event("plan", {"status": "awaiting_approval", "plan": {"sub_questions": [{"id": "sq1"}, {"id": "sq2"}]}})
    s.add_event("activity", {"kind": "llm_end", "model": "a/planner-model", "role": "planner", "input_tokens": 1000,
                             "output_tokens": 200, "elapsed_s": 3.0})
    s.add_event("activity", {"kind": "llm_end", "model": "b/cheap", "role": "validator", "input_tokens": 500,
                             "output_tokens": 50, "elapsed_s": 1.5})
    s.add_event("activity", {"kind": "llm_end", "model": "b/cheap", "role": "validator", "input_tokens": 500,
                             "output_tokens": 50, "elapsed_s": 1.5})
    s.add_event("activity", {"kind": "tool_end", "tool": "arxiv", "results": 4})
    s.add_event("activity", {"kind": "tool_end", "tool": "tavily", "results": 6})
    s.add_event("activity", {"kind": "degraded", "component": "tool:semantic_scholar", "error_kind": "rate_limited"})
    s.add_event("critic", {"round": 2, "verdict": "accept"})
    s.add_event("report", {"stats": {"retrieved": 10, "kept": 7, "dropped": 3, "claims": 12, "contradictions": 1,
                                     "gaps": 4, "gaps_pursued": 2, "hops": 1, "citations_verified": 9,
                                     "citations_removed": 0}})
    s.plan_seconds, s.run_seconds = 12.0, 100.5
    return s


def test_feed_totals_and_cost():
    s = _fed()
    assert s.sub_questions == 2 and s.calls == 3 and s.total_tokens == 2300 and s.critic_rounds == 2
    assert s.models["b/cheap"] == {"calls": 2, "input_tokens": 1000, "output_tokens": 100}
    assert s.roles["validator"] == {"calls": 2, "seconds": 3.0} and s.tools == {"arxiv": 4, "tavily": 6}
    assert len(s.degradations) == 1 and s.total_seconds == 112.5
    assert s.estimate_cost({"a/planner-model": (1e-6, 2e-6), "b/cheap": (1e-7, 2e-7)}) == pytest.approx(0.00152)
    assert s.estimate_cost({"a/planner-model": (1e-6, 2e-6)}) is None        # unknown price
    assert s.estimate_cost(None) is None and RS.RunStats().estimate_cost({"x": (1, 1)}) is None


def test_summary_markdown_and_session_rows():
    s = _fed()
    s.estimate_cost({"a/planner-model": (1e-6, 2e-6), "b/cheap": (1e-7, 2e-7)})
    md = s.summary_markdown()
    for needle in ("| **Time** | 112 s (plan 12 s, research 100 s)", "3 · 2,300 tokens · est. $0.002",
                   "10 retrieved, 7 kept, 3 dropped · 10 results from 2 tools",
                   "12 claims · 1 contradictions · 4 gaps raised, 2 pursued · 1 hop(s)",
                   "critic rounds 2 · 9 citations verified", "1 issue(s): tool:semantic_scholar/rate_limited",
                   "cheap x2 (1,100 tok)", "validator 3s"):
        assert needle in md, needle
    row = s.row()
    assert row[1].startswith("Does remote work") and row[2] == "112 s" and row[3] == "2,300" and row[-1] == "ok"
    assert row[5:10] == ["7", "12", "1", "1", "1"]
    table = RS.session_markdown([row, RS.RunStats(question="second | one").row()])
    assert table.startswith("| Time | Question |") and "second / one" in table         # pipes escaped
    assert table.index("second") < table.index("Does remote")                        # newest first
    assert RS.session_markdown([]) == ""


def test_error_and_report_degradations():
    s = RS.RunStats()
    s.add_event("report", {"stats": {"kept": 1}, "degradations": [{"component": "critic", "kind": "model_error"}]})
    s.add_event("error", {"message": "boom"})
    md = s.summary_markdown()
    assert "critic/model_error" in md and "| **Error** | boom |" in md and s.row()[-1] == "error"


@pytest.mark.anyio
async def test_get_prices_degrades_to_empty(monkeypatch):
    from deep_researcher import llm

    async def boom(*a, **k):
        raise RuntimeError("offline")

    monkeypatch.setattr(llm, "alist_models", boom)
    RS._prices.update({"at": 0.0, "rows": {}})
    assert await RS.get_prices(timeout=0.5) == {}
