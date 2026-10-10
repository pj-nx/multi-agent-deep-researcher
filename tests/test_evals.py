"""Phase 7: seeded cases, judges (deterministic + fake grader) and the suite runner on the fake graph."""
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from deep_researcher.evals import cases as C
from deep_researcher.evals import judges as J
from deep_researcher.evals import runner as R
from fakes import FakeStructured

pytestmark = pytest.mark.real_graph

ROOT = Path(__file__).resolve().parents[1]
A, B, X = "E-aaaaaaaaaa", "E-bbbbbbbbbb", "E-cccccccccc"


def report(**over):
    r = {
        "title": "Remote work and productivity", "question": "Does remote work raise productivity?",
        "executive_summary": f"Mixed evidence [{A}]. A dangling citation [{B}].",
        "key_findings": [f"Knowledge work shows gains in randomised trials [{A}]"],
        "summaries": {"sq1": {"summary": "Trials of remote work.", "claims": [
            {"text": "Trials show higher output in knowledge work", "evidence_ids": [A], "confidence": 0.8},
            {"text": "Output doubled (overstates)", "evidence_ids": [A], "confidence": 0.5},
            {"text": "wrong claim about manufacturing", "evidence_ids": [X], "confidence": 0.5}]}},
        "contradictions": [{"sub_question_id": "sq1", "claim_a": "productivity rises", "claim_b": "productivity falls",
                            "evidence_ids_a": [A], "evidence_ids_b": [X], "explanation": "different populations"}],
        "insights": [{"hypothesis": "Effects depend on sector", "confidence": 0.6,
                      "reasoning": [{"step": "trials in knowledge work show gains", "evidence_ids": [A]}]}],
        "limitations": ["few sources"], "open_questions": [],
        "sources": [{"id": A, "title": "Trial", "text": "The trial found higher output among knowledge workers."},
                    {"id": X, "title": "Other", "text": "Factory output fell during the pilot."}],
        "stats": {"kept": 2, "claims": 3, "contradictions": 1, "hops": 0},
        "plan": {"sub_questions": [{"id": "sq1"}]},
    }
    r.update(over)
    return r


class FakeGrader:
    """Dispatches on the judge schema: 'overstates' -> partial, 'wrong' -> unsupported; facets by keyword."""

    model_name = "fake/grader"

    def __init__(self):
        self.calls = []

    def with_structured_output(self, schema, **kwargs):
        assert kwargs.get("method") == "json_schema"
        return FakeStructured(lambda msgs: self._answer(schema, msgs[-1].content))

    def _answer(self, schema, prompt):
        self.calls.append(schema.__name__)
        if schema is J.GroundingVerdicts:
            items = []
            for m in re.finditer(r"<claim index=(\d+)>(.*?)</claim>", prompt, re.S):
                text = m.group(2)
                verdict = ("partially_supported" if "overstates" in text else
                           "unsupported" if "wrong" in text else "supported")
                items.append(J.ClaimVerdict(index=int(m.group(1)), verdict=verdict, reason="fake"))
            return J.GroundingVerdicts(items=items)
        if schema is J.CoverageVerdicts:
            rep = prompt.split("<report>", 1)[1].lower()
            items = [J.FacetVerdict(index=int(i), covered=f.split()[0].lower() in rep, where="fake")
                     for i, f in re.findall(r"<facet index=(\d+)>(.*?)</facet>", prompt)]
            return J.CoverageVerdicts(items=items)
        if schema is J.ContradictionMatch:
            return J.ContradictionMatch(matches="productivity" in prompt, reason="fake")
        raise AssertionError(schema)


# ----------------------------------------------------------------------------- cases

def test_seeded_questions_load_and_are_well_formed():
    cases = C.load_cases()
    assert len(cases) == 10
    assert len({c.id for c in cases}) == 10
    assert all(len(c.expected_facets) >= 3 for c in cases)
    assert sum(c.expect_contradiction for c in cases) >= 6
    assert all(c.contradiction_hint for c in cases if c.expect_contradiction)


def test_load_cases_filters_and_rejects_duplicates(tmp_path):
    cases = C.load_cases(ids=["four-day-week", "grid-batteries"])
    assert [c.id for c in cases] == ["four-day-week", "grid-batteries"]
    assert len(C.load_cases(limit=3)) == 3
    with pytest.raises(ValueError, match="unknown case ids"):
        C.load_cases(ids=["nope"])
    p = tmp_path / "q.jsonl"
    row = json.dumps({"id": "dup", "question": "Is water wet?", "expected_facets": ["x"]})
    p.write_text(f"# comment\n{row}\n\n{row}\n")
    with pytest.raises(ValueError, match="duplicate"):
        C.load_cases(p)


def test_cli_list_prints_cases():
    out = subprocess.run([sys.executable, "-m", "deep_researcher.evals", "list"], cwd=ROOT, capture_output=True,
                         text=True, check=True, env={"PYTHONPATH": "src", "PATH": "/usr/bin:/bin"}).stdout
    assert "four-day-week:" in out and "[contradiction]" in out and out.count("facets:") == 10


# ----------------------------------------------------------------------------- judges

def test_citation_integrity_counts_every_section():
    r = J.citation_integrity(report())
    assert r.score == 0.889                       # 9 citations, one dangling [E-bbbbbbbbbb]
    assert r.details["invalid"] == [("executive_summary", B)]
    assert J.citation_integrity(report(executive_summary="no cites", key_findings=[], summaries={}, insights=[],
                                       contradictions=[])).score == 1.0


@pytest.mark.anyio
async def test_groundedness_structural_without_model():
    r = await J.groundedness(report())
    assert r.mode == "structural" and r.score == 1.0
    bad = report()
    bad["summaries"]["sq1"]["claims"][0]["evidence_ids"] = ["E-deadbeef00"]
    assert (await J.groundedness(bad)).score == pytest.approx(2 / 3, abs=1e-3)
    assert (await J.groundedness(report(summaries={}))).score == 0.0


@pytest.mark.anyio
async def test_groundedness_with_grader_weights_partial_support():
    grader = FakeGrader()
    r = await J.groundedness(report(), llm=grader)
    assert r.mode == "model" and grader.calls == ["GroundingVerdicts"]
    assert r.score == 0.5                          # 1 + 0.5 + 0 over 3 claims
    assert [v["verdict"] for v in r.details["verdicts"]] == ["supported", "partially_supported", "unsupported"]
    # a claim citing a non-existent source scores 0 whatever the model says
    bad = report()
    bad["summaries"]["sq1"]["claims"][0]["evidence_ids"] = ["E-deadbeef00"]
    assert (await J.groundedness(bad, llm=grader)).score == pytest.approx(1 / 6, abs=1e-3)


@pytest.mark.anyio
async def test_coverage_keyword_fallback_and_model():
    facets = ["randomised trials of remote work", "knowledge work versus manufacturing", "zebra migration patterns"]
    kw = await J.coverage(report(), facets)
    assert kw.mode == "keyword" and kw.score == pytest.approx(2 / 3, abs=1e-3)
    assert [f["covered"] for f in kw.details["facets"]] == [True, True, False]
    model = await J.coverage(report(), facets, llm=FakeGrader())
    assert model.mode == "model" and model.score == pytest.approx(2 / 3, abs=1e-3)
    assert (await J.coverage(report(), [])).score is None


@pytest.mark.anyio
async def test_contradiction_recall_variants():
    case = C.EvalCase(id="c", question="Is it so?", expected_facets=["x"], expect_contradiction=True,
                      contradiction_hint="productivity rises in trials but falls in factories")
    assert (await J.contradiction_recall(report(), case)).score == 1.0
    assert (await J.contradiction_recall(report(contradictions=[]), case)).score == 0.0
    assert (await J.contradiction_recall(report(), case, llm=FakeGrader())).score == 1.0
    other = case.model_copy(update={"contradiction_hint": "zebras disagree"})
    r = report()
    r["contradictions"][0].update({"claim_a": "rain", "claim_b": "sun"})
    assert (await J.contradiction_recall(r, other, llm=FakeGrader())).score == 0.5
    none = case.model_copy(update={"expect_contradiction": False})
    assert (await J.contradiction_recall(report(), none)).score is None


@pytest.mark.anyio
async def test_judge_report_and_mean():
    case = C.EvalCase(id="c", question="Is it so?", expected_facets=["randomised trials", "zebras"],
                      expect_contradiction=True)
    results = await J.judge_report(report(), case, llm=FakeGrader())
    assert set(results) == set(J.JUDGE_NAMES)
    assert J.mean_score(results) == pytest.approx((0.889 + 0.5 + 0.5 + 1.0) / 4, abs=1e-3)


# ----------------------------------------------------------------------------- runner

def test_usage_and_cost_estimate():
    u = R.Usage()
    u.add({"model": "m1", "input_tokens": 1000, "output_tokens": 500})
    u.add({"model": "m1", "input_tokens": 1000, "output_tokens": 500})
    u.add({"model": "m2", "input_tokens": 10, "output_tokens": 0})
    assert u.total_tokens == 3010 and u.calls == 3
    assert R.estimate_cost(u, {"m1": (1e-6, 2e-6), "m2": (1e-6, 1e-6)}) == pytest.approx(0.00401)
    assert R.estimate_cost(u, {"m1": (1e-6, 2e-6)}) is None       # unknown price for m2
    assert R.estimate_cost(u, None) is None


@pytest.mark.anyio
async def test_run_suite_on_fake_graph_writes_results(tmp_path):
    pytest.importorskip("chromadb")
    from langgraph.checkpoint.memory import InMemorySaver

    from deep_researcher import graph as G
    from fakes import make_deps

    deps = make_deps(tmp_path)
    graph = G.build_graph(InMemorySaver(), deps)
    case = C.EvalCase(id="remote-work", question="Does remote work raise productivity?",
                      expected_facets=["randomised trials of remote work", "knowledge work gains", "zebras"],
                      expect_contradiction=True, contradiction_hint="productivity rises versus falls")
    done = []
    runs = await R.run_suite([case], out_dir=tmp_path / "out", max_hops=0, graph=graph, judge_llm=FakeGrader(),
                             prices={"unknown": (1e-6, 2e-6)}, settings={"max_hops": 0}, on_done=done.append)
    r = runs[0]
    assert r.ok and done == [r] and r.sub_questions == 3 and r.stats["kept"] >= 1
    assert r.usage.calls >= 5 and r.usage.total_tokens == r.usage.calls * 1200     # fake raw usage per call
    assert r.est_cost_usd == pytest.approx(r.usage.calls * (1000 * 1e-6 + 200 * 2e-6))
    assert set(r.judges) == set(J.JUDGE_NAMES)
    assert r.judges["citation_integrity"].score == 1.0        # report builder stripped the fake citations
    assert r.judges["contradiction_recall"].score == 1.0
    out = tmp_path / "out"
    assert {p.name for p in out.iterdir()} == {"remote-work.report.md", "remote-work.report.json",
                                               "remote-work.eval.json", "summary.json", "summary.md"}
    summary = json.loads((out / "summary.json").read_text())
    assert summary["aggregate"]["succeeded"] == 1 and summary["rows"][0]["case"] == "remote-work"
    md = (out / "summary.md").read_text()
    assert "| remote-work |" in md and "max_hops=0" in md and "**mean (1/1 ok)**" in md


@pytest.mark.anyio
async def test_run_case_records_failure_instead_of_raising():
    class Broken:
        async def astream(self, *a, **k):
            raise RuntimeError("boom")
            yield  # pragma: no cover

    case = C.EvalCase(id="x", question="Will it fail?", expected_facets=["y"])
    r = await R.run_case(case, max_hops=0, budget_seconds=None, graph=Broken())
    assert not r.ok and "boom" in r.error and r.judges == {}
    md = R.summary_markdown([r], R.aggregate([r]))
    assert "failed: RuntimeError: boom" in md and "(0/1 ok)" in md


@pytest.mark.anyio
async def test_judge_existing_matches_case_by_question(tmp_path):
    p = tmp_path / "r.json"
    p.write_text(json.dumps(report()))
    case = C.EvalCase(id="remote-work", question="Does remote work raise productivity", expected_facets=["trials"],
                      expect_contradiction=True)
    runs = await R.judge_existing([p], [case])
    assert runs[0].case_id == "remote-work" and set(runs[0].judges) == set(J.JUDGE_NAMES)
    unmatched = await R.judge_existing([p], [])
    assert unmatched[0].case_id == "r" and set(unmatched[0].judges) == {"citation_integrity", "groundedness"}
