"""Fakes shared by the graph and UI tests: schema-dispatching LLM, canned tools, in-memory store deps."""
import re

from langchain_core.embeddings import DeterministicFakeEmbedding

from deep_researcher.agents.deps import Deps
from deep_researcher.schemas import (
    AnalysisOutput,
    Claim,
    ContradictionOut,
    CriticGapOut,
    CritiqueOut,
    GapOut,
    Insight,
    InsightOutput,
    ReasoningStep,
    ReportOutput,
    ResearchPlan,
    SourceAssessment,
    SourceAssessments,
    SubQuestion,
)
from deep_researcher.store import EvidenceStore
from deep_researcher.tools.base import SearchResult


class FakeStructured:
    def __init__(self, fn):
        self.fn = fn

    async def ainvoke(self, messages):
        return self.fn(messages)


class FakeLLM:
    """Dispatches on the structured-output schema; records every prompt it saw."""

    def __init__(self, plan, *, gap_rounds=1, contradiction=True, verdicts=("accept",)):
        self.plan, self.gap_rounds, self.contradiction = plan, gap_rounds, contradiction
        self.verdicts = list(verdicts)          # consumed one per critic round; last one repeats
        self.prompts = []
        self.analysis_calls = 0
        self.insight_calls = 0
        self.critic_calls = 0

    def with_structured_output(self, schema, **kwargs):
        assert kwargs.get("method") == "json_schema"
        if kwargs.get("include_raw"):
            class Raw:
                usage_metadata = {"input_tokens": 1000, "output_tokens": 200, "total_tokens": 1200}
            return FakeStructured(lambda msgs: {"raw": Raw(), "parsed": self._answer(schema, msgs),
                                                "parsing_error": None})
        return FakeStructured(lambda msgs: self._answer(schema, msgs))

    def _answer(self, schema, msgs):
        prompt = msgs[-1].content
        self.prompts.append((schema.__name__, prompt))
        if schema is ResearchPlan:
            return self.plan
        if schema is SourceAssessments:
            ids = re.findall(r'<source id="([^"]+)"', prompt)
            items = []
            for i, eid in enumerate(ids):
                off_topic = "OFFTOPIC" in prompt.split(f'id="{eid}"', 1)[1].split("</source>", 1)[0]
                items.append(SourceAssessment(evidence_id=eid, relevance=0.1 if off_topic else 0.9,
                                              credibility=0.85 if i == 0 else 0.5, keep=not off_topic,
                                              issues=["off-topic"] if off_topic else []))
            return SourceAssessments(items=items)
        if schema is AnalysisOutput:
            self.analysis_calls += 1
            ids = re.findall(r"\[(E-[0-9a-f]{10})\]", prompt)
            sq_id = re.search(r"Sub-question \((\w+)\)", prompt).group(1)
            claims = [Claim(text=f"finding from {ids[0]}", evidence_ids=[ids[0]], confidence=0.8),
                      Claim(text="hallucinated", evidence_ids=["E-deadbeef00"], confidence=0.9)]  # must be dropped
            contradictions = []
            if self.contradiction and "RAISES" in prompt and "LOWERS" in prompt:
                a = next(i for i in ids if i in prompt.split("RAISES")[0][-400:]) if False else ids[0]
                b = ids[1] if len(ids) > 1 else ids[0]
                contradictions.append(ContradictionOut(claim_a="productivity rises", claim_b="productivity falls",
                                                       evidence_ids_a=[a], evidence_ids_b=[b],
                                                       explanation="different populations"))
            gaps = []
            hop_marker = prompt.count("gap")  # gap-derived sub-questions never raise further gaps
            if sq_id == "sq1" and self.gap_rounds > 0 and not sq_id.startswith("gap"):
                self.gap_rounds -= 1
                n = self.analysis_calls
                q = "What do long-term follow-ups show?" if self.gap_rounds >= 0 and n <= 2 else f"Open question {n}?"
                gaps.append(GapOut(question=q, source_types=["papers"],
                                   search_queries=["long-term follow-up"], reason="only short trials"))
            _ = hop_marker
            return AnalysisOutput(summary=f"Summary for {sq_id} citing [{ids[0]}].", claims=claims,
                                  contradictions=contradictions, gaps=gaps, confidence=0.7)
        if schema is InsightOutput:
            self.insight_calls += 1
            ids = re.findall(r"\[(E-[0-9a-f]{10})\]", prompt)
            if not ids:
                return InsightOutput(insights=[])
            revised = "Critique of the previous insights" in prompt
            return InsightOutput(insights=[
                Insight(hypothesis=("Revised: " if revised else "") + "Effects depend on sector",
                        reasoning=[ReasoningStep(step="trials in knowledge work show gains", evidence_ids=[ids[0]]),
                                   ReasoningStep(step="bogus citation", evidence_ids=["E-deadbeef00"]),
                                   ReasoningStep(step="therefore sector moderates", evidence_ids=[])],
                        confidence=0.6, falsifier="uniform gains across sectors"),
                Insight(hypothesis="Ungrounded speculation", reasoning=[ReasoningStep(step="x", evidence_ids=["E-deadbeef00"])],
                        confidence=0.9),
            ])
        if schema is CritiqueOut:
            self.critic_calls += 1
            verdict = self.verdicts[min(self.critic_calls - 1, len(self.verdicts) - 1)]
            follow_up = [CriticGapOut(question="What do manufacturing-sector trials show?", source_types=["papers"],
                                      search_queries=["manufacturing four-day week"], parent_sub_question_id="sq1")] \
                if verdict == "need_more_evidence" else []
            return CritiqueOut(verdict=verdict, issues=[f"issue from round {self.critic_calls}"] if verdict != "accept" else [],
                               follow_up=follow_up, strengths=["cites evidence"])
        if schema is ReportOutput:
            ids = re.findall(r"\[(E-[0-9a-f]{10})\]", prompt)
            if not ids:
                return ReportOutput(title="No evidence", executive_summary="Nothing was retrieved.",
                                    limitations=["no evidence"])
            return ReportOutput(title="Remote work and productivity: what the evidence says",
                                executive_summary=f"Overall the evidence is mixed [{ids[0]}]. A fake citation [E-deadbeef00] here.",
                                key_findings=[f"Knowledge work shows gains [{ids[0]}]", "Unsupported finding [E-deadbeef00]"],
                                limitations=["few sources"], open_questions=["long-run effects?"])
        raise AssertionError(f"unexpected schema {schema}")


class FakeTool:
    def __init__(self, name, source_type, n=2):
        self.name, self.source_type, self.n = name, source_type, n
        self.calls = []

    def _text(self, i, query):
        if self.source_type == "papers":
            verdict = "RAISES" if i % 2 == 0 else "LOWERS"
            return f"Evidence {i} about {query}: the trial {verdict} productivity. " * 4
        return f"Evidence {i} about {query}. " * 5

    async def search(self, query, limit=5):
        self.calls.append(query)
        out = [SearchResult(title=f"{self.name} {i} on {query[:12]}",
                            url=f"https://{self.name}.test/{abs(hash(query)) % 997}/{i}",
                            text=self._text(i, query), source_type=self.source_type,
                            published_at=f"2025-01-0{i + 1}") for i in range(self.n)]
        if self.source_type == "wikipedia":
            out.append(SearchResult(title="Tyler Joseph", url=f"https://{self.name}.test/offtopic/{abs(hash(query)) % 97}",
                                    text=f"OFFTOPIC musician biography {query}", source_type="wikipedia"))
        return out


def _plan():
    return ResearchPlan(question="Does remote work raise productivity?", sub_questions=[
        SubQuestion(id="a", question="What do randomised trials find about remote work productivity?",
                    source_types=["papers"], search_queries=["remote work productivity trial", "telework output RCT"]),
        SubQuestion(id="b", question="What have large employers announced about office returns?", source_types=["news"]),
        SubQuestion(id="c", question="Background on remote work definitions", source_types=["wikipedia"], depends_on=["a"]),
    ])


def make_deps(tmp_path):
    store = EvidenceStore(DeterministicFakeEmbedding(size=32), embedding_id="fake:32", path=str(tmp_path / "chroma"))
    tools = {"papers": FakeTool("arxiv", "papers"), "news": FakeTool("gdelt", "news", n=1),
             "wikipedia": FakeTool("wikipedia", "wikipedia", n=1)}

    def tools_for(source_types):
        return [tools[s] for s in source_types if s in tools]

    fake_llm = FakeLLM(_plan())
    d = Deps(llm=lambda role: fake_llm, store=lambda: store, tools_for=tools_for,
             embedding_id=lambda: "fake:32", model_names=lambda: "fake-model: all roles")
    d.fake_tools = tools
    d.fake_store = store
    d.fake_llm = fake_llm
    return d
