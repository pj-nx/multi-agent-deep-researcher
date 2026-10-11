"""Research Planner: decomposes the question into 3-7 sub-questions with source hints (structured output)."""
from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from deep_researcher.agents.common import call_structured
from deep_researcher.agents.deps import Deps
from deep_researcher.guardrails import check_question
from deep_researcher.schemas import ResearchPlan, SubQuestion
from deep_researcher.skills import skill_prompt

log = logging.getLogger(__name__)

SYSTEM = """You are the research planner for a multi-agent deep-research system.
Decompose the user's question into 3-7 focused sub-questions that together answer it. For each:
- make it answerable by retrieval (papers, news, web, reports, wikipedia, api), not by opinion;
- pick the source_types most likely to hold evidence (papers for empirical findings, news for
  recent events, wikipedia for background/definitions, reports for industry/government analysis,
  web as a general fallback);
- add depends_on when a sub-question only makes sense after another is answered;
- give a one-line rationale;
- give 2-3 search_queries: SHORT keyword strings (3-8 words, no punctuation, no "or"/"and",
  no parentheses) as you would type into a search box, e.g. "four-day week trial productivity".
  Search APIs (arXiv, Wikipedia, news) return nothing useful for long sentences.
Prefer specificity over breadth. Use ids sq1, sq2, ... Keep the original question verbatim."""


async def plan_node(state: dict[str, Any], deps: Deps) -> dict[str, Any]:
    question = state["question"]
    degradations = await check_question(question, deps.llm)    # raises GuardrailViolation to reject
    llm = deps.llm("planner")
    messages = [SystemMessage(SYSTEM + skill_prompt("planner")), HumanMessage(question)]
    plan: ResearchPlan = await call_structured(llm, ResearchPlan, messages, role="planner", label="plan")
    plan = normalise_plan(plan, question)
    return {"plan": plan.model_dump(), "degradations": degradations,
            "notes": [f"planner proposed {len(plan.sub_questions)} sub-questions"]}


def normalise_plan(plan: ResearchPlan, question: str) -> ResearchPlan:
    """Stable ids (sq1..sqN), dangling depends_on removed, original question kept."""
    old_to_new: dict[str, str] = {}
    subs: list[SubQuestion] = []
    for i, sq in enumerate(plan.sub_questions, start=1):
        new_id = f"sq{i}"
        old_to_new[sq.id] = new_id
        subs.append(sq.model_copy(update={"id": new_id}))
    for sq in subs:
        sq.depends_on = [old_to_new[d] for d in sq.depends_on if d in old_to_new and old_to_new[d] != sq.id]
        sq.search_queries = [q.strip()[:120] for q in sq.search_queries if q.strip()][:5]
    return ResearchPlan(question=question, sub_questions=subs, notes=plan.notes)
