"""Skills: short instruction blocks from ``skills.md`` appended to an agent's system prompt.

Each ``## name`` section of ``skills.md`` is one skill. :data:`AGENT_SKILLS` says which agents get which
skills; :func:`skill_prompt` renders them at call time. ``SKILLS=off`` turns injection off (prompts are then
exactly the agents' own ``SYSTEM`` text). Adding a skill is: add a section, add its name to an agent.
:func:`validate_skills` checks the file against the mapping (run at graph build and in tests); at call time a
broken file never raises, it degrades (see ``skill_prompt``).
"""
from __future__ import annotations

import functools
import logging
import os
import re
from pathlib import Path

log = logging.getLogger(__name__)
SKILLS_FILE = Path(__file__).with_name("skills.md")

AGENT_SKILLS: dict[str, tuple[str, ...]] = {
    "planner": ("adaptive-research-planning", "autonomous-decision-making", "search-quality",
                "academic-paper-search", "heavy-document-retrieval", "tool-call-efficiency",
                "research-depth-completeness"),
    "validator": ("content-safety-guardrails", "search-quality"),
    "analysis": ("citation-verification", "factual-correctness", "hallucination-prevention",
                 "context-window-management", "content-safety-guardrails"),
    "insight": ("citation-verification", "factual-correctness", "hallucination-prevention",
                "research-depth-completeness", "adaptive-research-planning"),
    "critic": ("citation-verification", "factual-correctness", "research-depth-completeness",
               "autonomous-decision-making", "tool-call-efficiency", "adaptive-research-planning"),
    "report": ("citation-verification", "factual-correctness", "hallucination-prevention",
               "research-depth-completeness", "content-safety-guardrails"),
    "guardrails": ("content-safety-guardrails",),
}


@functools.cache
def load_skills(path: Path = SKILLS_FILE) -> dict[str, str]:
    """``{skill name: instruction text}`` parsed from the ``## name`` sections of the file."""
    text = path.read_text(encoding="utf-8")
    parts = re.split(r"^## (\S+)\s*$", text, flags=re.MULTILINE)
    return {name: body.strip() for name, body in zip(parts[1::2], parts[2::2], strict=True)}


def enabled() -> bool:
    return (os.getenv("SKILLS") or "on").strip().lower() not in {"off", "false", "0", "no"}


def validate_skills(path: Path = SKILLS_FILE) -> list[str]:
    """Problems with the skills setup (empty = fine): unreadable file, missing or empty or unused skills."""
    try:
        loaded = load_skills(path)
    except (OSError, ValueError) as exc:
        return [f"cannot read {path.name}: {exc}"]
    problems = [f"{agent}: skill {n!r} is not defined in skills.md"
                for agent, names in AGENT_SKILLS.items() for n in names if n not in loaded]
    used = {n for names in AGENT_SKILLS.values() for n in names}
    problems += [f"skill {n!r} is defined but no agent uses it" for n in loaded if n not in used]
    problems += [f"skill {n!r} is empty" for n, body in loaded.items() if not body.strip()]
    return problems


def skill_prompt(agent: str) -> str:
    """Text to append to ``agent``'s system prompt ('' when disabled or the agent has no skills).

    Runs inside the agent's LLM call, so it never raises: a skill that cannot be loaded is left out and
    recorded as a run-health degradation (the agent still works from its own prompt).
    """
    if not enabled():
        return ""
    names = AGENT_SKILLS.get(agent, ())
    if not names:
        return ""
    try:
        skills = load_skills()
    except (OSError, ValueError) as exc:
        _report(agent, "file_unreadable", f"{type(exc).__name__}: {exc}")
        return ""
    present = [n for n in names if skills.get(n)]
    if len(present) < len(names):
        missing = [n for n in names if n not in present]
        _report(agent, "skill_missing", f"{agent} is missing skill(s) {missing}")
    if not present:
        return ""
    blocks = "\n\n".join(f"### Skill: {n}\n{skills[n]}" for n in present)
    return f"\n\n## Skills\nApply these skills in addition to the instructions above.\n\n{blocks}"


@functools.cache
def _report_once(agent: str, kind: str) -> None:
    from deep_researcher.health import degrade

    degrade("skills", kind, f"{agent}: prompt used without some skills", "warn")


def _report(agent: str, kind: str, detail: str) -> None:
    log.error("skills: %s (%s)", detail, kind)
    _report_once(agent, kind)
