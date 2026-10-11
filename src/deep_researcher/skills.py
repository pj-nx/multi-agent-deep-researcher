"""Skills: short instruction blocks from ``skills.md`` appended to an agent's system prompt.

Each ``## name`` section of ``skills.md`` is one skill. :data:`AGENT_SKILLS` says which agents get
which skills; :func:`skill_prompt` renders them. ``SKILLS=off`` turns injection off (prompts then are
exactly the agents' own ``SYSTEM`` text). Adding a skill is: add a section, add its name to an agent.
"""
from __future__ import annotations

import functools
import os
import re
from pathlib import Path

SKILLS_FILE = Path(__file__).with_name("skills.md")

AGENT_SKILLS: dict[str, tuple[str, ...]] = {
    "planner": ("adaptive-research-planning", "academic-paper-search", "heavy-document-retrieval"),
    "validator": ("content-safety-guardrails",),
    "analysis": ("citation-verification", "heavy-document-retrieval", "content-safety-guardrails"),
    "insight": ("citation-verification", "adaptive-research-planning"),
    "critic": ("citation-verification", "adaptive-research-planning"),
    "report": ("citation-verification", "content-safety-guardrails"),
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


def skill_prompt(agent: str) -> str:
    """Text to append to ``agent``'s system prompt ('' when disabled or the agent has no skills)."""
    if not enabled():
        return ""
    skills = load_skills()
    names = AGENT_SKILLS.get(agent, ())
    missing = [n for n in names if n not in skills]
    if missing:
        raise KeyError(f"skills.md has no section(s) {missing} (needed by {agent!r})")
    if not names:
        return ""
    blocks = "\n\n".join(f"### Skill: {n}\n{skills[n]}" for n in names)
    return f"\n\n## Skills\nApply these skills in addition to the instructions above.\n\n{blocks}"
