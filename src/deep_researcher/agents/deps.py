"""Injectable dependencies for graph nodes, so tests can swap LLMs, tools and the store."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


def _default_llm(role: str) -> Any:
    from deep_researcher.llm import get_llm

    return get_llm(role)


def _default_store() -> Any:
    from deep_researcher.store import get_store

    return get_store()


def _default_tools_for(source_types: list[str]) -> list[Any]:
    from deep_researcher.tools import tools_for

    return tools_for(source_types)  # type: ignore[arg-type]


@dataclass
class Deps:
    llm: Callable[[str], Any] = field(default=_default_llm)
    store: Callable[[], Any] = field(default=_default_store)
    tools_for: Callable[[list[str]], list[Any]] = field(default=_default_tools_for)
    fetch_full_text: bool = False          # enrich short results (GDELT titles, PDFs) with fetched text
    per_tool_limit: int = 5
    max_per_sub_question: int = 12
    tool_timeout_s: float = 45.0           # one slow/rate-limited API must not hold a branch hostage
    min_relevance: float = 0.2             # share of sub-question keywords a result must mention
