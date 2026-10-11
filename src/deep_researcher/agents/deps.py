# Copyright (c) 2026 C10 Hackathon G1
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


def _default_embedding_id() -> str:
    from deep_researcher.embeddings import embedding_id

    return embedding_id()


def _default_model_names() -> str:
    from deep_researcher.llm import ROLES, model_for

    pairs = {role: model_for(role) for role in ROLES}
    by_model: dict[str, list[str]] = {}
    for role, model in pairs.items():
        by_model.setdefault(model, []).append(role)
    return "; ".join(f"{m}: {', '.join(r)}" for m, r in by_model.items())


def _env_int(name: str, default: int) -> int:
    import os

    try:
        return max(1, int(os.getenv(name) or default))
    except ValueError:
        return default


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
    # parallel model calls inside validator / analysis (LLM_CONCURRENCY)
    llm_concurrency: int = field(default_factory=lambda: _env_int("LLM_CONCURRENCY", 8))
    analysis_k: int = 12                   # chunks shown to the analysis model per sub-question
    analysis_chunk_chars: int = 1600
    max_gap_branches: int = 4              # follow-up sub-questions pursued per hop (after de-duplication)
    max_revisions: int = 2                 # critic rounds that may send insights back
    embedding_id: Callable[[], str] = field(default=_default_embedding_id)
    model_names: Callable[[], str] = field(default=_default_model_names)
