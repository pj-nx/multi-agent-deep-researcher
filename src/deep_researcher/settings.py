"""Runtime settings: read/write the project's ``.env`` and apply changes to the running process.

Used by the UI's Settings tab. Secrets are never echoed back in full (see :func:`mask`); blank
inputs mean "leave unchanged" and the literal ``none`` clears a value.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

Kind = Literal["secret", "text", "choice", "bool"]


@dataclass(frozen=True)
class Field:
    key: str
    label: str
    kind: Kind = "text"
    group: str = "general"
    help: str = ""
    choices: tuple[str, ...] = ()


FIELDS: tuple[Field, ...] = (
    Field("OPENROUTER_API_KEY", "OpenRouter API key", "secret", "keys",
          "Required for every agent (openrouter.ai/keys)"),
    Field("TAVILY_API_KEY", "Tavily API key", "secret", "keys", "Enables web + reports sources"),
    Field("NEWSDATA_API_KEY", "newsdata.io API key", "secret", "keys",
          "News source (newsdata.io; otherwise GDELT, keyless)"),
    Field("SEMANTIC_SCHOLAR_API_KEY", "Semantic Scholar API key", "secret", "keys",
          "Optional; raises the rate limit (keyless calls often 429)"),
    Field("LANGSMITH_API_KEY", "LangSmith API key", "secret", "keys", "Tracing (optional)"),
    Field("LANGSMITH_TRACING", "LangSmith tracing", "bool", "keys", "true/false"),
    Field("LANGSMITH_PROJECT", "LangSmith project", "text", "keys"),
    Field("OPENROUTER_MODEL_HEAVY", "Heavy tier model", "choice", "models",
          "planner, analysis, insight, critic"),
    Field("OPENROUTER_MODEL_WORKER", "Worker tier model", "choice", "models", "retriever, report"),
    Field("OPENROUTER_MODEL_CHEAP", "Cheap tier model", "choice", "models", "validator, grader"),
    Field("OPENROUTER_FALLBACK_MODELS", "Fallback models", "text", "models",
          "comma-separated, tried in order when the primary is unavailable"),
    Field("OPENROUTER_REASONING_EFFORT", "Reasoning effort", "choice", "models",
          "caps hidden reasoning tokens", ("", "none", "minimal", "low", "medium", "high")),
    Field("EMBEDDINGS_PROVIDER", "Embeddings provider", "choice", "embeddings", "",
          ("local", "openrouter", "ollama", "openai")),
    Field("EMBEDDINGS_MODEL", "Embeddings model", "text", "embeddings",
          "blank = provider default; changing the model needs a new CHROMA_COLLECTION"),
    Field("EMBEDDINGS_DIMENSIONS", "Embedding dimensions", "text", "embeddings", "256, or 'native'"),
    Field("CHROMA_COLLECTION", "Chroma collection", "text", "embeddings",
          "one collection per embedding model"),
    Field("RETRIEVAL_LANGUAGES", "Allowed languages", "text", "run",
          "comma-separated ISO 639-1 codes, e.g. en or en,de,fr; articles in other languages are dropped"),
    Field("MAX_HOPS", "Default max hops", "text", "run"),
    Field("RUN_BUDGET_SECONDS", "Run budget (seconds)", "text", "run"),
)
FIELD_BY_KEY = {f.key: f for f in FIELDS}
CLEAR_SENTINEL = "none"


def env_path() -> Path:
    return Path(os.getenv("DEEP_RESEARCHER_ENV_FILE") or ".env")


def mask(value: str | None) -> str:
    """``sk-or-v1-abcd…wxyz`` style preview that never reveals the middle of a secret."""
    if not value:
        return "(not set)"
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:6]}…{value[-4:]} ({len(value)} chars)"


def current(key: str) -> str:
    return os.getenv(key) or ""


def display_value(field: Field) -> str:
    """What the form shows initially: masked for secrets, the live value otherwise."""
    return mask(current(field.key)) if field.kind == "secret" else current(field.key)


def read_env_file(path: Path | None = None) -> dict[str, str]:
    path = path or env_path()
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        m = re.match(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$", line)
        if m:
            out[m.group(1)] = _strip_value(m.group(2))
    return out


def _strip_value(raw: str) -> str:
    raw = raw.strip()
    if raw[:1] in {'"', "'"} and raw[-1:] == raw[:1] and len(raw) >= 2:
        return raw[1:-1]
    return re.split(r"\s+#", raw, maxsplit=1)[0].strip()


def write_env_updates(updates: dict[str, str | None], path: Path | None = None) -> Path:
    """Update ``KEY=`` lines in place (keeping order and comments), append new keys, ``None`` clears."""
    path = path or env_path()
    lines = path.read_text().splitlines() if path.exists() else []
    pending = dict(updates)
    out: list[str] = []
    for line in lines:
        m = re.match(r"^(\s*)(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)(\s*=)", line)
        key = m.group(2) if m else None
        if key in pending:
            value = pending.pop(key)
            out.append(f"{key}={_quote(value or '')}")
        else:
            out.append(line)
    if pending:
        out.append("")
        out.append("# added by the Settings page")
        out.extend(f"{k}={_quote(v or '')}" for k, v in pending.items())
    path.write_text("\n".join(out).rstrip("\n") + "\n")
    try:
        path.chmod(0o600)
    except OSError:
        pass
    return path


def _quote(value: str) -> str:
    return f'"{value}"' if any(c in value for c in " #'\"") else value


def apply_to_process(updates: dict[str, str | None]) -> None:
    """Set/unset env vars in this process and drop every cache that derives from them."""
    for key, value in updates.items():
        if value:
            os.environ[key] = value
        else:
            os.environ.pop(key, None)
    from deep_researcher import embeddings, graph, llm
    from deep_researcher.store import reset_store_cache

    llm.reset_cache()
    embeddings.reset_cache()
    reset_store_cache()
    graph.reset_graph()


def normalise(form: dict[str, Any]) -> dict[str, str | None]:
    """Form values -> env updates.

    Blank secret = unchanged (dropped); ``none`` = clear; bools -> true/false.
    """
    updates: dict[str, str | None] = {}
    for key, raw in form.items():
        field = FIELD_BY_KEY.get(key)
        if field is None:
            continue
        if field.kind == "bool":
            updates[key] = "true" if raw in (True, "true", "True", "1") else "false"
            continue
        value = str(raw if raw is not None else "").strip()
        if field.kind == "secret":
            if not value or value.startswith("(not set)") or "…" in value:
                continue                                   # unchanged
            updates[key] = None if value.lower() == CLEAR_SENTINEL else value
        else:
            if value.lower() == CLEAR_SENTINEL:
                updates[key] = None
            elif value != current(key):
                updates[key] = value or None
    return updates


def save(form: dict[str, Any], path: Path | None = None) -> tuple[dict[str, str | None], Path]:
    updates = normalise(form)
    target = write_env_updates(updates, path) if updates else (path or env_path())
    if updates:
        apply_to_process(updates)
    return updates, target


def summary() -> dict[str, Any]:
    """Effective configuration (secrets masked) for display."""
    from deep_researcher import embeddings, llm, tracing
    from deep_researcher.tools import language, registry

    return {
        "languages": language.allowed_languages(),
        "keys": {f.key: mask(current(f.key)) for f in FIELDS if f.kind == "secret"},
        "llm": llm.info(),
        "embeddings": embeddings.info(),
        "sources": {k: v["active"] for k, v in registry.info().items()},
        "tracing": tracing.info(),
        "env_file": str(env_path().resolve()),
    }
