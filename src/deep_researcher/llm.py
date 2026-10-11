"""LLM factory: every agent gets its chat model from here, routed through OpenRouter.

OpenRouter exposes hundreds of models behind one OpenAI-compatible endpoint, so the
project needs a single key (``OPENROUTER_API_KEY``) and can mix vendors per role.

Transport: ``langchain-openai``'s ``ChatOpenAI`` with ``base_url`` set to OpenRouter. The dedicated
``langchain-openrouter`` package was tried first and dropped: its bundled SDK hung indefinitely on
``ainvoke`` (event loop idle, our timeout never propagated) while raw HTTP to the same endpoint
answered in ~2 s. OpenRouter-specific routing (fallback ``models``, ``provider`` preferences) goes
through ``extra_body``; attribution headers through ``default_headers``.

Structured output: use ``with_structured_output(schema, method="json_schema")``. OpenRouter's
Anthropic route rejected the ``function_calling`` method for Opus with a provider 400 (verified).

Roles map to cost tiers; each can be overridden by environment variable:

    OPENROUTER_MODEL_HEAVY   planner, analysis, insight, critic   (default anthropic/claude-opus-5.5)
    OPENROUTER_MODEL_WORKER  retriever, report                    (default anthropic/claude-sonnet-5)
    OPENROUTER_MODEL_CHEAP   validator                            (default anthropic/claude-haiku-4.5)
    OPENROUTER_MODEL_JUDGE   grader (eval judges)                 (default openai/gpt-5-mini)
    OPENROUTER_MODEL_<ROLE>  e.g. OPENROUTER_MODEL_INSIGHT=openai/gpt-5.1  (beats the tier)

Routing knobs (all optional):

    OPENROUTER_FALLBACK_MODELS   comma list tried in order if the primary model is down
    OPENROUTER_PROVIDER_ORDER    comma list of provider slugs to prefer (e.g. anthropic,google-vertex)
    OPENROUTER_ALLOW_FALLBACKS   true|false (default true)
    OPENROUTER_ZDR               true → only zero-data-retention endpoints
    OPENROUTER_SORT              price|throughput|latency
    OPENROUTER_REASONING_EFFORT  none|minimal|low|medium|high  (`reasoning.effort`; "none" disables)
    OPENROUTER_REASONING_EFFORT_<ROLE>  same values, one role only (beats the global value), e.g.
                                 OPENROUTER_REASONING_EFFORT_ANALYSIS=low  (unset roles keep the global value)
    OPENROUTER_APP_URL / OPENROUTER_APP_TITLE   attribution headers shown on openrouter.ai

Usage::

    from deep_researcher.llm import get_llm
    planner_llm = get_llm("planner")                     # cached per role
    grader_llm  = get_llm("grader", temperature=0.0)     # overrides are not cached
"""
from __future__ import annotations

import functools
import os
from typing import Any, Literal, get_args

Role = Literal["planner", "retriever", "validator", "analysis", "insight", "critic", "report", "grader"]
ROLES: tuple[str, ...] = get_args(Role)

PROVIDER = "openrouter"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_APP_URL = "https://github.com/pj-nx/multi-agent-deep-researcher"
DEFAULT_APP_TITLE = "Multi-Agent Deep Researcher"

TIER_DEFAULTS: dict[str, str] = {
    "heavy": "anthropic/claude-opus-5.5",
    "worker": "anthropic/claude-sonnet-5",
    "cheap": "anthropic/claude-haiku-4.5",
    # Eval judges get their own tier, from a different vendor than the pipeline: a judge that shares
    # the validator's model inherits its blind spots and tends to approve its own family's output.
    "judge": "openai/gpt-5-mini",
}
ROLE_TIER: dict[str, str] = {
    "planner": "heavy",
    "analysis": "heavy",
    "insight": "heavy",
    "critic": "heavy",
    "retriever": "worker",
    "report": "worker",
    "validator": "cheap",
    "grader": "judge",
}
# Deterministic where we parse structured output; a little warmth where we want hypotheses.
ROLE_TEMPERATURE: dict[str, float] = {
    "planner": 0.2, "retriever": 0.0, "validator": 0.0, "analysis": 0.1,
    "insight": 0.6, "critic": 0.2, "report": 0.3, "grader": 0.0,
}
# Generous on purpose: on OpenRouter a model's hidden reasoning tokens count against max_tokens, and a
# structured-output call that runs out of budget fails outright (LengthFinishReasonError) rather than
# degrading. Observed: the critic spent 3.3k reasoning tokens before writing any JSON.
ROLE_MAX_TOKENS: dict[str, int] = {
    "planner": 8_000, "retriever": 4_000, "validator": 4_000, "analysis": 16_000,
    "insight": 16_000, "critic": 12_000, "report": 24_000, "grader": 2_000,
}

_TRUTHY = {"1", "true", "yes", "on"}


class LLMConfigError(RuntimeError):
    pass


def _env_bool(name: str) -> bool | None:
    v = os.getenv(name)
    return None if v is None or v == "" else v.strip().lower() in _TRUTHY


def _env_list(name: str) -> list[str]:
    return [x.strip() for x in os.getenv(name, "").split(",") if x.strip()]


def api_key() -> str | None:
    return os.getenv("OPENROUTER_API_KEY") or None


def enabled() -> bool:
    return api_key() is not None


def base_url() -> str:
    return os.getenv("OPENROUTER_API_BASE", DEFAULT_BASE_URL)


def model_for(role: str) -> str:
    """Resolve the OpenRouter model id for a role (role override > tier override > default)."""
    if role not in ROLE_TIER:
        raise LLMConfigError(f"unknown role {role!r}; expected one of {ROLES}")
    tier = ROLE_TIER[role]
    return (
        os.getenv(f"OPENROUTER_MODEL_{role.upper()}")
        or os.getenv(f"OPENROUTER_MODEL_{tier.upper()}")
        or TIER_DEFAULTS[tier]
    )


def provider_preferences() -> dict[str, Any] | None:
    """OpenRouter ``provider`` routing object built from env; None when nothing is set."""
    prefs: dict[str, Any] = {}
    if order := _env_list("OPENROUTER_PROVIDER_ORDER"):
        prefs["order"] = order
    if (allow := _env_bool("OPENROUTER_ALLOW_FALLBACKS")) is not None:
        prefs["allow_fallbacks"] = allow
    if _env_bool("OPENROUTER_ZDR"):
        prefs["zdr"] = True
    if sort := os.getenv("OPENROUTER_SORT"):
        prefs["sort"] = sort
    return prefs or None


def reasoning_effort(role: str | None = None) -> str | None:
    """Effort for ``role``: its ``OPENROUTER_REASONING_EFFORT_<ROLE>``, else the global value, else None."""
    name = "OPENROUTER_REASONING_EFFORT"
    if role and os.getenv(f"{name}_{role.upper()}"):
        name = f"{name}_{role.upper()}"
    v = (os.getenv(name) or "").strip().lower()
    if not v:
        return None
    if v not in {"none", "minimal", "low", "medium", "high"}:
        raise LLMConfigError(f"{name}={v!r}; expected none|minimal|low|medium|high")
    return v


def fallback_models() -> list[str]:
    return _env_list("OPENROUTER_FALLBACK_MODELS")


STRUCTURED_OUTPUT_METHOD = "json_schema"


def max_tokens(role: str) -> int:
    """Completion budget for ``role``; ``OPENROUTER_MAX_TOKENS`` caps every role (e.g. 300 for low credit)."""
    default = ROLE_MAX_TOKENS[role]
    raw = os.getenv("OPENROUTER_MAX_TOKENS", "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        raise LLMConfigError(f"OPENROUTER_MAX_TOKENS={raw!r}; expected an integer") from None


def attribution_headers() -> dict[str, str]:
    return {
        "HTTP-Referer": os.getenv("OPENROUTER_APP_URL", DEFAULT_APP_URL),
        "X-OpenRouter-Title": os.getenv("OPENROUTER_APP_TITLE", DEFAULT_APP_TITLE),
    }


def build_kwargs(role: str, **overrides: Any) -> dict[str, Any]:
    """Keyword arguments for ChatOpenAI (pointed at OpenRouter). Pure; no network, no langchain import."""
    if role not in ROLE_TIER:
        raise LLMConfigError(f"unknown role {role!r}; expected one of {ROLES}")
    model = overrides.pop("model", None) or model_for(role)
    extra_body: dict[str, Any] = dict(overrides.pop("extra_body", {}) or {})
    if (fb := fallback_models()) and "models" not in extra_body:
        extra_body["models"] = [model, *[m for m in fb if m != model]]
    if (prefs := provider_preferences()) and "provider" not in extra_body:
        extra_body["provider"] = prefs
    if (effort := reasoning_effort(role)) and "reasoning" not in extra_body:
        extra_body["reasoning"] = {"enabled": False} if effort == "none" else {"effort": effort}
    kwargs: dict[str, Any] = {
        "model": model,
        "temperature": ROLE_TEMPERATURE[role],
        "max_tokens": max_tokens(role),
        "max_retries": 2,
        "timeout": 120,
        "default_headers": attribution_headers(),
        "tags": [f"role:{role}", "openrouter"],
        "metadata": {"role": role, "model": model, "provider": PROVIDER},
    }
    if extra_body:
        kwargs["extra_body"] = extra_body
    kwargs.update(overrides)
    return kwargs


def make_llm(role: str, **overrides: Any):
    """Construct a fresh ChatOpenAI bound to OpenRouter for ``role``. Raises LLMConfigError without a key."""
    if not enabled():
        raise LLMConfigError(
            "OPENROUTER_API_KEY is not set. Get a key at https://openrouter.ai/keys and add it to .env "
            "(or the deployment's secrets)."
        )
    from langchain_openai import ChatOpenAI

    kwargs = build_kwargs(role, **overrides)
    return ChatOpenAI(api_key=api_key(), base_url=base_url(), **kwargs)


def structured(llm: Any, schema: Any, **kwargs: Any):
    """``with_structured_output`` with the method OpenRouter's Anthropic route accepts."""
    kwargs.setdefault("method", STRUCTURED_OUTPUT_METHOD)
    return llm.with_structured_output(schema, **kwargs)


@functools.cache
def _cached(role: str):
    return make_llm(role)


def get_llm(role: str, **overrides: Any):
    """Chat model for a role. Cached when no overrides are given."""
    return _cached(role) if not overrides else make_llm(role, **overrides)


def reset_cache() -> None:
    _cached.cache_clear()


def info() -> dict[str, Any]:
    """Health payload: provider, whether a key is present, resolved model per role. Never the key."""
    return {
        "provider": PROVIDER,
        "configured": enabled(),
        "base_url": base_url(),
        "models": {role: model_for(role) for role in ROLES},
        "fallback_models": fallback_models(),
        "provider_preferences": provider_preferences(),
        "reasoning_effort": _safe_reasoning_effort(),
    }


def _safe_reasoning_effort() -> str | None:
    try:
        return reasoning_effort()
    except LLMConfigError:
        return None


def _shape_models(rows: list[dict[str, Any]], prefix: str) -> list[dict[str, Any]]:
    return sorted(
        (
            {"id": m["id"], "context_length": m.get("context_length"),
             "prompt_price": m["pricing"].get("prompt"), "completion_price": m["pricing"].get("completion")}
            for m in rows if m["id"].startswith(prefix)
        ),
        key=lambda r: r["id"],
    )


def list_models(prefix: str = "", *, timeout: float = 15.0) -> list[dict[str, Any]]:
    """Live model catalogue from OpenRouter (public endpoint, no key needed).

    Blocking network call for CLI / notebook use. Inside async code use :func:`alist_models`.
    """
    import httpx

    resp = httpx.get(f"{base_url()}/models", timeout=timeout)
    resp.raise_for_status()
    return _shape_models(resp.json()["data"], prefix)


async def alist_models(prefix: str = "", *, timeout: float = 15.0) -> list[dict[str, Any]]:
    """Async variant of :func:`list_models` for use in routes and graph nodes."""
    import httpx

    async with httpx.AsyncClient(timeout=timeout) as c:
        resp = await c.get(f"{base_url()}/models")
    resp.raise_for_status()
    return _shape_models(resp.json()["data"], prefix)
