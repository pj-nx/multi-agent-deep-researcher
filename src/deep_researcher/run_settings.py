"""Per-run settings a client may send with a research request.

Hosted deployments (Vercel, Lambda) have no writable ``.env``, so the hosted UI cannot change the
server's configuration the way the Gradio Settings tab does. Instead each request may carry a
:class:`RunSettings` object: model per tier or role, reasoning effort, fallback models and a run
budget. The API turns it into a :class:`Deps` for that run only (:func:`make_deps`); nothing is
persisted and the server's env stays the default for everyone else.

API keys are deliberately not part of this: they stay on the server.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from deep_researcher.agents.deps import Deps
from deep_researcher.llm import ROLE_TIER, ROLES, TIER_DEFAULTS, model_for

TIERS: tuple[str, ...] = tuple(TIER_DEFAULTS)
MODEL_ID = r"^[a-z0-9][a-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._:-]*$"   # vendor/model[:variant]
Effort = Literal["none", "minimal", "low", "medium", "high"]


class RunSettings(BaseModel):
    """Overrides for one run. Keys of ``models`` are tiers (heavy/worker/cheap) or roles (planner, ...)."""

    models: dict[str, str] = Field(default_factory=dict, max_length=16)
    reasoning_effort: Effort | None = None
    fallback_models: list[str] = Field(default_factory=list, max_length=5)
    budget_seconds: int | None = Field(default=None, ge=0, le=3600)

    @field_validator("models")
    @classmethod
    def _check_models(cls, v: dict[str, str]) -> dict[str, str]:
        import re

        for key, model in v.items():
            if key not in TIERS and key not in ROLES:
                raise ValueError(f"unknown tier/role {key!r}; expected one of {TIERS + ROLES}")
            if not re.match(MODEL_ID, model):
                raise ValueError(f"{key}: {model!r} is not an OpenRouter model id (vendor/model)")
        return v

    @field_validator("fallback_models")
    @classmethod
    def _check_fallbacks(cls, v: list[str]) -> list[str]:
        import re

        for model in v:
            if not re.match(MODEL_ID, model):
                raise ValueError(f"{model!r} is not an OpenRouter model id (vendor/model)")
        return v

    def is_empty(self) -> bool:
        return not (self.models or self.reasoning_effort or self.fallback_models)

    def model_for(self, role: str) -> str:
        """Role override > tier override > the server's configuration."""
        return self.models.get(role) or self.models.get(ROLE_TIER[role], "") or model_for(role)

    def resolved_models(self) -> dict[str, str]:
        return {role: self.model_for(role) for role in ROLES}

    def summary(self) -> dict[str, Any]:
        """What the run will use, for logs and the activity stream (no secrets)."""
        out: dict[str, Any] = {"models": self.resolved_models()}
        if self.reasoning_effort:
            out["reasoning_effort"] = self.reasoning_effort
        if self.fallback_models:
            out["fallback_models"] = self.fallback_models
        if self.budget_seconds is not None:
            out["budget_seconds"] = self.budget_seconds
        return out


def effective_budget(settings: RunSettings | None, server_budget: int) -> int | None:
    """The client may shorten the run but never exceed the server's budget (``0`` = no server cap)."""
    if settings is None or settings.budget_seconds is None:
        return server_budget or None
    if server_budget:
        return min(settings.budget_seconds, server_budget) or None
    return settings.budget_seconds or None


def _llm_overrides(settings: RunSettings, role: str) -> dict[str, Any]:
    model = settings.model_for(role)
    extra_body: dict[str, Any] = {}
    if settings.fallback_models:
        extra_body["models"] = [model, *[m for m in settings.fallback_models if m != model]]
    if settings.reasoning_effort:
        effort = settings.reasoning_effort
        extra_body["reasoning"] = {"enabled": False} if effort == "none" else {"effort": effort}
    return {"model": model, "extra_body": extra_body}


def make_deps(settings: RunSettings | None) -> Deps | None:
    """A ``Deps`` whose models follow ``settings``; ``None`` when nothing is overridden (shared graph)."""
    if settings is None or settings.is_empty():
        return None
    from deep_researcher.llm import make_llm

    cache: dict[str, Any] = {}

    def llm(role: str) -> Any:
        if role not in cache:
            cache[role] = make_llm(role, **_llm_overrides(settings, role))
        return cache[role]

    def model_names() -> str:
        by_model: dict[str, list[str]] = {}
        for role, model in settings.resolved_models().items():
            by_model.setdefault(model, []).append(role)
        return "; ".join(f"{m}: {', '.join(r)}" for m, r in by_model.items())

    return Deps(llm=llm, model_names=model_names)
