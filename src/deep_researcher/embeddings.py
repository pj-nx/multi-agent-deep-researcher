"""Embedding model factory: local by default, hosted when you ask for it.

The evidence store needs one embedding model per collection. This module picks it from env so
local development needs no API key at all, while Vercel can point at a hosted endpoint.

    EMBEDDINGS_PROVIDER   local | openrouter | ollama | openai   (default: local)
    EMBEDDINGS_MODEL      provider-specific model id              (defaults below)
    EMBEDDINGS_DEVICE     local only: cpu | cuda | mps             (default: cpu)
    EMBEDDINGS_BASE_URL   ollama: server url (default http://localhost:11434)
                          openai: any OpenAI-compatible endpoint (OpenAI, vLLM, LM Studio)
                          openrouter: fixed to https://openrouter.ai/api/v1 unless overridden
    EMBEDDINGS_API_KEY    openai / openrouter; falls back to OPENAI_API_KEY / OPENROUTER_API_KEY
    EMBEDDINGS_DIMENSIONS openai only: request truncated vectors (text-embedding-3-*)

Defaults per provider (the first two are the SAME model, so a collection built locally can be
queried from a hosted deployment and vice versa):
    local       BAAI/bge-base-en-v1.5    768-d, ~440 MB, CPU           (uv sync --extra embeddings-local)
    openrouter  baai/bge-base-en-v1.5    768-d, $0.005 / M tokens       (uv sync --extra embeddings-openai)
    ollama      nomic-embed-text         768-d                          (uv sync --extra embeddings-ollama)
    openai      text-embedding-3-small   1536-d                         (uv sync --extra embeddings-openai)

``embedding_id()`` is the fingerprint the evidence store writes into collection metadata so a
collection is never queried with a different model than the one that built it. It is derived
from the *model*, not the provider, so ``local`` and ``openrouter`` serving the same weights
share a collection; the store's dimension check catches anything the name does not.
"""
from __future__ import annotations

import functools
import os
from dataclasses import asdict, dataclass
from typing import Any

PROVIDERS: tuple[str, ...] = ("local", "openrouter", "ollama", "openai")

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

DEFAULT_MODELS: dict[str, str] = {
    "local": "BAAI/bge-base-en-v1.5",
    "openrouter": "baai/bge-base-en-v1.5",
    "ollama": "nomic-embed-text",
    "openai": "text-embedding-3-small",
}
DEFAULT_BASE_URLS: dict[str, str | None] = {
    "local": None,
    "openrouter": OPENROUTER_BASE_URL,
    "ollama": "http://localhost:11434",
    "openai": None,  # the SDK default (api.openai.com) unless overridden
}
DEFAULT_KEY_ENV: dict[str, str | None] = {
    "local": None,
    "openrouter": "OPENROUTER_API_KEY",
    "ollama": None,
    "openai": "OPENAI_API_KEY",
}
INSTALL_HINT: dict[str, str] = {
    "local": "uv sync --extra embeddings-local",
    "openrouter": "uv sync --extra embeddings-openai",
    "ollama": "uv sync --extra embeddings-ollama   (and `ollama pull <model>`)",
    "openai": "uv sync --extra embeddings-openai",
}
# Providers that speak the OpenAI embeddings wire format.
_OPENAI_COMPATIBLE = {"openai", "openrouter"}


class EmbeddingsConfigError(RuntimeError):
    pass


@dataclass(frozen=True)
class EmbeddingSpec:
    provider: str
    model: str
    device: str = "cpu"
    base_url: str | None = None
    api_key_env: str | None = None
    dimensions: int | None = None
    normalize: bool = True

    @property
    def embedding_id(self) -> str:
        """Stable fingerprint: canonical model name plus requested dimensions.

        Provider is deliberately *not* part of it, so ``local`` + ``BAAI/bge-base-en-v1.5`` and
        ``openrouter`` + ``baai/bge-base-en-v1.5`` resolve to the same id (same weights).
        """
        return canonical_model(self.model, self.dimensions)

    def public(self) -> dict[str, Any]:
        d = asdict(self)
        d["embedding_id"] = self.embedding_id
        return d


def canonical_model(model: str, dimensions: int | None = None) -> str:
    """Lower-case, strip a leading ``sentence-transformers/`` org and any OpenRouter ``:variant``."""
    m = model.strip().lower().split(":", 1)[0]
    if m.startswith("sentence-transformers/"):
        m = m[len("sentence-transformers/"):]
    return f"{m}@{dimensions}" if dimensions else m


def spec() -> EmbeddingSpec:
    """Resolve the embedding configuration from the environment (pure; no imports, no network)."""
    provider = (os.getenv("EMBEDDINGS_PROVIDER") or "local").strip().lower()
    if provider not in PROVIDERS:
        raise EmbeddingsConfigError(f"EMBEDDINGS_PROVIDER={provider!r}; expected one of {PROVIDERS}")
    dims_raw = (os.getenv("EMBEDDINGS_DIMENSIONS") or "").strip()
    dims = int(dims_raw) if dims_raw.isdigit() else None
    if dims is not None and provider != "openai":
        dims = None  # only OpenAI's own text-embedding-3-* accept a dimensions parameter
    api_key_env = None
    if provider in _OPENAI_COMPATIBLE:
        api_key_env = "EMBEDDINGS_API_KEY" if os.getenv("EMBEDDINGS_API_KEY") else DEFAULT_KEY_ENV[provider]
    return EmbeddingSpec(
        provider=provider,
        model=os.getenv("EMBEDDINGS_MODEL") or DEFAULT_MODELS[provider],
        device=(os.getenv("EMBEDDINGS_DEVICE") or "cpu").strip().lower(),
        base_url=os.getenv("EMBEDDINGS_BASE_URL") or DEFAULT_BASE_URLS[provider],
        api_key_env=api_key_env,
        dimensions=dims,
    )


def embedding_id() -> str:
    return spec().embedding_id


def make_embeddings(s: EmbeddingSpec | None = None):
    """Construct a LangChain ``Embeddings`` for the spec. Imports the provider lazily."""
    s = s or spec()
    try:
        if s.provider == "local":
            from langchain_huggingface import HuggingFaceEmbeddings

            return HuggingFaceEmbeddings(
                model_name=s.model,
                model_kwargs={"device": s.device},
                encode_kwargs={"normalize_embeddings": s.normalize},
            )
        if s.provider == "ollama":
            from langchain_ollama import OllamaEmbeddings

            return OllamaEmbeddings(model=s.model, base_url=s.base_url)
        if s.provider in _OPENAI_COMPATIBLE:
            from langchain_openai import OpenAIEmbeddings

            key = os.getenv(s.api_key_env or "")
            if not key:
                raise EmbeddingsConfigError(
                    f"EMBEDDINGS_PROVIDER={s.provider} needs EMBEDDINGS_API_KEY or "
                    f"{DEFAULT_KEY_ENV[s.provider]}; or use EMBEDDINGS_PROVIDER=local to run without any key."
                )
            kwargs: dict[str, Any] = {"model": s.model, "api_key": key}
            if s.base_url:
                kwargs["base_url"] = s.base_url
            if s.dimensions:
                kwargs["dimensions"] = s.dimensions
            if s.provider == "openrouter" or (s.base_url and "api.openai.com" not in s.base_url):
                # Third-party endpoints want raw strings, not tiktoken token arrays, and
                # their model ids are unknown to tiktoken.
                kwargs["check_embedding_ctx_length"] = False
                kwargs["tiktoken_enabled"] = False
            if s.provider == "openrouter":
                kwargs["default_headers"] = {
                    "HTTP-Referer": os.getenv("OPENROUTER_APP_URL", "https://github.com/pj-nx/multi-agent-deep-researcher"),
                    "X-OpenRouter-Title": os.getenv("OPENROUTER_APP_TITLE", "Multi-Agent Deep Researcher"),
                }
            return OpenAIEmbeddings(**kwargs)
    except ImportError as exc:
        raise EmbeddingsConfigError(
            f"embeddings provider {s.provider!r} is not installed: {exc}. Run `{INSTALL_HINT[s.provider]}`."
        ) from exc
    raise EmbeddingsConfigError(f"unsupported provider {s.provider!r}")  # pragma: no cover


@functools.cache
def get_embeddings():
    """Process-wide embeddings instance for the configured spec (model load happens once)."""
    return make_embeddings()


def reset_cache() -> None:
    get_embeddings.cache_clear()


def info() -> dict[str, Any]:
    """Health payload. Never includes keys; never loads a model."""
    try:
        s = spec()
    except EmbeddingsConfigError as exc:
        return {"configured": False, "error": str(exc)}
    out = s.public()
    out["configured"] = True
    out["api_key_present"] = bool(os.getenv(s.api_key_env)) if s.api_key_env else None
    return out
