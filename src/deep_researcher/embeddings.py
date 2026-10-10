"""Embedding model factory: local by default, hosted when you ask for it.

The evidence store needs one embedding model per collection. This module picks it from env so
local development needs no API key at all, while a deployment can point at a hosted endpoint.

    EMBEDDINGS_PROVIDER    local | openrouter | ollama | openai   (default: local)
    EMBEDDINGS_MODEL       provider-specific model id              (defaults below)
    EMBEDDINGS_DIMENSIONS  output vector size (default: 256; "0" or "native" = model's full size)
    EMBEDDINGS_DEVICE      local only: cpu | cuda | mps             (default: cpu)
    EMBEDDINGS_BASE_URL    ollama: server url (default http://localhost:11434)
                           openai: any OpenAI-compatible endpoint (OpenAI, vLLM, LM Studio)
                           openrouter: fixed to https://openrouter.ai/api/v1 unless overridden
    EMBEDDINGS_API_KEY     openai / openrouter; falls back to OPENAI_API_KEY / OPENROUTER_API_KEY

Reduced dimensions only make sense for Matryoshka-trained models (the first N components carry
most of the meaning). Requesting them from a fixed-size model is refused up front for the known
families and by the backend for the rest (OpenRouter returns 400 for e.g. bge-base).

Defaults per provider, all 256-d:
    local       Snowflake/snowflake-arctic-embed-m-v1.5   MRL-trained for 256, ~440 MB, CPU
    openrouter  openai/text-embedding-3-small             $0.02 / M tokens
    openai      text-embedding-3-small                    same weights as the openrouter default
    ollama      nomic-embed-text                          MRL (v1.5); `dimensions` passed through

``embedding_id()`` is the fingerprint the evidence store writes into collection metadata so a
collection is never queried with a different model than the one that built it. It is derived from
the canonical *model* name plus dimensions, not the provider, so ``openrouter`` and ``openai``
serving text-embedding-3-small share a collection. The local default is a different model, so a
locally built collection is not portable to a hosted deployment (and vice versa) under these
defaults; set ``EMBEDDINGS_PROVIDER=openrouter`` locally if you need that.
"""
from __future__ import annotations

import functools
import logging
import os
from dataclasses import asdict, dataclass
from typing import Any

log = logging.getLogger(__name__)

PROVIDERS: tuple[str, ...] = ("local", "openrouter", "ollama", "openai")

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_DIMENSIONS = 256

DEFAULT_MODELS: dict[str, str] = {
    "local": "Snowflake/snowflake-arctic-embed-m-v1.5",
    "openrouter": "openai/text-embedding-3-small",
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
    "local": "uv sync --extra embeddings-local   (+ --extra embeddings-gemma for google/embeddinggemma-2)",
    "openrouter": "uv sync --extra embeddings-openai",
    "ollama": "uv sync --extra embeddings-ollama   (and `ollama pull <model>`)",
    "openai": "uv sync --extra embeddings-openai",
}
# Providers that speak the OpenAI embeddings wire format.
_OPENAI_COMPATIBLE = {"openai", "openrouter"}

# Model families known to be Matryoshka-trained (safe to truncate) or fixed-size (refuse).
_MRL_FAMILIES = ("text-embedding-3-", "qwen3-embedding", "voyage-4", "gemini-embedding", "embeddinggemma",
                 "nomic-embed-text", "snowflake-arctic-embed", "mxbai-embed-large", "jina-embeddings-v3")
_FIXED_FAMILIES = ("bge-", "e5-", "gte-", "all-minilm", "all-mpnet", "multi-qa-", "paraphrase-",
                   "text-embedding-ada", "mistral-embed", "codestral-embed")


class EmbeddingsConfigError(RuntimeError):
    pass


def canonical_model(model: str, dimensions: int | None = None) -> str:
    """Lower-case; drop a leading ``sentence-transformers/`` or ``openai/`` org and any ``:variant``."""
    m = model.strip().lower().split(":", 1)[0]
    for org in ("sentence-transformers/", "openai/"):
        if m.startswith(org):
            m = m[len(org):]
    return f"{m}@{dimensions}" if dimensions else m


def supports_dimensions(model: str) -> bool | None:
    """True for known Matryoshka families, False for known fixed-size ones, None when unknown."""
    m = canonical_model(model)
    if any(f in m for f in _MRL_FAMILIES):
        return True
    if any(f in m for f in _FIXED_FAMILIES):
        return False
    return None


@dataclass(frozen=True)
class EmbeddingSpec:
    provider: str
    model: str
    device: str = "cpu"
    base_url: str | None = None
    api_key_env: str | None = None
    dimensions: int | None = DEFAULT_DIMENSIONS
    normalize: bool = True

    @property
    def embedding_id(self) -> str:
        return canonical_model(self.model, self.dimensions)

    def public(self) -> dict[str, Any]:
        d = asdict(self)
        d["embedding_id"] = self.embedding_id
        return d


def _parse_dimensions(raw: str | None) -> int | None:
    v = (raw if raw is not None else str(DEFAULT_DIMENSIONS)).strip().lower()
    if v in {"", "0", "none", "native", "full"}:
        return None
    if not v.isdigit() or int(v) <= 0:
        raise EmbeddingsConfigError(
            f"EMBEDDINGS_DIMENSIONS={raw!r} must be a positive integer, 0 or 'native'")
    return int(v)


def spec() -> EmbeddingSpec:
    """Resolve the embedding configuration from the environment (pure; no imports, no network)."""
    provider = (os.getenv("EMBEDDINGS_PROVIDER") or "local").strip().lower()
    if provider not in PROVIDERS:
        raise EmbeddingsConfigError(f"EMBEDDINGS_PROVIDER={provider!r}; expected one of {PROVIDERS}")
    model = os.getenv("EMBEDDINGS_MODEL") or DEFAULT_MODELS[provider]
    dims = _parse_dimensions(os.getenv("EMBEDDINGS_DIMENSIONS"))
    if dims is not None and supports_dimensions(model) is False:
        raise EmbeddingsConfigError(
            f"{model!r} has a fixed output size and cannot produce {dims}-d vectors. Set "
            "EMBEDDINGS_DIMENSIONS=native to use its full size, or pick a Matryoshka model "
            "(text-embedding-3-small, qwen3-embedding, voyage-4, snowflake-arctic-embed-m-v1.5, "
            "nomic-embed-text)."
        )
    api_key_env = None
    if provider in _OPENAI_COMPATIBLE:
        api_key_env = "EMBEDDINGS_API_KEY" if os.getenv("EMBEDDINGS_API_KEY") else DEFAULT_KEY_ENV[provider]
    return EmbeddingSpec(
        provider=provider,
        model=model,
        device=(os.getenv("EMBEDDINGS_DEVICE") or "cpu").strip().lower(),
        base_url=os.getenv("EMBEDDINGS_BASE_URL") or DEFAULT_BASE_URLS[provider],
        api_key_env=api_key_env,
        dimensions=dims,
    )


def embedding_id() -> str:
    return spec().embedding_id


def _attribution_headers() -> dict[str, str]:
    return {
        "HTTP-Referer": os.getenv("OPENROUTER_APP_URL", "https://github.com/pj-nx/multi-agent-deep-researcher"),
        "X-OpenRouter-Title": os.getenv("OPENROUTER_APP_TITLE", "Multi-Agent Deep Researcher"),
    }


def make_embeddings(s: EmbeddingSpec | None = None):
    """Construct a LangChain ``Embeddings`` for the spec. Imports the provider lazily."""
    s = s or spec()
    try:
        if s.provider == "local":
            from langchain_huggingface import HuggingFaceEmbeddings

            model_kwargs: dict[str, Any] = {"device": s.device}
            if s.dimensions:
                model_kwargs["truncate_dim"] = s.dimensions  # sentence-transformers Matryoshka truncation
            e = HuggingFaceEmbeddings(model_name=s.model, model_kwargs=model_kwargs,
                                      encode_kwargs={"normalize_embeddings": s.normalize})
            # Retrieval models often ship a query prompt (e.g. arctic-embed, bge). Apply it to
            # queries only, when the model defines one; documents are embedded as-is.
            prompts = getattr(getattr(e, "_client", None), "prompts", None) or {}
            if prompts.get("query"):
                e.query_encode_kwargs = {**e.encode_kwargs, "prompt_name": "query"}
            return e
        if s.provider == "ollama":
            from langchain_ollama import OllamaEmbeddings

            kwargs: dict[str, Any] = {"model": s.model, "base_url": s.base_url}
            if s.dimensions:
                kwargs["dimensions"] = s.dimensions
            return OllamaEmbeddings(**kwargs)
        if s.provider in _OPENAI_COMPATIBLE:
            from langchain_openai import OpenAIEmbeddings

            key = os.getenv(s.api_key_env or "")
            if not key:
                raise EmbeddingsConfigError(
                    f"EMBEDDINGS_PROVIDER={s.provider} needs EMBEDDINGS_API_KEY or "
                    f"{DEFAULT_KEY_ENV[s.provider]}; or use EMBEDDINGS_PROVIDER=local to run without any key."
                )
            kwargs = {"model": s.model, "api_key": key}
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
                kwargs["default_headers"] = _attribution_headers()
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
