"""Embedding model factory: local by default, hosted when you ask for it.

The evidence store needs one embedding model per collection. This module picks it from env so
local development needs no API key at all, while Vercel can point at a hosted endpoint.

    EMBEDDINGS_PROVIDER   local | ollama | openai            (default: local)
    EMBEDDINGS_MODEL      provider-specific model id           (defaults below)
    EMBEDDINGS_DEVICE     local only: cpu | cuda | mps          (default: cpu)
    EMBEDDINGS_BASE_URL   ollama: server url (default http://localhost:11434)
                          openai: any OpenAI-compatible endpoint (OpenAI, OpenRouter, vLLM, LM Studio)
    EMBEDDINGS_API_KEY    openai only; falls back to OPENAI_API_KEY
    EMBEDDINGS_DIMENSIONS openai only: request truncated vectors (text-embedding-3-*)

Defaults per provider:
    local   BAAI/bge-small-en-v1.5    384-d, ~130 MB, runs on CPU   (uv sync --extra embeddings-local)
    ollama  nomic-embed-text          768-d                         (uv sync --extra embeddings-ollama)
    openai  text-embedding-3-small    1536-d                        (uv sync --extra embeddings-openai)

``embedding_id()`` is the fingerprint the evidence store writes into collection metadata so a
collection is never queried with a different model than the one that built it.
"""
from __future__ import annotations

import functools
import os
from dataclasses import asdict, dataclass
from typing import Any

PROVIDERS: tuple[str, ...] = ("local", "ollama", "openai")

DEFAULT_MODELS: dict[str, str] = {
    "local": "BAAI/bge-small-en-v1.5",
    "ollama": "nomic-embed-text",
    "openai": "text-embedding-3-small",
}
DEFAULT_BASE_URLS: dict[str, str | None] = {
    "local": None,
    "ollama": "http://localhost:11434",
    "openai": None,  # the SDK default (api.openai.com) unless overridden
}
INSTALL_HINT: dict[str, str] = {
    "local": "uv sync --extra embeddings-local",
    "ollama": "uv sync --extra embeddings-ollama   (and `ollama pull <model>`)",
    "openai": "uv sync --extra embeddings-openai",
}


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
        """Stable fingerprint: provider, model and (if requested) dimensions."""
        dims = f"@{self.dimensions}" if self.dimensions else ""
        return f"{self.provider}:{self.model}{dims}"

    def public(self) -> dict[str, Any]:
        d = asdict(self)
        d["embedding_id"] = self.embedding_id
        return d


def spec() -> EmbeddingSpec:
    """Resolve the embedding configuration from the environment (pure; no imports, no network)."""
    provider = (os.getenv("EMBEDDINGS_PROVIDER") or "local").strip().lower()
    if provider not in PROVIDERS:
        raise EmbeddingsConfigError(f"EMBEDDINGS_PROVIDER={provider!r}; expected one of {PROVIDERS}")
    dims_raw = (os.getenv("EMBEDDINGS_DIMENSIONS") or "").strip()
    dims = int(dims_raw) if dims_raw.isdigit() else None
    if dims is not None and provider != "openai":
        dims = None  # only OpenAI-style endpoints accept a dimensions parameter
    api_key_env = None
    if provider == "openai":
        api_key_env = "EMBEDDINGS_API_KEY" if os.getenv("EMBEDDINGS_API_KEY") else "OPENAI_API_KEY"
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
        if s.provider == "openai":
            from langchain_openai import OpenAIEmbeddings

            key = os.getenv(s.api_key_env or "OPENAI_API_KEY")
            if not key:
                raise EmbeddingsConfigError(
                    "EMBEDDINGS_PROVIDER=openai needs EMBEDDINGS_API_KEY or OPENAI_API_KEY; "
                    "or use EMBEDDINGS_PROVIDER=local to run without any key."
                )
            kwargs: dict[str, Any] = {"model": s.model, "api_key": key}
            if s.base_url:
                kwargs["base_url"] = s.base_url
            if s.dimensions:
                kwargs["dimensions"] = s.dimensions
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
