import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture
def emb(monkeypatch):
    import os
    for var in [v for v in os.environ if v.startswith("EMBEDDINGS_")] + ["OPENAI_API_KEY", "OPENROUTER_API_KEY"]:
        monkeypatch.delenv(var, raising=False)
    import deep_researcher.embeddings as mod
    mod = importlib.reload(mod)
    mod.reset_cache()
    return mod


def test_default_is_local_and_keyless(emb):
    s = emb.spec()
    assert s.provider == "local" and s.model == "BAAI/bge-base-en-v1.5" and s.api_key_env is None
    assert s.embedding_id == "baai/bge-base-en-v1.5"
    i = emb.info()
    assert i["configured"] is True and i["api_key_present"] is None and i["api_key_env"] is None
    assert set(i) == {"provider", "model", "device", "base_url", "api_key_env", "dimensions", "normalize",
                      "embedding_id", "configured", "api_key_present"}      # no secret-bearing fields


def test_provider_switches(emb, monkeypatch):
    monkeypatch.setenv("EMBEDDINGS_PROVIDER", "ollama")
    s = emb.spec()
    assert s.model == "nomic-embed-text" and s.base_url == "http://localhost:11434"

    monkeypatch.setenv("EMBEDDINGS_PROVIDER", "openai")
    monkeypatch.setenv("EMBEDDINGS_MODEL", "text-embedding-3-large")
    monkeypatch.setenv("EMBEDDINGS_DIMENSIONS", "1024")
    monkeypatch.setenv("EMBEDDINGS_BASE_URL", "https://openrouter.ai/api/v1")
    s = emb.spec()
    assert s.embedding_id == "text-embedding-3-large@1024"
    assert s.api_key_env == "OPENAI_API_KEY" and s.base_url == "https://openrouter.ai/api/v1"
    monkeypatch.setenv("EMBEDDINGS_API_KEY", "x")
    assert emb.spec().api_key_env == "EMBEDDINGS_API_KEY"


def test_dimensions_ignored_for_non_openai(emb, monkeypatch):
    monkeypatch.setenv("EMBEDDINGS_DIMENSIONS", "256")
    assert emb.spec().dimensions is None


def test_bad_provider(emb, monkeypatch):
    monkeypatch.setenv("EMBEDDINGS_PROVIDER", "pinecone")
    with pytest.raises(emb.EmbeddingsConfigError):
        emb.spec()
    assert emb.info()["configured"] is False


def test_openai_requires_key(emb, monkeypatch):
    monkeypatch.setenv("EMBEDDINGS_PROVIDER", "openai")
    pytest.importorskip("langchain_openai")
    with pytest.raises(emb.EmbeddingsConfigError, match="OPENAI_API_KEY"):
        emb.make_embeddings()


def test_missing_backend_gives_install_hint(emb, monkeypatch):
    monkeypatch.setenv("EMBEDDINGS_PROVIDER", "ollama")
    monkeypatch.setitem(sys.modules, "langchain_ollama", None)   # simulate not installed
    with pytest.raises(emb.EmbeddingsConfigError, match="embeddings-ollama"):
        emb.make_embeddings()


def test_local_constructs_with_expected_kwargs(emb, monkeypatch):
    pytest.importorskip("langchain_huggingface")
    captured = {}

    class FakeHF:
        def __init__(self, **kw):
            captured.update(kw)

    import langchain_huggingface
    monkeypatch.setattr(langchain_huggingface, "HuggingFaceEmbeddings", FakeHF)
    monkeypatch.setenv("EMBEDDINGS_DEVICE", "cuda")
    emb.make_embeddings()
    assert captured == {"model_name": "BAAI/bge-base-en-v1.5", "model_kwargs": {"device": "cuda"},
                        "encode_kwargs": {"normalize_embeddings": True}}


def test_openrouter_provider_shares_fingerprint_with_local(emb, monkeypatch):
    monkeypatch.setenv("EMBEDDINGS_PROVIDER", "openrouter")
    s = emb.spec()
    assert s.model == "baai/bge-base-en-v1.5" and s.base_url == "https://openrouter.ai/api/v1"
    assert s.api_key_env == "OPENROUTER_API_KEY"
    assert s.embedding_id == "baai/bge-base-en-v1.5"       # == local default → same collection


def test_canonical_model_normalisation(emb):
    assert emb.canonical_model("sentence-transformers/all-MiniLM-L6-v2") == "all-minilm-l6-v2"
    assert emb.canonical_model("liquid/lfm-2.5-embedding-350m:free") == "liquid/lfm-2.5-embedding-350m"
    assert emb.canonical_model("text-embedding-3-small", 512) == "text-embedding-3-small@512"


def test_openrouter_requires_openrouter_key(emb, monkeypatch):
    monkeypatch.setenv("EMBEDDINGS_PROVIDER", "openrouter")
    pytest.importorskip("langchain_openai")
    with pytest.raises(emb.EmbeddingsConfigError, match="OPENROUTER_API_KEY"):
        emb.make_embeddings()


def test_openrouter_constructs_openai_embeddings_for_third_party_endpoint(emb, monkeypatch):
    pytest.importorskip("langchain_openai")
    monkeypatch.setenv("EMBEDDINGS_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    e = emb.make_embeddings()
    assert e.model == "baai/bge-base-en-v1.5"
    assert e.openai_api_base == "https://openrouter.ai/api/v1"
    assert e.check_embedding_ctx_length is False and e.tiktoken_enabled is False
    assert e.default_headers["X-OpenRouter-Title"]
    assert e.openai_api_key.get_secret_value() == "sk-or-test"
