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
    assert s.provider == "local" and s.model == "Snowflake/snowflake-arctic-embed-m-v1.5"
    assert s.api_key_env is None and s.dimensions == 256
    assert s.embedding_id == "snowflake/snowflake-arctic-embed-m-v1.5@256"
    i = emb.info()
    assert i["configured"] is True and i["api_key_present"] is None and i["api_key_env"] is None
    assert set(i) == {"provider", "model", "device", "base_url", "api_key_env", "dimensions", "normalize",
                      "embedding_id", "configured", "api_key_present"}      # no secret-bearing fields


def test_provider_switches(emb, monkeypatch):
    monkeypatch.setenv("EMBEDDINGS_PROVIDER", "ollama")
    s = emb.spec()
    assert s.model == "nomic-embed-text" and s.base_url == "http://localhost:11434" and s.dimensions == 256

    monkeypatch.setenv("EMBEDDINGS_PROVIDER", "openai")
    monkeypatch.setenv("EMBEDDINGS_MODEL", "text-embedding-3-large")
    monkeypatch.setenv("EMBEDDINGS_DIMENSIONS", "1024")
    monkeypatch.setenv("EMBEDDINGS_BASE_URL", "https://openrouter.ai/api/v1")
    s = emb.spec()
    assert s.embedding_id == "text-embedding-3-large@1024"
    assert s.api_key_env == "OPENAI_API_KEY" and s.base_url == "https://openrouter.ai/api/v1"
    monkeypatch.setenv("EMBEDDINGS_API_KEY", "x")
    assert emb.spec().api_key_env == "EMBEDDINGS_API_KEY"


def test_dimensions_native_and_invalid(emb, monkeypatch):
    monkeypatch.setenv("EMBEDDINGS_DIMENSIONS", "native")
    assert emb.spec().dimensions is None and "@" not in emb.spec().embedding_id
    monkeypatch.setenv("EMBEDDINGS_DIMENSIONS", "0")
    assert emb.spec().dimensions is None
    monkeypatch.setenv("EMBEDDINGS_DIMENSIONS", "lots")
    with pytest.raises(emb.EmbeddingsConfigError):
        emb.spec()


def test_fixed_size_model_refuses_reduced_dimensions(emb, monkeypatch):
    monkeypatch.setenv("EMBEDDINGS_MODEL", "BAAI/bge-base-en-v1.5")       # 768 fixed, not Matryoshka
    with pytest.raises(emb.EmbeddingsConfigError, match="fixed output size"):
        emb.spec()
    monkeypatch.setenv("EMBEDDINGS_DIMENSIONS", "native")
    assert emb.spec().embedding_id == "baai/bge-base-en-v1.5"                # allowed at full size


def test_supports_dimensions_classification(emb):
    assert emb.supports_dimensions("openai/text-embedding-3-small") is True
    assert emb.supports_dimensions("Snowflake/snowflake-arctic-embed-m-v1.5") is True
    assert emb.supports_dimensions("qwen/qwen3-embedding-8b") is True
    assert emb.supports_dimensions("google/embeddinggemma-2") is True
    assert emb.supports_dimensions("baai/bge-m3") is False
    assert emb.supports_dimensions("sentence-transformers/all-MiniLM-L6-v2") is False
    assert emb.supports_dimensions("acme/mystery-embed") is None


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
    assert captured == {"model_name": "Snowflake/snowflake-arctic-embed-m-v1.5",
                        "model_kwargs": {"device": "cuda", "truncate_dim": 256},
                        "encode_kwargs": {"normalize_embeddings": True}}


def test_openrouter_and_openai_share_fingerprint(emb, monkeypatch):
    monkeypatch.setenv("EMBEDDINGS_PROVIDER", "openrouter")
    s = emb.spec()
    assert s.model == "openai/text-embedding-3-small" and s.base_url == "https://openrouter.ai/api/v1"
    assert s.api_key_env == "OPENROUTER_API_KEY"
    assert s.embedding_id == "text-embedding-3-small@256"
    monkeypatch.setenv("EMBEDDINGS_PROVIDER", "openai")
    assert emb.spec().embedding_id == "text-embedding-3-small@256"    # same collection


def test_canonical_model_normalisation(emb):
    assert emb.canonical_model("sentence-transformers/all-MiniLM-L6-v2") == "all-minilm-l6-v2"
    assert emb.canonical_model("liquid/lfm-2.5-embedding-350m:free") == "liquid/lfm-2.5-embedding-350m"
    assert emb.canonical_model("text-embedding-3-small", 512) == "text-embedding-3-small@512"
    assert emb.canonical_model("openai/text-embedding-3-small", 512) == "text-embedding-3-small@512"


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
    assert e.model == "openai/text-embedding-3-small" and e.dimensions == 256
    assert e.openai_api_base == "https://openrouter.ai/api/v1"
    assert e.check_embedding_ctx_length is False and e.tiktoken_enabled is False
    assert e.default_headers["X-OpenRouter-Title"]
    assert e.openai_api_key.get_secret_value() == "sk-or-test"
