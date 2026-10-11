# Copyright (c) 2026 C10 Hackathon G1
"""Settings: .env round trip, masking, blank/none semantics, live apply with cache resets."""
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from deep_researcher import settings as S


@pytest.fixture
def env_file(tmp_path, monkeypatch):
    p = tmp_path / ".env"
    p.write_text("# comment\nOPENROUTER_API_KEY=sk-or-v1-0123456789abcdef\n"
                 "EMBEDDINGS_PROVIDER=local   # inline comment\nQUOTED=\"a b\"\nexport EXPORTED=1\n")
    monkeypatch.setenv("DEEP_RESEARCHER_ENV_FILE", str(p))
    for k in ("OPENROUTER_API_KEY", "TAVILY_API_KEY", "OPENROUTER_MODEL_HEAVY", "LANGSMITH_TRACING",
              "EMBEDDINGS_PROVIDER", "MAX_HOPS"):
        monkeypatch.delenv(k, raising=False)
    return p


def test_mask_never_reveals_the_middle():
    assert S.mask("") == "(not set)" and S.mask(None) == "(not set)"
    assert S.mask("short") == "*****"
    m = S.mask("sk-or-v1-0123456789abcdef")
    assert m.startswith("sk-or-") and m.endswith("cdef (25 chars)") and "0123456789" not in m


def test_read_env_file_handles_comments_quotes_and_export(env_file):
    v = S.read_env_file()
    assert v["OPENROUTER_API_KEY"] == "sk-or-v1-0123456789abcdef"
    assert v["EMBEDDINGS_PROVIDER"] == "local" and v["QUOTED"] == "a b" and v["EXPORTED"] == "1"


def test_write_env_updates_in_place_and_appends(env_file):
    S.write_env_updates({"EMBEDDINGS_PROVIDER": "openrouter", "TAVILY_API_KEY": "tv ly",
                         "OPENROUTER_API_KEY": None})
    text = env_file.read_text()
    assert text.startswith("# comment\n")                              # comments kept
    assert "OPENROUTER_API_KEY=\n" in text                             # cleared, line kept in place
    assert text.index("EMBEDDINGS_PROVIDER=openrouter") < text.index("QUOTED")   # order kept
    assert 'TAVILY_API_KEY="tv ly"' in text and "# added by the Settings page" in text
    assert oct(env_file.stat().st_mode & 0o777) == "0o600"
    assert S.read_env_file()["TAVILY_API_KEY"] == "tv ly"


def test_normalise_blank_secret_unchanged_none_clears(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-existing")
    monkeypatch.setenv("MAX_HOPS", "1")
    form = {"OPENROUTER_API_KEY": "", "TAVILY_API_KEY": "sk-or-…cdef (25 chars)", "NEWSDATA_API_KEY": "none",
            "SEMANTIC_SCHOLAR_API_KEY": " newkey ", "LANGSMITH_TRACING": True, "MAX_HOPS": "1",
            "RUN_BUDGET_SECONDS": "300", "OPENROUTER_MODEL_HEAVY": "none", "UNKNOWN": "x"}
    u = S.normalise(form)
    assert "OPENROUTER_API_KEY" not in u and "TAVILY_API_KEY" not in u      # blank / masked => unchanged
    assert u["NEWSDATA_API_KEY"] is None and u["SEMANTIC_SCHOLAR_API_KEY"] == "newkey"
    assert u["LANGSMITH_TRACING"] == "true"
    assert "MAX_HOPS" not in u and u["RUN_BUDGET_SECONDS"] == "300"          # unchanged text dropped
    assert u["OPENROUTER_MODEL_HEAVY"] is None and "UNKNOWN" not in u


def test_save_applies_to_process_and_resets_caches(env_file, monkeypatch):
    from deep_researcher import llm

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-old")
    assert llm.model_for("planner") == "anthropic/claude-opus-5.5"
    before = llm.get_llm("grader")
    updates, path = S.save({"OPENROUTER_API_KEY": "sk-or-new-key-value", "OPENROUTER_MODEL_HEAVY": "openai/gpt-5.1",
                            "LANGSMITH_TRACING": False})
    assert path == env_file and updates["OPENROUTER_MODEL_HEAVY"] == "openai/gpt-5.1"
    assert os.environ["OPENROUTER_API_KEY"] == "sk-or-new-key-value"
    assert llm.model_for("planner") == "openai/gpt-5.1"                     # live
    assert llm.get_llm("grader") is not before                              # cache dropped
    assert S.read_env_file()["OPENROUTER_MODEL_HEAVY"] == "openai/gpt-5.1"  # persisted
    assert S.read_env_file()["OPENROUTER_API_KEY"] == "sk-or-new-key-value"
    assert S.save({"OPENROUTER_API_KEY": ""}) == ({}, env_file)             # nothing changed


def test_summary_masks_secrets(env_file, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-0123456789abcdef")
    info = S.summary()
    assert "0123456789" not in str(info) and info["keys"]["OPENROUTER_API_KEY"].startswith("sk-or-")
    assert info["llm"]["configured"] is True and "papers" in info["sources"]


def test_ui_settings_helpers(env_file, monkeypatch):
    pytest.importorskip("gradio")
    from deep_researcher import ui

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-0123456789abcdef")
    md = ui.settings_summary_md()
    assert "0123456789" not in md and "**Models:**" in md and "**Active sources:**" in md
    assert ui.save_settings({"OPENROUTER_API_KEY": ""}) == "Nothing changed."
    msg = ui.save_settings({"EMBEDDINGS_DIMENSIONS": "native"})
    assert msg.startswith("Saved to") and "EMBEDDINGS_DIMENSIONS set" in msg and "CHROMA_COLLECTION" in msg
    assert ui.build_app(graph=None) is not None                             # both tabs construct
