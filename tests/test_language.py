# Copyright (c) 2026 C10 Hackathon G1
"""Language gate: detection, declared metadata, allowed list, retriever filtering, API parameters."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from deep_researcher.tools import language as L

EN = "A UK trial of the four-day working week reported higher productivity and lower burnout among staff."
FR = "Une étude britannique sur la semaine de quatre jours fait état d'une productivité accrue et de moins de stress."
DE = "Eine britische Studie zur Vier-Tage-Woche berichtet von höherer Produktivität und weniger Burnout."


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for v in ("RETRIEVAL_LANGUAGES", "NEWSDATA_LANGUAGE", "WIKIPEDIA_LANG"):
        monkeypatch.delenv(v, raising=False)


def test_allowed_languages_env(monkeypatch):
    assert L.allowed_languages() == ["en"] and L.primary_language() == "en"
    monkeypatch.setenv("RETRIEVAL_LANGUAGES", " EN, de ,fr,")
    assert L.allowed_languages() == ["en", "de", "fr"]
    monkeypatch.setenv("RETRIEVAL_LANGUAGES", " ")
    assert L.allowed_languages() == ["en"]


def test_detect_and_normalise():
    assert L.detect(EN) == "en" and L.detect(FR) == "fr" and L.detect(DE) == "de"
    assert L.detect("x") is None and L.detect("") is None            # too short to trust
    assert L.normalise_code("english") == "en" and L.normalise_code("zh-cn") == "zh"
    assert L.normalise_code("EN") == "en" and L.normalise_code(None) is None
    assert L.language_name("en") == "english" and L.language_name("xx") == "xx"


def test_classify_declared_beats_detection_and_unknown_passes(monkeypatch):
    assert L.classify(EN) == (True, "en") and L.classify(FR) == (False, "fr")
    assert L.classify(EN, declared="french") == (False, "fr")       # metadata wins
    assert L.classify("short", declared=None) == (True, None)        # undecidable -> allowed
    monkeypatch.setenv("RETRIEVAL_LANGUAGES", "en,fr")
    assert L.classify(FR) == (True, "fr") and L.classify(DE) == (False, "de")


def test_retriever_language_gate_drops_and_counts():
    from deep_researcher.agents.retriever import language_gate, to_evidence
    from deep_researcher.tools.base import SearchResult

    rs = [SearchResult(title="UK trial", url="https://a", text=EN, source_type="news"),
          SearchResult(title="Étude", url="https://b", text=FR, source_type="news"),
          SearchResult(title="Studie", url="https://c", text=DE, source_type="papers"),
          SearchResult(title="Declared", url="https://d", text=EN, source_type="news", extra={"language": "german"}),
          SearchResult(title="Tiny", url="https://e", text="ok", source_type="web")]
    kept, dropped = language_gate(rs)
    assert [r.url for r in kept] == ["https://a", "https://e"]
    assert dropped == {"fr": 1, "de": 2}
    assert to_evidence(kept[0], "sq1").language == "en" and to_evidence(kept[1], "sq1").language is None


@pytest.mark.anyio
async def test_apis_receive_allowed_languages(monkeypatch):
    import httpx

    from deep_researcher.tools.news import GdeltTool, NewsDataTool
    from deep_researcher.tools.wikipedia import WikipediaTool

    monkeypatch.setenv("RETRIEVAL_LANGUAGES", "de,en")
    monkeypatch.setenv("NEWSDATA_API_KEY", "nk")
    seen = {}

    def handler(request):
        seen[request.url.host] = dict(request.url.params)
        if request.url.host == "api.gdeltproject.org":
            return httpx.Response(200, json={"articles": []}, headers={"content-type": "application/json"})
        if request.url.host.endswith("wikipedia.org"):
            return httpx.Response(200, json={"query": {"pages": []}})
        return httpx.Response(200, json={"status": "success", "results": []})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    for cls in (NewsDataTool, GdeltTool, WikipediaTool):
        monkeypatch.setattr(cls, "min_interval_s", 0.0)
        await cls(client).search("vier tage woche", limit=3)
    assert seen["newsdata.io"]["language"] == "de,en"
    assert seen["api.gdeltproject.org"]["sourcelang"] == "german"
    assert "de.wikipedia.org" in seen


def test_settings_expose_languages(monkeypatch):
    from deep_researcher import settings as S

    assert "RETRIEVAL_LANGUAGES" in S.FIELD_BY_KEY
    monkeypatch.setenv("RETRIEVAL_LANGUAGES", "en,fr")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    assert S.summary()["languages"] == ["en", "fr"] and L.info()["allowed"] == ["en", "fr"]
