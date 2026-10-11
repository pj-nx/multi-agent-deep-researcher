"""Retrieval tools against canned HTTP responses (httpx.MockTransport): parsers, enablement, failure modes."""
import json
import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from deep_researcher.tools import POLICIES, fetch_text, info, register, tools_for
from deep_researcher.tools.arxiv import ArxivTool
from deep_researcher.tools.base import RetrievalTool, SearchResult, iso_date
from deep_researcher.tools.fetch import html_to_text
from deep_researcher.tools.news import GdeltTool, gdelt_query, newsdata_query
from deep_researcher.tools.semantic_scholar import SemanticScholarTool
from deep_researcher.tools.web_search import TavilyReportsTool, TavilyTool
from deep_researcher.tools.wikipedia import WikipediaTool

ARXIV_ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/1206.0021v1</id>
    <published>2012-05-31T20:15:35Z</published>
    <title>Four-Day  Week\n  Productivity</title>
    <summary>  We study   a four-day week.\n Productivity rose. </summary>
    <author><name>A. Author</name></author>
    <arxiv:doi>10.1000/xyz</arxiv:doi>
    <arxiv:primary_category term="econ.GN"/>
    <link href="http://arxiv.org/abs/1206.0021v1" rel="alternate" type="text/html"/>
    <link title="pdf" href="http://arxiv.org/pdf/1206.0021v1" rel="related" type="application/pdf"/>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/9999.0001v1</id><published>2020-01-01T00:00:00Z</published>
    <title>Empty</title><summary>   </summary>
  </entry>
</feed>"""

S2_JSON = {"total": 2, "data": [
    {"paperId": "p1", "title": "Trial", "abstract": "Big abstract.", "url": "https://www.semanticscholar.org/paper/p1",
     "year": 2023, "publicationDate": "2023-04-02", "citationCount": 12, "openAccessPdf": {"url": "https://x/p1.pdf"},
     "externalIds": {"DOI": "10.1/abc", "ArXiv": "2301.00001"}, "venue": "J", "authors": [{"name": "B"}]},
    {"paperId": "p2", "title": "No abstract", "abstract": None, "url": "https://x/p2", "year": 2020},
]}
WIKI_JSON = {"query": {"pages": [
    {"pageid": 2, "title": "Second", "index": 2, "extract": "Second intro.", "fullurl": "https://en.wikipedia.org/wiki/Second",
     "touched": "2026-10-07T11:33:25Z", "length": 100},
    {"pageid": 1, "title": "Four-day workweek", "index": 1, "extract": "A four-day workweek is...",
     "fullurl": "https://en.wikipedia.org/wiki/Four-day_workweek", "touched": "2026-10-07T11:33:25Z"},
    {"pageid": 3, "title": "Empty", "index": 3, "extract": ""},
]}}
GDELT_JSON = {"articles": [{"url": "https://news.example/a", "title": "Banks order return", "seendate": "20250131T120000Z",
                            "domain": "news.example", "sourcecountry": "UK"}]}
NEWSDATA_JSON = {"status": "success", "totalResults": 2, "results": [
    {"article_id": "a1", "title": "T", "link": "https://n/1", "description": "D.", "content": "ONLY AVAILABLE IN PAID PLANS",
     "pubDate": "2025-02-01 10:00:00", "source_id": "reuters", "source_name": "Reuters", "source_priority": 12,
     "country": ["united kingdom"], "category": ["business"], "creator": ["A. Writer"]},
    {"article_id": "a2", "title": "T2", "link": "https://n/2", "description": "D2.", "content": "Full body text.",
     "pubDate": "2025-02-02 10:00:00", "source_name": "FT"},
]}
NEWSDATA_ERROR = {"status": "error", "results": {"message": "API key is invalid", "code": "Unauthorized"}}
TAVILY_JSON = {"results": [{"title": "W", "url": "https://w/1?utm_source=x", "content": "Web content", "score": 0.9,
                            "published_date": "2025-03-01"}]}


def _transport(status_by_host=None):
    status_by_host = status_by_host or {}

    def handler(request: httpx.Request) -> httpx.Response:
        host = request.url.host
        if host in status_by_host:
            return httpx.Response(status_by_host[host], text="rate limited", headers={"Retry-After": "0"})
        if host == "export.arxiv.org":
            assert request.url.params["search_query"].startswith("all:")
            return httpx.Response(200, text=ARXIV_ATOM, headers={"content-type": "application/atom+xml"})
        if host == "api.semanticscholar.org":
            return httpx.Response(200, json=S2_JSON)
        if host == "en.wikipedia.org":
            return httpx.Response(200, json=WIKI_JSON)
        if host == "api.gdeltproject.org":
            return httpx.Response(200, json=GDELT_JSON, headers={"content-type": "application/json"})
        if host == "newsdata.io":
            assert request.headers["x-access-key"] == "nk" and "apikey" not in str(request.url)
            assert request.url.params["size"] == "5" and request.url.params["language"] == "en"
            if request.url.params["q"] == "bad key":
                return httpx.Response(200, json=NEWSDATA_ERROR)
            return httpx.Response(200, json=NEWSDATA_JSON)
        if host == "api.tavily.com":
            body = json.loads(request.content)
            assert body["api_key"] == "tk"
            return httpx.Response(200, json=TAVILY_JSON)
        return httpx.Response(404)

    return httpx.MockTransport(handler)


@pytest.fixture
def client():
    return httpx.AsyncClient(transport=_transport())


@pytest.fixture(autouse=True)
def _fast(monkeypatch):
    for cls in (ArxivTool, SemanticScholarTool, GdeltTool, WikipediaTool):
        monkeypatch.setattr(cls, "min_interval_s", 0.0)
    for v in ("TAVILY_API_KEY", "NEWSDATA_API_KEY", "SEMANTIC_SCHOLAR_API_KEY", "NEWSDATA_COUNTRY"):
        monkeypatch.delenv(v, raising=False)


def test_gdelt_query_sanitiser():
    q = gdelt_query("Where do studies report divergent or negative results (healthcare, retail), and why?")
    assert "or" not in q.split() and "(" not in q and len(q.split()) <= 8
    assert q.startswith("studies report divergent negative results")
    assert gdelt_query("do it") == ""                      # nothing >= 3 chars survives


def test_iso_date():
    assert iso_date("2025-01-31T12:00:00Z") == "2025-01-31"
    assert iso_date("20250131T120000Z") == "2025-01-31"
    assert iso_date("2023") == "2023-01-01"
    assert iso_date(None) is None and iso_date("garbage") is None


@pytest.mark.anyio
async def test_arxiv_parser(client):
    res = await ArxivTool(client).search("four-day week", limit=5)
    assert len(res) == 1                                   # empty-summary entry dropped
    r = res[0]
    assert r.title == "Four-Day Week Productivity" and r.text == "We study a four-day week. Productivity rose."
    assert r.url == "http://arxiv.org/abs/1206.0021v1" and r.extra["pdf_url"].endswith("1206.0021v1")
    assert r.published_at == "2012-05-31" and r.extra["doi"] == "10.1000/xyz" and r.extra["primary_category"] == "econ.GN"


@pytest.mark.anyio
async def test_semantic_scholar_parser(client):
    res = await SemanticScholarTool(client).search("x")
    assert [r.title for r in res] == ["Trial"]            # abstract-less paper dropped
    assert res[0].published_at == "2023-04-02" and res[0].extra["citation_count"] == 12
    assert res[0].extra["pdf_url"] == "https://x/p1.pdf"


@pytest.mark.anyio
async def test_wikipedia_parser_orders_by_rank_and_drops_empty(client):
    res = await WikipediaTool(client).search("x")
    assert [r.title for r in res] == ["Four-day workweek", "Second"]
    assert res[0].url.endswith("/Four-day_workweek") and res[0].published_at == "2026-10-07"


def test_newsdata_query_strips_operators_and_caps_length():
    assert newsdata_query("four-day week AND productivity (UK) OR trial") == "four-day week productivity UK trial"
    assert len(newsdata_query("word " * 200)) <= 512


@pytest.mark.anyio
async def test_news_prefers_newsdata_when_key_present_else_gdelt(client, monkeypatch):
    assert [t.name for t in tools_for(["news"], client)] == ["gdelt"]
    monkeypatch.setenv("NEWSDATA_API_KEY", "nk")
    tools = tools_for(["news"], client)
    assert [t.name for t in tools] == ["newsdata"]
    res = await tools[0].search("four-day week", limit=5)
    assert [r.title for r in res] == ["T", "T2"]
    assert res[0].extra["source"] == "Reuters" and res[0].text == "D." and res[0].published_at == "2025-02-01"
    assert res[0].extra["needs_fetch"] is True                      # free-plan placeholder content dropped
    assert res[1].text == "D2. Full body text." and res[1].extra["needs_fetch"] is False
    assert await tools[0].search("bad key", limit=5) == []          # API-level error => empty, never raises
    g = await GdeltTool(client).search("banks office return mandate")
    assert g[0].extra["needs_fetch"] is True and g[0].published_at == "2025-01-31" and g[0].text == g[0].title


@pytest.mark.anyio
async def test_tavily_web_and_reports(client, monkeypatch):
    assert tools_for(["web", "reports"], client) == []     # no key → nothing
    monkeypatch.setenv("TAVILY_API_KEY", "tk")
    names = [t.name for t in tools_for(["web", "reports"], client)]
    assert names == ["tavily", "tavily_reports"]
    web = await TavilyTool(client).search("x")
    rep = await TavilyReportsTool(client).search("x")
    assert web[0].source_type == "web" and rep[0].source_type == "reports" and web[0].published_at == "2025-03-01"


@pytest.mark.anyio
async def test_http_errors_never_raise(monkeypatch):
    c = httpx.AsyncClient(transport=_transport({"export.arxiv.org": 429, "en.wikipedia.org": 500}))
    assert await ArxivTool(c).search("x") == []
    assert await WikipediaTool(c).search("x") == []

    def boom(request):
        raise httpx.ConnectError("nope")

    assert await ArxivTool(httpx.AsyncClient(transport=httpx.MockTransport(boom))).search("x") == []


def test_newsdata_size_is_capped_to_plan(monkeypatch):
    from deep_researcher.tools.news import NewsDataTool

    monkeypatch.setenv("NEWSDATA_API_KEY", "nk")
    seen = {}

    def handler(request):
        seen["size"] = request.url.params["size"]
        return httpx.Response(200, json={"status": "success", "results": []})

    import asyncio
    asyncio.run(NewsDataTool(httpx.AsyncClient(transport=httpx.MockTransport(handler))).search("x y", limit=40))
    assert seen["size"] == "10"


def test_registry_info_and_dedup(monkeypatch):
    i = info()
    assert i["papers"]["active"] == ["arxiv", "semantic_scholar"] and i["news"]["active"] == ["gdelt"]
    assert i["web"]["active"] == [] and i["web"]["tools"][0]["requires"] == ["TAVILY_API_KEY"]
    names = [t.name for t in tools_for(["papers", "papers", "wikipedia"])]
    assert names == ["arxiv", "semantic_scholar", "wikipedia"]


def test_register_custom_api_tool(monkeypatch):
    class FredTool(RetrievalTool):
        name = "fred"
        source_type = "api"
        requires_env = ("FRED_API_KEY",)

    monkeypatch.setattr(POLICIES["api"], "tools", [])
    register("api", FredTool)
    assert tools_for(["api"]) == []
    monkeypatch.setenv("FRED_API_KEY", "k")
    assert [t.name for t in tools_for(["api"])] == ["fred"]
    POLICIES["api"].tools.remove(FredTool)


def test_html_to_text_strips_boilerplate():
    html = "<html><head><style>x{}</style><script>bad()</script></head><body><nav>menu</nav>" \
           "<h1>Title</h1><p>Para one.</p><p>Para&nbsp;two.</p><footer>f</footer></body></html>"
    text = html_to_text(html)
    assert "bad()" not in text and "menu" not in text and "f" != text.strip()[-1]
    assert "Title" in text and "Para one." in text and "Para two." in text


def _mini_pdf(text: str = "Hello evidence from a PDF") -> bytes:
    """A valid one-page PDF with a correct xref table (pypdf needs the EOF marker and offsets)."""
    objs = [
        b"<</Type/Catalog/Pages 2 0 R>>",
        b"<</Type/Pages/Kids[3 0 R]/Count 1>>",
        b"<</Type/Page/Parent 2 0 R/MediaBox[0 0 300 144]/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>",
        None,  # content stream, filled below
        b"<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>",
    ]
    stream = f"BT /F1 18 Tf 20 100 Td ({text}) Tj ET".encode()
    objs[3] = b"<</Length " + str(len(stream)).encode() + b">>stream\n" + stream + b"\nendstream"
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += f"trailer<</Size {len(objs) + 1}/Root 1 0 R>>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)


MINI_PDF = _mini_pdf()


@pytest.mark.anyio
async def test_fetch_text_pdf_and_html_and_failure():
    def handler(request):
        if request.url.path == "/doc.pdf":
            return httpx.Response(200, content=MINI_PDF, headers={"content-type": "application/pdf"})
        if request.url.path == "/page":
            return httpx.Response(200, text="<html><body><p>Fetched body.</p></body></html>",
                                  headers={"content-type": "text/html; charset=utf-8"})
        if request.url.path == "/img":
            return httpx.Response(200, content=b"\x89PNG", headers={"content-type": "image/png"})
        return httpx.Response(404)

    c = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    pdf = await fetch_text("https://h/doc.pdf", client=c)
    assert pdf and pdf[1] == "pdf" and "Hello evidence" in pdf[0]
    html = await fetch_text("https://h/page", client=c)
    assert html == ("Fetched body.", "html")
    assert await fetch_text("https://h/img", client=c) is None
    assert await fetch_text("https://h/missing", client=c) is None


def test_search_result_usable():
    assert not SearchResult(title="t", url="", text="x", source_type="web").usable()
    assert not SearchResult(title="t", url="u", text="  ", source_type="web").usable()


def test_identical_searches_are_answered_once_even_in_parallel():
    import asyncio

    calls = []

    def handler(request):
        calls.append(str(request.url))
        return httpx.Response(200, text="<feed xmlns='http://www.w3.org/2005/Atom'></feed>")

    async def go():
        tool = ArxivTool(httpx.AsyncClient(transport=httpx.MockTransport(handler)))
        tool.min_interval_s = 0
        await asyncio.gather(*(tool.search("four-day week") for _ in range(4)))
        await tool.search("four-day week")                 # later, same run: still cached
        await tool.search("something else")
    asyncio.run(go())
    assert len(calls) == 2


def test_failed_searches_are_not_cached():
    import asyncio

    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(429)

    async def go():
        tool = ArxivTool(httpx.AsyncClient(transport=httpx.MockTransport(handler)))
        tool.min_interval_s = 0
        tool.max_retries = 0
        assert await tool.search("x") == [] and await tool.search("x") == []
    asyncio.run(go())
    assert len(calls) == 2
