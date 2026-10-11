"""Contextual Retriever: one branch per sub-question, fans out across the enabled tools.

Deterministic by design (no LLM in the loop): every tool for the sub-question's source types is
queried in parallel, results are de-duplicated by URL, optionally enriched with fetched full text,
converted to ``Evidence`` and written to the vector store. A ReAct variant that lets a model
reformulate queries can be layered on later without changing the state contract.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import time
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from deep_researcher.agents.common import emit
from deep_researcher.agents.deps import Deps
from deep_researcher.health import degrade, tool_errors
from deep_researcher.schemas import Evidence, SubQuestion
from deep_researcher.tools import language as lang
from deep_researcher.tools.base import SearchResult

log = logging.getLogger(__name__)

# Baseline prior by source type; the Source Validator (Phase 3) refines per document.
CREDIBILITY_PRIOR: dict[str, float] = {
    "papers": 0.8, "api": 0.75, "reports": 0.65, "wikipedia": 0.6, "news": 0.5, "web": 0.4,
}


def normalise_url(url: str) -> str:
    parts = urlsplit(url.strip())
    tracking = ("utm_", "fbclid", "ref=")
    query = "&".join(p for p in parts.query.split("&") if p and not p.lower().startswith(tracking))
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path.rstrip("/"), query, ""))


def evidence_id(url: str) -> str:
    return "E-" + hashlib.sha1(normalise_url(url).encode()).hexdigest()[:10]


def to_evidence(result: SearchResult, sub_question_id: str) -> Evidence:
    return Evidence(
        id=evidence_id(result.url), sub_question_id=sub_question_id, source_type=result.source_type,
        title=result.title[:500], url=result.url[:2000], text=result.text,
        published_at=result.published_at, credibility=CREDIBILITY_PRIOR.get(result.source_type, 0.5),
        language=result.extra.get("language_code"),
    )


def language_gate(results: list[SearchResult]) -> tuple[list[SearchResult], dict[str, int]]:
    """Keep results in the allowed languages; annotate each with its detected/declared code."""
    allowed = lang.allowed_languages()
    kept: list[SearchResult] = []
    dropped: dict[str, int] = {}
    for r in results:
        ok, code = lang.classify(f"{r.title}. {r.text}", r.extra.get("language"), allowed)
        r.extra["language_code"] = code
        if ok:
            kept.append(r)
        else:
            dropped[code or "?"] = dropped.get(code or "?", 0) + 1
    return kept, dropped


async def _maybe_enrich(results: list[SearchResult], deps: Deps) -> None:
    """Fetch full text for results that only have a title/short snippet or point at a PDF."""
    if not deps.fetch_full_text:
        return
    from deep_researcher.tools.fetch import fetch_text

    async def enrich(r: SearchResult) -> None:
        target = r.extra.get("pdf_url") or r.url
        if len(r.text) > 1500 and not r.extra.get("pdf_url"):
            return
        fetched = await fetch_text(target)
        if fetched and len(fetched[0]) > len(r.text):
            r.text = fetched[0]
            r.extra["fetched"] = fetched[1]

    await asyncio.gather(*(enrich(r) for r in results[:deps.max_per_sub_question]))


_STOP = {"what", "which", "where", "when", "does", "did", "have", "has", "been", "about", "with", "from",
         "that", "this", "these", "those", "their", "there", "into", "than", "then", "they", "them", "were",
         "will", "would", "should", "could", "also", "such", "like", "over", "under", "between", "after",
         "before", "major", "report", "reported", "studies", "study", "results", "find", "finds", "found",
         "example", "examples", "evidence", "across", "versus"}


def keywords(sq: SubQuestion) -> set[str]:
    """Content words (>= 4 chars, no stopwords) from the sub-question and its search queries."""
    text = " ".join([sq.question, *sq.search_queries]).lower()
    return {w for w in re.findall(r"[a-z][a-z0-9-]{3,}", text) if w not in _STOP}


def relevance(result: SearchResult, kws: set[str]) -> float:
    """Fraction of sub-question keywords that appear in the result's title + text (stem-tolerant)."""
    if not kws:
        return 1.0
    hay = f"{result.title} {result.text}".lower()
    hits = sum(1 for k in kws if k[: max(4, len(k) - 2)] in hay)
    return hits / len(kws)


def search_queries(sq: SubQuestion) -> list[str]:
    """Short keyword queries from the planner, falling back to a trimmed form of the question."""
    queries = [q for q in sq.search_queries if q.strip()]
    if queries:
        return queries
    words = [w.strip("?,.;:()\"'") for w in sq.question.split()]
    return [" ".join(w for w in words if w)[:120]]


async def _search_tool(tool: Any, queries: list[str], deps: Deps, sq_id: str = "",
                       errors: list[dict[str, Any]] | None = None) -> list[SearchResult]:
    """Run every query against one tool, bounded by a per-tool timeout; partial results survive.

    Failures the tool swallows (HTTP errors, bad payloads) and timeouts are appended to ``errors``.
    """
    errors = errors if errors is not None else []
    token = tool_errors.set(errors)
    out: list[SearchResult] = []
    seen: set[str] = set()
    started = time.monotonic()
    emit("tool_start", tool=tool.name, sub_question_id=sq_id, queries=len(queries))

    async def run() -> None:
        for q in queries:
            for r in await tool.search(q, deps.per_tool_limit):
                key = normalise_url(r.url)
                if key not in seen:
                    seen.add(key)
                    out.append(r)

    try:
        await asyncio.wait_for(run(), timeout=deps.tool_timeout_s)
    except TimeoutError:
        log.warning("%s: timed out after %ss (%d results kept)", tool.name, deps.tool_timeout_s, len(out))
        errors.append({"tool": tool.name, "kind": "timeout",
                       "detail": f"timed out after {deps.tool_timeout_s:g}s ({len(out)} results kept)"})
        emit("tool_timeout", tool=tool.name, sub_question_id=sq_id, results=len(out),
             elapsed_s=round(time.monotonic() - started, 1))
        return out
    finally:
        tool_errors.reset(token)
    emit("tool_end", tool=tool.name, sub_question_id=sq_id, results=len(out),
         elapsed_s=round(time.monotonic() - started, 1))
    return out


async def retrieve_node(payload: dict[str, Any], deps: Deps) -> dict[str, Any]:
    started = time.monotonic()
    sq = SubQuestion.model_validate(payload["sub_question"])
    hop = int(payload.get("hop", 0))
    deadline = float(payload.get("deadline") or 0)
    if deadline and time.time() > deadline:
        return {"retrieval_log": [{"sub_question_id": sq.id, "hop": hop, "skipped": "budget exhausted"}],
                "notes": [f"{sq.id}: skipped retrieval, budget exhausted"]}

    tools = deps.tools_for(list(sq.source_types))
    if not tools:
        return {"degradations": [degrade("retriever", "no_tools", f"no enabled tools for {sq.source_types}",
                                         sub_question_id=sq.id)],
                "retrieval_log": [{"sub_question_id": sq.id, "hop": hop, "tools": {}, "stored": 0,
                                   "warning": f"no enabled tools for {sq.source_types}"}],
                "notes": [f"{sq.id}: no enabled tools for {sq.source_types}"]}

    queries = search_queries(sq)
    tool_errs: dict[str, list[dict[str, Any]]] = {t.name: [] for t in tools}
    batches = await asyncio.gather(*(_search_tool(t, queries, deps, sq.id, tool_errs[t.name]) for t in tools),
                                   return_exceptions=True)
    degradations: list[dict[str, Any]] = []
    per_tool: dict[str, int] = {}
    seen: set[str] = set()
    results: list[SearchResult] = []
    for tool, batch in zip(tools, batches, strict=True):
        if isinstance(batch, BaseException):
            per_tool[tool.name] = -1
            degradations.append(degrade(f"tool:{tool.name}", "exception", f"{type(batch).__name__}: {batch}",
                                        sub_question_id=sq.id))
            continue
        per_tool[tool.name] = len(batch)
        for r in batch:
            key = normalise_url(r.url)
            if key in seen:
                continue
            seen.add(key)
            results.append(r)
    for name, errs in tool_errs.items():
        for kind in dict.fromkeys(e["kind"] for e in errs):
            same = [e for e in errs if e["kind"] == kind]
            degradations.append(degrade(f"tool:{name}", kind, same[0]["detail"], count=len(same),
                                        sub_question_id=sq.id))
    # Language gate first: only articles in RETRIEVAL_LANGUAGES (default English) become evidence.
    results, dropped_lang = language_gate(results)
    if dropped_lang:
        emit("language_filter", sub_question_id=sq.id, dropped=dropped_lang)
    # Lexical relevance gate: search APIs rank by text match and return off-topic hits for short
    # keyword queries (Wikipedia especially). The Source Validator (Phase 3) will grade with a model;
    # until then require a minimal overlap with the sub-question's vocabulary.
    kws = keywords(sq)
    before = len(results)
    scored = sorted(((relevance(r, kws), r) for r in results), key=lambda x: -x[0])
    results = [r for score, r in scored if score >= deps.min_relevance]
    filtered = before - len(results)
    results = results[: deps.max_per_sub_question]
    await _maybe_enrich(results, deps)

    evidence = [to_evidence(r, sq.id) for r in results if r.usable()]
    stored = 0
    if evidence:
        store = deps.store()
        emit("embed_start", sub_question_id=sq.id, sources=len(evidence))
        stored = await asyncio.to_thread(store.add_evidence, evidence)   # embedding is CPU-bound
        emit("embed_end", sub_question_id=sq.id, chunks=stored)

    elapsed = round(time.monotonic() - started, 1)
    return {
        "degradations": degradations,
        "evidence": [e.model_dump() for e in evidence],
        "retrieval_log": [{"sub_question_id": sq.id, "hop": hop, "tools": per_tool, "sources": len(evidence),
                           "chunks_stored": stored, "elapsed_s": elapsed, "queries": queries,
                           "filtered_irrelevant": filtered, "filtered_language": sum(dropped_lang.values()),
                           "languages_dropped": dropped_lang}],
        "notes": [f"{sq.id}: {len(evidence)} sources / {stored} chunks via {', '.join(per_tool)} "
                  f"in {elapsed}s"],
    }
