"""Shared plumbing for retrieval tools.

Every tool is a small class with ``name``, ``source_type``, the env vars it needs, and an async
``search(query, limit)`` that never raises: network errors, 429s and parse failures are logged and
yield an empty list, because one flaky source must not sink a parallel retrieval branch.
Per-tool throttling (min seconds between calls) keeps us inside the public APIs' etiquette.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
import weakref
from dataclasses import dataclass, field
from typing import Any, ClassVar

import httpx

from deep_researcher.health import record_tool_error
from deep_researcher.schemas import SourceType

log = logging.getLogger(__name__)

USER_AGENT = os.getenv(
    "RETRIEVAL_USER_AGENT",
    "deep-researcher/0.1 (+https://github.com/pj-nx/multi-agent-deep-researcher)",
)
DEFAULT_TIMEOUT = float(os.getenv("RETRIEVAL_TIMEOUT_SECONDS") or 30)

CACHE_TTL_SECONDS = float(os.getenv("RETRIEVAL_CACHE_TTL_SECONDS") or 600)   # 0 disables

# Search results per HTTP client: {(tool, query, limit): (stored_at, results)}, plus one lock per key so
# parallel sub-questions asking the same thing wait for the first answer instead of re-querying.
_search_cache: weakref.WeakKeyDictionary[httpx.AsyncClient, dict[str, dict[Any, Any]]] = (
    weakref.WeakKeyDictionary())

_shared_client: httpx.AsyncClient | None = None


def shared_client() -> httpx.AsyncClient:
    """One connection pool per process; tools accept an explicit client for tests."""
    global _shared_client
    if _shared_client is None or _shared_client.is_closed:
        _shared_client = httpx.AsyncClient(timeout=DEFAULT_TIMEOUT, follow_redirects=True,
                                           headers={"User-Agent": USER_AGENT})
    return _shared_client


@dataclass
class SearchResult:
    title: str
    url: str
    text: str
    source_type: SourceType
    published_at: str | None = None          # ISO-8601 date when known
    extra: dict[str, Any] = field(default_factory=dict)

    def usable(self) -> bool:
        return bool(self.url) and bool(self.text.strip())


class RetrievalTool:
    name: ClassVar[str] = "base"
    source_type: ClassVar[SourceType] = "web"
    requires_env: ClassVar[tuple[str, ...]] = ()
    min_interval_s: ClassVar[float] = 0.0      # politeness gap between calls to this API
    max_retries: ClassVar[int] = 1

    _locks: ClassVar[dict[str, asyncio.Lock]] = {}
    _last_call: ClassVar[dict[str, float]] = {}

    def __init__(self, client: httpx.AsyncClient | None = None) -> None:
        self._client = client

    @property
    def client(self) -> httpx.AsyncClient:
        return self._client or shared_client()

    @classmethod
    def enabled(cls) -> bool:
        return all(os.getenv(v) for v in cls.requires_env)

    @classmethod
    def describe(cls) -> dict[str, Any]:
        return {"name": cls.name, "source_type": cls.source_type, "requires": list(cls.requires_env),
                "enabled": cls.enabled()}

    async def _throttle(self) -> None:
        if self.min_interval_s <= 0:
            return
        lock = self._locks.setdefault(self.name, asyncio.Lock())
        async with lock:
            wait = self._last_call.get(self.name, 0.0) + self.min_interval_s - time.monotonic()
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_call[self.name] = time.monotonic()

    async def _get(self, url: str, *, params: dict[str, Any] | None = None,
                   headers: dict[str, str] | None = None) -> httpx.Response:
        return await self._request("GET", url, params=params, headers=headers)

    async def _request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        """One request with throttling and a single retry on 429/5xx honouring Retry-After."""
        attempt = 0
        while True:
            await self._throttle()
            resp = await self.client.request(method, url, **kwargs)
            if resp.status_code in (429, 500, 502, 503, 504) and attempt < self.max_retries:
                attempt += 1
                retry_after = resp.headers.get("Retry-After")
                delay = max(2.0, self.min_interval_s)
                if retry_after and retry_after.isdigit():
                    delay = float(retry_after)
                log.info("%s: HTTP %s, retrying in %.1fs", self.name, resp.status_code, delay)
                await asyncio.sleep(min(delay, 15.0))
                continue
            resp.raise_for_status()
            return resp

    async def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        """Never raises. Returns usable results only; identical searches within the TTL are answered once."""
        if CACHE_TTL_SECONDS <= 0:
            return await self._search_uncached(query, limit)
        store = _search_cache.setdefault(self.client, {"results": {}, "locks": {}})
        key = (self.name, query, limit)
        async with store["locks"].setdefault(key, asyncio.Lock()):
            hit = store["results"].get(key)
            if hit and time.monotonic() - hit[0] < CACHE_TTL_SECONDS:
                log.info("%s: cached results for %r", self.name, query[:60])
                return list(hit[1])
            results, ok = await self._search_checked(query, limit)
            if ok:                       # failures are never cached: the next caller retries
                store["results"][key] = (time.monotonic(), results)
            return list(results)

    async def _search_uncached(self, query: str, limit: int) -> list[SearchResult]:
        return (await self._search_checked(query, limit))[0]

    async def _search_checked(self, query: str, limit: int) -> tuple[list[SearchResult], bool]:
        """(results, succeeded). Failures are reported through health.record_tool_error."""
        try:
            results = await self._search(query, limit)
        except httpx.HTTPStatusError as exc:
            code = exc.response.status_code
            log.warning("%s: HTTP %s for %r", self.name, code, query[:80])
            record_tool_error(self.name, "rate_limited" if code == 429 else f"http_{code}",
                              f"HTTP {code} for {query[:60]!r}")
            return [], False
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
            log.warning("%s: failed for %r: %s", self.name, query[:80], str(exc)[:160])
            record_tool_error(self.name, "request_failed", f"{type(exc).__name__}: {str(exc)[:120]}")
            return [], False
        return [r for r in results if r.usable()][:limit], True

    async def _search(self, query: str, limit: int) -> list[SearchResult]:  # pragma: no cover - abstract
        raise NotImplementedError


def iso_date(value: str | None) -> str | None:
    """Normalise the date formats these APIs return to YYYY-MM-DD (or None)."""
    if not value:
        return None
    v = value.strip()
    if len(v) >= 10 and v[4] == "-" and v[7] == "-":
        return v[:10]
    if len(v) >= 8 and v[:8].isdigit():                 # GDELT: 20250131T120000Z
        return f"{v[:4]}-{v[4:6]}-{v[6:8]}"
    if len(v) == 4 and v.isdigit():                     # year only
        return f"{v}-01-01"
    return None


def clip(text: str, limit: int = 20_000) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit]
