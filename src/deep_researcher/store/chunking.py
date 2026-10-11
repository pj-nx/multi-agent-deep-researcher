"""Token-budgeted chunking with overlap, breaking on paragraph/sentence boundaries.

Defaults come from env so the whole app agrees on one chunk shape:

    CHUNK_SIZE=512      budget per chunk (default 512)
    CHUNK_OVERLAP=64    carried into the next chunk (default 64)
    CHUNK_UNIT=tokens   tokens | chars  (default tokens)

Tokens are counted with tiktoken's ``cl100k_base`` when it is available (it is an approximation
for non-OpenAI embedding models, within a few percent for English). If tiktoken or its encoding
file is unavailable (offline first run), a 4-characters-per-token heuristic is used instead.
"""
from __future__ import annotations

import functools
import logging
import os
from collections.abc import Callable

_BREAKS = ("\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ")
CHARS_PER_TOKEN_FALLBACK = 4.0
log = logging.getLogger(__name__)


def _env_int(name: str, default: int) -> int:
    raw = (os.getenv(name) or "").strip()
    return int(raw) if raw.isdigit() else default


def default_chunk_size() -> int:
    return _env_int("CHUNK_SIZE", 512)


def default_chunk_overlap() -> int:
    return _env_int("CHUNK_OVERLAP", 64)


def default_chunk_unit() -> str:
    unit = (os.getenv("CHUNK_UNIT") or "tokens").strip().lower()
    if unit not in {"tokens", "chars"}:
        raise ValueError(f"CHUNK_UNIT={unit!r}; expected 'tokens' or 'chars'")
    return unit


@functools.cache
def _token_counter() -> Callable[[str], int]:
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return lambda text: len(enc.encode(text, disallowed_special=()))
    except Exception as exc:  # noqa: BLE001 - offline or tiktoken missing: degrade to a heuristic
        log.warning("tiktoken unavailable (%s); estimating tokens as chars/%s", exc, CHARS_PER_TOKEN_FALLBACK)
        return lambda text: max(1, round(len(text) / CHARS_PER_TOKEN_FALLBACK))


def count_tokens(text: str) -> int:
    return _token_counter()(text)


def _chunk_chars(text: str, size: int, overlap: int) -> list[str]:
    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + size, n)
        if end < n:
            window = text[start:end]
            for brk in _BREAKS:
                cut = window.rfind(brk)
                if cut > size // 2:
                    end = start + cut + len(brk)
                    break
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return chunks


def chunk_text(text: str, *, size: int | None = None, overlap: int | None = None,
               unit: str | None = None) -> list[str]:
    """Split ``text`` into chunks of at most ``size`` units with ``overlap`` units carried over."""
    size = size if size is not None else default_chunk_size()
    overlap = overlap if overlap is not None else default_chunk_overlap()
    unit = unit or default_chunk_unit()
    if size <= 0:
        raise ValueError("size must be positive")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must be >= 0 and < size")
    text = text.strip()
    if not text:
        return []
    if unit == "chars":
        return _chunk_chars(text, size, overlap)
    if unit != "tokens":
        raise ValueError(f"unit must be 'tokens' or 'chars', got {unit!r}")

    # Token mode: convert the budget to characters using this text's own chars/token ratio, chunk
    # on natural boundaries, then tighten the ratio until every chunk fits the token budget.
    total_tokens = count_tokens(text)
    if total_tokens <= size:
        return [text]
    ratio = len(text) / total_tokens
    for _ in range(8):
        chunks = _chunk_chars(text, max(1, int(size * ratio)), max(0, int(overlap * ratio)))
        if all(count_tokens(c) <= size for c in chunks):
            return chunks
        ratio *= 0.9
    return [c for chunk in chunks for c in _hard_split(chunk, size)]  # pragma: no cover - pathological text


def _hard_split(chunk: str, size: int) -> list[str]:
    """Last resort for text whose chars/token ratio varies wildly: split on raw token windows."""
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        toks = enc.encode(chunk, disallowed_special=())
        return [enc.decode(toks[i:i + size]) for i in range(0, len(toks), size)]
    except Exception as exc:  # noqa: BLE001
        log.warning("token-based splitting failed (%s); falling back to character windows", exc)
        step = int(size * CHARS_PER_TOKEN_FALLBACK)
        return [chunk[i:i + step] for i in range(0, len(chunk), step)]
