"""Plain character chunking with overlap, breaking on paragraph/sentence boundaries when possible."""
from __future__ import annotations

_BREAKS = ("\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ")


def chunk_text(text: str, *, size: int = 1200, overlap: int = 150) -> list[str]:
    if size <= 0:
        raise ValueError("size must be positive")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must be >= 0 and < size")
    text = text.strip()
    if not text:
        return []
    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + size, n)
        if end < n:
            window = text[start:end]
            cut = -1
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
