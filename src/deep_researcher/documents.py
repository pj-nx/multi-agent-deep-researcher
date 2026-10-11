"""Uploaded documents as a research source (PLAN.md Phase 8).

A *document set* groups the files a user attaches to a run. Each file is parsed here (PDF via pypdf,
TXT / Markdown, HTML, DOCX when ``python-docx`` is installed), split into page-aligned *sections* of
roughly ``UPLOAD_SECTION_TOKENS`` tokens, and stored by :mod:`deep_researcher.store.uploads`. At run
time :class:`deep_researcher.tools.documents.DocumentsTool` searches the set's sections for each
sub-question and hands them to the ordinary retrieval pipeline, so uploaded text is validated,
chunked, cited and listed in the bibliography exactly like a web page or a paper.

Document text is data, never instructions: it is rendered inside the same delimiters as fetched
pages. Parsing is pypdf only (no JavaScript, no external fetches), the format is sniffed from the
bytes, control characters are stripped and every cap is enforced before parsing.
"""

from __future__ import annotations

import hashlib
import io
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from deep_researcher.agents.deps import Deps
from deep_researcher.schemas import DocumentInfo
from deep_researcher.store.chunking import chunk_text, count_tokens

UPLOAD_SCHEME = "upload://"
FORMATS: tuple[str, ...] = ("pdf", "txt", "md", "html", "docx")


class DocumentError(ValueError):
    """The file cannot be accepted (too big, too many pages, unsupported format, set full)."""


# ----------------------------------------------------------------------------- limits


def _env_int(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name) or default))
    except ValueError:
        return default


def max_file_bytes() -> int:
    return _env_int("UPLOAD_MAX_FILE_MB", 20) * 1_000_000


def max_pages() -> int:
    return _env_int("UPLOAD_MAX_PAGES", 200)


def max_docs_per_set() -> int:
    return _env_int("UPLOAD_MAX_DOCS_PER_SET", 10)


def section_tokens() -> int:
    return _env_int("UPLOAD_SECTION_TOKENS", 1500)


def retention_days() -> float:
    try:
        return float(os.getenv("UPLOAD_RETENTION_DAYS") or 7)
    except ValueError:
        return 7.0


def limits() -> dict[str, Any]:
    """Health payload."""
    return {
        "formats": list(FORMATS),
        "max_file_mb": max_file_bytes() // 1_000_000,
        "max_pages": max_pages(),
        "max_docs_per_set": max_docs_per_set(),
        "section_tokens": section_tokens(),
        "retention_days": retention_days(),
    }


# ----------------------------------------------------------------------------- urls


def new_set_id() -> str:
    return uuid.uuid4().hex


def doc_id_for(set_id: str, filename: str, data: bytes) -> str:
    return hashlib.sha1(f"{set_id}:{filename}:".encode() + hashlib.sha1(data).digest()).hexdigest()[:12]


def document_url(set_id: str, doc_id: str, page_start: int | None = None) -> str:
    """``upload://<set>/<doc>/p<page>``: the page is a path segment, not a fragment, because the retriever
    de-duplicates sources by normalised URL (fragments dropped) and every section must stay a distinct
    piece of evidence with its own id."""
    base = f"{UPLOAD_SCHEME}{set_id}/{doc_id}"
    return f"{base}/p{page_start}" if page_start else base


def is_upload_url(url: str) -> bool:
    return url.startswith(UPLOAD_SCHEME)


def parse_upload_url(url: str) -> tuple[str, str, int | None] | None:
    """``(set_id, doc_id, page)`` for an ``upload://`` URL, else None."""
    if not is_upload_url(url):
        return None
    parts = url[len(UPLOAD_SCHEME) :].split("#", 1)[0].split("/")
    set_id = parts[0] if parts else ""
    doc_id = parts[1] if len(parts) > 1 else ""
    page = None
    if len(parts) > 2 and parts[2].startswith("p") and parts[2][1:].isdigit():
        page = int(parts[2][1:])
    return set_id, doc_id, page


def display_source(title: str, url: str) -> str:
    """Bibliography text for an uploaded section ("Uploaded: file.pdf, p. 3-4") or the URL itself."""
    if not is_upload_url(url):
        return url
    name, _, pages = title.partition(" p.")
    return f"Uploaded: {name}" + (f", p. {pages}" if pages else "")


# ----------------------------------------------------------------------------- parsing


@dataclass
class ParsedDocument:
    kind: str  # pdf | txt | md | html | docx
    pages: list[str]  # one entry per page (text formats: one page)
    warnings: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n\n".join(p for p in self.pages if p)

    @property
    def no_text(self) -> bool:
        return not self.text.strip()


_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def clean_text(text: str) -> str:
    text = _CONTROL.sub(" ", text.replace("\r\n", "\n").replace("\r", "\n"))
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def sniff_kind(filename: str, data: bytes) -> str:
    """Format from the bytes first (a renamed file must not pick its parser by extension)."""
    head = data[:1024].lstrip()
    if head.startswith(b"%PDF-"):
        return "pdf"
    if data[:4] == b"PK\x03\x04" and b"word/" in data[:4000]:
        return "docx"
    low = head[:256].lower()
    if low.startswith(b"<!doctype html") or low.startswith(b"<html") or b"<html" in low:
        return "html"
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext in ("md", "markdown"):
        return "md"
    if ext in ("htm", "html"):
        return "html"
    if ext == "pdf":
        raise DocumentError(f"{filename}: not a PDF (no %PDF header)")
    if ext == "docx":
        raise DocumentError(f"{filename}: not a DOCX file")
    return "txt"


def _pdf_pages(data: bytes, limit: int) -> tuple[list[str], list[str]]:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    warnings = []
    total = len(reader.pages)
    if total > limit:
        warnings.append(f"only the first {limit} of {total} pages were read")
    pages = []
    for page in reader.pages[:limit]:
        try:
            pages.append(clean_text(page.extract_text() or ""))
        except Exception as exc:  # noqa: BLE001 - one broken page must not sink the document
            warnings.append(f"page {len(pages) + 1}: {type(exc).__name__}")
            pages.append("")
    return pages, warnings


def _docx_text(data: bytes) -> str:
    try:
        import docx  # type: ignore[import-not-found]
    except ImportError as exc:
        raise DocumentError("DOCX support needs the 'documents' extra: uv sync --extra documents") from exc
    d = docx.Document(io.BytesIO(data))
    parts = [p.text for p in d.paragraphs]
    for table in d.tables:
        for row in table.rows:
            parts.append(" | ".join(c.text for c in row.cells))
    return "\n".join(parts)


def parse_document(filename: str, data: bytes) -> ParsedDocument:
    """Parse one file into pages. Raises :class:`DocumentError` for files that must be refused."""
    if not data:
        raise DocumentError(f"{filename}: empty file")
    if len(data) > max_file_bytes():
        raise DocumentError(
            f"{filename}: {len(data) / 1e6:.1f} MB exceeds the {max_file_bytes() // 1_000_000} MB limit"
        )
    kind = sniff_kind(filename, data)
    if kind == "pdf":
        pages, warnings = _pdf_pages(data, max_pages())
        return ParsedDocument(kind, pages, warnings)
    if kind == "docx":
        return ParsedDocument(kind, [clean_text(_docx_text(data))])
    text = data.decode("utf-8", errors="replace")
    if kind == "html":
        from deep_researcher.tools.fetch import html_to_text

        text = html_to_text(text)
    return ParsedDocument(kind, [clean_text(text)])


# ----------------------------------------------------------------------------- sections


@dataclass(frozen=True)
class Section:
    index: int
    page_start: int
    page_end: int
    text: str

    @property
    def pages_label(self) -> str:
        return (
            f"{self.page_start}" if self.page_start == self.page_end else f"{self.page_start}-{self.page_end}"
        )


def split_sections(pages: list[str], *, target_tokens: int | None = None) -> list[Section]:
    """Page-aligned sections of about ``target_tokens``: consecutive pages are merged until the target
    is reached; a single page larger than twice the target is split on paragraph boundaries."""
    target = target_tokens or section_tokens()
    sections: list[Section] = []
    buf: list[str] = []
    buf_tokens = 0
    start = 1

    def flush(end: int) -> None:
        nonlocal buf, buf_tokens, start
        text = "\n\n".join(buf).strip()
        if text:
            sections.append(Section(len(sections), start, end, text))
        buf, buf_tokens = [], 0
        start = end + 1

    for n, page in enumerate(pages, start=1):
        page = page.strip()
        if not page:
            if not buf:
                start = n + 1
            continue
        tokens = count_tokens(page)
        if tokens > 2 * target:
            flush(n - 1)
            for piece in chunk_text(page, size=target, overlap=0):
                sections.append(Section(len(sections), n, n, piece))
            start = n + 1
            continue
        if buf and buf_tokens + tokens > target:
            flush(n - 1)
        buf.append(page)
        buf_tokens += tokens
    flush(len(pages))
    return sections


def describe(
    set_id: str,
    filename: str,
    data: bytes,
    parsed: ParsedDocument,
    sections: list[Section],
    language: str | None = None,
) -> DocumentInfo:
    return DocumentInfo(
        doc_id=doc_id_for(set_id, filename, data),
        filename=filename[:200],
        kind=parsed.kind,
        pages=len(parsed.pages),
        sections=len(sections),
        chars=len(parsed.text),
        no_text=parsed.no_text,
        warnings=parsed.warnings[:5],
        language=language,
        excerpt=parsed.text[:300],
    )


def excerpt_lines(documents: list[dict[str, Any]]) -> str:
    """Manifest lines for the planner prompt."""
    lines = []
    for d in documents:
        flag = " (no extractable text)" if d.get("no_text") else ""
        ex = (d.get("excerpt") or "").replace("\n", " ")[:200]
        lines.append(f"- {d['filename']} ({d.get('kind')}, {d.get('pages')} page(s)){flag}: {ex}")
    return "\n".join(lines)


# ----------------------------------------------------------------------------- run binding


def bind_documents(deps: Deps, set_id: str, store: Any = None) -> Deps:
    """A ``Deps`` whose ``tools_for`` also yields this run's :class:`DocumentsTool` for ``documents``."""
    from dataclasses import replace

    from deep_researcher.tools.documents import DocumentsTool

    base = deps.tools_for

    def tools_for(source_types: list[str]) -> list[Any]:
        tools = list(base(source_types))
        if "documents" in source_types:
            tools.append(DocumentsTool(set_id, store=store))
        return tools

    return replace(deps, tools_for=tools_for)


def stamp() -> float:
    return time.time()


def manifest_markdown(documents: list[dict[str, Any]]) -> str:
    """Short manifest for the UIs."""
    if not documents:
        return ""
    lines = []
    for d in documents:
        flag = " · **no extractable text** (scanned? not searchable)" if d.get("no_text") else ""
        warn = f" · {'; '.join(d['warnings'])}" if d.get("warnings") else ""
        lines.append(f"- **{d['filename']}** · {d.get('kind')} · {d.get('pages')} page(s) · "
                     f"{d.get('sections')} section(s){flag}{warn}")
    return "\n".join(lines)


async def graph_for_documents(documents: dict[str, Any] | None, *, deps: Deps | None = None,
                              store: Any = None):
    """A compiled graph whose retrievers can search ``documents['set_id']`` (None = the shared graph)."""
    from deep_researcher import graph as G

    if not documents or not documents.get("set_id"):
        return await G.graph_for(deps) if deps else None
    return await G.graph_for(bind_documents(deps or Deps(), documents["set_id"], store=store))


def ingest(
    files: list[tuple[str, bytes]],
    *,
    set_id: str | None = None,
    store: Any = None,
) -> tuple[str, list[DocumentInfo]]:
    """Parse, section, embed and store files; returns ``(set_id, manifest)``. Raises DocumentError."""
    from deep_researcher.store.uploads import get_upload_store
    from deep_researcher.tools import language as lang

    store = store or get_upload_store()
    set_id = set_id or new_set_id()
    existing = store.manifest(set_id)
    if len(existing) + len(files) > max_docs_per_set():
        raise DocumentError(f"a document set holds at most {max_docs_per_set()} files")
    manifest = list(existing)
    for filename, data in files:
        parsed = parse_document(filename, data)
        sections = [] if parsed.no_text else split_sections(parsed.pages)
        code = None
        if not parsed.no_text:
            _, code = lang.classify(parsed.text[:2000], None, lang.allowed_languages())
        info = describe(set_id, filename, data, parsed, sections, language=code)
        store.add_document(set_id, info, sections)
        manifest.append(info)
    return set_id, manifest
