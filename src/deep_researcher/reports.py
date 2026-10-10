"""Writing finished reports to disk (shared by the CLI and the Gradio UI)."""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any


def slugify(text: str, limit: int = 60) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower())[:limit].strip("-") or "report"


def reports_dir(default: str = "reports") -> Path:
    return Path(os.getenv("REPORTS_DIR") or default)


def write_report(question: str, markdown: str, report_json: dict[str, Any] | None,
                 out_dir: Path | None = None) -> tuple[Path, Path | None]:
    """Write ``<timestamp>-<slug>.md`` (+ ``.json``) and return their paths."""
    out = out_dir or reports_dir()
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{time.strftime('%Y%m%d-%H%M%S')}-{slugify(question)}"
    md_path = out / f"{stem}.md"
    md_path.write_text(markdown)
    json_path = None
    if report_json:
        json_path = out / f"{stem}.json"
        json_path.write_text(json.dumps(report_json, indent=2, default=str))
    return md_path, json_path
