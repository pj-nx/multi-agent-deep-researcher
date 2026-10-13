# Copyright (c) 2026 C10 Hackathon G1
"""Read saved research reports for the optional Streamlit dashboard."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from deep_researcher.reports import reports_dir


def list_reports(directory: Path | None = None) -> list[Path]:
    """Return saved JSON reports, newest first."""
    root = directory or reports_dir()
    return sorted(root.glob("*.json"), key=lambda path: (path.stat().st_mtime_ns, path.name), reverse=True)


def load_report(path: Path) -> dict[str, Any]:
    """Load one report and reject malformed or unexpected JSON shapes."""
    with path.open(encoding="utf-8") as report_file:
        report = json.load(report_file)
    if not isinstance(report, dict):
        raise ValueError(f"report {path} must contain a JSON object")
    return report


def load_markdown(path: Path) -> str | None:
    """Load a matching Markdown report when it exists."""
    markdown_path = path.with_suffix(".md")
    return markdown_path.read_text(encoding="utf-8") if markdown_path.is_file() else None


def auth_required(password: str | None) -> bool:
    """Authentication is disabled unless a dashboard password is configured."""
    return bool(password)


def verify_password(candidate: str, configured: str | None) -> bool:
    """Constant-time password check; an unconfigured dashboard remains open."""
    import hmac

    return not configured or hmac.compare_digest(candidate, configured)
