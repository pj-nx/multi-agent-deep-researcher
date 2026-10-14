# Copyright (c) 2026 C10 Hackathon G1
"""Optional Streamlit dashboard for browsing generated research reports.

Run with ``uv run python -m deep_researcher --dashboard [--out reports]``, or directly with
``REPORTS_DIR=reports uv run streamlit run src/deep_researcher/dashboard.py``.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from deep_researcher.dashboard_data import (
    auth_required,
    list_reports,
    load_markdown,
    load_report,
    verify_password,
)
from deep_researcher.reports import reports_dir


def launch_command(password: str | None) -> list[str]:
    """``streamlit run`` for this file; bound to localhost unless a dashboard password is configured."""
    import sys

    cmd = [sys.executable, "-m", "streamlit", "run", str(Path(__file__))]
    if not auth_required(password):
        cmd += ["--server.address", "127.0.0.1"]
    return cmd


def launch_dashboard(out_dir: Path | None = None) -> int:
    """Launch the dashboard in a Streamlit subprocess; ``out_dir`` overrides ``REPORTS_DIR``."""
    import subprocess

    env = dict(os.environ)
    if out_dir is not None:
        env["REPORTS_DIR"] = str(out_dir)
    return subprocess.call(launch_command(env.get("DASHBOARD_PASSWORD")), env=env)


def _authenticated(st: Any, password: str | None) -> bool:
    if not auth_required(password):
        return True
    if st.session_state.get("dashboard_authenticated"):
        return True
    candidate = st.text_input("Dashboard password", type="password")
    if st.button("Sign in"):
        if verify_password(candidate, password):
            st.session_state["dashboard_authenticated"] = True
            st.rerun()
        else:
            st.error("Incorrect password")
    return False


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="Deep Researcher", page_icon="🔎", layout="wide")
    st.title("Multi-Agent Deep Researcher")
    if not _authenticated(st, os.getenv("DASHBOARD_PASSWORD")):
        st.info("Sign in to view saved reports.")
        return

    # Streamlit reruns this script on every interaction; re-parse a report only when its file changes.
    @st.cache_data(show_spinner=False)
    def read(path: str, mtime_ns: int) -> dict[str, Any]:
        return load_report(Path(path))

    report_dir = reports_dir()
    report_data: dict[Path, dict[str, Any]] = {}
    unreadable: list[str] = []
    for path in list_reports(report_dir):
        try:
            report_data[path] = read(str(path), path.stat().st_mtime_ns)
        except (OSError, ValueError):  # JSONDecodeError is a ValueError; one bad file must not hide the rest
            unreadable.append(path.name)
    if unreadable:
        st.warning(f"Skipped {len(unreadable)} unreadable report(s): {', '.join(unreadable)}")
    if not report_data:
        st.info(f"No saved reports in `{report_dir}` yet. Run a research question with the CLI or Gradio UI.")
        return

    selected = st.selectbox(
        "Saved reports",
        list(report_data),
        format_func=lambda path: f"{path.stem} — {str(report_data[path].get('question') or '')[:90]}",
    )
    report = report_data[selected]
    st.header(str(report.get("title") or selected.stem))
    st.caption(str(report.get("question") or ""))
    stats = report.get("stats") or {}
    if stats:
        cols = st.columns(4)
        for col, (label, key) in zip(
            cols,
            (("Sources", "kept"), ("Claims", "claims"), ("Contradictions", "contradictions"),
             ("Verified citations", "citations_verified")),
            strict=True,
        ):
            col.metric(label, stats.get(key, 0))

    markdown = load_markdown(selected)
    if markdown is not None:
        st.markdown(markdown)
    else:
        st.markdown(str(report.get("executive_summary") or "_No Markdown report was saved._"))
        st.json(report)


if __name__ == "__main__":
    main()
