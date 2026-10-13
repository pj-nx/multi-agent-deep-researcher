# Copyright (c) 2026 C10 Hackathon G1
"""Optional Streamlit dashboard for browsing generated research reports.

Run with ``uv run streamlit run src/deep_researcher/dashboard.py -- --reports-dir reports``.
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

    from deep_researcher.reports import reports_dir

    st.set_page_config(page_title="Deep Researcher", page_icon="🔎", layout="wide")
    st.title("Multi-Agent Deep Researcher")
    if not _authenticated(st, os.getenv("DASHBOARD_PASSWORD")):
        st.info("Sign in to view saved reports.")
        return

    report_dir = Path(os.getenv("REPORTS_DIR") or reports_dir())
    reports = list_reports(report_dir)
    if not reports:
        st.info(f"No saved reports in `{report_dir}` yet. Run a research question with the CLI or Gradio UI.")
        return

    report_data = {path: load_report(path) for path in reports}
    selected = st.selectbox(
        "Saved reports",
        reports,
        format_func=lambda path: f"{path.stem} — {report_data[path].get('question', '')[:90]}",
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
