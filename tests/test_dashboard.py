# Copyright (c) 2026 C10 Hackathon G1
import json
import os

import pytest

from deep_researcher.dashboard_data import (
    auth_required,
    list_reports,
    load_markdown,
    load_report,
    verify_password,
)


def test_dashboard_auth_is_optional_and_password_checked() -> None:
    assert not auth_required(None)
    assert not auth_required("")
    assert verify_password("", None)
    assert verify_password("secret", "secret")
    assert not verify_password("incorrect", "secret")


def test_report_listing_is_newest_first_and_loads_markdown(tmp_path) -> None:
    old = tmp_path / "old.json"
    old.write_text(json.dumps({"question": "older"}), encoding="utf-8")
    new = tmp_path / "new.json"
    new.write_text(json.dumps({"question": "newer"}), encoding="utf-8")
    (tmp_path / "new.md").write_text("# New report", encoding="utf-8")
    old_time = old.stat().st_mtime_ns
    new_time = old_time + 1_000_000_000
    os.utime(old, ns=(old_time, old_time))
    os.utime(new, ns=(new_time, new_time))

    assert list_reports(tmp_path) == [new, old]
    assert load_report(new) == {"question": "newer"}
    assert load_markdown(new) == "# New report"
    assert load_markdown(old) is None


def test_report_loader_rejects_non_object_json(tmp_path) -> None:
    report = tmp_path / "invalid.json"
    report.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="JSON object"):
        load_report(report)


def test_password_check_handles_non_ascii() -> None:
    assert verify_password("pässwort", "pässwort")
    assert not verify_password("é", "secret")
    assert not verify_password("secret", "pässwort")


def test_dashboard_binds_localhost_unless_password_set() -> None:
    from deep_researcher.dashboard import launch_command

    assert launch_command(None)[-2:] == ["--server.address", "127.0.0.1"]
    assert "--server.address" not in launch_command("secret")


def test_cli_passes_out_dir_to_dashboard_and_rejects_question(monkeypatch, tmp_path) -> None:
    import sys

    import deep_researcher.__main__ as cli
    import deep_researcher.dashboard as dashboard

    launched = []
    monkeypatch.setattr(dashboard, "launch_dashboard", lambda out=None: launched.append(out) or 0)
    monkeypatch.setattr(sys, "argv", ["deep_researcher", "--dashboard", "--out", str(tmp_path)])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 0 and launched == [tmp_path]

    monkeypatch.setattr(sys, "argv", ["deep_researcher", "a question", "--dashboard"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2 and len(launched) == 1
