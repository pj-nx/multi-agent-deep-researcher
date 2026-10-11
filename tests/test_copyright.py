# Copyright (c) 2026 C10 Hackathon G1
"""Every source file and both UIs carry the project's copyright notice."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTICE = "Copyright (c) 2026 C10 Hackathon G1"


def test_every_python_file_has_the_notice():
    missing = []
    for folder in ("src", "api", "tests", "scripts"):
        for p in (ROOT / folder).rglob("*.py"):
            head = "\n".join(p.read_text().splitlines()[:3])
            if NOTICE not in head:
                missing.append(str(p.relative_to(ROOT)))
    assert not missing, f"files without the copyright header: {missing}"


def test_uis_show_the_notice():
    page = (ROOT / "public" / "index.html").read_text()
    assert NOTICE in page and 'id="dr-footer">© 2026 C10 Hackathon G1' in page
    from deep_researcher import ui

    assert ui.COPYRIGHT.startswith("© 2026 C10 Hackathon G1")
    assert NOTICE in (ROOT / "README.md").read_text()
