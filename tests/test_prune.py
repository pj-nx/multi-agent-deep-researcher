# Copyright (c) 2026 C10 Hackathon G1
"""Checkpoint pruning: old completed threads go, recent and unfinished ones stay."""
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest
from langgraph.checkpoint.base import empty_checkpoint
from langgraph.checkpoint.memory import InMemorySaver

from deep_researcher.prune import prune


async def _put(saver, thread_id, *, age_days, report):
    cp = empty_checkpoint()
    cp["ts"] = (datetime.now(UTC) - timedelta(days=age_days)).isoformat()
    cp["channel_values"] = {"report": "# done"} if report else {}
    cp["channel_versions"] = {"report": "1"} if report else {}
    cfg = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
    await saver.aput(cfg, cp, {"source": "loop", "step": 1, "parents": {}}, cp["channel_versions"])


async def _threads(saver):
    return {t.config["configurable"]["thread_id"] async for t in saver.alist(None)}


async def _seeded():
    s = InMemorySaver()
    await _put(s, "old-done", age_days=6, report=True)
    await _put(s, "new-done", age_days=1, report=True)
    await _put(s, "old-pending", age_days=6, report=False)
    return s


@pytest.mark.anyio
async def test_deletes_only_old_completed():
    saver = await _seeded()
    assert await prune(saver, days=4) == ["old-done"]
    assert await _threads(saver) == {"new-done", "old-pending"}


@pytest.mark.anyio
async def test_dry_run_changes_nothing():
    saver = await _seeded()
    assert await prune(saver, days=4, dry_run=True) == ["old-done"]
    assert await _threads(saver) == {"old-done", "new-done", "old-pending"}


@pytest.mark.anyio
async def test_include_unfinished():
    saver = await _seeded()
    assert set(await prune(saver, days=4, include_unfinished=True)) == {"old-done", "old-pending"}
    assert await _threads(saver) == {"new-done"}


@pytest.mark.anyio
async def test_sqlite_created_at_column_drives_pruning(tmp_path, monkeypatch):
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    from deep_researcher import graph as G

    monkeypatch.setenv("CHECKPOINTER", "sqlite")
    monkeypatch.setenv("CHECKPOINT_DB", str(tmp_path / "cp.sqlite"))
    saver = await G.get_checkpointer()
    assert isinstance(saver, AsyncSqliteSaver)
    try:
        await _put(saver, "old-done", age_days=0, report=True)
        await _put(saver, "new-done", age_days=0, report=True)
        await _put(saver, "old-pending", age_days=0, report=False)
        rows = await (await saver.conn.execute("SELECT thread_id, created_at FROM checkpoints")).fetchall()
        assert all(created for _, created in rows)                 # trigger dated every insert
        await saver.conn.execute("UPDATE checkpoints SET created_at = '2000-01-01T00:00:00Z' "
                                 "WHERE thread_id LIKE 'old-%'")
        await saver.conn.commit()
        assert await prune(saver, days=4) == ["old-done"]
        assert await _threads(saver) == {"new-done", "old-pending"}
    finally:
        await saver.conn.close()
        G.reset_graph()
