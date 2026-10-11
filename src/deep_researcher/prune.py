"""Delete old graph runs from the checkpointer (sqlite or postgres, whichever ``CHECKPOINTER`` selects).

    uv run python -m deep_researcher.prune [--days 4] [--include-unfinished] [--dry-run]

A thread counts as *completed* when its latest checkpoint holds a report. Threads still waiting for
plan approval (or that failed before reporting) are kept unless ``--include-unfinished`` is given.
Age is the ``created_at`` date of the thread's newest checkpoint (a column this project adds to
the ``checkpoints`` table; see ``graph.get_checkpointer``), compared in SQL so only old threads are
loaded. The in-memory saver has no such column and falls back to the checkpoint's own timestamp.
Deleting a thread removes its checkpoints and writes; the Chroma evidence store is untouched.
"""
from __future__ import annotations

import argparse
import asyncio
import os
from datetime import UTC, datetime, timedelta
from typing import Any

from dotenv import load_dotenv


async def _latest_by_thread(saver: Any) -> dict[str, Any]:
    """Newest checkpoint tuple per thread (checkpoints are listed newest first)."""
    latest: dict[str, Any] = {}
    async for tup in saver.alist(None):
        latest.setdefault(tup.config["configurable"]["thread_id"], tup)
    return latest


async def _stale_thread_ids(saver: Any, cutoff: datetime) -> list[str] | None:
    """Threads whose newest ``created_at`` is before ``cutoff``; None when the saver has no such column."""
    query = "SELECT thread_id FROM checkpoints GROUP BY thread_id HAVING MAX(created_at) < "
    if type(saver).__name__ == "AsyncSqliteSaver":
        cur = await saver.conn.execute(query + "?", (cutoff.strftime("%Y-%m-%dT%H:%M:%SZ"),))
        return [row[0] for row in await cur.fetchall()]
    if type(saver).__name__ == "AsyncPostgresSaver":
        cur = await saver.conn.execute(query + "%s", (cutoff,))
        return [row["thread_id"] for row in await cur.fetchall()]
    return None


def _is_old(checkpoint: dict[str, Any], cutoff: datetime) -> bool:
    return datetime.fromisoformat(checkpoint["ts"]).astimezone(UTC) < cutoff


async def prune(saver: Any, *, days: float = 4, include_unfinished: bool = False,
                dry_run: bool = False, now: datetime | None = None) -> list[str]:
    """Delete threads whose newest checkpoint is older than ``days``; returns the thread ids affected."""
    cutoff = (now or datetime.now(UTC)) - timedelta(days=days)
    stale = await _stale_thread_ids(saver, cutoff)
    doomed = []
    if stale is None:       # no created_at column: judge by each thread's checkpoint timestamp
        candidates = {t: tup for t, tup in (await _latest_by_thread(saver)).items()
                      if _is_old(tup.checkpoint, cutoff)}
    else:
        candidates = {}
        for thread_id in stale:
            tup = await saver.aget_tuple({"configurable": {"thread_id": thread_id}})
            if tup is not None:
                candidates[thread_id] = tup
    for thread_id, tup in candidates.items():
        if include_unfinished or tup.checkpoint.get("channel_values", {}).get("report"):
            doomed.append(thread_id)
    if not dry_run:
        for thread_id in doomed:
            await saver.adelete_thread(thread_id)
    return doomed


def _vacuum_sqlite(path: str) -> None:
    """Shrink the file after deletes (sqlite only frees pages internally otherwise)."""
    import sqlite3

    conn = sqlite3.connect(path, isolation_level=None)
    try:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        conn.execute("VACUUM")
    finally:
        conn.close()


async def _main(args: argparse.Namespace) -> int:
    from deep_researcher import graph as G

    saver = await G.get_checkpointer()
    try:
        removed = await prune(saver, days=args.days, include_unfinished=args.include_unfinished,
                              dry_run=args.dry_run)
    finally:
        conn = getattr(saver, "conn", None)      # the in-memory saver has no connection
        if conn is not None:
            await conn.close()
    verb = "would delete" if args.dry_run else "deleted"
    print(f"{verb} {len(removed)} run(s) older than {args.days:g} day(s)")
    if args.documents:
        from deep_researcher.store.uploads import get_upload_store

        store = get_upload_store()
        sets = store.expired_sets(args.days) if args.dry_run else store.expire(args.days)
        print(f"{verb} {len(sets)} uploaded document set(s) older than {args.days:g} day(s)")
    if removed and not args.dry_run and G.checkpointer_kind() == "sqlite":
        _vacuum_sqlite(os.getenv("CHECKPOINT_DB") or ".checkpoints.sqlite")
    return 0


def main() -> int:
    load_dotenv()
    p = argparse.ArgumentParser(prog="deep_researcher.prune", description=__doc__.split("\n\n")[0])
    p.add_argument("--days", type=float, default=4, help="delete runs whose last activity is older than this")
    p.add_argument("--include-unfinished", action="store_true",
                   help="also delete runs with no report (abandoned plan approvals, failures)")
    p.add_argument("--dry-run", action="store_true", help="report what would be deleted, change nothing")
    p.add_argument("--documents", action="store_true",
                   help="also delete uploaded document sets older than --days (the 'uploads' collection)")
    return asyncio.run(_main(p.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
