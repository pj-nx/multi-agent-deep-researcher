"""One-off migration: add ``created_at`` to an existing LangGraph ``checkpoints`` table.

    python scripts/add_checkpoint_created_at.py                 # sqlite: $CHECKPOINT_DB or default
    python scripts/add_checkpoint_created_at.py --db path.sqlite
    python scripts/add_checkpoint_created_at.py --postgres      # uses $DATABASE_URL

Safe to re-run (every step is conditional) and safe to skip: the app applies the same change on
startup. Run it beforehand to date old rows accurately: LangGraph checkpoint ids are time-based
UUIDv6 values, so each existing row gets the moment it was really written (the app's own
start-up migration can only stamp old rows with "now"). Existing rows and their data are not
modified; only the new column is filled. Back the database up first if it matters.
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from datetime import UTC, datetime

_GREGORIAN_TO_UNIX_100NS = 0x01B21DD213814000   # 1582-10-15 -> 1970-01-01, in 100 ns ticks


def uuid6_time(checkpoint_id: str) -> datetime | None:
    """Creation time encoded in a UUIDv6 (time_high|time_mid|ver|time_low); None if not a UUIDv6."""
    h = checkpoint_id.replace("-", "")
    if len(h) != 32 or h[12] != "6":
        return None
    ticks = int(h[0:8] + h[8:12] + h[13:16], 16)
    return datetime.fromtimestamp((ticks - _GREGORIAN_TO_UNIX_100NS) / 1e7, UTC)


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


NOW_SQLITE = "strftime('%Y-%m-%dT%H:%M:%SZ', 'now')"


def migrate_sqlite(path: str) -> str:
    if not os.path.exists(path):
        return f"{path} does not exist; nothing to migrate"
    conn = sqlite3.connect(path)
    try:
        if not conn.execute("SELECT 1 FROM sqlite_master WHERE name = 'checkpoints'").fetchone():
            return "no checkpoints table yet; the app creates and migrates it on first start"
        cols = [r[1] for r in conn.execute("PRAGMA table_info(checkpoints)")]
        if "created_at" not in cols:
            conn.execute("ALTER TABLE checkpoints ADD COLUMN created_at TEXT")
        rows = conn.execute("SELECT thread_id, checkpoint_ns, checkpoint_id FROM checkpoints "
                            "WHERE created_at IS NULL").fetchall()
        updates = [(_iso(uuid6_time(cid) or datetime.now(UTC)), t, ns, cid) for t, ns, cid in rows]
        conn.executemany("UPDATE checkpoints SET created_at = ? WHERE thread_id = ? "
                         "AND checkpoint_ns = ? AND checkpoint_id = ?", updates)
        conn.execute(
            "CREATE TRIGGER IF NOT EXISTS checkpoints_created_at AFTER INSERT ON checkpoints "
            "WHEN NEW.created_at IS NULL BEGIN "
            f"UPDATE checkpoints SET created_at = {NOW_SQLITE} WHERE thread_id = NEW.thread_id "
            "AND checkpoint_ns = NEW.checkpoint_ns AND checkpoint_id = NEW.checkpoint_id; END")
        conn.commit()
        return f"sqlite {path}: created_at ready, dated {len(updates)} existing row(s)"
    finally:
        conn.close()


def migrate_postgres(url: str) -> str:
    import psycopg

    with psycopg.connect(url, autocommit=False) as conn:
        if not conn.execute("SELECT to_regclass('checkpoints')").fetchone()[0]:
            return "no checkpoints table yet; the app creates and migrates it on first start"
        had = conn.execute("SELECT 1 FROM information_schema.columns WHERE table_name = 'checkpoints' "
                           "AND column_name = 'created_at'").fetchone()
        if not had:   # nullable first so old rows can be dated before the NOT NULL default applies
            conn.execute("ALTER TABLE checkpoints ADD COLUMN created_at timestamptz")
        rows = conn.execute("SELECT thread_id, checkpoint_ns, checkpoint_id FROM checkpoints "
                            "WHERE created_at IS NULL").fetchall()
        updates = [(uuid6_time(cid) or datetime.now(UTC), t, ns, cid) for t, ns, cid in rows]
        with conn.cursor() as cur:
            cur.executemany("UPDATE checkpoints SET created_at = %s WHERE thread_id = %s "
                            "AND checkpoint_ns = %s AND checkpoint_id = %s", updates)
        conn.execute("ALTER TABLE checkpoints ALTER COLUMN created_at SET DEFAULT now()")
        conn.execute("ALTER TABLE checkpoints ALTER COLUMN created_at SET NOT NULL")
        return f"postgres: created_at ready, dated {len(updates)} existing row(s)"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--db", default=os.getenv("CHECKPOINT_DB") or ".checkpoints.sqlite", help="sqlite file")
    p.add_argument("--postgres", action="store_true", help="migrate $DATABASE_URL instead of sqlite")
    args = p.parse_args()
    if args.postgres:
        url = os.getenv("DATABASE_URL")
        if not url:
            sys.exit("--postgres needs DATABASE_URL")
        print(migrate_postgres(url))
    else:
        print(migrate_sqlite(args.db))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
