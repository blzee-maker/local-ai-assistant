"""SQLite-backed hash cache so a rescan doesn't re-read the whole disk.

Content hashing is the expensive part of a scan. A file whose size and
modification time are both unchanged since last time cannot have different
content in any way that matters here, so its hash is reused.

The cache key deliberately includes size *and* mtime: keying on path alone would
serve a stale hash after an edit, silently reporting two files as duplicates
when they no longer are.
"""
from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

from app.analyzer.walker import FileEntry

SCHEMA = """
CREATE TABLE IF NOT EXISTS file_hashes (
    key    TEXT PRIMARY KEY,
    digest TEXT NOT NULL,
    seen   REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS scans (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    started   REAL NOT NULL,
    finished  REAL,
    files     INTEGER,
    bytes     INTEGER,
    summary   TEXT
);
CREATE TABLE IF NOT EXISTS observed_atimes (
    path            TEXT PRIMARY KEY,
    earliest_atime  REAL NOT NULL
);
"""


def entry_key(entry: FileEntry) -> str:
    """Identity of a file's *content* for caching purposes."""
    return f"{entry.path}|{entry.size}|{int(entry.mtime)}"


class HashCache:
    """A dict-like persistent map of content-key → digest."""

    def __init__(self, db_path: str | Path) -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path))
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def load(self) -> dict[str, str]:
        with closing(self._conn.execute("SELECT key, digest FROM file_hashes")) as cur:
            return dict(cur.fetchall())

    def save(self, hashes: dict[str, str]) -> None:
        import time

        now = time.time()
        self._conn.executemany(
            "INSERT OR REPLACE INTO file_hashes (key, digest, seen) VALUES (?, ?, ?)",
            [(k, v, now) for k, v in hashes.items()],
        )
        self._conn.commit()

    def prune(self, keep_keys: set[str]) -> int:
        """Drop entries for files that no longer exist, so the DB can't grow
        without bound across months of scans."""
        with closing(self._conn.execute("SELECT key FROM file_hashes")) as cur:
            existing = {row[0] for row in cur.fetchall()}
        dead = existing - keep_keys
        if dead:
            self._conn.executemany(
                "DELETE FROM file_hashes WHERE key = ?", [(k,) for k in dead]
            )
            self._conn.commit()
        return len(dead)

    # ── access-time preservation ─────────────────────────────────
    #
    # Hashing a file to compare it against another *reads* it, and reading
    # updates its last-access time. So the duplicate detector destroys the very
    # signal the staleness report depends on: after one scan, files that had sat
    # untouched for a year looked freshly used and vanished from the report.
    # (Observed directly — the "large and unused" list shrank from 5 files to 3
    # between two consecutive runs, purely because of our own reads.)
    #
    # The fix is to remember the *earliest* access time ever observed for each
    # path and prefer it. Restoring atime with os.utime() would be the
    # alternative, but that means writing to the user's files, which this
    # package promises never to do.
    def merge_atimes(self, observed: dict[str, float]) -> dict[str, float]:
        """Record the atimes we saw and return the earliest known per path."""
        with closing(
            self._conn.execute("SELECT path, earliest_atime FROM observed_atimes")
        ) as cur:
            stored = dict(cur.fetchall())

        merged: dict[str, float] = {}
        writes: list[tuple[str, float]] = []
        for path, atime in observed.items():
            previous = stored.get(path)
            if previous is None or atime < previous:
                merged[path] = atime
                writes.append((path, atime))
            else:
                merged[path] = previous

        if writes:
            self._conn.executemany(
                "INSERT OR REPLACE INTO observed_atimes (path, earliest_atime) "
                "VALUES (?, ?)",
                writes,
            )
            self._conn.commit()
        return merged

    def forget_atimes(self, keep_paths: set[str]) -> None:
        with closing(self._conn.execute("SELECT path FROM observed_atimes")) as cur:
            existing = {row[0] for row in cur.fetchall()}
        dead = existing - keep_paths
        if dead:
            self._conn.executemany(
                "DELETE FROM observed_atimes WHERE path = ?", [(p,) for p in dead]
            )
            self._conn.commit()

    def record_scan(
        self, started: float, finished: float, files: int, total_bytes: int, summary: str
    ) -> None:
        self._conn.execute(
            "INSERT INTO scans (started, finished, files, bytes, summary) "
            "VALUES (?, ?, ?, ?, ?)",
            (started, finished, files, total_bytes, summary),
        )
        self._conn.commit()

    def last_scan(self) -> tuple[float, str] | None:
        with closing(
            self._conn.execute(
                "SELECT finished, summary FROM scans WHERE finished IS NOT NULL "
                "ORDER BY finished DESC LIMIT 1"
            )
        ) as cur:
            row = cur.fetchone()
        return (row[0], row[1]) if row else None

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "HashCache":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
