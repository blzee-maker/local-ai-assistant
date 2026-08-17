"""Schedule state and a record of everything the daemon did unattended.

Two jobs, both about trust.

**Schedule survives restarts.** Without persistence, every restart looks like a
fresh start and a weekly disk scan would run on every launch — expensive work,
triggered by an unrelated action, on a machine with little headroom.

**Every run is recorded, including failures.** The daemon acts while nobody is
watching, which is exactly when "what did it do?" must have an answer that is not
a guess. The tool ledger covers tool invocations; this covers the jobs that drive
them.
"""
from __future__ import annotations

import json
import sqlite3
import time
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config import settings

JOURNAL_PATH = Path(settings.upload_dir).parent / "daemon.sqlite3"

SCHEMA = """
CREATE TABLE IF NOT EXISTS job_state (
    job          TEXT PRIMARY KEY,
    last_run     REAL,
    last_outcome TEXT,
    last_detail  TEXT,
    failures     INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS job_runs (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    job      TEXT NOT NULL,
    at       REAL NOT NULL,
    duration REAL NOT NULL,
    outcome  TEXT NOT NULL,
    detail   TEXT,
    payload  TEXT
);
CREATE INDEX IF NOT EXISTS idx_job_runs_at ON job_runs(at DESC);
CREATE TABLE IF NOT EXISTS seen_files (
    path      TEXT PRIMARY KEY,
    mtime     REAL NOT NULL,
    indexed_at REAL NOT NULL
);
"""


@dataclass
class JobRun:
    job: str
    at: float
    duration: float
    outcome: str
    detail: str | None = None
    payload: dict[str, Any] | None = None


@dataclass
class JobState:
    job: str
    last_run: float | None
    last_outcome: str | None
    last_detail: str | None
    failures: int


class Journal:
    def __init__(self, path: str | Path | None = None) -> None:
        self._path = Path(path or JOURNAL_PATH)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path))
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    # ── schedule ─────────────────────────────────────────────────
    def state(self, job: str) -> JobState:
        with closing(
            self._conn.execute(
                "SELECT job, last_run, last_outcome, last_detail, failures "
                "FROM job_state WHERE job = ?",
                (job,),
            )
        ) as cur:
            row = cur.fetchone()
        if row is None:
            return JobState(job=job, last_run=None, last_outcome=None,
                            last_detail=None, failures=0)
        return JobState(*row)

    def all_states(self) -> list[JobState]:
        with closing(
            self._conn.execute(
                "SELECT job, last_run, last_outcome, last_detail, failures "
                "FROM job_state ORDER BY job"
            )
        ) as cur:
            return [JobState(*row) for row in cur.fetchall()]

    def record_run(self, run: JobRun) -> None:
        self._conn.execute(
            "INSERT INTO job_runs (job, at, duration, outcome, detail, payload) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                run.job,
                run.at,
                run.duration,
                run.outcome,
                (run.detail or "")[:1000] or None,
                json.dumps(run.payload, default=str)[:4000] if run.payload else None,
            ),
        )
        failed = run.outcome in {"error", "failed"}
        previous = self.state(run.job)
        self._conn.execute(
            "INSERT OR REPLACE INTO job_state "
            "(job, last_run, last_outcome, last_detail, failures) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                run.job,
                run.at,
                run.outcome,
                (run.detail or "")[:500] or None,
                previous.failures + 1 if failed else 0,
            ),
        )
        self._conn.commit()

    def history(self, limit: int = 25, job: str | None = None) -> list[JobRun]:
        query = "SELECT job, at, duration, outcome, detail, payload FROM job_runs"
        params: tuple = ()
        if job:
            query += " WHERE job = ?"
            params = (job,)
        query += " ORDER BY at DESC LIMIT ?"
        params += (limit,)

        with closing(self._conn.execute(query, params)) as cur:
            rows = cur.fetchall()

        runs: list[JobRun] = []
        for name, at, duration, outcome, detail, payload in rows:
            parsed = None
            if payload:
                try:
                    parsed = json.loads(payload)
                except ValueError:
                    parsed = None
            runs.append(JobRun(name, at, duration, outcome, detail, parsed))
        return runs

    def runs_since(self, since: float) -> list[JobRun]:
        with closing(
            self._conn.execute(
                "SELECT job, at, duration, outcome, detail, payload FROM job_runs "
                "WHERE at >= ? ORDER BY at",
                (since,),
            )
        ) as cur:
            rows = cur.fetchall()
        return [JobRun(n, a, d, o, det, None) for n, a, d, o, det, _p in rows]

    # ── seen files ───────────────────────────────────────────────
    #
    # Tracked here rather than inferred from the vector store: a document the
    # user deliberately removed from the index must not be silently re-added on
    # the next sweep. "Have I seen this file?" and "is it indexed?" are
    # different questions, and conflating them fights the user.
    def known_files(self) -> dict[str, float]:
        with closing(self._conn.execute("SELECT path, mtime FROM seen_files")) as cur:
            return dict(cur.fetchall())

    def mark_seen(self, entries: dict[str, float]) -> None:
        if not entries:
            return
        now = time.time()
        self._conn.executemany(
            "INSERT OR REPLACE INTO seen_files (path, mtime, indexed_at) "
            "VALUES (?, ?, ?)",
            [(path, mtime, now) for path, mtime in entries.items()],
        )
        self._conn.commit()

    def forget_seen(self) -> int:
        with closing(self._conn.execute("SELECT COUNT(*) FROM seen_files")) as cur:
            count = int(cur.fetchone()[0])
        self._conn.execute("DELETE FROM seen_files")
        self._conn.commit()
        return count

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "Journal":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
