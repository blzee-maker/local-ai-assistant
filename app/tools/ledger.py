"""Per-tool consent and an audit trail of everything the assistant did.

`app/consent.py` governs one question: may we read the user's folders? That was
enough when reading was all the assistant could do. Once tools can change state,
a single yes/no is too blunt — approving disk analysis must not also approve
killing processes.

So permission is per tool and recorded, following the same shape as the folder
consent (rule 3): an explicit decision, stored, that does not silently widen.
READ tools inherit the folder grant rather than nagging separately; anything that
writes or destroys asks on first use.

The audit log exists for rule 4. An assistant that acts on your machine and keeps
no record of it is asking for trust it has not earned — and when something is
wrong, "what did it actually do?" needs an answer that is not a guess.
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

LEDGER_PATH = Path(settings.upload_dir).parent / "tool_ledger.sqlite3"

SCHEMA = """
CREATE TABLE IF NOT EXISTS tool_grants (
    tool       TEXT PRIMARY KEY,
    granted    INTEGER NOT NULL,
    risk       TEXT NOT NULL,
    decided_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS tool_audit (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    at         REAL NOT NULL,
    tool       TEXT NOT NULL,
    risk       TEXT NOT NULL,
    source     TEXT NOT NULL,
    arguments  TEXT NOT NULL,
    outcome    TEXT NOT NULL,
    detail     TEXT
);
"""


@dataclass
class AuditEntry:
    at: float
    tool: str
    risk: str
    source: str
    arguments: dict[str, Any]
    outcome: str
    detail: str | None = None


class ToolLedger:
    def __init__(self, path: str | Path | None = None) -> None:
        self._path = Path(path or LEDGER_PATH)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path))
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    # ── consent ──────────────────────────────────────────────────
    def decision(self, tool: str) -> bool | None:
        """True/False if decided, None if never asked."""
        with closing(
            self._conn.execute("SELECT granted FROM tool_grants WHERE tool = ?", (tool,))
        ) as cur:
            row = cur.fetchone()
        return None if row is None else bool(row[0])

    def record_decision(self, tool: str, risk: str, granted: bool) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO tool_grants (tool, granted, risk, decided_at) "
            "VALUES (?, ?, ?, ?)",
            (tool, int(granted), risk, time.time()),
        )
        self._conn.commit()

    def revoke(self, tool: str | None = None) -> int:
        """Withdraw one tool's permission, or all of them."""
        if tool is None:
            with closing(self._conn.execute("SELECT COUNT(*) FROM tool_grants")) as cur:
                count = cur.fetchone()[0]
            self._conn.execute("DELETE FROM tool_grants")
        else:
            count = self._conn.execute(
                "DELETE FROM tool_grants WHERE tool = ?", (tool,)
            ).rowcount
        self._conn.commit()
        return count

    def grants(self) -> list[tuple[str, bool, str, float]]:
        with closing(
            self._conn.execute(
                "SELECT tool, granted, risk, decided_at FROM tool_grants ORDER BY tool"
            )
        ) as cur:
            return [(t, bool(g), r, d) for t, g, r, d in cur.fetchall()]

    # ── audit ────────────────────────────────────────────────────
    def record(
        self,
        tool: str,
        risk: str,
        source: str,
        arguments: dict[str, Any],
        outcome: str,
        detail: str | None = None,
    ) -> None:
        """Log an invocation. Called for refusals too — a denied destructive
        call is exactly the event someone will later want to see."""
        self._conn.execute(
            "INSERT INTO tool_audit (at, tool, risk, source, arguments, outcome, detail) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                time.time(),
                tool,
                risk,
                source,
                json.dumps(arguments, default=str)[:2000],
                outcome,
                detail[:500] if detail else None,
            ),
        )
        self._conn.commit()

    def history(self, limit: int = 50, tool: str | None = None) -> list[AuditEntry]:
        query = (
            "SELECT at, tool, risk, source, arguments, outcome, detail FROM tool_audit"
        )
        params: tuple = ()
        if tool:
            query += " WHERE tool = ?"
            params = (tool,)
        query += " ORDER BY at DESC LIMIT ?"
        params += (limit,)

        with closing(self._conn.execute(query, params)) as cur:
            rows = cur.fetchall()

        entries: list[AuditEntry] = []
        for at, name, risk, source, args, outcome, detail in rows:
            try:
                parsed = json.loads(args)
            except ValueError:
                parsed = {"raw": args}
            entries.append(
                AuditEntry(at, name, risk, source, parsed, outcome, detail)
            )
        return entries

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "ToolLedger":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
