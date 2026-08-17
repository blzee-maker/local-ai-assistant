"""SQLite persistence for conversations and remembered facts.

Two different things live here, deliberately in one database:

* **Conversations** — the transcript, so closing the terminal stops throwing away
  what you said. Previously the whole session died with the process.
* **Memories** — durable facts the user asked to be kept ("my name is …", "I
  prefer …"), each stored with its embedding so recall is semantic rather than
  keyword-based.

Embeddings are stored as raw float32 blobs beside their text rather than in a
second FAISS index. Memories number in the dozens, not the millions, so a brute
force cosine pass is instant and the whole store stays one deletable file — which
matters more here than query speed, because a user who says "forget everything"
must get exactly that, not an orphaned index left on disk.
"""
from __future__ import annotations

import sqlite3
import time
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from config import settings

MEMORY_PATH = Path(settings.upload_dir).parent / "memory.sqlite3"

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at  REAL NOT NULL,
    last_active REAL NOT NULL,
    title       TEXT
);
CREATE TABLE IF NOT EXISTS messages (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role       TEXT NOT NULL,
    content    TEXT NOT NULL,
    at         REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id);
CREATE TABLE IF NOT EXISTS memories (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    text       TEXT NOT NULL,
    source     TEXT NOT NULL,
    created_at REAL NOT NULL,
    used_count INTEGER NOT NULL DEFAULT 0,
    dim        INTEGER,
    embedding  BLOB
);
"""


@dataclass
class Memory:
    id: int
    text: str
    source: str
    created_at: float
    used_count: int = 0

    @property
    def age_days(self) -> float:
        return (time.time() - self.created_at) / 86_400.0


@dataclass
class StoredMessage:
    role: str
    content: str
    at: float


@dataclass
class SessionInfo:
    id: int
    started_at: float
    last_active: float
    title: str | None
    message_count: int


class MemoryStore:
    def __init__(self, path: str | Path | None = None) -> None:
        self._path = Path(path or MEMORY_PATH)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path))
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(SCHEMA)
        self._conn.commit()
        # Cached vector matrix, invalidated on write. Rebuilding is cheap at this
        # scale and keeps recall from re-reading the table every turn.
        self._vectors: np.ndarray | None = None
        self._vector_ids: list[int] = []

    # ── sessions ─────────────────────────────────────────────────
    def create_session(self, title: str | None = None) -> int:
        now = time.time()
        cur = self._conn.execute(
            "INSERT INTO sessions (started_at, last_active, title) VALUES (?, ?, ?)",
            (now, now, title),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def latest_session(self) -> int | None:
        with closing(
            self._conn.execute("SELECT id FROM sessions ORDER BY last_active DESC LIMIT 1")
        ) as cur:
            row = cur.fetchone()
        return int(row[0]) if row else None

    def sessions(self, limit: int = 20) -> list[SessionInfo]:
        with closing(
            self._conn.execute(
                "SELECT s.id, s.started_at, s.last_active, s.title, "
                "       (SELECT COUNT(*) FROM messages m WHERE m.session_id = s.id) "
                "FROM sessions s ORDER BY s.last_active DESC LIMIT ?",
                (limit,),
            )
        ) as cur:
            return [SessionInfo(*row) for row in cur.fetchall()]

    def set_title(self, session_id: int, title: str) -> None:
        """Set the title only if there isn't one.

        The title is the *first* thing the user said. Guarding in SQL rather
        than at the call site because the caller cannot cheaply know whether an
        earlier message exists — a windowed lookback saw only recent messages
        and happily retitled the session on every turn.
        """
        self._conn.execute(
            "UPDATE sessions SET title = ? WHERE id = ? AND title IS NULL",
            (title[:120], session_id),
        )
        self._conn.commit()

    def delete_session(self, session_id: int) -> None:
        self._conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        self._conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        self._conn.commit()

    # ── messages ─────────────────────────────────────────────────
    def add_message(self, session_id: int, role: str, content: str) -> None:
        now = time.time()
        self._conn.execute(
            "INSERT INTO messages (session_id, role, content, at) VALUES (?, ?, ?, ?)",
            (session_id, role, content, now),
        )
        self._conn.execute(
            "UPDATE sessions SET last_active = ? WHERE id = ?", (now, session_id)
        )
        self._conn.commit()

    def messages(self, session_id: int, limit: int | None = None) -> list[StoredMessage]:
        query = "SELECT role, content, at FROM messages WHERE session_id = ? ORDER BY id"
        params: tuple[Any, ...] = (session_id,)
        if limit is not None:
            # Take the newest `limit`, then restore chronological order.
            query = (
                "SELECT role, content, at FROM (" + query + " DESC LIMIT ?) "
                "ORDER BY at"
            )
            params = (session_id, limit)
        with closing(self._conn.execute(query, params)) as cur:
            return [StoredMessage(*row) for row in cur.fetchall()]

    # ── memories ─────────────────────────────────────────────────
    def add_memory(
        self, text: str, source: str, embedding: np.ndarray | None = None
    ) -> int:
        blob = None
        dim = None
        if embedding is not None:
            vec = np.asarray(embedding, dtype=np.float32).reshape(-1)
            blob = vec.tobytes()
            dim = int(vec.shape[0])
        cur = self._conn.execute(
            "INSERT INTO memories (text, source, created_at, dim, embedding) "
            "VALUES (?, ?, ?, ?, ?)",
            (text, source, time.time(), dim, blob),
        )
        self._conn.commit()
        self._vectors = None
        return int(cur.lastrowid)

    def memories(self) -> list[Memory]:
        with closing(
            self._conn.execute(
                "SELECT id, text, source, created_at, used_count FROM memories "
                "ORDER BY created_at DESC"
            )
        ) as cur:
            return [Memory(*row) for row in cur.fetchall()]

    def count(self) -> int:
        with closing(self._conn.execute("SELECT COUNT(*) FROM memories")) as cur:
            return int(cur.fetchone()[0])

    def find_similar_text(self, text: str) -> Memory | None:
        """Exact-text match, used to avoid storing the same fact twice."""
        with closing(
            self._conn.execute(
                "SELECT id, text, source, created_at, used_count FROM memories "
                "WHERE lower(trim(text)) = lower(trim(?)) LIMIT 1",
                (text,),
            )
        ) as cur:
            row = cur.fetchone()
        return Memory(*row) if row else None

    def forget(self, memory_id: int) -> bool:
        changed = self._conn.execute(
            "DELETE FROM memories WHERE id = ?", (memory_id,)
        ).rowcount
        self._conn.commit()
        self._vectors = None
        return bool(changed)

    def forget_all(self) -> int:
        with closing(self._conn.execute("SELECT COUNT(*) FROM memories")) as cur:
            count = int(cur.fetchone()[0])
        self._conn.execute("DELETE FROM memories")
        self._conn.commit()
        self._vectors = None
        return count

    def mark_used(self, memory_ids: list[int]) -> None:
        if not memory_ids:
            return
        self._conn.executemany(
            "UPDATE memories SET used_count = used_count + 1 WHERE id = ?",
            [(i,) for i in memory_ids],
        )
        self._conn.commit()

    # ── vector search ────────────────────────────────────────────
    def _load_vectors(self) -> tuple[np.ndarray | None, list[int]]:
        if self._vectors is not None:
            return self._vectors, self._vector_ids

        with closing(
            self._conn.execute(
                "SELECT id, dim, embedding FROM memories WHERE embedding IS NOT NULL"
            )
        ) as cur:
            rows = cur.fetchall()

        if not rows:
            self._vectors, self._vector_ids = None, []
            return None, []

        ids: list[int] = []
        vectors: list[np.ndarray] = []
        expected = rows[0][1]
        for mem_id, dim, blob in rows:
            # A changed embedding model leaves old vectors incomparable. Skip
            # them rather than crashing on a shape mismatch.
            if dim != expected:
                continue
            ids.append(int(mem_id))
            vectors.append(np.frombuffer(blob, dtype=np.float32))

        matrix = np.vstack(vectors) if vectors else None
        self._vectors, self._vector_ids = matrix, ids
        return matrix, ids

    def search(self, query_vec: np.ndarray, k: int, min_score: float) -> list[tuple[Memory, float]]:
        """Cosine similarity over stored memories, above `min_score` only."""
        matrix, ids = self._load_vectors()
        if matrix is None or not ids:
            return []

        query = np.asarray(query_vec, dtype=np.float32).reshape(-1)
        if query.shape[0] != matrix.shape[1]:
            return []  # embedding model changed; nothing comparable

        # Vectors from fastembed are already L2-normalized, but normalize again
        # so a future embedder swap cannot silently produce garbage scores.
        matrix_norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix_norms[matrix_norms == 0] = 1.0
        query_norm = np.linalg.norm(query) or 1.0
        scores = (matrix / matrix_norms) @ (query / query_norm)

        order = np.argsort(-scores)[:k]
        by_id = {m.id: m for m in self.memories()}
        results: list[tuple[Memory, float]] = []
        for idx in order:
            score = float(scores[idx])
            if score < min_score:
                continue
            memory = by_id.get(ids[int(idx)])
            if memory is not None:
                results.append((memory, score))
        return results

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "MemoryStore":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
