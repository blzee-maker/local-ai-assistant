"""An append-only record of what the assistant did, so a session can be read back.

Written because a voice session is the one part of this program the user cannot
inspect while it runs. Chat scrolls; the loop between turns does not, and when
it sat silent there was nothing afterwards to say whether it had been waiting on
the microphone, had transcribed something and discarded it, or had gone deaf
when the window lapsed.

**JSONL, not sqlite.** The awake window is watched by a `threading.Timer`, so
state changes are written from at least two threads, and sqlite3 connections
belong to the thread that opened them — the mistake already made once in
`app/daemon/watcher.py`. An append to a file under a lock has no such rule, and
`O_APPEND` writes of a single short line do not interleave. It is also greppable
with no tooling, which matters for a log whose entire purpose is being read at
2am when something is broken.

**What the user said is not recorded by default.** These events carry the user's
own words — spoken into the room, or typed at the prompt — and a log of those is
a persistent record of the user that nobody asked for (the same objection as
inferring facts in the background). So the fields in `REDACTED_FIELDS` are
reduced to a shape, never their words, unless `log_transcripts` is switched on
deliberately. The redaction happens *here* rather than at each call site,
because a guarantee that depends on every caller remembering it is not a
guarantee (rule 1) — and the way to add a new one is to name it in that tuple,
so the safe thing is also the easy thing.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Iterator

from config import settings

# One id per process, so `logs` can separate this run from the last one. Short
# because it is read by eye, not joined on.
SESSION_ID = uuid.uuid4().hex[:8]

# Anything the user themselves produced. Spoken and typed input are the same
# question — "is there a record of what I said?" — so they get the same answer.
REDACTED_FIELDS = ("speech", "prompt")

_lock = threading.Lock()
_warned = False


def path() -> Path:
    return Path(settings.event_log_path)


def _redacted(text: str) -> Any:
    """What is safe to keep about something the user said or typed.

    The shape survives redaction because the shape is what most of these bugs
    are about: an empty transcript, a two-word fragment where a sentence was
    expected, a wake phrase that arrived as four words instead of three, a
    prompt far longer than the context window. None of that needs the words.
    """
    if settings.log_transcripts:
        return text
    words = len(text.split())
    return {"redacted": True, "words": words, "chars": len(text)}


def _rotate_if_large(target: Path) -> None:
    """Keep one previous file and no more.

    Size-based rather than dated: this log is read to answer "what just
    happened", so an unbounded history buys nothing on a machine that is short
    of disk as well as memory.
    """
    try:
        if target.stat().st_size < settings.event_log_max_bytes:
            return
    except OSError:
        return
    previous = target.with_suffix(target.suffix + ".1")
    try:
        os.replace(target, previous)
    except OSError:
        pass


def log(event: str, **fields: Any) -> None:
    """Record one event. Never raises.

    A logger that can end a conversation is worse than no logger (rule 10) — but
    a logger that fails silently forever is how you end up debugging against a
    file that was never being written. So the first failure is reported once, on
    stderr, and after that it stays quiet.
    """
    global _warned
    record: dict[str, Any] = {
        "at": round(time.time(), 3),
        "session": SESSION_ID,
        "event": event,
    }
    for key, value in fields.items():
        record[key] = _redacted(value) if key in REDACTED_FIELDS and value is not None else value

    try:
        target = path()
        with _lock:
            target.parent.mkdir(parents=True, exist_ok=True)
            _rotate_if_large(target)
            line = json.dumps(record, ensure_ascii=False, default=str)
            with open(target, "a", encoding="utf-8") as handle:
                handle.write(line + "\n")
    except Exception as exc:  # noqa: BLE001 - see docstring
        if not _warned:
            _warned = True
            print(f"event log unavailable ({exc})", file=sys.stderr)


def read(limit: int = 50, *, event: str | None = None,
         session: str | None = None) -> list[dict[str, Any]]:
    """The most recent entries, oldest first.

    Reads the whole file rather than seeking backwards: it is capped at
    `event_log_max_bytes` and the alternative is fiddly enough to get wrong.
    """
    entries = [
        entry
        for entry in _iter_entries(path())
        if (event is None or entry.get("event") == event)
        and (session is None or entry.get("session") == session)
    ]
    return entries[-limit:] if limit else entries


def _iter_entries(target: Path) -> Iterator[dict[str, Any]]:
    try:
        with open(target, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except ValueError:
                    # A half-written final line after a hard kill. Skipping it
                    # is right; refusing to read the log because of it is not.
                    continue
                if isinstance(entry, dict):
                    yield entry
    except OSError:
        return


def sessions(limit: int = 10) -> list[dict[str, Any]]:
    """Recent runs, newest first, with when each started and how big it was."""
    seen: dict[str, dict[str, Any]] = {}
    for entry in _iter_entries(path()):
        key = str(entry.get("session", "?"))
        info = seen.setdefault(key, {"session": key, "started": entry.get("at"), "events": 0})
        info["events"] += 1
        info["last"] = entry.get("at")
    return list(reversed(list(seen.values())))[:limit]
