"""The append-only event log.

Two properties matter more than the rest and are tested hardest: it never takes
the caller down with it (rule 10), and it does not write down what the user said
unless that was switched on deliberately.
"""
from __future__ import annotations

import json
import threading

import pytest

from app import eventlog
from config import settings


@pytest.fixture
def log_path(tmp_path, monkeypatch):
    target = tmp_path / "events.jsonl"
    monkeypatch.setattr(settings, "event_log_path", str(target), raising=False)
    monkeypatch.setattr(eventlog, "_warned", False, raising=False)
    return target


def _entries(target):
    return [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines() if line]


def test_an_event_is_one_json_line(log_path):
    eventlog.log("session.start", live=True)
    entries = _entries(log_path)
    assert len(entries) == 1
    assert entries[0]["event"] == "session.start"
    assert entries[0]["live"] is True
    assert "at" in entries[0] and "session" in entries[0]


def test_every_entry_carries_the_run_it_came_from(log_path):
    """So one bad session can be read on its own instead of interleaved with
    every run before it."""
    eventlog.log("a")
    eventlog.log("b")
    assert {e["session"] for e in _entries(log_path)} == {eventlog.SESSION_ID}


# ── what the user said ───────────────────────────────────────────
def test_speech_is_not_written_down_by_default(log_path, monkeypatch):
    """A voice log with transcripts on is a persistent record of the room that
    nobody asked for."""
    monkeypatch.setattr(settings, "log_transcripts", False, raising=False)
    eventlog.log("heard", speech="my bank password is hunter2")
    entry = _entries(log_path)[0]
    assert "hunter2" not in json.dumps(entry)
    assert entry["speech"]["redacted"] is True
    assert entry["speech"]["words"] == 5


def test_the_shape_survives_redaction(log_path, monkeypatch):
    """Because the shape is what most voice bugs are about — an empty
    transcript, or a fragment where a sentence was expected."""
    monkeypatch.setattr(settings, "log_transcripts", False, raising=False)
    eventlog.log("heard", speech="")
    entry = _entries(log_path)[0]
    assert entry["speech"]["words"] == 0
    assert entry["speech"]["chars"] == 0


def test_transcripts_are_recorded_when_that_is_switched_on(log_path, monkeypatch):
    monkeypatch.setattr(settings, "log_transcripts", True, raising=False)
    eventlog.log("heard", speech="open spotify")
    assert _entries(log_path)[0]["speech"] == "open spotify"


def test_every_declared_field_is_redacted_not_just_the_first(log_path, monkeypatch):
    """A guarantee that depends on every call site remembering it is not a
    guarantee, so the logger enforces it. Driven off the tuple rather than a
    fixed list, so a field added there is covered without anyone remembering
    to extend this test — and one added *elsewhere* fails it."""
    monkeypatch.setattr(settings, "log_transcripts", False, raising=False)
    secret = "sortcode 60 16 13 hunter2"
    eventlog.log("probe", **{field: secret for field in eventlog.REDACTED_FIELDS})
    entry = _entries(log_path)[0]
    assert "hunter2" not in json.dumps(entry)
    for field in eventlog.REDACTED_FIELDS:
        assert entry[field]["redacted"] is True
        assert entry[field]["words"] == 5


def test_typed_prompts_are_redacted_the_same_way_as_speech(log_path, monkeypatch):
    """Spoken and typed input are the same question — "is there a record of
    what I said?" — so they get the same answer."""
    monkeypatch.setattr(settings, "log_transcripts", False, raising=False)
    eventlog.log("turn.start", prompt="summarise my medical results")
    entry = _entries(log_path)[0]
    assert "medical" not in json.dumps(entry)
    assert entry["prompt"]["words"] == 4


def test_fields_outside_the_tuple_are_kept_verbatim(log_path):
    """Redaction is targeted, not blanket — the timings and model names this
    log exists for must survive it."""
    eventlog.log("turn.end", model="llama3.2:3b", took_s=1.4)
    entry = _entries(log_path)[0]
    assert entry["model"] == "llama3.2:3b"
    assert entry["took_s"] == 1.4


# ── it must never take the caller down ───────────────────────────
def test_a_broken_destination_does_not_raise(tmp_path, monkeypatch, capsys):
    """A logger that can end a conversation is worse than no logger."""
    monkeypatch.setattr(eventlog, "_warned", False, raising=False)
    # A path whose parent is a *file* — mkdir and open both fail.
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("x", encoding="utf-8")
    monkeypatch.setattr(settings, "event_log_path", str(blocker / "events.jsonl"),
                        raising=False)
    eventlog.log("session.start")          # must not raise
    assert "event log unavailable" in capsys.readouterr().err


def test_the_failure_is_reported_once_and_then_stays_quiet(tmp_path, monkeypatch, capsys):
    """Silent forever is how you debug against a file that was never written;
    once per line is how a log destroys the output it was meant to explain."""
    monkeypatch.setattr(eventlog, "_warned", False, raising=False)
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("x", encoding="utf-8")
    monkeypatch.setattr(settings, "event_log_path", str(blocker / "events.jsonl"),
                        raising=False)
    for _ in range(5):
        eventlog.log("tick")
    assert capsys.readouterr().err.count("event log unavailable") == 1


def test_reading_a_log_that_does_not_exist_returns_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "event_log_path", str(tmp_path / "absent.jsonl"),
                        raising=False)
    assert eventlog.read() == []


def test_a_half_written_line_does_not_poison_the_read(log_path):
    """A hard kill mid-write leaves a partial final line. Skipping it is right;
    refusing to read the log because of it is not."""
    eventlog.log("good")
    with open(log_path, "a", encoding="utf-8") as handle:
        handle.write('{"event": "truncated"')      # no newline, no closing brace
    entries = eventlog.read()
    assert [e["event"] for e in entries] == ["good"]


# ── written from more than one thread ────────────────────────────
def test_concurrent_writers_do_not_interleave(log_path):
    """The awake window is watched by a threading.Timer, so state changes
    arrive from at least two threads. This is why the log is a locked file
    append and not sqlite, whose connections belong to one thread."""
    def writer(n: int) -> None:
        for i in range(25):
            eventlog.log("tick", worker=n, i=i)

    threads = [threading.Thread(target=writer, args=(n,)) for n in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    entries = _entries(log_path)          # every line must still parse
    assert len(entries) == 100


# ── filtering and rotation ───────────────────────────────────────
def test_reading_can_filter_by_event(log_path):
    eventlog.log("heard")
    eventlog.log("decision", action="run")
    eventlog.log("heard")
    assert len(eventlog.read(event="heard")) == 2


def test_the_limit_returns_the_most_recent(log_path):
    for i in range(10):
        eventlog.log("tick", i=i)
    got = eventlog.read(limit=3)
    assert [e["i"] for e in got] == [7, 8, 9]


def test_the_log_rotates_instead_of_growing_without_bound(log_path, monkeypatch):
    monkeypatch.setattr(settings, "event_log_max_bytes", 200, raising=False)
    for i in range(50):
        eventlog.log("tick", i=i, padding="x" * 40)
    assert log_path.with_suffix(".jsonl.1").exists()
    assert log_path.stat().st_size < 2000


def test_runs_are_listed_with_their_size(log_path):
    eventlog.log("session.start")
    eventlog.log("heard")
    found = eventlog.sessions()
    assert len(found) == 1
    assert found[0]["session"] == eventlog.SESSION_ID
    assert found[0]["events"] == 2
