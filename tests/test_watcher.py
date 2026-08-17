"""Tests for the Downloads watcher.

The events themselves are the easy part. The risk is indexing a file that is
still being written — a download exists as a zero-byte file long before it is a
document — so most of this covers "is it finished?" rather than "did we notice?".
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from app.daemon.watcher import (
    DownloadsWatcher,
    is_indexable,
    wait_until_settled,
)


# ── what is worth indexing ───────────────────────────────────────
@pytest.mark.parametrize(
    "name", ["report.pdf", "notes.md", "readme.txt", "contract.docx", "guide.markdown"]
)
def test_documents_are_indexable(name):
    assert is_indexable(Path(name))


@pytest.mark.parametrize(
    "name",
    [
        "movie.mp4", "photo.jpg", "archive.zip", "installer.exe", "data.csv",
    ],
)
def test_unsupported_formats_are_ignored(name):
    assert not is_indexable(Path(name))


@pytest.mark.parametrize(
    "name",
    [
        "report.pdf.crdownload",   # Chrome, mid-download
        "report.pdf.part",         # Firefox, mid-download
        "notes.md.tmp",
        "~$contract.docx",         # Word owner-lock stub
        "~budget.md",
    ],
)
def test_in_progress_and_scratch_files_are_ignored(name):
    """A partial download is not a document, and indexing one records a
    fragment as if it were the finished file."""
    assert not is_indexable(Path(name))


# ── knowing when a file is finished ──────────────────────────────
def test_a_stable_file_settles(tmp_path):
    path = tmp_path / "done.pdf"
    path.write_bytes(b"%PDF-1.4 complete enough")
    assert wait_until_settled(path, settle_seconds=0.2, poll=0.05) is True


def test_an_empty_file_never_settles(tmp_path):
    """A creation event fires when the file exists, not when it has content."""
    path = tmp_path / "starting.pdf"
    path.write_bytes(b"")
    assert wait_until_settled(path, settle_seconds=0.2, poll=0.05, timeout=1.0) is False


def test_a_growing_file_is_not_settled_until_it_stops(tmp_path):
    """The actual failure this prevents: indexing a half-written download."""
    path = tmp_path / "growing.pdf"
    path.write_bytes(b"start")
    done = threading.Event()

    def grow():
        for _ in range(6):
            if done.is_set():
                return
            with path.open("ab") as handle:
                handle.write(b"x" * 1000)
            time.sleep(0.1)

    writer = threading.Thread(target=grow)
    writer.start()
    try:
        started = time.time()
        settled = wait_until_settled(path, settle_seconds=0.25, poll=0.05, timeout=5.0)
        elapsed = time.time() - started
        assert settled is True
        # It must have waited for the writer rather than returning immediately.
        assert elapsed >= 0.5
    finally:
        done.set()
        writer.join(timeout=5)


def test_a_vanished_file_reports_unsettled(tmp_path):
    path = tmp_path / "gone.pdf"
    path.write_bytes(b"temporary")
    path.unlink()
    assert wait_until_settled(path, settle_seconds=0.1, poll=0.05, timeout=0.5) is False


# ── queueing ─────────────────────────────────────────────────────
def test_repeated_events_for_one_file_collapse(tmp_path):
    """A single download emits many modify events; it must be handled once."""
    watcher = DownloadsWatcher(assistant=None, root=tmp_path)
    path = tmp_path / "report.pdf"

    for _ in range(5):
        watcher._enqueue(path)

    assert watcher._queue.qsize() == 1


def test_unindexable_paths_never_reach_the_queue(tmp_path):
    watcher = DownloadsWatcher(assistant=None, root=tmp_path)
    watcher._enqueue(tmp_path / "movie.mp4")
    watcher._enqueue(tmp_path / "report.pdf.crdownload")
    assert watcher._queue.qsize() == 0


# ── permission ───────────────────────────────────────────────────
def test_watcher_indexes_nothing_without_consent(tmp_path, monkeypatch):
    """Consent can be withdrawn while the daemon runs; that must take effect
    immediately, not at the next restart."""
    from app import consent

    monkeypatch.setattr(consent, "CONSENT_PATH", tmp_path / "consent.json")
    monkeypatch.setattr(consent, "current_roots", lambda: [str(tmp_path)])
    consent.decline()

    class RecordingAssistant:
        def __init__(self):
            self.ingested = []

        def ingest_path(self, path):
            self.ingested.append(path)
            return {"ok": True, "chunks": 1}

    assistant = RecordingAssistant()
    watcher = DownloadsWatcher(assistant=assistant, root=tmp_path)

    document = tmp_path / "secret.md"
    document.write_text("sensitive", encoding="utf-8")
    watcher._handle(document)

    assert assistant.ingested == []


def test_watcher_indexes_from_its_worker_thread(tmp_path, monkeypatch):
    """Regression: the journal was opened on the main thread and used on the
    worker thread, so the first real file raised "SQLite objects created in a
    thread can only be used in that same thread" and was never indexed. This
    exercises the full started-observer path, not just _handle().
    """
    from app import consent
    from app.daemon import journal as journal_module

    monkeypatch.setattr(consent, "CONSENT_PATH", tmp_path / "consent.json")
    monkeypatch.setattr(consent, "current_roots", lambda: [str(tmp_path)])
    consent.grant([str(tmp_path)])
    monkeypatch.setattr(journal_module, "JOURNAL_PATH", tmp_path / "daemon.sqlite3")

    indexed: list[str] = []
    errors: list[str] = []

    class RecordingAssistant:
        def ingest_path(self, path):
            indexed.append(Path(path).name)
            return {"ok": True, "chunks": 2}

    watched = tmp_path / "Downloads"
    watched.mkdir()

    watcher = DownloadsWatcher(
        assistant=RecordingAssistant(),
        root=watched,
        on_event=lambda level, msg: errors.append(msg) if level == "error" else None,
        settle_seconds=0.2,
    )
    assert watcher.start() is True
    try:
        document = watched / "arrived.md"
        document.write_text("# Arrived\n\nSome content worth indexing.", encoding="utf-8")

        deadline = time.time() + 20
        while time.time() < deadline and not indexed and not errors:
            time.sleep(0.2)
    finally:
        watcher.stop()

    assert not errors, f"watcher errored: {errors}"
    assert indexed == ["arrived.md"]


# ── graceful absence ─────────────────────────────────────────────
def test_missing_downloads_folder_degrades_quietly(tmp_path):
    """No doorbell is not an error — the sweep still covers everything."""
    events: list[tuple[str, str]] = []
    watcher = DownloadsWatcher(
        assistant=None,
        root=tmp_path / "does-not-exist",
        on_event=lambda level, msg: events.append((level, msg)),
    )

    assert watcher.start() is False
    assert any("sweeps only" in msg for _level, msg in events)
    watcher.stop()  # must be safe even though nothing started


def test_stop_is_safe_when_never_started(tmp_path):
    DownloadsWatcher(assistant=None, root=tmp_path).stop()
