"""The doorbell: instant indexing for Downloads while the daemon is awake.

Filesystem events are used for **Downloads only**, deliberately. It is not
cloud-synced, so there is no sync churn inventing events for files nobody
touched, and it is where a person actually saves the thing they are about to ask
about. Everywhere else stays on the periodic sweep.

The two mechanisms are not redundant, they cover different failures:

* **Events only work while the daemon is running.** Anything that arrives while
  it is stopped is never announced — those notifications are gone. The sweep
  that runs at startup is what catches up on them.
* **Events can be dropped** under load, or missed if a directory is replaced
  wholesale. The periodic sweep is the backstop that eventually sees everything.

So the doorbell is an optimisation on top of the mailbox, never a replacement.
If watchdog is missing or the observer fails to start, indexing keeps working at
sweep pace and says so (rule 10).

The hard part is not the events, it is knowing when a file is *finished*. A
browser writes `report.pdf.crdownload`, grows it for a minute, then renames it;
a copy fires a creation event when the file is zero bytes. Ingesting at first
sight would index a truncated document. So a path is only read once its size has
stopped changing.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Callable

from app.daemon.journal import Journal

# Formats the RAG pipeline can read. Kept in step with IndexNewFilesJob.
WATCHED_EXTENSIONS = {".pdf", ".txt", ".md", ".markdown", ".docx"}

# In-progress downloads and editor scratch files. Indexing these is either
# impossible or pointless, and they churn constantly.
IGNORED_SUFFIXES = {
    ".crdownload", ".part", ".partial", ".tmp", ".temp", ".download", ".opdownload",
}
IGNORED_PREFIXES = ("~$", ".~", "~")

# How long a file's size must hold steady before it counts as finished.
SETTLE_SECONDS = 2.0
SETTLE_POLL = 0.5
SETTLE_TIMEOUT = 120.0


def is_indexable(path: Path) -> bool:
    """Should this path ever be considered for indexing?"""
    name = path.name
    if any(name.startswith(prefix) for prefix in IGNORED_PREFIXES):
        return False
    if path.suffix.lower() in IGNORED_SUFFIXES:
        return False
    return path.suffix.lower() in WATCHED_EXTENSIONS


def wait_until_settled(
    path: Path,
    settle_seconds: float = SETTLE_SECONDS,
    poll: float = SETTLE_POLL,
    timeout: float = SETTLE_TIMEOUT,
) -> bool:
    """Block until `path` stops growing. False if it vanished or never settled.

    A creation event fires when the file exists, not when it is complete — a
    download can be zero bytes at that moment. Reading it then would index a
    fragment and record it as done.
    """
    deadline = time.time() + timeout
    last_size = -1
    stable_since: float | None = None

    while time.time() < deadline:
        try:
            size = path.stat().st_size
        except OSError:
            return False  # deleted, or renamed out from under us

        if size == last_size and size > 0:
            if stable_since is None:
                stable_since = time.time()
            elif time.time() - stable_since >= settle_seconds:
                return True
        else:
            stable_since = None
            last_size = size

        time.sleep(poll)
    return False


class DownloadsWatcher:
    """Watches one folder and indexes finished documents as they land."""

    def __init__(
        self,
        assistant: Any,
        root: Path | None = None,
        on_event: Callable[[str, str], None] | None = None,
        settle_seconds: float = SETTLE_SECONDS,
    ) -> None:
        self._assistant = assistant
        self._root = root
        self._on_event = on_event
        self._settle_seconds = settle_seconds
        self._queue: "Queue[Path]" = Queue()
        self._observer: Any = None
        self._worker: threading.Thread | None = None
        self._stopping = threading.Event()
        self._pending: set[str] = set()
        self._lock = threading.Lock()
        # Opened *by the worker thread*, not here. sqlite3 connections may only
        # be used on the thread that created them, and creating it in start()
        # (main thread) while using it in _handle() (worker thread) raised
        # "SQLite objects created in a thread can only be used in that same
        # thread" the first time a real file arrived.
        self._journal: Journal | None = None

    def _emit(self, level: str, message: str) -> None:
        if self._on_event is not None:
            self._on_event(level, message)

    # ── lifecycle ────────────────────────────────────────────────
    def resolve_root(self) -> Path | None:
        """Downloads, but only if the user actually allowed it."""
        if self._root is not None:
            return self._root if self._root.is_dir() else None

        from app import files as filesvc

        for label, path in filesvc.allowed_roots():
            if label.lower() == "downloads":
                return path
        return None

    def start(self) -> bool:
        """Begin watching. False (with a reason emitted) if unavailable."""
        root = self.resolve_root()
        if root is None:
            self._emit("info", "no Downloads folder to watch; sweeps only")
            return False

        try:
            from watchdog.events import FileSystemEventHandler
            from watchdog.observers import Observer
        except ImportError:
            self._emit("info", "watchdog not installed; sweeps only")
            return False

        watcher = self

        class _Handler(FileSystemEventHandler):
            def on_created(self, event):
                if not event.is_directory:
                    watcher._enqueue(Path(event.src_path))

            def on_modified(self, event):
                if not event.is_directory:
                    watcher._enqueue(Path(event.src_path))

            def on_moved(self, event):
                # The rename that ends a download: report.pdf.crdownload ->
                # report.pdf. The destination is the file we want.
                dest = getattr(event, "dest_path", None)
                if dest and not event.is_directory:
                    watcher._enqueue(Path(dest))

        try:
            self._observer = Observer()
            self._observer.schedule(_Handler(), str(root), recursive=False)
            self._observer.start()
        except Exception as exc:
            self._emit("info", f"could not watch Downloads ({exc}); sweeps only")
            self._observer = None
            return False

        self._worker = threading.Thread(
            target=self._process_queue, name="downloads-watcher", daemon=True
        )
        self._worker.start()
        self._emit("info", f"watching {root} for new documents")
        return True

    def stop(self) -> None:
        self._stopping.set()
        if self._observer is not None:
            try:
                self._observer.stop()
                self._observer.join(timeout=5)
            except Exception:
                pass
            self._observer = None
        if self._worker is not None:
            # The worker closes its own journal on the way out — closing it from
            # here would be the same cross-thread mistake in reverse.
            self._worker.join(timeout=SETTLE_TIMEOUT + 5)
            self._worker = None

    # ── queue ────────────────────────────────────────────────────
    def _enqueue(self, path: Path) -> None:
        """Called on the observer thread — must stay cheap.

        Ingesting here would block the observer and drop later events, so the
        path is handed to a worker. Duplicates are collapsed because a single
        download emits many modify events for the same file.
        """
        if not is_indexable(path):
            return
        key = str(path).lower()
        with self._lock:
            if key in self._pending:
                return
            self._pending.add(key)
        self._queue.put(path)

    def _process_queue(self) -> None:
        # The connection belongs to this thread for its whole life.
        self._journal = Journal()
        try:
            while not self._stopping.is_set():
                try:
                    path = self._queue.get(timeout=0.5)
                except Empty:
                    continue
                try:
                    self._handle(path)
                except Exception as exc:  # never let the worker die
                    self._emit("error", f"watcher failed on {path.name}: {exc}")
                finally:
                    with self._lock:
                        self._pending.discard(str(path).lower())
        finally:
            if self._journal is not None:
                self._journal.close()
                self._journal = None

    def _handle(self, path: Path) -> None:
        from app import consent

        # Re-checked per file, not once at startup: consent can be withdrawn
        # while the daemon is running, and that must take effect immediately.
        state, _record = consent.status()
        if state != "granted":
            return

        if not wait_until_settled(path, self._settle_seconds):
            return  # vanished, or still being written when we gave up
        if self._stopping.is_set():
            return

        journal = self._journal
        if journal is None:
            return

        try:
            mtime = path.stat().st_mtime
        except OSError:
            return

        # Shared with the periodic sweep, so a file is never indexed twice —
        # whichever mechanism sees it first records it.
        if journal.known_files().get(str(path)) == mtime:
            return

        result = self._assistant.ingest_path(str(path))
        journal.mark_seen({str(path): mtime})

        if result.get("ok"):
            self._emit("notable", f"Indexed {path.name} ({result.get('chunks', 0)} chunks)")
        else:
            self._emit("error", f"could not index {path.name}: {result.get('error')}")
