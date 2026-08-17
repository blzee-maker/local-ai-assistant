"""The loop that decides what runs and when.

Deliberately a plain loop over a persisted schedule rather than a threaded timer
per job. Jobs here are I/O-heavy and occasionally memory-hungry (a disk scan
hashes files); running them one at a time is the difference between a background
helper and a second workload competing with the user for a machine that already
has under a gigabyte free.

Failure handling follows rule 10: a job that raises is recorded and the loop
continues. Repeated failures back the job off exponentially rather than
hammering a broken dependency every interval.
"""
from __future__ import annotations

import signal
import time
from dataclasses import dataclass
from typing import Any, Callable

from app.daemon.jobs import Job, JobResult, default_jobs
from app.daemon.journal import Journal, JobRun
from config import settings

# After repeated failures, wait longer each time — capped so a job that starts
# working again is picked up within a reasonable window.
MAX_BACKOFF_MULTIPLIER = 8


@dataclass
class DueJob:
    job: Job
    due_at: float
    overdue_by: float


class Scheduler:
    def __init__(
        self,
        assistant: Any,
        jobs: list[Job] | None = None,
        journal: Journal | None = None,
        on_event: Callable[[str, str], None] | None = None,
    ) -> None:
        self._assistant = assistant
        self._jobs = jobs if jobs is not None else default_jobs()
        self._journal = journal or Journal()
        self._on_event = on_event
        self._stopping = False

    @property
    def journal(self) -> Journal:
        return self._journal

    @property
    def jobs(self) -> list[Job]:
        return list(self._jobs)

    def _emit(self, level: str, message: str) -> None:
        if self._on_event is not None:
            self._on_event(level, message)

    # ── scheduling ───────────────────────────────────────────────
    def next_due(self, job: Job, now: float | None = None) -> float:
        """When `job` should next run, honouring backoff after failures."""
        now = now if now is not None else time.time()
        state = self._journal.state(job.name)

        if state.last_run is None:
            # Never run: due immediately if it opts in, otherwise one full
            # interval away so a restart cannot trigger expensive work.
            return now if job.run_on_start else now + job.interval_s

        interval = job.interval_s
        if state.failures:
            interval *= min(2 ** state.failures, MAX_BACKOFF_MULTIPLIER)
        return state.last_run + interval

    def due_jobs(self, now: float | None = None) -> list[DueJob]:
        now = now if now is not None else time.time()
        due = []
        for job in self._jobs:
            at = self.next_due(job, now)
            if at <= now:
                due.append(DueJob(job=job, due_at=at, overdue_by=now - at))
        # Most overdue first, so a long-stopped daemon catches up sensibly.
        due.sort(key=lambda d: d.overdue_by, reverse=True)
        return due

    # ── execution ────────────────────────────────────────────────
    def run_job(self, job: Job) -> JobResult:
        """Run one job, recording the outcome whatever happens."""
        started = time.time()
        self._emit("start", f"{job.name} started")
        try:
            result = job.run(self._assistant, self._journal)
        except Exception as exc:
            result = JobResult(outcome="error", detail=f"{type(exc).__name__}: {exc}")
            self._emit("error", f"{job.name} failed: {exc}")
        else:
            level = {"ok": "ok", "skipped": "skip"}.get(result.outcome, "error")
            self._emit(level, f"{job.name}: {result.detail or result.outcome}")

        self._journal.record_run(
            JobRun(
                job=job.name,
                at=started,
                duration=time.time() - started,
                outcome=result.outcome,
                detail=result.detail,
                payload=result.payload or None,
            )
        )
        if result.notable:
            self._emit("notable", result.notable)
        return result

    def tick(self, now: float | None = None) -> list[tuple[Job, JobResult]]:
        """Run everything currently due. Returns what ran."""
        ran: list[tuple[Job, JobResult]] = []
        for entry in self.due_jobs(now):
            if self._stopping:
                break
            ran.append((entry.job, self.run_job(entry.job)))
        return ran

    def stop(self) -> None:
        self._stopping = True

    def run_forever(
        self, tick_seconds: float | None = None, watch: bool = True
    ) -> None:
        """Block, running due jobs until interrupted.

        Order matters. The catch-up sweep runs *before* the watcher starts:
        filesystem events only exist while the daemon is awake, so anything that
        arrived while it was stopped was never announced and is only found by
        sweeping. Arming the doorbell first would leave that gap unclosed.
        """
        interval = tick_seconds or settings.daemon_tick_seconds

        def handle_signal(_signum, _frame):
            self._emit("stop", "stopping after the current job")
            self.stop()

        # Ctrl-C should finish the job in flight rather than leaving a half-
        # written index behind.
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, handle_signal)
            except (ValueError, OSError, AttributeError):
                pass  # not the main thread, or unsupported on this platform

        self._emit("info", f"watching {len(self._jobs)} job(s)")

        # Catch up on everything missed while stopped, then arm the doorbell.
        try:
            self.tick()
        except Exception as exc:
            self._emit("error", f"scheduler error: {exc}")

        watcher = None
        if watch and not self._stopping:
            from app.daemon.watcher import DownloadsWatcher

            watcher = DownloadsWatcher(self._assistant, on_event=self._on_event)
            if not watcher.start():
                watcher = None

        try:
            while not self._stopping:
                slept = 0.0
                while slept < interval and not self._stopping:
                    time.sleep(min(1.0, interval - slept))
                    slept += 1.0
                if self._stopping:
                    break
                try:
                    self.tick()
                except Exception as exc:  # the loop itself must never die
                    self._emit("error", f"scheduler error: {exc}")
        finally:
            if watcher is not None:
                watcher.stop()
            self._journal.close()
            self._emit("stop", "stopped")


# ── briefing ─────────────────────────────────────────────────────
def briefing(journal: Journal, since_hours: float = 24.0) -> str:
    """What happened while you were away, as plain text.

    Composed deterministically. The facts are already known exactly, and asking
    a 3B model to phrase them risks inventing numbers that were never in the
    journal (rule 13).
    """
    since = time.time() - since_hours * 3600.0
    runs = journal.runs_since(since)
    if not runs:
        return "Nothing has run recently."

    lines: list[str] = []
    by_job: dict[str, list] = {}
    for run in runs:
        by_job.setdefault(run.job, []).append(run)

    for name, entries in sorted(by_job.items()):
        last = entries[-1]
        failures = sum(1 for e in entries if e.outcome in {"error", "failed"})
        summary = f"{name}: ran {len(entries)}x — {last.detail or last.outcome}"
        if failures:
            summary += f" ({failures} failure(s))"
        lines.append(summary)

    return "\n".join(lines)
