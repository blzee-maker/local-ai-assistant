"""The work the daemon does on a schedule.

Every job runs with **no confirmer attached**, so any tool needing permission
answers no. That is the point of the default set when the registry was built:
nobody is present to be asked, and an absent user is not consent.

Jobs must not raise. A job that throws is recorded as an error and the scheduler
carries on — one broken sweep should never take down the process that also runs
the others (rule 10).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from app.daemon.journal import Journal
from config import settings

HOUR = 3600.0
MINUTE = 60.0


@dataclass
class JobResult:
    outcome: str  # "ok" | "skipped" | "failed"
    detail: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    # Something a person would actually want told to them. Most sweeps find
    # nothing, and a briefing that reports every uneventful run is one nobody
    # reads (rule 4 is about being informative, not noisy).
    notable: str | None = None


@runtime_checkable
class Job(Protocol):
    name: str
    description: str
    interval_s: float
    run_on_start: bool

    def run(self, assistant: Any, journal: Journal) -> JobResult:
        ...


class IndexNewFilesJob(Job):
    """Index documents that appeared in the allowed folders since last sweep.

    Polling, not filesystem events. `watchdog` was the obvious choice and was
    rejected: Documents and Desktop are OneDrive-synced here, and sync churn
    produces a steady stream of create/modify events for files that did not
    change in any way the user would recognise. A watcher would also hold a
    thread and OS handles open permanently on a machine with little headroom,
    to notice files whose indexing is not urgent. A periodic mtime sweep reuses
    the walker that already exists, costs nothing between runs, and cannot miss
    a file because an event was dropped while the daemon was stopped.
    """

    name = "index_new_files"
    description = "Index new documents from your allowed folders"
    run_on_start = True

    # Only formats the RAG pipeline can actually read.
    EXTENSIONS = {".pdf", ".txt", ".md", ".markdown", ".docx"}
    MAX_PER_SWEEP = 20  # a first sweep of a full Downloads folder must not stall

    def __init__(self, interval_s: float | None = None) -> None:
        self.interval_s = interval_s or settings.daemon_index_interval_minutes * MINUTE

    def run(self, assistant: Any, journal: Journal) -> JobResult:
        from app import consent, files as filesvc
        from app.analyzer import walker

        state, _record = consent.status()
        if state != "granted":
            return JobResult(
                outcome="skipped",
                detail="file analysis has not been approved",
            )

        roots = filesvc.allowed_roots()
        if not roots:
            return JobResult(outcome="skipped", detail="no readable folders")

        entries, _stats = walker.walk(roots, scan_limit=50_000)
        known = journal.known_files()

        candidates = [
            entry
            for entry in entries
            if entry.suffix in self.EXTENSIONS
            and not entry.is_cloud_placeholder
            and known.get(str(entry.path)) != entry.mtime
        ]
        # Newest first: if there are more than a sweep's worth, the recent ones
        # are the ones the user is likely to ask about.
        candidates.sort(key=lambda e: e.mtime, reverse=True)

        first_sweep = not known
        if first_sweep:
            # Do not ingest a whole existing library on first run. Record what is
            # already there as seen, and index only what appears from now on —
            # the user asked for new files to be picked up, not for their entire
            # Documents folder to be silently absorbed.
            journal.mark_seen({str(e.path): e.mtime for e in entries})
            return JobResult(
                outcome="ok",
                detail=f"baseline recorded ({len(entries):,} files); new files from now on",
                payload={"baseline": len(entries)},
            )

        batch = candidates[: self.MAX_PER_SWEEP]
        indexed: list[str] = []
        failed: list[str] = []
        for entry in batch:
            result = assistant.ingest_path(str(entry.path))
            if result.get("ok"):
                indexed.append(entry.name)
            else:
                failed.append(entry.name)

        journal.mark_seen({str(e.path): e.mtime for e in batch})

        if not indexed and not failed:
            return JobResult(outcome="ok", detail="no new documents")

        detail = f"indexed {len(indexed)}"
        if failed:
            detail += f", {len(failed)} failed"
        remaining = len(candidates) - len(batch)
        if remaining > 0:
            detail += f", {remaining} queued for next sweep"

        return JobResult(
            outcome="ok",
            detail=detail,
            payload={"indexed": indexed, "failed": failed, "remaining": remaining},
            notable=(
                f"Indexed {len(indexed)} new document(s): {', '.join(indexed[:5])}"
                if indexed else None
            ),
        )


class DiskScanJob(Job):
    """Re-scan the disk periodically so chat answers stay current."""

    name = "disk_scan"
    description = "Re-scan your folders for duplicates, corruption and idle storage"
    run_on_start = False  # expensive; never on launch

    def __init__(self, interval_s: float | None = None) -> None:
        self.interval_s = interval_s or settings.daemon_scan_interval_hours * HOUR

    def run(self, assistant: Any, journal: Journal) -> JobResult:
        from app import consent
        from app.analyzer import run_scan
        from app.core import diskintent

        state, _record = consent.status()
        if state != "granted":
            return JobResult(
                outcome="skipped", detail="file analysis has not been approved"
            )

        report = run_scan()
        diskintent.save_report(report)

        from app.analyzer.service import human_bytes

        notable = None
        problems = len(report.problems)
        wasted = report.wasted_bytes
        # Only worth interrupting someone for if there is something to act on.
        if problems or wasted > 100 * 1024 * 1024:
            bits = []
            if wasted:
                bits.append(f"{human_bytes(wasted)} in duplicate files")
            if problems:
                bits.append(f"{problems} damaged or mislabelled file(s)")
            notable = "Disk scan found " + " and ".join(bits) + "."

        return JobResult(
            outcome="ok",
            detail=report.summary_line(),
            payload={
                "files": report.stats.files_seen,
                "wasted_bytes": wasted,
                "problems": problems,
            },
            notable=notable,
        )


class SystemHealthJob(Job):
    """Notice sustained resource pressure rather than a momentary spike."""

    name = "system_health"
    description = "Watch for sustained memory, disk or battery pressure"
    run_on_start = True

    MEMORY_ALERT = 90.0
    DISK_ALERT = 92.0
    BATTERY_ALERT = 15

    def __init__(self, interval_s: float | None = None) -> None:
        self.interval_s = interval_s or settings.daemon_health_interval_minutes * MINUTE

    def run(self, assistant: Any, journal: Journal) -> JobResult:
        from app.tools.system import system_snapshot

        snapshot = system_snapshot()
        memory = snapshot["memory"]
        alerts: list[str] = []

        if memory["percent_used"] >= self.MEMORY_ALERT:
            alerts.append(
                f"memory is {memory['percent_used']:.0f}% used "
                f"({memory['available_gb']:.2f} GB free)"
            )
        for disk in snapshot["disks"]:
            if disk["percent_used"] >= self.DISK_ALERT:
                alerts.append(
                    f"disk {disk['mount']} is {disk['percent_used']:.0f}% full "
                    f"({disk['free_gb']:.1f} GB left)"
                )
        battery = snapshot.get("battery")
        if battery and not battery["plugged_in"] and battery["percent"] <= self.BATTERY_ALERT:
            alerts.append(f"battery is at {battery['percent']}% and not charging")

        return JobResult(
            outcome="ok",
            detail="; ".join(alerts) if alerts else "healthy",
            payload={
                "memory_percent": memory["percent_used"],
                "cpu_percent": snapshot["cpu_percent"],
                "alerts": alerts,
            },
            notable=("Attention: " + "; ".join(alerts)) if alerts else None,
        )


def default_jobs() -> list[Job]:
    return [IndexNewFilesJob(), SystemHealthJob(), DiskScanJob()]
