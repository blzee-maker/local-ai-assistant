"""Answering questions about the user's disk from a cached scan.

A full scan takes tens of seconds — far too long to run inside a chat turn every
time someone asks "what's using my space?". So the last report is cached on disk
and reused, with its age stated in the answer. The user is told when the data is
from rather than being silently served stale numbers.

Scanning is never triggered implicitly by a chat message: it reads the user's
files, and that is a decision they make explicitly with `assistant scan`.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from config import settings

REPORT_CACHE = Path(settings.upload_dir).parent / "last_scan.json"

DISK_GROUNDING = (
    "The user asked about disk usage, duplicate files, corrupted files, or "
    "storage. Here are the findings from a local disk scan of their machine.\n"
    "Scan age: {age}\n\n"
    "--- SCAN RESULTS START ---\n{results}\n--- SCAN RESULTS END ---\n\n"
    "Answer using only these findings. Quote real file names and sizes. If the "
    "answer is not in the results, say so. Never suggest a deletion command; "
    "tell them to run `assistant scan --cleanup-script cleanup.ps1` and review "
    "it.\n\nUser's question: {question}"
)

NO_SCAN_NOTE = (
    "The user asked about their disk, but no scan results are available. Say so "
    "plainly and offer to run a scan. Do not invent file names, sizes, or "
    "findings, and do not tell the user to run a command themselves."
)

# After this, findings are still worth answering from but no longer worth
# presenting as current. The answer says how old they are and offers a refresh
# rather than quietly serving week-old numbers (rule 4).
STALE_AFTER_DAYS = 7.0


def is_stale(saved_at: float) -> bool:
    return (time.time() - saved_at) > STALE_AFTER_DAYS * 86_400

_DISK_VERBS = (
    "duplicate", "duplicates", "corrupt", "corrupted", "broken", "space",
    "storage", "disk", "cleanup", "clean up", "free up", "unused", "wasting",
    "wasted", "large files", "big files", "junk", "declutter", "full",
)
_DISK_CONTEXT = (
    "file", "files", "disk", "drive", "storage", "space", "folder", "downloads",
    "documents", "desktop", "gb", "mb", "scan",
)


# How full a drive is, is *not* a disk-scan question. The scan knows which
# files exist; it never learns the volume's capacity or free space. Asked "how
# much free space is on my drive?" the scan tool answered from file totals and
# reported 7.1 GB of storage, on a 415 GB disk, then apologised for not knowing
# the free space — having been handed the question it could not answer.
# Those figures are live and belong to system_status, so this tool stands aside.
_CAPACITY_PHRASES = (
    "free space", "space left", "space remaining", "available space",
    "space available", "how full", "capacity", "running out of space",
    "space is left", "space do i have", "space have i got",
)


def asks_about_drive_capacity(text: str) -> bool:
    """Is this about how full the drive is, rather than about its files?"""
    lowered = text.lower()
    return any(phrase in lowered for phrase in _CAPACITY_PHRASES)


def looks_like_disk_question(text: str) -> bool:
    """Cheap gate — only inject scan data when the question is about the disk."""
    t = text.lower()
    return any(v in t for v in _DISK_VERBS) and any(c in t for c in _DISK_CONTEXT)


def save_report(report: Any) -> None:
    """Cache a finished scan so chat can answer from it."""
    REPORT_CACHE.parent.mkdir(parents=True, exist_ok=True)
    from app.analyzer import report as report_mod

    payload = {
        "saved_at": time.time(),
        "summary": report.summary_line(),
        "prompt_text": report_mod.to_prompt_text(report),
    }
    REPORT_CACHE.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_report() -> dict[str, Any] | None:
    if not REPORT_CACHE.exists():
        return None
    try:
        return json.loads(REPORT_CACHE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def describe_age(saved_at: float) -> str:
    seconds = max(0.0, time.time() - saved_at)
    if seconds < 90:
        return "just now"
    minutes = seconds / 60
    if minutes < 90:
        return f"{minutes:.0f} minutes ago"
    hours = minutes / 60
    if hours < 36:
        return f"{hours:.0f} hours ago"
    return f"{hours / 24:.0f} days ago"


def ground_prompt(question: str) -> tuple[str | None, str | None]:
    """Return (grounded_prompt, correction_note) for a disk question."""
    cached = load_report()
    if cached is None:
        return None, NO_SCAN_NOTE
    return (
        DISK_GROUNDING.format(
            age=describe_age(cached.get("saved_at", 0.0)),
            results=cached.get("prompt_text", ""),
            question=question,
        ),
        None,
    )
