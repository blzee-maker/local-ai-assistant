"""Runs a full disk analysis and returns a single structured report.

Report-only by design. Nothing in this package deletes, moves, or modifies a
user's file — findings are presented and the decision stays with the person who
owns the data. See `report.deletion_script()` for the escape hatch, which writes
a reviewable script rather than acting on its own.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from app.analyzer import duplicates, integrity, staleness, walker
from app.analyzer.cache import HashCache, entry_key
from app.analyzer.duplicates import DuplicateGroup
from app.analyzer.integrity import IntegrityResult, Verdict
from app.analyzer.staleness import CategoryUsage, StaleFile
from app.analyzer.walker import FileEntry, WalkStats
from config import settings

# Verifying every file in a large profile costs more time than it returns, so
# integrity checking is capped and prioritises the formats that actually break:
# documents and media that arrive via interrupted downloads.
INTEGRITY_PRIORITY_EXTS = {
    ".pdf", ".docx", ".xlsx", ".pptx", ".zip", ".epub", ".jar",
    ".png", ".jpg", ".jpeg", ".gif", ".json", ".gz", ".tgz",
}


@dataclass
class ScanReport:
    started_at: float
    duration_s: float
    roots: list[str]
    stats: WalkStats
    duplicate_groups: list[DuplicateGroup] = field(default_factory=list)
    integrity_results: list[IntegrityResult] = field(default_factory=list)
    stale_files: list[StaleFile] = field(default_factory=list)
    usage: list[CategoryUsage] = field(default_factory=list)
    atime_usable: bool = True
    atime_note: str = ""
    integrity_checked: int = 0
    integrity_skipped_for_budget: int = 0

    # ── headline numbers ─────────────────────────────────────────
    @property
    def wasted_bytes(self) -> int:
        return duplicates.total_wasted(self.duplicate_groups)

    @property
    def stale_bytes(self) -> int:
        return staleness.total_stale_bytes(self.stale_files)

    @property
    def problems(self) -> list[IntegrityResult]:
        return [r for r in self.integrity_results if r.is_problem]

    @property
    def unverifiable_count(self) -> int:
        return sum(
            1 for r in self.integrity_results if r.verdict == Verdict.UNVERIFIABLE
        )

    @property
    def reclaimable_bytes(self) -> int:
        """Duplicates plus stale files. These sets can overlap, so this is an
        upper bound and is labelled as such wherever it is shown."""
        return self.wasted_bytes + self.stale_bytes

    def summary_line(self) -> str:
        return (
            f"{self.stats.files_seen:,} files, {human_bytes(self.stats.bytes_seen)}; "
            f"{len(self.duplicate_groups)} duplicate groups "
            f"({human_bytes(self.wasted_bytes)}), "
            f"{len(self.problems)} integrity problems, "
            f"{len(self.stale_files)} stale files ({human_bytes(self.stale_bytes)})"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "started_at": self.started_at,
            "duration_s": round(self.duration_s, 2),
            "roots": self.roots,
            "totals": {
                "files": self.stats.files_seen,
                "bytes": self.stats.bytes_seen,
                "cloud_placeholders": self.stats.cloud_placeholders,
                "hardlinks_collapsed": self.stats.hardlink_duplicates,
                "permission_errors": len(self.stats.permission_errors),
                "hit_scan_limit": self.stats.hit_scan_limit,
            },
            "duplicates": {
                "groups": len(self.duplicate_groups),
                "wasted_bytes": self.wasted_bytes,
                "top": [
                    {
                        "size": g.size,
                        "copies": g.count,
                        "wasted_bytes": g.wasted_bytes,
                        "keep": str(g.keeper.path),
                        "redundant": [str(e.path) for e in g.redundant],
                    }
                    for g in self.duplicate_groups[:25]
                ],
            },
            "integrity": {
                "checked": self.integrity_checked,
                "problems": [
                    {
                        "path": str(r.entry.path),
                        "verdict": r.verdict.value,
                        "detail": r.detail,
                    }
                    for r in self.problems
                ],
                "unverifiable": self.unverifiable_count,
                "skipped_for_budget": self.integrity_skipped_for_budget,
            },
            "stale": {
                "signal": "accessed" if self.atime_usable else "modified",
                "note": self.atime_note,
                "total_bytes": self.stale_bytes,
                "files": [
                    {
                        "path": str(s.entry.path),
                        "size": s.entry.size,
                        "age_days": round(s.age_days),
                        "disposable": s.is_disposable,
                    }
                    for s in self.stale_files[:25]
                ],
            },
            "usage": [
                {"category": c.label, "count": c.count, "bytes": c.total_bytes}
                for c in self.usage
            ],
        }


def human_bytes(n: int | float) -> str:
    step = 1024.0
    value = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(value) < step or unit == "TB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= step
    return f"{value:.1f} TB"


def run_scan(
    roots: list[tuple[str, Path]] | None = None,
    *,
    check_integrity: bool = True,
    integrity_budget: int = 3000,
    use_cache: bool = True,
    on_phase: Callable[[str], None] | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> ScanReport:
    """Walk, de-duplicate, verify, and rank — then hand back a report."""
    from app import files as filesvc

    started = time.time()
    scan_roots = roots if roots is not None else filesvc.allowed_roots()

    def phase(name: str) -> None:
        if on_phase is not None:
            on_phase(name)

    # ── walk ─────────────────────────────────────────────────────
    phase("Scanning folders")
    entries, stats = walk_roots(scan_roots)

    atime_usable, atime_note = walker.last_access_tracking()

    # ── duplicates ───────────────────────────────────────────────
    cache: HashCache | None = None
    hash_cache: dict[str, str] = {}
    if use_cache:
        cache = HashCache(Path(settings.vector_store_dir).parent / "analyzer.sqlite3")
        hash_cache = cache.load()

        # Must happen BEFORE hashing: reading a file to fingerprint it updates
        # its access time, so the staleness signal has to be banked first.
        earliest = cache.merge_atimes({str(e.path): e.atime for e in entries})
        for entry in entries:
            known = earliest.get(str(entry.path))
            if known is not None and known < entry.atime:
                entry.atime = known
        cache.forget_atimes({str(e.path) for e in entries})

    phase("Hashing candidates")
    groups = duplicates.find_duplicates(
        entries,
        on_progress=on_progress,
        hash_cache=hash_cache,
        cache_key=entry_key,
    )

    if cache is not None:
        cache.save(hash_cache)
        cache.prune({entry_key(e) for e in entries})

    # ── integrity ────────────────────────────────────────────────
    results: list[IntegrityResult] = []
    skipped_budget = 0
    if check_integrity:
        phase("Verifying file integrity")
        candidates = [
            e
            for e in entries
            if e.suffix in INTEGRITY_PRIORITY_EXTS and not e.is_cloud_placeholder
        ]
        # Largest first: a truncated 2 GB download matters more than a 3 KB icon.
        candidates.sort(key=lambda e: e.size, reverse=True)
        if len(candidates) > integrity_budget:
            skipped_budget = len(candidates) - integrity_budget
            candidates = candidates[:integrity_budget]

        for i, entry in enumerate(candidates, 1):
            results.append(integrity.check(entry))
            if on_progress is not None:
                on_progress(i, len(candidates))

    # ── stale + usage ────────────────────────────────────────────
    phase("Ranking storage")
    stale = staleness.find_stale(entries, use_atime=atime_usable)
    usage = staleness.usage_by_category(entries)

    report = ScanReport(
        started_at=started,
        duration_s=time.time() - started,
        roots=[str(p) for _label, p in scan_roots],
        stats=stats,
        duplicate_groups=groups,
        integrity_results=results,
        stale_files=stale,
        usage=usage,
        atime_usable=atime_usable,
        atime_note=atime_note,
        integrity_checked=len(results),
        integrity_skipped_for_budget=skipped_budget,
    )

    if cache is not None:
        cache.record_scan(
            started, time.time(), stats.files_seen, stats.bytes_seen, report.summary_line()
        )
        cache.close()

    return report


def walk_roots(
    scan_roots: list[tuple[str, Path]],
    on_progress: Callable[[int], None] | None = None,
) -> tuple[list[FileEntry], WalkStats]:
    return walker.walk(
        scan_roots,
        scan_limit=max(settings.file_search_scan_limit, 200_000),
        on_progress=on_progress,
    )
