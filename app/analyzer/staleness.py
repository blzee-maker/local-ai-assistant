"""Finding large files that nobody has touched in a long time.

"Unused" cannot be measured directly — the filesystem only records when a file
was last *modified* and, sometimes, last *accessed*. Which of those is available
changes the meaning of the result, so the caller is told which signal was used
and the report repeats it. See `walker.last_access_tracking()`.

Ranking is by `size × age` rather than size alone: a 4 GB ISO downloaded
yesterday is probably wanted, while a 300 MB installer last opened three years
ago is probably not. Sorting purely by size surfaces the former and buries the
latter, which is exactly backwards for reclaiming space.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from app.analyzer.walker import FileEntry

DAY = 86_400.0

# Below this, nothing is worth a line in a storage report.
MIN_INTERESTING_BYTES = 10 * 1024 * 1024  # 10 MB
MIN_AGE_DAYS = 180.0

# Extensions that are usually re-downloadable — worth calling out separately
# because deleting them is low-regret.
DISPOSABLE_EXTS = {
    ".iso", ".msi", ".exe", ".dmg", ".pkg", ".deb", ".rpm",
    ".zip", ".rar", ".7z", ".tar", ".gz", ".tgz", ".cab",
}


@dataclass
class StaleFile:
    entry: FileEntry
    age_days: float
    signal: str  # "accessed" or "modified"

    @property
    def score(self) -> float:
        """Bytes-years: how much space, weighted by how long it has sat there."""
        return self.entry.size * (self.age_days / 365.0)

    @property
    def is_disposable(self) -> bool:
        return self.entry.suffix in DISPOSABLE_EXTS


def find_stale(
    entries: list[FileEntry],
    *,
    use_atime: bool,
    min_bytes: int = MIN_INTERESTING_BYTES,
    min_age_days: float = MIN_AGE_DAYS,
    limit: int = 50,
) -> list[StaleFile]:
    """Large, long-untouched files, most reclaimable first."""
    now = time.time()
    signal = "accessed" if use_atime else "modified"
    results: list[StaleFile] = []

    for entry in entries:
        if entry.size < min_bytes:
            continue
        # A cloud placeholder occupies no local space — reporting it as
        # reclaimable storage would be simply wrong.
        if entry.is_cloud_placeholder:
            continue

        stamp = entry.atime if use_atime else entry.mtime
        age_days = max(0.0, (now - stamp) / DAY)
        if age_days < min_age_days:
            continue
        results.append(StaleFile(entry=entry, age_days=age_days, signal=signal))

    results.sort(key=lambda s: s.score, reverse=True)
    return results[:limit]


def total_stale_bytes(stale: list[StaleFile]) -> int:
    return sum(s.entry.size for s in stale)


# ── storage breakdown ────────────────────────────────────────────
@dataclass
class CategoryUsage:
    label: str
    count: int
    total_bytes: int


_CATEGORIES: list[tuple[str, set[str]]] = [
    ("Video", {".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".webm"}),
    ("Archives", {".zip", ".rar", ".7z", ".tar", ".gz", ".tgz", ".cab", ".iso"}),
    ("Installers", {".exe", ".msi", ".dmg", ".pkg", ".deb", ".rpm", ".appx"}),
    ("Images", {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp", ".heic", ".raw"}),
    ("Audio", {".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a", ".wma"}),
    ("Documents", {".pdf", ".docx", ".doc", ".xlsx", ".xls", ".pptx", ".ppt", ".txt", ".md", ".csv"}),
    ("Code / data", {".py", ".js", ".ts", ".json", ".xml", ".html", ".css", ".sql", ".ipynb"}),
]


def usage_by_category(entries: list[FileEntry]) -> list[CategoryUsage]:
    """Where the space actually goes, biggest category first."""
    buckets: dict[str, list[int]] = {label: [0, 0] for label, _ in _CATEGORIES}
    buckets["Other"] = [0, 0]

    lookup = {ext: label for label, exts in _CATEGORIES for ext in exts}

    for entry in entries:
        if entry.is_cloud_placeholder:
            continue
        label = lookup.get(entry.suffix, "Other")
        buckets[label][0] += 1
        buckets[label][1] += entry.size

    usage = [
        CategoryUsage(label=label, count=count, total_bytes=total)
        for label, (count, total) in buckets.items()
        if count
    ]
    usage.sort(key=lambda c: c.total_bytes, reverse=True)
    return usage
