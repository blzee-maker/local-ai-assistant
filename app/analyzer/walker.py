"""Filesystem walking that survives contact with a real Windows profile.

A naive ``os.walk`` over Downloads/Documents/Desktop goes wrong in four ways,
each of which this module handles explicitly:

1. **Cloud placeholders.** With OneDrive redirection (very common for Documents
   and Desktop), many "files" are stubs whose bytes live in the cloud. Opening
   one triggers a download. A duplicate-hasher that reads every file would
   quietly pull gigabytes onto a disk we were asked to *free up* — the exact
   opposite of the point. Placeholders are detected and reported, never read.
2. **Reparse points.** Junctions and symlinks can point back up the tree and
   turn a walk into an infinite loop, or make one file appear under many paths
   and register as its own duplicate.
3. **Hard links.** The same bytes under two names on NTFS. Reporting those as
   "duplicates you can delete" is wrong — deleting one frees nothing.
4. **Permission errors.** Parts of a profile are simply unreadable, and a scan
   must report that rather than abort.

Everything here is read-only: the walker stats and (later) reads files. Nothing
in this package writes to or deletes from a user folder.
"""
from __future__ import annotations

import os
import stat
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterator

# Windows file attributes relevant to cloud-backed storage.
FILE_ATTRIBUTE_REPARSE_POINT = 0x400
FILE_ATTRIBUTE_OFFLINE = 0x1000
FILE_ATTRIBUTE_RECALL_ON_OPEN = 0x40000
FILE_ATTRIBUTE_RECALL_ON_DATA_ACCESS = 0x400000

# A file whose bytes are not local. Reading it forces a network download.
_CLOUD_ATTRS = (
    FILE_ATTRIBUTE_OFFLINE
    | FILE_ATTRIBUTE_RECALL_ON_OPEN
    | FILE_ATTRIBUTE_RECALL_ON_DATA_ACCESS
)

# Directories that are never interesting and are often enormous.
SKIP_DIR_NAMES = {
    "node_modules", "__pycache__", ".git", ".svn", ".hg", ".venv", "venv",
    "$RECYCLE.BIN", "System Volume Information", ".cache", "AppData",
    ".gradle", ".nuget", "site-packages", ".next", "dist-info",
}


@dataclass
class FileEntry:
    """One file on disk, as observed by the walker. Read-only snapshot."""

    path: Path
    size: int
    mtime: float
    atime: float
    root_label: str
    is_cloud_placeholder: bool = False
    # NTFS file identity; lets us spot two names pointing at the same bytes.
    link_key: tuple[int, int] | None = None

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def suffix(self) -> str:
        return self.path.suffix.lower()

    @property
    def modified(self) -> datetime:
        return datetime.fromtimestamp(self.mtime, tz=timezone.utc).astimezone()

    @property
    def accessed(self) -> datetime:
        return datetime.fromtimestamp(self.atime, tz=timezone.utc).astimezone()


@dataclass
class WalkStats:
    """What the walk saw — surfaced in the report so numbers can be trusted."""

    files_seen: int = 0
    bytes_seen: int = 0
    dirs_skipped: int = 0
    cloud_placeholders: int = 0
    hardlink_duplicates: int = 0
    permission_errors: list[str] = field(default_factory=list)
    reparse_points_skipped: int = 0
    hit_scan_limit: bool = False


def _is_cloud_placeholder(attrs: int) -> bool:
    return bool(attrs & _CLOUD_ATTRS)


def _file_attributes(st: os.stat_result) -> int:
    return getattr(st, "st_file_attributes", 0)


def _link_key(st: os.stat_result) -> tuple[int, int] | None:
    """(volume, index) identity for a file, when the OS provides one."""
    ino = getattr(st, "st_ino", 0)
    dev = getattr(st, "st_dev", 0)
    if not ino:
        return None
    return (dev, ino)


def walk(
    roots: list[tuple[str, Path]],
    *,
    scan_limit: int = 200_000,
    skip_dirs: set[str] | None = None,
    on_progress: Callable[[int], None] | None = None,
) -> tuple[list[FileEntry], WalkStats]:
    """Collect file metadata under `roots`. Never opens a file.

    `roots` is a list of (label, path). Returns the entries plus a stats record
    describing what was skipped and why.
    """
    skip = SKIP_DIR_NAMES if skip_dirs is None else skip_dirs
    stats = WalkStats()
    entries: list[FileEntry] = []
    seen_links: set[tuple[int, int]] = set()
    visited_dirs: set[tuple[int, int]] = set()

    for label, root in roots:
        for dirpath, dirnames, filenames in os.walk(root, onerror=_record_error(stats)):
            here = Path(dirpath)

            # Prune uninteresting subtrees in place (os.walk honours this).
            kept = []
            for d in dirnames:
                if d in skip or d.startswith("."):
                    stats.dirs_skipped += 1
                    continue
                full = here / d
                if _is_reparse_point(full):
                    stats.reparse_points_skipped += 1
                    stats.dirs_skipped += 1
                    continue
                kept.append(d)
            dirnames[:] = kept

            # A directory reachable by two paths must only be walked once.
            try:
                dir_id = _link_key(here.stat())
            except OSError:
                dir_id = None
            if dir_id is not None:
                if dir_id in visited_dirs:
                    dirnames[:] = []
                    continue
                visited_dirs.add(dir_id)

            for name in filenames:
                if len(entries) >= scan_limit:
                    stats.hit_scan_limit = True
                    return entries, stats

                fp = here / name
                try:
                    st = fp.stat(follow_symlinks=False)
                except OSError as exc:
                    _note_permission(stats, fp, exc)
                    continue

                if not stat.S_ISREG(st.st_mode):
                    continue

                attrs = _file_attributes(st)
                if attrs & FILE_ATTRIBUTE_REPARSE_POINT:
                    stats.reparse_points_skipped += 1
                    continue

                key = _link_key(st)
                if key is not None and key in seen_links:
                    # Same bytes already counted under another name.
                    stats.hardlink_duplicates += 1
                    continue
                if key is not None:
                    seen_links.add(key)

                is_cloud = _is_cloud_placeholder(attrs)
                if is_cloud:
                    stats.cloud_placeholders += 1

                entries.append(
                    FileEntry(
                        path=fp,
                        size=st.st_size,
                        mtime=st.st_mtime,
                        atime=st.st_atime,
                        root_label=label,
                        is_cloud_placeholder=is_cloud,
                        link_key=key,
                    )
                )
                stats.files_seen += 1
                stats.bytes_seen += st.st_size

                if on_progress is not None and stats.files_seen % 200 == 0:
                    on_progress(stats.files_seen)

    if on_progress is not None:
        on_progress(stats.files_seen)
    return entries, stats


def _is_reparse_point(path: Path) -> bool:
    try:
        st = path.stat(follow_symlinks=False)
    except OSError:
        return False
    if _file_attributes(st) & FILE_ATTRIBUTE_REPARSE_POINT:
        return True
    return path.is_symlink()


def _record_error(stats: WalkStats) -> Callable[[OSError], None]:
    def handler(exc: OSError) -> None:
        _note_permission(stats, Path(getattr(exc, "filename", "") or "?"), exc)

    return handler


def _note_permission(stats: WalkStats, path: Path, exc: OSError) -> None:
    # Cap the list — a locked subtree could otherwise produce thousands.
    if len(stats.permission_errors) < 25:
        stats.permission_errors.append(f"{path}: {exc.strerror or exc}")


# ── last-access-time reliability ─────────────────────────────────
def last_access_tracking() -> tuple[bool, str]:
    """Is st_atime meaningful on this machine? Returns (usable, explanation).

    Windows can disable last-access-time updates for performance, in which case
    atime is a copy of mtime and any "you haven't opened this in a year" claim
    would be fiction. A report must know which signal it is actually using.
    """
    if sys.platform != "win32":
        return True, "POSIX: last-access times are tracked"

    import subprocess

    try:
        out = subprocess.run(
            ["fsutil", "behavior", "query", "disablelastaccess"],
            capture_output=True,
            text=True,
            timeout=10,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        ).stdout
    except Exception:
        return False, "could not determine last-access policy; using modified time"

    # Values 0 and 2 mean updates are enabled; 1 and 3 mean disabled.
    digits = [c for c in out if c.isdigit()]
    value = digits[-1] if digits else None
    if value in {"0", "2"}:
        return True, "last-access times are tracked on this system"
    if value in {"1", "3"}:
        return False, "Windows has last-access updates disabled; using modified time"
    return False, "last-access policy unknown; using modified time"
