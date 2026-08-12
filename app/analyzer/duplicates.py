"""Exact duplicate detection via staged hashing.

Hashing every file to find duplicates is the obvious approach and the wrong one:
on a 7 GB profile it reads 7 GB to discover that most files are unique. Files of
*different sizes* can never be identical, and among same-size files a mismatch in
the first or last few KB rules them out immediately.

So the work is staged, cheapest filter first:

    stage 1  group by size          — no I/O at all
    stage 2  hash 8 KB head + tail  — one seek per candidate
    stage 3  full BLAKE2b           — only for files that survived both

In practice stage 1 eliminates the overwhelming majority, and stage 3 runs on a
small remainder. BLAKE2b is used over SHA-256 because it is faster in pure
Python for the same collision resistance, and this is not a security boundary.

Cloud placeholders are never hashed — reading one would force a download.
"""
from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Iterable

from app.analyzer.walker import FileEntry

CHUNK = 1024 * 1024   # full-hash read size
EDGE = 8192           # bytes sampled from each end in stage 2

# Files below this are ignored: thousands of tiny identical stubs (empty
# __init__.py, .gitkeep) are duplicates in the literal sense but recovering
# their space is pointless and the noise buries real findings.
MIN_SIZE = 4096


@dataclass
class DuplicateGroup:
    """A set of files with byte-identical content."""

    digest: str
    size: int
    entries: list[FileEntry]

    @property
    def count(self) -> int:
        return len(self.entries)

    @property
    def wasted_bytes(self) -> int:
        """Space recoverable by keeping one copy."""
        return self.size * (self.count - 1)

    @property
    def keeper(self) -> FileEntry:
        """The copy worth keeping: oldest, on the theory that it is the
        original and the newer ones are 'file (1).pdf' style accidents."""
        return min(self.entries, key=lambda e: e.mtime)

    @property
    def redundant(self) -> list[FileEntry]:
        keeper = self.keeper
        return [e for e in self.entries if e.path != keeper.path]


def _hash_edges(entry: FileEntry) -> str | None:
    """Fingerprint from the first and last EDGE bytes. None if unreadable."""
    try:
        with entry.path.open("rb") as fh:
            head = fh.read(EDGE)
            if entry.size > EDGE * 2:
                fh.seek(-EDGE, 2)
                tail = fh.read(EDGE)
            else:
                tail = b""
        digest = hashlib.blake2b(head + tail, digest_size=16)
        digest.update(str(entry.size).encode())
        return digest.hexdigest()
    except OSError:
        return None


def _hash_full(entry: FileEntry) -> str | None:
    try:
        digest = hashlib.blake2b(digest_size=32)
        with entry.path.open("rb") as fh:
            while chunk := fh.read(CHUNK):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def find_duplicates(
    entries: Iterable[FileEntry],
    *,
    min_size: int = MIN_SIZE,
    on_progress: Callable[[int, int], None] | None = None,
    hash_cache: dict[str, str] | None = None,
    cache_key: Callable[[FileEntry], str] | None = None,
) -> list[DuplicateGroup]:
    """Group `entries` into sets of byte-identical files, largest waste first.

    `hash_cache` (keyed by `cache_key(entry)`) lets a repeat scan skip re-reading
    files whose size and mtime are unchanged.
    """
    candidates = [
        e
        for e in entries
        if e.size >= min_size and not e.is_cloud_placeholder
    ]

    # Stage 1 — group by exact size. Pure metadata, no I/O.
    by_size: dict[int, list[FileEntry]] = defaultdict(list)
    for entry in candidates:
        by_size[entry.size].append(entry)
    same_size = [group for group in by_size.values() if len(group) > 1]

    # Stage 2 — cheap edge fingerprint within each size group.
    stage3_groups: list[list[FileEntry]] = []
    for group in same_size:
        by_edge: dict[str, list[FileEntry]] = defaultdict(list)
        for entry in group:
            edge = _hash_edges(entry)
            if edge is not None:
                by_edge[edge].append(entry)
        stage3_groups.extend(g for g in by_edge.values() if len(g) > 1)

    # Stage 3 — full content hash on the survivors only.
    total = sum(len(g) for g in stage3_groups)
    done = 0
    by_digest: dict[tuple[str, int], list[FileEntry]] = defaultdict(list)

    for group in stage3_groups:
        for entry in group:
            digest: str | None = None
            key = cache_key(entry) if cache_key else None
            if hash_cache is not None and key is not None:
                digest = hash_cache.get(key)
            if digest is None:
                digest = _hash_full(entry)
                if digest is not None and hash_cache is not None and key is not None:
                    hash_cache[key] = digest
            if digest is not None:
                by_digest[(digest, entry.size)].append(entry)

            done += 1
            if on_progress is not None:
                on_progress(done, total)

    groups = [
        DuplicateGroup(digest=digest, size=size, entries=members)
        for (digest, size), members in by_digest.items()
        if len(members) > 1
    ]
    groups.sort(key=lambda g: g.wasted_bytes, reverse=True)
    return groups


def total_wasted(groups: list[DuplicateGroup]) -> int:
    return sum(g.wasted_bytes for g in groups)
