"""Tests for staged duplicate detection and the consent gate."""
from __future__ import annotations

from pathlib import Path

import pytest

from app.analyzer import duplicates
from app.analyzer.walker import FileEntry


def entry_for(path: Path, label: str = "test") -> FileEntry:
    st = path.stat()
    return FileEntry(
        path=path, size=st.st_size, mtime=st.st_mtime, atime=st.st_atime,
        root_label=label,
    )


def write(path: Path, content: bytes) -> Path:
    path.write_bytes(content)
    return path


BIG = b"x" * 10_000  # comfortably above duplicates.MIN_SIZE


def test_identical_files_are_grouped(tmp_path):
    a = write(tmp_path / "a.bin", BIG)
    b = write(tmp_path / "b.bin", BIG)
    groups = duplicates.find_duplicates([entry_for(a), entry_for(b)])
    assert len(groups) == 1
    assert groups[0].count == 2
    assert groups[0].wasted_bytes == len(BIG)


def test_different_content_same_size_is_not_a_duplicate(tmp_path):
    """Guards the size-bucket stage from being mistaken for the answer."""
    a = write(tmp_path / "a.bin", b"a" * 10_000)
    b = write(tmp_path / "b.bin", b"b" * 10_000)
    assert duplicates.find_duplicates([entry_for(a), entry_for(b)]) == []


def test_files_differing_only_in_the_middle_are_not_duplicates(tmp_path):
    """Stage 2 samples the head and tail; identical edges must not be enough."""
    head, tail = b"H" * 9000, b"T" * 9000
    a = write(tmp_path / "a.bin", head + b"AAAA" + tail)
    b = write(tmp_path / "b.bin", head + b"BBBB" + tail)
    assert duplicates.find_duplicates([entry_for(a), entry_for(b)]) == []


def test_tiny_files_are_ignored(tmp_path):
    a = write(tmp_path / "a.txt", b"hi")
    b = write(tmp_path / "b.txt", b"hi")
    assert duplicates.find_duplicates([entry_for(a), entry_for(b)]) == []


def test_cloud_placeholders_are_never_hashed(tmp_path):
    """Reading a cloud-only file forces a download; it must be excluded."""
    a = entry_for(write(tmp_path / "a.bin", BIG))
    b = entry_for(write(tmp_path / "b.bin", BIG))
    b.is_cloud_placeholder = True
    assert duplicates.find_duplicates([a, b]) == []


def test_keeper_is_the_oldest_copy(tmp_path):
    import os
    import time

    a = write(tmp_path / "original.bin", BIG)
    b = write(tmp_path / "copy (1).bin", BIG)
    old = time.time() - 86_400 * 30
    os.utime(a, (old, old))

    group = duplicates.find_duplicates([entry_for(a), entry_for(b)])[0]
    assert group.keeper.path == a
    assert [e.path for e in group.redundant] == [b]


def test_three_copies_report_two_wasted(tmp_path):
    entries = [
        entry_for(write(tmp_path / f"c{i}.bin", BIG)) for i in range(3)
    ]
    group = duplicates.find_duplicates(entries)[0]
    assert group.count == 3
    assert group.wasted_bytes == len(BIG) * 2
    assert len(group.redundant) == 2


def test_hash_cache_is_used_on_a_second_pass(tmp_path):
    a = write(tmp_path / "a.bin", BIG)
    b = write(tmp_path / "b.bin", BIG)
    entries = [entry_for(a), entry_for(b)]

    from app.analyzer.cache import entry_key

    cache: dict[str, str] = {}
    duplicates.find_duplicates(entries, hash_cache=cache, cache_key=entry_key)
    assert len(cache) == 2

    # A second run must reach the same conclusion from the cache alone.
    groups = duplicates.find_duplicates(entries, hash_cache=cache, cache_key=entry_key)
    assert len(groups) == 1 and groups[0].count == 2


def test_cache_key_changes_when_content_changes(tmp_path):
    """Keying on path alone would serve a stale hash after an edit."""
    import os
    import time

    from app.analyzer.cache import entry_key

    p = write(tmp_path / "a.bin", BIG)
    before = entry_key(entry_for(p))

    time.sleep(0.01)
    write(p, BIG + b"more")
    os.utime(p, (time.time() + 5, time.time() + 5))
    assert entry_key(entry_for(p)) != before
