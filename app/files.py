"""Safe local-file browsing, restricted to an allowlist of folders.

The whole security model lives here. The app can only see files that sit inside
one of the configured `allowed_file_dirs`; every requested path is resolved and
checked against those roots so a crafted path (``..\\..\\Windows\\...``) cannot
escape. Combined with binding the server to 127.0.0.1, this keeps the "read my
files" feature from becoming a way to exfiltrate the whole disk.
"""
from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from config import settings

SUPPORTED_EXTS = {".pdf", ".txt", ".md", ".markdown", ".docx"}


def _tokens(s: str) -> list[str]:
    """Split on spaces and filename separators so 'quokka notes' == 'quokka_notes'."""
    return [t for t in re.split(r"[\s._\-()]+", s.lower()) if t]


def _name_matches(query: str, filename: str) -> bool:
    """True if every query token appears in the filename's token stream."""
    q_tokens = _tokens(query)
    if not q_tokens:
        return True
    haystack = " ".join(_tokens(filename))
    return all(qt in haystack for qt in q_tokens)


def allowed_roots() -> list[tuple[str, Path]]:
    """(label, resolved_path) for each configured root that actually exists."""
    roots: list[tuple[str, Path]] = []
    seen: set[Path] = set()
    for entry in settings.allowed_file_dirs:
        p = Path(entry).resolve()
        if p in seen or not (p.exists() and p.is_dir()):
            continue
        seen.add(p)
        roots.append((p.name or str(p), p))
    return roots


def resolve_allowed(path_str: str) -> Path:
    """Resolve a path and confirm it is a supported file inside an allowed root.

    Raises ValueError otherwise — this is the guard that prevents traversal.
    """
    p = Path(path_str).resolve()
    if p.suffix.lower() not in SUPPORTED_EXTS:
        raise ValueError(f"Unsupported file type: {p.suffix or '(none)'}")
    for _label, root in allowed_roots():
        try:
            p.relative_to(root)
        except ValueError:
            continue
        if p.exists() and p.is_file():
            return p
    raise ValueError("Path is not inside an allowed folder")


def search_files(query: str, limit: int) -> list[dict[str, Any]]:
    """Find supported files whose name contains `query`, newest first."""
    q = query.lower().strip()
    matches: list[dict[str, Any]] = []
    scanned = 0
    scan_limit = settings.file_search_scan_limit

    for label, root in allowed_roots():
        for dirpath, _dirnames, filenames in os.walk(root):
            for name in filenames:
                scanned += 1
                if scanned > scan_limit:
                    return _finish(matches, limit)
                if Path(name).suffix.lower() not in SUPPORTED_EXTS:
                    continue
                if q and not _name_matches(q, name):
                    continue
                fp = Path(dirpath) / name
                try:
                    st = fp.stat()
                except OSError:
                    continue
                matches.append(
                    {
                        "path": str(fp),
                        "name": name,
                        "root": label,
                        "size_kb": round(st.st_size / 1024, 1),
                        "modified": datetime.fromtimestamp(st.st_mtime).strftime(
                            "%Y-%m-%d"
                        ),
                        "_mtime": st.st_mtime,
                    }
                )
    return _finish(matches, limit)


def find_latest(name: str, folder_hint: str | None = None) -> dict[str, Any] | None:
    """Best single match for `name`, newest first, optionally within a folder.

    Implements the "if several files share a name, pick the most recent" rule.
    """
    results = search_files(name, settings.file_search_max_results)
    if folder_hint:
        hint = folder_hint.lower().strip()
        scoped = [r for r in results if hint in r["root"].lower()]
        if scoped:
            results = scoped
    return results[0] if results else None


def _finish(matches: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    matches.sort(key=lambda m: m["_mtime"], reverse=True)
    for m in matches:
        m.pop("_mtime", None)
    return matches[:limit]
