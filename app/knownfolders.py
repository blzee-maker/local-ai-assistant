"""Resolve the user's real Downloads/Documents/Desktop folders.

Why this exists: ``Path.home() / "Documents"`` is wrong on Windows. When a folder
is redirected (OneDrive Backup, roaming profiles, a moved Downloads), the real
location moves but a **stale empty folder is often left behind at the old path**
— so the naive join silently resolves to a decoy directory with none of the
user's files in it. On the development machine this exact case was live:
``C:\\Users\\<u>\\Documents`` held 6 leftover items while the actual Documents
folder (427 files) sat in ``C:\\Users\\<u>\\OneDrive\\Documents``.

The only correct source on Windows is the shell's Known Folder API
(``SHGetKnownFolderPath``), which returns wherever the folder actually lives now.
Non-Windows platforms fall back to XDG-ish conventions.
"""
from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

# KNOWNFOLDERID GUIDs (shlobj.h).
_FOLDERID = {
    "Downloads": "374DE290-123F-4565-9164-39C4925E467B",
    "Documents": "FDD39AD0-238F-46AF-ADB4-6C85480369C7",
    "Desktop": "B4BFCC3A-DB2C-424C-B029-7FE99A87C641",
}

# Order matters only for display; callers treat the mapping as a set.
KNOWN_FOLDER_NAMES = tuple(_FOLDERID)


def _windows_known_folder(name: str) -> Path | None:
    """Ask the Windows shell where `name` actually lives. None if unavailable."""
    import ctypes
    from ctypes import wintypes

    class _GUID(ctypes.Structure):
        _fields_ = [
            ("Data1", ctypes.c_ulong),
            ("Data2", ctypes.c_ushort),
            ("Data3", ctypes.c_ushort),
            ("Data4", ctypes.c_ubyte * 8),
        ]

    guid = _GUID()
    ctypes.memmove(ctypes.byref(guid), uuid.UUID(_FOLDERID[name]).bytes_le, 16)

    out = ctypes.c_wchar_p()
    # KF_FLAG_DEFAULT (0) = current location, do not create if missing.
    hresult = ctypes.windll.shell32.SHGetKnownFolderPath(
        ctypes.byref(guid), 0, wintypes.HANDLE(0), ctypes.byref(out)
    )
    if hresult != 0 or not out.value:
        return None
    try:
        return Path(out.value)
    finally:
        ctypes.windll.ole32.CoTaskMemFree(out)


def _posix_known_folder(name: str) -> Path | None:
    """XDG user dirs, falling back to the conventional ~/Name."""
    xdg_key = {
        "Downloads": "XDG_DOWNLOAD_DIR",
        "Documents": "XDG_DOCUMENTS_DIR",
        "Desktop": "XDG_DESKTOP_DIR",
    }[name]
    from_env = os.environ.get(xdg_key)
    if from_env:
        return Path(os.path.expandvars(from_env)).expanduser()

    config = Path.home() / ".config" / "user-dirs.dirs"
    if config.exists():
        try:
            for line in config.read_text(encoding="utf-8", errors="ignore").splitlines():
                key, _, raw = line.strip().partition("=")
                if key.strip() != xdg_key:
                    continue
                value = raw.strip().strip('"')
                if value.startswith("$HOME"):
                    return Path.home() / value[len("$HOME") :].lstrip("/")
                return Path(value).expanduser()
        except OSError:
            pass
    return Path.home() / name


def known_folder(name: str) -> Path | None:
    """Real path of a known folder, or None if the platform can't resolve it."""
    if name not in _FOLDERID:
        raise ValueError(f"Unknown folder id: {name!r}")
    try:
        if sys.platform == "win32":
            return _windows_known_folder(name)
        return _posix_known_folder(name)
    except Exception:
        # Never let folder discovery take the app down — the caller falls back.
        return None


def user_folders() -> list[Path]:
    """Existing Downloads/Documents/Desktop paths, de-duplicated, order preserved.

    Falls back to ``~/Name`` only when the API gives us nothing, so a redirected
    profile resolves to the real location rather than a leftover decoy.
    """
    found: list[Path] = []
    seen: set[Path] = set()
    for name in KNOWN_FOLDER_NAMES:
        path = known_folder(name) or (Path.home() / name)
        try:
            resolved = path.resolve()
        except OSError:
            continue
        if resolved in seen or not resolved.is_dir():
            continue
        seen.add(resolved)
        found.append(resolved)
    return found
