"""Finding, launching, raising and closing desktop applications.

Win32 primitives only — no tool or prompt concepts — so the behaviour can be
tested directly against a real desktop, which is the only way any of this was
ever going to be got right (rule 11).

Three facts about Windows shaped everything here:

* **The Start Menu is not the list of installed apps.** Walking `*.lnk` under
  the two Start Menu roots finds Visual Studio Code and misses Spotify
  entirely, because Store apps have no shortcut file. `Get-StartApps` returns
  both — 272 entries on this machine against 195 shortcuts — so that is the
  resolver, and shortcuts are not consulted at all.
* **A Store app has no path to execute.** It has an AppID, launched through
  `shell:AppsFolder`. So launching goes through the shell for everything,
  rather than one code path for real executables and another for the rest.
* **Closing is not killing.** `psutil.terminate()` on Windows is
  `TerminateProcess`: the program is stopped where it stands and anything
  unsaved is gone. Posting `WM_CLOSE` to its windows is what clicking the X
  does — the app gets to prompt about unsaved work. Closing an app must be the
  second thing, never the first (rule 2).
"""
from __future__ import annotations

import ctypes
import os
import subprocess
import sys
import time
from ctypes import wintypes
from dataclasses import dataclass

WM_CLOSE = 0x0010
SW_RESTORE = 9
SW_SHOW = 5

# Declaring prototypes is not tidiness here, it is correctness. ctypes assumes
# every undeclared function returns a C int, so GetForegroundWindow handed back
# a 64-bit window handle truncated to 32 bits. Raising a window then always
# reported failure — the comparison against the real handle could never
# match — and the assistant told the user Windows had refused a focus change
# that had in fact worked.
def _bind() -> None:
    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32

    user32.GetForegroundWindow.restype = wintypes.HWND
    user32.GetForegroundWindow.argtypes = []
    user32.SetForegroundWindow.restype = wintypes.BOOL
    user32.SetForegroundWindow.argtypes = [wintypes.HWND]
    user32.ShowWindow.restype = wintypes.BOOL
    user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
    user32.BringWindowToTop.restype = wintypes.BOOL
    user32.BringWindowToTop.argtypes = [wintypes.HWND]
    user32.IsWindowVisible.restype = wintypes.BOOL
    user32.IsWindowVisible.argtypes = [wintypes.HWND]
    user32.IsIconic.restype = wintypes.BOOL
    user32.IsIconic.argtypes = [wintypes.HWND]
    user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
    user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    user32.GetWindowThreadProcessId.restype = wintypes.DWORD
    user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.c_void_p]
    user32.AttachThreadInput.restype = wintypes.BOOL
    user32.AttachThreadInput.argtypes = [wintypes.DWORD, wintypes.DWORD, wintypes.BOOL]
    user32.PostMessageW.restype = wintypes.BOOL
    user32.PostMessageW.argtypes = [
        wintypes.HWND, ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p
    ]
    user32.SetActiveWindow.restype = wintypes.HWND
    user32.SetActiveWindow.argtypes = [wintypes.HWND]
    kernel32.GetConsoleWindow.restype = wintypes.HWND
    kernel32.GetConsoleWindow.argtypes = []
    kernel32.GetCurrentThreadId.restype = wintypes.DWORD


if sys.platform == "win32":
    _bind()

# Rebuilt at most this often. Get-StartApps costs the best part of a second, and
# the set of installed applications does not change during a conversation.
APP_LIST_TTL_S = 300.0

_app_cache: tuple[float, list["InstalledApp"]] = (0.0, [])


@dataclass(frozen=True)
class InstalledApp:
    name: str
    app_id: str


@dataclass(frozen=True)
class WindowInfo:
    hwnd: int
    title: str
    pid: int
    process: str


def available() -> bool:
    return sys.platform == "win32"


# ── looking at what is on screen ─────────────────────────────────
def visible_windows() -> list[WindowInfo]:
    """Every visible top-level window with a title, and who owns it."""
    if not available():
        return []

    import psutil

    user32 = ctypes.windll.user32
    callback = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
    found: list[WindowInfo] = []

    def visit(hwnd: int, _param: int) -> bool:
        if not user32.IsWindowVisible(hwnd):
            return True
        length = user32.GetWindowTextLengthW(hwnd)
        if not length:
            return True
        buffer = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buffer, length + 1)
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        try:
            process = psutil.Process(pid.value).name()
        except Exception:
            process = ""
        found.append(WindowInfo(hwnd, buffer.value, pid.value, process))
        return True

    user32.EnumWindows(callback(visit), 0)
    return found


def _is_foreground(hwnd: int) -> bool:
    current = ctypes.windll.user32.GetForegroundWindow()
    return bool(current) and int(current) == int(hwnd)


def console_window() -> int:
    """The console this process is attached to, if any. 0 when detached."""
    if not available():
        return 0
    return int(ctypes.windll.kernel32.GetConsoleWindow() or 0)


def raise_window(hwnd: int) -> bool:
    """Restore a minimised window and put it in front.

    Two further tricks were tried and are deliberately absent, because neither
    made any difference when measured: a simulated ALT keypress (the usual
    advice for satisfying the "user provided input" condition) and a
    minimise-then-restore cycle. Both left the foreground window unchanged.
    Code that does not demonstrably help is code that hides the real cause.

    What blocks it is not the technique. Windows only lets the process that
    already owns the foreground hand it to another, so a focus change works
    when the assistant is the terminal the user is typing in, and does not from
    a background process. Confirmed in use: raising a window fails from a
    detached test harness and works from a real terminal session. The result is therefore checked and reported rather
    than assumed — telling someone a window was raised when it was not is
    worse than saying it could not be.
    """
    if not available():
        return False

    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32
    handle = wintypes.HWND(hwnd)

    if user32.IsIconic(handle):
        user32.ShowWindow(handle, SW_RESTORE)
    else:
        user32.ShowWindow(handle, SW_SHOW)

    if user32.SetForegroundWindow(handle) and _is_foreground(hwnd):
        return True

    foreground = user32.GetForegroundWindow()
    other = user32.GetWindowThreadProcessId(foreground, None)
    mine = kernel32.GetCurrentThreadId()
    attached = bool(user32.AttachThreadInput(mine, other, True))
    try:
        user32.BringWindowToTop(handle)
        user32.SetForegroundWindow(handle)
        user32.SetActiveWindow(handle)
    finally:
        if attached:
            user32.AttachThreadInput(mine, other, False)

    return _is_foreground(hwnd)


def close_window(hwnd: int) -> None:
    """Ask a window to close, the same way its X button does.

    `PostMessage`, not `SendMessage`: SendMessage blocks until the application
    has finished closing, and an app showing a "save changes?" dialog would
    freeze the assistant until the user answered it.
    """
    if available():
        ctypes.windll.user32.PostMessageW(wintypes.HWND(hwnd), WM_CLOSE, None, None)


# ── what is installed ────────────────────────────────────────────
def installed_apps(force: bool = False) -> list[InstalledApp]:
    """Every launchable application, desktop and Store alike.

    `Get-StartApps` is the only enumeration that covers both. The alternative,
    walking Start Menu shortcuts, silently misses every Store app: on this
    machine it finds Visual Studio Code and not Spotify.
    """
    global _app_cache
    if not available():
        return []

    cached_at, cached = _app_cache
    if not force and cached and (time.time() - cached_at) < APP_LIST_TTL_S:
        return cached

    try:
        completed = subprocess.run(
            [
                "powershell.exe", "-NoProfile", "-NonInteractive", "-Command",
                "Get-StartApps | ForEach-Object { $_.Name + '|' + $_.AppID }",
            ],
            capture_output=True,
            text=True,
            timeout=25,
            # Without this the console flashes on screen every time an app is
            # opened, which is precisely what the daemon work was about.
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except Exception:
        return cached

    apps: list[InstalledApp] = []
    for line in completed.stdout.splitlines():
        name, sep, app_id = line.partition("|")
        if sep and name.strip() and app_id.strip():
            apps.append(InstalledApp(name.strip(), app_id.strip()))

    if apps:
        _app_cache = (time.time(), apps)
    return apps or cached


# Shorthands people actually type. Kept deliberately tiny: every alias is a
# guess about what someone meant, and a wrong guess opens the wrong program.
# These two earn their place because the full name shares no word with them.
_ALIASES = {
    "vscode": "visual studio code",
    "vs code": "visual studio code",
}


def normalise(query: str) -> str:
    cleaned = " ".join(query.lower().split())
    return _ALIASES.get(cleaned, cleaned)


def score_name(query: str, candidate: str, penalise_length: bool = True) -> int:
    """How well a candidate name answers `query`. 0 means no match.

    Ranked rather than first-hit: "visual studio" matches four entries on this
    machine, and picking whichever the shell happened to list first is how you
    end up opening the Visual Studio Installer.
    """
    query = normalise(query)
    candidate = " ".join(candidate.lower().split())
    if not query or not candidate:
        return 0

    # Window titles are long by nature — "Build AI assistant ... - Visual
    # Studio Code" — so the length penalty that usefully separates two app
    # names wipes out a perfectly good title match. Scored without it there.
    penalty = (lambda extra: min(extra, 30)) if penalise_length else (lambda _extra: 0)
    if query == candidate:
        return 100
    if candidate.startswith(query):
        # Shorter is closer: "Visual Studio Code" beats "Visual Studio Installer"
        # for "visual studio code", and both beat a long unrelated prefix match.
        return 80 - penalty(len(candidate) - len(query))
    if query in candidate:
        return 50 - penalty(len(candidate) - len(query))
    # Every word present, in any order: "code visual studio" still finds it.
    words = query.split()
    if len(words) > 1 and all(word in candidate for word in words):
        return 40
    return 0


def find_installed_scored(query: str) -> list[tuple[int, InstalledApp]]:
    """Installed apps matching `query`, best first, with their scores.

    Scores are exposed because near-ties matter: "visual studio" matches
    Visual Studio 2017, Visual Studio Code and Visual Studio Installer within
    two points of each other, and silently picking the winner of that is how
    the assistant opens the installer. The caller asks instead.
    """
    scored = [(score_name(query, app.name), app) for app in installed_apps()]
    hits = [(score, app) for score, app in scored if score > 0]
    hits.sort(key=lambda pair: (-pair[0], len(pair[1].name)))
    return hits


def find_installed(query: str) -> list[InstalledApp]:
    """Installed apps matching `query`, best first."""
    return [app for _score, app in find_installed_scored(query)]


def find_windows(query: str) -> list[WindowInfo]:
    """Open windows matching `query`, best first.

    Matched against the process name and the window title, never against a
    joined command line: a substring test over everything a process mentions is
    how a search for daemons once matched the shell doing the searching.
    """
    scored: list[tuple[int, WindowInfo]] = []
    for window in visible_windows():
        stem = os.path.splitext(window.process)[0]
        score = max(
            score_name(query, stem),
            # Titles carry the application name, usually after the document:
            # "… - local-ai-assistant - Visual Studio Code". Worth less than a
            # process-name match, which cannot be a coincidence of content.
            score_name(query, window.title, penalise_length=False) - 20,
        )
        if score > 0:
            scored.append((score, window))
    scored.sort(key=lambda pair: -pair[0])
    return [window for _score, window in scored]


def launch(app: InstalledApp) -> bool:
    """Start an application through the shell.

    Everything goes through `shell:AppsFolder`, Store app or not, because a
    Store app has no executable path to run. `explorer.exe` is the launcher, so
    the new process is a child of the shell rather than of the assistant — it
    outlives the conversation, as a program the user opened should.
    """
    if not available():
        return False
    try:
        subprocess.Popen(
            ["explorer.exe", f"shell:AppsFolder\\{app.app_id}"],
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return True
    except Exception:
        return False


def wait_for_window(query: str, timeout_s: float = 12.0) -> WindowInfo | None:
    """Poll for an application's window after launching it.

    A fixed sleep is wrong in both directions: too short and the window is not
    up yet, too long and every launch feels broken. Applications here take
    anywhere from under a second to about eight.
    """
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        matches = find_windows(query)
        if matches:
            return matches[0]
        time.sleep(0.4)
    return None
