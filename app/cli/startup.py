"""Bringing the whole assistant up with one command.

`wake` exists because the parts that make this thing useful — the model server,
the background daemon — are separate processes a person should not have to
remember. Everything here is written to be run repeatedly and safely: each step
checks whether the work is already done before doing it, so waking an
already-awake assistant is a no-op rather than a second copy of everything.

Nothing here fails hard. A machine with no Ollama, no daemon and no microphone
should still end up at a working prompt with an honest account of what is
missing (rule 10).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from config import settings

# Where the daemon records its process id, so waking twice doesn't start a
# second one. Checked for liveness *and* identity — a stale file whose pid has
# been recycled must not be mistaken for a running daemon.
DAEMON_PID_FILE = Path(settings.upload_dir).parent / "daemon.pid"

OLLAMA_START_TIMEOUT = 60.0
OLLAMA_POLL = 0.5


@dataclass
class StepResult:
    ok: bool
    message: str
    skipped: bool = False


@dataclass
class WakeReport:
    steps: list[StepResult] = field(default_factory=list)

    def add(self, result: StepResult) -> StepResult:
        self.steps.append(result)
        return result

    @property
    def failures(self) -> list[StepResult]:
        return [s for s in self.steps if not s.ok]


# ── the model server ─────────────────────────────────────────────
def ollama_running(host: str | None = None) -> bool:
    base = (host or settings.ollama_host).rstrip("/")
    try:
        response = httpx.get(base + "/api/tags", timeout=2.0)
        return response.status_code == 200
    except httpx.HTTPError:
        return False


def find_ollama() -> str | None:
    """Locate the ollama binary; PATH first, then the usual install locations."""
    found = shutil.which("ollama")
    if found:
        return found

    candidates = [
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe",
        Path(r"C:\Program Files\Ollama\ollama.exe"),
        Path("/usr/local/bin/ollama"),
        Path("/usr/bin/ollama"),
    ]
    for path in candidates:
        try:
            if path.is_file():
                return str(path)
        except OSError:
            continue
    return None


def _detached_kwargs() -> dict:
    """Spawn flags for a child that must outlive its launcher, invisibly.

    `DETACHED_PROCESS` alone. Windows documents CREATE_NO_WINDOW,
    DETACHED_PROCESS and CREATE_NEW_CONSOLE as **mutually exclusive**, and the
    two wrong answers both showed up here:

    * OR-ing CREATE_NO_WINDOW with DETACHED_PROCESS is a contradictory request,
      not "extra hidden" — the daemon opened a visible console of its own and
      stray consoles flashed on screen.
    * CREATE_NO_WINDOW alone hides the window but leaves the child attached to
      the launcher's console, so the daemon died the moment the command that
      started it returned.

    DETACHED_PROCESS gives the child no console at all: nothing to show, and
    nothing to be killed with. Its output is handed a real file instead, since
    a process with no console still needs somewhere for stdout to go.
    """
    if sys.platform == "win32":
        return {"creationflags": getattr(subprocess, "DETACHED_PROCESS", 0)}
    return {"start_new_session": True}


# The daemon's own output. Sent to a file rather than DEVNULL: a background
# process whose errors are discarded is undiagnosable, and this cost real time —
# the daemon was crashing instantly on startup and simply appeared not to run.
DAEMON_LOG = Path(settings.upload_dir).parent / "daemon.log"


def _background_python() -> str:
    r"""The interpreter to launch detached children with: pythonw.exe if present.

    DETACHED_PROCESS alone is not enough, and the thing it misses is a
    *grandchild*. A detached process has no console at all, so when it starts a
    console application with default flags Windows allocates a **fresh** one —
    which on Windows 11 means a new Windows Terminal window. The virtualenv's
    python.exe is exactly such a starter: it is a redirector that re-launches
    the base interpreter. So `wake` opened a window titled
    `...\.venv\Scripts\python.exe` that nobody asked for, and the flag meant to
    hide the daemon was what created it.

    pythonw.exe is a GUI-subsystem binary: it never allocates a console, and
    neither does the base pythonw.exe its own redirector goes on to start. The
    whole chain stays invisible however many links it has.

    pythonw was tried once before and rejected for a reason that no longer
    holds: having no stdout of its own, the scheduler's first status line killed
    the daemon on launch. That was fatal only because output went to DEVNULL.
    Handed a real file (DAEMON_LOG) its stdout is valid and it runs normally —
    which is a second reason not to discard a background process's output.

    CREATE_NO_WINDOW on the ordinary interpreter also hides the window and was
    measured to work, but it is not used: it keeps the child inside console-land
    where survival depends on which console it ended up attached to, and it was
    already seen dying with its launcher once.
    """
    candidate = Path(sys.executable).with_name("pythonw.exe")
    return str(candidate) if candidate.is_file() else sys.executable


def ensure_ollama(timeout: float = OLLAMA_START_TIMEOUT) -> StepResult:
    if ollama_running():
        return StepResult(True, "model server already running", skipped=True)

    binary = find_ollama()
    if binary is None:
        return StepResult(
            False,
            "Ollama is not installed or not on PATH — install it from ollama.com",
        )

    try:
        subprocess.Popen(
            [binary, "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            **_detached_kwargs(),
        )
    except Exception as exc:
        return StepResult(False, "could not start Ollama: " + str(exc))

    # Poll rather than sleeping a fixed amount: it is usually ready in about a
    # second, and an arbitrary wait would make every wake feel slow.
    deadline = time.time() + timeout
    while time.time() < deadline:
        if ollama_running():
            return StepResult(True, "model server started")
        time.sleep(OLLAMA_POLL)
    return StepResult(False, "Ollama did not become ready in time")


def warm_model(assistant, timeout_s: float = 180.0) -> StepResult:
    """Force the model into memory now, rather than on the first real question.

    A cold load costs ~20s on this hardware. Paying it during a startup banner
    the user is already reading is far better than pausing after they have typed
    a question and are watching a blank line.
    """
    from app.engines import ChatMessage, GenerationOptions

    started = time.time()
    try:
        for event in assistant.engine.chat_stream(
            [ChatMessage(role="user", content="Reply with the single word: ready")],
            GenerationOptions(max_tokens=3, temperature=0.0),
        ):
            if event.done:
                break
            if time.time() - started > timeout_s:
                return StepResult(False, "model took too long to warm up")
    except Exception as exc:
        return StepResult(False, "model could not be loaded: " + str(exc))

    elapsed = time.time() - started
    return StepResult(True, f"{settings.default_model} warm ({elapsed:.0f}s)")


# ── the background daemon ────────────────────────────────────────
def _entry_script() -> Path:
    return Path(__file__).resolve().parents[2] / "assistant.py"


def is_daemon_cmdline(cmdline: list[str] | None, name: str = "") -> bool:
    """Is this argv *actually* our daemon?

    Matched against the argument list, never a joined string. A substring test
    over the whole command line ("assistant.py" and "daemon" both appear
    somewhere) matches any process that merely *mentions* those words — during
    development it matched the shell command being used to search for daemons,
    and terminated it. A process-killer that fires on a coincidental mention is
    exactly the asymmetric mistake rule 6 is about.

    So all three must hold: a Python interpreter, running this project's
    assistant.py, with `daemon` as a real argument.
    """
    if not cmdline:
        return False
    if name and Path(name).stem.lower() not in {"python", "pythonw"}:
        return False

    entry = str(_entry_script()).lower()
    args = [str(part).lower() for part in cmdline]

    interpreter = Path(args[0]).stem if args else ""
    if interpreter not in {"python", "pythonw"}:
        return False
    if not any(Path(arg).name == "assistant.py" and arg == entry for arg in args[1:]):
        return False
    return "daemon" in args[1:]


def find_daemons() -> list[int]:
    """Every running daemon, found by inspecting argument lists.

    The pid file alone is not enough. It records one pid, so a daemon that was
    started and then lost track of — the file overwritten, or the process
    surviving a crash — becomes invisible: `ensure_daemon` starts another and
    `stop_daemon` leaves the old one running. Two accumulated that way during
    development.
    """
    try:
        import psutil
    except ImportError:
        return []

    mine = os.getpid()
    matched: dict[int, int | None] = {}
    for process in psutil.process_iter(["pid", "name", "cmdline", "ppid"]):
        try:
            if process.info["pid"] == mine:
                continue
            if is_daemon_cmdline(process.info["cmdline"], process.info.get("name") or ""):
                matched[process.info["pid"]] = process.info.get("ppid")
        except Exception:
            continue

    # One launch produces two matching processes: the virtualenv's python.exe is
    # a redirector that re-runs the base interpreter with identical argv. Both
    # look like the daemon, and treating them as two led to the "duplicate"
    # being terminated — which was the actual worker, so the daemon appeared to
    # start and vanish. Keep only tree roots, so a launcher and its child count
    # once.
    return [pid for pid, parent in matched.items() if parent not in matched]


def daemon_pid() -> int | None:
    """A running daemon's pid, or None. Verifies identity, not just liveness."""
    running = find_daemons()
    if not running:
        return None

    # Prefer the recorded one so the reported pid stays stable across calls.
    try:
        recorded = int(DAEMON_PID_FILE.read_text(encoding="utf-8").strip())
        if recorded in running:
            return recorded
    except (OSError, ValueError):
        pass
    return running[0]


def ensure_daemon() -> StepResult:
    existing = daemon_pid()
    if existing is not None:
        return StepResult(
            True, f"background daemon already running (pid {existing})", skipped=True
        )

    entry = _entry_script()
    try:
        DAEMON_LOG.parent.mkdir(parents=True, exist_ok=True)
        log = DAEMON_LOG.open("a", encoding="utf-8", errors="replace")
    except OSError:
        log = None

    try:
        process = subprocess.Popen(
            [_background_python(), str(entry), "daemon", "run"],
            stdout=log or subprocess.DEVNULL,
            stderr=subprocess.STDOUT if log else subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            cwd=str(entry.parent),
            **_detached_kwargs(),
        )
    except Exception as exc:
        return StepResult(False, "could not start the daemon: " + str(exc))
    finally:
        if log is not None:
            log.close()  # the child keeps its own inherited handle

    try:
        DAEMON_PID_FILE.parent.mkdir(parents=True, exist_ok=True)
        DAEMON_PID_FILE.write_text(str(process.pid), encoding="utf-8")
    except OSError:
        pass  # the daemon runs regardless; we just cannot dedupe next time

    # Deliberately no "kill the duplicates" step here. One was written and
    # removed: it terminated a shell whose command line merely mentioned the
    # word daemon, and then the daemon itself. Killing processes on a heuristic
    # match is not worth the tidiness it buys (rule 6). `ensure_daemon` checks
    # before starting, and `sleep` stops everything it finds.
    return StepResult(True, f"background daemon started (pid {process.pid})")


def stop_daemon() -> StepResult:
    """Stop every daemon, not just the recorded one — strays must not survive."""
    running = find_daemons()
    if not running:
        DAEMON_PID_FILE.unlink(missing_ok=True)
        return StepResult(True, "no daemon running", skipped=True)

    import psutil

    stopped: list[int] = []
    failed: list[str] = []
    for pid in running:
        try:
            process = psutil.Process(pid)
            process.terminate()
            process.wait(timeout=10)
            stopped.append(pid)
        except psutil.NoSuchProcess:
            stopped.append(pid)
        except Exception as exc:
            failed.append(f"{pid} ({exc})")

    DAEMON_PID_FILE.unlink(missing_ok=True)
    if failed:
        return StepResult(False, "could not stop: " + ", ".join(failed))

    listed = ", ".join(str(pid) for pid in stopped)
    plural = "s" if len(stopped) > 1 else ""
    return StepResult(True, f"daemon{plural} stopped (pid {listed})")


# ── consent ──────────────────────────────────────────────────────
def consent_state() -> StepResult:
    """Report, never prompt. Waking up is not the moment to interrogate someone
    about folder permissions; the first command that needs them will ask."""
    from app import consent

    state, _record = consent.status()
    messages = {
        "granted": (True, "file access approved", False),
        "declined": (True, "file access declined — document and disk tools are off", True),
        "stale": (True, "folder list changed — will re-ask on first use", True),
        "none": (True, "file access not yet approved — will ask on first use", True),
    }
    ok, message, skipped = messages.get(state, (True, "consent state unknown", True))
    return StepResult(ok, message, skipped=skipped)
