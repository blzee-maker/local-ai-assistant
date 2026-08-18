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
    """Spawn flags that let a child outlive this process without a console."""
    if sys.platform == "win32":
        flags = 0
        for name in ("CREATE_NO_WINDOW", "DETACHED_PROCESS"):
            flags |= getattr(subprocess, name, 0)
        return {"creationflags": flags}
    return {"start_new_session": True}


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
def daemon_pid() -> int | None:
    """The running daemon's pid, or None. Verifies identity, not just liveness."""
    try:
        pid = int(DAEMON_PID_FILE.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None

    try:
        import psutil

        process = psutil.Process(pid)
        with process.oneshot():
            # A recycled pid could be anything; require it to look like ours.
            command = " ".join(process.cmdline()).lower()
        if "assistant" in command and "daemon" in command:
            return pid
    except Exception:
        return None
    return None


def ensure_daemon() -> StepResult:
    existing = daemon_pid()
    if existing is not None:
        return StepResult(
            True, f"background daemon already running (pid {existing})", skipped=True
        )

    entry = Path(__file__).resolve().parents[2] / "assistant.py"
    try:
        process = subprocess.Popen(
            [sys.executable, str(entry), "daemon", "run"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            cwd=str(entry.parent),
            **_detached_kwargs(),
        )
    except Exception as exc:
        return StepResult(False, "could not start the daemon: " + str(exc))

    try:
        DAEMON_PID_FILE.parent.mkdir(parents=True, exist_ok=True)
        DAEMON_PID_FILE.write_text(str(process.pid), encoding="utf-8")
    except OSError:
        pass  # the daemon runs regardless; we just cannot dedupe next time

    return StepResult(True, f"background daemon started (pid {process.pid})")


def stop_daemon() -> StepResult:
    pid = daemon_pid()
    if pid is None:
        return StepResult(True, "no daemon running", skipped=True)
    try:
        import psutil

        process = psutil.Process(pid)
        process.terminate()
        process.wait(timeout=10)
    except Exception as exc:
        return StepResult(False, "could not stop the daemon: " + str(exc))

    DAEMON_PID_FILE.unlink(missing_ok=True)
    return StepResult(True, f"daemon stopped (pid {pid})")


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
