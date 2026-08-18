"""Tests for `wake`, and for how background processes are identified.

The identification tests matter most. `stop_daemon` terminates whatever
`find_daemons` reports, so a loose match is a process-killer pointed at the
wrong target — during development a substring match killed the shell that was
being used to look for daemons, and then killed the daemon itself.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from app.cli.startup import (
    _background_python,
    _detached_kwargs,
    _entry_script,
    consent_state,
    find_ollama,
    is_daemon_cmdline,
    ollama_running,
)

ENTRY = str(_entry_script())
PY = r"C:\Projects\local-ai-assistant\.venv\Scripts\python.exe"
PYW = r"C:\Projects\local-ai-assistant\.venv\Scripts\pythonw.exe"


# ── identifying our own daemon ───────────────────────────────────
def test_the_real_daemon_is_recognised():
    assert is_daemon_cmdline([PY, ENTRY, "daemon", "run"], "python.exe")


def test_a_daemon_started_with_pythonw_is_still_recognised():
    """Background children run under pythonw, so `sleep` must know that shape
    too — otherwise it reports no daemon while one is running."""
    assert is_daemon_cmdline([PYW, ENTRY, "daemon", "run"], "pythonw.exe")


def test_other_assistant_commands_are_not_daemons():
    """`sleep` must not identify itself as the thing it is about to kill."""
    assert not is_daemon_cmdline([PY, ENTRY, "sleep", "buddy"], "python.exe")
    assert not is_daemon_cmdline([PY, ENTRY, "wake"], "python.exe")
    assert not is_daemon_cmdline([PY, ENTRY, "chat"], "python.exe")


@pytest.mark.parametrize(
    "cmdline,name",
    [
        (["powershell.exe", "-Command", "ps | grep assistant.py daemon"], "powershell.exe"),
        (["cmd.exe", "/c", "echo assistant.py daemon"], "cmd.exe"),
        (["notepad.exe", "assistant.py"], "notepad.exe"),
    ],
)
def test_processes_merely_mentioning_the_words_are_not_daemons(cmdline, name):
    """The bug that killed a live shell: matching a joined command line meant
    any process whose arguments happened to contain both words was a target."""
    assert not is_daemon_cmdline(cmdline, name)


def test_a_different_projects_daemon_is_not_ours():
    """Two checkouts on one machine must not stop each other."""
    assert not is_daemon_cmdline(
        [PY, r"C:\Other\checkout\assistant.py", "daemon", "run"], "python.exe"
    )


def test_empty_or_missing_cmdline_is_safe():
    assert not is_daemon_cmdline(None)
    assert not is_daemon_cmdline([])
    assert not is_daemon_cmdline([PY])


# ── spawn flags ──────────────────────────────────────────────────
@pytest.mark.skipif(sys.platform != "win32", reason="Windows process flags")
def test_background_children_are_detached_and_windowless():
    """Windows treats these flags as mutually exclusive. Both wrong answers
    happened: OR-ing them opened a visible console, and CREATE_NO_WINDOW alone
    left the daemon attached to its launcher so it died on return."""
    flags = _detached_kwargs()["creationflags"]

    assert flags == subprocess.DETACHED_PROCESS
    assert not flags & subprocess.CREATE_NO_WINDOW
    assert not flags & subprocess.CREATE_NEW_CONSOLE


@pytest.mark.skipif(sys.platform != "win32", reason="Windows console model")
def test_background_interpreter_cannot_allocate_a_console():
    r"""The window `wake` opened was never the daemon's own console.

    DETACHED_PROCESS leaves a process with no console, so the console app it
    starts next gets a brand-new one — and the venv's python.exe always starts
    one, being a redirector for the base interpreter. The visible window was
    that grandchild's, titled `...\.venv\Scripts\python.exe`. Only a
    GUI-subsystem interpreter keeps the whole chain quiet.
    """
    chosen = Path(_background_python())
    assert chosen.is_file()
    assert chosen.stem.lower() == "pythonw"


def test_daemon_output_goes_to_a_file_not_devnull():
    """A background process whose errors are discarded is undiagnosable: the
    daemon was crashing on startup and simply appeared not to run."""
    from app.cli.startup import DAEMON_LOG

    assert Path(DAEMON_LOG).is_absolute()
    assert DAEMON_LOG.name.endswith(".log")


# ── steps report rather than prompt ──────────────────────────────
def test_consent_step_never_blocks():
    """Waking up is not the moment to interrogate someone about permissions;
    the first command that needs them asks."""
    result = consent_state()
    assert result.ok  # never fails the wake, whatever the consent state
    assert result.message


def test_ollama_probe_is_safe_when_absent():
    """A machine with no model server must still reach a usable prompt."""
    assert ollama_running("http://127.0.0.1:9") is False


def test_ollama_lookup_returns_a_path_or_none():
    found = find_ollama()
    assert found is None or Path(found).exists()


# ---- what the user sees ----------------------------------------
def test_the_reply_label_is_the_assistant_s_name():
    """Being greeted by "Buddy" and then answered by "assistant" reads as two
    different programs."""
    from app.cli.repl import BOT_LABEL
    from config import settings

    assert BOT_LABEL == settings.assistant_name.lower()
    assert BOT_LABEL != "assistant"


def test_wake_does_not_resume_by_default():
    """The previous conversation is opt-in. Left on, it leaked old answers into
    new questions."""
    import inspect

    from app.cli.main import wake

    default = inspect.signature(wake).parameters["resume"].default
    assert getattr(default, "default", default) is False
