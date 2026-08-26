"""Tests for opening, raising and closing applications.

The dangerous parts here are not launching things. They are: closing the wrong
window, and claiming a window was brought to the front when it was not. Both are
covered through the real dispatch path where it matters, because testing a tool
directly proves nothing about the guard in front of it (rule 12).
"""
from __future__ import annotations

import sys

import pytest

from app import windows as win
from app.tools.apps import CloseAppTool, OpenAppTool, app_name_from
from app.tools.base import Risk, ToolContext

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Windows only")


def ctx(text: str, confirm=None) -> ToolContext:
    return ToolContext(assistant=None, request_text=text, confirm=confirm)


# ── reading the request ──────────────────────────────────────────
@pytest.mark.parametrize(
    "said,wanted",
    [
        ("open spotify", "spotify"),
        ("launch google chrome", "google chrome"),
        ("bring visual studio up on the main screen", "visual studio"),
        ("pull up vs code", "vs code"),
        ("close spotify please", "spotify"),
        ("switch to chrome", "chrome"),
        ("start notepad for me", "notepad"),
    ],
)
def test_the_app_name_survives_the_sentence(said, wanted):
    """The backstop calls tools with no arguments at all, and a 3B model omits
    them even when it does emit a call. A tool that only works when the model
    fills in a field is a tool that mostly does not work."""
    assert app_name_from(said) == wanted


def test_a_bare_verb_names_nothing():
    assert app_name_from("open") == ""
    assert app_name_from("close the window please") == ""


# ── matching installed apps ──────────────────────────────────────
@pytest.mark.machine
def test_store_apps_are_found_too():
    """Walking Start Menu shortcuts finds Visual Studio Code and misses Spotify
    entirely, because Store apps have no shortcut file."""
    names = [app.name.lower() for app in win.installed_apps()]
    assert names, "no installed applications enumerated"
    assert any("spotify" in name for name in names)
    assert any("visual studio code" in name for name in names)


def test_an_exact_name_beats_a_longer_one():
    assert win.score_name("notepad", "Notepad") > win.score_name("notepad", "Notepad++")
    assert win.score_name("visual studio code", "Visual Studio Code") == 100


@pytest.mark.machine
def test_shorthands_people_actually_type():
    assert [a.name for a in win.find_installed("vs code")[:1]] == ["Visual Studio Code"]
    assert [a.name for a in win.find_installed("vscode")[:1]] == ["Visual Studio Code"]


def test_a_window_title_still_matches_despite_its_length():
    """"Build AI assistant ... - Visual Studio Code" is a long string with a
    short app name at the end. Penalising length there scored a perfect match
    at zero, and "bring visual studio up" found nothing while it was open."""
    long_title = "Build AI assistant with x - local-ai-assistant - Visual Studio Code"
    assert win.score_name("visual studio", long_title, penalise_length=False) > 0


def test_nonsense_matches_nothing():
    assert win.find_installed("frobnicator3000") == []


# ── never close the wrong thing ──────────────────────────────────
def test_the_terminal_running_this_conversation_is_never_closed(monkeypatch):
    """Not hypothetical: "close terminal" would otherwise end the conversation
    that asked for it."""
    fake = win.WindowInfo(hwnd=1234, title="Command Prompt", pid=99, process="x.exe")
    monkeypatch.setattr(win, "find_windows", lambda _q: [fake])
    monkeypatch.setattr("app.tools.apps._own_pids", lambda: {99})

    refusal = CloseAppTool().precheck({"name": "terminal"}, ctx("close terminal"))

    assert refusal is not None
    assert not refusal.ok
    assert "running in" in (refusal.final_text or "")


def test_protected_system_processes_are_never_closed(monkeypatch):
    fake = win.WindowInfo(
        hwnd=1, title="Program Manager", pid=5, process="csrss.exe"
    )
    monkeypatch.setattr(win, "find_windows", lambda _q: [fake])
    monkeypatch.setattr("app.tools.apps._own_pids", lambda: set())

    refusal = CloseAppTool().precheck({"name": "csrss"}, ctx("close csrss"))

    assert refusal is not None
    assert "part of Windows" in (refusal.final_text or "")


def test_refusal_happens_before_any_prompt(monkeypatch, tmp_path):
    """The `end_process` lesson: prompting and *then* refusing trains people to
    wave prompts through, and the prompt itself was a lie."""
    from app.tools import ToolInvocation, ToolLedger, ToolRegistry

    fake = win.WindowInfo(hwnd=1, title="x", pid=5, process="csrss.exe")
    monkeypatch.setattr(win, "find_windows", lambda _q: [fake])
    monkeypatch.setattr("app.tools.apps._own_pids", lambda: set())

    asked: list[str] = []
    registry = ToolRegistry(ledger=ToolLedger(tmp_path / "ledger.sqlite3"))
    registry.register(CloseAppTool())
    result = registry.invoke(
        ToolInvocation(tool="close_app", arguments={"name": "csrss"}, source="model"),
        ctx("close csrss", confirm=lambda prompt: asked.append(prompt) or True),
    )

    assert asked == []
    assert not result.ok


# ── closing is not killing ───────────────────────────────────────
def test_closing_is_a_request_not_a_termination():
    """TerminateProcess takes unsaved work with it. WM_CLOSE is what the X
    button does: the application gets to prompt (rule 2)."""
    import inspect

    source = inspect.getsource(CloseAppTool.run)
    assert "close_window" in source
    assert "terminate" not in source.lower()
    assert "kill" not in source.lower()


def test_the_risk_levels_say_what_each_one_does():
    """Force-quitting stays a louder, separate thing."""
    from app.tools.system import EndProcessTool

    assert CloseAppTool().risk is Risk.WRITE
    assert OpenAppTool().risk is Risk.WRITE
    assert EndProcessTool().risk is Risk.DESTRUCTIVE


# ── honest reporting ─────────────────────────────────────────────
def test_a_failed_raise_is_not_reported_as_success(monkeypatch):
    """Windows only lets the foreground process hand over focus, so this can
    genuinely fail. Saying a window was raised when it was not is worse than
    saying it could not be."""
    fake = win.WindowInfo(hwnd=7, title="phase1.txt - Notepad", pid=8, process="notepad.exe")
    monkeypatch.setattr(win, "find_windows", lambda _q: [fake])
    monkeypatch.setattr(win, "raise_window", lambda _h: False)

    result = OpenAppTool().run({"name": "notepad"}, ctx("open notepad"))

    assert "would not let me" in (result.final_text or "")
    assert result.meta["action"] == "flashed"


def test_a_successful_raise_says_it_was_already_open(monkeypatch):
    fake = win.WindowInfo(hwnd=7, title="phase1.txt - Notepad", pid=8, process="notepad.exe")
    monkeypatch.setattr(win, "find_windows", lambda _q: [fake])
    monkeypatch.setattr(win, "raise_window", lambda _h: True)

    result = OpenAppTool().run({"name": "notepad"}, ctx("open notepad"))

    assert result.final_text == "Notepad was already open — brought it to the front."


def test_an_ambiguous_name_asks_rather_than_guessing(monkeypatch):
    """"visual studio" scores Visual Studio 2017 and Visual Studio Code
    identically on this machine. Guessing opens the wrong program."""
    monkeypatch.setattr(win, "find_windows", lambda _q: [])
    monkeypatch.setattr(
        win,
        "find_installed_scored",
        lambda _q: [
            (75, win.InstalledApp("Visual Studio 2017", "a")),
            (75, win.InstalledApp("Visual Studio Code", "b")),
        ],
    )
    launched: list[str] = []
    monkeypatch.setattr(win, "launch", lambda app: launched.append(app.name) or True)

    result = OpenAppTool().run({"name": "visual studio"}, ctx("open visual studio"))

    assert launched == []  # nothing was opened
    assert "Did you mean" in (result.final_text or "")


def test_an_unknown_app_says_so_without_inventing(monkeypatch):
    monkeypatch.setattr(win, "find_windows", lambda _q: [])
    monkeypatch.setattr(win, "find_installed_scored", lambda _q: [])

    result = OpenAppTool().run({"name": "frobnicator3000"}, ctx("open frobnicator3000"))

    assert not result.ok
    assert "could not find" in (result.final_text or "")


def test_outcomes_are_stated_not_generated(monkeypatch):
    """Every outcome here is already known exactly, so none of them is handed
    to a 3B model to phrase (rule 13)."""
    monkeypatch.setattr(win, "find_windows", lambda _q: [])
    monkeypatch.setattr(win, "find_installed_scored", lambda _q: [])

    for tool, text in ((OpenAppTool(), "open nothing"), (CloseAppTool(), "close nothing")):
        result = tool.run({"name": "nothing-at-all"}, ctx(text))
        assert result.final_text, tool.name
