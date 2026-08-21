"""Tests for running a disk scan from inside a conversation.

The scan is read-only, but it is expensive: minutes of file reading on a machine
with little memory to spare. So the guarantees worth testing are that it never
starts without a yes, that an absent user is not a yes, and that declining one
scan is not read as declining file analysis for ever.

Nothing here actually scans. Every test refuses at the prompt, or stops before
it, which is also the only way to keep the suite fast.
"""
from __future__ import annotations

import pytest

from app.tools.base import ToolContext, ToolResult
from app.tools.builtin import DiskReportTool
from app.tools.disk import RunDiskScanTool, perform_scan


def ctx(text: str, confirm=None, progress=None) -> ToolContext:
    return ToolContext(
        assistant=None, request_text=text, confirm=confirm, progress=progress
    )


@pytest.fixture
def granted(monkeypatch):
    """Folder consent already given, so the cost prompt is the only gate."""
    from app import consent

    monkeypatch.setattr(consent, "status", lambda: ("granted", None))
    monkeypatch.setattr(consent, "current_roots", lambda: [r"C:\Users\Om\Downloads"])
    return consent


# ── the tool is reachable at all ─────────────────────────────────
def test_the_scan_can_be_asked_for_in_words():
    """It was reachable only as `assistant scan`, so chat answered a question
    about duplicates by naming the command to type."""
    tool = RunDiskScanTool()
    for question in (
        "scan my files",
        "run a scan",
        "can you rescan my folders?",
        "do a fresh scan",
        "check my files for duplicates",
    ):
        assert tool.matches(question), question


def test_the_scan_is_read_only():
    """It opens files and writes none of them. The prompt it raises is about
    cost, not danger — mislabelling it destructive would train the user to wave
    through the prompts that do matter."""
    from app.tools.base import Risk

    assert RunDiskScanTool().risk is Risk.READ


# ── nothing starts without a yes ─────────────────────────────────
def test_an_absent_user_is_not_consent(granted):
    """The daemon wires no confirmer. A scan there must not start itself."""
    result = perform_scan(ctx("scan my files", confirm=None), "scan my files")

    assert not result.ok
    assert "declined" in result.display


def test_declining_says_so_exactly(granted):
    """A determined outcome is stated, never handed to a model to phrase: asked
    to explain a refusal, llama3.2:3b invents a reason and denies the
    capability exists."""
    result = perform_scan(
        ctx("scan my files", confirm=lambda _: False), "scan my files"
    )

    assert result.final_text
    assert "No scan run" in result.final_text


def test_the_prompt_says_what_it_will_read_and_what_it_costs(granted):
    asked: list[str] = []
    perform_scan(
        ctx("scan my files", confirm=lambda prompt: asked.append(prompt) or False),
        "scan my files",
    )

    assert len(asked) == 1
    prompt = asked[0].lower()
    assert "folder" in prompt
    assert "minutes" in prompt
    assert "nothing is modified" in prompt


def test_declining_one_scan_is_not_declining_file_analysis(monkeypatch):
    """"Not now" must never be recorded as "never". Silently widening a no is
    the mirror of silently widening a yes (rule 3)."""
    from app import consent

    monkeypatch.setattr(consent, "status", lambda: ("none", None))
    monkeypatch.setattr(consent, "current_roots", lambda: [r"C:\Users\Om\Downloads"])

    declined: list[bool] = []
    monkeypatch.setattr(consent, "decline", lambda: declined.append(True))
    monkeypatch.setattr(consent, "grant", lambda *a, **k: declined.append(False))

    perform_scan(ctx("scan my files", confirm=lambda _: False), "scan my files")
    assert declined == []


def test_a_previous_refusal_of_file_analysis_is_honoured(monkeypatch):
    from app import consent

    monkeypatch.setattr(consent, "status", lambda: ("declined", None))

    asked: list[str] = []
    result = perform_scan(
        ctx("scan my files", confirm=lambda p: asked.append(p) or True),
        "scan my files",
    )

    assert asked == []  # never prompted, because the answer was already no
    assert not result.ok
    assert "consent --grant" in (result.final_text or "")


# ── the report tool offers rather than deflects ──────────────────
def test_no_scan_on_record_offers_to_run_one(monkeypatch, granted):
    """It used to refuse and tell the user which command to type. That is the
    same failure as telling someone to run `systeminfo` themselves."""
    from app.core import diskintent

    monkeypatch.setattr(diskintent, "ground_prompt", lambda q: (None, "no scan"))

    asked: list[str] = []
    result = DiskReportTool().run(
        {}, ctx("do I have duplicate files?", confirm=lambda p: asked.append(p) or False)
    )

    assert len(asked) == 1  # it offered
    assert isinstance(result, ToolResult)
    assert "assistant scan" not in (result.final_text or "")


def test_stale_findings_are_answered_but_flagged(monkeypatch):
    """Rule 4: serve the degraded answer, and say that it is degraded."""
    import time

    from app.core import diskintent

    old = time.time() - 30 * 86_400
    monkeypatch.setattr(diskintent, "ground_prompt", lambda q: ("findings", None))
    monkeypatch.setattr(
        diskintent, "load_report", lambda: {"saved_at": old, "summary": "s"}
    )

    result = DiskReportTool().run({}, ctx("any duplicate files?"))

    assert result.ok
    assert "may be out of date" in result.content
    assert "rescan" in result.content


def test_fresh_findings_are_not_flagged(monkeypatch):
    import time

    from app.core import diskintent

    monkeypatch.setattr(diskintent, "ground_prompt", lambda q: ("findings", None))
    monkeypatch.setattr(
        diskintent, "load_report", lambda: {"saved_at": time.time(), "summary": "s"}
    )

    result = DiskReportTool().run({}, ctx("any duplicate files?"))

    assert result.ok
    assert "may be out of date" not in result.content


# ── progress ─────────────────────────────────────────────────────
def test_progress_is_optional_and_never_fatal():
    """A front end that cannot draw a progress line must not take the scan down
    with it (rule 10)."""
    silent = ctx("scan")
    silent.report_progress("walking")  # no sink wired: must be a no-op

    def explodes(_message: str) -> None:
        raise RuntimeError("terminal went away")

    ctx("scan", progress=explodes).report_progress("walking")


# ── the cleanup script, reachable from conversation ──────────────
def test_the_grounding_never_tells_the_user_to_type_a_command(monkeypatch):
    """It used to end every disk answer with "run `assistant scan
    --cleanup-script cleanup.ps1`" — the same deflection as telling someone to
    run `systeminfo` themselves, only better disguised."""
    from app.core import diskintent

    grounded, _note = diskintent.DISK_GROUNDING.format(
        age="just now", results="x", question="delete my duplicates"
    ), None

    assert "assistant scan" not in grounded
    assert "--cleanup-script" not in grounded
    assert "never name a command for them to type" in grounded.lower()
    assert "you cannot delete files" in grounded.lower()


def test_a_cleanup_request_writes_a_reviewable_script(monkeypatch, tmp_path):
    """Rule 2: destructive intent produces an artifact the user runs, never an
    action the assistant takes."""
    from app.core import diskintent
    from app.tools import disk as disk_tools

    target = tmp_path / "cleanup.ps1"
    monkeypatch.setattr(disk_tools, "CLEANUP_SCRIPT_PATH", target)
    monkeypatch.setattr(
        diskintent,
        "load_report",
        lambda: {
            "saved_at": 0.0,
            "cleanup_script": "# review me\nAdd-Type -AssemblyName Microsoft.VisualBasic",
            "cleanup_count": 3,
        },
    )

    result = disk_tools.PrepareCleanupTool().run({}, ctx("delete my duplicate files"))

    assert result.ok
    assert target.exists()
    assert "3 redundant copies" in (result.final_text or "")
    assert "Nothing has been deleted" in (result.final_text or "")


def test_the_script_is_written_where_powershell_can_read_it(monkeypatch, tmp_path):
    """PowerShell 5.1 decodes a BOM-less .ps1 as ANSI and mangles non-ASCII
    paths. A delete script pointed at a mangled path is the one mistake that
    must not be possible."""
    from app.core import diskintent
    from app.tools import disk as disk_tools

    target = tmp_path / "cleanup.ps1"
    monkeypatch.setattr(disk_tools, "CLEANUP_SCRIPT_PATH", target)
    monkeypatch.setattr(
        diskintent,
        "load_report",
        lambda: {"saved_at": 0.0, "cleanup_script": "# café", "cleanup_count": 1},
    )

    disk_tools.PrepareCleanupTool().run({}, ctx("clean up duplicates"))

    assert target.read_bytes().startswith(b"\xef\xbb\xbf")  # utf-8 BOM


def test_nothing_to_clean_says_so_rather_than_writing_an_empty_script(monkeypatch):
    from app.core import diskintent
    from app.tools import disk as disk_tools

    monkeypatch.setattr(
        diskintent,
        "load_report",
        lambda: {"saved_at": 0.0, "cleanup_script": "# none", "cleanup_count": 0},
    )

    result = disk_tools.PrepareCleanupTool().run({}, ctx("clean up duplicates"))

    assert result.ok
    assert "nothing for a cleanup script to do" in (result.final_text or "")


def test_no_scan_yet_offers_one_instead_of_refusing(monkeypatch, granted):
    from app.core import diskintent
    from app.tools import disk as disk_tools

    monkeypatch.setattr(diskintent, "load_report", lambda: None)

    asked: list[str] = []
    disk_tools.PrepareCleanupTool().run(
        {}, ctx("clean up duplicates", confirm=lambda p: asked.append(p) or False)
    )

    assert len(asked) == 1  # it offered to scan


def test_the_cleanup_tool_writes_a_file_but_deletes_nothing():
    import inspect

    from app.tools.disk import PrepareCleanupTool

    source = inspect.getsource(PrepareCleanupTool)
    assert "unlink" not in source
    assert "rmtree" not in source
    assert "remove(" not in source
