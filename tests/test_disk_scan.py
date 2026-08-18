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
