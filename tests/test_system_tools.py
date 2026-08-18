"""Tests for system awareness and process control.

`end_process` is the first thing this assistant can do that a user cannot undo,
so most of this file is about what it *refuses*. Each refusal below corresponds
to a way the naive version would have destroyed something.
"""
from __future__ import annotations

import os
import time

import pytest

from app.tools.base import Risk, ToolContext
from app.tools.system import (
    EndProcessTool,
    ProcessInfo,
    SystemStatusTool,
    TopProcessesTool,
    describe_snapshot,
    is_idle_process,
    is_protected,
    sample_processes,
    system_snapshot,
)


def ctx(text: str = "", confirm=None) -> ToolContext:
    return ToolContext(assistant=None, request_text=text, confirm=confirm)


# ── snapshot ─────────────────────────────────────────────────────
def test_snapshot_has_the_headline_numbers():
    snapshot = system_snapshot()
    assert snapshot["cpu_count"] >= 1
    assert 0 <= snapshot["memory"]["percent_used"] <= 100
    assert snapshot["memory"]["total_gb"] > 0
    assert snapshot["uptime_hours"] >= 0


def test_description_flags_memory_pressure():
    """A 3B model will not infer that 1.1GB free is a problem, and that is the
    actual question behind 'why is my laptop slow?'."""
    snapshot = {
        "cpu_percent": 5, "cpu_count": 8,
        "memory": {"total_gb": 7.8, "available_gb": 0.4, "percent_used": 95},
        "swap": {"total_gb": 0, "percent_used": 0},
        "battery": None, "disks": [], "uptime_hours": 2.0,
    }
    assert "memory is nearly exhausted" in describe_snapshot(snapshot).lower()


def test_description_stays_quiet_when_healthy():
    snapshot = {
        "cpu_percent": 5, "cpu_count": 8,
        "memory": {"total_gb": 16.0, "available_gb": 11.0, "percent_used": 31},
        "swap": {"total_gb": 0, "percent_used": 0},
        "battery": None, "disks": [], "uptime_hours": 2.0,
    }
    text = describe_snapshot(snapshot).lower()
    assert "nearly exhausted" not in text
    assert "saturated" not in text


# ── the idle-process trap ────────────────────────────────────────
def test_idle_process_is_recognised():
    assert is_idle_process(0, "System Idle Process")
    assert is_idle_process(1234, "system idle process")
    assert not is_idle_process(1234, "chrome.exe")


def test_process_list_excludes_the_idle_process():
    """Windows charges unused CPU to a pseudo-process, so it tops any CPU
    ranking on an idle machine. Reporting it as the biggest consumer states the
    exact opposite of the truth — that figure is how much CPU is free."""
    names = {p.name.lower() for p in sample_processes(interval=0.0)}
    assert "system idle process" not in names


def test_processes_are_ranked_and_capped():
    procs = sample_processes(interval=0.0, limit=5)
    assert len(procs) <= 5
    scores = [(p.cpu_percent, p.memory_mb) for p in procs]
    assert scores == sorted(scores, reverse=True)


# ── read tools ───────────────────────────────────────────────────
def test_status_tool_matches_resource_questions():
    tool = SystemStatusTool()
    assert tool.matches("why is my laptop so slow?")
    assert tool.matches("how much RAM am I using")
    assert not tool.matches("write me a poem about rain")


def test_top_processes_matches_the_obvious_phrasings():
    tool = TopProcessesTool()
    assert tool.matches("what's using my CPU?")
    assert tool.matches("which process is hogging memory")
    assert not tool.matches("summarise this document")


def test_read_tools_need_no_consent():
    assert SystemStatusTool().risk is Risk.READ
    assert TopProcessesTool().risk is Risk.READ
    assert not SystemStatusTool().risk.needs_consent


def test_status_tool_returns_grounding_text():
    result = SystemStatusTool().run({}, ctx("why is it slow?"))
    assert result.ok
    assert "Memory:" in result.content
    assert "why is it slow?" in result.content


# ── end_process: what it refuses ─────────────────────────────────
def test_end_process_is_destructive_and_confirmed_every_time():
    tool = EndProcessTool()
    assert tool.risk is Risk.DESTRUCTIVE
    assert tool.risk.needs_confirmation
    # No standing grant: each kill is approved against the specific process.
    assert not tool.risk.needs_consent


@pytest.mark.parametrize(
    "name",
    ["csrss.exe", "winlogon.exe", "services.exe", "lsass.exe", "System",
     "systemd", "init", "launchd"],
)
def test_critical_process_names_are_protected(name):
    """Ending these does not close a program, it takes the machine down."""
    import app.tools.system as system_module

    # Check against both platform tables, since the constant set is per-OS.
    combined = system_module._PROTECTED_WINDOWS | system_module._PROTECTED_POSIX
    assert name.lower() in combined


def test_protected_process_is_refused_through_the_registry_without_prompting(
    monkeypatch, tmp_path
):
    """The refusal must land before confirmation, in the real dispatch path.

    An earlier version of this test called `run()` directly and passed, while
    the actual flow prompted "End csrss.exe? Unsaved work will be lost", took
    the user's yes, and only then refused. Exercising the registry is the point.
    """
    from app.tools.base import ToolInvocation
    from app.tools.ledger import ToolLedger
    from app.tools.registry import ToolRegistry

    tool = EndProcessTool()
    target = ProcessInfo(pid=700, name="csrss.exe", cpu_percent=0.0,
                         memory_mb=5.0, create_time=time.time())
    monkeypatch.setattr(tool, "_resolve", lambda args: (target, None))

    registry = ToolRegistry(ToolLedger(tmp_path / "l.sqlite3"))
    registry.register(tool)

    asked: list[str] = []
    result = registry.invoke(
        ToolInvocation("end_process", {"pid": 700}, "model"),
        ctx(confirm=lambda p: asked.append(p) or True),
    )

    assert not result.ok
    assert "protected system process" in result.display
    assert asked == [], "the user must never be asked to approve this"
    assert "crash the machine" in (result.final_text or "")


def test_protected_process_is_refused_when_run_directly(monkeypatch):
    """Defence in depth: run() stays safe even without the registry."""
    tool = EndProcessTool()
    target = ProcessInfo(pid=700, name="csrss.exe", cpu_percent=0.0,
                         memory_mb=5.0, create_time=time.time())
    monkeypatch.setattr(tool, "_resolve", lambda args: (target, None))

    result = tool.run({"pid": 700}, ctx())
    assert not result.ok
    assert "protected system process" in result.display


def test_ending_the_model_server_is_refused(monkeypatch):
    tool = EndProcessTool()
    target = ProcessInfo(pid=900, name="ollama.exe", cpu_percent=0.0,
                         memory_mb=2000.0, create_time=time.time())
    monkeypatch.setattr(tool, "_resolve", lambda args: (target, None))

    result = tool.run({"pid": 900}, ctx())
    assert not result.ok
    assert "runs this assistant" in result.display


def test_ending_itself_is_refused(monkeypatch):
    tool = EndProcessTool()
    target = ProcessInfo(pid=os.getpid(), name="python.exe", cpu_percent=0.0,
                         memory_mb=50.0, create_time=time.time())
    monkeypatch.setattr(tool, "_resolve", lambda args: (target, None))

    result = tool.run({"pid": os.getpid()}, ctx())
    assert not result.ok
    assert "this assistant" in result.display


def test_missing_target_is_refused():
    result = EndProcessTool().run({}, ctx("kill it"))
    assert not result.ok
    assert "did not say which" in (result.correction or "")


def test_unknown_pid_is_reported_not_guessed():
    # A PID that cannot plausibly exist.
    result = EndProcessTool().run({"pid": 4_000_000}, ctx())
    assert not result.ok
    assert "not found" in (result.correction or "").lower() or "no process" in result.display


def test_ambiguous_name_refuses_and_lists(monkeypatch):
    """'close chrome' with fourteen chrome processes must not become fourteen
    terminations."""
    import app.tools.system as system_module

    now = time.time()
    fakes = [
        ProcessInfo(pid=101, name="chrome.exe", cpu_percent=1.0, memory_mb=300.0, create_time=now),
        ProcessInfo(pid=102, name="chrome.exe", cpu_percent=0.5, memory_mb=120.0, create_time=now),
    ]
    monkeypatch.setattr(system_module, "sample_processes", lambda **kw: fakes)

    result = EndProcessTool().run({"name": "chrome"}, ctx())
    assert not result.ok
    assert "need a PID" in result.display
    assert "101" in (result.correction or "") and "102" in (result.correction or "")


def test_name_with_no_match_is_reported(monkeypatch):
    import app.tools.system as system_module

    monkeypatch.setattr(system_module, "sample_processes", lambda **kw: [])
    result = EndProcessTool().run({"name": "definitelynotrunning"}, ctx())

    assert not result.ok
    assert "not running" in (result.correction or "").lower()


def test_recycled_pid_is_detected(monkeypatch):
    """Between choosing a target and the user approving it, the PID can be
    reused. Identity is re-checked at the moment of the kill.

    Uses a real spawned process, because the check needs a live PID that is
    neither ours nor protected — an earlier version passed `os.getpid()` and
    merely proved the self-check fires first.
    """
    import subprocess
    import sys as _sys

    child = subprocess.Popen(
        [_sys.executable, "-c", "import time; time.sleep(30)"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        # Same live PID, but claiming a start time from 1970: the identity check
        # must notice and refuse rather than terminating whatever now holds it.
        stale = ProcessInfo(pid=child.pid, name="chrome.exe", cpu_percent=0.0,
                            memory_mb=10.0, create_time=1.0)
        tool = EndProcessTool()
        monkeypatch.setattr(tool, "_resolve", lambda args: (stale, None))

        result = tool.run({"pid": stale.pid}, ctx())
        assert not result.ok
        assert "recycled" in result.display
        assert child.poll() is None, "the process must not have been terminated"
    finally:
        child.kill()
        child.wait(timeout=10)


def test_a_real_process_can_actually_be_ended():
    """The guards must not have made the tool useless — the happy path works on
    a process we spawned ourselves."""
    import subprocess
    import sys as _sys

    child = subprocess.Popen(
        [_sys.executable, "-c", "import time; time.sleep(30)"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        result = EndProcessTool().run({"pid": child.pid}, ctx("kill it"))
        assert result.ok, result.correction
        assert result.meta.get("exited") is True
        assert child.poll() is not None
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=10)


# ── confirmation wording ─────────────────────────────────────────
def test_confirmation_prompt_names_the_real_target(monkeypatch):
    """A user cannot evaluate "Run 'end_process' with {'pid': 8420}?" — the
    prompt has to say which program and how much it is using."""
    tool = EndProcessTool()
    target = ProcessInfo(pid=8420, name="Spotify.exe", cpu_percent=0.0,
                         memory_mb=150.0, create_time=time.time())
    monkeypatch.setattr(tool, "_resolve", lambda args: (target, None))

    prompt = tool.confirmation_prompt({"pid": 8420})
    assert "Spotify.exe" in prompt
    assert "8420" in prompt
    assert "Unsaved work" in prompt


def test_registry_uses_the_tool_prompt(monkeypatch, tmp_path):
    from app.tools.ledger import ToolLedger
    from app.tools.registry import ToolRegistry
    from app.tools.base import ToolInvocation

    tool = EndProcessTool()
    target = ProcessInfo(pid=8420, name="Spotify.exe", cpu_percent=0.0,
                         memory_mb=150.0, create_time=time.time())
    monkeypatch.setattr(tool, "_resolve", lambda args: (target, None))
    monkeypatch.setattr(
        tool, "run",
        lambda args, context: __import__(
            "app.tools.base", fromlist=["ToolResult"]
        ).ToolResult(ok=True, display="ended"),
    )

    registry = ToolRegistry(ToolLedger(tmp_path / "l.sqlite3"))
    registry.register(tool)

    prompts: list[str] = []
    registry.invoke(
        ToolInvocation("end_process", {"pid": 8420}, "model"),
        ctx(confirm=lambda p: prompts.append(p) or True),
    )
    assert any("Spotify.exe" in p for p in prompts)


# ── routing ──────────────────────────────────────────────────────
def test_kill_phrasing_does_not_trip_the_read_tools():
    """Overlapping matchers are fine — the model disambiguates — but a request
    to end something must at least reach the tool that can do it."""
    assert EndProcessTool().matches("kill chrome")
    assert EndProcessTool().matches("force quit spotify")
    assert not EndProcessTool().matches("what is using my memory")


# ── the fabrication bug ──────────────────────────────────────────
NATURAL_SYSTEM_QUESTIONS = [
    "give me my system information",
    "what are my PC specs?",
    "tell me about my computer",
    "system info",
    "what hardware do I have",
    "how much RAM do I have",
    "what processor is this",
    "show me my drives",
    "how big is my hard drive",
    "is my laptop powerful",
    "what are the specifications of this machine",
    "why is my laptop slow?",
    "how much free space is left",
    "what is my operating system",
]


@pytest.mark.parametrize("question", NATURAL_SYSTEM_QUESTIONS)
def test_natural_phrasings_reach_a_system_tool(question):
    """The gate missing a phrasing is not a missed feature, it is a lie.

    With a narrower trigger list, "give me my system information" matched
    nothing, so the model answered from imagination: 16 GB of RAM and three
    hard drives on a machine with 7.8 GB and one. For a read-only tool a false
    match costs one cheap local call, so the list is deliberately generous.
    """
    assert SystemStatusTool().matches(question) or TopProcessesTool().matches(question)


@pytest.mark.parametrize(
    "question",
    [
        "write me a poem about rain",
        "what is the capital of Peru",
        "summarise this document",
        "who wrote Hamlet",
        "translate this to French",
    ],
)
def test_ordinary_chat_still_costs_no_tool_call(question):
    """Generous must not mean indiscriminate — rule 9 still applies."""
    assert not SystemStatusTool().matches(question)
    assert not TopProcessesTool().matches(question)


def test_hardware_info_reports_real_values():
    """The snapshot had no CPU model at all, so even a correctly-routed
    question could only answer "[Unknown, unable to determine]"."""
    from app.tools.system import hardware_info

    info = hardware_info()
    assert info["cpu_name"] and info["cpu_name"] != "unknown"
    assert (info["cpu_cores_logical"] or 0) >= 1
    assert info["os"]
    assert info["architecture"]


def test_description_includes_the_processor_and_os():
    from app.tools.system import describe_snapshot, system_snapshot

    text = describe_snapshot(system_snapshot())
    assert "Processor:" in text
    assert "Operating system:" in text


def test_system_status_outscores_disk_report_on_a_hardware_question():
    """The tie-break that decides which tool the backstop runs."""
    from app.tools.builtin import DiskReportTool

    question = "Give me my system information: CPU, memory and disks."
    assert SystemStatusTool().match_score(question) > DiskReportTool().match_score(question)


# ── resisting the model's own earlier fabrications ───────────────
def test_grounding_overrides_earlier_conversation():
    """A model that fabricated once will copy itself when asked to "redo" it.

    The user's own transcript is in context, so a wrong answer from before a fix
    is a source the model happily reuses. The tool result has to say outright
    that it supersedes anything stated earlier.
    """
    content = SystemStatusTool().run({}, ctx("redo the system analysis")).content
    lowered = content.lower()
    assert "replace any hardware or storage numbers stated earlier" in lowered
    assert "your own previous answers" in lowered


def test_grounding_forbids_inventing_absent_sections():
    """It reported active connections and download speeds from a tool that
    collects no network data at all."""
    content = SystemStatusTool().run({}, ctx("full system analysis")).content
    assert "never invent a section" in content.lower()


def test_grounding_does_not_invite_an_apology():
    """Listing what was unavailable produced "I am unable to perform a
    comprehensive analysis due to the lack of network, GPU and temperature
    data" — turning a complete reading into an apology."""
    content = SystemStatusTool().run({}, ctx("system analysis")).content.lower()
    assert "never mention, apologise for" in content
    assert "complete as it stands" in content


def test_grounding_answers_the_question_asked_and_no_more():
    """The fix for the apology overshot into the opposite failure. Told to
    "report the lines above and nothing else", the assistant answered "how much
    of my memory is in use?" with the processor, the operating system, the
    memory and every drive — burying the one figure wanted among four
    that were not. Answering more than was asked is its own wrong answer."""
    content = SystemStatusTool().run({}, ctx("how much memory is in use?")).content
    lowered = content.lower()

    assert "answer the question that was actually asked" in lowered
    assert "only when the question is a general one" in lowered
    # The wording that caused the dump must not come back.
    assert "nothing else" not in lowered


def test_figures_state_which_way_round_they_run():
    """Given "415.8 GB (72% used)" the model reported "72% free space" — an
    inversion that turns a nearly-full disk into a healthy one. Every figure now
    names both halves explicitly."""
    from app.tools.system import describe_snapshot, system_snapshot

    text = describe_snapshot(system_snapshot())
    assert "in use" in text
    assert "still free" in text
    # The bare, invertible phrasing must be gone.
    assert "% used)" not in text


def test_drive_count_is_stated_explicitly():
    """It invented D: and E: drives. Saying how many exist closes that door."""
    from app.tools.system import describe_snapshot, system_snapshot

    snapshot = system_snapshot()
    text = describe_snapshot(snapshot)
    if snapshot["disks"]:
        assert "no other drives" in text.lower()
        assert f"Drives attached: {len(snapshot['disks'])}" in text


def test_snapshot_reports_used_alongside_free():
    from app.tools.system import system_snapshot

    snapshot = system_snapshot()
    assert "used_gb" in snapshot["memory"]
    total = snapshot["memory"]["total_gb"]
    parts = snapshot["memory"]["used_gb"] + snapshot["memory"]["available_gb"]
    assert abs(total - parts) < 0.05, "used + free should account for total"
    for disk in snapshot["disks"]:
        assert "used_gb" in disk


# ---- which tool answers a question about the drive ---------------
def test_capacity_questions_do_not_go_to_the_stale_disk_scan():
    """Asked "how much free space is on my drive?" the scan tool answered from
    file totals — "7.1 GB of total storage" on a 415 GB disk — and then
    apologised for not knowing the free space. It never had that figure: a scan
    knows which files exist, not the volume's capacity."""
    from app.tools.builtin import DiskReportTool
    from app.tools.system import SystemStatusTool

    disk, system = DiskReportTool(), SystemStatusTool()
    for question in (
        "how much free space is on my drive?",
        "how much space do I have left?",
        "how full is my disk?",
        "am I running out of space?",
    ):
        assert not disk.matches(question), question
        assert system.matches(question), question


def test_the_disk_scan_still_owns_questions_about_files():
    """Standing aside on capacity must not cost it its actual job."""
    from app.tools.builtin import DiskReportTool

    disk = DiskReportTool()
    for question in (
        "do I have duplicate files?",
        "which files are wasting storage?",
        "are any of my files corrupt?",
        "what is taking up space on my disk?",
    ):
        assert disk.matches(question), question


# ---- answering about *what* is using the machine -----------------
def test_a_memory_question_is_ordered_by_memory():
    """The list is truncated after sorting, so the order decides who appears
    at all. Sorted by CPU and cut to five, "the top 5 things consuming my
    memory" named the five busiest processes — on a real run the actual
    largest, llama-server.exe at 2,185 MB, was not in the list."""
    from app.tools.system import sample_processes

    # One sample, sorted two ways. Taking two samples compared different
    # moments: a process grew by half a megabyte between them and the assertion
    # failed on a system that was behaving correctly.
    everything = sample_processes(sort_by="memory")
    by_memory = everything[:5]
    by_cpu = sorted(everything, key=lambda p: p.cpu_percent, reverse=True)[:5]

    assert [p.memory_mb for p in by_memory] == sorted(
        (p.memory_mb for p in by_memory), reverse=True
    )
    # The point of the fix: the biggest memory user is never cut off.
    assert max(p.memory_mb for p in by_memory) >= max(p.memory_mb for p in by_cpu)


def test_the_cpu_ordering_is_by_cpu():
    from app.tools.system import sample_processes

    by_cpu = sample_processes(limit=5, sort_by="cpu")
    assert [p.cpu_percent for p in by_cpu] == sorted(
        (p.cpu_percent for p in by_cpu), reverse=True
    )


def test_the_ordering_follows_the_question():
    from app.tools.system import preferred_process_sort

    assert preferred_process_sort("top 5 things consuming my memory") == "memory"
    assert preferred_process_sort("what is eating my RAM?") == "memory"
    assert preferred_process_sort("what is using my CPU?") == "cpu"
    assert preferred_process_sort("what is running?") == "cpu"


def test_the_process_list_says_how_it_is_ordered():
    """Presented as a bare "top processes", a memory-ordered list reads as a
    CPU one and gets ranked wrongly in the answer."""
    from app.tools.system import describe_processes, sample_processes

    procs = sample_processes(limit=3, sort_by="memory")
    assert "by memory" in describe_processes(procs, 3, "memory")
    assert "by CPU" in describe_processes(procs, 3, "cpu")


def test_top_processes_matches_the_phrasing_that_missed():
    """"list top 5 things that are consuming my memory" matched none of the
    original triggers, so the tool was never a candidate and a totals-only
    reading answered unopposed — twice, identically."""
    from app.tools.system import TopProcessesTool

    tool = TopProcessesTool()
    for question in (
        "list top 5 things that are consuming my memory",
        "what programs are taking up my memory?",
        "which apps use the most RAM?",
        "what is the biggest memory hog?",
    ):
        assert tool.matches(question), question


def test_top_processes_is_scored_like_its_competitor():
    """An unscored tool scores 1 in the backstop while system_status returns
    its trigger count, so the totals reading won nearly every contest."""
    from app.tools.system import SystemStatusTool, TopProcessesTool

    question = "what processes are consuming the most memory?"
    assert TopProcessesTool().match_score(question) >= SystemStatusTool().match_score(
        question
    )


# ---- depth follows the request ----------------------------------
def test_a_detailed_memory_report_names_what_used_the_memory():
    """"Get me a detailed report on memory usage" came back as three lines of
    totals. A memory report that cannot say where the memory went is not one."""
    content = SystemStatusTool().run(
        {}, ctx("get me a detailed report on memory usage")
    ).content
    assert "Top processes" in content
    assert "asked for detail" in content


def test_a_narrow_question_still_gets_one_sentence():
    """The cap exists because the same tool once answered "how much of my
    memory is in use?" with the processor, the OS and every drive."""
    content = SystemStatusTool().run({}, ctx("how much memory is in use?")).content
    assert "in one short sentence" in content
    assert "Top processes" not in content


# ---- narrowing the reading, not the prompt ----------------------
def test_the_reading_is_narrowed_to_the_area_asked_about():
    """Filtering moved out of the prompt after two failures. Handed the whole
    machine and told to report only the memory, the model answered with the
    processor, the OS and every drive; told to give "every figure, one per
    line, largest first" it stripped the labels and returned "1. 7.43 GB /
    2. 0.35 GB / 3. 96%". A section never shown cannot be misquoted."""
    from app.tools.system import describe_snapshot, system_snapshot

    snapshot = system_snapshot()
    memory_only = describe_snapshot(snapshot, "memory")
    assert "Memory:" in memory_only
    assert "Processor:" not in memory_only
    assert "Drive " not in memory_only

    disk_only = describe_snapshot(snapshot, "disk")
    assert "Drive " in disk_only
    assert "Memory:" not in disk_only


def test_a_general_question_still_gets_the_whole_machine():
    """Narrowing must not cost the overview its content."""
    from app.tools.system import describe_snapshot, snapshot_focus, system_snapshot

    assert snapshot_focus("give me my system information") is None
    full = describe_snapshot(system_snapshot(), None)
    for expected in ("Processor:", "Memory:", "Drive ", "Uptime:"):
        assert expected in full, expected


def test_a_question_spanning_two_areas_is_not_narrowed():
    """"How much memory and disk space do I have?" is a general enquiry."""
    from app.tools.system import snapshot_focus

    assert snapshot_focus("how much memory and disk space do I have?") is None
    assert snapshot_focus("compare my RAM and my CPU") is None


def test_the_detailed_reading_carries_no_unrelated_sections():
    """The whole failure this replaced: a detailed memory report that led with
    the processor and ended with the drives."""
    content = SystemStatusTool().run(
        {}, ctx("get me a detailed report on memory usage")
    ).content
    assert "Memory:" in content
    assert "Top processes" in content
    assert "Processor:" not in content
    assert "Drive C" not in content

