"""Tests for the background scheduler.

The daemon runs while nobody is watching, so the properties that matter are the
ones about restraint: it must not run expensive work on launch, must not act on
permissions it was never given, must not silently absorb a user's library, and
must not die because one job broke.
"""
from __future__ import annotations

import time

import pytest

from app.daemon.jobs import Job, JobResult
from app.daemon.journal import Journal, JobRun
from app.daemon.scheduler import Scheduler, briefing


class FakeJob(Job):
    def __init__(
        self,
        name: str,
        interval_s: float = 60.0,
        run_on_start: bool = True,
        outcome: str = "ok",
        raises: bool = False,
        notable: str | None = None,
    ):
        self.name = name
        self.description = f"fake {name}"
        self.interval_s = interval_s
        self.run_on_start = run_on_start
        self.outcome = outcome
        self.raises = raises
        self.notable = notable
        self.runs = 0

    def run(self, assistant, journal) -> JobResult:
        self.runs += 1
        if self.raises:
            raise RuntimeError("job exploded")
        return JobResult(outcome=self.outcome, detail=f"{self.name} ran",
                         notable=self.notable)


@pytest.fixture
def journal(tmp_path) -> Journal:
    return Journal(tmp_path / "daemon.sqlite3")


def scheduler_for(jobs, journal, events=None) -> Scheduler:
    return Scheduler(
        assistant=None, jobs=jobs, journal=journal,
        on_event=(lambda level, msg: events.append((level, msg))) if events is not None else None,
    )


# ── scheduling ───────────────────────────────────────────────────
def test_run_on_start_jobs_are_due_immediately(journal):
    job = FakeJob("cheap", run_on_start=True)
    scheduler = scheduler_for([job], journal)
    assert [d.job.name for d in scheduler.due_jobs()] == ["cheap"]


def test_expensive_jobs_are_not_due_on_first_launch(journal):
    """A weekly disk scan must not fire because someone restarted the daemon."""
    job = FakeJob("scan", interval_s=3600, run_on_start=False)
    scheduler = scheduler_for([job], journal)

    assert scheduler.due_jobs() == []
    assert scheduler.next_due(job) == pytest.approx(time.time() + 3600, abs=5)


def test_job_is_not_due_again_until_its_interval_passes(journal):
    job = FakeJob("cheap", interval_s=600)
    scheduler = scheduler_for([job], journal)

    scheduler.run_job(job)
    assert scheduler.due_jobs() == []
    # ...but is due once the interval has elapsed.
    assert scheduler.due_jobs(now=time.time() + 601)


def test_schedule_survives_a_restart(tmp_path):
    """Without persistence every restart looks like a fresh start, and periodic
    work would run on every launch."""
    path = tmp_path / "daemon.sqlite3"
    job = FakeJob("cheap", interval_s=600)

    first = scheduler_for([job], Journal(path))
    first.run_job(job)
    first.journal.close()

    second = scheduler_for([FakeJob("cheap", interval_s=600)], Journal(path))
    assert second.due_jobs() == []


def test_most_overdue_job_runs_first(journal):
    old = FakeJob("old", interval_s=60)
    new = FakeJob("new", interval_s=60)
    scheduler = scheduler_for([old, new], journal)

    now = time.time()
    journal.record_run(JobRun("old", now - 600, 0.1, "ok"))
    journal.record_run(JobRun("new", now - 100, 0.1, "ok"))

    assert [d.job.name for d in scheduler.due_jobs(now)] == ["old", "new"]


# ── failure handling ─────────────────────────────────────────────
def test_a_raising_job_is_recorded_not_propagated(journal):
    job = FakeJob("broken", raises=True)
    scheduler = scheduler_for([job], journal)

    result = scheduler.run_job(job)
    assert result.outcome == "error"
    assert "job exploded" in result.detail
    assert journal.state("broken").failures == 1


def test_one_broken_job_does_not_stop_the_others(journal):
    broken = FakeJob("broken", raises=True)
    healthy = FakeJob("healthy")
    scheduler = scheduler_for([broken, healthy], journal)

    scheduler.tick()
    assert healthy.runs == 1


def test_repeated_failures_back_the_job_off(journal):
    job = FakeJob("broken", interval_s=60, raises=True)
    scheduler = scheduler_for([job], journal)

    scheduler.run_job(job)
    after_one = scheduler.next_due(job)
    scheduler.run_job(job)
    after_two = scheduler.next_due(job)

    assert after_two > after_one, "backoff should widen with repeated failure"


def test_success_clears_the_failure_count(journal):
    job = FakeJob("flaky", raises=True)
    scheduler = scheduler_for([job], journal)
    scheduler.run_job(job)
    assert journal.state("flaky").failures == 1

    job.raises = False
    scheduler.run_job(job)
    assert journal.state("flaky").failures == 0


# ── reporting ────────────────────────────────────────────────────
def test_notable_results_are_emitted(journal):
    events: list[tuple[str, str]] = []
    job = FakeJob("scan", notable="Found 3 damaged files")
    scheduler_for([job], journal, events).run_job(job)

    assert ("notable", "Found 3 damaged files") in events


def test_uneventful_runs_are_not_notable(journal):
    events: list[tuple[str, str]] = []
    scheduler_for([FakeJob("quiet")], journal, events).run_job(FakeJob("quiet"))

    assert not any(level == "notable" for level, _m in events)


def test_briefing_summarises_recent_runs(journal):
    now = time.time()
    journal.record_run(JobRun("index", now - 100, 1.0, "ok", "indexed 2"))
    journal.record_run(JobRun("index", now - 50, 1.0, "ok", "no new documents"))
    journal.record_run(JobRun("scan", now - 30, 9.0, "error", "disk unreadable"))

    text = briefing(journal, since_hours=1)
    assert "index: ran 2x" in text
    assert "scan" in text and "failure" in text


def test_briefing_is_quiet_when_nothing_ran(journal):
    assert "Nothing has run" in briefing(journal, since_hours=1)


# ── seen files ───────────────────────────────────────────────────
def test_seen_files_are_remembered(journal):
    journal.mark_seen({"C:/a.pdf": 100.0, "C:/b.pdf": 200.0})
    assert journal.known_files() == {"C:/a.pdf": 100.0, "C:/b.pdf": 200.0}


def test_seen_file_tracking_is_separate_from_the_index(journal):
    """A document the user deliberately removed from the index must not be
    silently re-added on the next sweep — 'seen' and 'indexed' are different
    questions."""
    journal.mark_seen({"C:/a.pdf": 100.0})
    known = journal.known_files()
    assert "C:/a.pdf" in known
    # Changing the file makes it a candidate again; leaving it alone does not.
    assert known["C:/a.pdf"] == 100.0


# ── permissions ──────────────────────────────────────────────────
def test_jobs_skip_when_file_consent_is_absent(tmp_path, monkeypatch):
    from app import consent
    from app.daemon.jobs import DiskScanJob, IndexNewFilesJob

    monkeypatch.setattr(consent, "CONSENT_PATH", tmp_path / "consent.json")
    monkeypatch.setattr(consent, "current_roots", lambda: [r"C:\A"])
    consent.decline()

    journal_obj = Journal(tmp_path / "d.sqlite3")
    for job in (IndexNewFilesJob(), DiskScanJob()):
        result = job.run(assistant=None, journal=journal_obj)
        assert result.outcome == "skipped"
        assert "approved" in result.detail


def test_an_absent_user_cannot_authorise_a_destructive_tool(tmp_path):
    """The daemon runs with no confirmer. Nobody is present to be asked, and an
    absent user is not consent — so destructive capabilities must refuse."""
    from app.tools.base import Risk, Tool, ToolContext, ToolInvocation, ToolResult
    from app.tools.ledger import ToolLedger
    from app.tools.registry import ToolRegistry

    class DangerousTool(Tool):
        name = "wipe"
        description = "destroy things"
        risk = Risk.DESTRUCTIVE

        def __init__(self):
            self.ran = False

        def schema(self) -> dict:
            return {"type": "function", "function": {"name": self.name,
                    "description": self.description,
                    "parameters": {"type": "object", "properties": {}}}}

        def matches(self, text: str) -> bool:
            return True

        def run(self, arguments: dict, context: ToolContext) -> ToolResult:
            self.ran = True
            return ToolResult(ok=True)

    tool = DangerousTool()
    registry = ToolRegistry(ToolLedger(tmp_path / "l.sqlite3"))
    registry.register(tool)

    # Exactly how a job would call it: no confirmer.
    result = registry.invoke(
        ToolInvocation("wipe", {}, "model"),
        ToolContext(assistant=None, request_text="", confirm=None),
    )

    assert not result.ok
    assert not tool.ran, "a destructive tool must not run unattended"
