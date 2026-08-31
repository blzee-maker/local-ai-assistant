"""Shared test setup.

The event log is redirected for every test without exception. Not tidiness: the
default destination is the user's real `data/events.jsonl`, and a suite that
appends a few hundred synthetic entries to it every run would quietly ruin the
one file whose entire job is telling you what really happened.
"""
from __future__ import annotations

import pytest

from config import settings


@pytest.fixture(autouse=True)
def _isolate_event_log(tmp_path, monkeypatch):
    monkeypatch.setattr(
        settings, "event_log_path", str(tmp_path / "events.jsonl"), raising=False
    )
    # Off is already the default; pinned so a test that flips it cannot leak
    # the setting into the next one and start recording speech.
    monkeypatch.setattr(settings, "log_transcripts", False, raising=False)
    yield
