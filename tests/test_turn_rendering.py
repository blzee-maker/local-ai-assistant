"""What one turn actually puts on screen when a tool ran first.

Written from a real transcript. Asked to open Spotify, the assistant rendered:

    you (voice) Open Spotify.
    Opened Spotify.g Spotify

— no speaker label, and the tail of the progress line still showing. The label
had been printed before the stream started, so the first carriage return wrote
over it, and the progress line was never wiped because the tool never sent
"done".
"""
from __future__ import annotations

import io
import sys

import pytest

from app.cli import render, repl
from app.core.assistant import AssistantEvent


@pytest.fixture(autouse=True)
def _clean_slate():
    render._transient_pending = False
    yield
    render._transient_pending = False


class _FakeMemory:
    def record(self, role, content):  # noqa: D102 - test double
        pass


class _FakeAssistant:
    """Emits a progress line first, the way a tool-running turn does."""

    def __init__(self, progress_messages, tokens, tool_used=None):
        self._progress = progress_messages
        self._tokens = tokens
        self._tool_used = tool_used
        self.memory = _FakeMemory()

    def chat_stream(self, _history, *, progress=None, **_kwargs):
        for message in self._progress:
            progress(message)
        for token in self._tokens:
            yield AssistantEvent(type="token", text=token)
        yield AssistantEvent(
            type="done",
            tool_used=self._tool_used,
            metrics={"determined": True},
        )


def _render(assistant) -> str:
    buffer = io.StringIO()
    real_out, real_file = sys.stdout, render.console.file
    sys.stdout = buffer
    render.console.file = buffer
    try:
        repl.run_turn(assistant, repl.Session(), "Open Spotify.")
    finally:
        sys.stdout, render.console.file = real_out, real_file
    return buffer.getvalue()


def test_the_speaker_label_survives_a_progress_line():
    """The label used to be printed before the stream and overwritten by the
    first carriage return, leaving a reply with nobody's name on it."""
    out = _render(_FakeAssistant(["opening Spotify"], ["Opened Spotify."]))
    assert repl.BOT_LABEL in out


def test_the_label_is_printed_exactly_once():
    out = _render(
        _FakeAssistant(["opening Spotify"], ["Opened ", "Spotify", "."])
    )
    assert out.count(repl.BOT_LABEL) == 1


def test_the_progress_line_is_wiped_before_the_reply():
    """The tool sends no "done" — this is the exact shape that produced
    "Opened Spotify.g Spotify"."""
    out = _render(_FakeAssistant(["opening Spotify"], ["Opened Spotify."]))
    # The reply is preceded by a full-width wipe, so nothing of the progress
    # line can still be sitting under it.
    assert " " * render.TRANSIENT_WIDTH in out
    assert render.transient_pending() is False


def test_no_progress_means_no_stray_blank_line():
    """Ordinary chat must not pay for the wipe."""
    out = _render(_FakeAssistant([], ["Hello."]))
    assert " " * render.TRANSIENT_WIDTH not in out
    assert repl.BOT_LABEL in out


def test_a_turn_that_produces_no_tokens_leaves_nothing_pending():
    class _Empty(_FakeAssistant):
        def chat_stream(self, _history, *, progress=None, **_kwargs):
            progress("thinking")
            yield AssistantEvent(type="done", metrics={})

    _render(_Empty([], []))
    assert render.transient_pending() is False


# ── what a text turn leaves in the log ───────────────────────────
# Voice was instrumented first, which left the far more common path — someone
# typing at the prompt — writing nothing at all. A log that is empty exactly
# when you reach for it is worse than knowing there isn't one.
def _logged_turn(assistant) -> list[dict]:
    from app import eventlog

    _render(assistant)
    return eventlog.read(limit=0)


def test_a_text_turn_is_bracketed_in_the_log():
    entries = _logged_turn(_FakeAssistant([], ["Hello."]))
    assert [e["event"] for e in entries] == ["turn.start", "turn.end"]


def test_the_typed_prompt_is_not_written_down():
    entries = _logged_turn(_FakeAssistant([], ["Hello."]))
    start = entries[0]
    assert "Spotify" not in str(start)
    assert start["prompt"]["redacted"] is True


def test_the_end_of_a_turn_records_what_would_explain_a_bad_answer():
    """Model, fallback, tool and timings — the things asked about after the
    fact, when the screen has long since scrolled."""
    entries = _logged_turn(
        _FakeAssistant([], ["Opened Spotify."], tool_used={"tool": "open_app"})
    )
    end = entries[-1]
    assert end["event"] == "turn.end"
    assert end["tool"] == "open_app"
    assert end["reply_chars"] == len("Opened Spotify.")
    assert end["determined"] is True
    assert end["took_s"] >= 0


def test_a_fallback_is_recorded_because_it_explains_a_worse_answer():
    """On a machine this tight a weaker model answers without anything
    failing, and that is the commonest cause of "why was that worse today"."""
    class _FellBack(_FakeAssistant):
        def chat_stream(self, _history, *, progress=None, **_kwargs):
            yield AssistantEvent(type="token", text="Sort of.")
            yield AssistantEvent(
                type="done",
                metrics={"model": "qwen2.5:1.5b", "fell_back": True},
            )

    entries = _logged_turn(_FellBack([], []))
    end = entries[-1]
    assert end["fell_back"] is True
    assert end["model"] == "qwen2.5:1.5b"


def test_a_failed_turn_records_the_error():
    class _Broken(_FakeAssistant):
        def chat_stream(self, _history, *, progress=None, **_kwargs):
            yield AssistantEvent(type="error", error="model would not load")

    entries = _logged_turn(_Broken([], []))
    assert entries[-1]["event"] == "turn.error"
    assert "would not load" in entries[-1]["error"]


def test_a_turn_that_produced_nothing_says_so():
    """No tokens and no error — silent on screen, and until now silent
    afterwards too."""
    class _Empty(_FakeAssistant):
        def chat_stream(self, _history, *, progress=None, **_kwargs):
            yield AssistantEvent(type="done", metrics={})

    entries = _logged_turn(_Empty([], []))
    assert entries[-1]["event"] == "turn.empty"
