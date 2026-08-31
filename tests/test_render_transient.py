"""The rewritten progress line, and the corruption it used to leave behind.

A carriage return moves the cursor back but erases nothing, so every one of
these cases is about what is still on screen *after* the line is done with.

The bug this was written from: a turn that launched Spotify rendered as
"Opened Spotify.g Spotify" with no speaker label on it. Two causes, both here
— the progress line was only wiped when a tool sent the literal message
"done" (one tool in the codebase ever did), and the speaker label was printed
before the stream, so the first carriage return wrote straight over it.
"""
from __future__ import annotations

import io
import sys

import pytest

from app.cli import render


@pytest.fixture(autouse=True)
def _clean_slate():
    """Module state is global, so a leaked pending flag would leak between
    tests — exactly the failure mode being tested against."""
    render._transient_pending = False
    yield
    render._transient_pending = False


def _capture(fn) -> str:
    buffer = io.StringIO()
    real = sys.stdout
    sys.stdout = buffer
    try:
        fn()
    finally:
        sys.stdout = real
    return buffer.getvalue()


def test_a_transient_line_is_padded_to_full_width():
    """The padding is the whole mechanism: a short message after a long one
    must cover it, not sit inside it."""
    out = _capture(lambda: render.transient("opening Spotify"))
    assert out.endswith("\r")
    assert len(out.rstrip("\r")) == render.TRANSIENT_WIDTH


def test_clearing_wipes_the_full_width():
    def go():
        render.transient("scanning 12000 files")
        render.clear_transient()

    out = _capture(go)
    assert out.endswith(" " * render.TRANSIENT_WIDTH + "\r")


def test_clearing_when_nothing_is_pending_writes_nothing():
    """Called unconditionally from every print path, so it has to be free when
    there is nothing to wipe — otherwise ordinary chat gains a blank line."""
    assert _capture(render.clear_transient) == ""


def test_a_tool_that_never_says_done_still_cannot_leave_residue():
    """The actual regression. `report_progress("done")` is emitted by exactly
    one tool; every other tool ends its work by simply stopping. Wiping must
    not depend on the tool remembering."""
    def go():
        render.transient("opening Spotify")   # no "done" ever sent
        render.clear_transient()              # the print path wipes regardless

    out = _capture(go)
    assert render.transient_pending() is False
    # Whatever the next write is, it lands on a blank line: the last thing
    # written before the cursor returned is all spaces.
    assert out.rstrip("\r").endswith(" " * 10)


def test_repeated_clears_are_idempotent():
    def go():
        render.transient("thinking")
        render.clear_transient()
        render.clear_transient()
        render.clear_transient()

    out = _capture(go)
    assert out.count(" " * render.TRANSIENT_WIDTH) == 1


def test_show_progress_still_wipes_on_the_done_message():
    """The old contract kept working — disk.py sends it and should stay
    correct, it is just no longer the only thing standing between a tool and a
    corrupted screen."""
    from app.cli import repl

    def go():
        repl.show_progress("scanning")
        assert render.transient_pending() is True
        repl.show_progress("done")

    _capture(go)
    assert render.transient_pending() is False
