"""Tests for the awake/asleep conversation gate.

`now` is injected everywhere so the follow-up window is exercised without real
time passing. The asymmetry from test_wake carries over: waking at the wrong
moment is worse than missing a follow-up, so the cases lean on what must NOT be
acted on once the window has lapsed.
"""
from __future__ import annotations

from app.voice.gate import ConversationGate, is_sleep_command

PHRASE = "wake up buddy"
WINDOW = 12.0


def gate() -> ConversationGate:
    return ConversationGate(PHRASE, WINDOW)


# ── asleep ───────────────────────────────────────────────────────
def test_asleep_ignores_ordinary_speech():
    g = gate()
    assert g.consider("what is the weather like today", now=0.0).action == "ignore"
    assert g.consider("I told my buddy about it", now=1.0).action == "ignore"


def test_the_wake_phrase_with_a_command_runs_it():
    g = gate()
    d = g.consider("Wake up buddy, how much memory is in use.", now=0.0)
    assert d.action == "run"
    assert d.command == "how much memory is in use"


def test_the_bare_wake_phrase_only_acknowledges():
    g = gate()
    d = g.consider("Wake up buddy.", now=0.0)
    assert d.action == "acknowledge"
    assert d.command == ""


# ── the follow-up window ─────────────────────────────────────────
def test_a_follow_up_within_the_window_needs_no_phrase():
    g = gate()
    g.consider("Wake up buddy.", now=0.0)          # awake until 12.0
    d = g.consider("what time is it", now=5.0)      # still inside the window
    assert d.action == "run"
    assert d.command == "what time is it"


def test_each_command_refreshes_the_window():
    g = gate()
    g.consider("Wake up buddy.", now=0.0)
    g.consider("what time is it", now=10.0)         # refreshes to 22.0
    d = g.consider("and tomorrow", now=20.0)        # would be stale off 0.0, but isn't
    assert d.action == "run"


def test_speech_after_the_window_lapses_is_ignored():
    g = gate()
    g.consider("Wake up buddy.", now=0.0)           # awake until 12.0
    d = g.consider("someone talking in the room", now=15.0)
    assert d.action == "ignore"


def test_it_must_be_re_woken_after_going_quiet():
    g = gate()
    g.consider("Wake up buddy, what is my disk usage.", now=0.0)
    assert g.consider("open spotify", now=30.0).action == "ignore"   # lapsed
    assert g.consider("Wake up buddy, open spotify.", now=31.0).action == "run"


# ── ending the conversation ──────────────────────────────────────
def test_a_sleep_phrase_ends_the_window_immediately():
    g = gate()
    g.consider("Wake up buddy.", now=0.0)
    d = g.consider("go to sleep", now=2.0)
    assert d.action == "sleep"
    # And it is genuinely asleep now, well inside the old window.
    assert g.consider("what time is it", now=3.0).action == "ignore"


def test_sleep_phrases_are_recognised():
    assert is_sleep_command("go to sleep")
    assert is_sleep_command("Never mind.")
    assert is_sleep_command("okay, that's all")
    assert not is_sleep_command("what is the sleep schedule")


# ── the documented trade-off ─────────────────────────────────────
def test_while_awake_any_speech_is_a_command():
    """This is the risk the short window exists to bound, asserted so it can't
    change silently: inside the window, speech never meant for the assistant is
    still acted on. Shrinking the window is the mitigation, not pretending
    otherwise."""
    g = gate()
    g.consider("Wake up buddy.", now=0.0)
    d = g.consider("no I was talking to you not the computer", now=3.0)
    assert d.action == "run"


def test_empty_transcript_is_ignored():
    assert gate().consider("", now=0.0).action == "ignore"
    assert gate().consider("   ", now=0.0).action == "ignore"
