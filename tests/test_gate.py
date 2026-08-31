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


def test_speech_after_the_window_lapses_is_never_run():
    """The guarantee is that it does not run. It reports the lapse as well now,
    but "reported" and "acted on" must not be allowed to drift into each other:
    assert the safety property, then the label."""
    g = gate()
    g.consider("Wake up buddy.", now=0.0)           # awake until 12.0
    d = g.consider("someone talking in the room", now=15.0)
    assert d.action != "run"
    assert d.command == ""
    assert d.action == "lapsed"


def test_the_lapse_is_reported_once_and_then_the_room_is_ignored():
    """Said once because the speaker was plainly addressing us and deserves to
    know why nothing happened. Not twice, because a gate that narrates every
    overheard sentence is one nobody leaves running."""
    g = gate()
    g.consider("Wake up buddy.", now=0.0)
    assert g.consider("are you there", now=15.0).action == "lapsed"
    assert g.consider("someone else talking", now=16.0).action == "ignore"
    assert g.consider("and more chatter", now=17.0).action == "ignore"


def test_a_gate_that_was_never_awake_reports_nothing():
    """"You spoke too late" and "that was not addressed to me" are different
    facts. Only the first is worth saying, and only if the window had opened."""
    g = gate()
    assert g.consider("open spotify", now=0.0).action == "ignore"
    assert g.consider("delete everything", now=1.0).action == "ignore"


def test_it_must_be_re_woken_after_going_quiet():
    g = gate()
    g.consider("Wake up buddy, what is my disk usage.", now=0.0)
    assert g.consider("open spotify", now=30.0).action != "run"      # lapsed
    assert g.consider("Wake up buddy, open spotify.", now=31.0).action == "run"


# ── being told the window is open, and how long for ──────────────
def test_wake_opens_the_window_without_a_spoken_phrase():
    """Typing `/listen` is the intent; asking for the wake phrase straight
    afterwards asks the same question twice."""
    g = gate()
    assert not g.is_awake(0.0)
    g.wake(0.0)
    assert g.is_awake(0.0)
    assert g.consider("what time is it", now=1.0).action == "run"


def test_remaining_decays_and_bottoms_out_at_zero():
    g = gate()
    g.wake(0.0)
    assert g.remaining(0.0) == WINDOW
    assert g.remaining(6.0) == WINDOW - 6.0
    # Never negative: the loop renders this number, and "-4s left" is worse
    # than useless.
    assert g.remaining(WINDOW + 5.0) == 0.0


def test_remaining_is_zero_while_asleep():
    assert gate().remaining(0.0) == 0.0


def test_a_reported_lapse_is_not_reported_again_by_the_gate():
    """The loop watches the clock as well, because it can sit blocked on the
    microphone for minutes. Whichever notices first suppresses the other."""
    g = gate()
    g.wake(0.0)
    g.note_lapse_reported()                          # the watchdog got there first
    assert g.consider("are you there", now=99.0).action == "ignore"


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


def test_the_window_is_readable():
    """The log records the window actually in force. Re-reading the configured
    default instead produced an entry saying 12s directly above one saying 2s
    remained."""
    assert ConversationGate(PHRASE, 2.0).window == 2.0
