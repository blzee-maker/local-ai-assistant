"""The voice loop's dispatch, driven by scripted transcripts.

No microphone and no model: `handle` (run a command) and `say` (speak a line)
are injected and just record what they were asked to do, so this checks the
wiring between the gate's decisions and the loop's actions — asleep ignores,
wake runs, a bare wake acknowledges, a sleep phrase says goodbye.

The gate is given a wide window so the wall-clock `time.time()` the loop reads
never lapses mid-list; the window's *expiry* is the gate's own test to own.
"""
from __future__ import annotations

from app.cli import voiceloop
from app.voice.gate import ConversationGate


def _wide_gate() -> ConversationGate:
    return ConversationGate("wake up buddy", follow_up_window_s=600.0)


def test_the_loop_runs_wakes_follows_up_and_sleeps():
    events: list[tuple[str, str]] = []
    transcripts = [
        "the weather is nice today",           # asleep → ignored
        "Wake up buddy, what time is it.",     # wake + command → run
        "and what about tomorrow",             # follow-up within window → run
        "go to sleep",                         # → sleep
        "open spotify",                        # asleep again → ignored
    ]
    voiceloop.run(
        None,
        gate=_wide_gate(),
        transcripts=transcripts,
        handle=lambda command: events.append(("run", command)),
        say=lambda text: events.append(("say", text)),
    )
    assert events == [
        ("run", "what time is it"),
        ("run", "and what about tomorrow"),
        ("say", "Going to sleep."),
    ]


def test_a_bare_wake_phrase_acknowledges_and_runs_nothing():
    events: list[tuple[str, str]] = []
    voiceloop.run(
        None,
        gate=_wide_gate(),
        transcripts=["Wake up buddy."],
        handle=lambda command: events.append(("run", command)),
        say=lambda text: events.append(("say", text)),
    )
    assert events == [("say", "Yes?")]


def test_speech_before_any_wake_is_never_run():
    events: list[tuple[str, str]] = []
    voiceloop.run(
        None,
        gate=_wide_gate(),
        transcripts=["delete everything", "open my bank account", "buddy help"],
        handle=lambda command: events.append(("run", command)),
        say=lambda text: events.append(("say", text)),
    )
    assert events == []


# ── replies come back as text unless asked otherwise ─────────────
def test_replies_are_not_spoken_by_default():
    """Speaking means holding Piper in memory beside the transcriber and the
    language model. On a machine sitting above 90% used that is the difference
    between a fallback that rescues an answer and one with nothing left to load
    into, so voice in / text out is the default."""
    import inspect

    from app.cli import voiceloop

    source = inspect.getsource(voiceloop.run)
    assert "session.speak = True" not in source


def test_a_silent_session_says_nothing_aloud(monkeypatch):
    """The acknowledgement and the goodbye are printed either way; only the
    audio is conditional."""
    from app.cli import voiceloop
    from app.cli.repl import Session

    spoken: list[str] = []
    monkeypatch.setattr(voiceloop, "_say", lambda _a, text: spoken.append(text))

    voiceloop.run(
        assistant=None,
        session=Session(speak=False),
        transcripts=["wake up buddy"],
        handle=lambda _c: None,
    )
    assert spoken == []


def test_a_speaking_session_still_can(monkeypatch):
    from app.cli import voiceloop
    from app.cli.repl import Session

    spoken: list[str] = []
    monkeypatch.setattr(voiceloop, "_say", lambda _a, text: spoken.append(text))

    voiceloop.run(
        assistant=None,
        session=Session(speak=True),
        transcripts=["wake up buddy"],
        handle=lambda _c: None,
    )
    assert spoken == ["Yes?"]


# ── saying what state it is in ───────────────────────────────────
# The complaint these came from: after a command finished, the screen showed
# nothing at all. A gate still listening and a gate that had gone deaf looked
# identical, and the window closing was never announced — so "did it work?"
# and "has it hung?" had the same answer on screen.


def _render(**kwargs) -> str:
    """Run the loop and return everything it printed."""
    import io
    import sys

    from app.cli import render

    buffer = io.StringIO()
    real_out, real_file = sys.stdout, render.console.file
    sys.stdout = buffer
    render.console.file = buffer
    try:
        voiceloop.run(None, handle=lambda _c: None, say=lambda _t: None, **kwargs)
    finally:
        sys.stdout, render.console.file = real_out, real_file
    return buffer.getvalue()


def test_the_state_is_shown_after_every_command():
    out = _render(
        gate=_wide_gate(),
        transcripts=["Wake up buddy, open spotify."],
    )
    assert "awake" in out
    assert "s left" in out


def test_a_lapsed_utterance_says_why_nothing_happened():
    """Silence here is the thing being fixed: the user spoke, was plainly
    addressing the assistant, and got no indication that the window had shut."""
    import time

    from app.voice.gate import ConversationGate

    stale = ConversationGate("wake up buddy", follow_up_window_s=12.0)
    stale.wake(time.time() - 600.0)          # it was awake; it no longer is

    out = _render(gate=stale, transcripts=["are you still there"])
    assert "asleep" in out
    assert "wake up buddy" in out


def test_a_scripted_session_is_still_asleep_by_default():
    """Starting awake is for a session a person explicitly asked for. The
    scripted path must keep exercising the asleep-by-default gate, or the
    tests guarding "speech before any wake is never run" stop meaning
    anything."""
    events: list[str] = []
    voiceloop.run(
        None,
        gate=_wide_gate(),
        transcripts=["open spotify", "delete everything"],
        handle=lambda command: events.append(command),
        say=lambda _t: None,
    )
    assert events == []


# ── the clock, watched separately from the microphone ────────────
def test_the_lapse_watch_fires_when_the_window_runs_out():
    """The loop sits blocked in capture, which waits for speech and may wait
    forever, so nothing there notices the window ending."""
    import threading

    fired = threading.Event()
    watch = voiceloop._LapseWatch(fired.set)
    watch.arm(0.05)
    assert fired.wait(timeout=2.0) is True


def test_arming_again_replaces_the_previous_timer():
    """Every command refreshes the window, so a stale timer must not announce
    a lapse that has been pushed back."""
    import threading
    import time

    calls: list[float] = []
    watch = voiceloop._LapseWatch(lambda: calls.append(time.time()))
    watch.arm(0.05)
    watch.arm(0.05)
    watch.arm(0.05)
    time.sleep(0.3)
    assert len(calls) == 1


def test_cancelling_stops_it_firing_at_all():
    import time

    calls: list[int] = []
    watch = voiceloop._LapseWatch(lambda: calls.append(1))
    watch.arm(0.05)
    watch.cancel()
    time.sleep(0.2)
    assert calls == []


def test_a_zero_window_arms_nothing():
    """`remaining()` bottoms out at zero, and a timer with no time left would
    announce the lapse a second time."""
    calls: list[int] = []
    watch = voiceloop._LapseWatch(lambda: calls.append(1))
    watch.arm(0.0)
    assert watch._timer is None


# ── the log the loop leaves behind ───────────────────────────────
# "It ignored me" and "it never heard me" are the two complaints a silent voice
# session produces, and they need different fixes. Only the recorded decision
# tells them apart after the fact.
def _logged(**kwargs) -> list[dict]:
    from app import eventlog

    voiceloop.run(None, handle=lambda _c: None, say=lambda _t: None, **kwargs)
    return eventlog.read(limit=0)


def test_a_session_is_bracketed_in_the_log():
    events = [e["event"] for e in _logged(gate=_wide_gate(), transcripts=[])]
    assert events[0] == "session.start"
    assert events[-1] == "session.end"


def test_every_decision_is_recorded_including_the_ignored_ones():
    entries = _logged(
        gate=_wide_gate(),
        transcripts=["someone else talking", "Wake up buddy, open spotify."],
    )
    decisions = [e["action"] for e in entries if e["event"] == "decision"]
    assert decisions == ["ignore", "run"]


def test_a_recorded_decision_does_not_carry_the_words():
    entries = _logged(gate=_wide_gate(), transcripts=["my card number is 4111"])
    decisions = [e for e in entries if e["event"] == "decision"]
    assert "4111" not in str(decisions)
    assert decisions[0]["speech"]["redacted"] is True


def test_a_completed_dispatch_records_how_long_it_took():
    entries = _logged(gate=_wide_gate(), transcripts=["Wake up buddy, open spotify."])
    done = [e for e in entries if e["event"] == "dispatch.done"]
    assert len(done) == 1
    assert done[0]["took_s"] >= 0


def test_the_state_shown_is_the_state_recorded():
    """A log that disagrees with the screen is worse than no log, because it is
    believed. Both go through `emit`, so they cannot drift."""
    entries = _logged(gate=_wide_gate(), transcripts=["Wake up buddy, open spotify."])
    awake = [e for e in entries if e["event"] == "awake"]
    assert awake and awake[-1]["remaining_s"] > 0


def test_which_of_the_two_watchers_noticed_the_lapse_is_recorded():
    """The clock and the next utterance can both notice. Knowing which did is
    the difference between a stuck timer and a stuck microphone."""
    import time

    from app.voice.gate import ConversationGate

    stale = ConversationGate("wake up buddy", follow_up_window_s=12.0)
    stale.wake(time.time() - 600.0)
    entries = _logged(gate=stale, transcripts=["are you still there"])
    lapsed = [e for e in entries if e["event"] == "lapsed"]
    assert lapsed and lapsed[0]["noticed_by"] == "speech"
