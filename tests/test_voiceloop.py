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
