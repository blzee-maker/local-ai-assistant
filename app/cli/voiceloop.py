"""Hands-free voice conversation.

Ties the microphone (`audio`) to the wake gate (`app.voice.gate`) to a spoken
turn (`repl.run_turn`). The capture is one-utterance-at-a-time and strictly
between turns, so the assistant's own voice is never fed back in as the next
command.

Everything the loop *does* — running a command, speaking a short line — is
injectable, so the dispatch logic is tested with scripted transcripts and no
audio hardware or model, the way the gate and the wake matcher are.
"""
from __future__ import annotations

import threading
import time
from typing import Callable, Iterable

from app import eventlog
from app.cli import audio, render
from app.cli.render import console, err_console
from app.core import Assistant
from app.voice import devices
from app.voice.gate import ConversationGate
from config import settings

from .repl import Session, run_turn

# How often to look for the nominated microphone while it is unplugged.
DEVICE_POLL_S = 2.0


def _live_transcripts(assistant: Assistant) -> Iterable[str]:
    """Capture, transcribe, and yield one utterance at a time, forever.

    Listening only happens while the nominated microphone is connected. When it
    is not, the loop waits rather than falling back to the built-in array: the
    whole point of naming a device is that the assistant does not listen to the
    room through whatever Windows happened to select.
    """
    wanted = devices.remembered()
    waiting_announced = False

    while True:
        if wanted is not None:
            device = devices.find(wanted)
            if device is None:
                if not waiting_announced:
                    eventlog.log("device.waiting", wanted=wanted)
                    render.clear_transient()
                    console.print(
                        f"[meta]waiting for {wanted} — not connected, "
                        "so nothing is being listened to[/meta]"
                    )
                    waiting_announced = True
                # Polling, because PortAudio offers no device-change signal and
                # the alternative is a hot-plug callback that does not exist on
                # this platform. Two seconds is imperceptible when putting buds
                # in and costs nothing measurable while waiting.
                time.sleep(DEVICE_POLL_S)
                continue
            if waiting_announced:
                eventlog.log("device.connected", device=device.name, index=device.index)
                render.clear_transient()
                console.print(f"[ok]{device.name} connected — listening again[/ok]")
                waiting_announced = False
            index = device.index
        else:
            index = None

        try:
            # The capture layer prints nothing at all, by design -- it is the
            # CLI's job to render. But with nothing drawn between turns, a loop
            # waiting on speech looks exactly like a loop that has hung, which
            # is the complaint this whole change answers.
            render.transient("listening…")
            started = time.time()
            path = audio.capture_utterance_to_tempfile(device=index)
            capture_s = round(time.time() - started, 2)
        except audio.RecordingError as exc:
            eventlog.log("capture.failed", error=str(exc), waiting=wanted is not None)
            # A device that vanished mid-capture is the ordinary case here, not
            # a failure: buds go back in the case, and the loop should wait for
            # them rather than end the conversation.
            if wanted is not None:
                waiting_announced = False
                time.sleep(DEVICE_POLL_S)
                continue
            render.clear_transient()
            err_console.print(f"[err]{exc}[/err]")
            return
        if path is None:
            render.clear_transient()
            return
        try:
            render.transient("transcribing…")
            started = time.time()
            text = assistant.transcribe(str(path))
            transcribe_s = round(time.time() - started, 2)
        except Exception as exc:
            eventlog.log("transcribe.failed", error=str(exc), capture_s=capture_s)
            render.clear_transient()
            console.print(f"[warn]couldn't transcribe that ({exc})[/warn]")
            continue
        finally:
            render.clear_transient()
            path.unlink(missing_ok=True)
        text = (text or "").strip()
        # Logged whether or not it is used: an utterance transcribed to nothing
        # is the single most confusing thing this loop does, because it looks
        # from outside exactly like never having heard anything at all.
        eventlog.log(
            "heard",
            speech=text,
            capture_s=capture_s,
            transcribe_s=transcribe_s,
            empty=not text,
        )
        if text:
            yield text


def _say(assistant: Assistant, text: str) -> None:
    """Speak a short system line (an acknowledgement, a goodbye). Never fatal."""
    try:
        wav, _summarized = assistant.speak(text)
    except Exception:
        return
    audio.play_wav(wav)


class _LapseWatch:
    """Announces the awake window closing at the moment it closes.

    Needed because the loop spends its time blocked inside
    `capture_utterance_to_tempfile`, which waits for speech and may wait
    forever. Nothing in there watches the clock, so the window used to run out
    in silence and the next thing the user learned about it was that a sentence
    had been ignored — from the screen, indistinguishable from a hang.

    A thread rather than a deadline on the capture call: capture already has a
    `max_seconds` bounding a *single utterance*, and shortening that to poll
    the clock would start cutting people off mid-sentence to fix what is really
    a rendering problem.
    """

    def __init__(self, on_lapse: Callable[[], None]) -> None:
        self._on_lapse = on_lapse
        self._timer: threading.Timer | None = None

    def arm(self, seconds: float) -> None:
        self.cancel()
        if seconds <= 0:
            return
        # daemon, so a pending timer can never hold the process open after
        # Ctrl-C — rule 10 applies to shutting down too.
        self._timer = threading.Timer(seconds, self._on_lapse)
        self._timer.daemon = True
        self._timer.start()

    def cancel(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None


def run(
    assistant: Assistant,
    session: Session | None = None,
    *,
    gate: ConversationGate | None = None,
    transcripts: Iterable[str] | None = None,
    handle: Callable[[str], None] | None = None,
    say: Callable[[str], None] | None = None,
) -> None:
    """Run the hands-free loop until Ctrl-C (or the transcript source ends)."""
    if transcripts is None and not audio.available():
        err_console.print(
            "[err]No microphone available.[/err] "
            "[hint]pip install sounddevice soundfile[/hint]"
        )
        return

    session = session or Session()
    # Voice in, text out, by default. Speaking the reply means holding Piper in
    # memory alongside the transcriber and the language model, and on a machine
    # that sits above 90% used that is the difference between a fallback that
    # rescues an answer and one with nothing left to load into. `--speak` turns
    # it on for anyone with the headroom.
    gate = gate or ConversationGate(settings.wake_phrase, settings.voice_follow_up_seconds)
    handle = handle or (lambda command: run_turn(assistant, session, command))
    say = say or (
        (lambda text: _say(assistant, text)) if session.speak else (lambda _text: None)
    )

    name = settings.assistant_name
    mode = "spoken aloud" if session.speak else "on screen"
    nominated = devices.remembered()
    if nominated:
        console.print(f"[meta]microphone: {nominated} (and no other)[/meta]")
    console.print(
        f"\n[bot]{name} is listening.[/bot] "
        f"[meta]say “{settings.wake_phrase}”, then talk. "
        f"Replies come back {mode}. Ctrl-C to stop.[/meta]"
    )

    source = transcripts if transcripts is not None else _live_transcripts(assistant)
    phrase = settings.wake_phrase
    live = transcripts is None

    def emit(event: str, text: str, style: str = "meta", **fields: object) -> None:
        """Every state change in this loop goes through here.

        One funnel rather than a console.print at each site, so what is shown
        and what is recorded cannot drift apart -- a log that disagrees with the
        screen is worse than no log, because it is believed.
        """
        eventlog.log(event, **fields)
        render.clear_transient()
        console.print(f"[{style}]{text}[/{style}]")

    def show_state() -> None:
        """Say, after every turn, whether anything is still being listened to.

        The loop printed nothing here before, so a finished command and a gate
        that had gone deaf looked identical on screen.
        """
        left = gate.remaining(time.time())
        if left > 0:
            emit(
                "awake",
                f"awake · {left:.0f}s left — just talk, or say “{phrase}” to reset",
                remaining_s=round(left, 1),
            )
        else:
            emit("asleep", f"{name.lower()} is asleep — say “{phrase}”")

    def on_lapse() -> None:
        gate.note_lapse_reported()
        emit(
            "lapsed",
            f"window closed — {name.lower()} is asleep; say “{phrase}”",
            noticed_by="clock",
        )

    watch = _LapseWatch(on_lapse)

    def rearm() -> None:
        # Scripted transcripts drive the clock themselves and finish in
        # microseconds; arming a real timer there would fire into a torn-down
        # test rather than into a terminal.
        if live:
            watch.arm(gate.remaining(time.time()))

    # Typing the command *is* the intent, so the session starts awake instead
    # of asking for the wake phrase immediately afterwards -- which made the
    # user say it twice to start one conversation. The window is unchanged and
    # the state is on screen; the phrase now re-arms a lapsed session rather
    # than opening the first one. Live only, so the scripted path keeps
    # exercising the asleep-by-default gate the tests were written against.
    eventlog.log(
        "session.start",
        mode="voice",
        live=live,
        speak=session.speak,
        device=nominated,
        window_s=gate.window,
    )

    if live:
        gate.wake(time.time())
        show_state()
        rearm()

    try:
        for text in source:
            decision = gate.consider(text, time.time())
            # The decision is recorded for every utterance, including the ones
            # that go nowhere. "It ignored me" and "it never heard me" are the
            # two complaints this log exists to tell apart, and only this line
            # distinguishes them.
            eventlog.log("decision", action=decision.action, speech=text)

            if decision.action == "ignore":
                continue  # not addressed to us, and already said so if it was
            if decision.action == "lapsed":
                watch.cancel()
                emit(
                    "lapsed",
                    f"that came after the window — {name.lower()} is asleep; "
                    f"say “{phrase}”",
                    noticed_by="speech",
                )
                continue
            if decision.action == "acknowledge":
                emit("awake", f"{name.lower()} (awake — go ahead)", style="bot")
                say("Yes?")
                show_state()
                rearm()
            elif decision.action == "sleep":
                watch.cancel()
                emit("sleep", f"{name.lower()} is asleep — say “{phrase}”")
                say("Going to sleep.")
            elif decision.action == "run":
                watch.cancel()  # nothing lapses while a command is running
                render.clear_transient()
                console.print(f"[user]you (voice)[/user] {decision.command}")
                started = time.time()
                try:
                    handle(decision.command)
                except Exception as exc:
                    # A failed turn must not end the conversation, but a turn
                    # that failed silently is exactly what this log is for.
                    eventlog.log("dispatch.failed", error=str(exc),
                                 took_s=round(time.time() - started, 2))
                    raise
                # Named for what it measures: the loop handing off and getting
                # control back. `run_turn` logs the turn itself, and with an
                # injected handler there is no turn at all — two different
                # facts that must not share one event name.
                eventlog.log("dispatch.done", took_s=round(time.time() - started, 2))
                show_state()
                rearm()
    except KeyboardInterrupt:
        eventlog.log("session.interrupted")
    finally:
        watch.cancel()

    eventlog.log("session.end", mode="voice")
    render.clear_transient()
    console.print(f"\n[meta]{name.lower()} stopped listening[/meta]")
