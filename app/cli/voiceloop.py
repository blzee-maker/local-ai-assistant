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

import time
from typing import Callable, Iterable

from app.cli import audio
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
                console.print(f"[ok]{device.name} connected — listening again[/ok]")
                waiting_announced = False
            index = device.index
        else:
            index = None

        try:
            path = audio.capture_utterance_to_tempfile(device=index)
        except audio.RecordingError as exc:
            # A device that vanished mid-capture is the ordinary case here, not
            # a failure: buds go back in the case, and the loop should wait for
            # them rather than end the conversation.
            if wanted is not None:
                waiting_announced = False
                time.sleep(DEVICE_POLL_S)
                continue
            err_console.print(f"[err]{exc}[/err]")
            return
        if path is None:
            return
        try:
            text = assistant.transcribe(str(path))
        except Exception as exc:
            console.print(f"[warn]couldn't transcribe that ({exc})[/warn]")
            continue
        finally:
            path.unlink(missing_ok=True)
        text = (text or "").strip()
        if text:
            yield text


def _say(assistant: Assistant, text: str) -> None:
    """Speak a short system line (an acknowledgement, a goodbye). Never fatal."""
    try:
        wav, _summarized = assistant.speak(text)
    except Exception:
        return
    audio.play_wav(wav)


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

    try:
        for text in source:
            decision = gate.consider(text, time.time())

            if decision.action == "ignore":
                continue  # asleep, or chatter after the window lapsed
            if decision.action == "acknowledge":
                console.print(f"[bot]{name.lower()}[/bot] [meta](awake — go ahead)[/meta]")
                say("Yes?")
            elif decision.action == "sleep":
                console.print(
                    f"[meta]{name.lower()} is asleep — say the wake phrase[/meta]"
                )
                say("Going to sleep.")
            elif decision.action == "run":
                console.print(f"[user]you (voice)[/user] {decision.command}")
                handle(decision.command)
    except KeyboardInterrupt:
        pass

    console.print(f"\n[meta]{name.lower()} stopped listening[/meta]")
