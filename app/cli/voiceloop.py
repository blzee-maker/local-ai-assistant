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
from app.voice.gate import ConversationGate
from config import settings

from .repl import Session, run_turn


def _live_transcripts(assistant: Assistant) -> Iterable[str]:
    """Capture, transcribe, and yield one utterance at a time, forever."""
    while True:
        try:
            path = audio.capture_utterance_to_tempfile()
        except audio.RecordingError as exc:
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
    session.speak = True  # a voice conversation is a spoken one
    gate = gate or ConversationGate(settings.wake_phrase, settings.voice_follow_up_seconds)
    handle = handle or (lambda command: run_turn(assistant, session, command))
    say = say or (lambda text: _say(assistant, text))

    name = settings.assistant_name
    console.print(
        f"\n[bot]{name} is listening.[/bot] "
        f"[meta]say “{settings.wake_phrase}” to wake it, then talk. "
        "Ctrl-C to stop.[/meta]"
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
                console.print(f"[meta]{name.lower()} is going back to sleep[/meta]")
                say("Going to sleep.")
            elif decision.action == "run":
                console.print(f"[user]you (voice)[/user] {decision.command}")
                handle(decision.command)
    except KeyboardInterrupt:
        pass

    console.print(f"\n[meta]{name.lower()} stopped listening[/meta]")
