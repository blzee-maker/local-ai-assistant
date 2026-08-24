"""The awake/asleep state machine for hands-free conversation.

Kept separate from `wake.py` (which only matches a phrase in a string) and from
the CLI loop (which owns the microphone) so the *decision* — should this
transcript be acted on? — is a pure function of state and can be tested without
audio hardware or a model, the way `wake.py` is.

The rule that shapes it: asleep, only the wake phrase gets through; awake, any
utterance is a command, but only for a short window. While awake the assistant
will act on whatever it hears — the television, someone else in the room — so
the window is deliberately brief and every command refreshes it. An absent user
is not consent.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from app.voice import wake

# Spoken ways to end the conversation early instead of waiting for the window to
# lapse. Normalised once at import so matching folds the same mishearings the
# wake phrase does ("that's all" → "that s all").
_SLEEP_PHRASES = (
    "go to sleep",
    "go back to sleep",
    "goodbye buddy",
    "bye buddy",
    "stop listening",
    "never mind",
    "nevermind",
    "that is all",
    # `normalise` turns an apostrophe into a space, so "that's all" arrives as
    # "that s all" — the spoken form has to be listed the way it actually lands.
    "that s all",
)
_SLEEP_NORMALISED = tuple(wake.normalise(p) for p in _SLEEP_PHRASES)

Action = Literal["ignore", "run", "acknowledge", "sleep"]


@dataclass
class Decision:
    """What the loop should do with one transcript."""

    action: Action
    command: str = ""


def is_sleep_command(text: str) -> bool:
    normalised = wake.normalise(text)
    return any(phrase and phrase in normalised for phrase in _SLEEP_NORMALISED)


class ConversationGate:
    def __init__(self, phrase: str, follow_up_window_s: float) -> None:
        self._phrase = phrase
        self._window = follow_up_window_s
        self._awake_until: float = 0.0

    def is_awake(self, now: float) -> bool:
        return now < self._awake_until

    def consider(self, transcript: str, now: float) -> Decision:
        """Decide what to do with `transcript`, updating the awake window.

        `now` is passed in rather than read from the clock so a test can drive
        the window deterministically.
        """
        text = (transcript or "").strip()
        if not text:
            return Decision("ignore")

        # The wake phrase always gets through, awake or not, and (re)opens the
        # window. Anything said after it in the same breath is the command.
        if wake.heard_wake(text, self._phrase):
            self._awake_until = now + self._window
            command = wake.strip_wake(text, self._phrase)
            return Decision("run", command) if command else Decision("acknowledge")

        if self.is_awake(now):
            if is_sleep_command(text):
                self._awake_until = 0.0
                return Decision("sleep")
            # A genuine follow-up — keep the conversation alive.
            self._awake_until = now + self._window
            return Decision("run", text)

        # Asleep and it wasn't the wake phrase: not for us.
        return Decision("ignore")
