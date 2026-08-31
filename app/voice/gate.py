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

Action = Literal["ignore", "run", "acknowledge", "sleep", "lapsed"]


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
        # Whether the window has been open and has since run out. Only then is
        # a missed utterance worth reporting: "you spoke too late" and "that
        # was not addressed to me" are different facts, and the loop used to
        # render both as the same silence.
        self._lapse_unreported = False

    @property
    def window(self) -> float:
        """How long a wake lasts, in seconds.

        Exposed so the log can record the window actually in force rather than
        re-reading the configured default — those are the same number in
        production and different ones anywhere a gate is constructed directly,
        and a log that contradicts itself two lines later is worse than one
        that says nothing.
        """
        return self._window

    def is_awake(self, now: float) -> bool:
        return now < self._awake_until

    def remaining(self, now: float) -> float:
        """Seconds left in the window; 0.0 when asleep.

        Exposed because the loop has to *show* the state — an assistant that
        has gone deaf while the screen looks identical to one that is listening
        is the failure this whole change exists to fix.
        """
        return max(0.0, self._awake_until - now)

    def wake(self, now: float) -> None:
        """Open the window without a spoken phrase.

        For an explicitly invoked session: typing `/listen` is the intent, and
        demanding the wake phrase straight afterwards asks the same question
        twice. The window is still bounded and still announced -- this changes
        who opens it, not how long it stays open.
        """
        self._awake_until = now + self._window
        self._lapse_unreported = True

    def note_lapse_reported(self) -> None:
        """Record that the window closing has already been announced.

        The loop watches the clock as well as the transcripts, because it can
        sit blocked on the microphone for minutes and the user needs to be told
        the moment listening stops -- not whenever they next happen to speak.
        Whichever of the two notices it first suppresses the other, so the fact
        is reported once.
        """
        self._lapse_unreported = False

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
            self._lapse_unreported = True
            command = wake.strip_wake(text, self._phrase)
            return Decision("run", command) if command else Decision("acknowledge")

        if self.is_awake(now):
            if is_sleep_command(text):
                self._awake_until = 0.0
                # Sleeping on request is already announced by the loop; saying
                # the window lapsed as well would report one event twice.
                self._lapse_unreported = False
                return Decision("sleep")
            # A genuine follow-up — keep the conversation alive.
            self._awake_until = now + self._window
            return Decision("run", text)

        # Asleep, and it wasn't the wake phrase. Say so once if the window had
        # been open, because the speaker was plainly talking to us and deserves
        # to know why nothing happened. After that, silence: a gate that
        # narrates every overheard sentence is a gate nobody leaves running.
        if self._lapse_unreported:
            self._lapse_unreported = False
            return Decision("lapsed")
        return Decision("ignore")
