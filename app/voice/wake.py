"""Recognising the wake phrase in a transcript.

There is no wake-word model here. Speech is transcribed and the text is matched,
which costs more CPU than a dedicated detector but buys three things that matter
more on this machine: any phrase can be the wake word (the assistant is called
Buddy, and no pretrained detector knows that name), nothing new has to be
downloaded, and setup never touches the network.

Matching is deliberately forgiving. Whisper returns punctuation and casing of
its own — "hey buddy" comes back as "Hey, Buddy." — and mishears a two-syllable
name in predictable ways. A detector that only fires on an exact string is a
detector that mostly does not fire.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher

# How close each spoken word must be to its target. Applied per word, not to
# the phrase as a whole — see `_words_match` for why. Measured against real
# mishearings; the numbers are recorded in `tests/test_wake.py`.
# 0.75 let "make up buddy" through: "make" against "wake" scores exactly 0.75,
# three characters of four. 0.80 rejects it while every mishearing Whisper
# actually produced still passes — see tests/test_wake.py for the recorded
# transcriptions.
SIMILARITY_THRESHOLD = 0.80

# Words Whisper commonly substitutes for "buddy". Kept explicit rather than
# relying on fuzzy distance alone, because "body" and "buddy" differ by one
# letter while "hey bud" is short enough to score badly against the full phrase.
_HOMOPHONES = {
    "body": "buddy",
    "bodie": "buddy",
    "bud": "buddy",
    "buddie": "buddy",
    "budy": "buddy",
    "buddy's": "buddy",
    "hay": "hey",
    "hai": "hey",
    # "a" was mapped to "hey" here and had to be removed: it turned "a buddy of
    # mine" into the wake phrase exactly. A fold that common is not a mishearing
    # correction, it is a false positive generator.
}

_PUNCTUATION = re.compile(r"[^\w\s]+")


def normalise(text: str) -> str:
    """Lower-case, strip punctuation, and fold known mishearings."""
    cleaned = _PUNCTUATION.sub(" ", (text or "").lower())
    words = [_HOMOPHONES.get(word, word) for word in cleaned.split()]
    return " ".join(words)


def _similar(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def find_wake(text: str, phrase: str) -> int | None:
    """Index just past the wake phrase in `text`, or None if it is not there.

    Returns a position rather than a boolean so the caller can keep whatever was
    said after it: "hey buddy, how much memory is free" is one utterance, and
    making the user pause and repeat themselves would be a worse assistant than
    one that never listened at all.
    """
    words = normalise(text).split()
    target = normalise(phrase).split()
    if not words or not target:
        return None

    span = len(target)
    # Only near the start. A wake word buried mid-sentence is far more likely to
    # be someone talking *about* the assistant than to it.
    for start in range(0, min(len(words), span + 2) - span + 1):
        window = words[start : start + span]
        if len(window) == span and _words_match(window, target):
            return start + span
    return None


def _words_match(heard: list[str], target: list[str]) -> bool:
    """Every word close enough, compared one at a time.

    Comparing the joined strings was tried first and is far too loose: "my
    buddy" scores 0.82 against "hey buddy", because seven of nine characters
    line up. Per word, "my" against "hey" scores 0.40 and is rejected, while a
    genuine mishearing of one word still passes on the strength of the others.
    """
    return all(
        _similar(spoken, wanted) >= SIMILARITY_THRESHOLD
        for spoken, wanted in zip(heard, target)
    )


def heard_wake(text: str, phrase: str) -> bool:
    return find_wake(text, phrase) is not None


def strip_wake(text: str, phrase: str) -> str:
    """What was said after the wake phrase, in the original casing.

    Works on the raw text rather than the normalised copy, so the question keeps
    the capitalisation and punctuation the transcriber produced — the model reads
    it better, and so does the user seeing it echoed.
    """
    index = find_wake(text, phrase)
    if index is None:
        return ""

    # Walk the same number of words in the original, since normalisation never
    # adds or removes words, only rewrites them.
    raw = _PUNCTUATION.sub(" ", text or "").split()
    remainder = " ".join(raw[index:]).strip()
    return remainder
