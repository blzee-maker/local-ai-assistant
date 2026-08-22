"""Tests for recognising the wake phrase.

The transcriptions quoted here are real: Piper spoke each phrase, Whisper
base.en transcribed it, and these are what came back. Guessing at what a
transcriber returns is how a threshold gets tuned against an imagination
instead of against the machine.

The asymmetry matters. A missed wake phrase costs one repeated sentence. A false
one wakes the assistant while you are talking to somebody else, so the cases
below lean heavily on things that must *not* trigger it.
"""
from __future__ import annotations

import pytest

from app.voice.wake import (
    SIMILARITY_THRESHOLD,
    find_wake,
    heard_wake,
    normalise,
    strip_wake,
)

PHRASE = "wake up buddy"


# ── what Whisper actually returned ───────────────────────────────
@pytest.mark.parametrize(
    "heard",
    [
        "Wake up buddy.",                                        # spoken plainly
        "Wake up buddy, how much memory is in use.",             # with a question
        "Wake up buddy, open Spotify.",
        "Wake up buddy what is taking up my disc space.",         # "disk" -> "disc"
        "wake up buddy",                                          # no punctuation
        "Wake up, buddy!",
    ],
)
def test_the_phrase_is_recognised_as_transcribed(heard):
    assert heard_wake(heard, PHRASE), heard


@pytest.mark.parametrize(
    "heard",
    [
        "The weather is nice today.",
        "I told my buddy about it.",
        "Hey there, how are you?",
        "Let me check the memory usage.",
        "A buddy of mine works there.",
        "Okay so what is next?",
        "I need to wake up early tomorrow.",
        "Make up your mind.",
        "Wake up, it is late.",
    ],
)
def test_ordinary_speech_does_not_wake_it(heard):
    """All nine were spoken and transcribed for real; none may trigger."""
    assert not heard_wake(heard, PHRASE), heard


# ── why the matcher works the way it does ────────────────────────
def test_matching_is_per_word_not_per_phrase():
    """Comparing joined strings was tried and is far too loose: with a two-word
    phrase, "my buddy" scored 0.82 against "hey buddy" because seven of nine
    characters line up. Per word, "my" against "hey" scores 0.40."""
    assert not heard_wake("my buddy", "hey buddy")
    assert not heard_wake("I saw my buddy", "hey buddy")


def test_the_threshold_rejects_a_near_miss_that_0_75_allowed():
    """"make" against "wake" is three characters of four: exactly 0.75. That let
    "make up buddy" through, which is why the threshold is 0.80."""
    from difflib import SequenceMatcher

    assert SequenceMatcher(None, "make", "wake").ratio() == pytest.approx(0.75)
    assert SIMILARITY_THRESHOLD > 0.75
    assert not heard_wake("make up buddy", PHRASE)


def test_a_common_mishearing_of_the_name_still_wakes_it():
    """Whisper writes "buddy" as "body" often enough to matter, and a two
    syllable name is exactly what a transcriber guesses at."""
    assert heard_wake("Wake up body.", PHRASE)
    assert heard_wake("wake up bud", PHRASE)


def test_the_phrase_must_come_first():
    """A wake word in the middle of a sentence is far more likely to be someone
    talking *about* the assistant than to it."""
    assert not heard_wake("I was going to wake up buddy later", PHRASE)


def test_a_partial_phrase_is_not_enough():
    assert not heard_wake("wake up", PHRASE)
    assert not heard_wake("buddy", PHRASE)
    assert not heard_wake("buddy wake up", PHRASE)


# ── keeping what was said after it ───────────────────────────────
def test_the_question_after_the_phrase_survives():
    """One utterance, not two. Making someone say the wake word and then wait
    to be asked again is worse than not listening at all."""
    assert (
        strip_wake("Wake up buddy, how much memory is in use.", PHRASE)
        == "how much memory is in use"
    )
    assert strip_wake("Wake up buddy, open Spotify.", PHRASE) == "open Spotify"


def test_the_bare_phrase_leaves_nothing_behind():
    assert strip_wake("Wake up buddy.", PHRASE) == ""


def test_casing_is_preserved_for_the_model():
    """The question goes on to the model and back to the user, so it keeps the
    capitalisation the transcriber produced rather than the folded copy."""
    assert "Spotify" in strip_wake("Wake up buddy, open Spotify.", PHRASE)


# ── the pieces ───────────────────────────────────────────────────
def test_normalising_folds_punctuation_and_known_mishearings():
    assert normalise("Wake up, Buddy!") == "wake up buddy"
    assert normalise("Wake up body.") == "wake up buddy"


def test_empty_input_is_never_a_wake():
    assert find_wake("", PHRASE) is None
    assert find_wake("   ", PHRASE) is None
    assert find_wake("wake up buddy", "") is None


def test_the_configured_phrase_matches_the_terminal_command():
    """One phrase to remember, not two: `wake up buddy` starts it from a shell
    and from the microphone."""
    from config import settings

    assert settings.wake_phrase == "wake up buddy"
    assert heard_wake("Wake up buddy.", settings.wake_phrase)


# ── silence must cost nothing, and invent nothing ────────────────
def test_transcription_is_gated_by_voice_activity_detection():
    """Without it, three seconds of silence took a full second to decode and
    came back as the word "You" — an invented utterance, which for an
    always-listening loop means waking at nothing. With it: 0.09s and ""."""
    import inspect

    from app.voice.stt import WhisperSTT

    source = inspect.getsource(WhisperSTT.transcribe)
    assert "vad_filter=True" in source
