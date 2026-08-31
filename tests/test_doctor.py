"""Tests for the diagnostic command.

Run as a subprocess with its output piped, deliberately. That is the exact
condition that broke it: Windows hands a piped stdout the locale codepage, so
the tick in "check passed" was unencodable and `assistant doctor | tail` died
with a UnicodeEncodeError while the same command in a console worked. Importing
the module and calling the function would never have caught it (rule 12).

One run of the real environment, shared by every test that reads it: doctor
talks to Ollama and reads the disk, so it costs a few seconds and there is
nothing to gain from repeating it. A second, deliberately broken run
(`degraded_run`) covers the one arrangement a working machine cannot produce.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def doctor_run() -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "assistant.py"), "doctor"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(PROJECT_ROOT),
        timeout=180,
    )


def test_doctor_survives_having_its_output_redirected(doctor_run):
    """The diagnostic tool crashing is worse than no diagnostic tool."""
    combined = doctor_run.stdout + doctor_run.stderr

    assert "UnicodeEncodeError" not in combined
    assert "Traceback" not in combined
    # 0 = everything passed, 1 = a check failed. Both are valid outcomes; a
    # crash is not.
    assert doctor_run.returncode in (0, 1), combined[-600:]


def test_non_ascii_output_survives_the_pipe(doctor_run):
    """The characters that caused the crash must still be arriving."""
    assert "✓" in doctor_run.stdout or "✗" in doctor_run.stdout


def test_a_warning_is_never_summarised_as_all_good(doctor_run):
    """It reported "All good." on a run that had just said the fallback model
    was missing. A summary contradicting the lines above it teaches people to
    stop reading the lines."""
    output = doctor_run.stdout
    if "!" in output:
        assert "All good." not in output
        assert "warning(s)" in output


@pytest.fixture(scope="module")
def degraded_run(tmp_path_factory) -> subprocess.CompletedProcess:
    """A run with a failed check AND a warning — the combination this machine
    cannot produce.

    Locally Ollama is up and the models are on disk, so doctor either passes
    outright or warns; CI has neither a model server nor an audio device, and
    that is the one arrangement where the summary dropped the warnings. Forced
    here through the real command rather than by calling the function, for the
    reason the module docstring gives.
    """
    env = {
        **os.environ,
        # Port 9 (discard) is closed, so the health check is refused rather
        # than hanging on a timeout.
        "ASSISTANT_OLLAMA_HOST": "http://127.0.0.1:9",
        # An empty directory holds no .bin, so the STT check warns.
        "ASSISTANT_WHISPER_CACHE_DIR": str(tmp_path_factory.mktemp("no-whisper")),
    }
    return subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "assistant.py"), "doctor"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(PROJECT_ROOT),
        env=env,
        timeout=180,
    )


def test_a_failed_check_does_not_swallow_the_warnings(degraded_run):
    """CI failed on exactly this: Ollama down and no audio device printed
    "Some checks failed." and named none of the three warnings above it. The
    summary short-circuited on the failure, so the only way to see the rest was
    to fix the server and run again."""
    output = degraded_run.stdout

    assert "Some checks failed." in output, output
    assert "!" in output, output      # the run really did warn
    assert "warning(s)" in output, output


def test_speech_to_text_is_checked(doctor_run):
    """TTS and the microphone were checked while the transcriber was not, even
    though it downloads ~140 MB on first use and everything spoken depends on
    it."""
    assert "STT model" in doctor_run.stdout


def test_the_fallback_model_is_reported_either_way(doctor_run):
    """It is only a safety net if it is actually on disk, and this machine runs
    close enough to its memory limit to need one."""
    # Only when the engine answered. With Ollama down, doctor never reaches the
    # model checks at all, and asserting unconditionally tests the environment
    # rather than the code.
    if "unreachable" in doctor_run.stdout:
        pytest.skip("model server is not running")
    assert "fallback model" in doctor_run.stdout


# ── the encoding fix itself ──────────────────────────────────────
def test_forcing_utf8_tolerates_a_stream_that_cannot_be_reconfigured():
    """pythonw has no stdout at all, and the daemon runs under it. Degrade,
    never crash (rule 10)."""
    from app.cli import render

    class Stubborn:
        def reconfigure(self, **_kwargs):
            raise ValueError("detached")

    original_out, original_err = sys.stdout, sys.stderr
    try:
        sys.stdout = sys.stderr = Stubborn()  # type: ignore[assignment]
        render._force_utf8()  # must not raise
    finally:
        sys.stdout, sys.stderr = original_out, original_err


def test_forcing_utf8_tolerates_a_stream_with_no_reconfigure_at_all():
    from app.cli import render

    original_out, original_err = sys.stdout, sys.stderr
    try:
        sys.stdout = sys.stderr = object()  # type: ignore[assignment]
        render._force_utf8()
    finally:
        sys.stdout, sys.stderr = original_out, original_err
