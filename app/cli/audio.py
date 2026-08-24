"""Microphone capture and speaker playback for the terminal.

The browser used to do both of these jobs. Replacing it means talking to the
sound card directly, which is the one place this project needs a native
dependency (PortAudio, via `sounddevice`).

Audio hardware is the least predictable part of any desktop app — no device, no
driver, exclusive-mode conflicts, a headless CI box. So every entry point here
degrades instead of raising: `available()` is checked before offering voice, and
playback falls back to the Windows stdlib player if PortAudio is missing.
"""
from __future__ import annotations

import io
import queue
import sys
import tempfile
import wave
from pathlib import Path

# Whisper resamples internally, but 16 kHz mono is its native rate — recording
# straight into it avoids a needless resample and keeps the file small.
SAMPLE_RATE = 16_000
CHANNELS = 1


def _sounddevice():
    """Import sounddevice lazily; None when PortAudio isn't usable."""
    try:
        import sounddevice as sd

        return sd
    except Exception:
        return None


def available() -> bool:
    """True if we can both capture and play audio."""
    sd = _sounddevice()
    if sd is None:
        return False
    try:
        sd.query_devices(kind="input")
        sd.query_devices(kind="output")
        return True
    except Exception:
        return False


def input_device_name() -> str | None:
    sd = _sounddevice()
    if sd is None:
        return None
    try:
        return str(sd.query_devices(kind="input")["name"]).strip()
    except Exception:
        return None


# ── playback ─────────────────────────────────────────────────────
def play_wav(data: bytes) -> None:
    """Play WAV bytes, blocking until finished. Ctrl-C stops playback only."""
    sd = _sounddevice()
    if sd is None:
        _play_wav_fallback(data)
        return

    import soundfile as sf

    try:
        samples, rate = sf.read(io.BytesIO(data), dtype="float32")
        sd.play(samples, rate)
        try:
            sd.wait()
        except KeyboardInterrupt:
            sd.stop()
    except Exception:
        _play_wav_fallback(data)


def _play_wav_fallback(data: bytes) -> None:
    """Windows-only stdlib playback, used when PortAudio is unavailable."""
    if sys.platform != "win32":
        return
    import winsound

    tmp = Path(tempfile.gettempdir()) / "assistant_tts.wav"
    tmp.write_bytes(data)
    try:
        winsound.PlaySound(str(tmp), winsound.SND_FILENAME)
    finally:
        tmp.unlink(missing_ok=True)


# ── capture ──────────────────────────────────────────────────────
class RecordingError(RuntimeError):
    """Raised when the microphone cannot be opened."""


def record_until(stop_signal, max_seconds: float = 120.0) -> bytes:
    """Record mono 16 kHz audio until `stop_signal()` returns, as WAV bytes.

    `stop_signal` is a blocking callable (typically `input`) — recording runs on
    PortAudio's own thread meanwhile, so the caller just waits for the user.
    """
    sd = _sounddevice()
    if sd is None:
        raise RecordingError(
            "No audio backend. Install PortAudio support: pip install sounddevice"
        )

    chunks: "queue.Queue[bytes]" = queue.Queue()

    def callback(indata, _frames, _time, status):  # PortAudio thread
        if status:  # overflow/underflow — drop the frame, keep recording
            return
        chunks.put(bytes(indata))

    try:
        stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="int16",
            callback=callback,
        )
    except Exception as exc:
        raise RecordingError(f"Could not open the microphone: {exc}") from exc

    with stream:
        try:
            stop_signal()
        except (KeyboardInterrupt, EOFError):
            pass

    frames: list[bytes] = []
    total = 0
    max_bytes = int(max_seconds * SAMPLE_RATE * 2)  # int16 == 2 bytes/sample
    while not chunks.empty():
        chunk = chunks.get()
        frames.append(chunk)
        total += len(chunk)
        if total >= max_bytes:
            break

    if not frames:
        raise RecordingError("No audio captured — is the microphone muted?")

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(CHANNELS)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(b"".join(frames))
    return buffer.getvalue()


def record_to_tempfile(stop_signal) -> Path:
    """Record and persist to a temp .wav — faster-whisper reads from a path."""
    data = record_until(stop_signal)
    return _persist(data)


def _persist(data: bytes) -> Path:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    try:
        tmp.write(data)
        return Path(tmp.name)
    finally:
        tmp.close()


def _frames_to_wav(frames: list[bytes]) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(CHANNELS)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(b"".join(frames))
    return buffer.getvalue()


# ── hands-free capture (one utterance at a time) ─────────────────
#
# These numbers are a starting point measured to be sane, not a tuning against
# this exact microphone — per rule 11 they want confirming on real hardware. The
# design is forgiving of loose values on purpose: faster-whisper's own VAD drops
# non-speech from whatever we hand it, and the wake phrase must still match, so
# an over-generous segment costs one wasted transcription, never a false action.
_FRAME = 480                    # samples per block — 30 ms at 16 kHz
_CALIBRATION_FRAMES = 15        # ~0.45 s sampled to estimate the room's noise floor
_THRESHOLD_MULT = 3.0           # speech must be this much louder than that floor
_MIN_THRESHOLD = 350.0          # …but never trust a floor of near-silence as the bar
_ONSET_FRAMES = 2               # consecutive loud frames before we call it speech
_PREROLL_FRAMES = 8             # ~0.24 s kept before onset so the first word isn't clipped
_SILENCE_HANG_FRAMES = 25       # ~0.75 s of quiet ends the utterance
_MIN_SPEECH_FRAMES = 8          # shorter than ~0.24 s is a cough, not a sentence


def _rms(frame: bytes) -> float:
    import numpy as np

    samples = np.frombuffer(frame, dtype=np.int16).astype(np.float32)
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(samples * samples)))


def capture_one_utterance(max_seconds: float = 15.0) -> bytes | None:
    """Listen until someone speaks, capture that one utterance, and return it as
    WAV bytes. Blocks (waiting for speech) until interrupted.

    Only one utterance, then it returns — the caller closes the loop after
    processing, so the microphone is *not* open while the assistant is talking.
    Leaving it open would feed Buddy's own spoken reply straight back in as the
    next command.

    Returns None only if the stream produced nothing usable.
    """
    import collections

    sd = _sounddevice()
    if sd is None:
        raise RecordingError(
            "No audio backend. Install PortAudio support: pip install sounddevice"
        )

    chunks: "queue.Queue[bytes]" = queue.Queue()

    def callback(indata, _frames, _time, status):  # PortAudio thread
        if status:
            return
        chunks.put(bytes(indata))

    try:
        stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="int16",
            blocksize=_FRAME,
            callback=callback,
        )
    except Exception as exc:
        raise RecordingError(f"Could not open the microphone: {exc}") from exc

    preroll: "collections.deque[bytes]" = collections.deque(maxlen=_PREROLL_FRAMES)
    noise: list[float] = []
    threshold: float | None = None
    speaking = False
    voiced: list[bytes] = []
    loud_run = 0
    silence_run = 0
    speech_frames = 0
    max_frames = int(max_seconds * SAMPLE_RATE / _FRAME)

    with stream:
        while True:
            try:
                frame = chunks.get(timeout=0.2)  # short wait keeps Ctrl-C responsive
            except queue.Empty:
                continue

            level = _rms(frame)

            # Estimate the room's noise floor before judging anything against it.
            if threshold is None:
                noise.append(level)
                preroll.append(frame)
                if len(noise) >= _CALIBRATION_FRAMES:
                    floor = sorted(noise)[len(noise) // 2]  # median resists a stray bang
                    threshold = max(floor * _THRESHOLD_MULT, _MIN_THRESHOLD)
                continue

            if not speaking:
                preroll.append(frame)
                loud_run = loud_run + 1 if level >= threshold else 0
                if loud_run >= _ONSET_FRAMES:
                    speaking = True
                    voiced = list(preroll)  # keep the run-up so the onset isn't clipped
                    silence_run = 0
                    speech_frames = loud_run
                continue

            voiced.append(frame)
            if level >= threshold:
                silence_run = 0
                speech_frames += 1
            else:
                silence_run += 1

            if silence_run >= _SILENCE_HANG_FRAMES:
                if speech_frames >= _MIN_SPEECH_FRAMES:
                    return _frames_to_wav(voiced)
                # Too short to be speech — a click or a cough. Reset and wait.
                speaking = False
                voiced = []
                preroll.clear()
                loud_run = silence_run = speech_frames = 0
            elif len(voiced) >= max_frames:
                return _frames_to_wav(voiced)


def capture_utterance_to_tempfile(max_seconds: float = 15.0) -> Path | None:
    """`capture_one_utterance`, written to a temp .wav for faster-whisper."""
    data = capture_one_utterance(max_seconds=max_seconds)
    return _persist(data) if data else None
