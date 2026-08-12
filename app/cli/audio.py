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
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    try:
        tmp.write(data)
        return Path(tmp.name)
    finally:
        tmp.close()
