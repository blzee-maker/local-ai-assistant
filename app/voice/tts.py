"""Text-to-speech behind a swappable TTSEngine seam.

Piper runs a small ONNX voice model fully offline — the right fit for an edge
target. Swapping to another engine (e.g. the OS SAPI5 voices) means writing one
class that satisfies TTSEngine.
"""
from __future__ import annotations

import io
import wave
from typing import Protocol, runtime_checkable


@runtime_checkable
class TTSEngine(Protocol):
    def synthesize_wav(self, text: str) -> bytes:
        """Return spoken `text` as the bytes of a WAV file."""
        ...


class PiperTTS(TTSEngine):
    def __init__(self, voice_path: str) -> None:
        from piper import PiperVoice  # imported lazily so startup stays cheap

        self._voice = PiperVoice.load(voice_path)

    def synthesize_wav(self, text: str) -> bytes:
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav:
            self._voice.synthesize_wav(text, wav)
        return buffer.getvalue()
