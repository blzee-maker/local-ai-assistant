"""VoiceService — lazy-loading holder for the STT and TTS engines.

On an 8GB machine, RAM is scarce. Neither model is loaded until it is first
used: TTS loads on the first spoken reply, STT only if the user actually uses
the microphone. Once loaded, an engine stays resident.
"""
from __future__ import annotations

from config import settings

from .stt import STTEngine, WhisperSTT
from .tts import PiperTTS, TTSEngine


class VoiceService:
    def __init__(self) -> None:
        self._tts: TTSEngine | None = None
        self._stt: STTEngine | None = None

    @property
    def tts(self) -> TTSEngine:
        if self._tts is None:
            self._tts = PiperTTS(settings.piper_voice_path)
        return self._tts

    @property
    def stt(self) -> STTEngine:
        if self._stt is None:
            self._stt = WhisperSTT(
                settings.whisper_model,
                settings.whisper_compute_type,
                settings.whisper_cache_dir,
            )
        return self._stt

    def synthesize(self, text: str) -> bytes:
        return self.tts.synthesize_wav(text)

    def transcribe(self, audio_path: str) -> str:
        return self.stt.transcribe(audio_path)
