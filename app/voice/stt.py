"""Speech-to-text behind a swappable STTEngine seam.

faster-whisper runs Whisper on CTranslate2 with int8 quantization — small memory
footprint, no GPU required. It decodes browser audio (webm/opus, wav, …) via the
bundled PyAV, so uploaded recordings can be transcribed directly.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class STTEngine(Protocol):
    def transcribe(self, audio_path: str) -> str:
        """Return the text spoken in the audio file at `audio_path`."""
        ...


class WhisperSTT(STTEngine):
    def __init__(
        self,
        model_size: str,
        compute_type: str = "int8",
        cache_dir: str | None = None,
    ) -> None:
        from faster_whisper import WhisperModel  # lazy import

        self._model = WhisperModel(
            model_size,
            device="cpu",
            compute_type=compute_type,
            download_root=cache_dir,
        )

    def transcribe(self, audio_path: str) -> str:
        segments, _info = self._model.transcribe(audio_path, beam_size=1)
        return "".join(segment.text for segment in segments).strip()
