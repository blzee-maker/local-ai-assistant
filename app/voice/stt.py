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
        # vad_filter is not a speed optimisation, it is a correctness one.
        # Without it, three seconds of silence took a full second to decode and
        # came back as the word "You" — an invented utterance, which for an
        # always-listening loop means waking up at nothing. Silero VAD ships
        # inside faster-whisper, so this costs no new dependency.
        segments, _info = self._model.transcribe(
            audio_path, beam_size=1, vad_filter=True
        )
        return "".join(segment.text for segment in segments).strip()
