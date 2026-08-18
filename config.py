"""Central configuration, loaded from environment / .env file.

Everything is read once into a module-level `settings` singleton so the rest of
the app never touches os.environ directly.
"""
from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.knownfolders import user_folders


def _default_file_dirs() -> list[str]:
    """Folders the app is allowed to browse/ingest from (only existing ones matter).

    Resolved through the OS known-folder API rather than joining onto ``~`` — a
    redirected profile (OneDrive Backup) leaves a decoy folder at the old path,
    and joining would point the whole assistant at it. See app/knownfolders.py.
    """
    project_samples = Path(__file__).resolve().parent / "sample_docs"
    return [str(p) for p in user_folders()] + [str(project_samples)]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="ASSISTANT_",
        extra="ignore",
    )

    # What the assistant calls itself. Used in the system prompt and the
    # greeting, so the name is consistent everywhere rather than being a banner
    # that the model itself knows nothing about.
    assistant_name: str = "Buddy"

    # Which LLMEngine implementation the factory builds.
    engine: str = "ollama"

    # Ollama connection + models.
    ollama_host: str = "http://localhost:11434"
    default_model: str = "llama3.2:latest"
    # Used automatically when the default model fails to load (typically out of
    # memory on a small machine). Set to "" to disable the safety net.
    fallback_model: str = "llama3.2:1b"

    # Ollama can route inference to hosted "-cloud" models. Allowing that would
    # send prompts and file contents off the machine, which is the one thing
    # this project promises not to do — so it is refused unless deliberately
    # enabled here. See app/engines/policy.py.
    allow_remote_models: bool = False

    # Generation defaults.
    temperature: float = 0.7
    request_timeout_s: float = 120.0
    # One fixed context window for ALL calls. Changing num_ctx between requests
    # forces Ollama to reload the model (~25s on CPU), so we keep it constant —
    # big enough to hold an injected document, small enough for 8GB.
    num_ctx: int = 8192

    # RAG (Phase 3).
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"  # 384-dim, via ONNX
    embedding_cache_dir: str = "models/embeddings"  # pin model here (survives temp cleanup)
    vector_store_dir: str = "data/vector_store"
    upload_dir: str = "data/uploads"
    chunk_size: int = 700      # characters per chunk
    chunk_overlap: int = 120   # characters shared between adjacent chunks
    rag_top_k: int = 4         # chunks retrieved per query

    # Voice (Phase 4).
    whisper_model: str = "base.en"        # faster-whisper size (tiny.en = lighter)
    whisper_compute_type: str = "int8"    # int8 = smallest memory on CPU
    whisper_cache_dir: str = "models/whisper"
    piper_voice_path: str = "models/piper/en_US-lessac-medium.onnx"
    tts_summary_threshold_chars: int = 400  # longer replies are summarized before speaking

    # Background daemon. Intervals are deliberately unhurried: this machine has
    # little memory headroom, and a scheduler that competes with the user for it
    # is worse than one that notices things a bit later.
    daemon_index_interval_minutes: float = 30.0
    daemon_health_interval_minutes: float = 15.0
    daemon_scan_interval_hours: float = 168.0   # weekly
    daemon_tick_seconds: float = 30.0           # how often due jobs are checked

    # Local file browsing (Phase 4.5). Only these roots are ever exposed.
    allowed_file_dirs: list[str] = Field(default_factory=_default_file_dirs)
    file_search_max_results: int = 25
    file_search_scan_limit: int = 20000  # cap files walked so a huge folder can't stall


settings = Settings()
