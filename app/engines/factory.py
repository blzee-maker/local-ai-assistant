"""Builds the configured LLMEngine.

This is the ONLY place that knows which concrete engine exists. To add a new
backend (e.g. llama-cpp-python), implement LLMEngine and add a branch here — no
caller changes.
"""
from __future__ import annotations

from config import settings

from .base import LLMEngine
from .ollama_engine import OllamaEngine


def build_engine() -> LLMEngine:
    name = settings.engine.lower()

    if name == "ollama":
        return OllamaEngine(
            host=settings.ollama_host,
            default_model=settings.default_model,
            temperature=settings.temperature,
            timeout_s=settings.request_timeout_s,
            num_ctx=settings.num_ctx,
        )

    # Future: elif name == "llamacpp": return LlamaCppEngine(...)

    raise ValueError(
        f"Unknown engine '{settings.engine}'. Valid options: 'ollama'."
    )
