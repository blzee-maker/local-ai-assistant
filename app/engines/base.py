"""The LLMEngine abstraction — the single seam between the app and whatever
actually runs the model.

The rest of the application (chat API, RAG, voice) depends ONLY on the types in
this module. Swapping Ollama for llama-cpp-python, or any other backend, means
writing one new class that satisfies `LLMEngine` and pointing the factory at it —
no other code changes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Protocol, runtime_checkable


@dataclass
class ChatMessage:
    """A single turn in a conversation."""

    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass
class StreamEvent:
    """One event emitted while streaming a response.

    Token events carry text. The final event has ``done=True`` and carries the
    metrics we surface in the UI (time-to-first-token, tokens/sec, counts).
    """

    token: str = ""
    done: bool = False

    # Populated only on the final (done) event.
    model: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None

    # True when the primary model could not be loaded and a smaller fallback
    # answered instead. Surfaced to the user: silently swapping in a weaker
    # model would make a worse answer look like the model's best effort.
    fell_back: bool = False

    # Client-side: wall-clock latency the user actually feels (includes any
    # cold model load on the first call).
    time_to_first_token_s: float | None = None

    # Server-side timings reported by the backend, cleanly separated so a cold
    # model load never contaminates the generation-throughput number.
    load_duration_s: float | None = None        # loading the model into RAM
    prompt_eval_duration_s: float | None = None  # prefill / prompt processing
    eval_duration_s: float | None = None         # token generation only
    total_duration_s: float | None = None

    @property
    def tokens_per_second(self) -> float | None:
        """Decode throughput: completion tokens over *generation* time only.

        Uses the backend's reported eval_duration (generation window), so model
        load and prompt prefill never distort it.
        """
        if self.completion_tokens and self.eval_duration_s and self.eval_duration_s > 0:
            return self.completion_tokens / self.eval_duration_s
        return None


@dataclass
class GenerationOptions:
    """Per-request knobs. All optional; None means "use the engine default"."""

    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    num_ctx: int | None = None  # context window; raise when injecting file text
    stop: list[str] = field(default_factory=list)


@dataclass
class ToolCall:
    """A tool invocation the model decided to make."""

    name: str
    arguments: dict


@dataclass
class AssistantTurn:
    """Result of a non-streaming turn: free text plus any tool calls."""

    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)


@runtime_checkable
class LLMEngine(Protocol):
    """Contract every inference backend must satisfy."""

    def health_check(self) -> bool:
        """Return True if the backend is reachable and ready to serve."""
        ...

    def chat(
        self,
        messages: list[ChatMessage],
        tools: list[dict] | None = None,
        options: GenerationOptions | None = None,
    ) -> AssistantTurn:
        """Non-streaming turn. If `tools` are given, the model may return
        tool calls instead of (or alongside) text."""
        ...

    def list_models(self) -> list[str]:
        """Return the names of locally available models."""
        ...

    def chat_stream(
        self,
        messages: list[ChatMessage],
        options: GenerationOptions | None = None,
    ) -> Iterator[StreamEvent]:
        """Stream a chat completion as a sequence of StreamEvents.

        Yields token events as text arrives, then exactly one final event with
        ``done=True`` and populated metrics.
        """
        ...
