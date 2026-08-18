
"""The Assistant orchestrator — everything the app *does*, with no UI attached.

This layer owns the behaviour that used to live inside FastAPI route handlers:
deciding whether a message wants a local file, grounding the prompt on retrieved
context, and condensing long replies before they are spoken. A front end (the
CLI, a future HTTP server, anything) drives it and renders the events it yields.

Two deliberate properties:

* **Nothing is loaded until it is needed.** The embedding model and the voice
  models are lazy, so `assistant models` or `assistant doctor` start instantly
  instead of paying for a FAISS + ONNX load they never use.
* **The caller's history is never mutated.** Grounding rewrites the last user
  message (injecting file text or retrieved context). The old web flow got away
  with doing that in place because the browser kept its own copy; a CLI holds
  one list for the whole session, so in-place rewriting would permanently
  overwrite what the user actually typed and poison every later turn. Every
  turn here works on a deep copy.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Literal

from app import files as filesvc
from app.engines import ChatMessage, GenerationOptions, build_engine
from app.engines.base import LLMEngine
from app.tools import Dispatch, ToolContext, ToolRegistry, default_tools
from config import settings

# The name lives in the prompt, not just the banner: a greeting that calls it
# Buddy while the model itself has never heard the name produces the obvious
# awkwardness the first time the user asks "what are you called?".
DEFAULT_SYSTEM_PROMPT = (
    "You are {name}, a helpful and concise assistant running fully offline on "
    "the user's own machine. You can read their documents, analyse their disk, "
    "report on system health, end processes they ask you to, and remember "
    "things they tell you to remember. Be direct and accurate. If you are "
    "unsure, say so. Never claim you lack a capability you actually have."
)


def system_prompt() -> str:
    return DEFAULT_SYSTEM_PROMPT.format(name=settings.assistant_name)


RAG_PROMPT_TEMPLATE = (
    "Answer the question using ONLY the context below. If the answer is not in "
    "the context, say you don't know based on the provided documents. Where "
    "relevant, cite sources inline like [1], [2].\n\n"
    "Context:\n{context}\n\nQuestion: {question}"
)

# How much of a file's text is injected into the prompt. Bounded so a large
# document cannot blow past num_ctx and silently truncate the conversation.
FILE_INJECTION_CHARS = 8000

# Tools that read the user's files, switched off by `--no-files` / `/files off`.
# Only these — that flag is about file access, not about muting every capability.
FILE_ACCESS_TOOLS = {"open_local_file"}


@dataclass
class AssistantEvent:
    """One event in a streamed turn: many `token`s, then exactly one terminal event."""

    type: Literal["token", "done", "error"]
    text: str = ""
    sources: list[dict[str, Any]] = field(default_factory=list)
    # What a tool did this turn, if anything. Named generically because the
    # registry means this is no longer always a file (rule 4: an action that ran
    # silently is indistinguishable from one that did not).
    tool_used: dict[str, Any] | None = None
    # Which remembered facts were injected this turn. Shown for the same reason:
    # an assistant that quietly consults a private dossier about you is worse
    # than one that says which facts it used.
    recalled: dict[str, Any] | None = None
    metrics: dict[str, Any] | None = None
    error: str | None = None


class Assistant:
    def __init__(
        self,
        engine: LLMEngine | None = None,
        tools: ToolRegistry | None = None,
    ) -> None:
        self._engine = engine or build_engine()
        self._rag: Any = None
        self._voice: Any = None
        self._memory: Any = None
        self._tools = tools
        self._tools_ready = tools is not None

    # ── lazily-built subsystems ──────────────────────────────────
    @property
    def engine(self) -> LLMEngine:
        return self._engine

    @property
    def tools(self) -> ToolRegistry:
        """Every capability the assistant can invoke.

        Built lazily so commands that never chat (`doctor`, `models`) don't open
        the ledger database they will not use.
        """
        if self._tools is None:
            self._tools = ToolRegistry()
        if not self._tools_ready:
            self._tools.register_all(default_tools())
            self._tools_ready = True
        return self._tools

    @property
    def memory(self) -> Any:
        """MemoryService — conversation persistence and recalled facts.

        Shares the RAG embedder rather than loading its own copy, but only
        reaches for it lazily: with nothing remembered, recall never touches the
        model at all.
        """
        if self._memory is None:
            from app.memory import MemoryService

            self._memory = MemoryService(embed_fn=lambda texts: self.rag.embed(texts))
        return self._memory

    @property
    def rag(self) -> Any:
        """RagService — loads the ONNX embedding model on first touch."""
        if self._rag is None:
            from app.rag import RagService

            self._rag = RagService()
        return self._rag

    @property
    def voice(self) -> Any:
        """VoiceService — its own STT/TTS models stay lazy inside it."""
        if self._voice is None:
            from app.voice import VoiceService

            self._voice = VoiceService()
        return self._voice

    # ── status ───────────────────────────────────────────────────
    def health(self) -> dict[str, Any]:
        healthy = self._engine.health_check()
        return {
            "engine": settings.engine,
            "model": settings.default_model,
            "healthy": healthy,
            "models": self._engine.list_models() if healthy else [],
        }

    # ── documents ────────────────────────────────────────────────
    def documents(self) -> list[dict[str, Any]]:
        return self.rag.documents()

    @property
    def document_chunks(self) -> int:
        return self.rag.count

    def reset_documents(self) -> None:
        self.rag.reset()

    def ingest_path(self, path: str) -> dict[str, Any]:
        """Ingest a local file, but only from inside an allowed root."""
        try:
            safe_path = filesvc.resolve_allowed(path)
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}
        try:
            result = self.rag.ingest_file(str(safe_path), safe_path.name)
        except Exception as exc:
            return {"ok": False, "error": str(exc), "source": safe_path.name}
        return {"ok": True, **result, "total_chunks": self.rag.count}

    def search_files(self, query: str, limit: int | None = None) -> list[dict[str, Any]]:
        return filesvc.search_files(query, limit or settings.file_search_max_results)

    # ── speech ───────────────────────────────────────────────────
    def transcribe(self, audio_path: str) -> str:
        return self.voice.transcribe(audio_path)

    def speak(self, text: str) -> tuple[bytes, bool]:
        """Synthesize `text` to WAV bytes. Returns (audio, was_summarized)."""
        text = text.strip()
        summarized = len(text) > settings.tts_summary_threshold_chars
        spoken = self.summarize_for_speech(text) if summarized else text
        return self.voice.synthesize(spoken), summarized

    def summarize_for_speech(self, text: str) -> str:
        """Condense a long reply into a couple of sentences fit for reading aloud."""
        messages = [
            ChatMessage(role="system", content="You write very short spoken summaries."),
            ChatMessage(
                role="user",
                content=(
                    "Condense the following into at most two short sentences to be "
                    "read aloud. Keep only the key outcome. Reply with only the "
                    "summary.\n\n" + text
                ),
            ),
        ]
        out: list[str] = []
        for event in self._engine.chat_stream(
            messages, GenerationOptions(max_tokens=90, temperature=0.2)
        ):
            if not event.done:
                out.append(event.token)
        return "".join(out).strip() or text

    # ── the turn ─────────────────────────────────────────────────
    def chat_stream(
        self,
        history: list[ChatMessage],
        *,
        use_rag: bool = False,
        allow_file_access: bool = True,
        model: str | None = None,
        temperature: float | None = None,
        confirm: Callable[[str], bool] | None = None,
        use_memory: bool = True,
    ) -> Iterator[AssistantEvent]:
        """Run one turn, yielding token events then a single done/error event.

        `history` is copied before any grounding rewrite, so the caller's list is
        left exactly as the user typed it.

        `confirm` lets the front end ask the user before a tool acts. Omitting it
        means every permission question answers "no" — a non-interactive caller
        must not authorise anything by default.
        """
        turn_history = copy.deepcopy(history)
        if not turn_history or turn_history[0].role != "system":
            turn_history.insert(
                0, ChatMessage(role="system", content=system_prompt())
            )

        last_user_text = next(
            (m.content for m in reversed(turn_history) if m.role == "user"), ""
        )

        # Recall runs before tools and RAG: a remembered fact is context for the
        # whole turn, not an alternative to it. Injected as its own system
        # message so it survives a tool rewriting the user's message.
        recalled: dict[str, Any] | None = None
        if use_memory:
            recall = self.memory.recall(last_user_text)
            if recall:
                turn_history.insert(1, ChatMessage(role="system", content=recall.as_prompt()))
                recalled = {
                    "count": len(recall.memories),
                    "display": recall.display(),
                    "texts": [m.text for m in recall.memories],
                    "scores": [round(s, 3) for s in recall.scores],
                }

        # One dispatch for every capability. This used to be a chain of keyword
        # gates where each new feature added a boolean to every sibling branch,
        # and overlapping gates were separated by ordering — invisible, untested,
        # and unable to survive a third capability.
        tool_used: dict[str, Any] | None = None
        # `allow_file_access` disables file *reading*, not every capability.
        # Gating the whole registry on it meant `--no-files` also silenced the
        # system tools, so "why is my laptop slow?" fell back to generic advice.
        excluded = set() if allow_file_access else FILE_ACCESS_TOOLS
        dispatch = self.tools.dispatch(
            last_user_text,
            self._engine,
            ToolContext(
                assistant=self,
                request_text=last_user_text,
                confirm=confirm,
            ),
            exclude=excluded,
        )

        # Some outcomes are already fully determined — "you declined, nothing
        # happened" has nothing for a model to compose, and letting a 3B model
        # phrase it produced confident falsehoods ("I am unable to terminate
        # processes on your system"). Answer exactly, and skip generation.
        if dispatch.result is not None and dispatch.result.final_text:
            yield AssistantEvent(type="token", text=dispatch.result.final_text)
            yield AssistantEvent(
                type="done",
                tool_used={
                    "tool": dispatch.invocation.tool if dispatch.invocation else "",
                    "display": dispatch.result.display,
                    "failed": True,
                },
                recalled=recalled,
                metrics={"model": model or settings.default_model, "determined": True},
            )
            return

        if dispatch.result is not None:
            last_user = next(
                (m for m in reversed(turn_history) if m.role == "user"), None
            )
            if dispatch.grounded and last_user is not None:
                last_user.content = dispatch.result.content
                tool_used = {
                    "tool": dispatch.invocation.tool if dispatch.invocation else "",
                    "display": dispatch.result.display,
                    **dispatch.result.meta,
                }
            elif dispatch.correction:
                # Keep the model honest: it wanted a capability that could not
                # deliver, so it says why instead of denying it has the ability.
                turn_history.insert(
                    1, ChatMessage(role="system", content=dispatch.correction)
                )
                tool_used = {
                    "tool": dispatch.invocation.tool if dispatch.invocation else "",
                    "display": dispatch.result.display,
                    "failed": True,
                }

        # RAG: retrieve context for the latest user turn and ground the prompt.
        # Skipped when a tool already grounded this message.
        sources: list[dict[str, Any]] = []
        if not dispatch.grounded and use_rag and self.rag.count > 0:
            last_user = next(
                (m for m in reversed(turn_history) if m.role == "user"), None
            )
            if last_user is not None:
                hits = self.rag.retrieve(last_user.content, settings.rag_top_k)
                if hits:
                    context = "\n\n".join(
                        f"[{i + 1}] (from {h['source']})\n{h['text']}"
                        for i, h in enumerate(hits)
                    )
                    last_user.content = RAG_PROMPT_TEMPLATE.format(
                        context=context, question=last_user.content
                    )
                    sources = [
                        {
                            "n": i + 1,
                            "source": h["source"],
                            "score": round(h["score"], 3),
                            "preview": h["text"][:160].replace("\n", " "),
                        }
                        for i, h in enumerate(hits)
                    ]

        # num_ctx is intentionally left at the engine's constant default — varying
        # it per request forces Ollama to reload the model (~25s on CPU).
        options = GenerationOptions(model=model, temperature=temperature)

        try:
            for event in self._engine.chat_stream(turn_history, options):
                if event.done:
                    yield AssistantEvent(
                        type="done",
                        sources=sources,
                        tool_used=tool_used,
                        recalled=recalled,
                        metrics={
                            "model": event.model,
                            "fell_back": event.fell_back,
                            "prompt_tokens": event.prompt_tokens,
                            "completion_tokens": event.completion_tokens,
                            "time_to_first_token_s": event.time_to_first_token_s,
                            "load_duration_s": event.load_duration_s,
                            "prompt_eval_duration_s": event.prompt_eval_duration_s,
                            "eval_duration_s": event.eval_duration_s,
                            "total_duration_s": event.total_duration_s,
                            "tokens_per_second": event.tokens_per_second,
                        },
                    )
                else:
                    yield AssistantEvent(type="token", text=event.token)
        except Exception as exc:  # surface engine/connection errors to the front end
            yield AssistantEvent(type="error", error=str(exc))
