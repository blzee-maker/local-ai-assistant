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
from typing import Any, Iterator, Literal

from app import files as filesvc
from app.core import fileintent
from app.engines import ChatMessage, GenerationOptions, build_engine
from app.engines.base import LLMEngine
from config import settings

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful, concise assistant running fully offline on the user's "
    "own machine. Be direct and accurate. If you are unsure, say so."
)

RAG_PROMPT_TEMPLATE = (
    "Answer the question using ONLY the context below. If the answer is not in "
    "the context, say you don't know based on the provided documents. Where "
    "relevant, cite sources inline like [1], [2].\n\n"
    "Context:\n{context}\n\nQuestion: {question}"
)

# How much of a file's text is injected into the prompt. Bounded so a large
# document cannot blow past num_ctx and silently truncate the conversation.
FILE_INJECTION_CHARS = 8000


@dataclass
class AssistantEvent:
    """One event in a streamed turn: many `token`s, then exactly one terminal event."""

    type: Literal["token", "done", "error"]
    text: str = ""
    sources: list[dict[str, Any]] = field(default_factory=list)
    opened_file: dict[str, Any] | None = None
    metrics: dict[str, Any] | None = None
    error: str | None = None


class Assistant:
    def __init__(self, engine: LLMEngine | None = None) -> None:
        self._engine = engine or build_engine()
        self._rag: Any = None
        self._voice: Any = None

    # ── lazily-built subsystems ──────────────────────────────────
    @property
    def engine(self) -> LLMEngine:
        return self._engine

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

    # ── model-driven local file opening ──────────────────────────
    def _try_open_file(
        self, history: list[ChatMessage]
    ) -> tuple[dict[str, Any] | None, str | None]:
        """Resolve a file request against `history` (already a private copy).

        Returns ``(opened_file_meta, correction_note)``:

        * ``(meta, None)`` — a file was opened and the last user message rewritten
        * ``(None, note)`` — nothing opened; `note` is a system hint that keeps the
          model honest ("couldn't find it") instead of the false and unhelpful
          "I can't access files"
        """
        last_user = next((m for m in reversed(history) if m.role == "user"), None)
        if last_user is None:
            return None, None
        request_text = last_user.content

        name: str | None = None
        folder: str | None = fileintent.extract_folder(request_text)
        try:
            turn = self._engine.chat(
                [
                    ChatMessage(role="system", content=fileintent.FILE_DECISION_SYSTEM),
                    ChatMessage(role="user", content=request_text),
                ],
                tools=[fileintent.FILE_TOOL],
                options=GenerationOptions(temperature=0.0),
            )
            for tc in turn.tool_calls:
                if tc.name == fileintent.FILE_TOOL_NAME:
                    name = str(tc.arguments.get("name") or "").strip() or None
                    folder = str(tc.arguments.get("folder") or "").strip() or folder
                    break
        except Exception:
            pass  # tool-calling unsupported or failed → heuristic below

        if not name:
            name = fileintent.extract_name(request_text)
        if not name:
            return None, (
                "The user asked for a file but the name was unclear. Ask them for "
                "the exact file name. Do not claim you cannot access files."
            )

        match = filesvc.find_latest(name, folder)
        if not match:
            where = f" in {folder}" if folder else ""
            return None, (
                f"You searched the user's folders for a file matching '{name}'"
                f"{where} but found none. Tell them you couldn't find that file and "
                "ask them to check the name. Do not claim you cannot access files."
            )

        try:
            text = self.rag.read_file(match["path"])
        except Exception as exc:
            return None, (
                f"You found '{match['name']}' but couldn't read it ({exc}). Tell "
                "the user the file could not be read."
            )

        # Persist for follow-up RAG questions (skip if already ingested).
        if match["name"] not in {d["source"] for d in self.rag.documents()}:
            try:
                self.rag.ingest_file(match["path"], match["name"])
            except Exception:
                pass

        last_user.content = fileintent.FILE_GROUNDING.format(
            name=match["name"],
            root=match["root"],
            modified=match["modified"],
            text=text[:FILE_INJECTION_CHARS],
            question=request_text,
        )
        return (
            {
                "name": match["name"],
                "root": match["root"],
                "modified": match["modified"],
                "chars": len(text),
            },
            None,
        )

    # ── the turn ─────────────────────────────────────────────────
    def chat_stream(
        self,
        history: list[ChatMessage],
        *,
        use_rag: bool = False,
        allow_file_access: bool = True,
        model: str | None = None,
        temperature: float | None = None,
    ) -> Iterator[AssistantEvent]:
        """Run one turn, yielding token events then a single done/error event.

        `history` is copied before any grounding rewrite, so the caller's list is
        left exactly as the user typed it.
        """
        turn_history = copy.deepcopy(history)
        if not turn_history or turn_history[0].role != "system":
            turn_history.insert(
                0, ChatMessage(role="system", content=DEFAULT_SYSTEM_PROMPT)
            )

        # Model-driven file opening: if the latest message asks for a local file,
        # let the model choose it, fetch the newest match, ground on its text.
        opened_file: dict[str, Any] | None = None
        last_user_text = next(
            (m.content for m in reversed(turn_history) if m.role == "user"), ""
        )
        if allow_file_access and fileintent.looks_like_file_request(last_user_text):
            opened_file, correction = self._try_open_file(turn_history)
            if correction:
                turn_history.insert(1, ChatMessage(role="system", content=correction))

        # RAG: retrieve context for the latest user turn and ground the prompt.
        # Skipped when a file was just opened — that message is already grounded.
        sources: list[dict[str, Any]] = []
        if opened_file is None and use_rag and self.rag.count > 0:
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
                        opened_file=opened_file,
                        metrics={
                            "model": event.model,
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
