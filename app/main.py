"""FastAPI backend for the Local Offline AI Assistant.

Serves a single-page chat UI and a streaming /api/chat endpoint. The endpoint
streams newline-delimited JSON (NDJSON): one object per token, then a final
object carrying latency metrics. Conversation history is held by the browser and
sent with each request, so the server stays stateless.
"""
from __future__ import annotations

import json
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterator

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel

from app import files as filesvc
from app.engines import ChatMessage, GenerationOptions, build_engine
from app.rag import RagService
from app.voice import VoiceService
from config import settings

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful, concise assistant running fully offline on the user's "
    "own machine. Be direct and accurate. If you are unsure, say so."
)

WEB_DIR = Path(__file__).parent / "web"

RAG_PROMPT_TEMPLATE = (
    "Answer the question using ONLY the context below. If the answer is not in "
    "the context, say you don't know based on the provided documents. Where "
    "relevant, cite sources inline like [1], [2].\n\n"
    "Context:\n{context}\n\nQuestion: {question}"
)

app = FastAPI(title="Local Offline AI Assistant")
engine = build_engine()
rag = RagService()  # loads the embedding model into memory at startup
voice = VoiceService()  # STT/TTS load lazily on first use


def summarize_for_speech(text: str) -> str:
    """Condense a long reply into a couple of sentences suitable for reading aloud."""
    messages = [
        ChatMessage(role="system", content="You write very short spoken summaries."),
        ChatMessage(
            role="user",
            content=(
                "Condense the following into at most two short sentences to be read "
                "aloud. Keep only the key outcome. Reply with only the summary.\n\n"
                + text
            ),
        ),
    ]
    out: list[str] = []
    for event in engine.chat_stream(
        messages, GenerationOptions(max_tokens=90, temperature=0.2)
    ):
        if not event.done:
            out.append(event.token)
    return "".join(out).strip() or text


# ── conversational file opening (model-driven) ───────────────────
FILE_TOOL_NAME = "open_local_file"
FILE_TOOL = {
    "type": "function",
    "function": {
        "name": FILE_TOOL_NAME,
        "description": (
            "Find and read a document from the user's local folders (Downloads, "
            "Documents, Desktop). Call this whenever the user asks to get, open, "
            "read, load, find, review, or summarize a file on their computer. If "
            "several files match, the most recent one is used."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Part of the filename to look for, e.g. 'resume' or 'abc'.",
                },
                "folder": {
                    "type": "string",
                    "description": "Optional folder: 'downloads', 'documents', or 'desktop'.",
                },
            },
            "required": ["name"],
        },
    },
}

FILE_DECISION_SYSTEM = (
    "You decide whether the user wants to open a local document. If they do, call "
    f"{FILE_TOOL_NAME} with the name they mentioned (and folder if stated). Do not "
    "call it for anything else."
)

FILE_GROUNDING = (
    "The user asked to work with a local file. Its contents were retrieved for you.\n"
    "File: {name} (from {root}, modified {modified})\n\n"
    "--- FILE CONTENT START ---\n{text}\n--- FILE CONTENT END ---\n\n"
    "Answer the user's request using the file above: {question}"
)

_FILE_VERBS = (
    "get", "grab", "open", "read", "load", "fetch", "find", "locate", "pull",
    "show", "review", "summarize", "summarise", "look at", "look up", "access", "pick",
)
_FILE_NOUNS = (
    "file", "document", "doc", "pdf", "docx", "resume", "cv", "download",
    "desktop", "folder", "named", "called", ".pdf", ".docx", ".txt", ".md",
)


def looks_like_file_request(text: str) -> bool:
    """Cheap gate so normal chat never pays the tool-calling cost."""
    t = text.lower()
    return any(v in t for v in _FILE_VERBS) and any(n in t for n in _FILE_NOUNS)


_STOPWORDS = {
    "get", "grab", "open", "read", "load", "fetch", "find", "locate", "pull",
    "show", "review", "summarize", "summarise", "look", "up", "at", "access",
    "pick", "give", "tell", "say", "says", "the", "a", "an", "my", "me", "you",
    "your", "it", "its", "that", "this", "what", "whats", "about", "from", "in",
    "into", "on", "of", "and", "then", "please", "can", "could", "would", "will",
    "file", "files", "document", "documents", "doc", "docs", "pdf", "folder",
    "named", "called", "downloads", "download", "desktop", "content", "contents",
    "latest", "recent", "newest", "there", "is", "are", "with", "for", "to",
}
_FOLDERS = ("downloads", "documents", "desktop")


def _extract_folder(text: str) -> str | None:
    t = text.lower()
    for f in _FOLDERS:
        if f in t:
            return f
    return None


def _extract_name(text: str) -> str | None:
    """Reduce the message to its distinctive words (drop verbs/articles/folders).

    Order-independent, so 'the quokka notes file' and 'file named quokka' both
    yield 'quokka notes' / 'quokka'.
    """
    quoted = re.search(r"[\"']([^\"']{2,})[\"']", text)
    if quoted:
        return quoted.group(1).strip()
    words = re.findall(r"[a-z0-9][a-z0-9._\-]*", text.lower())
    keep = [w for w in words if w not in _STOPWORDS and len(w) > 1]
    return " ".join(keep) if keep else None


def try_open_file(history: list[ChatMessage]) -> tuple[dict[str, Any] | None, str | None]:
    """Resolve a file request. Returns (opened_file_meta, correction_note).

    - (meta, None)  → a file was opened and the prompt grounded on it
    - (None, note)  → could not open; `note` is a system hint so the model
      responds honestly ("couldn't find it") instead of "I can't access files"
    """
    last_user = next((m for m in reversed(history) if m.role == "user"), None)
    if last_user is None:
        return None, None
    request_text = last_user.content

    # Model picks the file (honours "let the model decide"); heuristic backstop.
    name: str | None = None
    folder: str | None = _extract_folder(request_text)
    try:
        turn = engine.chat(
            [
                ChatMessage(role="system", content=FILE_DECISION_SYSTEM),
                ChatMessage(role="user", content=request_text),
            ],
            tools=[FILE_TOOL],
            options=GenerationOptions(temperature=0.0),
        )
        for tc in turn.tool_calls:
            if tc.name == FILE_TOOL_NAME:
                name = str(tc.arguments.get("name") or "").strip() or None
                folder = str(tc.arguments.get("folder") or "").strip() or folder
                break
    except Exception:
        pass  # tool-calling unsupported/failed → heuristic below

    if not name:
        name = _extract_name(request_text)
    if not name:
        return None, (
            "The user asked for a file but the name was unclear. Ask them for the "
            "exact file name. Do not claim you cannot access files."
        )

    match = filesvc.find_latest(name, folder)
    if not match:
        where = f" in {folder}" if folder else ""
        return None, (
            f"You searched the user's folders for a file matching '{name}'{where} "
            "but found none. Tell them you couldn't find that file and ask them to "
            "check the name. Do not claim you cannot access files."
        )

    try:
        text = rag.read_file(match["path"])
    except Exception as exc:
        return None, (
            f"You found '{match['name']}' but couldn't read it ({exc}). Tell the "
            "user the file could not be read."
        )

    # Persist for follow-up RAG questions (skip if already ingested).
    if match["name"] not in {d["source"] for d in rag.documents()}:
        try:
            rag.ingest_file(match["path"], match["name"])
        except Exception:
            pass

    last_user.content = FILE_GROUNDING.format(
        name=match["name"],
        root=match["root"],
        modified=match["modified"],
        text=text[:8000],
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


# ── API models ───────────────────────────────────────────────────
class ClientMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[ClientMessage]
    model: str | None = None
    temperature: float | None = None
    use_rag: bool = False
    allow_file_access: bool = True


class SpeakRequest(BaseModel):
    text: str


class IngestPathRequest(BaseModel):
    path: str


# ── routes ───────────────────────────────────────────────────────
@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/info")
def info() -> dict:
    """Header data for the UI: which engine/model, and is it reachable."""
    healthy = engine.health_check()
    return {
        "engine": settings.engine,
        "model": settings.default_model,
        "healthy": healthy,
        "models": engine.list_models() if healthy else [],
    }


@app.get("/api/documents")
def documents() -> dict:
    return {"count": rag.count, "documents": rag.documents()}


@app.post("/api/ingest")
async def ingest(file: UploadFile = File(...)) -> dict:
    """Accept a .txt/.md/.pdf upload, embed it, and add it to the vector store."""
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    dest = upload_dir / Path(file.filename or "upload").name

    with dest.open("wb") as out:
        shutil.copyfileobj(file.file, out)

    try:
        result = rag.ingest_file(str(dest), dest.name)
    except Exception as exc:
        return {"ok": False, "error": str(exc), "source": dest.name}

    return {"ok": True, **result, "total_chunks": rag.count}


@app.post("/api/rag/reset")
def rag_reset() -> dict:
    rag.reset()
    return {"ok": True, "count": rag.count}


@app.get("/api/files/roots")
def file_roots() -> dict:
    """The folders the assistant is allowed to browse."""
    return {"roots": [{"name": n, "path": str(p)} for n, p in filesvc.allowed_roots()]}


@app.get("/api/files/search")
def file_search(q: str = "") -> dict:
    """Search allowed folders for supported files matching `q`."""
    return {"results": filesvc.search_files(q, settings.file_search_max_results)}


@app.post("/api/files/ingest")
def file_ingest(req: IngestPathRequest) -> dict:
    """Ingest a local file by path — but only if it is inside an allowed root."""
    try:
        safe_path = filesvc.resolve_allowed(req.path)
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}
    try:
        result = rag.ingest_file(str(safe_path), safe_path.name)
    except Exception as exc:
        return {"ok": False, "error": str(exc), "source": safe_path.name}
    return {"ok": True, **result, "total_chunks": rag.count}


@app.post("/api/transcribe")
async def transcribe(file: UploadFile = File(...)) -> dict:
    """Speech-to-text: accept a recorded audio blob and return the transcript."""
    suffix = Path(file.filename or "rec.webm").suffix or ".webm"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        shutil.copyfileobj(file.file, tmp)
        tmp.close()
        text = voice.transcribe(tmp.name)
        return {"ok": True, "text": text}
    except Exception as exc:
        return {"ok": False, "error": str(exc), "text": ""}
    finally:
        Path(tmp.name).unlink(missing_ok=True)


@app.post("/api/speak")
def speak(req: SpeakRequest) -> Response:
    """Text-to-speech: return WAV audio. Long text is summarized before speaking."""
    text = req.text.strip()
    summarized = len(text) > settings.tts_summary_threshold_chars
    spoken = summarize_for_speech(text) if summarized else text
    audio = voice.synthesize(spoken)
    return Response(
        content=audio,
        media_type="audio/wav",
        headers={"X-Summarized": "true" if summarized else "false"},
    )


@app.post("/api/chat")
def chat(req: ChatRequest) -> StreamingResponse:
    # Prepend a server-controlled system prompt unless the client already sent one.
    history = [ChatMessage(role=m.role, content=m.content) for m in req.messages]
    if not history or history[0].role != "system":
        history.insert(0, ChatMessage(role="system", content=DEFAULT_SYSTEM_PROMPT))

    # Model-driven file opening: if the latest message asks for a local file,
    # let the model choose it, fetch the newest match, and ground on its text.
    opened_file: dict[str, Any] | None = None
    last_user_text = next(
        (m.content for m in reversed(history) if m.role == "user"), ""
    )
    if req.allow_file_access and looks_like_file_request(last_user_text):
        opened_file, correction = try_open_file(history)
        if correction:
            # Keep the model honest when it wanted a file but none was opened.
            history.insert(1, ChatMessage(role="system", content=correction))

    # RAG: retrieve context for the latest user turn and ground the prompt.
    # Skipped when a file was just opened (that message is already grounded).
    sources: list[dict[str, Any]] = []
    if opened_file is None and req.use_rag and rag.count > 0:
        last_user = next((m for m in reversed(history) if m.role == "user"), None)
        if last_user is not None:
            hits = rag.retrieve(last_user.content, settings.rag_top_k)
            if hits:
                context = "\n\n".join(
                    f"[{i + 1}] (from {h['source']})\n{h['text']}"
                    for i, h in enumerate(hits)
                )
                # Ground the model on retrieved context (server-side only; the
                # browser keeps the user's original wording in its history).
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

    # num_ctx is intentionally left to the engine's constant default — varying it
    # per request would force Ollama to reload the model each time.
    options = GenerationOptions(model=req.model, temperature=req.temperature)

    def ndjson_stream() -> Iterator[str]:
        try:
            for event in engine.chat_stream(history, options):
                if event.done:
                    yield json.dumps(
                        {
                            "type": "done",
                            "sources": sources,
                            "opened_file": opened_file,
                            "metrics": {
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
                        }
                    ) + "\n"
                else:
                    yield json.dumps({"type": "token", "text": event.token}) + "\n"
        except Exception as exc:  # surface engine/connection errors to the UI
            yield json.dumps({"type": "error", "error": str(exc)}) + "\n"

    return StreamingResponse(ndjson_stream(), media_type="application/x-ndjson")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
