"""FastAPI front end — a thin adapter over `app.core.Assistant`.

All assistant behaviour lives in the core; this module only translates HTTP in
and NDJSON/audio out. Kept deliberately dumb so the CLI and this server can
never drift apart in what they actually do.
"""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Iterator

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel

from app import files as filesvc
from app.core import Assistant
from app.engines import ChatMessage
from config import settings

WEB_DIR = Path(__file__).parent / "web"

app = FastAPI(title="Local Offline AI Assistant")
assistant = Assistant()


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
    return assistant.health()


@app.get("/api/documents")
def documents() -> dict:
    return {"count": assistant.document_chunks, "documents": assistant.documents()}


@app.post("/api/ingest")
async def ingest(file: UploadFile = File(...)) -> dict:
    """Accept a .txt/.md/.pdf upload, embed it, and add it to the vector store."""
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    dest = upload_dir / Path(file.filename or "upload").name
    with dest.open("wb") as out:
        shutil.copyfileobj(file.file, out)
    return assistant.ingest_path(str(dest))


@app.post("/api/rag/reset")
def rag_reset() -> dict:
    assistant.reset_documents()
    return {"ok": True, "count": assistant.document_chunks}


@app.get("/api/files/roots")
def file_roots() -> dict:
    return {"roots": [{"name": n, "path": str(p)} for n, p in filesvc.allowed_roots()]}


@app.get("/api/files/search")
def file_search(q: str = "") -> dict:
    return {"results": assistant.search_files(q)}


@app.post("/api/files/ingest")
def file_ingest(req: IngestPathRequest) -> dict:
    return assistant.ingest_path(req.path)


@app.post("/api/transcribe")
async def transcribe(file: UploadFile = File(...)) -> dict:
    """Speech-to-text: accept a recorded audio blob and return the transcript."""
    suffix = Path(file.filename or "rec.webm").suffix or ".webm"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        shutil.copyfileobj(file.file, tmp)
        tmp.close()
        return {"ok": True, "text": assistant.transcribe(tmp.name)}
    except Exception as exc:
        return {"ok": False, "error": str(exc), "text": ""}
    finally:
        Path(tmp.name).unlink(missing_ok=True)


@app.post("/api/speak")
def speak(req: SpeakRequest) -> Response:
    """Text-to-speech: return WAV audio. Long text is summarized before speaking."""
    audio, summarized = assistant.speak(req.text)
    return Response(
        content=audio,
        media_type="audio/wav",
        headers={"X-Summarized": "true" if summarized else "false"},
    )


@app.post("/api/chat")
def chat(req: ChatRequest) -> StreamingResponse:
    history = [ChatMessage(role=m.role, content=m.content) for m in req.messages]

    def ndjson_stream() -> Iterator[str]:
        for event in assistant.chat_stream(
            history,
            use_rag=req.use_rag,
            allow_file_access=req.allow_file_access,
            model=req.model,
            temperature=req.temperature,
        ):
            if event.type == "token":
                yield json.dumps({"type": "token", "text": event.text}) + "\n"
            elif event.type == "done":
                yield json.dumps(
                    {
                        "type": "done",
                        "sources": event.sources,
                        "opened_file": event.opened_file,
                        "metrics": event.metrics,
                    }
                ) + "\n"
            else:
                yield json.dumps({"type": "error", "error": event.error}) + "\n"

    return StreamingResponse(ndjson_stream(), media_type="application/x-ndjson")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
