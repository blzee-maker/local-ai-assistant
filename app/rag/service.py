"""RagService — ties embedding, chunking, and the vector store together.

One instance is created at app startup and shared. Loading the embedder pulls
the ONNX model into memory (downloaded once on first ever run).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from pypdf import PdfReader

from config import settings

from .chunker import chunk_text
from .embedder import Embedder
from .store import VectorStore


class RagService:
    def __init__(self, store_dir: str | None = None) -> None:
        """`store_dir` overrides the configured vector-store location.

        Defaults to `settings.vector_store_dir` for normal use. The eval harness
        passes a throwaway directory so a benchmark run neither reads nor
        pollutes the user's real index — otherwise scores would depend on
        whatever happened to be ingested that day, and reruns would drift.
        """
        self._embedder = Embedder(settings.embedding_model, settings.embedding_cache_dir)
        self._store = VectorStore(store_dir or settings.vector_store_dir, self._embedder.dim)

    # ── ingestion ────────────────────────────────────────────────
    def ingest_file(self, path: str, source_name: str) -> dict[str, Any]:
        text = self._read(path)
        chunks = chunk_text(text, settings.chunk_size, settings.chunk_overlap)
        if not chunks:
            return {"source": source_name, "chunks": 0}

        vectors = self._embedder.embed(chunks)
        base = self._store.count
        metas = [
            {"id": base + i, "text": c, "source": source_name}
            for i, c in enumerate(chunks)
        ]
        self._store.add(vectors, metas)
        return {"source": source_name, "chunks": len(chunks)}

    # ── embedding ────────────────────────────────────────────────
    def embed(self, texts: list[str]):
        """Public embedding access, shared with the memory store.

        Memory needs vectors too, and loading a second copy of the ONNX model
        would cost ~86MB for nothing on an 8GB target (rule 9). One model, two
        consumers.
        """
        return self._embedder.embed(texts)

    @property
    def embedding_dim(self) -> int:
        return self._embedder.dim

    # ── retrieval ────────────────────────────────────────────────
    def retrieve(self, query: str, k: int | None = None) -> list[dict[str, Any]]:
        query_vec = self._embedder.embed([query])
        return self._store.search(query_vec, k or settings.rag_top_k)

    # ── introspection ────────────────────────────────────────────
    def documents(self) -> list[dict[str, Any]]:
        return self._store.sources()

    @property
    def count(self) -> int:
        return self._store.count

    def reset(self) -> None:
        self._store.reset()

    # ── file reading ─────────────────────────────────────────────
    def read_file(self, path: str) -> str:
        """Public: extract plain text from a supported file (pdf/docx/txt/md)."""
        return self._read(path)

    @staticmethod
    def _read(path: str) -> str:
        p = Path(path)
        ext = p.suffix.lower()
        if ext == ".pdf":
            reader = PdfReader(str(p))
            return "\n\n".join((page.extract_text() or "") for page in reader.pages)
        if ext == ".docx":
            import docx  # python-docx, imported lazily

            document = docx.Document(str(p))
            return "\n\n".join(par.text for par in document.paragraphs if par.text)
        return p.read_text(encoding="utf-8", errors="ignore")
