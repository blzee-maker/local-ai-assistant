"""Text embedding via fastembed (ONNX Runtime).

We use fastembed rather than sentence-transformers on purpose: it runs the same
all-MiniLM-L6-v2 model on ONNX Runtime instead of PyTorch, cutting install size
and memory dramatically — the right trade for an 8GB edge target. Vectors are
L2-normalized so a FAISS inner-product index computes cosine similarity.
"""
from __future__ import annotations

import numpy as np
from fastembed import TextEmbedding


class Embedder:
    def __init__(self, model_name: str, cache_dir: str | None = None) -> None:
        self._model = TextEmbedding(model_name=model_name, cache_dir=cache_dir)
        self._dim: int | None = None

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an (n, dim) float32 array of L2-normalized embeddings."""
        vecs = np.asarray(list(self._model.embed(texts)), dtype="float32")
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._dim = int(self.embed(["dimension probe"]).shape[1])
        return self._dim
