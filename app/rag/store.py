"""On-disk FAISS vector store with parallel chunk metadata.

Uses IndexFlatIP over L2-normalized vectors, so inner product == cosine
similarity. The index and its metadata are persisted so ingested documents
survive a restart.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import faiss
import numpy as np


class VectorStore:
    def __init__(self, dir_path: str, dim: int) -> None:
        self._dir = Path(dir_path)
        self._dim = dim
        self._index = faiss.IndexFlatIP(dim)
        self._meta: list[dict[str, Any]] = []
        self._load()

    # ── mutation ─────────────────────────────────────────────────
    def add(self, vectors: np.ndarray, metas: list[dict[str, Any]]) -> None:
        self._index.add(vectors)
        self._meta.extend(metas)
        self._save()

    def reset(self) -> None:
        self._index = faiss.IndexFlatIP(self._dim)
        self._meta = []
        self._save()

    # ── query ────────────────────────────────────────────────────
    def search(self, query_vec: np.ndarray, k: int) -> list[dict[str, Any]]:
        if self._index.ntotal == 0:
            return []
        k = min(k, self._index.ntotal)
        scores, idxs = self._index.search(query_vec, k)
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0:
                continue
            results.append({**self._meta[int(idx)], "score": float(score)})
        return results

    def sources(self) -> list[dict[str, Any]]:
        counts = Counter(m["source"] for m in self._meta)
        return [{"source": s, "chunks": n} for s, n in counts.items()]

    @property
    def count(self) -> int:
        return self._index.ntotal

    # ── persistence ──────────────────────────────────────────────
    def _save(self) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(self._dir / "index.faiss"))
        (self._dir / "meta.json").write_text(
            json.dumps(self._meta, ensure_ascii=False), encoding="utf-8"
        )

    def _load(self) -> None:
        idx_path = self._dir / "index.faiss"
        meta_path = self._dir / "meta.json"
        if not (idx_path.exists() and meta_path.exists()):
            return
        index = faiss.read_index(str(idx_path))
        # If a stale index has a different dimension, start clean instead of crashing.
        if index.d != self._dim:
            return
        self._index = index
        self._meta = json.loads(meta_path.read_text(encoding="utf-8"))
