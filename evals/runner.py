"""The eval runner — builds a throwaway index, then measures the two layers.

RAG has two failure modes that need separating, because the fix for each is
completely different:

  1. **Retrieval failed.** The answer was never in the prompt. Tune chunking,
     `rag_top_k`, or the embedding model. Changing the LLM will not help.
  2. **Generation failed.** The right chunk *was* in the prompt and the model
     still got it wrong, or ignored it, or embellished. Tune the prompt template
     or the model. Changing the chunker will not help.

A single end-to-end accuracy number cannot tell those apart, so this runner
reports them as separate stages and cross-tabulates them at the end.

Retrieval-only runs need no LLM at all, which makes them fast and fully
deterministic — the right thing to run on every commit.
"""
from __future__ import annotations

import json
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

from app.core.assistant import RAG_PROMPT_TEMPLATE
from app.engines import ChatMessage
from config import settings

from . import metrics
from .judge import judge_grounded

DATASET_PATH = Path(__file__).parent / "dataset.json"


@dataclass
class EvalReport:
    """Everything one run produced, ready to render or serialize."""

    retrieval: list[metrics.RetrievalScore] = field(default_factory=list)
    generation: list[metrics.GenerationScore] = field(default_factory=list)
    retrieval_summary: dict[str, Any] = field(default_factory=dict)
    generation_summary: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    chunks_indexed: int = 0
    wall_time_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config,
            "chunks_indexed": self.chunks_indexed,
            "wall_time_s": round(self.wall_time_s, 2),
            "retrieval": self.retrieval_summary,
            "generation": self.generation_summary,
            "cases": [
                {
                    "id": g.case_id,
                    "kind": g.kind,
                    "question": g.question,
                    "answer": g.answer,
                    "correct": g.correct,
                    "refused": g.refused,
                    "grounded": g.grounded,
                    "judge_reason": g.judge_reason,
                    "latency_s": round(g.latency_s, 2),
                    "retrieval_hit": g.retrieval.hit if g.retrieval else None,
                    "retrieval_rank": g.retrieval.rank if g.retrieval else None,
                }
                for g in self.generation
            ]
            or [
                {
                    "id": r.case_id,
                    "kind": r.kind,
                    "hit": r.hit,
                    "rank": r.rank,
                    "top_score": round(r.top_score, 3),
                }
                for r in self.retrieval
            ],
        }


def load_dataset(path: Path | None = None) -> dict[str, Any]:
    return json.loads((path or DATASET_PATH).read_text(encoding="utf-8"))


class EvalRunner:
    """Owns a disposable RAG index for the duration of one run."""

    def __init__(
        self,
        dataset: dict[str, Any] | None = None,
        top_k: int | None = None,
        model: str | None = None,
    ) -> None:
        self.dataset = dataset or load_dataset()
        self.top_k = top_k or settings.rag_top_k
        self.model = model or settings.default_model
        self._tmpdir: str | None = None
        self._rag: Any = None
        self._assistant: Any = None

    # ── lifecycle ────────────────────────────────────────────────
    def __enter__(self) -> "EvalRunner":
        self._tmpdir = tempfile.mkdtemp(prefix="rag-eval-")
        from app.rag import RagService

        self._rag = RagService(store_dir=self._tmpdir)

        doc = Path(self.dataset["document"])
        if not doc.exists():
            raise FileNotFoundError(f"eval document missing: {doc}")
        self._rag.ingest_file(str(doc), doc.name)
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._tmpdir:
            shutil.rmtree(self._tmpdir, ignore_errors=True)

    @property
    def assistant(self) -> Any:
        """An Assistant wired to the *throwaway* index, not the user's."""
        if self._assistant is None:
            from app.core import Assistant

            self._assistant = Assistant()
            self._assistant._rag = self._rag  # inject the hermetic store
        return self._assistant

    @property
    def chunks_indexed(self) -> int:
        return self._rag.count if self._rag else 0

    def _config(self) -> dict[str, Any]:
        return {
            "embedding_model": settings.embedding_model,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "top_k": self.top_k,
            "llm": self.model,
            "document": self.dataset["document"],
        }

    # ── stage 1: retrieval only (no LLM, deterministic) ──────────
    def run_retrieval(self) -> list[metrics.RetrievalScore]:
        scores: list[metrics.RetrievalScore] = []
        for case in self.dataset["cases"]:
            if case.get("kind") == "absent":
                continue
            hits = self._rag.retrieve(case["question"], self.top_k)
            scores.append(
                metrics.score_retrieval(
                    case_id=case["id"],
                    kind=case.get("kind", "fact"),
                    hits=hits,
                    must_retrieve=case.get("must_retrieve", []),
                )
            )
        return scores

    # ── stage 2: end-to-end generation ───────────────────────────
    def run_generation(
        self,
        judge: bool = True,
        on_case: Callable[[str], None] | None = None,
    ) -> list[metrics.GenerationScore]:
        scores: list[metrics.GenerationScore] = []

        for case in self.dataset["cases"]:
            if on_case:
                on_case(case["id"])

            hits = self._rag.retrieve(case["question"], self.top_k)
            context = "\n\n".join(
                f"[{i + 1}] (from {h['source']})\n{h['text']}"
                for i, h in enumerate(hits)
            )

            retrieval = (
                metrics.score_retrieval(
                    case_id=case["id"],
                    kind=case.get("kind", "fact"),
                    hits=hits,
                    must_retrieve=case.get("must_retrieve", []),
                )
                if case.get("kind") != "absent"
                else None
            )

            answer, latency = self._answer(case["question"])
            score = metrics.score_generation(case, answer, latency, retrieval)

            if judge:
                grounded, reason = judge_grounded(
                    self.assistant.engine,
                    question=case["question"],
                    context=context,
                    answer=answer,
                    model=self.model,
                )
                score.grounded = grounded
                score.judge_reason = reason

            scores.append(score)
        return scores

    def _answer(self, question: str) -> tuple[str, float]:
        """Run one turn through the real assistant path with RAG on.

        Temperature is pinned to 0 so a rerun measures the system, not sampling
        noise. `allow_file_access=False` keeps the file-opening branch in
        `chat_stream` from hijacking a question like "what do I need to reset
        the box" and grounding it on a real file from the user's disk instead.
        """
        started = time.perf_counter()
        parts: list[str] = []
        for event in self.assistant.chat_stream(
            [ChatMessage(role="user", content=question)],
            use_rag=True,
            allow_file_access=False,
            model=self.model,
            temperature=0.0,
        ):
            if event.type == "token":
                parts.append(event.text)
            elif event.type == "error":
                return f"[engine error: {event.error}]", time.perf_counter() - started
        return "".join(parts).strip(), time.perf_counter() - started

    # ── orchestration ────────────────────────────────────────────
    def run(
        self,
        retrieval_only: bool = False,
        judge: bool = True,
        on_case: Callable[[str], None] | None = None,
    ) -> EvalReport:
        started = time.perf_counter()
        report = EvalReport(config=self._config(), chunks_indexed=self.chunks_indexed)

        if retrieval_only:
            report.retrieval = self.run_retrieval()
        else:
            report.generation = self.run_generation(judge=judge, on_case=on_case)
            report.retrieval = [
                g.retrieval for g in report.generation if g.retrieval is not None
            ]
            report.generation_summary = metrics.aggregate_generation(report.generation)

        report.retrieval_summary = metrics.aggregate_retrieval(report.retrieval)
        report.wall_time_s = time.perf_counter() - started
        return report


def cross_tabulate(report: EvalReport) -> dict[str, list[str]]:
    """Split answerable failures by *which stage* broke.

    This is the payoff for measuring the stages separately — it turns "accuracy
    is 71%" into a specific instruction about what to go fix.
    """
    buckets: dict[str, list[str]] = {
        "both_ok": [],
        "retrieval_failed": [],      # nothing to answer from — fix chunking/top_k
        "generation_failed": [],     # context was right, model fumbled — fix prompt/model
        "both_failed": [],
    }
    for g in report.generation:
        if g.retrieval is None:
            continue
        key = (
            "both_ok" if g.retrieval.hit and g.correct
            else "generation_failed" if g.retrieval.hit
            else "retrieval_failed" if g.correct
            else "both_failed"
        )
        buckets[key].append(g.case_id)
    return buckets
