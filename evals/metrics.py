"""Scoring primitives for the eval harness.

These are deliberately dumb, deterministic string functions rather than anything
clever. An eval harness whose *metrics* are themselves fuzzy cannot tell you
whether a score moved because the system changed or because the scorer did — so
everything here is exact-match after normalization, and unit-tested in
`tests/test_evals.py`. The one genuinely fuzzy judgement (groundedness) is
quarantined in `judge.py` and reported as a separate, clearly-labelled number.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# Phrases a grounded model uses when the context does not contain the answer.
# Drawn from the escape hatch in RAG_PROMPT_TEMPLATE ("say you don't know based
# on the provided documents") plus the paraphrases llama3.2 actually emits.
_REFUSAL_MARKERS = (
    "don't know",
    "do not know",
    "not in the context",
    "not mentioned",
    "not specified",
    "not stated",
    "not provided",
    "not available",
    "not included",
    "not contain",
    "doesn't contain",
    "does not contain",
    "no information",
    "not find",
    "cannot determine",
    "can't determine",
    "unable to determine",
    "isn't mentioned",
    "based on the provided documents",
)


def normalize(text: str) -> str:
    """Lowercase and collapse every run of non-alphanumerics to one space.

    This is what makes `26-month`, `26 month`, and `26  Month` compare equal
    without resorting to a fuzzy ratio. Note it also flattens `16,384` to
    `16 384`, which is why the dataset lists both comma'd and bare forms in
    `answer_any_of` rather than relying on the normalizer to guess.
    """
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def contains(haystack: str, needle: str) -> bool:
    """Normalized substring test, padded so `26 month` cannot match `126 months`."""
    h = f" {normalize(haystack)} "
    n = normalize(needle)
    if not n:
        return False
    # Pad only the left: a trailing pad would break `620 gram` matching
    # `620 grams`, which is the exact prefix behaviour we want to keep.
    return f" {n}" in h


def any_of(haystack: str, needles: list[str]) -> bool:
    return any(contains(haystack, n) for n in needles)


def looks_like_refusal(answer: str) -> bool:
    """True if the answer declines rather than asserts.

    Guarded against the false positive that matters: an answer that *states a
    fact* and then adds a caveat ("...the warranty is 26 months, though pricing
    is not mentioned") is not a refusal.
    """
    return any(contains(answer, m) for m in _REFUSAL_MARKERS)


# ── retrieval scoring ────────────────────────────────────────────
@dataclass
class RetrievalScore:
    """How well retrieval did on one case, before any LLM is involved."""

    case_id: str
    kind: str
    hit: bool                  # did any retrieved chunk contain the gold fact?
    rank: int | None           # 1-based position of the first chunk that did
    reciprocal_rank: float     # 1/rank, or 0.0 — averaged into MRR
    top_score: float           # cosine similarity of the best chunk
    hit_score: float | None    # cosine similarity of the chunk that hit
    n_retrieved: int
    sources: list[str] = field(default_factory=list)


def score_retrieval(
    case_id: str,
    kind: str,
    hits: list[dict[str, Any]],
    must_retrieve: list[str],
) -> RetrievalScore:
    """Find the first retrieved chunk containing every gold substring.

    `must_retrieve` is AND-ed: all fragments must appear in the *same* chunk.
    A fact split across two chunks is a chunking failure, and scoring it as a
    hit would hide exactly the bug this metric exists to catch.
    """
    rank: int | None = None
    hit_score: float | None = None
    for i, h in enumerate(hits, start=1):
        text = h.get("text", "")
        if all(contains(text, frag) for frag in must_retrieve):
            rank, hit_score = i, float(h.get("score", 0.0))
            break

    return RetrievalScore(
        case_id=case_id,
        kind=kind,
        hit=rank is not None,
        rank=rank,
        reciprocal_rank=(1.0 / rank) if rank else 0.0,
        top_score=float(hits[0]["score"]) if hits else 0.0,
        hit_score=hit_score,
        n_retrieved=len(hits),
        sources=[h.get("source", "?") for h in hits],
    )


# ── generation scoring ───────────────────────────────────────────
@dataclass
class GenerationScore:
    """How well the end-to-end answer did on one case."""

    case_id: str
    kind: str
    question: str
    answer: str
    correct: bool              # the primary pass/fail for this case
    refused: bool
    partial: bool | None       # `also_expect` satisfied, when the case has one
    grounded: bool | None      # LLM judge verdict; None if judging was skipped
    judge_reason: str = ""
    latency_s: float = 0.0
    retrieval: RetrievalScore | None = None


def score_generation(
    case: dict[str, Any],
    answer: str,
    latency_s: float,
    retrieval: RetrievalScore | None,
) -> GenerationScore:
    """Grade one answer against its case.

    The two case kinds are graded in opposite directions, which is the whole
    point of including `absent` questions: an answerable question is correct
    when it *asserts* the gold fact, and an unanswerable one is correct when it
    *refuses*. A system that scores well on one and badly on the other is either
    hallucinating or uselessly over-cautious, and a single blended accuracy
    number would hide both failures.
    """
    refused = looks_like_refusal(answer)

    if case.get("expect_refusal"):
        correct = refused
        partial = None
    else:
        correct = any_of(answer, case.get("answer_any_of", []))
        extra = case.get("also_expect")
        partial = any_of(answer, extra) if extra else None

    return GenerationScore(
        case_id=case["id"],
        kind=case.get("kind", "fact"),
        question=case["question"],
        answer=answer,
        correct=correct,
        refused=refused,
        partial=partial,
        grounded=None,
        latency_s=latency_s,
        retrieval=retrieval,
    )


# ── aggregation ──────────────────────────────────────────────────
def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def aggregate_retrieval(scores: list[RetrievalScore]) -> dict[str, Any]:
    """Roll per-case retrieval scores into the headline numbers.

    `absent` cases are excluded — they have no gold chunk to retrieve, so
    including them would drag hit-rate down for correct behaviour.
    """
    scored = [s for s in scores if s.kind != "absent"]
    if not scored:
        return {"n": 0, "hit_rate": 0.0, "mrr": 0.0, "mean_top_score": 0.0}

    by_kind: dict[str, dict[str, Any]] = {}
    for s in scored:
        b = by_kind.setdefault(s.kind, {"n": 0, "hits": 0})
        b["n"] += 1
        b["hits"] += int(s.hit)
    for b in by_kind.values():
        b["hit_rate"] = b["hits"] / b["n"]

    return {
        "n": len(scored),
        "hit_rate": _mean([float(s.hit) for s in scored]),
        "mrr": _mean([s.reciprocal_rank for s in scored]),
        "mean_top_score": _mean([s.top_score for s in scored]),
        "mean_hit_score": _mean([s.hit_score for s in scored if s.hit_score is not None]),
        "misses": [s.case_id for s in scored if not s.hit],
        "by_kind": by_kind,
    }


def aggregate_generation(scores: list[GenerationScore]) -> dict[str, Any]:
    """Roll per-case answers up, keeping answerable and absent cases separate."""
    if not scores:
        return {"n": 0}

    answerable = [s for s in scores if s.kind != "absent"]
    absent = [s for s in scores if s.kind == "absent"]
    judged = [s for s in scores if s.grounded is not None]

    return {
        "n": len(scores),
        "accuracy": _mean([float(s.correct) for s in scores]),
        "answerable_n": len(answerable),
        "answerable_accuracy": _mean([float(s.correct) for s in answerable]),
        "absent_n": len(absent),
        "refusal_accuracy": _mean([float(s.correct) for s in absent]),
        # The headline safety number: an answerable question wrongly refused.
        "over_refusal_rate": _mean([float(s.refused) for s in answerable]),
        "groundedness": _mean([float(bool(s.grounded)) for s in judged]) if judged else None,
        "judged_n": len(judged),
        "mean_latency_s": _mean([s.latency_s for s in scores]),
        "failures": [s.case_id for s in scores if not s.correct],
    }
