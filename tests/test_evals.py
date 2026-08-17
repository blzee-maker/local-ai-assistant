"""Tests for the eval harness's own scoring functions.

An eval suite is measuring equipment, and unverified equipment is worse than
none: a scorer with a false-positive bug reports green while the pipeline rots.
Every case below is a way a naive implementation silently mis-scores.

Nothing here needs Ollama or the embedding model — these are pure functions.

Run with:  python -m pytest tests/test_evals.py -v
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals import metrics


# ── normalization ────────────────────────────────────────────────
def test_punctuation_variants_compare_equal():
    """`26-month` and `26 months` must not be scored as a miss."""
    assert metrics.contains("The warranty is 26 months.", "26-month")
    assert metrics.contains("a 26-month limited warranty", "26 month")


def test_case_and_whitespace_are_ignored():
    assert metrics.contains("XTS-AES-256 at rest", "xts aes 256")
    assert metrics.contains("weighs  620   grams", "620 gram")


def test_substring_cannot_match_across_a_longer_number():
    """Guards the classic false positive: `26 month` inside `126 months`."""
    assert not metrics.contains("a 126 month plan", "26 month")


def test_empty_needle_never_matches():
    """Otherwise a case with an empty expectation would silently always pass."""
    assert not metrics.contains("anything at all", "")


# ── refusal detection ────────────────────────────────────────────
@pytest.mark.parametrize(
    "answer",
    [
        "I don't know based on the provided documents.",
        "The price is not mentioned in the context.",
        "That information is not specified.",
        "The provided documents do not contain the CEO's name.",
    ],
)
def test_refusals_are_detected(answer):
    assert metrics.looks_like_refusal(answer)


@pytest.mark.parametrize(
    "answer",
    [
        "The hardware warranty is 26 months [1].",
        "Halden Systems was founded in 2019 by Priya Venkataraman.",
    ],
)
def test_confident_answers_are_not_refusals(answer):
    assert not metrics.looks_like_refusal(answer)


# ── retrieval scoring ────────────────────────────────────────────
def hit(text: str, score: float = 0.5, source: str = "doc.md") -> dict:
    return {"text": text, "score": score, "source": source}


def test_rank_and_reciprocal_rank_track_position():
    hits = [hit("irrelevant"), hit("also irrelevant"), hit("a 26-month warranty")]
    s = metrics.score_retrieval("c", "fact", hits, ["26-month"])
    assert s.hit and s.rank == 3
    assert s.reciprocal_rank == pytest.approx(1 / 3)


def test_a_miss_scores_zero_not_none():
    s = metrics.score_retrieval("c", "fact", [hit("nothing useful")], ["26-month"])
    assert not s.hit and s.rank is None and s.reciprocal_rank == 0.0


def test_fragments_must_land_in_the_same_chunk():
    """A fact split across two chunks is a chunking failure, not a hit."""
    hits = [hit("Priya Venkataraman founded it"), hit("in the year 2019")]
    s = metrics.score_retrieval("c", "fact", hits, ["Priya Venkataraman", "2019"])
    assert not s.hit


def test_empty_retrieval_does_not_crash():
    s = metrics.score_retrieval("c", "fact", [], ["anything"])
    assert not s.hit and s.top_score == 0.0 and s.n_retrieved == 0


# ── generation scoring ───────────────────────────────────────────
def test_answerable_case_is_graded_on_asserting_the_fact():
    case = {"id": "w", "kind": "fact", "question": "?", "answer_any_of": ["26 month"]}
    assert metrics.score_generation(case, "It is 26 months [1].", 1.0, None).correct


def test_answerable_case_fails_when_it_refuses():
    case = {"id": "w", "kind": "fact", "question": "?", "answer_any_of": ["26 month"]}
    score = metrics.score_generation(case, "I don't know.", 1.0, None)
    assert not score.correct and score.refused


def test_absent_case_is_graded_in_the_opposite_direction():
    """A refusal is the *correct* answer when the fact is not in the document."""
    case = {"id": "p", "kind": "absent", "question": "?", "expect_refusal": True}
    assert metrics.score_generation(case, "That is not mentioned.", 1.0, None).correct


def test_absent_case_fails_on_a_hallucinated_number():
    case = {"id": "p", "kind": "absent", "question": "?", "expect_refusal": True}
    assert not metrics.score_generation(case, "It costs $4,200.", 1.0, None).correct


# ── aggregation ──────────────────────────────────────────────────
def gen(case_id, kind, correct, refused=False, grounded=None):
    return metrics.GenerationScore(
        case_id=case_id, kind=kind, question="?", answer="",
        correct=correct, refused=refused, partial=None, grounded=grounded,
    )


def test_absent_cases_are_excluded_from_retrieval_hit_rate():
    """They have no gold chunk, so counting them would understate retrieval."""
    scores = [
        metrics.score_retrieval("a", "fact", [hit("26-month")], ["26-month"]),
        metrics.score_retrieval("b", "absent", [], ["nothing"]),
    ]
    summary = metrics.aggregate_retrieval(scores)
    assert summary["n"] == 1 and summary["hit_rate"] == 1.0


def test_answerable_and_refusal_accuracy_are_reported_separately():
    scores = [
        gen("a", "fact", True), gen("b", "fact", True),
        gen("c", "absent", False),
    ]
    s = metrics.aggregate_generation(scores)
    assert s["answerable_accuracy"] == 1.0
    assert s["refusal_accuracy"] == 0.0
    assert s["accuracy"] == pytest.approx(2 / 3)


def test_over_refusal_counts_only_answerable_cases():
    """Refusing an `absent` question is correct and must not inflate this."""
    scores = [gen("a", "fact", False, refused=True), gen("b", "absent", True, refused=True)]
    assert metrics.aggregate_generation(scores)["over_refusal_rate"] == 1.0


def test_unjudged_cases_are_excluded_from_groundedness():
    """A failed judge call must not be averaged in as a failure."""
    scores = [gen("a", "fact", True, grounded=True), gen("b", "fact", True, grounded=None)]
    s = metrics.aggregate_generation(scores)
    assert s["judged_n"] == 1 and s["groundedness"] == 1.0


# ── the dataset itself ───────────────────────────────────────────
def test_dataset_is_wellformed():
    """Catches the eval author's own mistakes before they become fake scores."""
    data = json.loads(
        (Path(__file__).parent.parent / "evals" / "dataset.json").read_text(encoding="utf-8")
    )
    assert Path(data["document"]).exists()

    seen = set()
    for case in data["cases"]:
        assert case["id"] not in seen, f"duplicate case id: {case['id']}"
        seen.add(case["id"])
        assert case["question"].strip()
        if case.get("expect_refusal"):
            assert case["kind"] == "absent"
        else:
            # An answerable case with no expectation would always pass.
            assert case.get("answer_any_of"), f"{case['id']} has no answer expectation"
            assert case.get("must_retrieve"), f"{case['id']} has no retrieval expectation"


def test_gold_fragments_actually_appear_in_the_source_document():
    """If a `must_retrieve` fragment isn't in the doc, the case is unpassable."""
    root = Path(__file__).parent.parent
    data = json.loads((root / "evals" / "dataset.json").read_text(encoding="utf-8"))
    doc = (root / data["document"]).read_text(encoding="utf-8")
    for case in data["cases"]:
        for fragment in case.get("must_retrieve", []):
            assert metrics.contains(doc, fragment), (
                f"{case['id']}: '{fragment}' is not in {data['document']}"
            )
