"""Tests for the CI gate on the eval harness.

The gate decides whether a change is allowed to land, so the thing it must never
do is pass a run that measured nothing. That is not hypothetical: with Ollama
down, every case came back `[engine error: ...]`, the harness scored it 8%
answerable accuracy, printed "retrieval OK, model fumbled → tune
RAG_PROMPT_TEMPLATE or try a larger model", and exited zero. Nothing was wrong
with the prompt or the model. A benchmark that misattributes an infrastructure
failure sends you tuning something that was never broken.
"""
from __future__ import annotations

from evals import EvalReport
from evals import report as eval_report
from evals.metrics import GenerationScore


def score(case_id: str, answer: str, correct: bool = True) -> GenerationScore:
    return GenerationScore(
        case_id=case_id,
        kind="fact",
        question="q",
        answer=answer,
        correct=correct,
        refused=False,
        partial=False,
        grounded=None,
        judge_reason="",
        latency_s=0.1,
        retrieval=None,
    )


def report_with(*answers: str) -> EvalReport:
    report = EvalReport()
    report.generation = [score(f"c{i}", a) for i, a in enumerate(answers)]
    report.generation_summary = {
        "n": len(answers),
        "answerable_accuracy": 1.0,
        "absent_n": 0,
    }
    report.retrieval_summary = {"n": len(answers), "hit_rate": 1.0}
    return report


# ── a run that measured nothing is never a pass ──────────────────
def test_a_dead_model_server_fails_the_gate():
    """Perfect arithmetic, no actual measurement. Exiting zero here is a broken
    build calling itself green."""
    broken = report_with("[engine error: connection refused]")

    assert eval_report.engine_failures(broken) == ["c0"]
    assert eval_report.passed(broken, 0.0) is False


def test_even_one_engine_error_fails_the_gate():
    """A partial outage still means the number is not comparable to the last
    one, which is the only thing a benchmark is for."""
    mixed = report_with("a real answer", "[engine error: timeout]")

    assert eval_report.engine_failures(mixed) == ["c1"]
    assert eval_report.passed(mixed, 0.0) is False


def test_a_healthy_run_passes_on_its_numbers():
    healthy = report_with("a real answer", "another real answer")

    assert eval_report.engine_failures(healthy) == []
    assert eval_report.passed(healthy, 0.95) is True


def test_a_healthy_run_still_fails_a_threshold_it_misses():
    healthy = report_with("a real answer")
    healthy.retrieval_summary = {"n": 12, "hit_rate": 0.5}

    assert eval_report.passed(healthy, 0.95) is False


# ── what the gate actually checks ────────────────────────────────
def test_refusal_accuracy_is_gated_too():
    """The three `absent` cases are the strongest part of the dataset: they test
    that the assistant says "that is not in your documents" instead of inventing
    an answer. Leaving them out of the gate would let abstention rot."""
    report = report_with("a real answer")
    report.generation_summary = {
        "n": 1,
        "answerable_accuracy": 1.0,
        "absent_n": 3,
        "refusal_accuracy": 0.0,
    }

    assert eval_report.passed(report, 0.95) is False


def test_an_empty_report_is_not_a_pass():
    """Nothing measured, nothing proved."""
    assert eval_report.passed(EvalReport(), 0.0) is False
