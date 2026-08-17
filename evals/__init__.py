"""Evaluation harness for the assistant's RAG pipeline.

Run it with `python assistant.py eval` (see evals/README.md for what the
numbers mean and how to act on them).
"""
from .runner import EvalReport, EvalRunner, cross_tabulate, load_dataset

__all__ = ["EvalReport", "EvalRunner", "cross_tabulate", "load_dataset"]
