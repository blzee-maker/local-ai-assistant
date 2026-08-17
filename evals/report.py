"""Rich rendering for an EvalReport — the only module that knows what a run looks like.

Kept out of `runner.py` on purpose: the runner produces plain dataclasses, so
the same run can be printed as a table, dumped as JSON for CI, or diffed against
a previous run without the measurement code caring.
"""
from __future__ import annotations

from typing import Any

from rich.table import Table

from app.cli.render import console

from .runner import EvalReport, cross_tabulate


def _pct(x: float | None) -> str:
    return "—" if x is None else f"{x * 100:.0f}%"


def _verdict(value: float, good: float, ok: float) -> str:
    style = "ok" if value >= good else "warn" if value >= ok else "err"
    return f"[{style}]{_pct(value)}[/{style}]"


def print_config(report: EvalReport) -> None:
    c = report.config
    console.print(
        f"[meta]{c['document']} → {report.chunks_indexed} chunks · "
        f"chunk {c['chunk_size']}/{c['chunk_overlap']} · top_k {c['top_k']} · "
        f"{c['llm']}[/meta]\n"
    )


def print_retrieval(report: EvalReport) -> None:
    s = report.retrieval_summary
    if not s.get("n"):
        return

    table = Table(title="Stage 1 — Retrieval", title_justify="left", header_style="bold")
    table.add_column("case")
    table.add_column("kind", style="dim")
    table.add_column("hit", justify="center")
    table.add_column("rank", justify="right")
    table.add_column("score", justify="right")

    for r in report.retrieval:
        table.add_row(
            r.case_id,
            r.kind,
            "[ok]✓[/ok]" if r.hit else "[err]✗[/err]",
            str(r.rank) if r.rank else "—",
            f"{r.hit_score:.3f}" if r.hit_score is not None else f"[dim]{r.top_score:.3f}[/dim]",
        )
    console.print(table)
    console.print(
        f"  hit@{report.config['top_k']} {_verdict(s['hit_rate'], 0.9, 0.7)}   "
        f"MRR [bold]{s['mrr']:.3f}[/bold]   "
        f"[meta]mean top score {s['mean_top_score']:.3f}[/meta]\n"
    )


def print_generation(report: EvalReport) -> None:
    s = report.generation_summary
    if not s.get("n"):
        return

    table = Table(title="Stage 2 — Generation", title_justify="left", header_style="bold")
    table.add_column("case")
    table.add_column("kind", style="dim")
    table.add_column("ok", justify="center")
    table.add_column("grounded", justify="center")
    table.add_column("latency", justify="right", style="dim")
    table.add_column("answer", overflow="ellipsis", max_width=52)

    for g in report.generation:
        grounded = (
            "[ok]✓[/ok]" if g.grounded is True
            else "[err]✗[/err]" if g.grounded is False
            else "[dim]—[/dim]"
        )
        answer = " ".join(g.answer.split())
        table.add_row(
            g.case_id,
            g.kind,
            "[ok]✓[/ok]" if g.correct else "[err]✗[/err]",
            grounded,
            f"{g.latency_s:.1f}s",
            answer,
        )
    console.print(table)

    console.print(
        f"  answerable accuracy {_verdict(s['answerable_accuracy'], 0.9, 0.7)} "
        f"[meta]({s['answerable_n']} cases)[/meta]   "
        f"refusal accuracy {_verdict(s['refusal_accuracy'], 1.0, 0.66)} "
        f"[meta]({s['absent_n']} cases)[/meta]"
    )
    line = f"  over-refusal {_pct(s['over_refusal_rate'])}"
    if s.get("groundedness") is not None:
        line += (
            f"   groundedness {_pct(s['groundedness'])} "
            f"[meta](judged by {report.config['llm']}, advisory)[/meta]"
        )
    console.print(line)
    console.print(f"  [meta]mean latency {s['mean_latency_s']:.1f}s[/meta]\n")


def print_diagnosis(report: EvalReport) -> None:
    """Turn the two stages into a statement about what to go fix."""
    if not report.generation:
        return
    buckets = cross_tabulate(report)
    if not any(buckets[k] for k in ("retrieval_failed", "generation_failed", "both_failed")):
        console.print("[ok]All answerable cases passed both stages.[/ok]\n")
        return

    console.print("[bold]Diagnosis[/bold]")
    if buckets["generation_failed"]:
        console.print(
            f"  [warn]retrieval OK, model fumbled[/warn] → {', '.join(buckets['generation_failed'])}"
        )
        console.print("    [hint]tune RAG_PROMPT_TEMPLATE or try a larger model[/hint]")
    if buckets["retrieval_failed"]:
        console.print(
            f"  [warn]answered right despite a retrieval miss[/warn] → "
            f"{', '.join(buckets['retrieval_failed'])}"
        )
        console.print("    [hint]likely answered from model priors — not real grounding[/hint]")
    if buckets["both_failed"]:
        console.print(f"  [err]retrieval missed and answer wrong[/err] → {', '.join(buckets['both_failed'])}")
        console.print("    [hint]tune chunk_size/overlap or raise rag_top_k[/hint]")
    console.print()


def print_report(report: EvalReport) -> None:
    print_config(report)
    print_retrieval(report)
    print_generation(report)
    print_diagnosis(report)
    console.print(f"[meta]completed in {report.wall_time_s:.1f}s[/meta]")


def passed(report: EvalReport, threshold: float) -> bool:
    """Gate for CI: every measured headline must clear `threshold`."""
    checks: list[float] = []
    if report.retrieval_summary.get("n"):
        checks.append(report.retrieval_summary["hit_rate"])
    if report.generation_summary.get("n"):
        checks.append(report.generation_summary["answerable_accuracy"])
        if report.generation_summary["absent_n"]:
            checks.append(report.generation_summary["refusal_accuracy"])
    return all(c >= threshold for c in checks) if checks else False
