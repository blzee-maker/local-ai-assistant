"""Terminal rendering helpers — the only module that knows what output looks like.

Streaming is written straight to stdout rather than through a Rich live region.
A live region re-renders the whole block on every token, which on Windows
terminals flickers badly and mangles wrapped text; a plain incremental write
stays smooth and is what makes generation *feel* fast.
"""
from __future__ import annotations

import sys
from typing import Any

from rich.console import Console
from rich.table import Table
from rich.theme import Theme

THEME = Theme(
    {
        "user": "bold cyan",
        "bot": "bold green",
        "meta": "dim",
        "warn": "yellow",
        "err": "bold red",
        "ok": "green",
        "hint": "dim italic",
    }
)

console = Console(theme=THEME, soft_wrap=False)
err_console = Console(theme=THEME, stderr=True)


def stream_token(text: str) -> None:
    """Write one token with no markup interpretation and no buffering delay."""
    sys.stdout.write(text)
    sys.stdout.flush()


def end_stream() -> None:
    sys.stdout.write("\n")
    sys.stdout.flush()


def fmt_duration(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    return f"{seconds:.2f}s"


def metrics_line(metrics: dict[str, Any]) -> str:
    """One-line latency strip: what the user felt, plus decode throughput."""
    parts = [str(metrics.get("model") or "?")]

    ttft = metrics.get("time_to_first_token_s")
    if ttft is not None:
        parts.append(f"first token {fmt_duration(ttft)}")

    tps = metrics.get("tokens_per_second")
    if tps:
        parts.append(f"{tps:.1f} tok/s")

    completion = metrics.get("completion_tokens")
    prompt = metrics.get("prompt_tokens")
    if completion is not None:
        parts.append(f"{completion} out / {prompt or 0} in")

    total = metrics.get("total_duration_s")
    if total is not None:
        parts.append(f"total {fmt_duration(total)}")

    load = metrics.get("load_duration_s")
    # Only worth showing when the model actually cold-loaded.
    if load and load > 0.5:
        parts.append(f"[warn]model load {fmt_duration(load)}[/warn]")

    return "  ·  ".join(parts)


def print_metrics(metrics: dict[str, Any] | None) -> None:
    if metrics:
        console.print(f"[meta]{metrics_line(metrics)}[/meta]")


def print_sources(sources: list[dict[str, Any]]) -> None:
    if not sources:
        return
    console.print("[meta]sources:[/meta]")
    for s in sources:
        console.print(
            f"  [meta][{s['n']}] {s['source']}  (score {s['score']:.3f})[/meta]"
        )


def print_opened_file(opened: dict[str, Any] | None) -> None:
    if not opened:
        return
    console.print(
        f"[meta]opened {opened['name']} from {opened['root']} "
        f"· modified {opened['modified']} · {opened['chars']:,} chars[/meta]"
    )


def documents_table(docs: list[dict[str, Any]], total_chunks: int) -> Table:
    table = Table(title=f"Indexed documents — {total_chunks} chunks", title_style="bot")
    table.add_column("Source")
    table.add_column("Chunks", justify="right")
    for d in sorted(docs, key=lambda x: -x["chunks"]):
        table.add_row(d["source"], str(d["chunks"]))
    return table


def files_table(results: list[dict[str, Any]]) -> Table:
    table = Table(title=f"{len(results)} matching file(s)", title_style="bot")
    table.add_column("#", justify="right", style="meta")
    table.add_column("Name")
    table.add_column("Folder", style="meta")
    table.add_column("Size", justify="right")
    table.add_column("Modified", style="meta")
    for i, r in enumerate(results, 1):
        table.add_row(
            str(i), r["name"], r["root"], f"{r['size_kb']:,.1f} KB", r["modified"]
        )
    return table
