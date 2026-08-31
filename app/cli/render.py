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

def _force_utf8() -> None:
    """Make stdout and stderr able to carry the characters we print.

    Windows gives a piped stdout the locale codepage, cp1252 here, so the tick
    in "check passed" is unencodable and the *diagnostic tool* dies with a
    UnicodeEncodeError — `assistant doctor | tail` crashed while the same
    command in a console worked. A checker that cannot survive being redirected
    is not much of a checker.

    errors="replace" rather than "strict": a stray character in a file name
    should degrade to a question mark, never end the command (rule 10).
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue  # pythonw with no stdout at all, or an odd wrapper
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass


_force_utf8()

console = Console(theme=THEME, soft_wrap=False)
err_console = Console(theme=THEME, stderr=True)


# Wide enough to cover the longest phase name plus a count. The padding is the
# point: a carriage return moves the cursor back but erases nothing, so a short
# message after a long one leaves the tail of the long one on screen.
TRANSIENT_WIDTH = 64

# Whether a rewritten line is currently sitting on screen unwiped.
#
# Tracked here rather than left to callers because the previous design made
# wiping the *tool's* job — the line was only cleared when a tool reported the
# literal message "done", and exactly one tool in the codebase ever did. Every
# other tool left its progress line behind for the next write to land on top
# of, which is how "Opened Spotify." over "...opening Spotify" rendered as
# "Opened Spotify.g Spotify". A tool that forgets to announce it finished must
# not be able to corrupt the screen.
_transient_pending = False


def transient(message: str) -> None:
    """Draw a line that the next write is expected to overwrite."""
    global _transient_pending
    sys.stdout.write(f"  {message:<{TRANSIENT_WIDTH - 2}}\r")
    sys.stdout.flush()
    _transient_pending = True


def clear_transient() -> None:
    """Wipe a pending rewritten line, if there is one.

    Safe to call unconditionally — that is the whole point. Every path that
    prints real output calls this first, so correctness no longer depends on
    each tool remembering to send a completion message.
    """
    global _transient_pending
    if not _transient_pending:
        return
    sys.stdout.write(" " * TRANSIENT_WIDTH + "\r")
    sys.stdout.flush()
    _transient_pending = False


def transient_pending() -> bool:
    """Whether a rewritten line is still on screen. For tests."""
    return _transient_pending


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
    if metrics.get("determined"):
        # No generation happened — a latency strip would be meaningless.
        return "answered directly (no model call)"
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
    if not metrics:
        return
    # Say it plainly when a weaker model answered — otherwise a degraded reply
    # reads as the main model's best effort.
    if metrics.get("fell_back"):
        console.print(
            f"[warn]The main model could not be loaded; answered with "
            f"{metrics.get('model')} instead.[/warn]"
        )
    console.print(f"[meta]{metrics_line(metrics)}[/meta]")


def print_sources(sources: list[dict[str, Any]]) -> None:
    if not sources:
        return
    console.print("[meta]sources:[/meta]")
    for s in sources:
        console.print(
            f"  [meta][{s['n']}] {s['source']}  (score {s['score']:.3f})[/meta]"
        )


def print_tool_used(used: dict[str, Any] | None) -> None:
    """Show what a tool did. Rule 4: an action nobody saw is indistinguishable
    from one that never happened."""
    if not used:
        return
    display = used.get("display") or used.get("tool") or ""
    if not display:
        return
    style = "warn" if used.get("failed") else "meta"
    console.print(f"[{style}]{display}[/{style}]")


def print_recalled(recalled: dict[str, Any] | None) -> None:
    """Name the facts that were consulted. An assistant that silently reads a
    private dossier about you is worse than one that shows its working."""
    if not recalled:
        return
    console.print(f"[meta]{recalled['display']}:[/meta]")
    for text, score in zip(recalled.get("texts", []), recalled.get("scores", [])):
        console.print(f"  [meta]· {text}  ({score:.2f})[/meta]")


def memories_table(memories: list[Any]) -> Table:
    table = Table(title=f"{len(memories)} remembered fact(s)", title_style="bot")
    table.add_column("#", justify="right", style="meta")
    table.add_column("Fact")
    table.add_column("Stored", style="meta")
    table.add_column("Used", justify="right", style="meta")
    for memory in memories:
        age = (
            "today" if memory.age_days < 1
            else f"{memory.age_days:.0f}d ago"
        )
        table.add_row(str(memory.id), memory.text, age, str(memory.used_count))
    return table


def sessions_table(sessions: list[Any]) -> Table:
    import datetime

    table = Table(title=f"{len(sessions)} conversation(s)", title_style="bot")
    table.add_column("#", justify="right", style="meta")
    table.add_column("Started", style="meta")
    table.add_column("Messages", justify="right")
    table.add_column("First message")
    for info in sessions:
        started = datetime.datetime.fromtimestamp(info.started_at).strftime("%m-%d %H:%M")
        title = (info.title or "")[:60]
        table.add_row(str(info.id), started, str(info.message_count), title)
    return table


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
