"""Command-line entry point.

`chat` is the interactive front door; everything else is a one-shot command so
the assistant composes with other tools (`assistant ask "..." | less`).
"""
from __future__ import annotations

import json
import sys
import time

import typer

from app.cli import render
from app.cli.render import console, err_console
from app.core import Assistant
from app.engines import ChatMessage
from config import settings

cli = typer.Typer(
    name="assistant",
    help="Local offline AI assistant — chat, documents, and disk analysis.",
    no_args_is_help=True,
    add_completion=False,
)


def _assistant() -> Assistant:
    return Assistant()


@cli.command()
def chat(
    rag: bool = typer.Option(False, "--rag", help="Ground answers in indexed documents."),
    files: bool = typer.Option(
        True, "--files/--no-files", help="Let the model open local files by name."
    ),
    model: str = typer.Option(None, "--model", "-m", help="Model for this session."),
    speak: bool = typer.Option(False, "--speak", help="Read replies aloud."),
    resume: bool = typer.Option(False, "--resume", "-r", help="Continue the last conversation."),
) -> None:
    """Start an interactive chat session."""
    from app.cli.repl import Session, run

    run(
        _assistant(),
        Session(use_rag=rag, allow_files=files, model=model, speak=speak),
        resume=resume,
    )


@cli.command()
def ask(
    prompt: list[str] = typer.Argument(None, help="Your question. Reads stdin if omitted."),
    rag: bool = typer.Option(False, "--rag", help="Ground the answer in indexed documents."),
    files: bool = typer.Option(
        True, "--files/--no-files", help="Let the model open local files by name."
    ),
    model: str = typer.Option(None, "--model", "-m", help="Model to use."),
    as_json: bool = typer.Option(False, "--json", help="Emit one JSON object instead of prose."),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Answer only — no metrics."),
) -> None:
    """Ask a single question and exit. Pipe-friendly."""
    text = " ".join(prompt).strip() if prompt else ""
    if not text and not sys.stdin.isatty():
        from app.cli.repl import clean_input

        text = clean_input(sys.stdin.read())
    if not text:
        err_console.print("[err]Nothing to ask.[/err] [hint]assistant ask \"...\"[/hint]")
        raise typer.Exit(1)

    assistant = _assistant()
    history = [ChatMessage(role="user", content=text)]

    collected: list[str] = []
    terminal = None
    for event in assistant.chat_stream(
        history, use_rag=rag, allow_file_access=files, model=model
    ):
        if event.type == "token":
            collected.append(event.text)
            if not as_json:
                render.stream_token(event.text)
        else:
            terminal = event

    reply = "".join(collected)

    if terminal is not None and terminal.type == "error":
        if not as_json:
            err_console.print(f"\n[err]{terminal.error}[/err]")
        else:
            print(json.dumps({"ok": False, "error": terminal.error}))
        raise typer.Exit(1)

    if as_json:
        print(
            json.dumps(
                {
                    "ok": True,
                    "answer": reply,
                    "sources": terminal.sources if terminal else [],
                    "tool_used": terminal.tool_used if terminal else None,
                    "recalled": terminal.recalled if terminal else None,
                    "metrics": terminal.metrics if terminal else None,
                },
                indent=2,
            )
        )
        return

    render.end_stream()
    if not quiet and terminal is not None:
        render.print_recalled(terminal.recalled)
        render.print_tool_used(terminal.tool_used)
        render.print_sources(terminal.sources)
        render.print_metrics(terminal.metrics)


@cli.command()
def ingest(
    paths: list[str] = typer.Argument(..., help="File(s) to index. Must be in an allowed folder."),
) -> None:
    """Index local files so `--rag` can ground answers in them."""
    assistant = _assistant()
    failures = 0
    for path in paths:
        result = assistant.ingest_path(path)
        if result.get("ok"):
            console.print(
                f"[ok]✓[/ok] {result['source']} — {result['chunks']} chunks "
                f"[meta]({result['total_chunks']} total)[/meta]"
            )
        else:
            failures += 1
            console.print(f"[err]✗[/err] {path}: {result.get('error')}")
    if failures:
        raise typer.Exit(1)


@cli.command()
def docs(
    reset: bool = typer.Option(False, "--reset", help="Delete everything from the index."),
) -> None:
    """List indexed documents."""
    assistant = _assistant()
    if reset:
        assistant.reset_documents()
        console.print("[ok]document index cleared[/ok]")
        return
    entries = assistant.documents()
    if not entries:
        console.print("[warn]nothing indexed yet[/warn] [hint]assistant ingest <path>[/hint]")
        return
    console.print(render.documents_table(entries, assistant.document_chunks))


@cli.command()
def find(
    query: list[str] = typer.Argument(None, help="Filename fragment to search for."),
) -> None:
    """Search the folders the assistant is allowed to read."""
    results = _assistant().search_files(" ".join(query) if query else "")
    if not results:
        console.print("[warn]no matching files[/warn]")
        raise typer.Exit(1)
    console.print(render.files_table(results))


@cli.command()
def say(
    text: list[str] = typer.Argument(None, help="Text to speak. Reads stdin if omitted."),
    out: str = typer.Option(None, "--out", "-o", help="Write a .wav file instead of playing."),
) -> None:
    """Speak text aloud with Piper (offline TTS)."""
    from app.cli import audio

    words = " ".join(text).strip() if text else ""
    if not words and not sys.stdin.isatty():
        from app.cli.repl import clean_input

        words = clean_input(sys.stdin.read())
    if not words:
        err_console.print("[err]Nothing to say.[/err]")
        raise typer.Exit(1)

    wav, summarized = _assistant().speak(words)
    if summarized:
        console.print("[meta](condensed to a spoken summary)[/meta]")

    if out:
        from pathlib import Path

        Path(out).write_bytes(wav)
        console.print(f"[ok]wrote {out}[/ok]")
        return

    if not audio.available():
        err_console.print(
            "[err]No audio output.[/err] [hint]use --out to write a .wav instead[/hint]"
        )
        raise typer.Exit(1)
    audio.play_wav(wav)


@cli.command()
def listen(
    ask_it: bool = typer.Option(False, "--ask", help="Send the transcript to the model."),
) -> None:
    """Record from the microphone and transcribe it (offline Whisper)."""
    from app.cli.repl import Session, record_question, run_turn

    assistant = _assistant()
    text = record_question(assistant)
    if not text:
        raise typer.Exit(1)

    if not ask_it:
        print(text)
        return

    console.print(f"[user]you (voice)[/user] {text}")
    run_turn(assistant, Session(), text)


@cli.command()
def scan(
    export: str = typer.Option(None, "--export", "-e", help="Write a Markdown report."),
    as_json: bool = typer.Option(False, "--json", help="Emit the report as JSON."),
    no_integrity: bool = typer.Option(False, "--no-integrity", help="Skip format checks."),
    no_cache: bool = typer.Option(False, "--no-cache", help="Re-hash everything."),
    top: int = typer.Option(10, "--top", "-n", help="Rows per section."),
    cleanup_script: str = typer.Option(
        None, "--cleanup-script", help="Write a reviewable delete script (never runs it)."
    ),
) -> None:
    """Analyse your allowed folders for duplicates, corruption, and idle storage."""
    from pathlib import Path

    from rich.progress import (
        BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn,
    )

    from app.analyzer import report as report_mod
    from app.analyzer import run_scan
    from app.consent import ensure_consent

    if not ensure_consent(console):
        raise typer.Exit(1)

    if as_json:
        result = run_scan(check_integrity=not no_integrity, use_cache=not no_cache)
    else:
        with Progress(
            SpinnerColumn(),
            TextColumn("[meta]{task.description}[/meta]"),
            BarColumn(bar_width=28),
            TextColumn("[meta]{task.completed}/{task.total}[/meta]"),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("Starting", total=None)

            def on_phase(name: str) -> None:
                progress.update(task, description=name, completed=0, total=None)

            def on_progress(done: int, total: int) -> None:
                progress.update(task, completed=done, total=total or None)

            result = run_scan(
                check_integrity=not no_integrity,
                use_cache=not no_cache,
                on_phase=on_phase,
                on_progress=on_progress,
            )

    # Cache the findings so `ask`/`chat` can answer disk questions without
    # re-scanning (which would take tens of seconds inside a chat turn).
    from app.core import diskintent

    diskintent.save_report(result)

    if as_json:
        print(json.dumps(result.to_dict(), indent=2, default=str))
        return

    report_mod.render_console(result, console, top=top)
    console.print(
        "[meta]You can now ask about this in chat, e.g. "
        '"which duplicate files are wasting the most space?"[/meta]'
    )

    if export:
        Path(export).write_text(report_mod.to_markdown(result), encoding="utf-8")
        console.print(f"\n[ok]report written to {export}[/ok]")

    if cleanup_script:
        target = Path(cleanup_script)
        count = report_mod.deletion_script(result, target)
        console.print(
            f"[ok]cleanup script written to {target}[/ok] "
            f"[meta]{count} file(s) listed — review it before running[/meta]"
        )


@cli.command(name="eval")
def eval_cmd(
    retrieval_only: bool = typer.Option(
        False, "--retrieval-only", help="Skip the LLM. Fast, deterministic, CI-friendly."
    ),
    no_judge: bool = typer.Option(
        False, "--no-judge", help="Skip LLM-as-judge groundedness scoring (roughly 2x faster)."
    ),
    top_k: int = typer.Option(None, "--top-k", help="Override rag_top_k for this run."),
    model: str = typer.Option(None, "--model", "-m", help="Model under test."),
    as_json: bool = typer.Option(False, "--json", help="Emit the report as JSON."),
    threshold: float = typer.Option(
        None, "--threshold", help="Exit non-zero if any headline metric falls below this (0-1)."
    ),
) -> None:
    """Benchmark the RAG pipeline against the golden dataset in evals/."""
    from rich.progress import Progress, SpinnerColumn, TextColumn

    from evals import EvalRunner
    from evals import report as eval_report

    try:
        with EvalRunner(top_k=top_k, model=model) as runner:
            if retrieval_only or as_json:
                result = runner.run(retrieval_only=retrieval_only, judge=not no_judge)
            else:
                total = len(runner.dataset["cases"])
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[meta]{task.description}[/meta]"),
                    console=console,
                    transient=True,
                ) as progress:
                    task = progress.add_task("Starting", total=total)
                    done = 0

                    def on_case(case_id: str) -> None:
                        nonlocal done
                        done += 1
                        progress.update(
                            task, description=f"{case_id} ({done}/{total})", completed=done
                        )

                    result = runner.run(judge=not no_judge, on_case=on_case)
    except FileNotFoundError as exc:
        console.print(f"[err]✗[/err] {exc}")
        raise typer.Exit(1)

    if as_json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        eval_report.print_report(result)

    if threshold is not None and not eval_report.passed(result, threshold):
        if not as_json:
            console.print(f"[err]below threshold {threshold:.0%}[/err]")
        raise typer.Exit(1)


@cli.command()
def models() -> None:
    """List models available from the local inference backend."""
    health = _assistant().health()
    if not health["healthy"]:
        err_console.print(f"[err]{health['engine']} backend unreachable[/err]")
        raise typer.Exit(1)
    for name in health["models"]:
        marker = "[ok]*[/ok]" if name == health["model"] else " "
        console.print(f" {marker} {name}")


@cli.command()
def consent(
    grant: bool = typer.Option(False, "--grant", help="Approve local file analysis."),
    revoke: bool = typer.Option(False, "--revoke", help="Withdraw approval."),
) -> None:
    """Show or change permission for the assistant to analyse your files."""
    from app import consent as consent_mod

    if grant and revoke:
        err_console.print("[err]--grant and --revoke are mutually exclusive[/err]")
        raise typer.Exit(1)

    if revoke:
        consent_mod.revoke()
        console.print("[ok]Approval withdrawn.[/ok] [meta]You'll be asked again next time.[/meta]")
        return

    if grant:
        record = consent_mod.grant()
        console.print("[ok]File analysis approved for:[/ok]")
        for root in record.approved_roots:
            console.print(f"  [meta]{root}[/meta]")
        return

    state, record = consent_mod.status()
    labels = {
        "none": "[warn]not yet asked[/warn]",
        "granted": "[ok]granted[/ok]",
        "declined": "[err]declined[/err]",
        "stale": "[warn]needs re-approval — the folder list changed[/warn]",
    }
    console.print(f"Status: {labels[state]}")
    if record and record.approved_roots:
        import datetime

        when = datetime.datetime.fromtimestamp(record.decided_at).strftime("%Y-%m-%d %H:%M")
        console.print(f"[meta]decided {when}[/meta]")
        for root in record.approved_roots:
            console.print(f"  [meta]{root}[/meta]")


daemon_cli = typer.Typer(
    name="daemon",
    help="Background jobs: indexing new files, watching health, periodic scans.",
    no_args_is_help=True,
    add_completion=False,
)
cli.add_typer(daemon_cli)


def _daemon_printer():
    """Render scheduler events as they happen."""
    styles = {
        "start": "meta", "ok": "ok", "skip": "meta",
        "error": "err", "notable": "warn", "info": "meta", "stop": "meta",
    }

    def emit(level: str, message: str) -> None:
        import datetime

        stamp = datetime.datetime.now().strftime("%H:%M:%S")
        style = styles.get(level, "meta")
        prefix = "!" if level == "notable" else "·"
        console.print(f"[meta]{stamp}[/meta] {prefix} [{style}]{message}[/{style}]")

    return emit


@daemon_cli.command("run")
def daemon_run(
    once: bool = typer.Option(False, "--once", help="Run everything due, then exit."),
    tick: float = typer.Option(None, "--tick", help="Seconds between checks."),
    watch: bool = typer.Option(
        True, "--watch/--no-watch",
        help="Also index Downloads instantly via filesystem events.",
    ),
) -> None:
    """Start the background daemon in the foreground (Ctrl-C to stop)."""
    from app.daemon import Scheduler

    scheduler = Scheduler(_assistant(), on_event=_daemon_printer())

    if once:
        ran = scheduler.tick()
        if not ran:
            console.print("[meta]nothing was due[/meta]")
        return

    console.print("[bot]Assistant daemon[/bot] [meta]Ctrl-C to stop[/meta]")
    scheduler.run_forever(tick_seconds=tick, watch=watch)


@daemon_cli.command("once")
def daemon_once(
    job: str = typer.Argument(..., help="Job name, or 'all' for everything due."),
) -> None:
    """Run a single job now, ignoring its schedule."""
    from app.daemon import Scheduler

    scheduler = Scheduler(_assistant(), on_event=_daemon_printer())

    if job == "all":
        if not scheduler.tick():
            console.print("[meta]nothing was due[/meta]")
        return

    target = next((j for j in scheduler.jobs if j.name == job), None)
    if target is None:
        err_console.print(
            f"[err]no such job: {job}[/err] "
            f"[hint]known: {', '.join(j.name for j in scheduler.jobs)}[/hint]"
        )
        raise typer.Exit(1)

    result = scheduler.run_job(target)
    raise typer.Exit(0 if result.outcome in {"ok", "skipped"} else 1)


@daemon_cli.command("status")
def daemon_status(
    history: int = typer.Option(0, "--history", "-n", help="Also show recent runs."),
) -> None:
    """Show what is scheduled, when it last ran, and what it found."""
    import datetime

    from rich.table import Table

    from app.daemon import Scheduler

    scheduler = Scheduler(_assistant())
    journal = scheduler.journal
    now = time.time()

    table = Table(title="Scheduled jobs", title_style="bot")
    table.add_column("Job")
    table.add_column("Every", justify="right", style="meta")
    table.add_column("Last run", style="meta")
    table.add_column("Next", justify="right", style="meta")
    table.add_column("Outcome")

    for job in scheduler.jobs:
        state = journal.state(job.name)
        due = scheduler.next_due(job, now)

        if job.interval_s >= 3600:
            every = f"{job.interval_s / 3600:.0f}h"
        else:
            every = f"{job.interval_s / 60:.0f}m"

        if state.last_run is None:
            last = "never"
        else:
            last = datetime.datetime.fromtimestamp(state.last_run).strftime("%m-%d %H:%M")

        remaining = due - now
        nxt = "due now" if remaining <= 0 else (
            f"{remaining / 3600:.1f}h" if remaining >= 3600 else f"{remaining / 60:.0f}m"
        )

        outcome = state.last_outcome or "—"
        style = {"ok": "ok", "skipped": "meta", "error": "err", "failed": "err"}.get(
            outcome, "meta"
        )
        detail = f" [meta]{state.last_detail}[/meta]" if state.last_detail else ""
        table.add_row(job.name, every, last, nxt, f"[{style}]{outcome}[/{style}]{detail}")

    console.print(table)

    if history:
        runs = journal.history(limit=history)
        if runs:
            recent = Table(title=f"Last {len(runs)} run(s)", title_style="bot")
            recent.add_column("When", style="meta")
            recent.add_column("Job")
            recent.add_column("Took", justify="right", style="meta")
            recent.add_column("Outcome")
            recent.add_column("Detail", style="meta", overflow="fold")
            for run in runs:
                when = datetime.datetime.fromtimestamp(run.at).strftime("%m-%d %H:%M:%S")
                style = {"ok": "ok", "skipped": "meta"}.get(run.outcome, "err")
                recent.add_row(
                    when, run.job, f"{run.duration:.1f}s",
                    f"[{style}]{run.outcome}[/{style}]", run.detail or "",
                )
            console.print(recent)


@daemon_cli.command("briefing")
def daemon_briefing(
    hours: float = typer.Option(24.0, "--hours", help="How far back to summarise."),
    speak: bool = typer.Option(False, "--speak", help="Read the briefing aloud."),
) -> None:
    """Summarise what the daemon did while you were away."""
    from app.daemon import Journal, briefing

    with Journal() as journal:
        text = briefing(journal, since_hours=hours)

    console.print(f"\n[bot]Briefing — last {hours:.0f}h[/bot]")
    for line in text.splitlines():
        console.print(f"  {line}")

    if speak:
        from app.cli import audio

        if not audio.available():
            err_console.print("[err]no audio output[/err]")
            raise typer.Exit(1)
        wav, _summarized = _assistant().speak(text)
        audio.play_wav(wav)


@cli.command()
def system(
    processes: int = typer.Option(10, "--processes", "-p", help="Top processes to show."),
    as_json: bool = typer.Option(False, "--json", help="Emit the snapshot as JSON."),
) -> None:
    """Show what this machine is doing right now."""
    from rich.table import Table

    from app.tools.system import sample_processes, system_snapshot

    snapshot = system_snapshot()

    if as_json:
        procs = sample_processes(limit=processes)
        print(json.dumps({**snapshot, "processes": [p.__dict__ for p in procs]}, indent=2))
        return

    memory_info = snapshot["memory"]
    used_style = "err" if memory_info["percent_used"] >= 85 else "ok"

    console.print("\n[bot]System[/bot]")
    console.print(
        f"  CPU        {snapshot['cpu_percent']:>5.0f}%  "
        f"[meta]{snapshot['cpu_count']} cores[/meta]"
    )
    console.print(
        f"  Memory     [{used_style}]{memory_info['percent_used']:>5.0f}%[/{used_style}]  "
        f"[meta]{memory_info['available_gb']:.2f} GB free of "
        f"{memory_info['total_gb']:.2f} GB[/meta]"
    )
    if snapshot["swap"]["total_gb"]:
        console.print(
            f"  Swap       {snapshot['swap']['percent_used']:>5.0f}%  "
            f"[meta]{snapshot['swap']['total_gb']:.1f} GB total[/meta]"
        )
    battery = snapshot.get("battery")
    if battery:
        state = "charging" if battery["plugged_in"] else "on battery"
        left = f", ~{battery['minutes_left']} min left" if battery["minutes_left"] else ""
        console.print(f"  Battery    {battery['percent']:>5}%  [meta]{state}{left}[/meta]")
    console.print(f"  Uptime     {snapshot['uptime_hours']:>5.1f}h")

    for disk in snapshot["disks"]:
        style = "err" if disk["percent_used"] >= 90 else "meta"
        console.print(
            f"  {disk['mount']:<10} [{style}]{disk['percent_used']:>5.0f}%[/{style}]  "
            f"[meta]{disk['free_gb']:.1f} GB free of {disk['total_gb']:.1f} GB[/meta]"
        )

    procs = sample_processes(limit=processes)
    table = Table(title=f"Top {len(procs)} processes", title_style="bot")
    table.add_column("PID", justify="right", style="meta")
    table.add_column("Name")
    table.add_column("CPU", justify="right")
    table.add_column("Memory", justify="right")
    for proc in procs:
        table.add_row(
            str(proc.pid), proc.name,
            f"{proc.cpu_percent:.1f}%", f"{proc.memory_mb:,.0f} MB",
        )
    console.print()
    console.print(table)

    if memory_info["percent_used"] >= 85:
        console.print(
            "[warn]Memory is nearly exhausted — the most likely cause of "
            "slowness on this machine.[/warn]"
        )


@cli.command()
def memory(
    remember: str = typer.Option(None, "--remember", help="Store a fact."),
    forget: str = typer.Option(None, "--forget", help="Delete a fact by id, or 'all'."),
    sessions: bool = typer.Option(False, "--sessions", help="List saved conversations."),
    show: int = typer.Option(None, "--show", help="Replay a saved conversation by id."),
) -> None:
    """Show or change what the assistant remembers about you."""
    assistant = _assistant()
    mem = assistant.memory

    if remember:
        stored, outcome = mem.remember(remember)
        if stored is None:
            err_console.print(f"[err]{outcome}[/err]")
            raise typer.Exit(1)
        console.print(f"[ok]{outcome}[/ok] [meta]#{stored.id} {stored.text}[/meta]")
        return

    if forget:
        if forget.lower() == "all":
            count = mem.forget_all()
            console.print(f"[ok]forgot {count} fact(s)[/ok]")
        elif forget.isdigit():
            if mem.forget(int(forget)):
                console.print(f"[ok]forgot #{forget}[/ok]")
            else:
                console.print(f"[warn]no memory #{forget}[/warn]")
                raise typer.Exit(1)
        else:
            err_console.print("[err]--forget takes an id or 'all'[/err]")
            raise typer.Exit(1)
        return

    if sessions:
        saved = mem.sessions()
        if not saved:
            console.print("[warn]no saved conversations[/warn]")
            return
        console.print(render.sessions_table(saved))
        console.print("[meta]assistant memory --show <#>  ·  assistant chat --resume[/meta]")
        return

    if show is not None:
        stored = mem.store.messages(show)
        if not stored:
            console.print(f"[warn]no conversation #{show}[/warn]")
            raise typer.Exit(1)
        for message in stored:
            label = "you" if message.role == "user" else "assistant"
            style = "user" if message.role == "user" else "bot"
            console.print(f"[{style}]{label}[/{style}] {message.content[:500]}\n")
        return

    facts = mem.memories()
    if not facts:
        console.print(
            "[warn]nothing remembered yet[/warn] "
            '[hint]assistant memory --remember "..."[/hint]'
        )
        console.print(
            "[meta]Facts are only stored when you ask — nothing is inferred in "
            "the background.[/meta]"
        )
        return
    console.print(render.memories_table(facts))
    console.print("[meta]assistant memory --forget <id|all>[/meta]")


@cli.command()
def tools(
    grant: str = typer.Option(None, "--grant", help="Permit a tool by name."),
    revoke: str = typer.Option(None, "--revoke", help="Withdraw a tool's permission."),
    revoke_all: bool = typer.Option(False, "--revoke-all", help="Withdraw every permission."),
    audit: bool = typer.Option(False, "--audit", help="Show what tools have done."),
    limit: int = typer.Option(20, "--limit", "-n", help="Audit entries to show."),
) -> None:
    """List the assistant's capabilities, their permissions, and their history."""
    import datetime

    from rich.table import Table

    from app.tools import Risk, ToolLedger, default_tools

    ledger = ToolLedger()

    if revoke_all:
        count = ledger.revoke()
        console.print(f"[ok]withdrew {count} permission(s)[/ok]")
        return

    if revoke:
        count = ledger.revoke(revoke)
        if count:
            console.print(f"[ok]permission for {revoke} withdrawn[/ok]")
        else:
            console.print(f"[warn]no recorded permission for {revoke}[/warn]")
        return

    if grant:
        registered = {t.name: t for t in default_tools()}
        tool = registered.get(grant)
        if tool is None:
            err_console.print(
                f"[err]no such tool: {grant}[/err] "
                f"[hint]known: {', '.join(sorted(registered))}[/hint]"
            )
            raise typer.Exit(1)
        ledger.record_decision(tool.name, tool.risk.value, True)
        console.print(f"[ok]{grant} permitted[/ok]")
        return

    if audit:
        entries = ledger.history(limit=limit)
        if not entries:
            console.print("[warn]no tool activity recorded yet[/warn]")
            return
        table = Table(title=f"Last {len(entries)} tool action(s)", title_style="bot")
        table.add_column("When", style="meta")
        table.add_column("Tool")
        table.add_column("Risk", style="meta")
        table.add_column("Via", style="meta")
        table.add_column("Outcome")
        for entry in entries:
            when = datetime.datetime.fromtimestamp(entry.at).strftime("%m-%d %H:%M")
            style = {"ok": "ok", "denied": "warn", "error": "err"}.get(
                entry.outcome, "meta"
            )
            table.add_row(
                when, entry.tool, entry.risk, entry.source,
                f"[{style}]{entry.outcome}[/{style}]",
            )
        console.print(table)
        return

    decisions = dict((name, granted) for name, granted, _r, _d in ledger.grants())
    table = Table(title="Capabilities", title_style="bot")
    table.add_column("Tool")
    table.add_column("Risk")
    table.add_column("Permission")
    table.add_column("What it does", style="meta")

    risk_style = {Risk.READ: "meta", Risk.WRITE: "warn", Risk.DESTRUCTIVE: "err"}
    for tool in sorted(default_tools(), key=lambda t: t.name):
        if tool.risk.needs_confirmation:
            # "not required" would be true but badly misleading here: there is no
            # standing grant precisely because every single call is confirmed.
            permission = "[warn]asks every time[/warn]"
        elif not tool.risk.needs_consent:
            permission = "[meta]not required[/meta]"
        elif decisions.get(tool.name) is True:
            permission = "[ok]granted[/ok]"
        elif decisions.get(tool.name) is False:
            permission = "[err]declined[/err]"
        else:
            permission = "[warn]will ask[/warn]"
        table.add_row(
            tool.name,
            f"[{risk_style[tool.risk]}]{tool.risk.value}[/{risk_style[tool.risk]}]",
            permission,
            tool.description,
        )
    console.print(table)
    console.print("[meta]assistant tools --audit  ·  --grant <name>  ·  --revoke <name>[/meta]")


@cli.command()
def doctor() -> None:
    """Check that every piece of the stack is present and reachable."""
    from app import files as filesvc

    ok = True
    console.print("[bot]Local AI Assistant — diagnostics[/bot]\n")

    assistant = _assistant()
    health = assistant.health()
    if health["healthy"]:
        console.print(f"[ok]✓[/ok] {health['engine']} reachable at {settings.ollama_host}")
        if health["model"] in health["models"]:
            console.print(f"[ok]✓[/ok] model {health['model']} is pulled")
        else:
            ok = False
            console.print(
                f"[err]✗[/err] model {health['model']} not pulled "
                f"[hint]ollama pull {health['model']}[/hint]"
            )

        # The fallback is only a safety net if it is actually on disk.
        fallback = settings.fallback_model
        if not fallback:
            console.print("[meta]·[/meta] no fallback model configured")
        elif fallback in health["models"]:
            console.print(f"[ok]✓[/ok] fallback model {fallback} is pulled")
        else:
            console.print(
                f"[warn]![/warn] fallback model {fallback} is not pulled "
                f"[hint]ollama pull {fallback}[/hint] "
                "[meta](no safety net if the main model runs out of memory)[/meta]"
            )

        # Name any cloud models present, and confirm they are being refused.
        remote = getattr(assistant.engine, "list_remote_models", lambda: [])()
        if remote:
            state = (
                "[err]ALLOWED — prompts can leave this machine[/err]"
                if settings.allow_remote_models
                else "[ok]blocked[/ok]"
            )
            console.print(
                f"[meta]·[/meta] {len(remote)} cloud model(s) in the registry, {state}: "
                f"[meta]{', '.join(remote)}[/meta]"
            )
            if settings.allow_remote_models:
                ok = False
    else:
        ok = False
        console.print(
            f"[err]✗[/err] {health['engine']} unreachable at {settings.ollama_host} "
            "[hint]start it with: ollama serve[/hint]"
        )

    roots = filesvc.allowed_roots()
    if roots:
        console.print(f"[ok]✓[/ok] {len(roots)} allowed folder(s):")
        for label, path in roots:
            console.print(f"    [meta]{label} → {path}[/meta]")
    else:
        ok = False
        console.print("[err]✗[/err] no readable allowed folders")

    try:
        console.print(
            f"[ok]✓[/ok] document index: {assistant.document_chunks} chunks from "
            f"{len(assistant.documents())} document(s)"
        )
    except Exception as exc:
        ok = False
        console.print(f"[err]✗[/err] document index unavailable: {exc}")

    from pathlib import Path

    from app.cli import audio

    voice_path = Path(settings.piper_voice_path)
    if voice_path.exists():
        console.print(f"[ok]✓[/ok] TTS voice present ({voice_path.name})")
    else:
        console.print(
            f"[warn]![/warn] TTS voice missing at {voice_path} "
            "[meta](spoken replies unavailable)[/meta]"
        )

    if audio.available():
        console.print(f"[ok]✓[/ok] audio devices ready (in: {audio.input_device_name()})")
    else:
        console.print(
            "[warn]![/warn] no audio device [meta](/mic and /speak unavailable)[/meta]"
        )

    console.print()
    console.print("[ok]All good.[/ok]" if ok else "[err]Some checks failed.[/err]")
    raise typer.Exit(0 if ok else 1)


def entrypoint() -> None:
    try:
        cli()
    except KeyboardInterrupt:
        err_console.print("\n[meta]cancelled[/meta]")
        sys.exit(130)


if __name__ == "__main__":
    entrypoint()
