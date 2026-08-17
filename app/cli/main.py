"""Command-line entry point.

`chat` is the interactive front door; everything else is a one-shot command so
the assistant composes with other tools (`assistant ask "..." | less`).
"""
from __future__ import annotations

import json
import sys

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
) -> None:
    """Start an interactive chat session."""
    from app.cli.repl import Session, run

    run(
        _assistant(),
        Session(use_rag=rag, allow_files=files, model=model, speak=speak),
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
                    "metrics": terminal.metrics if terminal else None,
                },
                indent=2,
            )
        )
        return

    render.end_stream()
    if not quiet and terminal is not None:
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
        if not tool.risk.needs_consent:
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
