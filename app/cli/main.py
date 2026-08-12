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
) -> None:
    """Start an interactive chat session."""
    from app.cli.repl import Session, run

    run(
        _assistant(),
        Session(use_rag=rag, allow_files=files, model=model),
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
                    "opened_file": terminal.opened_file if terminal else None,
                    "metrics": terminal.metrics if terminal else None,
                },
                indent=2,
            )
        )
        return

    render.end_stream()
    if not quiet and terminal is not None:
        render.print_opened_file(terminal.opened_file)
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

    voice_path = settings.piper_voice_path
    from pathlib import Path

    if Path(voice_path).exists():
        console.print(f"[ok]✓[/ok] TTS voice present ({Path(voice_path).name})")
    else:
        console.print(f"[warn]![/warn] TTS voice missing at {voice_path} [meta](voice replies disabled)[/meta]")

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
