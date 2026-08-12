"""The interactive chat loop.

Holds the conversation in memory for the life of the session and drives
`Assistant.chat_stream`. Slash commands change session state without leaving the
loop, which is what makes a terminal assistant usable — toggling RAG or opening
a file shouldn't cost you your context.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory

from app.cli import audio, render
from app.cli.render import console
from app.core import Assistant
from app.engines import ChatMessage
from config import settings

HISTORY_FILE = Path(settings.upload_dir).parent / "cli_history.txt"

BANNER = """[bot]Local Offline AI Assistant[/bot]
[meta]Everything runs on this machine. Type /help for commands, /exit to quit.[/meta]"""

HELP = """[bot]Commands[/bot]
  [user]/rag[/user] [meta]on|off[/meta]      ground answers in your indexed documents
  [user]/files[/user] [meta]on|off[/meta]    allow the model to open local files by name
  [user]/model[/user] [meta]<name>[/meta]    switch model for this session (blank = show current)
  [user]/temp[/user] [meta]<0.0-2.0>[/meta]  sampling temperature
  [user]/speak[/user] [meta]on|off[/meta]    read replies aloud (Piper, offline)
  [user]/mic[/user]              record a question from the microphone
  [user]/ingest[/user] [meta]<path>[/meta]   index a local file for RAG
  [user]/find[/user] [meta]<query>[/meta]    search your allowed folders
  [user]/docs[/user]             list indexed documents
  [user]/reset[/user]            wipe the document index
  [user]/clear[/user]            forget this conversation (keeps the index)
  [user]/help[/user]  [user]/exit[/user]"""


@dataclass
class Session:
    """Everything that persists across turns in one CLI run."""

    history: list[ChatMessage] = field(default_factory=list)
    use_rag: bool = False
    allow_files: bool = True
    speak: bool = False
    model: str | None = None
    temperature: float | None = None

    def status(self) -> str:
        bits = [
            f"rag {'on' if self.use_rag else 'off'}",
            f"files {'on' if self.allow_files else 'off'}",
        ]
        if self.speak:
            bits.append("speak on")
        if self.model:
            bits.append(self.model)
        return "  ·  ".join(bits)


def _parse_toggle(arg: str, current: bool) -> bool:
    """`/rag` with no argument flips; `/rag on|off` sets explicitly."""
    a = arg.strip().lower()
    if a in {"on", "true", "yes", "1"}:
        return True
    if a in {"off", "false", "no", "0"}:
        return False
    return not current


def handle_command(assistant: Assistant, session: Session, line: str) -> bool:
    """Run a slash command. Returns False when the session should end."""
    cmd, _, arg = line[1:].partition(" ")
    cmd = cmd.lower().strip()
    arg = arg.strip()

    if cmd in {"exit", "quit", "q"}:
        return False

    if cmd == "help":
        console.print(HELP)

    elif cmd == "rag":
        session.use_rag = _parse_toggle(arg, session.use_rag)
        if session.use_rag and assistant.document_chunks == 0:
            console.print(
                "[warn]RAG on, but nothing is indexed yet — use /ingest <path>.[/warn]"
            )
        else:
            console.print(f"[ok]rag {'on' if session.use_rag else 'off'}[/ok]")

    elif cmd == "files":
        session.allow_files = _parse_toggle(arg, session.allow_files)
        console.print(f"[ok]file access {'on' if session.allow_files else 'off'}[/ok]")

    elif cmd == "model":
        if not arg:
            console.print(f"[meta]model: {session.model or settings.default_model}[/meta]")
        else:
            session.model = arg
            console.print(f"[ok]model set to {arg}[/ok]")

    elif cmd == "temp":
        try:
            session.temperature = float(arg)
            console.print(f"[ok]temperature {session.temperature}[/ok]")
        except ValueError:
            console.print("[err]usage: /temp 0.7[/err]")

    elif cmd == "speak":
        if not audio.available():
            console.print(
                "[err]No audio output available.[/err] "
                "[hint]pip install sounddevice soundfile[/hint]"
            )
        else:
            session.speak = _parse_toggle(arg, session.speak)
            console.print(f"[ok]speak {'on' if session.speak else 'off'}[/ok]")

    elif cmd == "mic":
        text = record_question(assistant)
        if text:
            console.print(f"[user]you (voice)[/user] {text}")
            run_turn(assistant, session, text)

    elif cmd == "ingest":
        if not arg:
            console.print("[err]usage: /ingest <path>[/err]")
        else:
            result = assistant.ingest_path(arg)
            if result.get("ok"):
                console.print(
                    f"[ok]indexed {result['source']} — {result['chunks']} chunks "
                    f"({result['total_chunks']} total)[/ok]"
                )
                session.use_rag = True
            else:
                console.print(f"[err]{result.get('error')}[/err]")

    elif cmd == "find":
        results = assistant.search_files(arg)
        if results:
            console.print(render.files_table(results))
        else:
            console.print("[warn]no matching files[/warn]")

    elif cmd == "docs":
        docs = assistant.documents()
        if docs:
            console.print(render.documents_table(docs, assistant.document_chunks))
        else:
            console.print("[warn]nothing indexed yet[/warn]")

    elif cmd == "reset":
        assistant.reset_documents()
        console.print("[ok]document index cleared[/ok]")

    elif cmd == "clear":
        session.history.clear()
        console.print("[ok]conversation cleared[/ok]")

    else:
        console.print(f"[err]unknown command /{cmd}[/err] [hint]try /help[/hint]")

    return True


def run_turn(assistant: Assistant, session: Session, text: str) -> None:
    """Send one user message and stream the reply."""
    session.history.append(ChatMessage(role="user", content=text))

    console.print("[bot]assistant[/bot] ", end="")
    collected: list[str] = []
    terminal = None
    try:
        for event in assistant.chat_stream(
            session.history,
            use_rag=session.use_rag,
            allow_file_access=session.allow_files,
            model=session.model,
            temperature=session.temperature,
        ):
            if event.type == "token":
                collected.append(event.text)
                render.stream_token(event.text)
            else:
                terminal = event
    except KeyboardInterrupt:
        # Abandon this generation but keep the session alive.
        render.end_stream()
        console.print("[warn]interrupted[/warn]")

    reply = "".join(collected)
    if reply:
        render.end_stream()

    if terminal is not None and terminal.type == "error":
        console.print(f"[err]{terminal.error}[/err]")
        # A failed turn must not leave a dangling user message in history.
        session.history.pop()
        return

    if not reply:
        session.history.pop()
        return

    session.history.append(ChatMessage(role="assistant", content=reply))

    if terminal is not None:
        render.print_opened_file(terminal.opened_file)
        render.print_sources(terminal.sources)
        render.print_metrics(terminal.metrics)

    if session.speak:
        speak_reply(assistant, reply)


def speak_reply(assistant: Assistant, reply: str) -> None:
    """Synthesize and play a reply. Never fatal — a dead speaker shouldn't end
    the conversation, so failures degrade to a warning and the text stands."""
    try:
        wav, summarized = assistant.speak(reply)
    except Exception as exc:
        console.print(f"[warn]could not speak that ({exc})[/warn]")
        return
    if summarized:
        console.print("[meta](speaking a condensed summary)[/meta]")
    audio.play_wav(wav)


def record_question(assistant: Assistant) -> str | None:
    """Record from the mic and transcribe it. Returns None if nothing usable."""
    if not audio.available():
        console.print(
            "[err]No microphone available.[/err] "
            "[hint]pip install sounddevice soundfile[/hint]"
        )
        return None

    device = audio.input_device_name()
    console.print(
        f"[bot]recording[/bot] [meta]{device or 'default input'} — "
        "press Enter to stop[/meta]"
    )
    try:
        wav_path = audio.record_to_tempfile(input)
    except audio.RecordingError as exc:
        console.print(f"[err]{exc}[/err]")
        return None

    try:
        console.print("[meta]transcribing…[/meta]")
        text = assistant.transcribe(str(wav_path))
    except Exception as exc:
        console.print(f"[err]transcription failed: {exc}[/err]")
        return None
    finally:
        wav_path.unlink(missing_ok=True)

    text = (text or "").strip()
    if not text:
        console.print("[warn]didn't catch anything[/warn]")
        return None
    return text


# PowerShell prepends a UTF-8 BOM when piping text into a program, so the first
# line of a scripted session arrives as "﻿/help" and silently misses the
# slash-command check — the command gets sent to the model as chat instead.
_INVISIBLE = "﻿​‎‏"


def clean_input(line: str) -> str:
    return line.lstrip(_INVISIBLE).strip()


def _make_reader():
    """Return a prompt function suited to how the CLI was invoked.

    prompt_toolkit needs a real terminal — it raises when stdin is a pipe. So a
    piped session (`echo /docs | assistant chat`, or a test script) falls back to
    plain input(), trading history and editing for the ability to run headless.
    """
    if not sys.stdin.isatty():
        def read_plain() -> str:
            return input()

        return read_plain

    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    prompt = PromptSession(history=FileHistory(str(HISTORY_FILE)))

    def read_rich() -> str:
        return prompt.prompt("\nyou › ")

    return read_rich


def run(assistant: Assistant, session: Session | None = None) -> None:
    session = session or Session()
    console.print(BANNER)

    health = assistant.health()
    if not health["healthy"]:
        console.print(
            f"[err]Cannot reach the {health['engine']} backend.[/err] "
            "[hint]Is `ollama serve` running? Run `doctor` for details.[/hint]"
        )
    else:
        console.print(f"[meta]{health['model']}  ·  {session.status()}[/meta]")

    read_line = _make_reader()

    while True:
        try:
            line = clean_input(read_line())
        except KeyboardInterrupt:
            continue  # Ctrl-C clears the input line
        except EOFError:
            break  # Ctrl-D exits, and ends a piped script cleanly

        if not line:
            continue
        if line.startswith("/"):
            if not handle_command(assistant, session, line):
                break
            continue

        run_turn(assistant, session, line)

    console.print("\n[meta]bye[/meta]")
