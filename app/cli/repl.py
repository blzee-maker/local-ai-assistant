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

# What the assistant is called on screen. Lower-cased to sit beside the
# "you" prompt, and taken from the configured name rather than hard-coded:
# being greeted by "Buddy" and then answered by "assistant" reads as two
# different programs.
BOT_LABEL = settings.assistant_name.lower()


def banner() -> str:
    """Greeting for a directly-launched session.

    Built from the configured name so the assistant is called the same thing
    here, in the system prompt, and in whatever the user typed to start it.
    """
    return (
        f"[bot]{settings.assistant_name}[/bot]\n"
        "[meta]Everything runs on this machine. "
        "Type /help for commands, /exit to quit.[/meta]"
    )


HELP = """[bot]Commands[/bot]
  [user]/rag[/user] [meta]on|off[/meta]      ground answers in your indexed documents
  [user]/files[/user] [meta]on|off[/meta]    allow the model to open local files by name
  [user]/model[/user] [meta]<name>[/meta]    switch model for this session (blank = show current)
  [user]/temp[/user] [meta]<0.0-2.0>[/meta]  sampling temperature
  [user]/speak[/user] [meta]on|off[/meta]    read replies aloud (Piper, offline)
  [user]/mic[/user]              record a question from the microphone
  [user]/ingest[/user] [meta]<path>[/meta]   index a local file for RAG
  [user]/find[/user] [meta]<query>[/meta]    search your allowed folders
  [user]/remember[/user] [meta]<fact>[/meta] store something for future sessions
  [user]/memories[/user]         show what the assistant remembers
  [user]/forget[/user] [meta]<id|all>[/meta] delete a remembered fact
  [user]/history[/user]          replay this saved conversation
  [user]/resume[/user]           bring the previous conversation into context
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
    # Whether an earlier conversation has been pulled in. Guards /resume
    # against appending the same transcript twice.
    resumed: bool = False

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
            # Validate now rather than letting the next turn fail — this is the
            # exact path a cloud model would otherwise enter through.
            from app.engines.policy import RemoteModelBlocked, check_model

            try:
                check_model(arg, allow_remote=settings.allow_remote_models)
            except RemoteModelBlocked as exc:
                console.print(f"[err]{exc}[/err]")
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

    elif cmd == "remember":
        if not arg:
            console.print("[err]usage: /remember <fact>[/err]")
        else:
            memory, outcome = assistant.memory.remember(arg)
            if memory is None:
                console.print(f"[err]{outcome}[/err]")
            else:
                console.print(f"[ok]{outcome}[/ok] [meta]#{memory.id} {memory.text}[/meta]")

    elif cmd == "memories":
        memories = assistant.memory.memories()
        if memories:
            console.print(render.memories_table(memories))
        else:
            console.print("[warn]nothing remembered yet[/warn] [hint]/remember <fact>[/hint]")

    elif cmd == "forget":
        if arg.lower() == "all":
            count = assistant.memory.forget_all()
            console.print(f"[ok]forgot {count} fact(s)[/ok]")
        elif arg.isdigit():
            if assistant.memory.forget(int(arg)):
                console.print(f"[ok]forgot #{arg}[/ok]")
            else:
                console.print(f"[warn]no memory #{arg}[/warn]")
        else:
            console.print("[err]usage: /forget <id|all>[/err]")

    elif cmd == "history":
        stored = assistant.memory.history()
        if not stored:
            console.print("[warn]nothing saved for this conversation yet[/warn]")
        else:
            for message in stored:
                label = "you" if message.role == "user" else BOT_LABEL
                style = "user" if message.role == "user" else "bot"
                console.print(f"[{style}]{label}[/{style}] {message.content[:400]}")

    elif cmd == "resume":
        if session.resumed:
            console.print("[warn]already resumed[/warn] "
                          "[hint]/clear first to start fresh[/hint]")
        else:
            restored = load_conversation(
                session, assistant.memory.previous_messages(limit=20)
            )
            if restored:
                console.print(f"[ok]brought back {restored} message(s)[/ok]")
            else:
                console.print("[warn]no earlier conversation to resume[/warn]")

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
        session.resumed = False
        console.print("[ok]conversation cleared[/ok]")

    else:
        console.print(f"[err]unknown command /{cmd}[/err] [hint]try /help[/hint]")

    return True


def load_conversation(session: Session, messages) -> int:
    """Put stored messages into this session's context. Returns how many."""
    for message in messages:
        session.history.append(
            ChatMessage(role=message.role, content=message.content)
        )
    session.resumed = session.resumed or bool(messages)
    return len(messages)


def confirm_action(prompt: str) -> bool:
    """Ask before a tool acts. Anything but an explicit yes is a no."""
    console.print(f"\n[warn]{prompt}[/warn]")
    try:
        answer = input("Allow? [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        console.print("[meta]no answer — treating as no[/meta]")
        return False
    return answer in {"y", "yes"}


# Wide enough to cover the longest phase name plus a count. The padding is the
# point: a carriage return moves the cursor back but erases nothing, so a short
# message after a long one leaves the tail of the long one on screen.
PROGRESS_WIDTH = 64


def show_progress(message: str) -> None:
    """One rewritten line for work measured in minutes.

    Overwrites in place rather than scrolling: a scan emits hundreds of updates
    and a screen of them buries the conversation that follows.
    """
    if message == "done":
        # Wipe the line so the answer does not start halfway along it.
        console.print(" " * PROGRESS_WIDTH, end="\r", highlight=False)
        return
    console.print(
        f"[meta]  {message:<{PROGRESS_WIDTH - 2}}[/meta]", end="\r", highlight=False
    )


def run_turn(assistant: Assistant, session: Session, text: str) -> None:
    """Send one user message and stream the reply."""
    session.history.append(ChatMessage(role="user", content=text))

    console.print(f"[bot]{BOT_LABEL}[/bot] ", end="")
    collected: list[str] = []
    terminal = None
    try:
        for event in assistant.chat_stream(
            session.history,
            use_rag=session.use_rag,
            allow_file_access=session.allow_files,
            model=session.model,
            temperature=session.temperature,
            confirm=confirm_action,
            progress=show_progress,
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
        # Persist as we go rather than at exit: a crash or a closed terminal
        # should not cost the conversation, which is the whole point.
        assistant.memory.record("user", text)
        assistant.memory.record("assistant", reply)

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
        render.print_recalled(terminal.recalled)
        render.print_tool_used(terminal.tool_used)
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


def read_plain() -> str:
    """Input with no line editing. Works anywhere, including a bare pipe."""
    return input()


def _make_reader():
    """Return a prompt function suited to how the CLI was invoked.

    prompt_toolkit wants a real terminal, so a piped session (`echo /docs |
    assistant chat`) falls back to plain input(), trading history and editing
    for the ability to run headless.

    The isatty() check alone is not enough. Under some hosts — `powershell
    -File script.ps1`, a shortcut, a scheduled task — stdin can look interactive
    while Windows has given the process no console screen buffer, and
    constructing a PromptSession then raises NoConsoleScreenBufferError and
    takes the whole session down. So construction is guarded too: no line
    editing is a small loss, a crash on startup is not (rule 10).
    """
    if not sys.stdin.isatty():
        return read_plain

    try:
        HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        prompt = PromptSession(history=FileHistory(str(HISTORY_FILE)))
    except Exception:
        return read_plain

    def read_rich() -> str:
        try:
            return prompt.prompt("\nyou › ")
        except (KeyboardInterrupt, EOFError):
            raise
        except Exception:
            # The terminal went away mid-session; keep the conversation alive.
            return read_plain()

    return read_rich


def run(
    assistant: Assistant,
    session: Session | None = None,
    resume: bool = False,
    greet: bool = True,
) -> None:
    """`greet=False` when the caller already introduced itself — `wake` prints
    its own startup report, and following it with a second banner reads as the
    program starting twice."""
    session = session or Session()
    if greet:
        console.print(banner())

    # Off by default. Silently re-feeding the last transcript made the
    # assistant answer from it instead of from the question: asked how much
    # memory was in use, it replayed the whole machine report from an
    # earlier session. Nothing is lost — the conversation is still on
    # disk, and /resume or `wake --resume` brings it back on request.
    assistant.memory.start_session(resume=resume)
    if resume:
        restored = load_conversation(session, assistant.memory.history(limit=20))
        console.print(
            f"[meta]resumed — {restored} earlier message(s)[/meta]"
            if restored
            else "[meta]no earlier conversation to resume[/meta]"
        )

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
