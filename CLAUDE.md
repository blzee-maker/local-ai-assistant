# Local Offline AI Assistant — working agreement

A privacy-first assistant that runs entirely on the user's machine, driven from
the terminal. Target hardware: **8GB RAM**. Everything below was earned by
something going wrong; none of it is decoration.

---

## The golden rules

### I. Promises to the user

**1. Nothing leaves the machine — and the guarantee is enforced, not documented.**
`gpt-oss:120b-cloud` once sat one `/model` command away from shipping the user's
file contents to a server while the UI still said "running fully offline." A
guarantee that depends on the user not typing the wrong thing is not a guarantee.
Enforce it where the decision resolves (`app/engines/policy.py`), including
per-request overrides — not just the configured default.

**2. The assistant proposes; the user disposes.**
Nothing deletes, moves, or modifies a user's files. Destructive intent produces a
*reviewable artifact* the user runs themselves, targeting the Recycle Bin rather
than permanent deletion. `tests/test_safety.py` enforces this structurally.

**3. Consent is explicit, recorded, and scoped — and never widens silently.**
Approving Downloads once must not become approval of a folder added months later.
Consent records a fingerprint of its scope and re-prompts when that scope changes
(`app/consent.py`). Declining is a first-class, remembered outcome.

**4. Say it plainly when the answer is degraded.**
A fallback model answering, a spoken summary replacing full text, a stale scan —
all announced. Silently serving a worse answer is a small lie that compounds.

### II. Honesty about what we know

**5. Claim only what you can prove; name the gap otherwise.**
The integrity checker reports `unverifiable` so that "no problems found" is never
mistaken for "everything is healthy." Even `ok` is a bounded claim — the file
*decodes*, not that its content is complete. When a limit exists, write a test
named after it.

**6. A false positive is worse than a false negative when the user might act on it.**
Telling someone 21 working photos are corrupt invites them to delete good files.
Where consequences are asymmetric, so is the caution.

**7. Record why a path was rejected, at the point of the decision.**
`os.utime()` would have fixed the access-time bug and broken rule 2. Byte-marker
scanning failed in both directions. Those dead ends live in comments beside the
code that replaced them, so nobody re-walks them.

### III. How we build and verify

**8. Behaviour lives in the core; front ends only render.**
`app/core/assistant.py` owns what the assistant *does* and yields events. The CLI
renders them. This exists because all behaviour was once trapped in FastAPI
handlers, so deleting the web UI would have deleted the brain. One seam per
external dependency: `LLMEngine`, `STTEngine`, `TTSEngine`.

**9. Respect the 8GB budget.**
Lazy-load models (embedder, Whisper, Piper load on first use). Hold `num_ctx`
constant — changing it forces Ollama to reload the model (~25s on CPU). Don't
make a command pay for a subsystem it never touches.

**10. Degrade, never crash.**
No audio device → warn and continue. Model won't load → fall back and say so. A
dead speaker must never end a conversation.

**11. Verify against a real machine, not against the code.**
Every significant bug here was found by running it: the OneDrive decoy folder,
Samsung SEFT trailers, hashing clobbering its own staleness signal, PowerShell's
BOM on piped stdin, PowerShell's ANSI default for `.ps1`. Reading the code found
none of them. Run it against real data before claiming it works.

**12. Test both directions, through the real path.**
Healthy inputs must pass *and* damaged inputs must fail. A verifier tested only
on bad input is half-tested — that is how 21 false positives survive. And test
safety guards through the **actual dispatch path**: a test that called a tool
directly proved a protected process was refused, while the real flow still
prompted the user to approve killing it first. Testing the unit is not testing
the guard.

**13. When the answer is already known, state it — don't ask a model to phrase it.**
Told to explain a refused action, llama3.2:3b repeatedly claimed "I am unable to
terminate processes on your system" — false, and enough to convince someone the
capability does not exist — despite explicit instructions not to. Refusals,
cancellations and other fully-determined outcomes are emitted as fixed text with
no generation at all (`ToolResult.final_text`).

---

## Architecture

```
app/cli/        Terminal front end (Typer + Rich). Rendering only.
app/core/       Assistant orchestrator + intent modules. UI-neutral.
app/engines/    LLMEngine seam; Ollama impl; offline policy guard.
app/rag/        Chunking, ONNX embeddings, FAISS store.
app/voice/      Whisper STT + Piper TTS behind swappable seams; wake-phrase
                matching (wake.py) and the awake/asleep gate (gate.py).
app/analyzer/   Read-only disk analysis. Never writes to user files.
app/tools/      Tool registry: one dispatch path for every capability.
app/memory/     Conversation persistence + explicitly-remembered facts.
app/daemon/     Scheduled background jobs. Runs with no confirmer attached.
app/windows.py  Win32 window + app-launch primitives. UI-neutral.
app/consent.py  Recorded, scoped permission for reading user folders.
evals/          Golden-dataset RAG benchmark.
```

Front ends depend on `app/core`, never the reverse.

## Commands

```bash
wake up buddy                      # start everything, then talk
python assistant.py doctor          # diagnose the whole stack first
python assistant.py chat            # REPL; /rag /files /speak /mic /listen
python assistant.py converse        # hands-free: say the wake phrase, then talk
python assistant.py converse        # hands-free: say the wake phrase, then talk
python assistant.py ask "..."       # one-shot, pipeable, --json
python assistant.py scan            # disk analysis (also runnable from chat)
python -m pytest tests/ -q          # 428 tests
```

## Environment gotchas (Windows)

These cost real debugging time. Do not rediscover them.

- **PowerShell prepends a UTF-8 BOM** when piping text into a program, so the
  first piped line arrives as `﻿...`. Strip it (`app/cli/repl.clean_input`).
- **PowerShell 5.1 reads BOM-less `.ps1` as ANSI**, mangling non-ASCII paths.
  Write generated scripts as `utf-8-sig`.
- **`&&` and `||` do not exist** in PowerShell 5.1; use `;` and `if ($?)`.
- **Here-strings break on apostrophes** in this harness — write long git commit
  messages to a file and use `git commit -F`.
- **`Path.home() / "Documents"` is wrong.** OneDrive redirection leaves a stale
  decoy folder at the old path. Use `app/knownfolders.py`.
- **OneDrive placeholder files** download when read. Check the cloud attributes
  before opening anything (`app/analyzer/walker.py`).
- **`prompt_toolkit` needs a real console, not just a TTY.** Under
  `powershell -File`, a shortcut or a scheduled task, stdin can look
  interactive while Windows gives the process no console screen buffer, and
  constructing a `PromptSession` raises `NoConsoleScreenBufferError`. Guard
  the construction, not just `isatty()` (`app/cli/repl._make_reader`).
- **sqlite3 connections belong to one thread.** Opening a `Journal` on the main
  thread and using it from a worker raises "SQLite objects created in a thread
  can only be used in that same thread". Open the connection *in* the thread
  that will use it (see `app/daemon/watcher.py`).
- **Relative paths follow the shell, not the project.** Data locations are
  anchored to `PROJECT_ROOT` in `config.py`; a relative default silently gives
  every working directory its own empty `data/` folder.
- **cmd.exe cannot see PowerShell profile functions.** Shipping a command means
  both a profile function *and* a `.cmd` shim on PATH (`bin/`).
- **Windows process-creation flags are mutually exclusive.** CREATE_NO_WINDOW,
  DETACHED_PROCESS and CREATE_NEW_CONSOLE cannot be combined: OR-ing the first
  two opens a visible console, CREATE_NO_WINDOW alone leaves the child tied to
  its launcher's console. Background children use DETACHED_PROCESS and write to
  a log file (`app/cli/startup.py`).
- **A detached process has no console, so the next process it starts gets a
  brand-new one.** Hiding the child is not enough. The venv's `python.exe` is a
  redirector that re-launches the base interpreter, and that grandchild opened
  a Windows Terminal window titled `...\.venv\Scripts\python.exe` — the flag
  meant to hide the daemon was what created the window. Launch detached
  children with `pythonw.exe`: a GUI-subsystem binary never allocates a
  console, at any depth of the chain.
- **`pythonw.exe` has no stdout of its own.** Anything that prints dies instantly
  under it *unless* it is handed a real file handle — which a background process
  needs regardless, since DEVNULL makes its crashes undiagnosable.
- **Never identify a process by substring over its joined command line.** A
  match on "assistant.py" plus "daemon" hit the shell searching for daemons and
  killed it. Match the argument list.
- **An undeclared ctypes function returns a C int.** So `GetForegroundWindow()`
  handed back a 64-bit HWND truncated to 32 bits, every raise-to-front
  compared unequal, and the assistant reported a focus change as refused
  when it had worked. Declare `restype`/`argtypes` for every Win32 call.
- **The Start Menu is not the list of installed apps.** Walking `*.lnk`
  finds Visual Studio Code and misses Spotify, because Store apps have no
  shortcut file. `Get-StartApps` returns both (272 vs 195 here), and its
  AppIDs launch through `shell:AppsFolder`.
- **Only the foreground process may hand over the foreground.**
  `SetForegroundWindow` works from the terminal the user is typing in and
  is refused from a background process. The ALT-keypress and
  minimise/restore workarounds were measured and changed nothing, so the
  result is checked and reported rather than assumed.
- **A piped stdout gets the locale codepage, not UTF-8.** So `doctor | tail`
  died on the tick character while the same command in a console worked.
  `app/cli/render` reconfigures both streams to UTF-8 with `errors="replace"`.
- **PortAudio caches the device list when it initialises.** A microphone
  plugged in after the assistant started is invisible until it is torn down
  and brought back (`sd._terminate()`/`sd._initialize()`, about 20 ms).
- **MME truncates device names at 31 characters**, so one microphone appears
  as both `Microphone Array (Intel Smart` and its full name under other host
  APIs. Match device names as substrings in either direction.
- **Windows tells PortAudio only a device's name** — no form factor, no
  bus, no built-in flag. So "is this an external microphone?" cannot be
  answered, only guessed, and it fails both ways: a virtual audio cable
  reads as external and one pair of earbuds enumerates as three devices.
  Have the user nominate a device instead (`app/voice/devices.py`).
- **Memory is genuinely tight** (~7.8GB total, often <1GB free). Ollama returns
  HTTP 5xx when a model cannot load. That is what the fallback model is for.

## Conventions

- Comments explain *why*, especially rejected alternatives (rule 7). Match the
  existing density — this codebase comments decisions, not syntax.
- Type hints throughout; `from __future__ import annotations` at the top.
- New capability that acts on the user's machine → it goes through the tool
  registry with a declared risk level, recorded consent, and an audit entry.
  Never a bespoke side path (see `app/tools/`).
- Keyword pre-filters are for *cost*, not correctness: they exist so ordinary
  chat doesn't pay for tool selection. They must never be the only thing
  deciding whether a destructive action runs.
- For a *read* tool, be generous with those keywords. The failure modes are not
  symmetric: a false match costs one cheap local call, while a missed match
  means the model answers from imagination — asked "give me my system
  information" it reported 16GB of RAM and three drives on a machine with 7.8GB
  and one. And when several read tools match but the model picks none, run the
  best-scoring one rather than nothing.
- The prompt must forbid inventing machine or file facts outright. No keyword
  list will ever be complete, so that instruction is the real backstop.
- If a capability exists only as a command, chat cannot use it and the
  assistant ends up naming the command instead of doing the work. The disk
  scan was CLI-only for that reason; it is now a tool that asks, reports
  progress, and answers from what it found.
- A tool's `precheck` runs before *any* prompt, consent included. Asking
  permission for something about to be refused is the same lie as
  confirming it: "Allow 'close_app'?" was granted moments before the tool
  refused to close a protected process.
- Closing an app is not killing it. `psutil.terminate()` on Windows is
  `TerminateProcess` and takes unsaved work with it; `WM_CLOSE` is what the
  X button does and lets the app prompt. Hence `close_app` is WRITE and
  `end_process` stays DESTRUCTIVE.
- Never close a window belonging to this process or its ancestors —
  "close terminal" would otherwise end the conversation that asked.
- Declining one expensive action is "not now", never "never". Recording it
  as a withdrawal of consent widens a no the way rule 3 forbids widening a
  yes.
- A carriage return moves the cursor back but erases nothing — pad a
  rewritten progress line, or the tail of the previous one stays on screen.
- Filter in the tool, not in the prompt. Asking a 3B model to ignore most
  of what it was handed does not work: given the whole machine and told to
  report only the memory it answered with the processor, the OS and every
  drive. A section never shown cannot be misquoted.
- Sort before you truncate, by whatever the question asked about. A
  CPU-sorted list cut to five answered "the top 5 things consuming my
  memory" without the actual largest, a 2.2 GB process, appearing at all.
- A tool that competes with another for the same questions needs its own
  `match_score`. The registry scores an unscored tool as 1, so it loses to
  anything returning a trigger count — `top_processes` lost nearly every
  resource question to a reading holding no per-process data.
- Depth follows the request; scope does not widen with it. "Detailed" means
  more about the subject asked about, never more subjects.
- Answering more than was asked is its own wrong answer. Told to "report the
  lines above and nothing else", the assistant met "how much of my memory is
  in use?" with the processor, the OS, the memory and every drive. Ground a
  tool with the whole reading, then tell it to answer the question asked.
- Context is never re-fed silently. `wake` used to resume the last
  conversation by default, so a stale answer became the source for a new
  question. Resuming is opt-in (`/resume`, `wake --resume`) and announced.
- The assistant is called the same thing everywhere. Greeting someone as
  Buddy and then labelling the reply `assistant` reads as two programs.
- A tool must stand aside from questions it cannot answer. The disk scan
  knows which files exist, never a volume's capacity — asked for free space
  it replied "7.1 GB of total storage" on a 415 GB disk, then apologised.
- Naming a microphone is a convenience, not the guarantee. Listening is off
  until switched on and visible while it runs; "my buds are connected" is
  not consent, the way approved folder access is not consent to scan.
- Voice in does not imply voice out. Speaking a reply holds Piper in memory
  beside the transcriber and the model, which on this machine is the
  difference between a fallback that rescues an answer and one with nothing
  left to load into.
- Never infer and store facts about the user in the background. Memory is
  written when they ask for it, shown when it is used, and genuinely deleted
  when they forget it.
- An absent user is not consent. Anything running unattended gets no confirmer,
  so permission-requiring capabilities refuse; scheduled work must also check
  that the consent it depends on is still granted, not assume it.
- Background work is a guest on this machine: one job at a time, expensive work
  never on launch, and a first sweep records a baseline rather than ingesting
  everything it finds.
- Model choices are measured on this project's own prompts, not on
  reputation. The configured fallback was `llama3.2:1b` because it shares a
  family with the main model; measured, it held **more** memory (2.36 GB)
  than `qwen2.5:1.5b` (1.98 GB) at equal routing accuracy. Record the
  numbers next to the constant (`config.fallback_model`).
- A safety net must cover every path that can hit the failure. `chat_stream`
  fell back when the model would not load; `chat` did not, so on a
  memory-tight machine every *tool selection* silently dropped to keyword
  scoring at exactly the moment the fallback existed for.
- A summary must not contradict the lines above it. `doctor` printed
  "All good." on a run that had just warned the fallback model was missing.
- Similarity thresholds are *measured against the real embedder*, not guessed.
  A plausible-looking constant tuned against a fake silently dropped real
  recalls; record the probe numbers next to the constant.
- A tool that grounds a turn **replaces** retrieval, so a read tool with a
  generous keyword net can silently switch the user's documents off. Four of
  the eval's four generation failures were `system_status` claiming document
  questions on "hardware", "power" and "system" — the last matching *inside*
  "Halden Systems". When documents are loaded they go on the selection menu
  (`search_documents`), so the model chooses between reading the machine and
  reading what the user gave it, instead of picking the least-bad machine tool.
- **Read the answer, not just the score.** "I cannot provide information about
  the peak power draw *of the machine*" was scored as a refusal for three
  sessions. It was a correct answer to a question about the wrong subject, and
  the four words that said so were in every transcript. A metric that buckets an
  answer cannot tell you it answered something else.
- When a rewrite and a model swap both move a number by *exactly zero*, stop
  tuning and go find what neither of them touches. Prompt A/B produced three
  identical scores; so did the model comparison. Both were measuring a path the
  failing cases never took.
- Separating retrieval from generation is what made the bug findable: retrieval
  scored 100% on cases whose chunks were never sent to the model. Two stages
  disagreeing that hard is a signal about the *wiring between* them, not about
  either one.
