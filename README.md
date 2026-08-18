# Local Offline AI Assistant

A privacy-first AI assistant that runs **entirely on your machine**, driven from
the command line. No external APIs at runtime — your conversations, documents,
and voice never leave the device.

Built to run within tight edge constraints (target: **8GB RAM**), demonstrating
data-privacy, latency-bound, and edge-deployment engineering.

## Architecture

```
Terminal (CLI)
      │
      ▼
Assistant core ──► LLMEngine (interface) ──► Ollama ──► quantized SLM
   ├─ Whisper (faster-whisper)   speech → text   [offline STT]
   ├─ Piper                       text → speech   [offline TTS]
   ├─ RAG: MiniLM embeddings + FAISS vector store
   └─ Disk analyzer: duplicates, integrity, stale storage
```

Two seams carry the design:

- **`LLMEngine`** (`app/engines/base.py`) — the app never touches Ollama
  directly. Swapping the inference backend (e.g. to `llama-cpp-python` for
  air-gapped devices) is a one-line config change, not a rewrite.
- **`Assistant`** (`app/core/assistant.py`) — all behaviour lives here with no
  UI attached. The CLI is a rendering layer over it, so a second front end can
  never drift from what the assistant actually does.

## "Offline" — what it means here

- **At runtime: fully offline.** Nothing is sent to any external server.
- **At setup: one-time downloads.** Models are pulled from the internet *once*
  (`ollama pull ...`, `pip install ...`). After that, unplug the network and it works.

### The guarantee is enforced, not just documented

Ollama can also route inference to **hosted models** — anything tagged `-cloud`,
such as `gpt-oss:120b-cloud`. They appear in `ollama list` looking exactly like
local models and are selected the same way, so a single `/model gpt-oss:120b-cloud`
would have shipped prompts, retrieved chunks, and injected file contents to a
remote server while the UI still claimed to be running offline.

Nothing in the app chose a cloud model, but nothing stopped one either — and a
privacy guarantee that relies on the user not typing the wrong thing is not a
guarantee. So `app/engines/policy.py` validates every point where a model name is
resolved, including the `--model` flag and the `/model` command:

```
$ python assistant.py ask "hello" --model gpt-oss:120b-cloud
'gpt-oss:120b-cloud' is a cloud-hosted model — using it would send your prompts
and file contents off this machine, which this assistant exists to prevent.
```

Cloud models are also hidden from `models` (nothing should offer a choice it
would then refuse) and reported by `doctor`. To opt in deliberately, set
`ASSISTANT_ALLOW_REMOTE_MODELS=true` — at which point `doctor` reports it as a
failed check, because at that point the headline claim of this README is false.

## The models

One LLM does all the generation; the rest are task-specific networks.

| Model | Size | Role |
|---|---|---|
| `llama3.2:latest` (Llama 3.2 3B) | 2.0 GB | Chat, RAG answers, tool-calling, summarization |
| `llama3.2:1b` | 1.3 GB | Fallback if the main model can't load |
| `all-MiniLM-L6-v2` (ONNX) | 86 MB | RAG embeddings, 384-dim |
| `faster-whisper base.en` (int8) | 139 MB | Speech → text |
| `Piper en_US-lessac-medium` | 60 MB | Text → speech |

**The fallback is a real safety net, not a config entry.** On a memory-tight
machine a model load fails with an HTTP 5xx, which is exactly what happened
during development on an 8GB box with ~250 MB free. When that occurs the engine
retries once with `fallback_model` and **says so** — silently substituting a
weaker model would make a degraded answer look like the main model's best effort:

```
The main model could not be loaded; answered with llama3.2:1b instead.
```

The retry only fires before the first token; restarting mid-stream would splice
two different answers together. It also only fires on 5xx — a refused connection
means Ollama is down, where different weights would not help. Run
`ollama pull llama3.2:1b` to arm it; `doctor` warns when it isn't pulled.

## Prerequisites

- **Python 3.10+** (tested on 3.13)
- **[Ollama](https://ollama.com/download)** installed and running
- A pulled model — default `llama3.2:latest` (Llama 3.2 3B, ~2GB)

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
copy .env.example .env         # then edit if needed

ollama serve                   # if not already running
ollama pull llama3.2:latest    # if not already pulled
```

Check everything is wired up:

```bash
python assistant.py doctor
```

## Usage

One command brings everything up and starts a conversation:

```powershell
wake up buddy
```

That starts the model server if it is down, preloads the model so your first
question isn't the one that pays for the cold start, launches the background
daemon, reports what it found, and drops you into chat with your last
conversation resumed:

```
Waking Buddy...
  v model server started
  v llama3.2:latest warm (25s)
  v background daemon started (pid 18076)
  v file access approved

Buddy is ready.  /help for commands
```

Every step is idempotent — waking an already-awake assistant is a no-op, not a
second copy of everything. `sleep buddy` stops the daemon; the model server is
left alone, since other things on the machine may be using it.

Install it once with `scripts/install_shell.ps1`, which adds `wake`, `buddy` and
an `ai` alias to your PowerShell profile between marked lines you can delete.

### The individual commands

```bash
python assistant.py chat                    # interactive session
python assistant.py ask "what is 2+2?"      # one-shot
python assistant.py ask "summarize this" --rag --json | jq .answer
```

| Command | What it does |
|---|---|
| `wake` | Start everything and drop into conversation |
| `sleep` | Stop the background daemon |
| `chat` | Interactive REPL with slash commands and history |
| `ask` | One-shot question; reads stdin, `--json` for scripting |
| `ingest <path>` | Index a document for retrieval |
| `docs` | List indexed documents (`--reset` to wipe) |
| `find <query>` | Search your allowed folders |
| `scan` | Analyse your folders for duplicates, corruption, idle storage |
| `consent` | Show or change permission to analyse your files |
| `tools` | List capabilities, permissions, and what they've done |
| `memory` | Show or change what the assistant remembers about you |
| `system` | CPU, memory, disk, battery, and top processes |
| `daemon` | Background jobs: `run`, `once`, `status`, `briefing` |
| `say <text>` | Speak text aloud (`--out file.wav` to save instead) |
| `listen` | Record from the mic and transcribe (`--ask` to answer it) |
| `models` | List locally available models |
| `doctor` | Diagnose backend, folders, index, audio, and voice |
| `eval` | Benchmark the RAG pipeline against a golden dataset |

Inside `chat`: `/rag`, `/files`, `/speak`, `/mic`, `/model`, `/temp`, `/ingest`,
`/find`, `/docs`, `/reset`, `/clear`, `/help`, `/exit`.

## Chat with your documents (RAG)

```bash
python assistant.py ingest sample_docs/nimbusedge_handbook.md
python assistant.py ask "How long is the NimbusEdge warranty?" --rag
```

Documents are chunked, embedded (all-MiniLM-L6-v2 via ONNX), and stored in a
local FAISS index under `data/vector_store/` that survives restarts. Answers
print a **sources** line with cosine similarity per chunk.

[`sample_docs/nimbusedge_handbook.md`](sample_docs/nimbusedge_handbook.md)
contains invented facts (a 26-month warranty, HIPAA/GDPR/FIPS certs) the model
cannot know, so a correct answer *proves* retrieval is working. Try the same
question with and without `--rag` to see the difference.

### Design notes

- **Embeddings run on ONNX (fastembed), not PyTorch** — the same MiniLM model at
  a fraction of the install size and memory. The right trade for an 8GB target.
- **Structure-aware chunking** (`app/rag/chunker.py`) splits on Markdown headings
  and keeps each heading attached to its body. This matters: naive fixed-width
  chunking sliced "HIPAA" in half and detached the "Compliance" heading, dropping
  that chunk from rank 1 to rank 6 and causing the model to miss the answer.

## The background daemon

```bash
python assistant.py daemon run          # foreground, Ctrl-C to stop
python assistant.py daemon status -n 10 # schedule + recent runs
python assistant.py daemon once <job>   # run one now
python assistant.py daemon briefing     # what happened while you were away
```

| Job | Every | Does |
|---|---|---|
| `index_new_files` | 30m | Sweeps every folder for documents that appeared |
| `system_health` | 15m | Watches for sustained memory / disk / battery pressure |
| `disk_scan` | weekly | Refreshes the duplicate & integrity report |

Plus a filesystem watcher on Downloads for near-instant indexing (below). Drop a
document in Downloads and it becomes answerable without you doing anything.

### Restraint is the whole design

The daemon acts while nobody is watching, so most of its design is about what it
declines to do:

- **An absent user is not consent.** Jobs run with no confirmer attached, so any
  capability needing permission refuses outright. This is why the registry's
  "no confirmer means no" default was built before anything could run unattended.
- **Revoking file consent stops it.** Both file-touching jobs skip with
  *"file analysis has not been approved"* rather than proceeding on a permission
  granted for something else.
- **Expensive work never runs on launch.** A weekly disk scan is scheduled a full
  interval out on first start, so restarting the daemon can't trigger it.
- **The first sweep records a baseline instead of ingesting everything.** You
  asked for *new* files to be picked up, not for your entire Documents folder to
  be silently absorbed. 1,970 files were noted as seen; nothing was indexed.
- **Quiet runs stay quiet.** Only findings worth acting on are flagged (`!`); a
  briefing that reports every uneventful sweep is one nobody reads.
- **Jobs run one at a time**, because on a machine with under a gigabyte free a
  parallel scheduler is a second workload competing with you.

### The doorbell and the mailbox

Two mechanisms, covering different failures:

- **The doorbell** (`watchdog`) watches **Downloads only** and indexes documents
  within seconds of them landing.
- **The mailbox** (a periodic sweep) checks every folder on a timer.

Downloads gets the doorbell because it is not cloud-synced and it is where you
actually save the thing you are about to ask about. Documents and Desktop stay on
the sweep: they are OneDrive-synced here, and sync churn emits a constant stream
of create/modify events for files nobody touched.

**The doorbell only rings while the daemon is awake.** Anything that arrives
while it is stopped is never announced — those notifications are simply gone. So
the catch-up sweep runs *first* at startup, and only then is the watcher armed.
The doorbell is an optimisation on top of the mailbox, never a replacement:

```
01:28:27 · watching 3 job(s)                              ← mailbox catches up
01:28:27 · watching C:\Users\Om\Downloads for new documents  ← doorbell armed
01:28:50 ! Indexed doorbell_test.md (1 chunks)            ← 6s after the file landed
```

If watchdog is missing or the observer cannot start, indexing keeps working at
sweep pace and says so. `--no-watch` disables the doorbell.

### A file appearing is not a file being finished

The hard part is not noticing the file, it is knowing when it is *done*. A
browser writes `report.pdf.crdownload`, grows it for a minute, then renames it;
a plain copy fires a creation event while the file is still zero bytes. Indexing
at first sight records a fragment as though it were the document.

So `.crdownload`, `.part`, `.tmp` and `~$` scratch files are ignored outright,
and a path is only read once its **size has stopped changing**. In the live test
above the write ran from `01:28:41` to `01:28:44` and indexing happened at
`01:28:50` — it waited.

### Failure doesn't compound

A job that raises is recorded and the loop continues — one broken sweep must not
take down the process running the others. Repeated failures back the job off
exponentially (capped) rather than hammering a broken dependency every interval,
and a single success clears the count.

## System awareness

```bash
python assistant.py system                          # CPU, memory, disk, battery, processes
python assistant.py ask "why is my laptop slow?"    # a real answer, from real numbers
```

```
Based on the current state of your machine:
You are experiencing slowness due to low available memory
(0.53 GB free out of 7.78 GB).
system: 13% CPU · 0.53 GB free of 7.78 GB
```

The snapshot flags pressure explicitly ("memory is nearly exhausted") because a
3B model will not infer that 0.5 GB free is a problem — and that inference *is*
the question being asked.

**"System Idle Process" is excluded from process listings.** Windows charges
unused CPU time to it, so on an idle machine it tops any CPU ranking at 70–90%.
Reporting it as the biggest consumer states the exact opposite of the truth:
that figure is how much CPU is *free*.

### Ending a process — the first irreversible capability

`end_process` is `destructive`, so it is confirmed on **every** call, naming the
actual target rather than the tool call:

```
End Spotify.exe (PID 8420, 150 MB)? Unsaved work will be lost.
Allow? [y/N]
```

There is deliberately **no standing grant** for destructive tools. A persistent
"yes, you may destroy things" is precisely the permission that should not exist —
and asking for one produced a worse experience anyway, since a standing prompt
can only say "Allow 'end_process'?", forcing a decision about a category before
the user learns which program is about to close.

What it refuses, in the order each failure would hurt:

- **Protected processes**, before ever prompting. Ending `csrss.exe` doesn't
  close a program, it bluescreens Windows. Found by running it: an early version
  prompted *"End csrss.exe? Unsaved work will be lost"*, accepted the yes, and
  only then refused — a prompt for an action that was never possible, training
  exactly the wrong habit.
- **Ambiguous names.** "close chrome" with fourteen `chrome.exe` processes must
  not become fourteen terminations. The matches are listed and you pick a PID.
- **Recycled PIDs.** Between the model naming PID 8420 and you approving it, that
  PID can belong to something else. Name and start time are re-verified at the
  moment of the kill, not when the target was chosen.
- **Itself and its model server**, which would end the conversation mid-sentence.

### Refusals are answered directly, not generated

When you decline, the reply is fixed text and no model call happens at all:

```
Cancelled — nothing was changed.
answered directly (no model call)
```

This is not an optimisation. Asked to phrase a refused kill, llama3.2:3b
repeatedly answered *"I am unable to terminate processes on your system"* —
false, and enough to convince someone the capability doesn't exist — despite
explicit instructions not to. Where the truth is known exactly, it is stated
exactly rather than delegated to a small model.

## Memory

Conversations persist, and facts you ask it to keep come back later:

```bash
python assistant.py chat --resume            # continue the last conversation
python assistant.py memory                   # what it remembers
python assistant.py memory --remember "..."
python assistant.py memory --forget 3        # or --forget all
python assistant.py memory --sessions        # past conversations
```

In a session: `/remember`, `/memories`, `/forget <id|all>`, `/history`.

### Nothing is inferred in the background

The obvious design is to run an extraction pass over every turn asking the model
"what's worth remembering here?" It was rejected twice over. It costs a second
generation per turn on a machine where a turn already takes 20 seconds, and more
importantly it means silently accumulating inferred claims about someone that
they never approved, cannot see the reasoning behind, and which may simply be
wrong.

**So a fact is stored only when you ask for one to be stored** — through a tool
that declares `write` risk and therefore asks permission the first time, and is
audited every time. Recall is automatic, because reading back something you
explicitly asked to be kept is the thing you asked for.

Every recall is shown, with its similarity score:

```
Your dog's name is Rex, and he's a beagle.
recalled 1 memory:
  · The user's dog is called Rex and is a beagle  (0.57)
```

An assistant that quietly consults a private dossier about you is worse than one
that shows its working.

### The relevance floor was measured, not guessed

Injecting vaguely-related personal trivia into a 3B model's context makes answers
worse, so recall only fires above a cosine floor. That number came from probing
the real embedder:

| Query type | Score range |
|---|---|
| Unrelated ("capital of Peru", "reverse a list", "7×6") | 0.05 – 0.14 |
| Related ("what is my dog called", "how do you like to answer me") | 0.34 – 0.57 |

A first guess of `0.45` sat *inside* the positive range and silently dropped a
genuine recall. The floor is `0.28`, in the empty band between the clusters with
roughly 2× margin over the highest false match. Unit tests use a fake embedder
and pass their own threshold, so the constant is only meaningful against the
real model — which is why it was set from measurement rather than intuition.

Memory shares the RAG embedder rather than loading a second copy, and with
nothing stored it never touches the model at all.

## Capabilities (the tool registry)

Everything the assistant can *do* — read a document, report on your disk, and
whatever comes next — registers as a tool with a declared risk level:

```bash
python assistant.py tools           # what it can do, and what you've permitted
python assistant.py tools --audit   # what it actually did
python assistant.py tools --revoke <name>
```

| Risk | Consent | Example |
|---|---|---|
| `read` | inherits folder consent | open a document, report a scan, system status |
| `write` | asked once, remembered | remember a fact |
| `destructive` | **confirmed every call, no standing grant** | end a process |

### Why a registry rather than more `if` branches

The first two capabilities were hardwired into the turn loop, each with its own
keyword gate and grounding template. They already collided: *"find my duplicate
files"* matched **both** the file gate and the disk gate. It was resolved by
checking disk first — an invisible, untested ordering that would not have
survived a third capability. Every feature also added a boolean to every sibling
branch, so the dispatch condition had grown to four terms for two features.

Now both tools are offered to the model together and it picks based on the
sentence. The audit log shows the routing decision (`via: model`), so the
collision case is observable rather than a matter of faith.

Three rules keep it honest:

- **Ordinary chat pays nothing.** Cheap keyword matchers run first; if none
  match, no tool-selection call is made at all.
- **The backstop refuses to guess.** If the model picks nothing and exactly one
  tool matched, that tool runs. If *several* matched, it declines — guessing
  wrong is worse than doing nothing.
- **The backstop never fires for a destructive tool.** A keyword match is enough
  to guess "they want to read a file". It is not enough to decide something gets
  deleted.

Nothing is authorised by default: with no confirmer wired up — a script, or the
future background daemon — every permission question answers *no*.

## Disk analysis

```bash
python assistant.py scan                                 # report to the terminal
python assistant.py scan --export report.md              # full findings to a file
python assistant.py scan --cleanup-script cleanup.ps1    # a script you review and run
python assistant.py ask "what's wasting the most space?"  # ask about the last scan
```

The first run asks permission, showing exactly which folders it would read.
Approval is recorded per folder in `data/consent.json`, and **re-prompts if the
folder list ever changes** — approving Downloads once must not silently become
approval of a folder added months later. `assistant consent --revoke` withdraws it.

The report covers four things:

- **Duplicates** — byte-identical files, ranked by recoverable space
- **Integrity** — files whose format is verifiably broken or mislabelled
- **Large and unused** — big files nobody has opened in a long time
- **Storage breakdown** — where the space actually goes

### It never deletes anything

The scanner is read-only; `tests/test_safety.py` enforces that structurally by
parsing the package for write calls. `--cleanup-script` writes a **PowerShell
script you review and run yourself**, and it sends files to the **Recycle Bin**
rather than deleting them. The assistant proposes; you dispose.

### On "corrupt files" — what is actually being claimed

Generic corruption detection is not possible. A truncated MP4 and a healthy one
are both just bytes. So this verifies only what it can prove — PDF structure,
ZIP/OOXML containers, image decoding, JSON parsing, signature-vs-extension
agreement — and reports everything else as `unverifiable` rather than implying
it is healthy. The report states how many files it could not check, so "no
problems found" is never mistaken for a clean bill of health.

Even `ok` is a specific claim: the file **decodes**, not that its content is
complete. No decoder can know how many rows a photographer originally captured
(see `tests/test_integrity.py::test_known_limit_repaired_stream_is_reported_decodable`).

### What a real profile taught this code

Every one of these was found by running the scanner against a real Windows
profile, not by reading the code:

- **21 phone photos were reported corrupt.** Samsung appends a `SEFT` metadata
  trailer *after* the JPEG end-of-image marker, so a tail-only check never saw
  it. Searching a wider window for the marker then failed the opposite way — a
  stray `FFD9` inside a motion photo's embedded video made a half-truncated file
  look complete. Verdicts now come from an actual decode.
- **Encrypted PDFs were reported corrupt.** A password-protected file is intact,
  merely closed to us → `unverifiable`.
- **Office `~$name.xlsx` lock stubs were reported corrupt.** Normal artifacts.
- **Hashing destroyed the signal it depended on.** Reading a file to fingerprint
  it updates its access time, so the "unused files" list shrank from 5 entries to
  3 between consecutive runs purely because of the scan's own reads. The earliest
  access time ever observed is now persisted and preferred. Restoring times with
  `os.utime()` was rejected: this package must never write to user files.
- **The cleanup script was BOM-less UTF-8**, which PowerShell 5.1 decodes as
  ANSI — any path with an accent or emoji would have been mangled, and a delete
  script aimed at a mangled path is not acceptable. Now `utf-8-sig`.

Three files really were mislabelled: a government portal had saved Java
serialized object streams with a `.pdf` extension.

### Cost control

Hashing every file to find duplicates reads the whole disk to discover that most
files are unique. Instead: group by size (no I/O), fingerprint 8 KB from each end,
and full-hash only the survivors. Results are cached in SQLite keyed on path +
size + mtime, so a rescan is near-instant. A 7 GB / 1,970-file profile scans in
about 25 seconds cold.

## Talk to it (voice)

Fully offline speech, both directions:

```bash
python assistant.py chat --speak      # replies are read aloud
python assistant.py listen --ask      # speak a question, get an answer
python assistant.py say "hello there"
```

In a session, `/mic` records until you press Enter, then transcribes with
**faster-whisper** and sends it. `/speak on` reads replies aloud with **Piper**.
For a long answer the assistant first condenses it to a short spoken summary —
the full text still prints, so you don't sit through a minute of narration.

### Design notes

- Both voice models **load lazily** — Piper on the first spoken reply, Whisper
  only if you use the mic — so they don't occupy RAM until needed. On an 8GB
  machine that headroom is the whole budget.
- Whisper uses **int8** quantization (smallest CPU memory), pinned to
  `models/whisper/`; the Piper voice lives in `models/piper/`.
- Capture is **16 kHz mono**, Whisper's native rate, avoiding a resample.
- Audio hardware is the least predictable part of a desktop app, so
  `app/cli/audio.py` degrades rather than raising: voice commands check
  `available()` first, and playback falls back to the Windows stdlib player if
  PortAudio is missing. A dead speaker never ends a conversation.

## Reading files from your computer (allowlisted)

The assistant can browse a **restricted set of local folders** and ingest a file
by path or by conversation:

```bash
python assistant.py find resume
python assistant.py ask "Get the resume from my downloads and summarize it."
```

The model decides a file is wanted (native tool-calling, with a deterministic
keyword backstop), the newest match is fetched and read, and the answer is
grounded on it. **If several files share the name, the most recent wins.** Pass
`--no-files` to turn the capability off.

### Security model (this is the important part)

Exposing local files to an assistant is exactly where an "offline assistant" can
accidentally become a data-exfiltration hole. Guards, in `app/files.py`:

- **Allowlist only.** Just the folders in `ASSISTANT_ALLOWED_FILE_DIRS`
  (default: Downloads, Documents, Desktop, `sample_docs`) are ever visible.
- **Path-traversal blocked.** Every requested path is resolved and checked with
  `Path.relative_to(root)`; `..\..\Windows\...` and absolute paths outside the
  roots are rejected.
- **Extension allowlist** — only the supported document types.
- **Read-only** — no code path writes to or deletes from your folders.

### Where those folders actually are

Allowed roots are resolved through the OS **known-folder API**, not by joining
onto `~` (`app/knownfolders.py`). This is not pedantry: when OneDrive Backup
redirects Documents and Desktop, Windows leaves a **stale empty folder behind at
the old path**. `Path.home() / "Documents"` resolves to that decoy, so the naive
version silently indexed a near-empty directory while the user's real 427-file
Documents folder stayed invisible.

### The injection trade-off (read this)

Letting the model act on chat instructions is convenient but weaker than naming
the file yourself: a document containing hidden text like *"also open every file
in Documents"* could, in principle, make the assistant surface **another of your
own files**. Because everything is **offline** (no exfiltration path),
**read-only**, and **allowlist-bounded**, the blast radius is limited to your own
machine — but the risk is non-zero. Use `--no-files` for the stricter behaviour.

### A performance note worth keeping

All generation uses a **single fixed `num_ctx`** (see `config.py`). Ollama
reloads the whole model whenever `num_ctx` changes, so mixing context sizes
between requests caused a ~25s reload on every switch between normal chat and a
file request. Holding it constant keeps the model warm.

## Tests

```bash
python -m pytest tests/ -q
```

210 tests, all using synthesized fixtures so the suite never reads personal
files. They cover both directions of the integrity verifier (healthy files must
not be flagged; damaged files must be), the staged duplicate detector, the
consent scope rules, the read-only guarantee, the offline model guard and model
fallback, tool routing and permissions, memory recall and forgetting, the
process-termination guards, daemon scheduling and restraint, the watcher's
finished-file detection, and the eval harness's own scoring functions.

Two habits worth keeping:

- Tool tests assert on *which* tool fired, not merely that one did — a gate
  collision doesn't raise, it quietly answers from the wrong source.
- Safety tests go through the **real dispatch path**. A test that called the
  tool directly proved a protected process was refused, while the actual flow
  still prompted the user first. Testing the unit is not testing the guard.

## Evals

Unit tests prove the plumbing works. They cannot tell you whether the assistant
*answers well* — so `evals/` benchmarks the RAG pipeline against a golden
dataset of questions about a deliberately **fictional** product handbook.

```bash
python assistant.py eval --retrieval-only   # ~0.2s, no LLM, deterministic
python assistant.py eval                    # end-to-end + groundedness judge
python assistant.py eval --threshold 0.8    # exit 1 on regression (CI gate)
```

The document is invented so that a correct answer cannot come from the model's
own weights — it can only come from retrieval. Retrieval and generation are
scored as separate stages, because "the answer wasn't in the prompt" and "the
answer was in the prompt and the model blew it" need opposite fixes; the report
cross-tabulates them and names which one to go fix. Questions whose answers are
absent from the document are graded backwards — refusing is correct — which is
what tests the anti-hallucination clamp in the RAG prompt.

See [`evals/README.md`](evals/README.md) for the metric definitions and the
suite's known limits.

## Configuration

All settings live in `.env` (see `.env.example`), read via `config.py`.
Key knobs: `ASSISTANT_ENGINE`, `ASSISTANT_DEFAULT_MODEL`, `ASSISTANT_OLLAMA_HOST`.

## Future work

- **Optional HTTP server.** The project shipped a browser UI (FastAPI + a
  single-page app) through Phase 4; it was removed in favour of a CLI-only tool.
  Because all behaviour now lives in `app/core/assistant.py` behind a UI-neutral
  event stream, restoring a server is a thin adapter — roughly 150 lines mapping
  the same events to NDJSON — not a rewrite. Useful later for a phone client or
  a system-tray app. See git history at tag `pre-cli` for the original.
- **Jarvis-style extensions** — wake-word voice loop, a general tool registry
  with a permission ledger, persistent long-term memory, a proactive background
  daemon, and system awareness/control.
