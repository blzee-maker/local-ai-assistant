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
app/voice/      Whisper STT + Piper TTS behind swappable seams.
app/analyzer/   Read-only disk analysis. Never writes to user files.
app/tools/      Tool registry: one dispatch path for every capability.
app/memory/     Conversation persistence + explicitly-remembered facts.
app/daemon/     Scheduled background jobs. Runs with no confirmer attached.
app/consent.py  Recorded, scoped permission for reading user folders.
evals/          Golden-dataset RAG benchmark.
```

Front ends depend on `app/core`, never the reverse.

## Commands

```bash
python assistant.py doctor          # diagnose the whole stack first
python assistant.py chat            # REPL; /rag /files /speak /mic /model
python assistant.py ask "..."       # one-shot, pipeable, --json
python assistant.py scan            # disk analysis (consent-gated)
python -m pytest tests/ -q          # 91 tests
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
- **`prompt_toolkit` needs a real TTY**; fall back to `input()` when piped.
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
- Never infer and store facts about the user in the background. Memory is
  written when they ask for it, shown when it is used, and genuinely deleted
  when they forget it.
- An absent user is not consent. Anything running unattended gets no confirmer,
  so permission-requiring capabilities refuse; scheduled work must also check
  that the consent it depends on is still granted, not assume it.
- Background work is a guest on this machine: one job at a time, expensive work
  never on launch, and a first sweep records a baseline rather than ingesting
  everything it finds.
- Similarity thresholds are *measured against the real embedder*, not guessed.
  A plausible-looking constant tuned against a fake silently dropped real
  recalls; record the probe numbers next to the constant.
