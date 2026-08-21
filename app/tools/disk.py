"""Running a disk scan from inside a conversation.

Scanning used to be reachable only as `assistant scan`, and chat could do no
more than report on whatever that command had last produced. Asked about
duplicates with no scan on record, the assistant told the user which command to
type — the same failure as telling someone to run `systeminfo`, only better
disguised. The assistant does the work; it does not hand it back.

It is still not something that happens behind the user's back. A scan reads
every file in the approved folders and takes minutes, so it asks first, says how
long that is likely to be, and warns when memory is too tight for it to go
quickly. What it never does is modify anything: the scan is read-only, and
deletion remains a script the user reviews and runs themselves (rule 2).
"""
from __future__ import annotations

import time

from app.tools.base import Risk, Tool, ToolContext, ToolResult
from config import PROJECT_ROOT


def diskintent_cleanup_words() -> tuple[str, ...]:
    from app.core import diskintent

    return diskintent._CLEANUP_VERBS + diskintent._CLEANUP_NOUNS

# Below this, the model and a full scan are competing for the same few hundred
# megabytes and the scan crawls. Said out loud rather than left for the user to
# discover by watching a progress line stall (rule 4).
TIGHT_MEMORY_GB = 1.0

# How often progress is allowed to speak. The walk fires a callback per file,
# and a line per file is not progress, it is noise.
PROGRESS_INTERVAL_S = 1.5

# Written into the project rather than the user's Desktop or Documents:
# somewhere predictable, and somewhere that is not their space to litter.
CLEANUP_SCRIPT_PATH = PROJECT_ROOT / "cleanup.ps1"


def _free_memory_gb() -> float | None:
    try:
        import psutil

        return psutil.virtual_memory().available / (1024**3)
    except Exception:
        return None


def _estimate() -> str:
    """An honest sentence about cost, with the memory caveat when it applies."""
    base = "It reads every file in those folders and usually takes a few minutes."
    free = _free_memory_gb()
    if free is not None and free < TIGHT_MEMORY_GB:
        return (
            f"{base} Memory is tight right now ({free:.1f} GB free), so expect "
            "the slow end of that."
        )
    return base


def perform_scan(context: ToolContext, question: str) -> ToolResult:
    """Ask, scan, cache, and hand back grounding for the original question.

    Shared by both disk tools so there is one scan path, and so an offer to scan
    made while answering behaves exactly like an explicit request for one.
    """
    from app import consent
    from app.analyzer import run_scan
    from app.core import diskintent

    state, _record = consent.status()
    if state == "declined":
        return ToolResult.failure(
            "File analysis was previously declined by the user.",
            display="file analysis declined",
            final_text=(
                "File analysis is switched off — you declined it previously. "
                "Turn it back on with `assistant consent --grant` and I can scan "
                "then."
            ),
        )

    roots = consent.current_roots()
    if state == "granted":
        prompt = (
            f"Scan {len(roots)} folder(s) for duplicates, corrupt files and idle "
            f"storage? {_estimate()} Nothing is modified."
        )
    else:
        listed = ", ".join(roots) if roots else "your folders"
        prompt = (
            f"Analyse your files for the first time? I would read {listed}. "
            f"{_estimate()} Nothing is modified, and nothing leaves this machine."
        )

    if not context.ask_confirmation(prompt):
        # Deliberately not recorded as a refusal of file analysis. Declining
        # *this* scan means "not now"; consent.decline() would mean "never", and
        # silently widening a no is the mirror of silently widening a yes
        # (rule 3).
        return ToolResult.failure(
            "The user declined to run a scan just now.",
            display="scan declined",
            final_text="No scan run — you said no. Ask me again whenever you want one.",
        )

    if state != "granted":
        consent.grant()

    context.report_progress("starting scan")
    last_spoke = 0.0

    def on_phase(name: str) -> None:
        nonlocal last_spoke
        last_spoke = time.time()
        context.report_progress(name)

    def on_progress(done: int, total: int) -> None:
        nonlocal last_spoke
        now = time.time()
        if now - last_spoke < PROGRESS_INTERVAL_S:
            return
        last_spoke = now
        context.report_progress(f"{done:,} of {total:,}" if total else f"{done:,}")

    try:
        report = run_scan(on_phase=on_phase, on_progress=on_progress)
    except Exception as exc:
        # Degrade, never crash: a scan that dies half way must leave the
        # conversation intact and say what happened (rule 10).
        return ToolResult.failure(
            f"The disk scan failed: {exc}",
            display="scan failed",
            final_text=f"The scan could not finish: {exc}",
        )

    diskintent.save_report(report)
    context.report_progress("done")

    grounded, note = diskintent.ground_prompt(question)
    if grounded is None:
        # The scan ran but its cache could not be read back. Say that, rather
        # than answering from nothing (rule 5).
        return ToolResult.failure(
            note or "The scan finished but its results could not be read back.",
            display="scan results unavailable",
        )

    return ToolResult(
        ok=True,
        content=grounded,
        display=f"scanned {report.stats.files_seen:,} files in {report.duration_s:.0f}s",
        meta={
            "files_seen": report.stats.files_seen,
            "duration_s": report.duration_s,
            "summary": report.summary_line(),
        },
    )


class RunDiskScanTool(Tool):
    """Scan the approved folders now, then answer from what it found."""

    name = "run_disk_scan"
    description = "Scan your folders now for duplicates, corrupt files and idle storage"

    # READ, and deliberately so: the scan opens files and writes none of them.
    # The prompt it raises is about *cost*, not danger — minutes of work on a
    # machine with little memory to spare — and the risk tiers do not model
    # cost. So the tool asks for itself rather than mislabelling a read as
    # destructive, which would train the user to wave through prompts that
    # matter.
    risk = Risk.READ

    _TRIGGERS = (
        "scan", "rescan", "re-scan", "run a scan", "do a scan", "scan again",
        "analyse my files", "analyze my files", "analyse my disk",
        "analyze my disk", "check my files", "check my disk", "go through my",
        "look through my", "refresh the scan", "update the scan", "new scan",
        "fresh scan", "file analysis",
    )

    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    "Run a fresh scan of the approved folders for duplicate "
                    "files, corrupt or mislabelled files, and large files that "
                    "have not been touched in a long time. Takes minutes and "
                    "asks the user first. Use when they ask for a scan, or when "
                    "the existing findings are too old to trust."
                ),
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        }

    def matches(self, text: str) -> bool:
        lowered = text.lower()
        return any(trigger in lowered for trigger in self._TRIGGERS)

    def match_score(self, text: str) -> int:
        from app.tools.system import count_triggers

        return count_triggers(text, self._TRIGGERS)

    def run(self, arguments: dict, context: ToolContext) -> ToolResult:
        return perform_scan(context, context.request_text)


class PrepareCleanupTool(Tool):
    """Write the reviewable delete script, for the user to check and run."""

    name = "prepare_cleanup"
    description = "Write a reviewable script listing duplicate files to remove"

    # WRITE: it creates one new file of its own and touches nothing else. The
    # deletion itself stays with the user, which is the whole point of handing
    # them a script instead of a result (rule 2).
    risk = Risk.WRITE

    # Both the trigger words and the gate live in app.core.diskintent, so this
    # tool and the report tool cannot disagree about which requests are asking
    # for removal rather than for a description.
    _TRIGGERS = diskintent_cleanup_words()

    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    "Write a reviewable PowerShell script that would send "
                    "redundant duplicate copies to the Recycle Bin. Never "
                    "deletes anything itself. Use when the user asks to clean "
                    "up, delete duplicates, or free up space."
                ),
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        }

    def matches(self, text: str) -> bool:
        from app.core import diskintent

        return diskintent.asks_to_clean_up(text)

    def match_score(self, text: str) -> int:
        from app.tools.system import count_triggers

        return count_triggers(text, self._TRIGGERS)

    def run(self, arguments: dict, context: ToolContext) -> ToolResult:
        from app.core import diskintent

        cached = diskintent.load_report() or {}
        script = cached.get("cleanup_script")
        if not script:
            # No scan, or one cached before scripts were kept. Offer to do the
            # work rather than naming a command to type.
            return perform_scan(context, context.request_text)

        count = int(cached.get("cleanup_count") or 0)
        if count == 0:
            return ToolResult(
                ok=True,
                display="nothing to clean up",
                final_text=(
                    "There are no redundant duplicate copies to remove, so "
                    "there is nothing for a cleanup script to do."
                ),
                meta={"cleanup_count": 0},
            )

        target = CLEANUP_SCRIPT_PATH
        try:
            # utf-8-sig and CRLF for the same reason the CLI writes it that way:
            # PowerShell 5.1 reads a BOM-less .ps1 as ANSI and mangles any
            # non-ASCII path in it, and a delete script pointed at a mangled
            # path is the one mistake that must not be possible here.
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(script, encoding="utf-8-sig", newline="\r\n")
        except OSError as exc:
            return ToolResult.failure(
                f"The cleanup script could not be written: {exc}",
                display="could not write script",
                final_text=f"I could not write the cleanup script: {exc}",
            )

        age = diskintent.describe_age(cached.get("saved_at", 0.0))
        return ToolResult(
            ok=True,
            display=f"wrote cleanup script ({count} files)",
            final_text=(
                f"Written to {target}. It lists {count} redundant copies, based "
                f"on the scan from {age}, and sends them to the Recycle Bin so "
                "anything you change your mind about is recoverable. Nothing has "
                "been deleted: read it, then run it yourself if you agree."
            ),
            meta={"path": str(target), "cleanup_count": count, "scan_age": age},
        )
