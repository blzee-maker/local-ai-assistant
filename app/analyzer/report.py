"""Rendering a ScanReport — for the terminal, for a file, and for the model.

Three consumers, three shapes:

* `render_console` — what a person reads
* `to_markdown`    — what gets saved or shared
* `to_prompt_text` — a compact digest the LLM can answer questions from

The console view leads with caveats rather than burying them. A storage report
that says "reclaim 4 GB" without mentioning that a third of the files could not
be verified is telling a story, not reporting findings.
"""
from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.table import Table

from app.analyzer.integrity import Verdict
from app.analyzer.service import ScanReport, human_bytes

_VERDICT_STYLE = {
    Verdict.CORRUPT: "err",
    Verdict.MISLABELLED: "warn",
    Verdict.EMPTY: "meta",
}


def render_console(report: ScanReport, console: Console, *, top: int = 10) -> None:
    console.print()
    console.print("[bot]Disk analysis[/bot]")
    console.print(
        f"[meta]{report.stats.files_seen:,} files · "
        f"{human_bytes(report.stats.bytes_seen)} · "
        f"scanned in {report.duration_s:.1f}s[/meta]"
    )
    for root in report.roots:
        console.print(f"[meta]  {root}[/meta]")

    _render_caveats(report, console)
    _render_duplicates(report, console, top)
    _render_integrity(report, console, top)
    _render_stale(report, console, top)
    _render_usage(report, console)
    _render_bottom_line(report, console)


def _render_caveats(report: ScanReport, console: Console) -> None:
    """Anything that limits how far the numbers below can be trusted."""
    notes: list[str] = []
    s = report.stats

    if s.cloud_placeholders:
        notes.append(
            f"{s.cloud_placeholders:,} cloud-only files were not read "
            "(downloading them would defeat the purpose)"
        )
    if s.hardlink_duplicates:
        notes.append(
            f"{s.hardlink_duplicates:,} hard-linked paths collapsed "
            "(same bytes, deleting one frees nothing)"
        )
    if s.reparse_points_skipped:
        notes.append(f"{s.reparse_points_skipped:,} junctions/symlinks skipped")
    if s.permission_errors:
        notes.append(f"{len(s.permission_errors)} paths were unreadable")
    if s.hit_scan_limit:
        notes.append("scan limit reached — results are partial")
    if not report.atime_usable:
        notes.append(report.atime_note)
    else:
        notes.append(
            "hashing updates access times, so the earliest time seen across "
            "previous scans is used for the unused-files list"
        )

    if notes:
        console.print("\n[warn]Caveats[/warn]")
        for note in notes:
            console.print(f"  [meta]· {note}[/meta]")


def _render_duplicates(report: ScanReport, console: Console, top: int) -> None:
    console.print(
        f"\n[bot]Duplicates[/bot] [meta]{len(report.duplicate_groups)} groups · "
        f"{human_bytes(report.wasted_bytes)} recoverable[/meta]"
    )
    if not report.duplicate_groups:
        console.print("  [ok]none found[/ok]")
        return

    table = Table(show_header=True, header_style="meta", box=None, pad_edge=False)
    table.add_column("Wasted", justify="right")
    table.add_column("Copies", justify="right")
    table.add_column("Keep [meta](oldest)[/meta]", overflow="fold")

    for group in report.duplicate_groups[:top]:
        keeper = group.keeper
        table.add_row(
            human_bytes(group.wasted_bytes),
            str(group.count),
            f"{keeper.name}\n[meta]{keeper.path.parent}[/meta]",
        )
    console.print(table)
    if len(report.duplicate_groups) > top:
        console.print(f"  [meta]… {len(report.duplicate_groups) - top} more groups[/meta]")


def _render_integrity(report: ScanReport, console: Console, top: int) -> None:
    problems = report.problems
    console.print(
        f"\n[bot]Integrity[/bot] [meta]{report.integrity_checked:,} files verified · "
        f"{len(problems)} problems[/meta]"
    )

    if problems:
        table = Table(show_header=True, header_style="meta", box=None, pad_edge=False)
        table.add_column("Verdict")
        table.add_column("File", overflow="fold")
        table.add_column("Detail", overflow="fold")
        for result in problems[:top]:
            style = _VERDICT_STYLE.get(result.verdict, "warn")
            table.add_row(
                f"[{style}]{result.verdict.value}[/{style}]",
                result.entry.name,
                result.detail,
            )
        console.print(table)
        if len(problems) > top:
            console.print(f"  [meta]… {len(problems) - top} more[/meta]")
    else:
        console.print("  [ok]no corruption detected in verifiable formats[/ok]")

    # Never let "no problems" be mistaken for "everything is healthy".
    if report.unverifiable_count or report.integrity_skipped_for_budget:
        bits = []
        if report.unverifiable_count:
            bits.append(f"{report.unverifiable_count:,} had no applicable verifier")
        if report.integrity_skipped_for_budget:
            bits.append(
                f"{report.integrity_skipped_for_budget:,} skipped to stay within budget"
            )
        console.print(f"  [meta]not a clean bill of health: {'; '.join(bits)}[/meta]")


def _render_stale(report: ScanReport, console: Console, top: int) -> None:
    signal = "last opened" if report.atime_usable else "last modified"
    console.print(
        f"\n[bot]Large and unused[/bot] [meta]{len(report.stale_files)} files · "
        f"{human_bytes(report.stale_bytes)} · by {signal}[/meta]"
    )
    if not report.stale_files:
        console.print("  [ok]nothing large sitting idle[/ok]")
        return

    table = Table(show_header=True, header_style="meta", box=None, pad_edge=False)
    table.add_column("Size", justify="right")
    table.add_column("Age", justify="right")
    table.add_column("File", overflow="fold")

    for stale in report.stale_files[:top]:
        years = stale.age_days / 365.0
        age = f"{years:.1f}y" if years >= 1 else f"{stale.age_days:.0f}d"
        marker = " [meta](re-downloadable)[/meta]" if stale.is_disposable else ""
        table.add_row(
            human_bytes(stale.entry.size),
            age,
            f"{stale.entry.name}{marker}\n[meta]{stale.entry.path.parent}[/meta]",
        )
    console.print(table)
    if len(report.stale_files) > top:
        console.print(f"  [meta]… {len(report.stale_files) - top} more[/meta]")


def _render_usage(report: ScanReport, console: Console) -> None:
    if not report.usage:
        return
    console.print("\n[bot]Where the space goes[/bot]")
    widest = max(c.total_bytes for c in report.usage) or 1
    for category in report.usage[:8]:
        bar = "█" * max(1, round(24 * category.total_bytes / widest))
        console.print(
            f"  {category.label:<13} {human_bytes(category.total_bytes):>9}  "
            f"[meta]{bar} {category.count:,} files[/meta]"
        )


def _render_bottom_line(report: ScanReport, console: Console) -> None:
    console.print(
        f"\n[bot]Up to {human_bytes(report.reclaimable_bytes)} could be reclaimed[/bot] "
        "[meta](duplicates + unused; these overlap, so treat it as a ceiling)[/meta]"
    )
    console.print(
        "[meta]Nothing was changed. Review before deleting anything — "
        "`scan --export report.md` writes the full list.[/meta]"
    )


# ── file output ──────────────────────────────────────────────────
def to_markdown(report: ScanReport) -> str:
    lines: list[str] = []
    add = lines.append

    add("# Disk analysis report\n")
    add(f"- **Files scanned:** {report.stats.files_seen:,}")
    add(f"- **Total size:** {human_bytes(report.stats.bytes_seen)}")
    add(f"- **Duration:** {report.duration_s:.1f}s")
    add(f"- **Roots:** {', '.join(report.roots)}\n")

    add("## Caveats\n")
    add(f"- Cloud-only files not read: {report.stats.cloud_placeholders:,}")
    add(f"- Hard-linked paths collapsed: {report.stats.hardlink_duplicates:,}")
    add(f"- Junctions/symlinks skipped: {report.stats.reparse_points_skipped:,}")
    add(f"- Unreadable paths: {len(report.stats.permission_errors)}")
    add(f"- Staleness signal: {'last access' if report.atime_usable else 'last modified'}")
    add(f"  ({report.atime_note})\n")

    add(f"## Duplicates — {human_bytes(report.wasted_bytes)} recoverable\n")
    if report.duplicate_groups:
        for group in report.duplicate_groups:
            add(
                f"### {group.count} copies × {human_bytes(group.size)} "
                f"→ {human_bytes(group.wasted_bytes)} wasted"
            )
            add(f"- **keep:** `{group.keeper.path}`")
            for entry in group.redundant:
                add(f"- redundant: `{entry.path}`")
            add("")
    else:
        add("None found.\n")

    add(f"## Integrity — {len(report.problems)} problems\n")
    if report.problems:
        add("| Verdict | File | Detail |")
        add("|---|---|---|")
        for result in report.problems:
            add(f"| {result.verdict.value} | `{result.entry.path}` | {result.detail} |")
        add("")
    else:
        add("No corruption detected in verifiable formats.\n")
    add(
        f"Verified {report.integrity_checked:,} files; "
        f"{report.unverifiable_count:,} had no applicable verifier and are "
        f"**not** asserted to be healthy.\n"
    )

    signal = "last opened" if report.atime_usable else "last modified"
    add(f"## Large and unused — {human_bytes(report.stale_bytes)} (by {signal})\n")
    if report.stale_files:
        add("| Size | Age | File |")
        add("|---|---|---|")
        for stale in report.stale_files:
            add(
                f"| {human_bytes(stale.entry.size)} | {stale.age_days:.0f}d | "
                f"`{stale.entry.path}` |"
            )
        add("")
    else:
        add("Nothing large sitting idle.\n")

    add("## Where the space goes\n")
    add("| Category | Files | Size |")
    add("|---|---|---|")
    for category in report.usage:
        add(f"| {category.label} | {category.count:,} | {human_bytes(category.total_bytes)} |")
    add("")
    add(
        "> Nothing was modified by this scan. All findings are advisory; "
        "review each file before deleting it."
    )
    return "\n".join(lines)


def to_prompt_text(report: ScanReport, *, limit: int = 12) -> str:
    """A compact digest the model can answer questions from."""
    lines: list[str] = [
        f"Disk scan of {', '.join(report.roots)}",
        f"{report.stats.files_seen:,} files, {human_bytes(report.stats.bytes_seen)} total.",
        "",
        f"DUPLICATES: {len(report.duplicate_groups)} groups wasting "
        f"{human_bytes(report.wasted_bytes)}.",
    ]
    for group in report.duplicate_groups[:limit]:
        lines.append(
            f"- {group.count} copies of {group.keeper.name} "
            f"({human_bytes(group.size)} each, {human_bytes(group.wasted_bytes)} wasted); "
            f"keep {group.keeper.path}"
        )

    lines.append("")
    lines.append(
        f"INTEGRITY: {report.integrity_checked:,} files verified, "
        f"{len(report.problems)} problems, "
        f"{report.unverifiable_count:,} unverifiable (health unknown, not assumed good)."
    )
    for result in report.problems[:limit]:
        lines.append(f"- {result.verdict.value}: {result.entry.path} — {result.detail}")

    signal = "last opened" if report.atime_usable else "last modified"
    lines.append("")
    lines.append(
        f"UNUSED (by {signal}): {len(report.stale_files)} files, "
        f"{human_bytes(report.stale_bytes)}."
    )
    for stale in report.stale_files[:limit]:
        lines.append(
            f"- {human_bytes(stale.entry.size)}, {stale.age_days:.0f} days old: "
            f"{stale.entry.path}"
        )

    lines.append("")
    lines.append("STORAGE BY CATEGORY:")
    for category in report.usage[:8]:
        lines.append(
            f"- {category.label}: {human_bytes(category.total_bytes)} "
            f"({category.count:,} files)"
        )

    if report.stats.cloud_placeholders:
        lines.append("")
        lines.append(
            f"NOTE: {report.stats.cloud_placeholders:,} cloud-only files were not "
            "inspected; they occupy no local disk space."
        )
    return "\n".join(lines)


def deletion_script(report: ScanReport, path: Path) -> int:
    """Write a **reviewable** PowerShell script that sends redundant duplicate
    copies to the Recycle Bin.

    Deliberately not executed: the assistant proposes, the user disposes. Recycle
    Bin rather than permanent delete, so a mistake stays recoverable. Returns the
    number of files the script would remove.
    """
    lines = [
        "# Generated by the local AI assistant — REVIEW BEFORE RUNNING.",
        "# Sends redundant duplicate copies to the Recycle Bin (recoverable).",
        "# The oldest copy of each group is kept and is never listed here.",
        "",
        "Add-Type -AssemblyName Microsoft.VisualBasic",
        "",
    ]
    count = 0
    for group in report.duplicate_groups:
        lines.append(
            f"# {group.count} copies of {group.keeper.name} "
            f"({human_bytes(group.size)} each) — keeping {group.keeper.path}"
        )
        for entry in group.redundant:
            escaped = str(entry.path).replace("'", "''")
            lines.append(
                "[Microsoft.VisualBasic.FileIO.FileSystem]::DeleteFile("
                f"'{escaped}', 'OnlyErrorDialogs', 'SendToRecycleBin')"
            )
            count += 1
        lines.append("")

    lines.append(f"# {count} file(s) would be sent to the Recycle Bin.")

    # utf-8-sig, not utf-8: Windows PowerShell 5.1 decodes a BOM-less .ps1 as
    # ANSI. Without the BOM, any path containing a non-ASCII character (an
    # accent, an emoji in a filename) is silently mangled — and a delete script
    # pointed at a mangled path is exactly the kind of mistake that must not be
    # possible here.
    path.write_text("\n".join(lines), encoding="utf-8-sig", newline="\r\n")
    return count
