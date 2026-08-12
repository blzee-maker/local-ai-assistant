"""First-run consent for reading the user's folders.

The assistant reads personal directories. That should never happen silently on
first launch, and an approval given once should not silently widen later.

Two rules make that concrete:

* **Consent is per-root, and recorded.** `data/consent.json` stores exactly which
  folders were approved and when.
* **Scope changes re-prompt.** The stored record includes a fingerprint of the
  approved root set. If configuration later adds a folder, the fingerprint stops
  matching and the user is asked again — approving Downloads once must not
  quietly become approval of a folder added months later.

Declining is a first-class outcome: it is recorded too, so the prompt does not
reappear on every command.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from config import settings

CONSENT_VERSION = 1
CONSENT_PATH = Path(settings.upload_dir).parent / "consent.json"


@dataclass
class ConsentRecord:
    version: int = CONSENT_VERSION
    granted: bool = False
    decided_at: float = 0.0
    approved_roots: list[str] = field(default_factory=list)
    scope_fingerprint: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "granted": self.granted,
            "decided_at": self.decided_at,
            "approved_roots": self.approved_roots,
            "scope_fingerprint": self.scope_fingerprint,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConsentRecord":
        return cls(
            version=int(data.get("version", 0)),
            granted=bool(data.get("granted", False)),
            decided_at=float(data.get("decided_at", 0.0)),
            approved_roots=list(data.get("approved_roots", [])),
            scope_fingerprint=str(data.get("scope_fingerprint", "")),
        )


def fingerprint(roots: list[str]) -> str:
    """Stable identity for a set of folders, order-independent."""
    joined = "\n".join(sorted(str(r).lower() for r in roots))
    return hashlib.blake2b(joined.encode("utf-8"), digest_size=16).hexdigest()


def load() -> ConsentRecord | None:
    if not CONSENT_PATH.exists():
        return None
    try:
        return ConsentRecord.from_dict(json.loads(CONSENT_PATH.read_text("utf-8")))
    except (OSError, ValueError):
        return None  # unreadable record == no record; we will ask again


def save(record: ConsentRecord) -> None:
    CONSENT_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONSENT_PATH.write_text(
        json.dumps(record.to_dict(), indent=2), encoding="utf-8"
    )


def current_roots() -> list[str]:
    from app import files as filesvc

    return [str(path) for _label, path in filesvc.allowed_roots()]


def status() -> tuple[str, ConsentRecord | None]:
    """One of: 'none', 'granted', 'declined', 'stale'."""
    record = load()
    if record is None or record.version != CONSENT_VERSION:
        return "none", record
    if fingerprint(current_roots()) != record.scope_fingerprint:
        return "stale", record
    return ("granted" if record.granted else "declined"), record


def grant(roots: list[str] | None = None) -> ConsentRecord:
    approved = roots if roots is not None else current_roots()
    record = ConsentRecord(
        granted=True,
        decided_at=time.time(),
        approved_roots=approved,
        scope_fingerprint=fingerprint(approved),
    )
    save(record)
    return record


def decline() -> ConsentRecord:
    roots = current_roots()
    record = ConsentRecord(
        granted=False,
        decided_at=time.time(),
        approved_roots=[],
        scope_fingerprint=fingerprint(roots),
    )
    save(record)
    return record


def revoke() -> None:
    CONSENT_PATH.unlink(missing_ok=True)


# ── interactive prompt ───────────────────────────────────────────
_EXPLAINER = """[bot]Permission to analyse your files[/bot]

The assistant can scan these folders to find duplicate files, corrupted
downloads, and large files you no longer use:
"""

_GUARANTEES = """
[meta]What it does:[/meta]
  · reads file names, sizes, and dates
  · reads file contents only to fingerprint duplicates and verify formats
  · never opens cloud-only files (that would download them)

[meta]What it never does:[/meta]
  · delete, move, or modify anything — the report is advisory
  · send anything off this machine; the scan is entirely local
"""


def prompt_for_consent(console) -> bool:
    """Ask once, record the answer. Returns True if analysis may proceed."""
    roots = current_roots()
    console.print(_EXPLAINER)
    for root in roots:
        console.print(f"  [user]{root}[/user]")
    console.print(_GUARANTEES)

    try:
        answer = input("Allow local file analysis? [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        console.print("\n[warn]No answer given — analysis not started.[/warn]")
        return False

    if answer in {"y", "yes"}:
        grant(roots)
        console.print("[ok]Approved.[/ok] [meta]Revoke any time with: assistant consent --revoke[/meta]")
        return True

    decline()
    console.print(
        "[warn]Declined.[/warn] [meta]No files were read. "
        "Run `assistant consent --grant` to change this.[/meta]"
    )
    return False


def ensure_consent(console) -> bool:
    """Gate any file-analysis command. Prompts on first use or scope change."""
    state, record = status()

    if state == "granted":
        return True

    if state == "declined":
        console.print(
            "[warn]File analysis was previously declined.[/warn] "
            "[hint]assistant consent --grant[/hint]"
        )
        return False

    if state == "stale":
        console.print(
            "[warn]The set of folders has changed since you approved analysis.[/warn]"
        )
        if record and record.approved_roots:
            console.print("[meta]Previously approved:[/meta]")
            for root in record.approved_roots:
                console.print(f"  [meta]{root}[/meta]")
        console.print()

    return prompt_for_consent(console)
