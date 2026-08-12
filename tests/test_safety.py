"""Tests for the guarantees the assistant makes about a user's files.

These are the promises in the README: allowlist-bounded, read-only, and never
analysed without recorded consent. They are the ones worth a regression test,
because breaking them quietly is how a privacy-first tool stops being one.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from app import consent, files as filesvc


# ── allowlist / traversal ────────────────────────────────────────
def test_traversal_outside_allowed_roots_is_rejected():
    with pytest.raises(ValueError):
        filesvc.resolve_allowed(r"C:\Windows\System32\drivers\etc\hosts")


def test_relative_traversal_is_rejected():
    roots = filesvc.allowed_roots()
    if not roots:
        pytest.skip("no allowed roots on this machine")
    _label, root = roots[0]
    with pytest.raises(ValueError):
        filesvc.resolve_allowed(str(root / ".." / ".." / "Windows" / "win.ini"))


def test_unsupported_extension_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        filesvc.resolve_allowed(str(tmp_path / "payload.exe"))


def test_sample_docs_file_resolves():
    """A file genuinely inside an allowed root must still work."""
    sample = Path("sample_docs/nimbusedge_handbook.md").resolve()
    if not sample.exists():
        pytest.skip("sample doc missing")
    assert filesvc.resolve_allowed(str(sample)) == sample


# ── consent ──────────────────────────────────────────────────────
def test_fingerprint_is_order_independent():
    a = consent.fingerprint([r"C:\A", r"C:\B"])
    b = consent.fingerprint([r"C:\B", r"C:\A"])
    assert a == b


def test_fingerprint_changes_when_a_root_is_added():
    before = consent.fingerprint([r"C:\A"])
    after = consent.fingerprint([r"C:\A", r"C:\B"])
    assert before != after


def test_adding_a_root_invalidates_prior_consent(tmp_path, monkeypatch):
    """Approving Downloads once must not silently extend to a folder added
    later — the stored fingerprint has to stop matching."""
    monkeypatch.setattr(consent, "CONSENT_PATH", tmp_path / "consent.json")

    monkeypatch.setattr(consent, "current_roots", lambda: [r"C:\A"])
    consent.grant()
    assert consent.status()[0] == "granted"

    monkeypatch.setattr(consent, "current_roots", lambda: [r"C:\A", r"C:\B"])
    assert consent.status()[0] == "stale"


def test_declining_is_remembered(tmp_path, monkeypatch):
    monkeypatch.setattr(consent, "CONSENT_PATH", tmp_path / "consent.json")
    monkeypatch.setattr(consent, "current_roots", lambda: [r"C:\A"])
    consent.decline()
    assert consent.status()[0] == "declined"


def test_revoke_clears_the_record(tmp_path, monkeypatch):
    monkeypatch.setattr(consent, "CONSENT_PATH", tmp_path / "consent.json")
    monkeypatch.setattr(consent, "current_roots", lambda: [r"C:\A"])
    consent.grant()
    consent.revoke()
    assert consent.status()[0] == "none"


def test_corrupt_consent_file_means_ask_again(tmp_path, monkeypatch):
    path = tmp_path / "consent.json"
    path.write_text("{ not json", encoding="utf-8")
    monkeypatch.setattr(consent, "CONSENT_PATH", path)
    assert consent.status()[0] == "none"


# ── read-only guarantee ──────────────────────────────────────────
def test_analyzer_package_contains_no_write_calls():
    """A structural check that the scanner cannot modify user files.

    Parsed with `ast` rather than grepped: a substring search also matches the
    word inside comments and docstrings, and this module explains in prose why
    `os.utime` was rejected — which a naive scan then reported as a violation.
    Only real call expressions count.

    Blunt, but it fails loudly the day someone adds a convenience
    'just delete it for them' call to the analysis path.
    """
    import ast

    forbidden = {
        "remove", "unlink", "rmtree", "move", "rename", "utime",
        "rmdir", "truncate", "replace",
    }
    # report.py writes the report and cleanup script the user explicitly asked
    # for, to a path they supplied — never to a scanned file.
    exempt_files = {"report.py"}
    offenders: list[str] = []

    for py in Path("app/analyzer").glob("*.py"):
        if py.name in exempt_files:
            continue
        tree = ast.parse(py.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = (
                func.attr if isinstance(func, ast.Attribute)
                else func.id if isinstance(func, ast.Name)
                else None
            )
            if name in forbidden:
                offenders.append(f"{py.name}:{node.lineno} {name}()")

    assert not offenders, f"write operations found in analyzer: {offenders}"


def test_analyzer_opens_files_read_only():
    """Every open() in the analyzer must use a read mode."""
    import ast

    offenders: list[str] = []
    for py in Path("app/analyzer").glob("*.py"):
        tree = ast.parse(py.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not (isinstance(func, ast.Attribute) and func.attr == "open"):
                continue
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    if any(c in arg.value for c in "wax+"):
                        offenders.append(f"{py.name}:{node.lineno} open({arg.value!r})")

    assert not offenders, f"non-read opens in analyzer: {offenders}"
