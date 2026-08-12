"""Tests for the integrity verifier.

Every case here came from a real misclassification during development. The
verifier's failure mode that matters is the false positive: telling someone a
working file is corrupt invites them to delete it. So the suite checks both
directions — healthy files must pass, damaged files must not.

Fixtures are synthesized, so the suite is self-contained and does not read the
developer's personal folders.

Run with:  python -m pytest tests/ -v
"""
from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pytest

from app.analyzer import integrity
from app.analyzer.integrity import Verdict
from app.analyzer.walker import FileEntry


def entry_for(path: Path) -> FileEntry:
    st = path.stat()
    return FileEntry(
        path=path, size=st.st_size, mtime=st.st_mtime, atime=st.st_atime,
        root_label="test",
    )


def verdict_of(path: Path) -> Verdict:
    return integrity.check(entry_for(path)).verdict


@pytest.fixture
def jpeg_bytes() -> bytes:
    """A *noisy* JPEG, deliberately.

    A flat-colour image compresses to almost nothing, so half of it still holds
    a complete decodable picture and truncation tests silently pass for the
    wrong reason. Random pixels resist compression, so cutting the file really
    does remove image data.
    """
    import os

    from PIL import Image

    buf = io.BytesIO()
    noise = Image.frombytes("RGB", (320, 240), os.urandom(320 * 240 * 3))
    noise.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


@pytest.fixture
def png_bytes() -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (10, 200, 90)).save(buf, format="PNG")
    return buf.getvalue()


# ── healthy files must not be flagged ────────────────────────────
def test_plain_jpeg_is_ok(tmp_path, jpeg_bytes):
    p = tmp_path / "photo.jpg"
    p.write_bytes(jpeg_bytes)
    assert verdict_of(p) == Verdict.OK


def test_jpeg_with_trailing_metadata_is_ok(tmp_path, jpeg_bytes):
    """The original bug: 21 phone photos called corrupt.

    Samsung appends a 'SEFT' metadata trailer after the JPEG end-of-image
    marker. A verifier that only inspects the final bytes never sees the marker
    and declares a perfectly good photo truncated.
    """
    p = tmp_path / "motion_photo.jpg"
    trailer = b"\x00\x00\x01\n\x0e\x00\x00\x00Image_UTC_Data1700000000000" + b"\x00" * 64 + b"SEFT"
    p.write_bytes(jpeg_bytes + trailer)
    assert verdict_of(p) == Verdict.OK


def test_png_with_appended_data_is_ok(tmp_path, png_bytes):
    p = tmp_path / "image.png"
    p.write_bytes(png_bytes + b"\x00" * 512)
    assert verdict_of(p) == Verdict.OK


def test_valid_zip_is_ok(tmp_path):
    p = tmp_path / "archive.zip"
    with zipfile.ZipFile(p, "w") as zf:
        zf.writestr("hello.txt", "hi")
    assert verdict_of(p) == Verdict.OK


def test_valid_docx_is_ok(tmp_path):
    p = tmp_path / "doc.docx"
    with zipfile.ZipFile(p, "w") as zf:
        zf.writestr("[Content_Types].xml", "<Types/>")
        zf.writestr("word/document.xml", "<document/>")
    assert verdict_of(p) == Verdict.OK


def test_valid_json_is_ok(tmp_path):
    p = tmp_path / "data.json"
    p.write_text(json.dumps({"a": [1, 2, 3]}), encoding="utf-8")
    assert verdict_of(p) == Verdict.OK


# ── damaged files must be caught ─────────────────────────────────
@pytest.mark.parametrize("keep", [0.5, 0.25, 0.05])
def test_truncated_jpeg_is_corrupt(tmp_path, jpeg_bytes, keep):
    p = tmp_path / "cut.jpg"
    p.write_bytes(jpeg_bytes[: int(len(jpeg_bytes) * keep)])
    assert verdict_of(p) == Verdict.CORRUPT


def test_verdict_does_not_depend_on_end_marker_position(tmp_path, jpeg_bytes):
    """Byte-marker searching was tried for this and was wrong in both directions.

    Looking for an end marker near EOF called good photos truncated (their marker
    sat before a metadata trailer) and called a half-file complete (a stray FFD9
    appeared inside embedded video). Verdicts now come from the decoder, so
    marker position must not sway them: a complete file stays OK with trailing
    junk appended, and a cut file stays CORRUPT.
    """
    intact = tmp_path / "intact_with_trailer.jpg"
    intact.write_bytes(jpeg_bytes + b"TRAILINGJUNK" * 40)
    assert verdict_of(intact) == Verdict.OK

    cut = tmp_path / "cut.jpg"
    cut.write_bytes(jpeg_bytes[: len(jpeg_bytes) // 2])
    assert verdict_of(cut) == Verdict.CORRUPT


def test_known_limit_repaired_stream_is_reported_decodable(tmp_path, jpeg_bytes):
    """A documented boundary, not an oversight.

    Cutting a JPEG and appending a valid end-of-image marker produces a file
    that genuinely decodes — every image viewer opens it, showing a shortened
    picture. The checker verifies that a file *decodes*, which is not the same
    as proving its content is complete; no decoder can know how many rows the
    photographer originally captured.

    Reporting this as corrupt would contradict what the user sees when they open
    it, so it is reported OK. The report's wording is chosen to match: it claims
    files are readable, never that they are pristine.
    """
    p = tmp_path / "repaired.jpg"
    p.write_bytes(jpeg_bytes[: len(jpeg_bytes) // 2] + b"\xff\xd9")
    assert verdict_of(p) == Verdict.OK


def test_corrupt_zip_is_caught(tmp_path):
    p = tmp_path / "broken.zip"
    p.write_bytes(b"PK\x03\x04" + b"\x00" * 500)
    assert verdict_of(p) == Verdict.CORRUPT


def test_docx_missing_manifest_is_caught(tmp_path):
    p = tmp_path / "fake.docx"
    with zipfile.ZipFile(p, "w") as zf:
        zf.writestr("random.txt", "not really a docx")
    assert verdict_of(p) == Verdict.CORRUPT


def test_invalid_json_is_caught(tmp_path):
    p = tmp_path / "broken.json"
    p.write_text('{"a": [1, 2,', encoding="utf-8")
    assert verdict_of(p) == Verdict.CORRUPT


def test_empty_file_is_empty(tmp_path):
    p = tmp_path / "nothing.pdf"
    p.write_bytes(b"")
    assert verdict_of(p) == Verdict.EMPTY


def test_pdf_without_signature_is_flagged(tmp_path):
    p = tmp_path / "notreally.pdf"
    p.write_bytes(b"just some text, definitely not a pdf" * 20)
    assert verdict_of(p) in {Verdict.CORRUPT, Verdict.MISLABELLED}


# ── identification ───────────────────────────────────────────────
def test_png_named_jpg_is_mislabelled(tmp_path, png_bytes):
    p = tmp_path / "actually_png.jpg"
    p.write_bytes(png_bytes)
    assert verdict_of(p) == Verdict.MISLABELLED


def test_java_stream_named_pdf_is_identified(tmp_path):
    """Three real 'PDF' downloads turned out to be Java serialized streams.
    Naming what the file actually is beats 'missing PDF signature'."""
    p = tmp_path / "Form_pdf_123.pdf"
    p.write_bytes(b"\xac\xed\x00\x05ur\x00\x13[Ljava.lang.Object;" + b"\x00" * 200)
    result = integrity.check(entry_for(p))
    assert result.verdict == Verdict.MISLABELLED
    assert "java" in result.detail.lower()


# ── the honest-refusal contract ──────────────────────────────────
def test_office_lock_file_is_not_reported_corrupt(tmp_path):
    """Word/Excel leave "~$name.docx" stubs beside open documents. They carry a
    document extension but are not documents, so a signature check calls them
    corrupt — a false alarm about a file the user never created."""
    p = tmp_path / "~$budget.xlsx"
    p.write_bytes(b"\x00\x01 owner lock stub")
    assert verdict_of(p) == Verdict.UNVERIFIABLE


def test_unknown_format_is_unverifiable_not_ok(tmp_path):
    """An unrecognised format must never be reported as healthy."""
    p = tmp_path / "mystery.xyz"
    p.write_bytes(b"\x01\x02\x03\x04" * 100)
    assert verdict_of(p) == Verdict.UNVERIFIABLE


def test_cloud_placeholder_is_skipped_not_read(tmp_path):
    """Reading a cloud-only file would download it — the opposite of the point."""
    p = tmp_path / "cloud.pdf"
    p.write_bytes(b"%PDF-1.4 stub")
    entry = entry_for(p)
    entry.is_cloud_placeholder = True
    result = integrity.check(entry)
    assert result.verdict == Verdict.SKIPPED
