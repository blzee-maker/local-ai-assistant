"""Format-level integrity verification — and an explicit refusal to guess.

Generic "is this file corrupt?" is not answerable. A truncated MP4 and a healthy
MP4 are both just bytes; without decoding every format in existence there is no
signal. Tools that claim otherwise are usually reporting extension mismatches
and calling it corruption.

So this module verifies only what it can actually prove, and says so:

    OK            structure parsed and checks out
    CORRUPT       the format's own rules are violated (truncated, bad container)
    MISLABELLED   contents don't match the extension (.pdf that is really a ZIP)
    EMPTY         zero bytes
    UNREADABLE    permission denied or I/O error
    SKIPPED       cloud placeholder — reading it would force a download
    UNVERIFIABLE  no verifier for this format; nothing is claimed either way

`UNVERIFIABLE` is the important one. A file landing there is *not* asserted to be
healthy — the report says how many files could not be checked, so the number is
never mistaken for a clean bill of health.
"""
from __future__ import annotations

import contextlib
import json
import logging
import zipfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from app.analyzer.walker import FileEntry


@contextlib.contextmanager
def _quiet_pypdf():
    """Mute pypdf's per-object recovery chatter.

    Real-world PDFs are full of minor spec violations that pypdf recovers from
    while logging a line each ("Ignoring wrong pointing object ..."). Scanning a
    few hundred of them buries the actual report under thousands of lines. These
    are recovered conditions, not findings — a genuine failure raises instead.
    """
    logger = logging.getLogger("pypdf")
    previous = logger.level
    logger.setLevel(logging.CRITICAL)
    try:
        yield
    finally:
        logger.setLevel(previous)


class Verdict(str, Enum):
    OK = "ok"
    CORRUPT = "corrupt"
    MISLABELLED = "mislabelled"
    EMPTY = "empty"
    UNREADABLE = "unreadable"
    SKIPPED = "skipped"
    UNVERIFIABLE = "unverifiable"


@dataclass
class IntegrityResult:
    entry: FileEntry
    verdict: Verdict
    detail: str = ""

    @property
    def is_problem(self) -> bool:
        return self.verdict in {Verdict.CORRUPT, Verdict.MISLABELLED, Verdict.EMPTY}


# Leading bytes that identify a format. Order matters only for readability.
_MAGIC: list[tuple[bytes, str]] = [
    (b"%PDF-", "pdf"),
    (b"PK\x03\x04", "zip"),
    (b"\x89PNG\r\n\x1a\n", "png"),
    (b"\xff\xd8\xff", "jpeg"),
    (b"GIF87a", "gif"),
    (b"GIF89a", "gif"),
    (b"BM", "bmp"),
    (b"\x1f\x8b", "gzip"),
    (b"Rar!\x1a\x07", "rar"),
    (b"7z\xbc\xaf\x27\x1c", "7z"),
    (b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1", "ole"),  # legacy .doc/.xls/.ppt
    (b"ID3", "mp3"),
    (b"OggS", "ogg"),
    (b"fLaC", "flac"),
    (b"\x00\x00\x01\x00", "ico"),
    (b"MZ", "exe"),
    # Java object serialization. Worth naming explicitly: a government portal
    # here saved three "PDF" downloads as Java serialized streams, and "missing
    # PDF signature" is far less useful than saying what the file actually is.
    (b"\xac\xed\x00", "java-serialized"),
    (b"<!DOCTYPE html", "html"),
    (b"<html", "html"),
]

# Which magic families are legitimate for a given extension.
_EXPECTED: dict[str, set[str]] = {
    ".pdf": {"pdf"},
    ".png": {"png"},
    ".jpg": {"jpeg"},
    ".jpeg": {"jpeg"},
    ".gif": {"gif"},
    ".bmp": {"bmp"},
    ".ico": {"ico"},
    ".docx": {"zip"},
    ".xlsx": {"zip"},
    ".pptx": {"zip"},
    ".zip": {"zip"},
    ".jar": {"zip"},
    ".epub": {"zip"},
    ".doc": {"ole"},
    ".xls": {"ole"},
    ".ppt": {"ole"},
    ".gz": {"gzip"},
    ".tgz": {"gzip"},
    ".rar": {"rar"},
    ".7z": {"7z"},
    ".exe": {"exe"},
    ".dll": {"exe"},
    ".webp": {"riff"},
    ".wav": {"riff"},
    ".avi": {"riff"},
}

# Extensions we can structurally validate, not merely sniff.
_DEEP_CHECKED = {
    ".pdf", ".docx", ".xlsx", ".pptx", ".zip", ".jar", ".epub",
    ".png", ".jpg", ".jpeg", ".gif", ".json", ".gz", ".tgz",
}


def _sniff(head: bytes) -> str | None:
    if head[:4] == b"RIFF":
        return "riff"
    for signature, family in _MAGIC:
        if head.startswith(signature):
            return family
    return None


def check(entry: FileEntry) -> IntegrityResult:
    """Verify one file as far as its format allows."""
    if entry.is_cloud_placeholder:
        return IntegrityResult(
            entry, Verdict.SKIPPED, "cloud-only file; not downloaded to check"
        )
    if entry.size == 0:
        return IntegrityResult(entry, Verdict.EMPTY, "zero bytes")

    path = entry.path
    ext = entry.suffix

    # Word/Excel write a "~$name.docx" owner-lock stub next to an open document.
    # It carries the document's extension but is never an OOXML file, so the
    # signature check reports it as corrupt. It is a normal, transient artifact.
    if path.name.startswith("~$"):
        return IntegrityResult(
            entry, Verdict.UNVERIFIABLE, "Office lock file, not a document"
        )

    try:
        with path.open("rb") as fh:
            head = fh.read(64)
    except OSError as exc:
        return IntegrityResult(entry, Verdict.UNREADABLE, exc.strerror or str(exc))

    family = _sniff(head)
    expected = _EXPECTED.get(ext)

    # A known extension whose bytes say otherwise.
    if expected and family and family not in expected:
        return IntegrityResult(
            entry,
            Verdict.MISLABELLED,
            f"extension says {ext}, contents look like {family}",
        )
    if expected and family is None:
        return IntegrityResult(
            entry, Verdict.CORRUPT, f"missing {ext} file signature"
        )

    if ext in {".docx", ".xlsx", ".pptx", ".zip", ".jar", ".epub"}:
        return _check_zip(entry)
    if ext == ".pdf":
        return _check_pdf(entry)
    if ext == ".png":
        return _check_png(entry)
    if ext in {".jpg", ".jpeg"}:
        return _check_jpeg(entry)
    if ext == ".gif":
        return _check_gif(entry)
    if ext == ".json":
        return _check_json(entry)
    if ext in {".gz", ".tgz"}:
        return _check_gzip(entry)

    if ext in _DEEP_CHECKED:
        return IntegrityResult(entry, Verdict.OK, "signature valid")

    return IntegrityResult(entry, Verdict.UNVERIFIABLE, "no verifier for this format")


def _check_zip(entry: FileEntry) -> IntegrityResult:
    """OOXML documents are ZIP containers — a truncated download fails here."""
    try:
        with zipfile.ZipFile(entry.path) as zf:
            broken = zf.testzip()
            if broken is not None:
                return IntegrityResult(entry, Verdict.CORRUPT, f"bad CRC in {broken}")
            names = zf.namelist()
            if not names:
                return IntegrityResult(entry, Verdict.CORRUPT, "empty archive")
            if entry.suffix in {".docx", ".xlsx", ".pptx"} and "[Content_Types].xml" not in names:
                return IntegrityResult(
                    entry, Verdict.CORRUPT, "OOXML container missing [Content_Types].xml"
                )
        return IntegrityResult(entry, Verdict.OK, "archive verified")
    except zipfile.BadZipFile as exc:
        return IntegrityResult(entry, Verdict.CORRUPT, f"bad archive: {exc}")
    except OSError as exc:
        return IntegrityResult(entry, Verdict.UNREADABLE, exc.strerror or str(exc))


def _check_pdf(entry: FileEntry) -> IntegrityResult:
    """Header, EOF marker, then a real parse of the cross-reference table."""
    try:
        with entry.path.open("rb") as fh:
            fh.seek(max(0, entry.size - 2048))
            tail = fh.read()
        if b"%%EOF" not in tail:
            return IntegrityResult(
                entry, Verdict.CORRUPT, "no %%EOF marker — likely truncated"
            )
    except OSError as exc:
        return IntegrityResult(entry, Verdict.UNREADABLE, exc.strerror or str(exc))

    try:
        from pypdf import PdfReader
        from pypdf.errors import PdfReadError

        try:
            with _quiet_pypdf():
                reader = PdfReader(str(entry.path), strict=False)
                # A password-protected PDF is intact, just closed to us. Calling
                # it corrupt would be a lie that could get a good file deleted.
                if getattr(reader, "is_encrypted", False):
                    try:
                        if reader.decrypt("") == 0:  # empty user password
                            return IntegrityResult(
                                entry,
                                Verdict.UNVERIFIABLE,
                                "encrypted (password-protected); contents not checked",
                            )
                    except Exception:
                        return IntegrityResult(
                            entry,
                            Verdict.UNVERIFIABLE,
                            "encrypted (password-protected); contents not checked",
                        )
                pages = len(reader.pages)
        except PdfReadError as exc:
            if "decrypt" in str(exc).lower():
                return IntegrityResult(
                    entry,
                    Verdict.UNVERIFIABLE,
                    "encrypted (password-protected); contents not checked",
                )
            return IntegrityResult(entry, Verdict.CORRUPT, f"unparseable: {exc}")
        except Exception as exc:
            return IntegrityResult(entry, Verdict.CORRUPT, f"unparseable: {exc}")
        if pages == 0:
            return IntegrityResult(entry, Verdict.CORRUPT, "no pages")
        return IntegrityResult(entry, Verdict.OK, f"{pages} page(s)")
    except ImportError:
        return IntegrityResult(entry, Verdict.OK, "header and EOF marker present")


# Cameras and phones routinely append data *after* an image's end marker:
# Samsung writes a "SEFT" metadata trailer, and motion photos embed a whole MP4.
# The first version of this check only looked at the final 32 bytes and so
# declared 21 perfectly good photos truncated. Never assume the end marker is
# the last thing in the file.
_TRAILER_WINDOW = 1024 * 1024      # first look: covers metadata trailers
_TRAILER_WINDOW_MAX = 32 * 1024 * 1024  # escalation: covers embedded video


def _find_end_marker(entry: FileEntry, marker: bytes) -> bool | None:
    """Is `marker` present near the end of the file? None if unreadable.

    Searches a window back from EOF, widening once before giving up, so trailing
    metadata or an appended video doesn't read as truncation.
    """
    for window in (_TRAILER_WINDOW, _TRAILER_WINDOW_MAX):
        span = min(entry.size, window)
        try:
            with entry.path.open("rb") as fh:
                fh.seek(entry.size - span)
                if marker in fh.read(span):
                    return True
        except OSError:
            return None
        if span == entry.size:
            break  # whole file already searched; widening cannot help
    return False


def _decode_image(path: Path, tolerant: bool) -> tuple[bool, str]:
    """Try to decode an image. Returns (succeeded, message)."""
    from PIL import Image, ImageFile

    previous = ImageFile.LOAD_TRUNCATED_IMAGES
    ImageFile.LOAD_TRUNCATED_IMAGES = tolerant
    try:
        with Image.open(path) as img:
            img.load()
        return True, ""
    except Exception as exc:
        return False, str(exc)
    finally:
        ImageFile.LOAD_TRUNCATED_IMAGES = previous


def _check_image(entry: FileEntry, marker: bytes, label: str) -> IntegrityResult:
    """Verify an image by decoding it, arbitrated by its end marker.

    One decoder's strictness is not the same thing as corruption. Pillow refuses
    some structurally complete images over minor spec deviations — a real photo
    here failed to open yet decoded to a full 682x678 bitmap once tolerance was
    enabled. Reporting that as corrupt would tell someone a working file is
    broken, which is worse than saying nothing.

    So the end marker decides whether the file is *complete on disk*, and the
    decoder decides whether the bytes are *usable*:

        strict decode works                  -> OK
        lenient works + end marker present   -> OK, with a note
        end marker missing                   -> CORRUPT (genuinely truncated)
        nothing decodes                      -> CORRUPT
    """
    try:
        import PIL  # noqa: F401
    except ImportError:
        found = _find_end_marker(entry, marker)
        if found is None:
            return IntegrityResult(entry, Verdict.UNREADABLE, "could not read file")
        if not found:
            return IntegrityResult(
                entry, Verdict.CORRUPT, f"no {label} end marker — truncated"
            )
        return IntegrityResult(entry, Verdict.OK, f"{label} structure intact")

    ok, message = _decode_image(entry.path, tolerant=False)
    if ok:
        return IntegrityResult(entry, Verdict.OK, f"{label} decodes cleanly")

    # The decoder distinguishes the two failure modes better than any byte
    # pattern can. Searching for an end marker was tried first and was wrong:
    # inside a motion photo's embedded video, a stray FFD9 made a file chopped
    # in half look complete. The decoder knows how much data it still needed.
    if "truncated" in message.lower():
        return IntegrityResult(entry, Verdict.CORRUPT, f"{label} truncated: {message}")

    tolerant_ok, tolerant_message = _decode_image(entry.path, tolerant=True)
    if tolerant_ok:
        return IntegrityResult(
            entry,
            Verdict.OK,
            f"{label} complete and viewable; strict decoder objected ({message})",
        )
    return IntegrityResult(
        entry, Verdict.CORRUPT, f"{label} will not decode: {tolerant_message}"
    )


def _check_png(entry: FileEntry) -> IntegrityResult:
    return _check_image(entry, b"IEND", "PNG")


def _check_jpeg(entry: FileEntry) -> IntegrityResult:
    return _check_image(entry, b"\xff\xd9", "JPEG")


def _check_gif(entry: FileEntry) -> IntegrityResult:
    return _check_image(entry, b"\x3b", "GIF")


def _check_json(entry: FileEntry) -> IntegrityResult:
    # Only parse small JSON; a huge one costs more than the finding is worth.
    if entry.size > 32 * 1024 * 1024:
        return IntegrityResult(entry, Verdict.UNVERIFIABLE, "too large to parse")
    try:
        text = entry.path.read_text(encoding="utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        return IntegrityResult(entry, Verdict.CORRUPT, f"invalid encoding: {exc.reason}")
    except OSError as exc:
        return IntegrityResult(entry, Verdict.UNREADABLE, exc.strerror or str(exc))
    try:
        json.loads(text)
    except json.JSONDecodeError as exc:
        return IntegrityResult(entry, Verdict.CORRUPT, f"invalid JSON at line {exc.lineno}")
    return IntegrityResult(entry, Verdict.OK, "valid JSON")


def _check_gzip(entry: FileEntry) -> IntegrityResult:
    import gzip

    if entry.size > 512 * 1024 * 1024:
        return IntegrityResult(entry, Verdict.UNVERIFIABLE, "too large to decompress")
    try:
        with gzip.open(entry.path, "rb") as fh:
            while fh.read(1024 * 1024):
                pass
    except (OSError, EOFError, gzip.BadGzipFile) as exc:
        return IntegrityResult(entry, Verdict.CORRUPT, f"bad gzip stream: {exc}")
    return IntegrityResult(entry, Verdict.OK, "decompresses cleanly")
