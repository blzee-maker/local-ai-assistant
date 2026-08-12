"""Structure-aware text chunking.

Naive fixed-width chunking cuts words and, worse, separates a heading from the
text under it — which wrecks retrieval, since headings carry a lot of meaning.
This chunker instead:

  1. splits the document into sections at Markdown headings,
  2. keeps each section's heading attached to its body (prefixed onto every
     chunk of that section, so the heading's meaning is never lost),
  3. packs whole paragraphs up to the size budget, only hard-splitting a single
     paragraph that is itself larger than the budget.

Plain text with no headings still benefits: it packs on paragraph boundaries
instead of slicing mid-word.
"""
from __future__ import annotations

import re

_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+.*$", re.MULTILINE)


def _hard_split(text: str, size: int, overlap: int) -> list[str]:
    """Last-resort character split for a paragraph bigger than the budget."""
    size = max(size, 100)
    chunks: list[str] = []
    start, n = 0, len(text)
    while start < n:
        end = min(start + size, n)
        if end < n:
            window = text[start:end]
            for sep in (". ", " "):
                idx = window.rfind(sep)
                if idx != -1 and idx > size * 0.5:
                    end = start + idx + len(sep)
                    break
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return chunks


def _split_sections(text: str) -> list[tuple[str, str]]:
    """Return [(heading, body)] pairs. heading is '' for pre-heading text."""
    matches = list(_HEADING.finditer(text))
    if not matches:
        return [("", text.strip())]

    sections: list[tuple[str, str]] = []
    if matches[0].start() > 0:
        pre = text[: matches[0].start()].strip()
        if pre:
            sections.append(("", pre))

    for i, m in enumerate(matches):
        heading = m.group().strip().lstrip("#").strip()
        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[body_start:body_end].strip()
        sections.append((heading, body))
    return sections


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    text = text.strip()
    if not text:
        return []

    chunks: list[str] = []
    for heading, body in _split_sections(text):
        prefix = f"{heading}\n" if heading else ""
        full = (prefix + body).strip()

        # Whole section fits — keep it intact (heading + body together).
        if len(full) <= size:
            if full:
                chunks.append(full)
            continue

        # Otherwise pack paragraphs, always re-attaching the heading.
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
        buf = ""
        for para in paragraphs:
            candidate = f"{buf}\n\n{para}".strip() if buf else para
            if len(prefix + candidate) <= size:
                buf = candidate
                continue
            if buf:
                chunks.append((prefix + buf).strip())
                buf = ""
            if len(prefix + para) <= size:
                buf = para
            else:  # single oversized paragraph
                for piece in _hard_split(para, size - len(prefix), overlap):
                    chunks.append((prefix + piece).strip())
        if buf:
            chunks.append((prefix + buf).strip())

    return [c for c in chunks if c]
