"""Deciding whether a chat message is asking to open a local file.

Two paths, in order:

1. **The model decides** — native tool-calling, given the ``open_local_file``
   schema below. This is the honest path: the model sees the phrasing.
2. **Heuristic backstop** — if tool-calling is unsupported or returns nothing,
   strip the message down to its distinctive words and search on those.

A cheap keyword gate runs first so ordinary chat never pays for a tool-calling
round-trip.
"""
from __future__ import annotations

import re

FILE_TOOL_NAME = "open_local_file"

FILE_TOOL = {
    "type": "function",
    "function": {
        "name": FILE_TOOL_NAME,
        "description": (
            "Find and read a document from the user's local folders (Downloads, "
            "Documents, Desktop). Call this whenever the user asks to get, open, "
            "read, load, find, review, or summarize a file on their computer. If "
            "several files match, the most recent one is used."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Part of the filename to look for, e.g. 'resume' or 'abc'.",
                },
                "folder": {
                    "type": "string",
                    "description": "Optional folder: 'downloads', 'documents', or 'desktop'.",
                },
            },
            "required": ["name"],
        },
    },
}

FILE_DECISION_SYSTEM = (
    "You decide whether the user wants to open a local document. If they do, call "
    f"{FILE_TOOL_NAME} with the name they mentioned (and folder if stated). Do not "
    "call it for anything else."
)

FILE_GROUNDING = (
    "The user asked to work with a local file. Its contents were retrieved for you.\n"
    "File: {name} (from {root}, modified {modified})\n\n"
    "--- FILE CONTENT START ---\n{text}\n--- FILE CONTENT END ---\n\n"
    "Answer the user's request using the file above: {question}"
)

_FILE_VERBS = (
    "get", "grab", "open", "read", "load", "fetch", "find", "locate", "pull",
    "show", "review", "summarize", "summarise", "look at", "look up", "access", "pick",
)
_FILE_NOUNS = (
    "file", "document", "doc", "pdf", "docx", "resume", "cv", "download",
    "desktop", "folder", "named", "called", ".pdf", ".docx", ".txt", ".md",
)

_STOPWORDS = {
    "get", "grab", "open", "read", "load", "fetch", "find", "locate", "pull",
    "show", "review", "summarize", "summarise", "look", "up", "at", "access",
    "pick", "give", "tell", "say", "says", "the", "a", "an", "my", "me", "you",
    "your", "it", "its", "that", "this", "what", "whats", "about", "from", "in",
    "into", "on", "of", "and", "then", "please", "can", "could", "would", "will",
    "file", "files", "document", "documents", "doc", "docs", "pdf", "folder",
    "named", "called", "downloads", "download", "desktop", "content", "contents",
    "latest", "recent", "newest", "there", "is", "are", "with", "for", "to",
}

_FOLDERS = ("downloads", "documents", "desktop")


def looks_like_file_request(text: str) -> bool:
    """Cheap gate so normal chat never pays the tool-calling cost."""
    t = text.lower()
    return any(v in t for v in _FILE_VERBS) and any(n in t for n in _FILE_NOUNS)


def extract_folder(text: str) -> str | None:
    t = text.lower()
    for f in _FOLDERS:
        if f in t:
            return f
    return None


def extract_name(text: str) -> str | None:
    """Reduce the message to its distinctive words (drop verbs/articles/folders).

    Order-independent, so 'the quokka notes file' and 'file named quokka' both
    yield 'quokka notes' / 'quokka'.
    """
    quoted = re.search(r"[\"']([^\"']{2,})[\"']", text)
    if quoted:
        return quoted.group(1).strip()
    words = re.findall(r"[a-z0-9][a-z0-9._\-]*", text.lower())
    keep = [w for w in words if w not in _STOPWORDS and len(w) > 1]
    return " ".join(keep) if keep else None
