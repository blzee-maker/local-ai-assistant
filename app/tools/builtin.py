"""The capabilities that shipped before the registry existed, as tools.

Both were previously hardwired into `Assistant.chat_stream` with their own
keyword gate, their own LLM call, and their own grounding template. Their gates
overlapped — "find my duplicate files" matched both — and the collision was
resolved by checking disk first. That ordering was invisible, untested, and
would not survive a third capability.

Here the overlap is handled where it belongs: both tools are offered to the model
together, so it picks based on the sentence rather than on which `if` came first.
The disk tool additionally narrows its own matcher, so the two only genuinely
compete when the message really is ambiguous.
"""
from __future__ import annotations

from app.core import diskintent, fileintent
from app.tools.base import Risk, Tool, ToolContext, ToolResult


class OpenLocalFileTool(Tool):
    """Find and read a document from the user's allowlisted folders."""

    name = "open_local_file"
    description = "Read a document from the user's Downloads, Documents, or Desktop"
    risk = Risk.READ

    def schema(self) -> dict:
        return fileintent.FILE_TOOL

    def matches(self, text: str) -> bool:
        # A disk-health question is never a request to open one document, even
        # though "find my duplicate files" satisfies the file gate's grammar.
        if diskintent.looks_like_disk_question(text):
            return False
        return fileintent.looks_like_file_request(text)

    def run(self, arguments: dict, context: ToolContext) -> ToolResult:
        from app import files as filesvc

        assistant = context.assistant
        request_text = context.request_text

        name = str(arguments.get("name") or "").strip() or None
        folder = str(arguments.get("folder") or "").strip() or None

        # Backstop invocations arrive with no arguments; recover them from the
        # sentence rather than giving up.
        if not folder:
            folder = fileintent.extract_folder(request_text)
        if not name:
            name = fileintent.extract_name(request_text)
        if not name:
            return ToolResult.failure(
                "The user asked for a file but the name was unclear. Ask them for "
                "the exact file name. Do not claim you cannot access files."
            )

        match = filesvc.find_latest(name, folder)
        if not match:
            where = f" in {folder}" if folder else ""
            return ToolResult.failure(
                f"You searched the user's folders for a file matching '{name}'"
                f"{where} but found none. Tell them you couldn't find that file and "
                "ask them to check the name. Do not claim you cannot access files.",
                display=f"no file matching '{name}'",
            )

        try:
            text = assistant.rag.read_file(match["path"])
        except Exception as exc:
            return ToolResult.failure(
                f"You found '{match['name']}' but couldn't read it ({exc}). Tell "
                "the user the file could not be read.",
                display=f"could not read {match['name']}",
            )

        # Persist for follow-up RAG questions (skip if already ingested).
        if match["name"] not in {d["source"] for d in assistant.documents()}:
            try:
                assistant.rag.ingest_file(match["path"], match["name"])
            except Exception:
                pass

        from app.core.assistant import FILE_INJECTION_CHARS

        return ToolResult(
            ok=True,
            content=fileintent.FILE_GROUNDING.format(
                name=match["name"],
                root=match["root"],
                modified=match["modified"],
                text=text[:FILE_INJECTION_CHARS],
                question=request_text,
            ),
            display=(
                f"opened {match['name']} from {match['root']} "
                f"· modified {match['modified']} · {len(text):,} chars"
            ),
            meta={
                "name": match["name"],
                "root": match["root"],
                "modified": match["modified"],
                "chars": len(text),
            },
        )


class DiskReportTool(Tool):
    """Answer questions about disk usage from the last completed scan."""

    name = "disk_report"
    description = "Report duplicate files, corrupted files, and storage use from the last scan"
    risk = Risk.READ

    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    "Answer questions about the user's disk: duplicate files, "
                    "corrupted or mislabelled files, large unused files, and where "
                    "storage is going. Uses the most recent local scan."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "topic": {
                            "type": "string",
                            "enum": ["duplicates", "integrity", "unused", "usage", "all"],
                            "description": "Which part of the report the user is asking about.",
                        }
                    },
                    "required": [],
                },
            },
        }

    def matches(self, text: str) -> bool:
        return diskintent.looks_like_disk_question(text)

    def run(self, arguments: dict, context: ToolContext) -> ToolResult:
        grounded, note = diskintent.ground_prompt(context.request_text)
        if grounded is None:
            # No scan on record. Refuse honestly rather than inventing findings.
            return ToolResult.failure(note or diskintent.NO_SCAN_NOTE,
                                      display="no disk scan available")

        cached = diskintent.load_report() or {}
        age = diskintent.describe_age(cached.get("saved_at", 0.0))
        return ToolResult(
            ok=True,
            content=grounded,
            display=f"using disk scan from {age}",
            meta={"scan_age": age, "summary": cached.get("summary", "")},
        )


class RememberTool(Tool):
    """Store a fact the user explicitly asked to be kept.

    WRITE rather than READ: it puts durable, personal information on disk. That
    earns a permission prompt the first time, and an audit entry every time —
    the user should be able to answer "what does it know about me, and when did
    it decide to keep that?" without guessing.
    """

    name = "remember"
    description = "Store something the user explicitly asked you to remember"
    risk = Risk.WRITE

    # Only phrasings that *ask* for storage. "I remember that film" is not a
    # request to remember anything, so the verb alone is not enough.
    _TRIGGERS = (
        "remember that", "remember this", "remember my", "remember i",
        "don't forget", "dont forget", "keep in mind", "make a note",
        "note that", "take note", "store this", "save this",
        "from now on", "for future reference",
    )

    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    "Store a fact for future conversations. Use only when the user "
                    "explicitly asks you to remember, note, or not forget something."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "fact": {
                            "type": "string",
                            "description": (
                                "The fact to store, rewritten as a standalone "
                                "statement, e.g. 'The user's dog is called Rex'."
                            ),
                        }
                    },
                    "required": ["fact"],
                },
            },
        }

    def matches(self, text: str) -> bool:
        lowered = text.lower()
        return any(trigger in lowered for trigger in self._TRIGGERS)

    def run(self, arguments: dict, context: ToolContext) -> ToolResult:
        fact = str(arguments.get("fact") or "").strip()
        if not fact:
            # Backstop invocations carry no arguments; keep the sentence rather
            # than storing nothing, since the user did ask for something.
            fact = context.request_text.strip()
        if not fact:
            return ToolResult.failure(
                "The user asked you to remember something but did not say what. "
                "Ask them what to remember."
            )

        memory, outcome = context.assistant.memory.remember(fact)
        if memory is None:
            return ToolResult.failure(
                "Nothing could be stored. Tell the user plainly.",
                display="nothing to remember",
            )

        return ToolResult(
            ok=True,
            content=(
                f"You have stored this for future conversations: \"{memory.text}\"\n"
                "Confirm briefly to the user that you will remember it. Do not "
                "invent any other details.\n\n"
                f"User's message: {context.request_text}"
            ),
            display=f"{outcome}: {memory.text}",
            meta={"memory_id": memory.id, "text": memory.text, "outcome": outcome},
        )


def default_tools() -> list[Tool]:
    """The tools registered for a normal session."""
    return [OpenLocalFileTool(), DiskReportTool(), RememberTool()]
