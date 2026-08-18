"""The Tool contract — what a capability must declare to be callable.

Before this existed, each capability grew its own path through the turn loop: a
keyword gate, a bespoke LLM call, and a hand-written grounding template. Two
capabilities already collided ("find my duplicate files" matched both the file
gate and the disk gate) and were separated by *ordering*, which does not survive
a third. Every new feature also added a term to every sibling branch, so the
coupling grew quadratically.

A tool declares four things:

* **schema** — what the model sees, so one selection call covers everything
* **risk** — what it may do to the machine, which drives consent and audit
* **backstop** — a deterministic matcher, because a 3B model mis-selecting a
  destructive tool is exactly the asymmetric case rule 6 warns about
* **run** — the handler, which returns text for grounding, never raw side effects

Rules 2 and 3 are enforced structurally here rather than left to each tool's
good intentions: a tool cannot mutate anything without declaring a risk level
that forces consent and an audit entry.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Protocol, runtime_checkable


class Risk(str, Enum):
    """What a tool is permitted to do. Drives consent and confirmation.

    The split is by *consequence*, not by subsystem: what matters to the user is
    whether something can change or destroy their data, not which module it
    lives in.
    """

    READ = "read"            # observes only; no lasting change
    WRITE = "write"          # creates or modifies something the user will see
    DESTRUCTIVE = "destructive"  # can lose data; always confirmed, never implicit

    @property
    def needs_consent(self) -> bool:
        """A standing, remembered grant. WRITE only.

        READ inherits the folder consent already granted for scanning.
        DESTRUCTIVE deliberately has *no* standing grant: a persistent
        "yes, you may destroy things" is precisely the permission that should
        not exist, and asking for one produced a worse experience too — the
        standing prompt can only say "Allow 'end_process'?", so a user was being
        asked to approve a category before ever learning which program was
        about to be closed. Per-invocation confirmation replaces it entirely.
        """
        return self is Risk.WRITE

    @property
    def needs_confirmation(self) -> bool:
        """Per-invocation confirmation, naming the specific target."""
        return self is Risk.DESTRUCTIVE


@dataclass
class ToolResult:
    """What a tool hands back.

    `content` is grounding text spliced into the prompt — tools inform the
    model's answer rather than replacing it. `display` is what the user is shown
    about the action itself, which satisfies rule 4: an action that ran silently
    is indistinguishable from one that did not.
    """

    ok: bool
    content: str = ""
    display: str = ""
    # A note injected as a system message when a tool could not do its job, so
    # the model says "I couldn't find that" instead of "I have no file access".
    correction: str | None = None
    # A reply that is already fully determined, used verbatim with no generation.
    #
    # For an outcome like "you declined, so nothing happened" there is nothing
    # for a model to compose, and llama3.2:3b repeatedly ignored the instruction
    # not to invent a reason — answering a refused process kill with "I am
    # unable to terminate processes on your system", which is false and would
    # convince someone the capability does not exist. Where the truth is known
    # exactly, state it exactly.
    final_text: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def failure(
        cls, correction: str, display: str = "", final_text: str | None = None
    ) -> "ToolResult":
        return cls(
            ok=False, correction=correction, display=display, final_text=final_text
        )


@runtime_checkable
class Tool(Protocol):
    """Contract every capability satisfies."""

    name: str
    description: str
    risk: Risk

    def schema(self) -> dict:
        """OpenAI/Ollama-style function schema shown to the model."""
        ...

    def matches(self, text: str) -> bool:
        """Cheap deterministic gate: could this message plausibly want me?

        Two jobs. First, cost — ordinary chat must not pay for tool selection
        (rule 9). Second, it is the backstop when the model fails to emit a
        tool call at all. It is deliberately *not* the authority on whether a
        destructive tool runs; see CLAUDE.md conventions.
        """
        ...

    def match_score(self, text: str) -> int:
        """How strongly this message points at me. Higher wins a tie.

        Used only when several tools match and the model declines to choose.
        Counting how much of what the user actually said belongs to this tool is
        a better tie-break than registration order: "give me my system
        information: CPU, memory and disks" matches both the system reporter and
        the disk reporter, but overwhelmingly the former.
        """
        ...

    def run(self, arguments: dict, context: "ToolContext") -> ToolResult:
        """Execute. Must not raise for expected failures — return a ToolResult
        carrying a correction so the model can be honest about what happened."""
        ...


@dataclass
class ToolContext:
    """Everything a tool is allowed to reach.

    Passed in rather than imported so tools stay testable and cannot quietly
    acquire new capabilities: whatever is not here, a tool cannot touch.
    """

    assistant: Any                      # app.core.Assistant
    request_text: str                   # the user's message, unmodified
    confirm: Callable[[str], bool] | None = None  # front end asks the user
    # Front end shows progress for work that takes minutes. Optional, because a
    # tool must behave identically with nobody watching — the daemon wires
    # neither of these, and a scan there simply runs without narration.
    progress: Callable[[str], None] | None = None

    def ask_confirmation(self, prompt: str) -> bool:
        """Destructive tools call this. No confirmer wired up means no.

        Defaulting to False matters: a non-interactive caller (a script, the
        future daemon) must not silently authorise destruction just because
        nobody was there to object.
        """
        if self.confirm is None:
            return False
        return self.confirm(prompt)

    def report_progress(self, message: str) -> None:
        """Say what a long tool is doing. Silent when nothing is listening."""
        if self.progress is None:
            return
        try:
            self.progress(message)
        except Exception:
            # A front end that fails to draw a progress line must not take the
            # scan down with it (rule 10).
            pass


@dataclass
class ToolInvocation:
    """One decision to call a tool, before it runs. Recorded either way."""

    tool: str
    arguments: dict
    source: str  # "model" | "backstop"
