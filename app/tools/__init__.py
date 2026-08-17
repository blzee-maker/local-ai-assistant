"""Tool registry: one dispatch path for every capability that acts.

Anything the assistant can do to the user's machine registers here with a
declared risk level, recorded consent, and an audit entry. Never a bespoke side
path — see CLAUDE.md.
"""
from .base import Risk, Tool, ToolContext, ToolInvocation, ToolResult
from .ledger import ToolLedger
from .registry import Dispatch, ToolRegistry


def default_tools() -> list[Tool]:
    """The tools registered for a normal session.

    Imported inside the function on purpose: the builtin tools reuse the intent
    helpers in `app.core`, and `app.core.assistant` imports this package — so an
    eager `from .builtin import ...` here is a circular import. Deferring it also
    means a command that never chats never builds the tools.
    """
    from .builtin import default_tools as _builtin_default_tools

    return _builtin_default_tools()

__all__ = [
    "Risk",
    "Tool",
    "ToolContext",
    "ToolInvocation",
    "ToolResult",
    "ToolRegistry",
    "ToolLedger",
    "Dispatch",
    "default_tools",
]
