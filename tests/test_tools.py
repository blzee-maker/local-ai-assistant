"""Tests for the tool registry, permissions, and audit trail.

The failure mode here is silence. A gate collision does not raise — the wrong
capability simply wins, and the user gets a plausible answer from the wrong
source. So these tests assert on *which* tool fired, not merely that something
did (rule 12).
"""
from __future__ import annotations

import pytest

from app.engines.base import AssistantTurn, ToolCall
from app.tools.base import Risk, Tool, ToolContext, ToolInvocation, ToolResult
from app.tools.ledger import ToolLedger
from app.tools.registry import ToolRegistry


# ── doubles ──────────────────────────────────────────────────────
class FakeTool(Tool):
    def __init__(self, name: str, keyword: str, risk: Risk = Risk.READ):
        self.name = name
        self.description = f"fake {name}"
        self.risk = risk
        self.keyword = keyword
        self.calls: list[dict] = []

    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {"name": self.name, "description": self.description,
                         "parameters": {"type": "object", "properties": {}}},
        }

    def matches(self, text: str) -> bool:
        return self.keyword in text.lower()

    def run(self, arguments: dict, context: ToolContext) -> ToolResult:
        self.calls.append(arguments)
        return ToolResult(ok=True, content=f"{self.name} ran", display=self.name)


class ExplodingTool(FakeTool):
    def run(self, arguments: dict, context: ToolContext) -> ToolResult:
        raise RuntimeError("kaboom")


class FakeEngine:
    """Engine stub that returns a chosen tool call, or none."""

    def __init__(self, tool_name: str | None = None, raises: bool = False):
        self.tool_name = tool_name
        self.raises = raises
        self.tools_offered: list[str] = []

    def chat(self, messages, tools=None, options=None) -> AssistantTurn:
        if self.raises:
            raise RuntimeError("no tool support")
        self.tools_offered = [t["function"]["name"] for t in (tools or [])]
        if self.tool_name is None:
            return AssistantTurn(content="just chatting")
        return AssistantTurn(content="", tool_calls=[ToolCall(self.tool_name, {})])


@pytest.fixture
def ledger(tmp_path) -> ToolLedger:
    return ToolLedger(tmp_path / "ledger.sqlite3")


def ctx(assistant=None, text="hi", confirm=None) -> ToolContext:
    return ToolContext(assistant=assistant, request_text=text, confirm=confirm)


# ── registration ─────────────────────────────────────────────────
def test_duplicate_registration_is_rejected(ledger):
    registry = ToolRegistry(ledger)
    registry.register(FakeTool("a", "alpha"))
    with pytest.raises(ValueError):
        registry.register(FakeTool("a", "beta"))


# ── selection ────────────────────────────────────────────────────
def test_plain_chat_costs_no_tool_call(ledger):
    """Rule 9: ordinary conversation must not pay for tool selection."""
    registry = ToolRegistry(ledger)
    registry.register(FakeTool("files", "file"))
    engine = FakeEngine("files")

    assert registry.select("what is the capital of France?", engine) is None
    assert engine.tools_offered == []  # the model was never consulted


def test_only_matching_tools_are_offered_to_the_model(ledger):
    registry = ToolRegistry(ledger)
    registry.register_all([FakeTool("files", "file"), FakeTool("disk", "duplicate")])
    engine = FakeEngine("files")

    registry.select("open my file", engine)
    assert engine.tools_offered == ["files"]


def test_model_choice_wins_when_several_tools_match(ledger):
    """The collision case, resolved by the model rather than by `if` order."""
    registry = ToolRegistry(ledger)
    registry.register_all([FakeTool("files", "find"), FakeTool("disk", "find")])
    engine = FakeEngine("disk")

    invocation = registry.select("find my duplicate files", engine)
    assert invocation is not None
    assert invocation.tool == "disk"
    assert invocation.source == "model"
    assert sorted(engine.tools_offered) == ["disk", "files"]


def test_backstop_fires_when_the_model_picks_nothing(ledger):
    registry = ToolRegistry(ledger)
    registry.register(FakeTool("files", "file"))

    invocation = registry.select("open my file", FakeEngine(None))
    assert invocation is not None
    assert invocation.tool == "files"
    assert invocation.source == "backstop"


def test_backstop_fires_when_tool_calling_is_unsupported(ledger):
    registry = ToolRegistry(ledger)
    registry.register(FakeTool("files", "file"))

    invocation = registry.select("open my file", FakeEngine(raises=True))
    assert invocation is not None
    assert invocation.source == "backstop"


def test_backstop_picks_the_best_match_among_read_tools(ledger):
    """Declining whenever two tools matched sounded prudent and was wrong.

    Asked "give me my system information: CPU, memory and disks", both the
    system and disk reporters matched, the 3B model chose neither, and the user
    got no data at all — which is how a question about their hardware ended up
    answered from imagination. For read-only tools, running the better match
    costs a cheap local call; running nothing costs the answer.
    """
    class ScoredTool(FakeTool):
        def __init__(self, name, keyword, score):
            super().__init__(name, keyword)
            self._score = score

        def match_score(self, text: str) -> int:
            return self._score

    registry = ToolRegistry(ledger)
    registry.register_all([ScoredTool("weak", "find", 1), ScoredTool("strong", "find", 4)])

    invocation = registry.select("find something", FakeEngine(None))
    assert invocation is not None
    assert invocation.tool == "strong"
    assert invocation.source == "backstop"


def test_backstop_is_deterministic_on_a_tie(ledger):
    """Equal scores must not resolve differently run to run."""
    registry = ToolRegistry(ledger)
    registry.register_all([FakeTool("first", "find"), FakeTool("second", "find")])

    picks = {
        registry.select("find something", FakeEngine(None)).tool for _ in range(5)
    }
    assert picks == {"first"}


def test_excluded_tools_are_not_offered(ledger):
    registry = ToolRegistry(ledger)
    registry.register_all([FakeTool("files", "find"), FakeTool("disk", "find")])
    engine = FakeEngine("disk")

    registry.select("find something", engine, exclude={"files"})
    assert engine.tools_offered == ["disk"]


def test_disabling_file_access_does_not_mute_other_capabilities():
    """Regression: `--no-files` once gated the whole registry, so "why is my
    laptop slow?" fell back to generic advice instead of reading the machine.
    That flag is about file access, not about silencing everything."""
    from app.core.assistant import FILE_ACCESS_TOOLS
    from app.tools import default_tools

    names = {tool.name for tool in default_tools()}
    assert FILE_ACCESS_TOOLS == {"open_local_file"}
    # Everything else must survive the flag.
    assert names - FILE_ACCESS_TOOLS, "no capabilities left after excluding file tools"
    assert "system_status" not in FILE_ACCESS_TOOLS


def test_backstop_never_fires_for_a_destructive_tool(ledger):
    """Rule 6: a keyword match is enough to guess 'read a file'. It is not
    enough to decide something gets destroyed."""
    registry = ToolRegistry(ledger)
    registry.register(FakeTool("wipe", "delete", risk=Risk.DESTRUCTIVE))

    assert registry.select("delete stuff", FakeEngine(None)) is None


# ── consent ──────────────────────────────────────────────────────
def test_read_tools_need_no_separate_consent(ledger):
    registry = ToolRegistry(ledger)
    tool = FakeTool("reader", "read")
    registry.register(tool)

    result = registry.invoke(ToolInvocation("reader", {}, "model"), ctx())
    assert result.ok
    assert tool.calls == [{}]


def test_write_tool_asks_once_then_remembers(ledger):
    registry = ToolRegistry(ledger)
    tool = FakeTool("writer", "write", risk=Risk.WRITE)
    registry.register(tool)

    asked: list[str] = []

    def confirm(prompt: str) -> bool:
        asked.append(prompt)
        return True

    registry.invoke(ToolInvocation("writer", {}, "model"), ctx(confirm=confirm))
    registry.invoke(ToolInvocation("writer", {}, "model"), ctx(confirm=confirm))

    assert len(asked) == 1, "consent should be recorded, not re-asked"
    assert len(tool.calls) == 2


def test_declining_is_remembered_and_not_re_asked(ledger):
    registry = ToolRegistry(ledger)
    tool = FakeTool("writer", "write", risk=Risk.WRITE)
    registry.register(tool)

    asked: list[str] = []

    def refuse(prompt: str) -> bool:
        asked.append(prompt)
        return False

    first = registry.invoke(ToolInvocation("writer", {}, "model"), ctx(confirm=refuse))
    second = registry.invoke(ToolInvocation("writer", {}, "model"), ctx(confirm=refuse))

    assert not first.ok and not second.ok
    assert len(asked) == 1
    assert tool.calls == []


def test_destructive_tool_confirms_every_single_time(ledger):
    """A one-time grant is not enough when each call can lose different data."""
    registry = ToolRegistry(ledger)
    tool = FakeTool("wipe", "delete", risk=Risk.DESTRUCTIVE)
    registry.register(tool)

    asked: list[str] = []

    def confirm(prompt: str) -> bool:
        asked.append(prompt)
        return True

    registry.invoke(ToolInvocation("wipe", {}, "model"), ctx(confirm=confirm))
    registry.invoke(ToolInvocation("wipe", {}, "model"), ctx(confirm=confirm))

    # Exactly one prompt per call, and no standing grant in between.
    assert len(asked) == 2
    assert len(tool.calls) == 2


def test_destructive_tools_have_no_standing_grant(ledger):
    """A persistent "yes, you may destroy things" is the one permission that
    should not be storable. Every call is confirmed on its own merits."""
    registry = ToolRegistry(ledger)
    registry.register(FakeTool("wipe", "delete", risk=Risk.DESTRUCTIVE))

    registry.invoke(ToolInvocation("wipe", {}, "model"), ctx(confirm=lambda _p: True))

    assert ledger.decision("wipe") is None
    assert not Risk.DESTRUCTIVE.needs_consent
    assert Risk.DESTRUCTIVE.needs_confirmation


def test_no_confirmer_means_no(ledger):
    """A script or the future daemon must not authorise by default just because
    nobody was present to object."""
    registry = ToolRegistry(ledger)
    tool = FakeTool("writer", "write", risk=Risk.WRITE)
    registry.register(tool)

    result = registry.invoke(ToolInvocation("writer", {}, "model"), ctx(confirm=None))
    assert not result.ok
    assert tool.calls == []


# ── resilience ───────────────────────────────────────────────────
def test_a_crashing_tool_does_not_end_the_turn(ledger):
    """Rule 10: degrade, never crash."""
    registry = ToolRegistry(ledger)
    registry.register(ExplodingTool("boom", "boom"))

    result = registry.invoke(ToolInvocation("boom", {}, "model"), ctx())
    assert not result.ok
    assert "boom" in (result.correction or "")


def test_unknown_tool_is_reported_not_raised(ledger):
    registry = ToolRegistry(ledger)
    result = registry.invoke(ToolInvocation("ghost", {}, "model"), ctx())
    assert not result.ok


# ── audit ────────────────────────────────────────────────────────
def test_every_invocation_is_recorded(ledger):
    registry = ToolRegistry(ledger)
    registry.register(FakeTool("reader", "read"))

    registry.invoke(ToolInvocation("reader", {"q": 1}, "model"), ctx())
    history = ledger.history()

    assert len(history) == 1
    assert history[0].tool == "reader"
    assert history[0].outcome == "ok"
    assert history[0].arguments == {"q": 1}
    assert history[0].source == "model"


def test_refusals_are_audited_too(ledger):
    """A denied destructive call is exactly the event someone will look for."""
    registry = ToolRegistry(ledger)
    registry.register(FakeTool("wipe", "delete", risk=Risk.DESTRUCTIVE))

    registry.invoke(ToolInvocation("wipe", {}, "model"), ctx(confirm=lambda _p: False))
    history = ledger.history()

    assert len(history) == 1
    assert history[0].outcome == "denied"


def test_crashes_are_audited(ledger):
    registry = ToolRegistry(ledger)
    registry.register(ExplodingTool("boom", "boom"))
    registry.invoke(ToolInvocation("boom", {}, "model"), ctx())

    assert ledger.history()[0].outcome == "error"


def test_revoking_a_grant_makes_it_ask_again(ledger):
    registry = ToolRegistry(ledger)
    registry.register(FakeTool("writer", "write", risk=Risk.WRITE))

    registry.invoke(ToolInvocation("writer", {}, "model"), ctx(confirm=lambda _p: True))
    assert ledger.decision("writer") is True

    ledger.revoke("writer")
    assert ledger.decision("writer") is None


# ── dispatch ─────────────────────────────────────────────────────
def test_dispatch_grounds_the_prompt(ledger):
    registry = ToolRegistry(ledger)
    registry.register(FakeTool("files", "file"))

    dispatch = registry.dispatch("open my file", FakeEngine("files"), ctx())
    assert dispatch.grounded
    assert dispatch.result.content == "files ran"


def test_dispatch_is_empty_for_ordinary_chat(ledger):
    registry = ToolRegistry(ledger)
    registry.register(FakeTool("files", "file"))

    dispatch = registry.dispatch("hello there", FakeEngine("files"), ctx())
    assert dispatch.invocation is None
    assert not dispatch.grounded
    assert ledger.history() == []
