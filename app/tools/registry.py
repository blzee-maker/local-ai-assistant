"""Tool selection and dispatch — one decision point for every capability.

Replaces the chain of keyword gates that used to live in `Assistant.chat_stream`,
where each capability added a boolean to every sibling branch and collisions were
resolved by ordering. Here, candidates are gathered by cheap matchers, the model
picks from one combined schema list, and a deterministic backstop covers the case
where a 3B model emits no tool call at all.

Selection order matters and is deliberate:

1. **No candidate matches** → return immediately. Ordinary chat never pays for a
   tool-selection round trip (rule 9), which is the whole reason the cheap
   matchers exist.
2. **The model picks** from the candidates' schemas — one call, not one per
   capability.
3. **Backstop** — if the model picks nothing but exactly one candidate matched,
   use it. Ambiguity is *not* resolved by guessing: with several candidates and
   no model choice, we decline rather than pick wrong.

The backstop deliberately refuses to fire for destructive tools. A deterministic
keyword match is good enough to decide "they probably want to read a file"; it is
not good enough to decide "they want something deleted" (rule 6).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from app.engines.base import ChatMessage, GenerationOptions
from app.tools.base import Risk, Tool, ToolContext, ToolInvocation, ToolResult
from app.tools.ledger import ToolLedger

SELECTION_SYSTEM = (
    "You decide whether one of the available tools should handle the user's "
    "message. Call a tool only when it clearly applies. If plain conversation "
    "answers the message, call nothing."
)


@dataclass
class Dispatch:
    """The outcome of one turn's tool handling."""

    invocation: ToolInvocation | None = None
    result: ToolResult | None = None

    @property
    def grounded(self) -> bool:
        """True when a tool produced prompt context, so RAG should stand down."""
        return self.result is not None and self.result.ok and bool(self.result.content)

    @property
    def correction(self) -> str | None:
        return self.result.correction if self.result else None


class ToolRegistry:
    def __init__(self, ledger: ToolLedger | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        self._ledger = ledger
        self._owns_ledger = ledger is None

    # ── registration ─────────────────────────────────────────────
    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered")
        self._tools[tool.name] = tool

    def register_all(self, tools: Iterable[Tool]) -> None:
        for tool in tools:
            self.register(tool)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    @property
    def tools(self) -> list[Tool]:
        return list(self._tools.values())

    @property
    def ledger(self) -> ToolLedger:
        if self._ledger is None:
            self._ledger = ToolLedger()
        return self._ledger

    # ── selection ────────────────────────────────────────────────
    def candidates(self, text: str, exclude: set[str] | None = None) -> list[Tool]:
        """Tools whose cheap matcher thinks they might apply.

        `exclude` turns individual capabilities off for a turn — a user who
        disabled file access has disabled *that*, not the assistant's ability to
        tell them why their machine is slow.
        """
        blocked = exclude or set()
        return [
            tool
            for tool in self._tools.values()
            if tool.name not in blocked and tool.matches(text)
        ]

    def select(
        self, text: str, engine, exclude: set[str] | None = None
    ) -> ToolInvocation | None:
        """Choose a tool for `text`, or None for ordinary conversation."""
        candidates = self.candidates(text, exclude)
        if not candidates:
            return None

        by_name = {tool.name: tool for tool in candidates}

        try:
            turn = engine.chat(
                [
                    ChatMessage(role="system", content=SELECTION_SYSTEM),
                    ChatMessage(role="user", content=text),
                ],
                tools=[tool.schema() for tool in candidates],
                options=GenerationOptions(temperature=0.0),
            )
            for call in turn.tool_calls:
                if call.name in by_name:
                    return ToolInvocation(
                        tool=call.name, arguments=call.arguments or {}, source="model"
                    )
        except Exception:
            # Tool-calling unsupported, or the model is unreachable. Fall through
            # to the backstop rather than failing the whole turn.
            pass

        # Backstop: only when the match is unambiguous, and never for something
        # that could destroy data on a keyword's say-so.
        if len(candidates) == 1 and candidates[0].risk is not Risk.DESTRUCTIVE:
            return ToolInvocation(tool=candidates[0].name, arguments={}, source="backstop")
        return None

    # ── execution ────────────────────────────────────────────────
    def invoke(self, invocation: ToolInvocation, context: ToolContext) -> ToolResult:
        """Run a chosen tool, enforcing consent, confirmation, and audit."""
        tool = self._tools.get(invocation.tool)
        if tool is None:
            return ToolResult.failure(
                f"A tool named '{invocation.tool}' was requested but does not exist. "
                "Tell the user you cannot do that."
            )

        ledger = self.ledger
        risk = tool.risk.value

        # Consent — recorded once per tool, for anything that changes state.
        if tool.risk.needs_consent:
            decision = ledger.decision(tool.name)
            if decision is False:
                ledger.record(
                    tool.name, risk, invocation.source, invocation.arguments, "denied",
                    "permission previously declined",
                )
                return ToolResult.failure(
                    f"'{tool.name}' was previously declined by the user.",
                    display=f"{tool.name} is not permitted",
                    final_text=(
                        f"'{tool.name}' is turned off — you declined it "
                        f"previously. Re-enable it with `assistant tools --grant "
                        f"{tool.name}`."
                    ),
                )
            if decision is None:
                granted = context.ask_confirmation(
                    f"Allow '{tool.name}' ({risk})? {tool.description}"
                )
                ledger.record_decision(tool.name, risk, granted)
                if not granted:
                    ledger.record(
                        tool.name, risk, invocation.source, invocation.arguments,
                        "denied", "permission refused at prompt",
                    )
                    return ToolResult.failure(
                        f"The user declined permission for '{tool.name}'.",
                        display=f"{tool.name} denied",
                        final_text=(
                            "Not done — you declined permission for "
                            f"'{tool.name}'. You can allow it later with "
                            f"`assistant tools --grant {tool.name}`."
                        ),
                    )

        # Pre-flight — refuse *before* asking, when the answer is already no.
        #
        # Found by running it: asked to end csrss.exe, the assistant prompted
        # "End csrss.exe? Unsaved work will be lost", accepted the user's yes,
        # and only then refused. That prompt is a lie — the action was never
        # possible — and approving destructive prompts that get silently blocked
        # is exactly the habit not to train.
        precheck = getattr(tool, "precheck", None)
        if callable(precheck):
            try:
                refusal = precheck(invocation.arguments, context)
            except Exception:
                refusal = None
            if refusal is not None:
                ledger.record(
                    tool.name, risk, invocation.source, invocation.arguments,
                    "refused", refusal.display or refusal.correction,
                )
                return refusal

        # Confirmation — per invocation, for destructive actions only.
        if tool.risk.needs_confirmation:
            # Prefer the tool's own wording. "Run 'end_process' with {'pid':
            # 8420}?" is not something a person can evaluate; "End Spotify.exe
            # (PID 8420, 150 MB)?" is. Informed consent needs the former to
            # read like the latter.
            prompt = None
            describe = getattr(tool, "confirmation_prompt", None)
            if callable(describe):
                try:
                    prompt = describe(invocation.arguments)
                except Exception:
                    prompt = None
            if not prompt:
                prompt = (
                    f"Run '{tool.name}' with {invocation.arguments}? "
                    "This can lose data."
                )
            approved = context.ask_confirmation(prompt)
            if not approved:
                ledger.record(
                    tool.name, risk, invocation.source, invocation.arguments,
                    "denied", "confirmation refused",
                )
                return ToolResult.failure(
                    f"The user declined to confirm '{tool.name}'.",
                    display=f"{tool.name} cancelled",
                    final_text="Cancelled — nothing was changed.",
                )

        try:
            result = tool.run(invocation.arguments, context)
        except Exception as exc:
            # A crashing tool must not take the conversation with it (rule 10).
            ledger.record(
                tool.name, risk, invocation.source, invocation.arguments, "error", str(exc)
            )
            return ToolResult.failure(
                f"The tool '{tool.name}' failed ({exc}). Tell the user plainly; do "
                "not invent a result."
            )

        ledger.record(
            tool.name,
            risk,
            invocation.source,
            invocation.arguments,
            "ok" if result.ok else "failed",
            result.display or result.correction,
        )
        return result

    def dispatch(
        self,
        text: str,
        engine,
        context: ToolContext,
        exclude: set[str] | None = None,
    ) -> Dispatch:
        """Select and run in one step. Returns an empty Dispatch for plain chat."""
        invocation = self.select(text, engine, exclude)
        if invocation is None:
            return Dispatch()
        return Dispatch(invocation=invocation, result=self.invoke(invocation, context))
