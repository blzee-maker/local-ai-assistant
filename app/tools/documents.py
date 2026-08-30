"""Letting the user's own documents contest a question with the machine tools.

Every generation failure in the eval suite was this, and none of them was a
generation failure. Grounding is exclusive — `chat_stream` retrieves only when
no tool claimed the turn — and tool selection ran with no idea a document store
existed. So the model was handed a menu of machine readers and asked which one
answers "How long is the hardware warranty on the NimbusEdge X1?". It picked
`system_status`, whose triggers include "hardware"; retrieval never ran; and the
model, shown this laptop's CPU and battery, said it could not tell you the
warranty. That answer was *correct for the question it was actually asked*. The
harness scored it as a refusal and sent three sessions hunting a prompt bug.

Four cases were stolen this way, which was every failure in the suite:

    warranty-length      "hardware"  in "hardware warranty"
    peak-power           "power"     in "peak power draw"
    warranty-paraphrase  "hardware"  in "if the hardware breaks"
    founders             "system"    in "Halden Systems"

`absent-ceo` was stolen too and *passed*, because refusing to name the CEO of
this laptop looks identical to correctly abstaining about the document. A case
passing for the wrong reason is the same defect wearing a green tick.

The fix is not a better keyword list. "What is the peak power draw?" is genuinely
ambiguous — a laptop has one and so does the appliance in the handbook — and no
prefilter can resolve it, because the thing that resolves it is knowing the user
ingested a document that answers it. Nor is it a similarity threshold: measured
against the real embedder on the golden dataset, the two populations overlap.

    document questions        machine questions
    0.752 warranty-length     0.489 "give me my system information"
    0.477 founders            0.364 "what is my CPU usage right now?"
    0.410 absent-ceo          0.312 "how much of my memory is in use?"
    0.396 warranty-paraphrase 0.225 "what is running on my laptop?"
    0.350 peak-power          0.184 "how much free space do I have?"

`peak-power` at 0.350 sits *below* a plain CPU question at 0.364, so any cut
that saves peak-power steals ordinary machine questions (rule 6, and the
existing rule that thresholds are measured rather than guessed).

So the documents go on the menu instead, and the model chooses between reading
the machine and reading what the user gave it. That is a decision it can
actually make, unlike the one it was being asked to make before.
"""
from __future__ import annotations

from app.tools.base import Risk, Tool, ToolContext, ToolResult


class SearchDocumentsTool:
    """A marker that wins the selection and then gets out of the way.

    It deliberately returns no content. `Dispatch.grounded` is false without
    content, so `chat_stream` falls through to the retrieval path it already
    had — same chunks, same `[1] [2]` context, same inline citations. Producing
    the grounding here instead would fork RAG into two implementations, and the
    second one would be the one without citations.
    """

    name = "search_documents"
    description = "Answer from the documents the user has loaded"
    risk = Risk.READ

    # Matches everything, so it can contest any question a machine tool claims.
    # `contests_only` is what stops that from taxing ordinary chat: with nothing
    # to contest it is dropped, and "hello" never reaches tool selection.
    contests_only = True

    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    "Search the documents the user has loaded and answer from "
                    "them. Use this whenever the question is about a document, "
                    "a product, a company, a policy or a person — anything that "
                    "would be written down rather than measured from this "
                    "computer's hardware."
                ),
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        }

    def matches(self, text: str) -> bool:
        return True

    def match_score(self, text: str) -> int:
        """Never wins the backstop — only the model's explicit choice.

        The backstop fires when the model emits no tool call at all, and there
        it has to separate "about this computer" from "about the handbook"
        deterministically. Three signals were tried against both populations and
        none of them separates. They are recorded here so nobody re-walks them:

        **Retrieval similarity.** The document questions score 0.752 down to
        0.350; the machine questions score 0.489 down to 0.184. `peak-power`
        (0.350) sits *below* "what is my CPU usage right now?" (0.364), so any
        cut that rescues it takes ordinary machine questions with it.

        **A reference to this machine** ("my", "this laptop", "do I have").
        Cleaner, and still wrong in both directions: "if the hardware breaks, how
        long am I covered for?" reads as deictic on "am I", while "what is using
        all the RAM?", "is the computer running hot?" and "show me the disk
        usage" carry no possessive at all and are plainly about the machine.

        **Trigger count**, the thing the backstop already ranks by. Both
        populations sit at 1: "hardware" in "hardware warranty" scores exactly
        what "memory" in "how much of my memory is in use?" does.

        So the backstop keeps the behaviour it had, which is at least measured,
        and this tool stays out of it at zero. That leaves one narrow gap — a
        document question whose model call emits nothing, which is likeliest
        under the memory pressure that makes the model drop tool calls in the
        first place. Closing it by guessing would trade a rare wrong answer for
        a common one.
        """
        return 0

    def run(self, arguments: dict, context: ToolContext) -> ToolResult:
        return ToolResult(ok=True, content="", display="answering from your documents")
