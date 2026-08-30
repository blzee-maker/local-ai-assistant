"""Tests for documents and machine readings competing for the same question.

Every generation failure the eval suite had was this bug, and none of them was a
generation failure. `chat_stream` grounds on a tool *or* on retrieval, never
both, and tool selection ran with no idea a document store existed — so a
machine reader whose triggers include "hardware", "power" and "system" quietly
took four document questions and retrieval never ran for them.

The model's answers were the tell, and three sessions read past it: "I cannot
provide information about the peak power draw **of the machine**". It was not
refusing the handbook. It was correctly reporting that a CPU-and-battery reading
has no wattage in it. The harness scored that as a generation failure and sent
the investigation after the prompt, then after the model — neither of which
touches tool dispatch, which is exactly why neither changed the number.

Both directions, through the real dispatch path (rule 12). Testing the tool
alone would prove nothing: the bug lived entirely in which tool got picked.
"""
from __future__ import annotations

import pytest

from app.core import Assistant
from app.engines.base import StreamEvent
from app.tools import Risk


class NoToolCallEngine:
    """A model that never emits a tool call, which llama3.2:3b does often.

    Selection therefore falls to the deterministic backstop, so these tests
    measure the part of the decision that does not move between runs.
    """

    def health_check(self) -> bool:
        return True

    def chat(self, messages, tools=None, options=None):
        from types import SimpleNamespace

        return SimpleNamespace(tool_calls=[])

    def chat_stream(self, messages, options=None):
        yield StreamEvent(token="ok")
        yield StreamEvent(done=True, model="fake")


class Store:
    """A document index holding `count` chunks."""

    def __init__(self, count: int) -> None:
        self.count = count
        self.asked: list[str] = []

    def retrieve(self, query: str, top_k: int) -> list[dict]:
        self.asked.append(query)
        return [{"source": "handbook.md", "text": "26-month warranty", "score": 0.7}]


def assistant_with_documents(count: int = 6) -> tuple[Assistant, Store]:
    a = Assistant(engine=NoToolCallEngine())
    store = Store(count)
    a._rag = store
    return a, store


def candidate_names(assistant, text: str) -> set[str]:
    from app.core.assistant import FILE_ACCESS_TOOLS

    return {t.name for t in assistant.tools.candidates(text, set(FILE_ACCESS_TOOLS))}


# ── the tool must stand aside from what it does not hold ─────────
def test_system_status_does_not_claim_a_question_about_wattage():
    """The reading has battery percent and plugged-in state and no watts at all,
    so "peak power draw" was a question it advertised for and could not answer.
    A bare "power" trigger was the whole of that claim."""
    a, _ = assistant_with_documents()

    assert "system_status" not in candidate_names(a, "what is the peak power draw?")


@pytest.mark.parametrize(
    "question",
    [
        "is my battery charging?",
        "am I plugged in?",
        "how much of my memory is in use?",
        "why is my laptop so slow?",
        "what are my hardware specs?",
        "how much free space do I have?",
    ],
)
def test_the_machine_reader_still_claims_what_it_does_hold(question):
    """The other direction, and the one that matters more: narrowing a trigger
    list is how a read tool starts missing the questions it exists for, and a
    miss means the model answers from imagination — it once reported 16GB of RAM
    and three drives on a machine with 7.8GB and one."""
    a, _ = assistant_with_documents()

    assert "system_status" in candidate_names(a, question)


# ── documents contest, and only when there are documents ─────────
@pytest.mark.parametrize(
    "question",
    [
        "How long is the hardware warranty on the NimbusEdge X1?",
        "If the hardware breaks, how long am I covered for?",
        "Who founded Halden Systems and when?",
    ],
)
def test_documents_are_on_the_menu_when_a_machine_tool_claims_the_question(question):
    """"hardware" in "hardware warranty", "system" in "Halden Systems". The
    model was being asked which machine reader answers a question about a
    handbook, with no option to say "neither, read the documents"."""
    a, _ = assistant_with_documents()

    assert "search_documents" in candidate_names(a, question)


def test_no_documents_means_no_document_option():
    """Offering it against an empty index would let the model answer "that is in
    your documents" about a store holding none."""
    a, _ = assistant_with_documents(count=0)
    events = list(
        a.chat_stream(
            [__import__("app.engines", fromlist=["ChatMessage"]).ChatMessage(
                role="user", content="How long is the hardware warranty?")],
            use_rag=True,
        )
    )

    assert events  # the turn completed rather than raising
    from app.core.assistant import FILE_ACCESS_TOOLS

    names = {
        t.name
        for t in a.tools.candidates(
            "How long is the hardware warranty?",
            set(FILE_ACCESS_TOOLS) | {"search_documents"},
        )
    }
    assert "search_documents" not in names


# ── choosing the documents must actually run retrieval ───────────
def test_choosing_the_documents_grounds_on_retrieval_not_on_the_tool():
    """`search_documents` returns no content on purpose, so `Dispatch.grounded`
    stays false and the existing retrieval path runs — same chunks, same
    citations. A tool that produced its own grounding would be a second copy of
    RAG, and it would be the copy without inline sources."""
    from app.engines import ChatMessage
    from app.tools import ToolContext, ToolInvocation

    a, store = assistant_with_documents()
    result = a.tools.invoke(
        ToolInvocation(tool="search_documents", arguments={}, source="model"),
        ToolContext(assistant=a, request_text="q", confirm=None, progress=None),
    )

    assert result.ok is True
    assert result.content == "", "content here would fork RAG in two"

    events = list(
        a.chat_stream(
            [ChatMessage(role="user", content="what does the handbook say?")],
            use_rag=True,
        )
    )
    done = [e for e in events if e.type == "done"][-1]

    assert store.asked, "retrieval never ran"
    assert done.sources, "the answer lost its citations"


def test_the_backstop_still_prefers_the_machine_when_the_model_says_nothing():
    """Not the behaviour anyone would design, and it is here deliberately.

    When the model emits no tool call the choice falls to a keyword count, and
    three separate signals were measured against both question populations
    without separating them — see `SearchDocumentsTool.match_score`. So the
    backstop keeps what it had rather than guessing, and this test pins the
    known gap instead of pretending it is closed: a document question that gets
    no tool call from the model still loses to the machine reader.

    It fails loudly if someone later makes the documents win the backstop, which
    is the point — that change needs a measurement, not an intuition."""
    a, _ = assistant_with_documents()

    invocation = a.tools.select(
        "How long is the hardware warranty on the NimbusEdge X1?",
        NoToolCallEngine(),
        {"open_local_file"},
    )

    assert invocation is not None
    assert invocation.source == "backstop"
    assert invocation.tool == "system_status"


# ── cost: ordinary chat must not pay for this ────────────────────
@pytest.mark.parametrize("message", ["hello", "thanks!", "what is 2 + 2?", "who are you?"])
def test_ordinary_chat_never_reaches_tool_selection(message):
    """`search_documents` matches every message so it can contest any of them,
    which without `contests_only` would drag every "hello" through a
    tool-selection round trip on a machine with 8GB of RAM (rule 9)."""
    a, _ = assistant_with_documents()

    assert candidate_names(a, message) == set()


def test_the_document_tool_never_wins_the_backstop_by_itself():
    """The backstop breaks ties by counting trigger words and this tool has none
    to count. Scoring it above zero would hand it every unclaimed question by
    accident; at zero it changes an outcome only when the model deliberately
    picked the documents over the machine."""
    a, _ = assistant_with_documents()
    docs = a.tools.get("search_documents")

    assert docs.risk is Risk.READ
    assert docs.match_score("anything at all") == 0
