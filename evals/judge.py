"""LLM-as-judge for groundedness (a.k.a. faithfulness).

Correctness asks "is the answer right?" — a string match settles that, because
the dataset knows the gold fact. Groundedness asks a different and unanswerable-
by-string question: "is every claim in this answer actually supported by the
retrieved context, or did the model embellish?" An answer can be perfectly
correct and still ungrounded — it got the fact right from its own weights while
the retrieved chunks said nothing about it. That is a latent failure which will
bite the moment the question is about something the model *doesn't* already
know, so it is worth measuring separately.

Three deliberate constraints on the judge:

* **Temperature 0.** A judge that scores differently on reruns is worse than no
  judge, because it turns a regression into a coin flip.
* **It judges support, never truth.** The prompt forbids using outside
  knowledge. Asking a 3B model to arbitrate facts would just measure the judge's
  ignorance; asking it whether text A is entailed by text B is a much easier
  task that small models do acceptably.
* **It is the same local model.** Keeping the assistant fully offline means the
  judge is llama3.2, not GPT-4. That is a real accuracy ceiling and the report
  labels the number as advisory rather than authoritative.
"""
from __future__ import annotations

from app.engines import ChatMessage, GenerationOptions
from app.engines.base import LLMEngine

JUDGE_SYSTEM = (
    "You are a strict grading assistant. You judge only whether an ANSWER is "
    "supported by the given CONTEXT. You never use outside knowledge, and you "
    "never judge whether the answer is true in the real world — only whether "
    "the context supports it."
)

JUDGE_PROMPT = """CONTEXT:
{context}

QUESTION: {question}

ANSWER: {answer}

Is every factual claim in the ANSWER supported by the CONTEXT above?

Rules:
- If the ANSWER states a fact that does not appear in the CONTEXT, it is UNSUPPORTED.
- If the ANSWER declines to answer or says the information is not available, it is SUPPORTED.
- Ignore style, length, and helpfulness. Judge support only.

Reply with exactly one word on the first line: SUPPORTED or UNSUPPORTED.
On the second line, give a one-sentence reason."""


def judge_grounded(
    engine: LLMEngine,
    question: str,
    context: str,
    answer: str,
    model: str | None = None,
) -> tuple[bool | None, str]:
    """Return `(grounded, reason)`; `(None, reason)` if the judge call failed.

    A failed judge yields None rather than False so that infrastructure trouble
    (Ollama down mid-run) is never silently reported as a groundedness failure.
    """
    if not answer.strip():
        return None, "empty answer"

    messages = [
        ChatMessage(role="system", content=JUDGE_SYSTEM),
        ChatMessage(
            role="user",
            content=JUDGE_PROMPT.format(
                context=context or "(no context retrieved)",
                question=question,
                answer=answer,
            ),
        ),
    ]

    try:
        turn = engine.chat(
            messages,
            options=GenerationOptions(model=model, temperature=0.0, max_tokens=120),
        )
    except Exception as exc:
        return None, f"judge unavailable: {exc}"

    text = (turn.content or "").strip()
    if not text:
        return None, "judge returned nothing"

    head = text.splitlines()[0].strip().upper()
    reason = " ".join(text.splitlines()[1:]).strip()[:200]

    # Check UNSUPPORTED first — "UNSUPPORTED" contains "SUPPORTED" as a substring.
    if "UNSUPPORTED" in head or "NOT SUPPORTED" in head:
        return False, reason or text[:200]
    if "SUPPORTED" in head:
        return True, reason or text[:200]
    return None, f"unparseable verdict: {text[:120]}"
