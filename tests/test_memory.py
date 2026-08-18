"""Tests for conversation persistence and remembered facts.

Memory is the feature most able to surprise someone unpleasantly: it keeps
personal statements on disk. So the tests cover not only that recall works, but
that nothing is stored the user did not ask for, and that forgetting is real.

Embeddings are faked with a deterministic bag-of-words vector so the suite stays
fast and does not need the ONNX model. The store only requires that similar text
produces similar vectors, which this satisfies.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.memory.service import MemoryService
from app.memory.store import MemoryStore

# Content words only. An earlier version included "is"/"my"/"user", which made
# "what is the capital of Peru?" score 0.45 against a fact about a dog purely on
# the word "is" — the fixture, not the threshold, was at fault. A real embedder
# gives stopwords almost no weight, so the fake must not either.
VOCAB = [
    "dog", "cat", "rex", "name", "called", "prefer", "coffee", "tea",
    "python", "rust", "deadline", "friday",
]


def fake_embed(texts: list[str]) -> np.ndarray:
    """Deterministic bag-of-words vectors — similar text, similar direction."""
    rows = []
    for text in texts:
        lowered = text.lower()
        vec = np.array([lowered.count(word) for word in VOCAB], dtype=np.float32)
        norm = np.linalg.norm(vec)
        rows.append(vec / norm if norm else vec)
    return np.vstack(rows)


@pytest.fixture
def service(tmp_path) -> MemoryService:
    store = MemoryStore(tmp_path / "memory.sqlite3")
    return MemoryService(embed_fn=fake_embed, store=store, min_score=0.3)


# ── conversation persistence ─────────────────────────────────────
def test_a_new_session_starts_empty(service):
    """Waking up no longer replays yesterday. Resuming silently made the
    assistant answer from the old transcript: asked how much memory was in use,
    it repeated a whole machine report from an earlier conversation."""
    service.start_session()
    service.record("user", "how much memory is in use?")

    service.start_session()  # a second wake, no resume
    assert service.history() == []


def test_previous_messages_reaches_past_the_session_in_progress(service):
    """/resume runs *after* a fresh session exists, so `history` would return
    the empty one just created. It has to look one further back."""
    service.start_session()
    service.record("user", "remind me about the deadline")
    service.record("assistant", "Friday")

    service.start_session()
    earlier = service.previous_messages()
    assert [m.content for m in earlier] == ["remind me about the deadline", "Friday"]


def test_previous_messages_is_empty_on_a_first_ever_run(service):
    service.start_session()
    assert service.previous_messages() == []


def test_conversation_survives_a_new_service(tmp_path):
    """The whole point: closing the terminal must not discard the transcript."""
    path = tmp_path / "memory.sqlite3"

    first = MemoryService(embed_fn=fake_embed, store=MemoryStore(path))
    session_id = first.start_session()
    first.record("user", "hello there")
    first.record("assistant", "hi")
    first.close()

    second = MemoryService(embed_fn=fake_embed, store=MemoryStore(path))
    resumed = second.start_session(resume=True)
    assert resumed == session_id
    assert [m.content for m in second.history()] == ["hello there", "hi"]


def test_resume_picks_the_most_recent_conversation(service):
    service.start_session()
    service.record("user", "first conversation")
    second_id = service.start_session()
    service.record("user", "second conversation")

    assert service.start_session(resume=True) == second_id


def test_starting_fresh_does_not_resume(service):
    first = service.start_session()
    service.record("user", "hello")
    assert service.start_session() != first
    assert service.history() == []


def test_first_user_message_titles_the_session(service):
    service.start_session()
    service.record("user", "how do I renew my passport")
    service.record("assistant", "…")
    service.record("user", "and the fee?")

    assert service.sessions()[0].title == "how do I renew my passport"


def test_recording_without_a_session_is_a_no_op(service):
    """A one-shot `ask` must not litter the store with empty sessions."""
    service.record("user", "hello")
    assert service.sessions() == []


def test_history_limit_returns_the_newest_in_order(service):
    service.start_session()
    for i in range(6):
        service.record("user", f"message {i}")

    recent = service.history(limit=3)
    assert [m.content for m in recent] == ["message 3", "message 4", "message 5"]


# ── remembering ──────────────────────────────────────────────────
def test_remember_then_recall(service):
    service.remember("The user's dog is called Rex")
    recall = service.recall("what is my dog called?")

    assert recall
    assert "Rex" in recall.memories[0].text


def test_unrelated_questions_recall_nothing(service):
    """Below the relevance floor nothing is injected. Padding a 3B model's
    context with unrelated personal trivia makes answers worse, not better."""
    service.remember("The user's dog is called Rex")
    assert not service.recall("what is the capital of Peru?")


def test_recall_is_empty_when_nothing_is_stored(service):
    assert not service.recall("anything at all")


def test_recall_never_embeds_when_the_store_is_empty(tmp_path):
    """Rule 9: do not load the embedding model to discover there is nothing."""
    calls: list[list[str]] = []

    def counting_embed(texts):
        calls.append(texts)
        return fake_embed(texts)

    service = MemoryService(
        embed_fn=counting_embed, store=MemoryStore(tmp_path / "m.sqlite3")
    )
    service.recall("what is my name?")
    assert calls == []


def test_duplicate_facts_are_not_stored_twice(service):
    first, _ = service.remember("The user prefers coffee")
    second, outcome = service.remember("the user prefers COFFEE  ")

    assert second is not None and first is not None
    assert second.id == first.id
    assert outcome == "already remembered"
    assert len(service.memories()) == 1


def test_blank_facts_are_rejected(service):
    memory, outcome = service.remember("   ")
    assert memory is None
    assert outcome == "nothing to remember"


def test_long_facts_are_truncated(service):
    memory, _ = service.remember("x " * 600)
    assert memory is not None
    assert len(memory.text) <= 501


def test_a_fact_stored_without_embeddings_is_still_kept(tmp_path):
    """Rule 10: a broken embedder loses recall, not the user's data."""

    def broken_embed(texts):
        raise RuntimeError("model unavailable")

    service = MemoryService(
        embed_fn=broken_embed, store=MemoryStore(tmp_path / "m.sqlite3")
    )
    memory, outcome = service.remember("The user's cat is called Mo")

    assert memory is not None
    assert outcome == "remembered"
    assert len(service.memories()) == 1


def test_recall_records_usage(service):
    service.remember("The user prefers tea")
    service.recall("do I prefer tea or coffee?")

    assert service.memories()[0].used_count == 1


def test_most_relevant_memory_ranks_first(service):
    service.remember("The user prefers rust")
    service.remember("The user's dog is called Rex")

    recall = service.recall("what is my dog called")
    assert "Rex" in recall.memories[0].text


# ── forgetting ───────────────────────────────────────────────────
def test_forget_removes_a_fact_and_its_recall(service):
    memory, _ = service.remember("The user's dog is called Rex")
    assert service.recall("what is my dog called")

    assert service.forget(memory.id) is True
    assert service.memories() == []
    assert not service.recall("what is my dog called")


def test_forget_all_clears_everything(service):
    service.remember("The user prefers coffee")
    service.remember("The user's dog is called Rex")

    assert service.forget_all() == 2
    assert service.memories() == []


def test_forgetting_an_unknown_id_reports_false(service):
    assert service.forget(999) is False


def test_forgetting_survives_a_restart(tmp_path):
    """A user who says 'forget that' must find it gone next time, not cached."""
    path = tmp_path / "memory.sqlite3"
    first = MemoryService(embed_fn=fake_embed, store=MemoryStore(path))
    memory, _ = first.remember("The user's dog is called Rex")
    first.forget(memory.id)
    first.close()

    second = MemoryService(embed_fn=fake_embed, store=MemoryStore(path))
    assert second.memories() == []
    assert not second.recall("what is my dog called")


# ── prompt shape ─────────────────────────────────────────────────
def test_recall_prompt_lists_the_facts(service):
    service.remember("The user's dog is called Rex")
    prompt = service.recall("what is my dog called").as_prompt()

    assert "Rex" in prompt
    # The model must be told not to blurt memories out unprompted.
    assert "only if they are relevant" in prompt
