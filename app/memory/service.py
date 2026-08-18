"""MemoryService — durable conversation and recalled facts.

**Memories are created explicitly, never inferred in the background.**

The tempting design is to run an extraction pass over every turn, asking the
model "what's worth remembering here?" It was rejected for two reasons. First
cost: that is a second generation per turn on a machine where a turn already
takes 20 seconds, which breaks rule 9 for a feature the user did not ask for.
Second, and more important, it is exactly the surprising behaviour rule 3 warns
about — silently accumulating inferred claims about someone, which they never
approved, cannot see the reasoning behind, and may simply be wrong.

So a fact is stored when the user asks for one to be stored, via a tool that
declares WRITE risk and therefore asks permission the first time. Recall is
automatic, because reading back something you explicitly asked to be kept is
what you asked for.

Recall is also gated on relevance, not just proximity: below `min_score` nothing
is injected. Padding a 3B model's context with vaguely-related personal trivia
makes answers worse, not better.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from app.memory.store import Memory, MemoryStore, SessionInfo, StoredMessage

# Cosine floor for injecting a memory — measured against the real MiniLM
# embedder, not guessed. Over a probe set of related and unrelated questions:
#
#   unrelated queries ("capital of Peru", "reverse a list", "7 times 6")
#       scored 0.05 - 0.14
#   related queries ("what is my dog called", "how do you like to answer me")
#       scored 0.34 - 0.57
#
# A first attempt at 0.45 sat inside the positive range and silently dropped a
# genuine recall. 0.28 sits in the empty band between the two clusters, with
# roughly 2x margin above the highest false match. The unit tests use a fake
# embedder and pass their own threshold, so this number is only meaningful
# against the real model.
DEFAULT_MIN_SCORE = 0.28
DEFAULT_TOP_K = 3

RECALL_PREAMBLE = (
    "Things the user previously asked you to remember. Use them only if they are "
    "relevant to the current message; do not mention them otherwise.\n{facts}"
)


@dataclass
class Recall:
    """What recall found for one turn."""

    memories: list[Memory] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.memories)

    def as_prompt(self) -> str:
        facts = "\n".join(f"- {m.text}" for m in self.memories)
        return RECALL_PREAMBLE.format(facts=facts)

    def display(self) -> str:
        count = len(self.memories)
        return f"recalled {count} memor{'y' if count == 1 else 'ies'}"


class MemoryService:
    def __init__(
        self,
        embed_fn: Callable[[list[str]], object] | None = None,
        store: MemoryStore | None = None,
        min_score: float = DEFAULT_MIN_SCORE,
        top_k: int = DEFAULT_TOP_K,
    ) -> None:
        self._store = store or MemoryStore()
        self._embed = embed_fn
        self._min_score = min_score
        self._top_k = top_k
        self._session_id: int | None = None

    @property
    def store(self) -> MemoryStore:
        return self._store

    # ── conversation ─────────────────────────────────────────────
    def start_session(self, resume: bool = False) -> int:
        """Begin or resume a conversation. Returns the session id."""
        if resume:
            existing = self._store.latest_session()
            if existing is not None:
                self._session_id = existing
                return existing
        self._session_id = self._store.create_session()
        return self._session_id

    @property
    def session_id(self) -> int | None:
        return self._session_id

    def record(self, role: str, content: str) -> None:
        """Persist one message. No-op when no session is active, so a one-shot
        `ask` doesn't accumulate junk sessions."""
        if self._session_id is None:
            return
        self._store.add_message(self._session_id, role, content)
        # The first user message doubles as the session's title; set_title only
        # writes when the session has none yet.
        if role == "user":
            self._store.set_title(self._session_id, content)

    def history(self, limit: int | None = None) -> list[StoredMessage]:
        if self._session_id is None:
            return []
        return self._store.messages(self._session_id, limit=limit)

    def previous_messages(self, limit: int = 20) -> list[StoredMessage]:
        """The conversation before this one.

        Resuming is opt-in, so by the time someone asks for it a fresh session
        is already under way and `history` would return that empty session.
        """
        earlier = self._store.latest_session(exclude=self._session_id)
        if earlier is None:
            return []
        return self._store.messages(earlier, limit=limit)

    def sessions(self, limit: int = 20) -> list[SessionInfo]:
        return self._store.sessions(limit)

    # ── facts ────────────────────────────────────────────────────
    def remember(self, text: str, source: str = "explicit") -> tuple[Memory | None, str]:
        """Store a fact. Returns (memory, human-readable outcome)."""
        text = " ".join(text.split()).strip()
        if not text:
            return None, "nothing to remember"
        if len(text) > 500:
            text = text[:500].rstrip() + "…"

        existing = self._store.find_similar_text(text)
        if existing is not None:
            return existing, "already remembered"

        embedding = None
        if self._embed is not None:
            try:
                vectors = self._embed([text])
                embedding = vectors[0]
            except Exception:
                # A memory without a vector is still worth keeping — it just
                # won't surface through semantic recall (rule 10).
                embedding = None

        memory_id = self._store.add_memory(text, source, embedding)
        stored = next(
            (m for m in self._store.memories() if m.id == memory_id), None
        )
        return stored, "remembered"

    def recall(self, query: str) -> Recall:
        """Find facts relevant to `query`. Cheap and silent when empty."""
        # Rule 9: with nothing stored, never load the embedding model just to
        # discover there is nothing to recall.
        if self._embed is None or self._store.count() == 0:
            return Recall()
        try:
            vectors = self._embed([query])
        except Exception:
            return Recall()

        hits = self._store.search(vectors[0], self._top_k, self._min_score)
        if not hits:
            return Recall()

        self._store.mark_used([m.id for m, _s in hits])
        return Recall(
            memories=[m for m, _s in hits], scores=[s for _m, s in hits]
        )

    def memories(self) -> list[Memory]:
        return self._store.memories()

    def forget(self, memory_id: int) -> bool:
        return self._store.forget(memory_id)

    def forget_all(self) -> int:
        return self._store.forget_all()

    def close(self) -> None:
        self._store.close()
