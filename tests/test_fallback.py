"""Tests for the memory safety net.

This machine has ~7.8 GB of RAM and routinely sits above 90% used, so "the main
model will not load" is an ordinary Tuesday, not an edge case. Ollama answers
that with an HTTP 5xx, and the fallback is what turns it into a slightly worse
answer instead of a dead conversation (rule 10).

Both directions, through the real path (rule 12): the fallback must engage on
the failure it exists for, and must stay out of the way of every other failure.
A retry on a connection refused would just be noise while the server is down.
"""
from __future__ import annotations

import httpx
import pytest

from app.engines import ChatMessage, GenerationOptions
from app.engines.ollama_engine import OllamaEngine

MAIN = "llama3.2:latest"
FALLBACK = "qwen2.5:1.5b"


def engine() -> OllamaEngine:
    return OllamaEngine(
        host="http://127.0.0.1:11434",
        default_model=MAIN,
        fallback_model=FALLBACK,
        timeout_s=5.0,
    )


class FakeResponse:
    """Minimal stand-in for the pieces of httpx.Response the engine touches."""

    def __init__(self, status: int, payload: dict | None = None) -> None:
        self.status_code = status
        self._payload = payload or {}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"{self.status_code}", request=None, response=self  # type: ignore[arg-type]
            )

    def json(self) -> dict:
        return self._payload


def out_of_memory_for(model: str, monkeypatch, tried: list[str]):
    """Ollama refusing to load one specific model, as it does when RAM runs out."""

    def fake_post(url: str, json: dict, timeout: float = 0.0, **_kwargs):
        tried.append(json["model"])
        if json["model"] == model:
            return FakeResponse(500)
        return FakeResponse(
            200, {"message": {"content": "answered", "tool_calls": []}}
        )

    monkeypatch.setattr(httpx, "post", fake_post)


# ── the direction that matters ───────────────────────────────────
def test_tool_selection_falls_back_when_the_main_model_will_not_load(
    monkeypatch,
):
    """Selection had no safety net while streaming did, so on a machine tight
    enough for the main model to fail, every routing decision was falling
    through to keyword scoring alone."""
    tried: list[str] = []
    out_of_memory_for(MAIN, monkeypatch, tried)

    turn = engine().chat([ChatMessage(role="user", content="hello")])

    assert tried == [MAIN, FALLBACK]
    assert turn.content == "answered"


def test_the_fallback_is_not_tried_when_the_server_is_simply_down(monkeypatch):
    """Retrying with different weights cannot fix a refused connection; it just
    doubles the wait before the user is told the truth."""
    tried: list[str] = []

    def refuse(url: str, json: dict, timeout: float = 0.0, **_kwargs):
        tried.append(json["model"])
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(httpx, "post", refuse)

    with pytest.raises(httpx.ConnectError):
        engine().chat([ChatMessage(role="user", content="hello")])

    assert tried == [MAIN]


def test_a_4xx_is_not_a_memory_problem(monkeypatch):
    """Only a 5xx looks like a failed model load. A bad request is our bug and
    must surface, not be masked by a retry."""
    tried: list[str] = []

    def bad_request(url: str, json: dict, timeout: float = 0.0, **_kwargs):
        tried.append(json["model"])
        return FakeResponse(404)

    monkeypatch.setattr(httpx, "post", bad_request)

    with pytest.raises(httpx.HTTPStatusError):
        engine().chat([ChatMessage(role="user", content="hello")])

    assert tried == [MAIN]


def test_the_original_error_is_reported_when_the_fallback_also_fails(monkeypatch):
    """Commonly the fallback was never pulled. The first error describes the
    real problem, so that is the one the user gets."""
    tried: list[str] = []

    def always_fail(url: str, json: dict, timeout: float = 0.0, **_kwargs):
        tried.append(json["model"])
        return FakeResponse(503)

    monkeypatch.setattr(httpx, "post", always_fail)

    with pytest.raises(httpx.HTTPStatusError):
        engine().chat([ChatMessage(role="user", content="hello")])

    assert tried == [MAIN, FALLBACK]


def test_a_model_never_falls_back_to_itself(monkeypatch):
    """Same model twice is a doubled wait for an identical failure."""
    tried: list[str] = []
    out_of_memory_for(MAIN, monkeypatch, tried)

    same = OllamaEngine(
        host="http://127.0.0.1:11434",
        default_model=MAIN,
        fallback_model=MAIN,
        timeout_s=5.0,
    )
    with pytest.raises(httpx.HTTPStatusError):
        same.chat([ChatMessage(role="user", content="hello")])

    assert tried == [MAIN]


# ── the fallback is a real, local model ──────────────────────────
def test_the_configured_fallback_is_not_a_cloud_model():
    """The offline guarantee is enforced where the model is resolved, and the
    fallback is a model name like any other."""
    from app.engines.policy import check_model
    from config import settings

    check_model(settings.fallback_model, allow_remote=False)


def test_the_fallback_differs_from_the_main_model():
    from config import settings

    assert settings.fallback_model != settings.default_model


def test_options_survive_the_retry(monkeypatch):
    """The fallback answers the same question, with the same settings, or it is
    not a fallback."""
    seen: list[dict] = []

    def capture(url: str, json: dict, timeout: float = 0.0, **_kwargs):
        seen.append(json)
        if json["model"] == MAIN:
            return FakeResponse(500)
        return FakeResponse(200, {"message": {"content": "ok", "tool_calls": []}})

    monkeypatch.setattr(httpx, "post", capture)

    engine().chat(
        [ChatMessage(role="user", content="a question")],
        tools=[{"type": "function", "function": {"name": "t"}}],
        options=GenerationOptions(temperature=0.0, num_ctx=8192),
    )

    assert len(seen) == 2
    first, retry = seen
    assert retry["messages"] == first["messages"]
    assert retry["tools"] == first["tools"]
    assert retry["options"] == first["options"]
