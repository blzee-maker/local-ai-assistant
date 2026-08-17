"""Tests for the offline guarantee and the memory fallback.

The offline guarantee is the project's central claim, so it gets tests that fail
loudly rather than a comment asking people to be careful. Cloud models look
exactly like local ones in `ollama list` and are selected identically — the only
thing standing between a user and a remote inference call is this check.
"""
from __future__ import annotations

import httpx
import pytest

from app.engines.base import ChatMessage, GenerationOptions
from app.engines.ollama_engine import OllamaEngine
from app.engines.policy import (
    RemoteModelBlocked,
    check_model,
    is_remote_model,
    partition_models,
)


# ── identifying hosted models ────────────────────────────────────
@pytest.mark.parametrize(
    "name",
    [
        "gpt-oss:120b-cloud",
        "deepseek-v3.1:671b-cloud",
        "qwen3-coder:480b-cloud",
        "GPT-OSS:120B-CLOUD",
    ],
)
def test_cloud_models_are_detected(name):
    assert is_remote_model(name) is True


@pytest.mark.parametrize(
    "name",
    [
        "llama3.2:latest",
        "llama3.2:1b",
        "qwen3:4b",
        "mistral",
        "",
        # A local model whose name merely contains the word cloud.
        "cloudy-llm:7b",
        "nimbus:cloud-tuned",
    ],
)
def test_local_models_are_not_flagged(name):
    assert is_remote_model(name) is False


def test_partition_splits_local_from_remote():
    local, remote = partition_models(
        ["llama3.2:latest", "gpt-oss:120b-cloud", "qwen3:4b"]
    )
    assert local == ["llama3.2:latest", "qwen3:4b"]
    assert remote == ["gpt-oss:120b-cloud"]


# ── the guard ────────────────────────────────────────────────────
def test_cloud_model_is_refused_by_default():
    with pytest.raises(RemoteModelBlocked):
        check_model("gpt-oss:120b-cloud", allow_remote=False)


def test_cloud_model_is_permitted_when_explicitly_enabled():
    check_model("gpt-oss:120b-cloud", allow_remote=True)  # must not raise


def test_local_model_always_passes():
    check_model("llama3.2:latest", allow_remote=False)


def test_engine_refuses_a_cloud_default_model():
    with pytest.raises(RemoteModelBlocked):
        OllamaEngine(host="http://localhost:11434", default_model="gpt-oss:120b-cloud")


def test_per_request_override_cannot_smuggle_in_a_cloud_model():
    """The real hole: `/model gpt-oss:120b-cloud` and `--model` bypass the
    configured default entirely."""
    engine = OllamaEngine(
        host="http://localhost:11434", default_model="llama3.2:latest"
    )
    with pytest.raises(RemoteModelBlocked):
        engine._resolve_model(GenerationOptions(model="gpt-oss:120b-cloud"))


def test_resolve_falls_back_to_the_default_model():
    engine = OllamaEngine(
        host="http://localhost:11434", default_model="llama3.2:latest"
    )
    assert engine._resolve_model(None) == "llama3.2:latest"
    assert engine._resolve_model(GenerationOptions()) == "llama3.2:latest"


# ── the memory fallback ──────────────────────────────────────────
def _engine(**kw) -> OllamaEngine:
    return OllamaEngine(
        host="http://localhost:11434",
        default_model="llama3.2:latest",
        fallback_model="llama3.2:1b",
        **kw,
    )


def _http_error(status: int) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "http://localhost:11434/api/chat")
    response = httpx.Response(status, request=request)
    return httpx.HTTPStatusError("boom", request=request, response=response)


def test_server_error_triggers_the_fallback():
    """A 5xx is what an out-of-memory model load looks like."""
    engine = _engine()
    assert engine._should_try_fallback("llama3.2:latest", _http_error(500)) is True


def test_connection_failure_does_not_trigger_the_fallback():
    """Ollama being down is not fixed by different weights."""
    engine = _engine()
    exc = httpx.ConnectError("refused")
    assert engine._should_try_fallback("llama3.2:latest", exc) is False


def test_client_error_does_not_trigger_the_fallback():
    engine = _engine()
    assert engine._should_try_fallback("llama3.2:latest", _http_error(404)) is False


def test_fallback_does_not_retry_itself():
    engine = _engine()
    assert engine._should_try_fallback("llama3.2:1b", _http_error(500)) is False


def test_no_fallback_configured_means_no_retry():
    engine = OllamaEngine(
        host="http://localhost:11434",
        default_model="llama3.2:latest",
        fallback_model="",
    )
    assert engine._should_try_fallback("llama3.2:latest", _http_error(500)) is False


def test_fallback_is_reported_not_hidden(monkeypatch):
    """A weaker model answering must be visible, or a degraded reply looks like
    the main model's best effort."""
    engine = _engine()
    calls: list[str] = []

    def fake_stream(messages, opts, model, fell_back=False):
        calls.append(model)
        if model == "llama3.2:latest":
            raise _http_error(500)
        yield __import__("app.engines.base", fromlist=["StreamEvent"]).StreamEvent(
            token="hi"
        )
        yield __import__("app.engines.base", fromlist=["StreamEvent"]).StreamEvent(
            done=True, model=model, fell_back=fell_back
        )

    monkeypatch.setattr(engine, "_stream_once", fake_stream)
    events = list(engine.chat_stream([ChatMessage(role="user", content="hi")]))

    assert calls == ["llama3.2:latest", "llama3.2:1b"]
    assert events[-1].fell_back is True
    assert events[-1].model == "llama3.2:1b"


def test_original_error_surfaces_when_the_fallback_also_fails(monkeypatch):
    """Commonly the fallback was never pulled. The first error describes the
    real problem, so that is the one worth reporting."""
    engine = _engine()

    def always_fails(messages, opts, model, fell_back=False):
        raise _http_error(500 if model == "llama3.2:latest" else 404)
        yield  # pragma: no cover - generator marker

    monkeypatch.setattr(engine, "_stream_once", always_fails)
    with pytest.raises(httpx.HTTPStatusError) as caught:
        list(engine.chat_stream([ChatMessage(role="user", content="hi")]))
    assert caught.value.response.status_code == 500
