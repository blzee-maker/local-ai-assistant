"""Ollama-backed implementation of LLMEngine.

Talks to the local Ollama service over its REST API using httpx. We hit the raw
endpoints (rather than the `ollama` python package) to keep the data flow fully
transparent and dependency-light.
"""
from __future__ import annotations

import json
import time
from typing import Iterator

import httpx

from .base import (
    AssistantTurn,
    ChatMessage,
    GenerationOptions,
    LLMEngine,
    StreamEvent,
    ToolCall,
)
from .policy import check_model, partition_models


class OllamaEngine(LLMEngine):
    def __init__(
        self,
        host: str,
        default_model: str,
        temperature: float = 0.7,
        timeout_s: float = 120.0,
        num_ctx: int | None = None,
        fallback_model: str | None = None,
        allow_remote_models: bool = False,
    ) -> None:
        self._host = host.rstrip("/")
        self._default_model = default_model
        self._temperature = temperature
        self._timeout_s = timeout_s
        self._num_ctx = num_ctx
        self._fallback_model = fallback_model or None
        self._allow_remote = allow_remote_models

        check_model(default_model, allow_remote=allow_remote_models)
        if self._fallback_model:
            check_model(self._fallback_model, allow_remote=allow_remote_models)

    def _resolve_model(self, options: GenerationOptions | None) -> str:
        """Pick the model for a request and enforce the offline guarantee.

        Every path that names a model goes through here — including per-request
        overrides from `/model` and `--model`, which is exactly where a cloud
        model would otherwise slip in.
        """
        name = (options.model if options else None) or self._default_model
        check_model(name, allow_remote=self._allow_remote)
        return name

    def _should_try_fallback(self, model: str, exc: Exception) -> bool:
        """Only retry when a smaller model could plausibly help.

        A 5xx from Ollama is what an out-of-memory model load looks like, and on
        a memory-tight machine that is the common failure. A refused connection
        means the server is down, where retrying with different weights is
        pointless noise.
        """
        if not self._fallback_model or model == self._fallback_model:
            return False
        return (
            isinstance(exc, httpx.HTTPStatusError)
            and exc.response.status_code >= 500
        )

    # ── readiness ────────────────────────────────────────────────
    def health_check(self) -> bool:
        try:
            resp = httpx.get(f"{self._host}/api/tags", timeout=5.0)
            return resp.status_code == 200
        except httpx.HTTPError:
            return False

    def list_models(self) -> list[str]:
        """Locally runnable models. Cloud entries are filtered out unless
        explicitly allowed, so nothing offers the user a model this engine
        would then refuse to run."""
        resp = httpx.get(f"{self._host}/api/tags", timeout=5.0)
        resp.raise_for_status()
        names = [m["name"] for m in resp.json().get("models", [])]
        if self._allow_remote:
            return names
        local, _remote = partition_models(names)
        return local

    def list_remote_models(self) -> list[str]:
        """Cloud-hosted models present in the registry — surfaced by `doctor`
        so the user knows they exist and that they are being refused."""
        try:
            resp = httpx.get(f"{self._host}/api/tags", timeout=5.0)
            resp.raise_for_status()
        except httpx.HTTPError:
            return []
        _local, remote = partition_models(
            [m["name"] for m in resp.json().get("models", [])]
        )
        return remote

    # ── tool-calling (non-streaming) ─────────────────────────────
    def chat(
        self,
        messages: list[ChatMessage],
        tools: list[dict] | None = None,
        options: GenerationOptions | None = None,
    ) -> AssistantTurn:
        opts = options or GenerationOptions()
        payload: dict = {
            "model": self._resolve_model(opts),
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": False,
            "options": {
                "temperature": (
                    opts.temperature if opts.temperature is not None else self._temperature
                ),
            },
        }
        if tools:
            payload["tools"] = tools
        ctx = opts.num_ctx if opts.num_ctx is not None else self._num_ctx
        if ctx is not None:
            payload["options"]["num_ctx"] = ctx

        resp = httpx.post(
            f"{self._host}/api/chat", json=payload, timeout=self._timeout_s
        )
        resp.raise_for_status()
        message = resp.json().get("message", {})

        tool_calls: list[ToolCall] = []
        for tc in message.get("tool_calls") or []:
            fn = tc.get("function", {})
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            tool_calls.append(ToolCall(name=fn.get("name", ""), arguments=args or {}))

        return AssistantTurn(content=message.get("content", ""), tool_calls=tool_calls)

    # ── generation ───────────────────────────────────────────────
    def chat_stream(
        self,
        messages: list[ChatMessage],
        options: GenerationOptions | None = None,
    ) -> Iterator[StreamEvent]:
        """Stream a reply, dropping to the fallback model if the primary can't load.

        The retry only happens before any token has been emitted. Once text has
        reached the user, restarting with a different model would splice two
        different answers together, so a mid-stream failure is surfaced as-is.
        """
        opts = options or GenerationOptions()
        model = self._resolve_model(opts)

        try:
            first_attempt = self._stream_once(messages, opts, model)
            first_event = next(first_attempt)
        except StopIteration:
            return
        except Exception as exc:
            if not self._should_try_fallback(model, exc):
                raise
            fallback = self._fallback_model
            assert fallback is not None
            try:
                retry = self._stream_once(messages, opts, fallback, fell_back=True)
                first_event = next(retry)
            except StopIteration:
                return
            except Exception:
                # The fallback failed too (commonly: never pulled). The original
                # error describes the real problem, so report that one.
                raise exc
            yield first_event
            yield from retry
            return

        yield first_event
        yield from first_attempt

    def _stream_once(
        self,
        messages: list[ChatMessage],
        opts: GenerationOptions,
        model: str,
        fell_back: bool = False,
    ) -> Iterator[StreamEvent]:
        payload: dict = {
            "model": model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": True,
            "options": {
                "temperature": (
                    opts.temperature
                    if opts.temperature is not None
                    else self._temperature
                ),
            },
        }
        if opts.max_tokens is not None:
            payload["options"]["num_predict"] = opts.max_tokens
        ctx = opts.num_ctx if opts.num_ctx is not None else self._num_ctx
        if ctx is not None:
            payload["options"]["num_ctx"] = ctx
        if opts.stop:
            payload["options"]["stop"] = opts.stop

        start = time.perf_counter()
        first_token_at: float | None = None

        with httpx.stream(
            "POST",
            f"{self._host}/api/chat",
            json=payload,
            timeout=self._timeout_s,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line)

                content = chunk.get("message", {}).get("content", "")
                if content:
                    if first_token_at is None:
                        first_token_at = time.perf_counter() - start
                    yield StreamEvent(token=content)

                if chunk.get("done"):
                    # Ollama reports all durations in nanoseconds.
                    def _sec(key: str) -> float | None:
                        ns = chunk.get(key)
                        return (ns / 1e9) if ns else None

                    yield StreamEvent(
                        done=True,
                        model=model,
                        fell_back=fell_back,
                        prompt_tokens=chunk.get("prompt_eval_count"),
                        completion_tokens=chunk.get("eval_count"),
                        time_to_first_token_s=first_token_at,
                        load_duration_s=_sec("load_duration"),
                        prompt_eval_duration_s=_sec("prompt_eval_duration"),
                        eval_duration_s=_sec("eval_duration"),
                        total_duration_s=_sec("total_duration"),
                    )
