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


class OllamaEngine(LLMEngine):
    def __init__(
        self,
        host: str,
        default_model: str,
        temperature: float = 0.7,
        timeout_s: float = 120.0,
        num_ctx: int | None = None,
    ) -> None:
        self._host = host.rstrip("/")
        self._default_model = default_model
        self._temperature = temperature
        self._timeout_s = timeout_s
        self._num_ctx = num_ctx

    # ── readiness ────────────────────────────────────────────────
    def health_check(self) -> bool:
        try:
            resp = httpx.get(f"{self._host}/api/tags", timeout=5.0)
            return resp.status_code == 200
        except httpx.HTTPError:
            return False

    def list_models(self) -> list[str]:
        resp = httpx.get(f"{self._host}/api/tags", timeout=5.0)
        resp.raise_for_status()
        return [m["name"] for m in resp.json().get("models", [])]

    # ── tool-calling (non-streaming) ─────────────────────────────
    def chat(
        self,
        messages: list[ChatMessage],
        tools: list[dict] | None = None,
        options: GenerationOptions | None = None,
    ) -> AssistantTurn:
        opts = options or GenerationOptions()
        payload: dict = {
            "model": opts.model or self._default_model,
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
        opts = options or GenerationOptions()
        model = opts.model or self._default_model

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
                        prompt_tokens=chunk.get("prompt_eval_count"),
                        completion_tokens=chunk.get("eval_count"),
                        time_to_first_token_s=first_token_at,
                        load_duration_s=_sec("load_duration"),
                        prompt_eval_duration_s=_sec("prompt_eval_duration"),
                        eval_duration_s=_sec("eval_duration"),
                        total_duration_s=_sec("total_duration"),
                    )
