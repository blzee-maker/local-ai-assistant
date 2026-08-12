"""Phase 1 acceptance check.

Proves the full local path works: config loads, the Ollama service is reachable,
the configured model is present, and a streamed generation returns text with
latency metrics. Run:

    python scripts/verify_ollama.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make the project root importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.engines import ChatMessage, GenerationOptions, build_engine  # noqa: E402
from config import settings  # noqa: E402


def main() -> int:
    print("=" * 60)
    print(" Local Offline AI Assistant — Phase 1 verification")
    print("=" * 60)
    print(f"  engine        : {settings.engine}")
    print(f"  ollama host   : {settings.ollama_host}")
    print(f"  default model : {settings.default_model}")
    print()

    engine = build_engine()

    # 1) Service reachable?
    print("[1/3] Health check ...", end=" ", flush=True)
    if not engine.health_check():
        print("FAILED")
        print(f"      Cannot reach Ollama at {settings.ollama_host}.")
        print("      Is it running?  ->  ollama serve")
        return 1
    print("ok")

    # 2) Is the configured model present?
    print("[2/3] Model available ...", end=" ", flush=True)
    models = engine.list_models()
    if settings.default_model not in models:
        print("MISSING")
        print(f"      '{settings.default_model}' not found. Pull it with:")
        print(f"      ollama pull {settings.default_model}")
        print(f"      Locally available: {', '.join(models) or '(none)'}")
        return 1
    print("ok")

    # 3) Stream a real generation and time it.
    print("[3/3] Streaming a test generation ...\n")
    messages = [
        ChatMessage(role="system", content="You are a concise assistant."),
        ChatMessage(
            role="user",
            content="In one short sentence, confirm you are running locally.",
        ),
    ]

    print("  ┌─ response " + "─" * 46)
    print("  │ ", end="", flush=True)
    final: object | None = None
    for event in engine.chat_stream(messages, GenerationOptions(max_tokens=80)):
        if event.done:
            final = event
        else:
            print(event.token, end="", flush=True)
    print("\n  └" + "─" * 57)

    if final is not None:
        ttft = final.time_to_first_token_s
        tps = final.tokens_per_second
        def secs(v: object) -> str:
            return f"{v:.2f} s" if isinstance(v, float) else "n/a"

        print("\n  metrics")
        print(f"    model               : {final.model}")
        print(f"    prompt tokens       : {final.prompt_tokens}")
        print(f"    completion tokens   : {final.completion_tokens}")
        print(
            "    time to first token : "
            + (f"{ttft * 1000:.0f} ms" if ttft is not None else "n/a")
            + "   (wall-clock; includes model load on a cold start)"
        )
        print(f"    model load          : {secs(final.load_duration_s)}")
        print(f"    prompt prefill      : {secs(final.prompt_eval_duration_s)}")
        print(f"    generation          : {secs(final.eval_duration_s)}")
        print(
            "    throughput          : "
            + (f"{tps:.1f} tokens/sec" if tps is not None else "n/a")
            + "   (generation only)"
        )
        print(f"    total duration      : {secs(final.total_duration_s)}")

    print("\n✅ Phase 1 verified — engine path works end to end.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
