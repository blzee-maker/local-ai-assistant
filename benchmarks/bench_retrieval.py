"""Measure the local RAG retrieval path without invoking an LLM.

Run from the repository root with ``.venv\\Scripts\\python.exe
benchmarks\\bench_retrieval.py``. The corpus is built from the repository's
configured sample_docs directory into a temporary FAISS store, so the user's
normal index is neither read nor changed.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = Path(__file__).resolve().parent
SUPPORTED_SUFFIXES = {".docx", ".md", ".pdf", ".txt"}
SAMPLE_CHECKS = (
    ("What is the peak power draw of the X1?", "41 watts"),
    ("Who founded Halden Systems, and in what year?", "Priya Venkataraman"),
    ("How can the wireless radios be physically disabled?", "kill switch"),
)

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _environment() -> dict[str, Any]:
    import platform
    import winreg

    import faiss
    import numpy
    import onnxruntime
    import psutil
    import fastembed

    with winreg.OpenKey(
        winreg.HKEY_LOCAL_MACHINE,
        r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
    ) as key:
        cpu_model = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()

    memory = psutil.virtual_memory()
    heavy_processes = []
    current_pid = os.getpid()
    for process in psutil.process_iter(["pid", "name", "memory_info"]):
        info = process.info
        usage = info.get("memory_info")
        if info.get("pid") != current_pid and usage and usage.rss >= 200 * 1024**2:
            heavy_processes.append(
                {"name": info.get("name") or "unknown", "rss_mib": usage.rss / 1024**2}
            )
    heavy_processes.sort(key=lambda item: item["rss_mib"], reverse=True)

    return {
        "cpu_model": cpu_model,
        "ram_total_bytes": memory.total,
        "ram_available_bytes_before_benchmark": memory.available,
        "os": platform.platform(),
        "python": platform.python_version(),
        "libraries": {
            "faiss": getattr(faiss, "__version__", "unknown"),
            "onnxruntime": onnxruntime.__version__,
            "numpy": numpy.__version__,
            "fastembed": getattr(fastembed, "__version__", "unknown"),
        },
        "onnxruntime_available_providers": onnxruntime.get_available_providers(),
        "gpu_used": False,
        "gpu_usage_basis": (
            "Captured from InferenceSession.get_providers() on the live ONNX "
            "Runtime session used by FastEmbed."
        ),
        "other_processes_at_least_200_mib_rss": heavy_processes,
    }


def _build_corpus(store_dir: Path, corpus_dir: Path) -> None:
    from app.rag.service import RagService

    files = sorted(
        path
        for path in corpus_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )
    if not files:
        raise RuntimeError(f"No supported documents found under {corpus_dir}")

    rag = RagService(store_dir=str(store_dir))
    ingested = [
        rag.ingest_file(str(path), path.name)
        for path in files
    ]
    manifest = {
        "corpus_dir": str(corpus_dir),
        "files": [path.name for path in files],
        "file_count": len(files),
        "total_chunks": rag.count,
        "ingestion": ingested,
    }
    (store_dir / "benchmark_corpus.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _percentile_95(values: list[float]) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * 0.95
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _summarize(values: list[float]) -> dict[str, float]:
    return {
        "median_ms": statistics.median(values),
        "mean_ms": statistics.mean(values),
        "p95_ms": _percentile_95(values),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def _worker(store_dir: Path) -> dict[str, Any]:
    import onnxruntime

    from app.rag.embedder import Embedder
    from app.rag.service import RagService
    from app.rag.store import VectorStore
    from config import settings

    queries = [
        line.strip()
        for line in (BENCHMARK_DIR / "queries.txt").read_text(
            encoding="utf-8-sig"
        ).splitlines()
        if line.strip()
    ]
    if len(queries) < 30:
        raise RuntimeError(f"Expected at least 30 queries; found {len(queries)}")

    manifest = json.loads(
        (store_dir / "benchmark_corpus.json").read_text(encoding="utf-8")
    )
    environment = _environment()

    # Instrument the project's existing methods; RagService.retrieve remains
    # the only retrieval entry point used for every timed query.
    active_query: str | None = None
    component_times: dict[str, float] = {}
    original_embed = Embedder.embed
    original_search = VectorStore.search
    original_session = onnxruntime.InferenceSession
    selected_providers: list[str] = []

    def observed_session(*args, **kwargs):
        session = original_session(*args, **kwargs)
        selected_providers.extend(session.get_providers())
        return session

    def timed_embed(embedder: Embedder, texts: list[str]):
        start = time.perf_counter()
        result = original_embed(embedder, texts)
        elapsed = (time.perf_counter() - start) * 1000
        if active_query is not None and texts == [active_query]:
            component_times["embedding_ms"] = elapsed
        return result

    def timed_search(vector_store: VectorStore, query_vec, k: int):
        start = time.perf_counter()
        result = original_search(vector_store, query_vec, k)
        elapsed = (time.perf_counter() - start) * 1000
        if active_query is not None:
            component_times["search_ms"] = elapsed
        return result

    Embedder.embed = timed_embed
    VectorStore.search = timed_search
    onnxruntime.InferenceSession = observed_session
    cold_query = queries[0]
    cold_start = time.perf_counter()
    rag = RagService(store_dir=str(store_dir))
    active_query = cold_query
    cold_results = rag.retrieve(cold_query)
    cold_total_ms = (time.perf_counter() - cold_start) * 1000
    cold_components = dict(component_times)
    active_query = None
    if not cold_results:
        raise RuntimeError(f"Cold query returned no chunks: {cold_query}")

    def run_group(k: int | None, label: str) -> dict[str, Any]:
        for query in queries[:5]:
            rag.retrieve(query, k)

        measured: dict[str, list[float]] = {
            "embedding": [],
            "search": [],
            "total": [],
        }
        returned_count = 0
        for _ in range(3):
            for query in queries:
                nonlocal active_query
                active_query = query
                component_times.clear()
                start = time.perf_counter()
                results = rag.retrieve(query, k)
                total_ms = (time.perf_counter() - start) * 1000
                if not results:
                    raise RuntimeError(
                        f"{label} retrieval returned no chunks for query: {query}"
                    )
                if "embedding_ms" not in component_times or "search_ms" not in component_times:
                    raise RuntimeError("Retrieval component timing was not captured")
                measured["embedding"].append(component_times["embedding_ms"])
                measured["search"].append(component_times["search_ms"])
                measured["total"].append(total_ms)
                returned_count += len(results)
        active_query = None
        return {
            "top_k": settings.rag_top_k if k is None else k,
            "query_count": len(queries),
            "runs_per_query": 3,
            "warmup_queries_excluded": 5,
            "measured_retrievals": len(queries) * 3,
            "chunks_returned_across_measured_retrievals": returned_count,
            "timings": {
                name: _summarize(values)
                for name, values in measured.items()
            },
        }

    default_result = run_group(None, "default top_k")
    groups = [default_result]
    if settings.rag_top_k != 5:
        groups.append(run_group(5, "top_k=5"))

    samples = []
    for query, expected in SAMPLE_CHECKS:
        results = rag.retrieve(query)
        if not results:
            raise RuntimeError(f"Sanity-check query returned no chunks: {query}")
        top_text = results[0]["text"]
        if expected.casefold() not in top_text.casefold():
            raise RuntimeError(
                f"Top result for {query!r} did not contain expected evidence "
                f"{expected!r}: {top_text[:400]!r}"
            )
        samples.append(
            {
                "query": query,
                "expected_evidence": expected,
                "top_result_source": results[0]["source"],
                "top_result_score": results[0]["score"],
                "top_result_text": top_text,
            }
        )

    environment["onnxruntime_selected_providers"] = list(dict.fromkeys(selected_providers))
    environment["gpu_used"] = any(
        "CUDA" in provider or "Dml" in provider or "Tensorrt" in provider
        for provider in environment["onnxruntime_selected_providers"]
    )
    return {
        "measurement": {
            "query_count": len(queries),
            "runs_per_query": 3,
            "warmup_queries_excluded_per_top_k": 5,
            "cold_start_ms": cold_total_ms,
            "cold_start_scope": (
                "RagService construction (including its embedding dimension probe "
                "and FAISS index load) plus the first retrieval; excludes Python "
                "interpreter startup and imports."
            ),
            "cold_start_caveat": (
                "The model was already present in the local cache and downloads "
                "were disabled. The corpus index was built immediately before "
                "measurement, so filesystem-cache state was not controlled."
            ),
            "cold_query": cold_query,
            "cold_query_component_times": cold_components,
            "timing_clock": "time.perf_counter()",
            "p95_method": "linear interpolation at rank (n - 1) * 0.95",
            "retrieval_path": "RagService.retrieve -> Embedder.embed -> VectorStore.search",
            "llm_generation_included": False,
        },
        "environment": environment,
        "corpus": {
            **manifest,
            "index_dimension": rag.embedding_dim,
            "index_type": type(rag._store._index).__name__,
        },
        "benchmarks": groups,
        "sanity_samples": samples,
        "caveats": [
            f"The in-repository sample corpus contains {manifest['file_count']} file "
            f"and {manifest['total_chunks']} vectors; these measurements do not "
            "establish latency at larger corpus sizes.",
            "The cold-start figure measures a fresh benchmark process with local "
            "model files already cached; it excludes interpreter startup/imports "
            "and any first-time model download.",
            "Other background processes were active; their observed memory use is "
            "listed in the environment section.",
        ],
    }


def _render_markdown(results: dict[str, Any]) -> str:
    lines = [
        "# Local RAG retrieval benchmark",
        "",
        "LLM generation is excluded. Warm statistics include only the timed retrieval calls; "
        "the first cold retrieval and five warm-up calls per top-k are excluded from them.",
        "",
        "## Measured latency",
        "",
        "| top_k | stage | median (ms) | mean (ms) | p95 (ms) | min (ms) | max (ms) |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for group in results["benchmarks"]:
        for stage, stats in group["timings"].items():
            lines.append(
                f"| {group['top_k']} | {stage} | {stats['median_ms']} | "
                f"{stats['mean_ms']} | {stats['p95_ms']} | {stats['min_ms']} | "
                f"{stats['max_ms']} |"
            )
    measurement = results["measurement"]
    corpus = results["corpus"]
    lines.extend(
        [
            "",
            "## Run and corpus",
            "",
            f"- Queries: {measurement['query_count']}; measured runs per query: "
            f"{measurement['runs_per_query']}; measured retrievals per top_k: "
            f"{measurement['query_count'] * measurement['runs_per_query']}.",
            f"- Cold first retrieval: {measurement['cold_start_ms']} ms "
            f"({measurement['cold_start_scope']})",
            f"- Cold-start caveat: {measurement['cold_start_caveat']}",
            f"- Corpus: {corpus['file_count']} files, {corpus['total_chunks']} "
            f"chunks/vectors, dimension {corpus['index_dimension']}, "
            f"{corpus['index_type']}.",
            f"- Corpus files: {', '.join(corpus['files'])}.",
            f"- Default top_k: {results['benchmarks'][0]['top_k']}.",
            f"- Cold first-query embedding: "
            f"{measurement['cold_query_component_times'].get('embedding_ms')} ms; "
            f"FAISS search: "
            f"{measurement['cold_query_component_times'].get('search_ms')} ms.",
            "",
            "## Environment",
            "",
        ]
    )
    env = results["environment"]
    lines.extend(
        [
            f"- CPU: {env['cpu_model']}.",
            f"- RAM: {env['ram_total_bytes']} bytes total; "
            f"{env['ram_available_bytes_before_benchmark']} bytes available before run.",
            f"- OS: {env['os']}; Python: {env['python']}.",
            "- Libraries: " + ", ".join(
                f"{name} {version}" for name, version in env["libraries"].items()
            ) + ".",
            f"- ONNX Runtime providers available: "
            f"{', '.join(env['onnxruntime_available_providers'])}; selected: "
            f"{', '.join(env['onnxruntime_selected_providers']) or 'not exposed by fastembed'}.",
            f"- GPU used: {env['gpu_used']}.",
            "- Other processes with at least 200 MiB RSS: "
            + (
                ", ".join(
                    f"{process['name']} ({process['rss_mib']} MiB)"
                    for process in env["other_processes_at_least_200_mib_rss"]
                )
                or "none observed"
            )
            + ".",
            "",
            "## Caveats",
            "",
        ]
    )
    lines.extend(f"- {caveat}" for caveat in results["caveats"])
    lines.extend(["", "## Retrieval sanity checks", ""])
    for sample in results["sanity_samples"]:
        lines.extend(
            [
                f"- Query: {sample['query']}",
                f"  - Top result: {sample['top_result_text']}",
                f"  - Source: {sample['top_result_source']}; "
                f"score: {sample['top_result_score']}; checked for "
                f"`{sample['expected_evidence']}`.",
            ]
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--store-dir", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--corpus-dir", type=Path, default=ROOT / "sample_docs")
    args = parser.parse_args()

    if args.build_only:
        if args.store_dir is None:
            parser.error("--store-dir is required in build mode")
        _build_corpus(args.store_dir, args.corpus_dir.resolve())
        return
    if args.worker:
        if args.store_dir is None:
            parser.error("--store-dir is required in worker mode")
        print("BENCHMARK_JSON:" + json.dumps(_worker(args.store_dir)))
        return

    environment = os.environ.copy()
    environment.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "PYTHONIOENCODING": "utf-8",
        }
    )
    with tempfile.TemporaryDirectory(prefix="local-rag-benchmark-") as temp:
        store_dir = Path(temp) / "vector_store"
        common = [sys.executable, str(Path(__file__).resolve())]
        build = subprocess.run(
            common
            + [
                "--build-only",
                "--store-dir",
                str(store_dir),
                "--corpus-dir",
                str(args.corpus_dir.resolve()),
            ],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if build.returncode:
            raise RuntimeError(
                "Corpus preparation failed:\n" + (build.stderr or build.stdout)[-4000:]
            )
        worker = subprocess.run(
            common + ["--worker", "--store-dir", str(store_dir)],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if worker.returncode:
            raise RuntimeError(
                "Benchmark process failed:\n" + (worker.stderr or worker.stdout)[-4000:]
            )
        marker = next(
            (
                line[len("BENCHMARK_JSON:") :]
                for line in reversed(worker.stdout.splitlines())
                if line.startswith("BENCHMARK_JSON:")
            ),
            None,
        )
        if marker is None:
            raise RuntimeError(
                "Benchmark process returned no result payload:\n"
                + worker.stdout[-4000:]
            )
        results = json.loads(marker)

    json_path = BENCHMARK_DIR / "results.json"
    markdown_path = BENCHMARK_DIR / "results.md"
    json_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    markdown_path.write_text(_render_markdown(results), encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {markdown_path}")
    print(_render_markdown(results))


if __name__ == "__main__":
    main()
