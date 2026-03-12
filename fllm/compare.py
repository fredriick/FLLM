"""
compare.py — Benchmark multiple models side-by-side.

Loads each model sequentially, runs the standard benchmark suite,
then prints a comparison table and optionally saves results.

Usage:
  fllm compare qwen llama3 phi4
  fllm compare qwen llama3 --output comparison.json
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from statistics import mean, median
from typing import Dict, List, Optional

from .benchmark import PROMPTS, RunResult, BenchReport, _now, _print_run


# ---------------------------------------------------------------------------
# Comparison result
# ---------------------------------------------------------------------------

@dataclass
class ComparisonEntry:
    """Summary for one model in the comparison."""
    model_label: str
    family_key: str
    quant_method: str
    size_label: str
    estimated_gb: float
    backend: str
    mean_tps: float = 0.0
    median_tps: float = 0.0
    peak_tps: float = 0.0
    mean_latency: float = 0.0
    # Per-prompt breakdown
    prompt_tps: Dict[str, float] = field(default_factory=dict)
    rating: str = ""
    runs: List[RunResult] = field(default_factory=list)


@dataclass
class ComparisonReport:
    """Full comparison across multiple models."""
    timestamp: str = ""
    hw_tier: str = ""
    entries: List[ComparisonEntry] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Comparison runner
# ---------------------------------------------------------------------------

N_REPEATS = 2  # fewer repeats per model to keep total time reasonable
WARMUP_PROMPT = "Hello, how are you?"


def run_comparison(
    families: List[str],
    runner,  # LLMRunner instance
    output_path: Optional[Path] = None,
) -> ComparisonReport:
    """
    Benchmark each model family and produce a comparison report.
    Models are loaded and unloaded sequentially to manage memory.
    """
    from .benchmark import llamacpp_generate_fn
    from .backends.llamacpp import _gpu_layers

    report = ComparisonReport(
        timestamp=_now(),
        hw_tier=runner._hw.tier if runner._hw else "",
    )

    total = len(families)

    for idx, family in enumerate(families, 1):
        print(f"\n{'═' * 58}")
        print(f"  [{idx}/{total}] Benchmarking: {family}")
        print(f"{'═' * 58}")

        # Select and download model
        try:
            sel = runner.select_model(family)
        except Exception as e:
            print(f"  ⚠  Skipping {family}: {e}")
            continue

        try:
            path = runner.download()
        except Exception as e:
            print(f"  ⚠  Download failed for {family}: {e}")
            continue

        model_label = f"{sel.family.display} {sel.size.label}"
        backend = runner._build_backend(runner._hw, sel)

        print(f"  Model  : {model_label} ({sel.quant_method})")
        print(f"  Size   : ~{sel.estimated_size_gb:.1f} GB")
        print(f"  Backend: {backend.name}")

        entry = ComparisonEntry(
            model_label=model_label,
            family_key=family,
            quant_method=sel.quant_method,
            size_label=sel.size.label,
            estimated_gb=sel.estimated_size_gb,
            backend=backend.name,
        )

        # Load model
        llm = None
        try:
            from llama_cpp import Llama

            ngl = _gpu_layers(runner._hw, sel)
            try:
                _fd = os.dup(2)
                os.dup2(os.open(os.devnull, os.O_WRONLY), 2)
                try:
                    llm = Llama(
                        model_path=str(path),
                        n_ctx=sel.context_tokens,
                        n_gpu_layers=999 if ngl < 0 else ngl,
                        verbose=False,
                    )
                finally:
                    os.dup2(_fd, 2)
                    os.close(_fd)
            except (ValueError, RuntimeError):
                print("  ⚠  GPU init failed, retrying CPU-only …")
                llm = Llama(
                    model_path=str(path),
                    n_ctx=sel.context_tokens,
                    n_gpu_layers=0,
                    verbose=False,
                )
        except ImportError:
            print("  ⚠  llama-cpp-python required. Skipping.")
            continue

        gen_fn = llamacpp_generate_fn(llm)

        # Warmup
        print("  Warming up …", end="", flush=True)
        try:
            gen_fn(WARMUP_PROMPT)
            print(" done\n")
        except Exception as e:
            print(f" FAILED: {e}")
            del llm
            gc.collect()
            continue

        # Run benchmark prompts
        all_tps: List[float] = []

        for label, prompt in PROMPTS.items():
            tps_runs: List[float] = []
            for i in range(N_REPEATS):
                result = _time_run(gen_fn, label, prompt)
                tps_runs.append(result.tokens_per_second)
                entry.runs.append(result)
                _print_run(result, i + 1, N_REPEATS)

            avg = mean(tps_runs)
            entry.prompt_tps[label] = round(avg, 2)
            all_tps.extend(tps_runs)
            print(f"  └ {label:6s} average: {avg:6.1f} tok/s\n")

        # Aggregates
        if all_tps:
            entry.mean_tps = round(mean(all_tps), 2)
            entry.median_tps = round(median(all_tps), 2)
            entry.peak_tps = round(max(all_tps), 2)

        latencies = [r.total_seconds for r in entry.runs]
        if latencies:
            entry.mean_latency = round(mean(latencies), 3)

        entry.rating = _rate(entry.mean_tps)

        report.entries.append(entry)

        # Unload model to free memory
        del llm
        gc.collect()
        print(f"  Model unloaded. Memory freed.")

    # Print comparison
    print_comparison(report)

    # Save
    if output_path:
        _save_report(report, output_path)

    return report


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _time_run(gen_fn, label: str, prompt: str) -> RunResult:
    prompt_tokens = len(prompt) // 4
    t0 = time.perf_counter()
    try:
        output = gen_fn(prompt)
    except Exception as e:
        output = f"[ERROR: {e}]"
    elapsed = time.perf_counter() - t0

    out_tokens = max(1, len(output) // 4)
    tps = out_tokens / elapsed if elapsed > 0 else 0.0

    return RunResult(
        prompt_label=label,
        prompt_tokens=prompt_tokens,
        output_tokens=out_tokens,
        total_seconds=round(elapsed, 3),
        tokens_per_second=round(tps, 2),
        output_preview=output[:80].replace("\n", " "),
    )


def _rate(tps: float) -> str:
    if tps >= 50:
        return "Excellent"
    elif tps >= 20:
        return "Good"
    elif tps >= 5:
        return "Acceptable"
    else:
        return "Slow"


def _save_report(report: ComparisonReport, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = asdict(report)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\n  Saved → {path}\n")


# ---------------------------------------------------------------------------
# Print comparison table
# ---------------------------------------------------------------------------

def print_comparison(report: ComparisonReport):
    """Print a side-by-side comparison table."""
    entries = report.entries
    if not entries:
        print("\n  No models were benchmarked.\n")
        return

    # Sort by mean TPS descending
    entries.sort(key=lambda e: e.mean_tps, reverse=True)

    print(f"\n{'═' * 78}")
    print(f"  MODEL COMPARISON — {report.timestamp}")
    print(f"  Hardware Tier: {report.hw_tier}")
    print(f"{'═' * 78}\n")

    # Header
    print(f"  {'#':<4} {'Model':<24} {'Quant':<10} {'Size':<8} "
          f"{'Mean TPS':<10} {'Peak TPS':<10} {'Latency':<10} {'Rating'}")
    print(f"  {'─' * 4} {'─' * 24} {'─' * 10} {'─' * 8} "
          f"{'─' * 10} {'─' * 10} {'─' * 10} {'─' * 12}")

    for i, e in enumerate(entries, 1):
        winner = " 🏆" if i == 1 and len(entries) > 1 else ""
        print(f"  {i:<4} {e.model_label:<24} {e.quant_method:<10} "
              f"{e.estimated_gb:<7.1f}G "
              f"{e.mean_tps:<10.1f} {e.peak_tps:<10.1f} "
              f"{e.mean_latency:<9.3f}s {e.rating}{winner}")

    # Per-prompt breakdown
    prompt_labels = list(PROMPTS.keys())
    if any(e.prompt_tps for e in entries):
        print(f"\n  Per-prompt breakdown (avg tok/s):\n")
        header = f"  {'Model':<24}"
        for p in prompt_labels:
            header += f" {p:<10}"
        print(header)
        print(f"  {'─' * 24}" + f" {'─' * 10}" * len(prompt_labels))

        for e in entries:
            row = f"  {e.model_label:<24}"
            for p in prompt_labels:
                val = e.prompt_tps.get(p, 0)
                row += f" {val:<10.1f}"
            print(row)

    # Speed comparison
    if len(entries) >= 2:
        fastest = entries[0]
        slowest = entries[-1]
        if slowest.mean_tps > 0:
            ratio = fastest.mean_tps / slowest.mean_tps
            print(f"\n  {fastest.model_label} is {ratio:.1f}x faster than "
                  f"{slowest.model_label}")

    print(f"\n{'═' * 78}\n")
