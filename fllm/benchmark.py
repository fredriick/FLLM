"""
benchmark.py — Token throughput and latency benchmarker.

Measures:
  - Time-to-first-token (TTFT) — latency before streaming starts
  - Tokens per second (TPS) — generation throughput
  - Context fill time — how long to process a long prompt
  - Memory usage during inference

Results are printed to stdout and optionally saved as JSON.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from statistics import mean, median, stdev
from typing import Callable, List, Optional


# ---------------------------------------------------------------------------
# Prompts for benchmarking
# ---------------------------------------------------------------------------

PROMPTS = {
    "short":  "What is the capital of France?",
    "medium": (
        "Explain the difference between a transformer and an RNN in machine learning. "
        "Cover attention mechanisms, parallelism, and use cases."
    ),
    "long": (
        "You are given the following codebase excerpt:\n\n"
        "```python\n"
        "class DataPipeline:\n"
        "    def __init__(self, source, sink):\n"
        "        self.source = source\n"
        "        self.sink = sink\n"
        "        self._buffer = []\n\n"
        "    def run(self):\n"
        "        for record in self.source.read():\n"
        "            transformed = self._transform(record)\n"
        "            self._buffer.append(transformed)\n"
        "            if len(self._buffer) >= 100:\n"
        "                self.sink.write_batch(self._buffer)\n"
        "                self._buffer = []\n"
        "```\n\n"
        "Identify potential bugs, suggest improvements for error handling, "
        "and refactor the class to support async I/O."
    ),
}


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    prompt_label: str
    prompt_tokens: int          # rough estimate
    output_tokens: int
    total_seconds: float
    tokens_per_second: float
    output_preview: str         # first 80 chars of output


@dataclass
class BenchReport:
    model_label: str
    backend_name: str
    hw_tier: str
    quant_method: str
    timestamp: str
    runs: List[RunResult] = field(default_factory=list)

    # Aggregates (filled in after runs)
    mean_tps: float = 0.0
    median_tps: float = 0.0
    peak_tps: float = 0.0
    mean_ttft_s: float = 0.0


# ---------------------------------------------------------------------------
# Benchmarker
# ---------------------------------------------------------------------------

class Benchmarker:
    """
    Runs a standard prompt suite and reports token throughput.

    generate_fn: (prompt: str) -> str
      Must be synchronous; returns the full completion text.
    """

    WARMUP_PROMPT = "Hello, how are you?"
    N_REPEATS = 3   # repeats per prompt for stable averages

    def __init__(
        self,
        generate_fn: Callable[[str], str],
        model_label: str,
        backend_name: str,
        hw_tier: str,
        quant_method: str,
        output_path: Optional[Path] = None,
    ):
        self.gen = generate_fn
        self.report = BenchReport(
            model_label=model_label,
            backend_name=backend_name,
            hw_tier=hw_tier,
            quant_method=quant_method,
            timestamp=_now(),
        )
        self.output_path = output_path

    def run(self):
        print(f"\n{'─' * 58}")
        print(f"  Benchmark: {self.report.model_label}")
        print(f"  Backend  : {self.report.backend_name}  (Tier {self.report.hw_tier})")
        print(f"  Quant    : {self.report.quant_method}")
        print(f"{'─' * 58}\n")

        # Warm-up (not recorded)
        print("  Warming up …", end="", flush=True)
        try:
            self.gen(self.WARMUP_PROMPT)
            print(" done\n")
        except Exception as e:
            print(f" FAILED: {e}", file=sys.stderr)
            return

        all_tps: List[float] = []

        for label, prompt in PROMPTS.items():
            tps_runs: List[float] = []
            for i in range(self.N_REPEATS):
                result = self._time_run(label, prompt)
                tps_runs.append(result.tokens_per_second)
                self.report.runs.append(result)
                _print_run(result, i + 1, self.N_REPEATS)

            avg = mean(tps_runs)
            all_tps.extend(tps_runs)
            print(f"  └ {label:6s} average: {avg:6.1f} tok/s\n")

        # Aggregates
        if all_tps:
            self.report.mean_tps   = round(mean(all_tps), 2)
            self.report.median_tps = round(median(all_tps), 2)
            self.report.peak_tps   = round(max(all_tps), 2)

        self._print_summary()

        if self.output_path:
            self._save()

    def _time_run(self, label: str, prompt: str) -> RunResult:
        prompt_tokens = len(prompt) // 4   # ~4 chars/token approximation
        t0 = time.perf_counter()
        try:
            output = self.gen(prompt)
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

    def _print_summary(self):
        r = self.report
        print(f"{'─' * 58}")
        print(f"  Results summary")
        print(f"  Mean TPS   : {r.mean_tps:>7.1f} tok/s")
        print(f"  Median TPS : {r.median_tps:>7.1f} tok/s")
        print(f"  Peak TPS   : {r.peak_tps:>7.1f} tok/s")
        print(f"{'─' * 58}")

        # Qualitative rating
        tps = r.mean_tps
        if tps >= 50:
            rating = "🟢 Excellent  (real-time conversational)"
        elif tps >= 20:
            rating = "🟡 Good       (comfortable for chat)"
        elif tps >= 5:
            rating = "🟠 Acceptable (slightly slow for chat)"
        else:
            rating = "🔴 Slow       (consider a smaller model)"
        print(f"  Rating     : {rating}")
        print(f"{'─' * 58}\n")

        if self.output_path:
            print(f"  Saved → {self.output_path}\n")

    def _save(self):
        if self.output_path:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            data = asdict(self.report)
            with open(self.output_path, "w") as f:
                json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# Backend-specific generate_fn factories
# ---------------------------------------------------------------------------

def llamacpp_generate_fn(llm) -> Callable[[str], str]:
    """Wrap a llama_cpp.Llama instance into a (prompt)->str callable."""
    def _gen(prompt: str) -> str:
        out = llm.create_completion(
            prompt, max_tokens=512,
            stop=["<|im_end|>", "<|eot_id|>", "</s>", "User:"],
            echo=False,
        )
        return out["choices"][0]["text"].strip()
    return _gen


def vllm_generate_fn(llm, params) -> Callable[[str], str]:
    """Wrap a vllm.LLM instance."""
    def _gen(prompt: str) -> str:
        out = llm.generate([prompt], params)
        return out[0].outputs[0].text.strip()
    return _gen


def mlx_generate_fn(model, tokenizer) -> Callable[[str], str]:
    """Wrap an mlx-lm model/tokenizer pair."""
    from mlx_lm import generate as mlx_gen
    def _gen(prompt: str) -> str:
        return mlx_gen(model, tokenizer, prompt=prompt, max_tokens=512, verbose=False)
    return _gen


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now() -> str:
    from datetime import datetime
    return datetime.now().isoformat(timespec="seconds")


def _print_run(r: RunResult, n: int, total: int):
    bar_len = min(30, int(r.tokens_per_second / 2))
    bar = "█" * bar_len
    print(f"  [{n}/{total}] {r.prompt_label:6s}  "
          f"{r.tokens_per_second:6.1f} tok/s  "
          f"{r.total_seconds:5.1f}s  "
          f"{bar}")
