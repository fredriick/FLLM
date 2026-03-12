"""
metrics.py — Usage metrics tracking for FLLM.

Tracks per-request and aggregate statistics:
  - Tokens per second (TPS)
  - Time to first token (TTFT)
  - Total latency per request
  - Prompt / completion token counts
  - Request counts (success / error)
  - Estimated cost (configurable $/token)

Metrics are held in memory and optionally persisted to
~/.cache/fllm/metrics/<model>.json for historical review.

Usage:
  from fllm.metrics import MetricsTracker

  tracker = MetricsTracker(model_label="Qwen2.5 3B Q4_K_M")
  with tracker.track_request() as req:
      # ... run inference ...
      req.record(prompt_tokens=42, completion_tokens=128, ttft=0.12)

  print(tracker.summary())
"""

from __future__ import annotations

import json
import time
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Request-level metrics
# ---------------------------------------------------------------------------

@dataclass
class RequestMetrics:
    """Metrics for a single inference request."""
    timestamp: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    ttft_seconds: float = 0.0          # time to first token
    total_seconds: float = 0.0         # total request latency
    tokens_per_second: float = 0.0     # completion tokens / total seconds
    stream: bool = False
    success: bool = True
    error: str = ""

    def finalize(self):
        """Compute derived fields."""
        self.total_tokens = self.prompt_tokens + self.completion_tokens
        if self.total_seconds > 0 and self.completion_tokens > 0:
            self.tokens_per_second = round(
                self.completion_tokens / self.total_seconds, 2
            )


# ---------------------------------------------------------------------------
# Request context manager
# ---------------------------------------------------------------------------

class RequestTracker:
    """Context manager returned by MetricsTracker.track_request()."""

    def __init__(self):
        self._metrics = RequestMetrics()
        self._start: float = 0.0
        self._first_token_recorded = False

    def record_first_token(self):
        """Call when the first token is generated (for TTFT)."""
        if not self._first_token_recorded and self._start > 0:
            self._metrics.ttft_seconds = round(
                time.perf_counter() - self._start, 4
            )
            self._first_token_recorded = True

    def record(
        self,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        ttft: Optional[float] = None,
        stream: bool = False,
        success: bool = True,
        error: str = "",
    ):
        """Record final metrics for the request."""
        self._metrics.prompt_tokens = prompt_tokens
        self._metrics.completion_tokens = completion_tokens
        self._metrics.stream = stream
        self._metrics.success = success
        self._metrics.error = error
        if ttft is not None:
            self._metrics.ttft_seconds = round(ttft, 4)

    @property
    def metrics(self) -> RequestMetrics:
        return self._metrics


# ---------------------------------------------------------------------------
# Aggregate summary
# ---------------------------------------------------------------------------

@dataclass
class MetricsSummary:
    """Aggregate metrics across all tracked requests."""
    model_label: str = ""
    uptime_seconds: float = 0.0
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    mean_tps: float = 0.0
    median_tps: float = 0.0
    peak_tps: float = 0.0
    min_tps: float = 0.0
    mean_ttft_seconds: float = 0.0
    mean_latency_seconds: float = 0.0
    median_latency_seconds: float = 0.0
    requests_per_minute: float = 0.0
    estimated_cost_usd: float = 0.0


# ---------------------------------------------------------------------------
# Metrics tracker
# ---------------------------------------------------------------------------

class MetricsTracker:
    """
    Thread-safe usage metrics tracker.

    Tracks all requests and provides real-time and historical summaries.
    """

    # Default cost rates (per 1M tokens) — configurable
    DEFAULT_PROMPT_COST = 0.0    # local inference = free
    DEFAULT_COMPLETION_COST = 0.0

    def __init__(
        self,
        model_label: str = "unknown",
        persist_dir: Optional[Path] = None,
        prompt_cost_per_1m: float = 0.0,
        completion_cost_per_1m: float = 0.0,
    ):
        self.model_label = model_label
        self.persist_dir = persist_dir
        self.prompt_cost_per_1m = prompt_cost_per_1m
        self.completion_cost_per_1m = completion_cost_per_1m

        self._requests: List[RequestMetrics] = []
        self._lock = threading.Lock()
        self._start_time = time.time()

    @contextmanager
    def track_request(self):
        """Context manager to track a single request."""
        tracker = RequestTracker()
        tracker._start = time.perf_counter()
        tracker._metrics.timestamp = datetime.now().isoformat(timespec="seconds")

        try:
            yield tracker
        except Exception as e:
            tracker.record(success=False, error=str(e))
            raise
        finally:
            elapsed = time.perf_counter() - tracker._start
            tracker._metrics.total_seconds = round(elapsed, 4)
            tracker._metrics.finalize()

            with self._lock:
                self._requests.append(tracker._metrics)

    def summary(self) -> MetricsSummary:
        """Compute aggregate summary of all tracked requests."""
        with self._lock:
            requests = list(self._requests)

        s = MetricsSummary(model_label=self.model_label)
        s.uptime_seconds = round(time.time() - self._start_time, 1)
        s.total_requests = len(requests)

        if not requests:
            return s

        successful = [r for r in requests if r.success]
        failed = [r for r in requests if not r.success]

        s.successful_requests = len(successful)
        s.failed_requests = len(failed)

        s.total_prompt_tokens = sum(r.prompt_tokens for r in requests)
        s.total_completion_tokens = sum(r.completion_tokens for r in requests)
        s.total_tokens = s.total_prompt_tokens + s.total_completion_tokens

        # TPS stats (only from successful requests with completions)
        tps_values = [r.tokens_per_second for r in successful if r.tokens_per_second > 0]
        if tps_values:
            s.mean_tps = round(mean(tps_values), 2)
            s.median_tps = round(median(tps_values), 2)
            s.peak_tps = round(max(tps_values), 2)
            s.min_tps = round(min(tps_values), 2)

        # TTFT stats
        ttft_values = [r.ttft_seconds for r in successful if r.ttft_seconds > 0]
        if ttft_values:
            s.mean_ttft_seconds = round(mean(ttft_values), 4)

        # Latency stats
        latency_values = [r.total_seconds for r in successful if r.total_seconds > 0]
        if latency_values:
            s.mean_latency_seconds = round(mean(latency_values), 4)
            s.median_latency_seconds = round(median(latency_values), 4)

        # Requests per minute
        if s.uptime_seconds > 0:
            s.requests_per_minute = round(
                s.total_requests / (s.uptime_seconds / 60), 2
            )

        # Cost estimate
        s.estimated_cost_usd = round(
            (s.total_prompt_tokens / 1_000_000) * self.prompt_cost_per_1m +
            (s.total_completion_tokens / 1_000_000) * self.completion_cost_per_1m,
            6,
        )

        return s

    def summary_dict(self) -> Dict:
        """Return summary as a plain dict (JSON-serializable)."""
        return asdict(self.summary())

    def recent(self, n: int = 10) -> List[Dict]:
        """Return the last N request metrics as dicts."""
        with self._lock:
            recent = self._requests[-n:]
        return [asdict(r) for r in recent]

    def reset(self):
        """Clear all recorded metrics."""
        with self._lock:
            self._requests.clear()
            self._start_time = time.time()

    def persist(self):
        """Save current metrics to disk."""
        if not self.persist_dir:
            return

        self.persist_dir.mkdir(parents=True, exist_ok=True)

        # Save summary
        summary_path = self.persist_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(self.summary_dict(), f, indent=2)

        # Append request log
        log_path = self.persist_dir / "requests.jsonl"
        with self._lock:
            requests = list(self._requests)

        with open(log_path, "a") as f:
            for r in requests:
                f.write(json.dumps(asdict(r)) + "\n")

    @staticmethod
    def load_summary(persist_dir: Path) -> Optional[Dict]:
        """Load a previously saved summary from disk."""
        summary_path = persist_dir / "summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                return json.load(f)
        return None

    @staticmethod
    def load_request_log(persist_dir: Path, last_n: int = 50) -> List[Dict]:
        """Load the last N entries from the request log."""
        log_path = persist_dir / "requests.jsonl"
        if not log_path.exists():
            return []

        lines = log_path.read_text().strip().split("\n")
        lines = lines[-last_n:]
        return [json.loads(line) for line in lines if line.strip()]


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------

def print_summary(summary: MetricsSummary):
    """Print a formatted metrics summary to stdout."""
    uptime = _format_duration(summary.uptime_seconds)

    print(f"\n{'─' * 55}")
    print(f"  FLLM Usage Metrics")
    print(f"{'─' * 55}")
    print(f"  Model        : {summary.model_label}")
    print(f"  Uptime       : {uptime}")
    print(f"  Requests     : {summary.total_requests}"
          f"  ({summary.successful_requests} ok, {summary.failed_requests} failed)")
    print(f"  Req/min      : {summary.requests_per_minute}")
    print(f"{'─' * 55}")
    print(f"  Tokens")
    print(f"    Prompt     : {summary.total_prompt_tokens:,}")
    print(f"    Completion : {summary.total_completion_tokens:,}")
    print(f"    Total      : {summary.total_tokens:,}")
    print(f"{'─' * 55}")
    print(f"  Performance")
    print(f"    Mean TPS   : {summary.mean_tps:>8.1f} tok/s")
    print(f"    Median TPS : {summary.median_tps:>8.1f} tok/s")
    print(f"    Peak TPS   : {summary.peak_tps:>8.1f} tok/s")
    print(f"    Min TPS    : {summary.min_tps:>8.1f} tok/s")
    print(f"    Mean TTFT  : {summary.mean_ttft_seconds * 1000:>8.1f} ms")
    print(f"    Mean Lat.  : {summary.mean_latency_seconds:>8.3f} s")
    print(f"    Median Lat.: {summary.median_latency_seconds:>8.3f} s")

    if summary.estimated_cost_usd > 0:
        print(f"{'─' * 55}")
        print(f"  Est. Cost    : ${summary.estimated_cost_usd:.6f}")

    print(f"{'─' * 55}\n")


def print_recent_requests(requests: List[Dict], n: int = 10):
    """Print a table of recent requests."""
    if not requests:
        print("\n  No requests recorded yet.\n")
        return

    print(f"\n  Recent Requests (last {min(n, len(requests))})")
    print(f"  {'Time':<20} {'Prompt':<8} {'Compl.':<8} {'TPS':<10} {'Latency':<10} {'Status'}")
    print(f"  {'─' * 20} {'─' * 8} {'─' * 8} {'─' * 10} {'─' * 10} {'─' * 8}")

    for r in requests[-n:]:
        ts = r.get("timestamp", "")[-8:]  # HH:MM:SS
        pt = r.get("prompt_tokens", 0)
        ct = r.get("completion_tokens", 0)
        tps = r.get("tokens_per_second", 0)
        lat = r.get("total_seconds", 0)
        ok = "ok" if r.get("success", True) else "FAIL"
        print(f"  {ts:<20} {pt:<8} {ct:<8} {tps:<10.1f} {lat:<10.3f} {ok}")

    print()


def _format_duration(seconds: float) -> str:
    """Format seconds into a human-readable duration string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        m, s = divmod(int(seconds), 60)
        return f"{m}m {s}s"
    else:
        h, rem = divmod(int(seconds), 3600)
        m, s = divmod(rem, 60)
        return f"{h}h {m}m {s}s"
