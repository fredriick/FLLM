#!/usr/bin/env python3
"""
cli.py — FLLM command-line interface.

Usage:
  fllm info                              Hardware profile as JSON
  fllm models                            All supported model families
  fllm run <family>                      Detect, download, launch server
  fllm run <family> --mode interactive   Interactive chat
  fllm run <family> --mode bench         Throughput benchmark
  fllm sessions                          List saved chat sessions
  fllm bench <family>                    Run benchmark standalone
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


BANNER = r"""
  ███████╗██╗     ██╗     ███╗   ███╗
  ██╔════╝██║     ██║     ████╗ ████║
  █████╗  ██║     ██║     ██╔████╔██║
  ██╔══╝  ██║     ██║     ██║╚██╔╝██║
  ██║     ███████╗███████╗██║ ╚═╝ ██║
  ╚═╝     ╚══════╝╚══════╝╚═╝     ╚═╝

  Universal LLM Runner  v0.1.0
"""


def _run_args(sub):
    """Shared arguments for any subcommand that loads a model."""
    sub.add_argument("family",
                     help="Model family (e.g. qwen, llama3, deepseek, gemma3).")
    sub.add_argument("--tier", choices=["A","B","C"], default=None,
                     help="Override hardware tier.")
    sub.add_argument("--backend", choices=["llama.cpp","vllm","mlx","airllm"], default=None,
                     help="Override backend.")
    sub.add_argument("--compression", choices=["4bit","8bit",None], default=None,
                     help="AirLLM compression (4bit or 8bit).")
    sub.add_argument("--context", type=int, default=None,
                     help="Context length (default: auto). Lower = less memory.")
    sub.add_argument("--model-path", type=Path, default=None,
                     help="Use a local GGUF file instead of downloading.")
    sub.add_argument("--cache-dir", type=Path, default=None,
                       help="Override model cache directory (~/.cache/fllm).")
    sub.add_argument("--no-spec", action="store_true",
                     help="Disable speculative decoding.")
    sub.add_argument("--verbose", action="store_true",
                     help="Show hardware detection debug output.")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="fllm",
        description="Universal LLM runner — auto-detects hardware, picks the right model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  fllm info\n"
            "  fllm models\n"
            "  fllm run qwen\n"
            "  fllm run llama3 --mode interactive\n"
            "  fllm run deepseek --system 'You are a coding assistant.'\n"
            "  fllm run qwen --tier A --backend vllm --port 9000\n"
            "  fllm run phi4 --model-path ~/models/phi-4-Q4_K_M.gguf\n"
            "  fllm bench qwen\n"
            "  fllm sessions\n"
        ),
    )
    sub = p.add_subparsers(dest="command", metavar="COMMAND")

    # ── info ─────────────────────────────────────────────────────────────────
    sub.add_parser("info", help="Print hardware profile as JSON.")

    # ── scan ─────────────────────────────────────────────────────────────────
    sub.add_parser("scan", help="Scan hardware and recommend models that fit.")

    # ── models ───────────────────────────────────────────────────────────────
    mp = sub.add_parser("models", help="List supported and downloaded models.")
    mp.add_argument("--remove", metavar="FILENAME", default=None,
                    help="Remove a downloaded model by filename (e.g. Qwen2.5-3B-Instruct-Q4_K_M.gguf)")

    # ── sessions ─────────────────────────────────────────────────────────────
    sp = sub.add_parser("sessions", help="List saved chat sessions.")
    sp.add_argument("--cache-dir", type=Path, default=None)

    # ── templates ─────────────────────────────────────────────────────────────
    tp = sub.add_parser("templates", help="List or view prompt templates.")
    tp.add_argument("action", nargs="?", choices=["list", "show"],
                    default="list", help="list = show all, show = display one template.")
    tp.add_argument("name", nargs="?", default=None,
                    help="Template name (for 'show' action).")

    # ── config ────────────────────────────────────────────────────────────────
    cp = sub.add_parser("config", help="View or initialize config file (~/.fllm/config.yaml).")
    cp.add_argument("action", nargs="?", choices=["init", "show", "path"],
                    default="show",
                    help="init = create default config, show = display current, path = print path.")

    # ── run ──────────────────────────────────────────────────────────────────
    rp = sub.add_parser("run", help="Download and launch a model.")
    _run_args(rp)
    rp.add_argument("--mode", choices=["server","interactive","bench"],
                    default="server",
                    help="server (default) | interactive chat | bench throughput.")
    rp.add_argument("--port", type=int, default=8080,
                    help="API server port (default: 8080).")
    rp.add_argument("--system", default=None, dest="system_prompt",
                    help="Custom system prompt for interactive mode.")
    rp.add_argument("--template", default=None,
                    help="Use a named prompt template (e.g. coding, creative, math).")
    rp.add_argument("--web", action="store_true",
                    help="Launch browser-based chat UI.")
    rp.add_argument("--bench-output", type=Path, default=None,
                    help="Save benchmark JSON to this path.")

    # ── serve (OpenAI-compatible API) ────────────────────────────────────────
    svp = sub.add_parser("serve", help="OpenAI-compatible API server (drop-in replacement).")
    _run_args(svp)
    svp.add_argument("--port", type=int, default=8080,
                     help="API server port (default: 8080).")
    svp.add_argument("--web", action="store_true",
                     help="Include browser-based chat UI.")

    # ── metrics ───────────────────────────────────────────────────────────────
    metp = sub.add_parser("metrics", help="View usage metrics for a running or past server.")
    metp.add_argument("--model", default=None,
                      help="Model key to view metrics for (default: show all).")
    metp.add_argument("--live", default=None, metavar="URL",
                      help="Fetch live metrics from a running server (e.g. http://127.0.0.1:8080).")
    metp.add_argument("--recent", type=int, default=0, metavar="N",
                      help="Show last N requests.")

    # ── bench ─────────────────────────────────────────────────────────────────
    bp = sub.add_parser("bench", help="Benchmark token throughput for a model.")
    _run_args(bp)
    bp.add_argument("--output", type=Path, default=None,
                    help="Save benchmark results to JSON file.")

    return p


# ── Command handlers ──────────────────────────────────────────────────────────

def cmd_info(args):
    from fllm.launcher import LLMRunner
    print(BANNER)
    LLMRunner(verbose=getattr(args, "verbose", False)).info()


def cmd_scan(_args):
    print(BANNER)
    from fllm.launcher import LLMRunner
    LLMRunner().scan()


def cmd_models(args):
    print(BANNER)
    from fllm.launcher import LLMRunner
    runner = LLMRunner()
    remove_name = getattr(args, "remove", None)
    if remove_name:
        runner.remove_model(remove_name)
    else:
        runner.models()


def cmd_sessions(args):
    cache = getattr(args, "cache_dir", None) or Path.home() / ".cache" / "fllm"
    session_dir = cache / "sessions"
    if not session_dir.exists() or not list(session_dir.glob("*.json")):
        print("\n  No sessions saved yet.\n")
        return
    sessions = sorted(session_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    print(f"\n  Saved chat sessions ({len(sessions)} total)\n")
    for p in sessions[:20]:
        kb = p.stat().st_size / 1024
        mtime = p.stat().st_mtime
        from datetime import datetime
        dt = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
        print(f"  {p.stem:<40}  {kb:6.1f} KB  {dt}")
    print(f"\n  Load in interactive mode with:  /load <name>\n")


def cmd_templates(args):
    print(BANNER)
    from fllm.templates import print_templates, print_template_detail

    action = getattr(args, "action", "list")
    name = getattr(args, "name", None)

    if action == "show" and name:
        print_template_detail(name)
    elif action == "show" and not name:
        print("\n  Usage: fllm templates show <name>\n")
    else:
        print_templates()


def cmd_config(args):
    print(BANNER)
    from fllm.config import load_config, init_config, print_config, config_path

    action = getattr(args, "action", "show")

    if action == "init":
        p = init_config()
        print(f"\n  Config created at: {p}\n")
        print_config(load_config())
    elif action == "path":
        print(f"\n  {config_path()}\n")
    else:
        config = load_config()
        print_config(config)


def cmd_run(args):
    from fllm.launcher import LLMRunner
    from fllm.config import load_config, apply_config_defaults
    print(BANNER)

    config = load_config()
    args = apply_config_defaults(args, config, model_key=args.family)

    runner = LLMRunner(
        cache_dir=args.cache_dir,
        verbose=args.verbose,
        force_tier=args.tier,
        force_backend=args.backend,
        use_speculative=not args.no_spec,
        compression=args.compression,
        context=getattr(args, "context", None),
    )
    # Resolve system prompt: --system > --template > model default
    system_prompt = getattr(args, "system_prompt", None)
    template_name = getattr(args, "template", None)
    if not system_prompt and template_name:
        from fllm.templates import get_template
        system_prompt = get_template(template_name)
        if system_prompt:
            print(f"  Using template: {template_name}\n")

    runner.run(
        family=args.family,
        mode=args.mode,
        port=getattr(args, "port", 8080),
        model_path=args.model_path,
        system_prompt=system_prompt,
        no_spec=args.no_spec,
        web=getattr(args, "web", False),
    )


def cmd_serve(args):
    from fllm.launcher import LLMRunner
    from fllm.config import load_config, apply_config_defaults
    print(BANNER)

    config = load_config()
    args = apply_config_defaults(args, config, model_key=args.family)

    runner = LLMRunner(
        cache_dir=args.cache_dir,
        verbose=args.verbose,
        force_tier=args.tier,
        force_backend=args.backend,
        use_speculative=not args.no_spec,
        compression=args.compression,
        context=getattr(args, "context", None),
    )
    runner.serve(
        family=args.family,
        port=getattr(args, "port", 8080),
        model_path=args.model_path,
        web=getattr(args, "web", False),
    )


def cmd_metrics(args):
    print(BANNER)
    from fllm.metrics import (
        MetricsTracker, MetricsSummary, print_summary, print_recent_requests,
    )

    live_url = getattr(args, "live", None)
    model_key = getattr(args, "model", None)
    recent_n = getattr(args, "recent", 0)

    # Live mode — fetch from running server
    if live_url:
        import urllib.request
        import json

        try:
            url = f"{live_url.rstrip('/')}/v1/metrics"
            with urllib.request.urlopen(url, timeout=5) as resp:
                data = json.loads(resp.read())
            summary = MetricsSummary(**data)
            print_summary(summary)
        except Exception as e:
            print(f"\n  Failed to fetch metrics from {live_url}: {e}\n")
            return

        if recent_n > 0:
            try:
                url = f"{live_url.rstrip('/')}/v1/metrics/recent?n={recent_n}"
                with urllib.request.urlopen(url, timeout=5) as resp:
                    data = json.loads(resp.read())
                print_recent_requests(data.get("requests", []), recent_n)
            except Exception as e:
                print(f"  Failed to fetch recent requests: {e}\n")
        return

    # Offline mode — read from disk
    metrics_base = Path.home() / ".cache" / "fllm" / "metrics"

    if not metrics_base.exists():
        print("\n  No metrics data found.")
        print("  Start a server with 'fllm serve <model>' to begin tracking.\n")
        return

    if model_key:
        model_dir = metrics_base / model_key
        data = MetricsTracker.load_summary(model_dir)
        if data:
            summary = MetricsSummary(**data)
            print_summary(summary)
        else:
            print(f"\n  No metrics found for model '{model_key}'.\n")

        if recent_n > 0:
            requests = MetricsTracker.load_request_log(model_dir, recent_n)
            print_recent_requests(requests, recent_n)
    else:
        # Show all models
        found = False
        for model_dir in sorted(metrics_base.iterdir()):
            if model_dir.is_dir():
                data = MetricsTracker.load_summary(model_dir)
                if data:
                    found = True
                    summary = MetricsSummary(**data)
                    print_summary(summary)

        if not found:
            print("\n  No metrics data found.")
            print("  Start a server with 'fllm serve <model>' to begin tracking.\n")


def cmd_bench(args):
    from fllm.launcher import LLMRunner
    print(BANNER)
    runner = LLMRunner(
        cache_dir=args.cache_dir,
        verbose=args.verbose,
        force_tier=args.tier,
        force_backend=args.backend,
        use_speculative=not args.no_spec,
    )
    runner.bench(
        family=args.family,
        model_path=args.model_path,
        output=getattr(args, "output", None),
    )


def main():
    parser = _build_parser()
    args = parser.parse_args()

    dispatch = {
        "info":     cmd_info,
        "scan":     cmd_scan,
        "models":   cmd_models,
        "sessions":   cmd_sessions,
        "templates":  cmd_templates,
        "config":     cmd_config,
        "run":      cmd_run,
        "serve":    cmd_serve,
        "metrics":  cmd_metrics,
        "bench":    cmd_bench,
    }

    fn = dispatch.get(args.command)
    if fn is None:
        print(BANNER)
        parser.print_help()
        sys.exit(0)

    fn(args)


if __name__ == "__main__":
    main()
