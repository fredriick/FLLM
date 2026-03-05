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

    # ── models ───────────────────────────────────────────────────────────────
    sub.add_parser("models", help="List all supported model families.")

    # ── sessions ─────────────────────────────────────────────────────────────
    sp = sub.add_parser("sessions", help="List saved chat sessions.")
    sp.add_argument("--cache-dir", type=Path, default=None)

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
    rp.add_argument("--bench-output", type=Path, default=None,
                    help="Save benchmark JSON to this path.")

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


def cmd_models(_args):
    print(BANNER)
    from fllm.launcher import LLMRunner
    LLMRunner().models()


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


def cmd_run(args):
    from fllm.launcher import LLMRunner
    print(BANNER)
    runner = LLMRunner(
        cache_dir=args.cache_dir,
        verbose=args.verbose,
        force_tier=args.tier,
        force_backend=args.backend,
        use_speculative=not args.no_spec,
        compression=args.compression,
    )
    runner.run(
        family=args.family,
        mode=args.mode,
        port=getattr(args, "port", 8080),
        model_path=args.model_path,
        system_prompt=getattr(args, "system_prompt", None),
        no_spec=args.no_spec,
    )


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
        "models":   cmd_models,
        "sessions": cmd_sessions,
        "run":      cmd_run,
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
