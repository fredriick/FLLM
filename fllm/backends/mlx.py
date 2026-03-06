"""
backends/mlx.py — Apple Silicon backend via mlx-lm.

EXPERIMENTAL: mlx-lm backend requires MLX-format models (mlx-community HuggingFace org),
not GGUF. This backend will not work with the standard GGUF download pipeline.
Planned for v0.2.0 with a separate MLX registry and downloader path.

Uses the MLX framework (Apple's Metal-optimised ML library).
Runs in unified memory: no CPU↔GPU transfer overhead.
Supports InteractiveChat with full template + /command support.
"""

from __future__ import annotations

import atexit
import gc
import signal
import subprocess
import sys
from pathlib import Path
from typing import Optional

from ..scout import HardwareProfile
from ..selector import ModelSelection
from ..registry import FamilyEntry
from ..speculative import SpecConfig
from ..chat import ChatSession, TemplateRenderer, InteractiveChat


class MLXBackend:
    name = "mlx-lm  (Apple Silicon)"

    def __init__(self, hw: HardwareProfile, sel: ModelSelection, family: Optional[FamilyEntry] = None):
        self.hw = hw
        self.sel = sel
        self.family = family

    def is_available(self) -> bool:
        try:
            import mlx_lm  # noqa: F401
            return True
        except ImportError:
            return False

    def install_hint(self) -> str:
        return (
            "mlx-lm is not installed.\n"
            "  pip install mlx-lm\n"
            "Requires macOS 14+ on Apple Silicon (M1 or later)."
        )

    def _mlx_model_id(self) -> str:
        """Resolve the mlx-community model ID, falling back gracefully."""
        if self.sel.mlx_repo:
            return self.sel.mlx_repo
        # Generic fallback: try mlx-community convention
        name = self.family.display.replace(" ", "-")
        return f"mlx-community/{name}-{self.sel.size.label}-Instruct-4bit"

    # ── Server ────────────────────────────────────────────────────────────────

    def launch_server(self, model_path: Path, port: int = 8080,
                      spec: Optional[SpecConfig] = None):
        if not self.is_available():
            print(f"ERROR: {self.install_hint()}", file=sys.stderr)
            sys.exit(1)

        model_id = self._mlx_model_id()
        cmd = [
            sys.executable, "-m", "mlx_lm.server",
            "--model", model_id,
            "--port",  str(port),
        ]

        print(f"\n  ▶  mlx-lm server  →  http://127.0.0.1:{port}/v1")
        print(f"     Model   : {model_id}")
        print(f"     Context : {self.sel.context_tokens:,} tokens")
        print(f"     Memory  : {self.hw.total_ram_gb:.0f} GB unified")
        print(f"\n  OpenAI-compatible API at http://127.0.0.1:{port}/v1\n")

        proc = subprocess.Popen(cmd)

        def _kill_server():
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()

        atexit.register(_kill_server)
        signal.signal(signal.SIGTERM, lambda *_: (_kill_server(), sys.exit(0)))

        try:
            proc.wait()
        except KeyboardInterrupt:
            _kill_server()
        sys.exit(proc.returncode or 0)

    # ── Interactive ───────────────────────────────────────────────────────────

    def launch_interactive(
        self,
        model_path: Path,
        session: ChatSession,
        renderer: TemplateRenderer,
        spec: Optional[SpecConfig] = None,
    ):
        if not self.is_available():
            print(f"ERROR: {self.install_hint()}", file=sys.stderr)
            sys.exit(1)

        from mlx_lm import load, generate

        model_id = self._mlx_model_id()
        print(f"\n  Loading {model_id} into unified memory …")
        model, tokenizer = load(model_id)

        def _generate(prompt: str) -> str:
            return generate(
                model, tokenizer,
                prompt=prompt,
                max_tokens=512,
                verbose=False,
            )

        def _cleanup():
            nonlocal model, tokenizer
            del model, tokenizer
            gc.collect()

        chat = InteractiveChat(session, renderer, _generate, cleanup_fn=_cleanup)
        chat.run()
