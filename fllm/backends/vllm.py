"""
backends/vllm.py — vLLM backend for Tier A (NVIDIA/AMD high-end GPU).

Features:
  - Continuous batching + PagedAttention (built into vLLM)
  - Tensor parallelism across multiple GPUs
  - AWQ / bitsandbytes quantization pass-through
  - InteractiveChat session with full template + /command support
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


class VLLMBackend:
    name = "vLLM"

    def __init__(self, hw: HardwareProfile, sel: ModelSelection, family: Optional[FamilyEntry] = None):
        self.hw = hw
        self.sel = sel
        self.family = family

    def is_available(self) -> bool:
        try:
            import vllm  # noqa: F401
            return True
        except ImportError:
            return False

    def install_hint(self) -> str:
        return (
            "vLLM is not installed.\n"
            "  pip install vllm\n"
            "Requires CUDA 11.8+ and an NVIDIA (or ROCm AMD) GPU.\n"
            "See https://docs.vllm.ai for platform-specific wheels."
        )

    # ── Server ────────────────────────────────────────────────────────────────

    def launch_server(self, model_path: Path, port: int = 8080,
                      spec: Optional[SpecConfig] = None):
        if not self.is_available():
            print(f"ERROR: {self.install_hint()}", file=sys.stderr)
            sys.exit(1)

        n_gpus = max(1, len([g for g in self.hw.gpus if g.vendor in ("nvidia", "amd")]))
        model_id = self.sel.hf_repo   # vLLM reads HF format, not GGUF

        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model",                model_id,
            "--tensor-parallel-size", str(n_gpus),
            "--max-model-len",        str(self.sel.context_tokens),
            "--port",                 str(port),
            "--host",                 "127.0.0.1",
            "--enable-chunked-prefill",   # continuous batching improvement
        ]

        # Quantization
        if self.sel.quant_bits == 4:
            cmd += ["--quantization", "awq"]
        elif self.sel.quant_bits == 8:
            cmd += ["--quantization", "bitsandbytes",
                    "--load-format", "bitsandbytes"]

        # Speculative decoding (vLLM ≥ 0.5 supports --speculative-model)
        if spec and spec.enabled and spec.draft_model_path:
            cmd += [
                "--speculative-model",         str(spec.draft_model_path),
                "--num-speculative-tokens",    str(spec.n_draft_tokens),
            ]

        spec_str = (f"✓  n={spec.n_draft_tokens}" if spec and spec.enabled else "off")
        print(f"\n  ▶  vLLM server  →  http://127.0.0.1:{port}/v1")
        print(f"     Model         : {model_id}")
        print(f"     GPU count     : {n_gpus}")
        print(f"     Context       : {self.sel.context_tokens:,} tokens")
        print(f"     Quant         : {self.sel.quant_method}")
        print(f"     Spec decoding : {spec_str}")
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

        from vllm import LLM, SamplingParams

        n_gpus = max(1, len([g for g in self.hw.gpus if g.vendor in ("nvidia", "amd")]))

        kwargs: dict = dict(
            model=self.sel.hf_repo,
            tensor_parallel_size=n_gpus,
            max_model_len=self.sel.context_tokens,
        )
        if self.sel.quant_bits == 4:
            kwargs["quantization"] = "awq"

        llm = LLM(**kwargs)
        params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=512,
                                stop=["<|im_end|>", "<|eot_id|>", "</s>"])

        def _generate(prompt: str) -> str:
            out = llm.generate([prompt], params)
            return out[0].outputs[0].text.strip()

        def _cleanup():
            nonlocal llm
            del llm
            try:
                import torch
                torch.cuda.empty_cache()
            except (ImportError, RuntimeError):
                pass
            gc.collect()

        chat = InteractiveChat(session, renderer, _generate, cleanup_fn=_cleanup)
        chat.run()
