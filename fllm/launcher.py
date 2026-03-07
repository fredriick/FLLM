"""
launcher.py — Orchestrates the full pipeline:
  detect -> select -> download -> launch
"""
from __future__ import annotations
import gc, json, os, sys, warnings
from pathlib import Path
from typing import Literal, Optional

from .scout import HardwareScout, HardwareProfile
from .selector import ModelSelector, ModelSelection
from .downloader import ModelDownloader, default_cache_dir
from .registry import resolve as resolve_family
from .speculative import SpeculativeManager
from .chat import ChatSession, TemplateRenderer, InteractiveChat
from .backends.llamacpp import LlamaCppBackend
from .backends.vllm import VLLMBackend
from .backends.mlx import MLXBackend
from .backends.airllm import AirLLMBackend

Mode = Literal["server", "interactive", "bench"]

class LLMRunner:
    def __init__(self, cache_dir=None, verbose=False, force_tier=None,
                 force_backend=None, use_speculative=True, compression=None,
                 context=None):
        self.cache_dir = cache_dir or default_cache_dir()
        self.verbose = verbose
        self.force_tier = force_tier
        self.force_backend = force_backend
        self.use_speculative = use_speculative
        self.compression = compression
        self.context = context
        self._hw = None
        self._sel = None

    def detect(self):
        if self._hw is None:
            self._hw = HardwareScout(verbose=self.verbose).detect_all()
            if self.force_tier:
                self._hw.tier = self.force_tier.upper()
            if self.context:
                self._hw.max_context_tokens = self.context
        return self._hw

    def select_model(self, family):
        self.detect()
        self._sel = ModelSelector(self._hw).select(family)
        return self._sel

    def download(self):
        if self._sel is None:
            raise RuntimeError("Call select_model() first.")
        return ModelDownloader(self.cache_dir).ensure(self._sel)

    def run(self, family, mode="server", port=8080, model_path=None,
            system_prompt=None, no_spec=False):
        hw = self.detect()
        _print_hw(hw)
        sel = self.select_model(family)
        _print_model(sel)
        path = model_path or self.download()

        for w in hw.warnings:
            print(f"  !  {w}")

        # Speculative decoding check
        spec = None
        if self.use_speculative and not no_spec:
            fam_entry = resolve_family(family)
            if fam_entry:
                spec = SpeculativeManager(hw, fam_entry, self.cache_dir).evaluate()
                if not spec.enabled:
                    print(f"  i  Spec decode off: {spec.reason_disabled}")

        backend = self._build_backend(hw, sel)
        print(f"\n  Backend: {backend.name}")

        if not backend.is_available():
            print(f"\n  Backend not installed.\n  {backend.install_hint()}")
            sys.exit(1)

        if mode == "bench":
            self._run_bench(sel, path, backend)
        elif mode == "server":
            backend.launch_server(path, port=port)
        else:
            self._run_interactive(sel, path, backend,
                                  system_prompt or sel.family.default_system)

    def info(self):
        hw = self.detect()
        print(json.dumps({
            "tier": hw.tier, "os": hw.os, "arch": hw.arch,
            "recommended_backend": hw.recommended_backend,
            "cpu": {"model": hw.cpu_model, "physical_cores": hw.cpu_physical_cores,
                    "logical_cores": hw.cpu_logical_cores, "instructions": hw.cpu_instructions},
            "ram": {"total_gb": round(hw.total_ram_gb, 1), "type": hw.ram_type,
                    "speed_mhz": hw.ram_speed_mhz, "dual_channel": hw.dual_channel},
            "storage": {"type": hw.storage_type, "pcie_version": hw.pcie_version,
                        "pcie_lanes": hw.pcie_lanes},
            "gpus": [{"name": g.name, "vendor": g.vendor, "vram_gb": round(g.vram_gb,1),
                      "architecture": g.architecture, "compute_capability": g.compute_capability,
                      "bandwidth_gbps": g.memory_bandwidth_gbps} for g in hw.gpus],
            "unified_memory": hw.unified_memory,
            "optimal_quant_bits": hw.optimal_quant_bits,
            "max_context_tokens": hw.max_context_tokens,
            "warnings": hw.warnings,
        }, indent=2))

    def models(self):
        from .registry import list_families
        families = list_families()
        print(f"\n  Supported models ({len(families)} total)\n")
        for f in families:
            sizes = ", ".join(s.label for s in f.sizes)
            print(f"  {f.key:<12} {f.display:<20} {sizes}")
        print(f"\n  Run 'fllm run <model>' to start.\n")

    def bench(self, family, model_path=None, output=None):
        hw = self.detect()
        _print_hw(hw)
        sel = self.select_model(family)
        _print_model(sel)
        path = model_path or self.download()
        backend = self._build_backend(hw, sel)
        if not backend.is_available():
            print(f"  {backend.install_hint()}")
            sys.exit(1)
        self._run_bench(sel, path, backend, output_path=output)

    def _run_interactive(self, sel, path, backend, system_prompt):
        renderer = TemplateRenderer(sel.family)
        session = ChatSession(
            family_key=sel.family.key,
            model_label=f"{sel.family.display} {sel.size.label} {sel.quant_method}",
            system_prompt=system_prompt,
            context_limit=sel.context_tokens,
        )
        generate_fn, cleanup_fn = self._make_generate_fn(backend, path, sel)
        InteractiveChat(session=session, renderer=renderer,
                        generate_fn=generate_fn, cleanup_fn=cleanup_fn).run()

    def _make_generate_fn(self, backend, path, sel):
        """Return (generate_fn, cleanup_fn) tuple."""
        if "llama" in backend.name.lower():
            try:
                from llama_cpp import Llama
                from .backends.llamacpp import _gpu_layers
                ngl = _gpu_layers(self._hw, sel)
                try:
                    _fd = os.dup(2)
                    os.dup2(os.open(os.devnull, os.O_WRONLY), 2)
                    try:
                        llm = Llama(model_path=str(path), n_ctx=sel.context_tokens,
                                    n_gpu_layers=ngl if ngl >= 0 else 999, verbose=False)
                    finally:
                        os.dup2(_fd, 2)
                        os.close(_fd)
                except (ValueError, RuntimeError):
                    print("  ⚠  GPU init failed, retrying CPU-only …", file=sys.stderr)
                    llm = Llama(model_path=str(path), n_ctx=sel.context_tokens,
                                n_gpu_layers=0, verbose=False)
                def _gen(prompt):
                    out = llm.create_completion(prompt, max_tokens=1024,
                                               stop=["<|im_end|>","<|eot_id|>","<end_of_turn>"])
                    return out["choices"][0]["text"].strip()
                def _cleanup():
                    nonlocal llm
                    del llm
                    gc.collect()
                return _gen, _cleanup
            except ImportError:
                pass
        if "mlx" in backend.name.lower() or "mlc" in backend.name.lower():
            try:
                from mlx_lm import load, generate as mlx_gen
                model, tokenizer = load(sel.mlx_repo or sel.hf_repo)
                def _gen(prompt):
                    return mlx_gen(model, tokenizer, prompt=prompt, max_tokens=1024, verbose=False)
                def _cleanup():
                    nonlocal model, tokenizer
                    del model, tokenizer
                    gc.collect()
                return _gen, _cleanup
            except ImportError:
                pass
        return lambda prompt: "[Install llama-cpp-python for interactive mode]", None

    def _run_bench(self, sel, path, backend, output_path=None):
        from .benchmark import Benchmarker, llamacpp_generate_fn
        try:
            from llama_cpp import Llama
            from .backends.llamacpp import _gpu_layers
            ngl = _gpu_layers(self._hw, sel)
            try:
                _fd = os.dup(2)
                os.dup2(os.open(os.devnull, os.O_WRONLY), 2)
                try:
                    llm = Llama(model_path=str(path), n_ctx=sel.context_tokens,
                                n_gpu_layers=ngl if ngl >= 0 else 999, verbose=False)
                finally:
                    os.dup2(_fd, 2)
                    os.close(_fd)
            except (ValueError, RuntimeError):
                print("  ⚠  GPU init failed, retrying CPU-only …", file=sys.stderr)
                llm = Llama(model_path=str(path), n_ctx=sel.context_tokens,
                            n_gpu_layers=0, verbose=False)
            raw = llamacpp_generate_fn(llm)
        except ImportError:
            print("  llama-cpp-python required for bench.", file=sys.stderr)
            sys.exit(1)
        save = output_path or (
            self.cache_dir.parent / "benchmarks" /
            f"{sel.family.key}_{sel.size.label}_{sel.quant_method}.json"
        )
        try:
            Benchmarker(generate_fn=raw,
                        model_label=f"{sel.family.display} {sel.size.label}",
                        backend_name=backend.name, hw_tier=self._hw.tier,
                        quant_method=sel.quant_method, output_path=save).run()
        finally:
            del llm
            gc.collect()

    def _build_backend(self, hw, sel):
        name = self.force_backend or hw.recommended_backend
        if name == "vllm":
            b = VLLMBackend(hw, sel)
            return b if b.is_available() else LlamaCppBackend(hw, sel)
        if name == "mlx":
            warnings.warn(
                "[experimental] mlx-lm backend requires MLX-format models from mlx-community. "
                "Falling back to llama.cpp (Metal). Use --backend mlx to force.",
                stacklevel=2
            )
            return LlamaCppBackend(hw, sel)
        if name == "airllm":
            b = AirLLMBackend(hw, sel, compression=self.compression)
            return b if b.is_available() else LlamaCppBackend(hw, sel)
        return LlamaCppBackend(hw, sel)


def _print_hw(hw):
    tier_desc = {"A": "High-End GPU (A)", "B": "Unified Memory (B)", "C": "CPU/Low-End (C)"}
    print(f"\n{'─'*55}")
    print(f"  Hardware  : {tier_desc.get(hw.tier, hw.tier)}")
    print(f"  OS        : {hw.os} ({hw.arch})")
    print(f"  CPU       : {hw.cpu_model}")
    print(f"  RAM       : {hw.total_ram_gb:.1f} GB {hw.ram_type}")
    print(f"  Storage   : {hw.storage_type}")
    for g in hw.gpus:
        print(f"  GPU       : {g.name}  ({g.vram_gb:.1f} GB)")
    if not hw.gpus:
        print(f"  GPU       : none")
    print(f"{'─'*55}")

def _print_model(sel):
    mem = "Full VRAM" if sel.fits_in_vram else ("Partial offload" if sel.needs_cpu_offload else "RAM")
    print(f"\n  Model    : {sel.family.display} {sel.size.label}")
    print(f"  Quant    : {sel.quant_method} ({sel.quant_bits}-bit)")
    print(f"  Size     : ~{sel.estimated_size_gb:.1f} GB")
    print(f"  Context  : {sel.context_tokens:,} tokens")
    print(f"  Memory   : {mem}")
    print(f"  Template : {sel.family.chat_template}")
    print(f"  Repo     : {sel.gguf_repo}")
    print(f"{'─'*55}")
