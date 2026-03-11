"""
backends/llamacpp.py — llama.cpp backend (Tier B + C, GPU offload).

Supports:
  - llama-server binary  (API server mode)
  - llama-cli binary     (interactive mode)
  - llama-cpp-python     (Python bindings fallback for both)
  - Speculative decoding via --draft-model flag
  - InteractiveChat session with templates, /commands, and autosave
"""

from __future__ import annotations

import atexit
import gc
import os
import signal
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

from ..scout import HardwareProfile
from ..selector import ModelSelection
from ..registry import FamilyEntry
from ..speculative import SpecConfig
from ..chat import ChatSession, TemplateRenderer, InteractiveChat


# ---------------------------------------------------------------------------
# Binary resolution
# ---------------------------------------------------------------------------

_SERVER_NAMES = ["llama-server", "server"]
_CLI_NAMES    = ["llama-cli", "main"]
_EXTRA_DIRS = [
    Path.home() / ".local" / "bin",
    Path("/opt/llama.cpp/bin"),
    Path("/usr/local/bin"),
]
if sys.platform == "win32":
    _EXTRA_DIRS.append(
        Path(os.environ.get("LOCALAPPDATA", "")) / "llama.cpp"
    )


def _find(candidates: list[str]) -> Optional[str]:
    for name in candidates:
        hit = shutil.which(name)
        if hit:
            return hit
    for d in _EXTRA_DIRS:
        for name in candidates:
            p = d / name
            if p.exists():
                return str(p)
    return None


# ---------------------------------------------------------------------------
# GPU layer count
# ---------------------------------------------------------------------------

def _gpu_layers(hw: HardwareProfile, sel: ModelSelection) -> int:
    """0 = CPU, -1 = all layers on GPU, N = partial offload."""
    if not hw.gpus or hw.tier == "C":
        return 0
    if hw.tier == "B":
        return -1   # Metal / Vulkan unified
    # Tier A — partial or full
    avail = hw.total_vram_gb * 0.85
    if sel.estimated_size_gb <= avail:
        return -1
    frac = avail / sel.estimated_size_gb
    n_layers = sel.size.n_layers
    return max(1, int(n_layers * frac))


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------

class LlamaCppBackend:
    name = "llama.cpp"

    def __init__(self, hw: HardwareProfile, sel: ModelSelection, family: Optional[FamilyEntry] = None):
        self.hw = hw
        self.sel = sel
        self.family = family

    def is_available(self) -> bool:
        if _find(_SERVER_NAMES) or _find(_CLI_NAMES):
            return True
        try:
            import llama_cpp  # noqa: F401
            return True
        except ImportError:
            return False

    def install_hint(self) -> str:
        return (
            "llama.cpp is not installed.\n"
            "  pip install llama-cpp-python        # Python bindings (easiest)\n"
            "  brew install llama.cpp              # macOS\n"
            "  https://github.com/ggerganov/llama.cpp/releases  # pre-built binaries"
        )

    # ── Server ────────────────────────────────────────────────────────────────

    def launch_server(self, model_path: Path, port: int = 8080, spec: Optional[SpecConfig] = None, web: bool = False):
        binary = _find(_SERVER_NAMES)
        if not binary:
            return self._py_server(model_path, port, web=web)

        ngl     = _gpu_layers(self.hw, self.sel)
        threads = max(1, self.hw.cpu_physical_cores - 1)
        ctx     = self.sel.context_tokens

        cmd = [
            binary,
            "--model",    str(model_path),
            "--ctx-size", str(ctx),
            "--threads",  str(threads),
            "--port",     str(port),
            "--host",     "127.0.0.1",
            "--flash-attn",
        ]
        if ngl != 0:
            cmd += ["--n-gpu-layers", str(999 if ngl < 0 else ngl)]

        # Speculative decoding
        if spec and spec.enabled and spec.draft_model_path:
            cmd += ["--draft-model", str(spec.draft_model_path),
                    "--draft", str(spec.n_draft_tokens)]

        _print_server_info(port, ngl, ctx, threads, spec)
        if web:
            print(f"  i  llama-server has a built-in web UI at http://127.0.0.1:{port}")
            _open_browser(port)
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
        generate_fn, cleanup_fn = self._make_generate_fn(model_path, spec)
        chat = InteractiveChat(session, renderer, generate_fn, cleanup_fn=cleanup_fn)
        chat.run()

    def _make_generate_fn(self, model_path: Path, spec: Optional[SpecConfig]):
        """Return (generate_fn, cleanup_fn) using llama-cpp-python."""
        try:
            from llama_cpp import Llama
        except ImportError:
            print("ERROR: llama-cpp-python not installed.\n"
                  "  pip install llama-cpp-python", file=sys.stderr)
            sys.exit(1)

        ngl = _gpu_layers(self.hw, self.sel)

        kwargs: dict = dict(
            model_path=str(model_path),
            n_ctx=self.sel.context_tokens,
            n_gpu_layers=999 if ngl < 0 else ngl,
            verbose=False,
        )
        if spec and spec.enabled and spec.draft_model_path:
            # llama-cpp-python ≥0.2.90 supports draft_model
            kwargs["draft_model"] = str(spec.draft_model_path)

        try:
            # Suppress noisy Metal/GPU shader errors on first attempt
            _fd = os.dup(2)
            os.dup2(os.open(os.devnull, os.O_WRONLY), 2)
            try:
                llm = Llama(**kwargs)
            finally:
                os.dup2(_fd, 2)
                os.close(_fd)
        except (ValueError, RuntimeError):
            # Metal backend may fail under Rosetta 2 or incompatible macOS
            print("  ⚠  GPU init failed, retrying CPU-only …", file=sys.stderr)
            kwargs["n_gpu_layers"] = 0
            try:
                llm = Llama(**kwargs)
            except (ValueError, RuntimeError):
                print("ERROR: Failed to load model. If on Apple Silicon under Rosetta 2,\n"
                      "  try: pip install 'llama-cpp-python==0.2.90'", file=sys.stderr)
                sys.exit(1)

        def _generate(prompt: str) -> str:
            out = llm(prompt, max_tokens=512, stop=["<|im_end|>", "<|eot_id|>",
                                                     "</s>", "User:", "\nUser:"])
            return out["choices"][0]["text"].strip()

        def _cleanup():
            nonlocal llm
            del llm
            gc.collect()

        return _generate, _cleanup

    # ── Python-bindings server fallback ───────────────────────────────────────

    def _py_server(self, model_path: Path, port: int, web: bool = False):
        try:
            from llama_cpp.server.app import create_app
            from llama_cpp.server.settings import ModelSettings, ServerSettings, Settings
            import uvicorn
        except ImportError:
            print("ERROR: Neither llama-server binary nor llama-cpp-python found.",
                  file=sys.stderr)
            print(self.install_hint(), file=sys.stderr)
            sys.exit(1)

        ngl = _gpu_layers(self.hw, self.sel)

        # Force CPU mode if FLLM_CPU_ONLY is set or Metal fails
        if os.environ.get("FLLM_CPU_ONLY", "").lower() in ("1", "true", "yes"):
            ngl = 0
            print("  ℹ  FLLM_CPU_ONLY=1 forcing CPU mode", file=sys.stderr)

        server_settings = ServerSettings(
            host="127.0.0.1",
            port=port,
        )

        try:
            model_settings = ModelSettings(
                model=str(model_path),
                n_ctx=self.sel.context_tokens,
                n_gpu_layers=999 if ngl < 0 else ngl,
            )
            app = create_app(
                model_settings=[model_settings],
                server_settings=server_settings,
            )
        except (ValueError, RuntimeError):
            if ngl == 0:
                raise
            print("  ⚠  GPU init failed, retrying CPU-only …", file=sys.stderr)
            model_settings = ModelSettings(
                model=str(model_path),
                n_ctx=self.sel.context_tokens,
                n_gpu_layers=0,
            )
            app = create_app(
                model_settings=[model_settings],
                server_settings=server_settings,
            )

        model_label = f"{self.sel.family.display} {self.sel.size.label} {self.sel.quant_method}"
        if web:
            from ..webui import mount_webui
            mount_webui(app, model_label)
            print(f"\n  ▶  llama-cpp-python server  →  http://127.0.0.1:{port}/v1")
            print(f"  ▶  Web UI                   →  http://127.0.0.1:{port}")
            _open_browser(port)
        else:
            print(f"\n  ▶  llama-cpp-python server  →  http://127.0.0.1:{port}/v1")

        uvicorn.run(app, host="127.0.0.1", port=port)


def _open_browser(port: int, delay: float = 1.5):
    """Open the browser after a short delay to let the server start."""
    import threading
    import webbrowser

    def _go():
        import time
        time.sleep(delay)
        webbrowser.open(f"http://127.0.0.1:{port}")

    threading.Thread(target=_go, daemon=True).start()


def _print_server_info(port, ngl, ctx, threads, spec):
    ngl_str = "all" if ngl < 0 else (str(ngl) if ngl else "none  (CPU)")
    spec_str = (f"✓  draft={spec.draft_model_path.name}, n={spec.n_draft_tokens}"
                if spec and spec.enabled else "off")
    print(f"\n  ▶  llama-server  →  http://127.0.0.1:{port}/v1")
    print(f"     GPU layers     : {ngl_str}")
    print(f"     Context        : {ctx:,} tokens")
    print(f"     Threads        : {threads}")
    print(f"     Spec decoding  : {spec_str}")
    print(f"\n  OpenAI-compatible API at http://127.0.0.1:{port}/v1\n")
