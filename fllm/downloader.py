"""
downloader.py — Fetches GGUF model files from HuggingFace Hub.

Uses huggingface_hub under the hood for auth, resume, and integrity checks.
Falls back to plain urllib if hf_hub is unavailable.
"""

from __future__ import annotations

import os
import sys
import urllib.request
from pathlib import Path
from typing import Optional

from .selector import ModelSelection


# ---------------------------------------------------------------------------
# Cache location  (~/.cache/fllm/models/)
# ---------------------------------------------------------------------------

def default_cache_dir() -> Path:
    base = Path(os.environ.get("FLLM_CACHE", Path.home() / ".cache" / "fllm" / "models"))
    base.mkdir(parents=True, exist_ok=True)
    return base


# ---------------------------------------------------------------------------
# Downloader
# ---------------------------------------------------------------------------

class ModelDownloader:
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or default_cache_dir()

    def ensure(self, selection: ModelSelection) -> Path:
        """
        Return local path to the GGUF file, downloading if not cached.
        """
        dest = self.cache_dir / selection.gguf_repo.replace("/", "--") / selection.gguf_filename

        if dest.exists():
            print(f"  ✓ Using cached model: {dest}")
            return dest

        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"  ↓ Downloading {selection.family} {selection.size_label} "
              f"({selection.quant_method}, ~{selection.estimated_size_gb:.1f} GB) …")

        try:
            return self._download_hf_hub(selection, dest)
        except ImportError:
            print("  huggingface_hub not installed; falling back to urllib …", file=sys.stderr)
            return self._download_urllib(selection, dest)

    # ── HF Hub (preferred) ────────────────────────────────────────────────

    def _download_hf_hub(self, sel: ModelSelection, dest: Path) -> Path:
        from huggingface_hub import hf_hub_download

        local = hf_hub_download(
            repo_id=sel.gguf_repo,
            filename=sel.gguf_filename,
            local_dir=str(dest.parent),
            local_dir_use_symlinks=False,
        )
        return Path(local)

    # ── urllib fallback ───────────────────────────────────────────────────

    def _download_urllib(self, sel: ModelSelection, dest: Path) -> Path:
        url = (
            f"https://huggingface.co/{sel.gguf_repo}/resolve/main/{sel.gguf_filename}"
        )

        tmp = dest.with_suffix(".tmp")
        try:
            with urllib.request.urlopen(url) as response:
                total = int(response.headers.get("Content-Length", 0))
                downloaded = 0
                chunk = 1024 * 1024  # 1 MB

                with open(tmp, "wb") as f:
                    while True:
                        buf = response.read(chunk)
                        if not buf:
                            break
                        f.write(buf)
                        downloaded += len(buf)
                        self._progress(downloaded, total)

            tmp.rename(dest)
            print()   # newline after progress bar
        except Exception:
            if tmp.exists():
                tmp.unlink()
            raise

        return dest

    @staticmethod
    def _progress(done: int, total: int):
        if total:
            pct = done / total * 100
            bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
            mb_done = done / 1024 / 1024
            mb_total = total / 1024 / 1024
            print(f"\r  [{bar}] {pct:5.1f}%  {mb_done:.0f}/{mb_total:.0f} MB",
                  end="", flush=True)
        else:
            mb = done / 1024 / 1024
            print(f"\r  Downloaded {mb:.1f} MB …", end="", flush=True)
