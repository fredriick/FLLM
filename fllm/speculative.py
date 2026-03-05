"""
speculative.py — Speculative decoding support.

Speculative decoding uses a small "draft" model to propose tokens,
which the large "target" model then verifies in parallel.
Speedup: typically 2–4× on long generations.

Works with llama.cpp (--draft-model flag) and vLLM (ngram / draft model).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .registry import FamilyEntry
from .scout import HardwareProfile


@dataclass
class SpecConfig:
    enabled: bool
    draft_model_path: Optional[Path]
    n_draft_tokens: int          # tokens to speculate per step (llama.cpp: --draft)
    reason_disabled: str = ""    # why it's off, if disabled


class SpeculativeManager:
    """
    Decides whether speculative decoding is viable for the current
    hardware + model combination, and downloads the draft model if so.
    """

    # Minimum free VRAM (GB) required to hold a draft model alongside the main model
    VRAM_HEADROOM_GB = 2.0

    def __init__(self, hw: HardwareProfile, family: FamilyEntry, cache_dir: Path):
        self.hw = hw
        self.family = family
        self.cache_dir = cache_dir

    def evaluate(self) -> SpecConfig:
        """Return a SpecConfig describing whether/how to use spec decoding."""
        if not self._hardware_supports():
            return SpecConfig(
                enabled=False,
                draft_model_path=None,
                n_draft_tokens=0,
                reason_disabled=self._disabled_reason(),
            )

        if not self.family.draft_gguf_repo:
            return SpecConfig(
                enabled=False,
                draft_model_path=None,
                n_draft_tokens=0,
                reason_disabled=f"No draft model defined for {self.family.display}.",
            )

        draft_path = self._draft_path()
        if not draft_path.exists():
            print(f"  ↓ Downloading draft model for speculative decoding …")
            draft_path = self._download_draft()

        n_draft = self._draft_tokens()
        print(f"  ⚡ Speculative decoding enabled  "
              f"(draft={self.family.draft_gguf_file}, n_draft={n_draft})")

        return SpecConfig(
            enabled=True,
            draft_model_path=draft_path,
            n_draft_tokens=n_draft,
        )

    # ── Internal ─────────────────────────────────────────────────────────────

    def _hardware_supports(self) -> bool:
        """Spec decoding makes sense only when GPU/unified memory has headroom."""
        if self.hw.tier == "A":
            return self.hw.total_vram_gb >= (4.0 + self.VRAM_HEADROOM_GB)
        if self.hw.tier == "B":
            return self.hw.total_ram_gb >= 16.0
        # Tier C (CPU): spec decoding rarely helps because draft + verify are both slow
        return False

    def _disabled_reason(self) -> str:
        if self.hw.tier == "C":
            return "CPU-only: speculative decoding overhead exceeds benefit."
        if self.hw.tier == "A" and self.hw.total_vram_gb < 4.0 + self.VRAM_HEADROOM_GB:
            return f"Insufficient VRAM headroom (need >{4 + self.VRAM_HEADROOM_GB:.0f} GB free)."
        return "Hardware does not meet requirements."

    def _draft_tokens(self) -> int:
        """
        Tokens to speculate per step.
        More VRAM → can afford more draft tokens → higher potential speedup.
        """
        if self.hw.tier == "B":
            return 5
        vram = self.hw.total_vram_gb
        if vram >= 48:
            return 8
        elif vram >= 24:
            return 6
        else:
            return 4

    def _draft_path(self) -> Path:
        repo_slug = self.family.draft_gguf_repo.replace("/", "--")
        return self.cache_dir / repo_slug / self.family.draft_gguf_file

    def _download_draft(self) -> Path:
        dest = self._draft_path()
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            from huggingface_hub import hf_hub_download
            local = hf_hub_download(
                repo_id=self.family.draft_gguf_repo,
                filename=self.family.draft_gguf_file,
                local_dir=str(dest.parent),
            )
            return Path(local)
        except ImportError:
            import urllib.request
            url = (
                f"https://huggingface.co/{self.family.draft_gguf_repo}"
                f"/resolve/main/{self.family.draft_gguf_file}"
            )
            tmp = dest.with_suffix(".tmp")
            try:
                urllib.request.urlretrieve(url, str(tmp))
                tmp.rename(dest)
            except Exception as e:
                print(f"  ✗ Draft model download failed: {e}", file=sys.stderr)
                if tmp.exists():
                    tmp.unlink()
                return dest   # Caller checks .exists()
            return dest
