"""selector.py — Maps HardwareProfile -> optimal ModelSelection via registry."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from .scout import HardwareProfile
from .registry import FamilyEntry, SizeEntry, resolve, list_families

@dataclass
class ModelSelection:
    family: FamilyEntry
    size: SizeEntry
    quant_method: str
    quant_bits: int
    gguf_repo: str
    gguf_filename: str
    hf_repo: str
    mlx_repo: Optional[str]
    estimated_size_gb: float
    fits_in_vram: bool
    needs_cpu_offload: bool
    context_tokens: int

_QUANT_BITS = {"Q2_K":2,"Q3_K_M":3,"Q4_K_M":4,"Q5_K_M":5,"Q6_K":6,"Q8_0":8,"fp16":16}
_BITS_TO_QUANT = {2:"Q2_K",3:"Q3_K_M",4:"Q4_K_M",5:"Q5_K_M",6:"Q6_K",8:"Q8_0",16:"fp16"}

def _est(params_b, bits): return round(params_b * bits / 8 * 1.1, 1)

class ModelSelector:
    def __init__(self, profile: HardwareProfile):
        self.hw = profile

    def select(self, family_hint: str) -> ModelSelection:
        entry = resolve(family_hint)
        if entry is None:
            known = ", ".join(f.key for f in list_families())
            raise ValueError(f"Unknown model family '{family_hint}'. Known: {known}")
        return self._pick(entry)

    def _pick(self, entry: FamilyEntry) -> ModelSelection:
        hw = self.hw
        budget = (hw.total_vram_gb * 0.85 if hw.tier == "A"
                  else hw.total_ram_gb * 0.70 if hw.tier == "B"
                  else hw.total_ram_gb * 0.55)
        bits = hw.optimal_quant_bits
        quant = _BITS_TO_QUANT.get(bits, "Q4_K_M")
        chosen = None
        for size in reversed(entry.sizes):
            q = size.quant_override or quant
            b = _QUANT_BITS.get(q, bits)
            if _est(size.params_b, b) <= budget:
                chosen, quant, bits = size, q, b
                break
        if chosen is None:
            chosen = entry.sizes[0]
            for fq, fb in [("Q3_K_M",3),("Q2_K",2)]:
                if _est(chosen.params_b, fb) <= budget:
                    quant, bits = fq, fb
                    break
            hw.warnings.append(f"Memory constrained — using {chosen.label} {quant}.")
        est = _est(chosen.params_b, bits)
        fits = hw.tier == "A" and est <= hw.total_vram_gb * 0.90
        label = chosen.label
        def fill(t):
            return (t or "").replace("{size}", label).replace("{name}", entry.display).replace("{quant}", quant)
        return ModelSelection(
            family=entry, size=chosen, quant_method=quant, quant_bits=bits,
            gguf_repo=fill(entry.gguf_repo_template),
            gguf_filename=fill(entry.gguf_file_template),
            hf_repo=fill(entry.hf_repo_template),
            mlx_repo=fill(entry.mlx_repo_template) if entry.mlx_repo_template else None,
            estimated_size_gb=est, fits_in_vram=fits,
            needs_cpu_offload=hw.tier == "A" and not fits,
            context_tokens=hw.max_context_tokens,
        )
