"""selector.py — Maps HardwareProfile -> optimal ModelSelection via registry."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from .scout import HardwareProfile
from .registry import FamilyEntry, SizeEntry, resolve, list_families

# KV cache estimate for llama.cpp with PagedAttention
# Rough rule: ~2MB per 1B parameters per 1K context tokens (conservative)
_KV_CACHE_MB_PER_1B_PER_1K = 2.0  # MB per 1B params per 1K tokens

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
    kv_cache_overhead_gb: float
    fits_in_vram: bool
    needs_cpu_offload: bool
    context_tokens: int

_QUANT_BITS = {"Q2_K":2,"Q3_K_M":3,"Q4_K_M":4,"Q5_K_M":5,"Q6_K":6,"Q8_0":8,"fp16":16}
_BITS_TO_QUANT = {2:"Q2_K",3:"Q3_K_M",4:"Q4_K_M",5:"Q5_K_M",6:"Q6_K",8:"Q8_0",16:"fp16"}

def _est(params_b, bits):
    """Estimate model weights size in GB."""
    return round(params_b * bits / 8 * 1.1, 1)

def _est_kv_cache(params_b: float, context_tokens: int) -> float:
    """Estimate KV cache memory in GB based on model size and context length."""
    mb_per_token = params_b * _KV_CACHE_MB_PER_1B_PER_1K * (context_tokens / 1024)
    total_mb = mb_per_token * 32  # Assume ~32 layers for typical models
    return round(total_mb / 1024, 2)  # Convert to GB

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
        context = hw.max_context_tokens

        # Calculate KV cache overhead for target context length
        # Start with smallest size to calculate minimum KV cache
        smallest_size = entry.sizes[0]
        min_kv_cache = _est_kv_cache(smallest_size.params_b, context)

        # For larger models, KV cache is proportionally larger
        # Use more conservative budgets to prevent OOM
        budget = (hw.total_vram_gb * 0.60 if hw.tier == "A"  # Leave 40% for KV cache + overhead
                  else hw.total_ram_gb * 0.40 if hw.tier == "B"  # Leave 60% for KV cache + overhead
                  else hw.total_ram_gb * 0.30)  # Leave 70% for KV cache + overhead

        bits = hw.optimal_quant_bits
        quant = _BITS_TO_QUANT.get(bits, "Q4_K_M")
        chosen = None

        for size in reversed(entry.sizes):
            model_size = _est(size.params_b, bits)
            kv_cache = _est_kv_cache(size.params_b, context)
            total_needed = model_size + kv_cache

            if total_needed <= budget:
                chosen = size
                break

        if chosen is None:
            # Try reducing context length if model doesn't fit
            for reduced_context in [2048, 1024]:
                for size in reversed(entry.sizes):
                    model_size = _est(size.params_b, bits)
                    kv_cache = _est_kv_cache(size.params_b, reduced_context)
                    total_needed = model_size + kv_cache

                    if total_needed <= budget:
                        chosen = size
                        hw.warnings.append(
                            f"Reduced context to {reduced_context} tokens to fit {size.label} in memory."
                        )
                        context = reduced_context
                        break
                if chosen:
                    break

        if chosen is None:
            # Last resort: use smallest model with heaviest quant
            chosen = entry.sizes[0]
            for fq, fb in [("Q3_K_M",3),("Q2_K",2)]:
                model_size = _est(chosen.params_b, fb)
                kv_cache = _est_kv_cache(chosen.params_b, 512)  # Reduce context heavily
                total_needed = model_size + kv_cache
                if total_needed <= budget:
                    quant, bits = fq, fb
                    context = 512
                    break
            hw.warnings.append(f"Memory constrained — using {chosen.label} {quant} with {context} context.")

        est = _est(chosen.params_b, bits)
        kv_overhead = _est_kv_cache(chosen.params_b, context)
        fits = hw.tier == "A" and (est + kv_overhead) <= hw.total_vram_gb * 0.60

        # Warn if context is too large for unified memory (Tier B)
        if hw.tier == "B" and context >= 4096 and (est + kv_overhead) > hw.total_ram_gb * 0.4:
            hw.warnings.append(
                f"High context ({context} tokens) may cause OOM on {hw.total_ram_gb:.0f}GB unified memory. "
                f"Consider using --tier C or a smaller model."
            )

        label = chosen.label
        def fill(t):
            return (t or "").replace("{size}", label).replace("{name}", entry.display).replace("{quant}", quant)
        return ModelSelection(
            family=entry, size=chosen, quant_method=quant, quant_bits=bits,
            gguf_repo=fill(entry.gguf_repo_template),
            gguf_filename=fill(entry.gguf_file_template),
            hf_repo=fill(entry.hf_repo_template),
            mlx_repo=fill(entry.mlx_repo_template) if entry.mlx_repo_template else None,
            estimated_size_gb=est, kv_cache_overhead_gb=kv_overhead,
            fits_in_vram=fits,
            needs_cpu_offload=hw.tier == "A" and not fits,
            context_tokens=context,
        )
