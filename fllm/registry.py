"""
registry.py — Central model registry.

All model families, sizes, quantization tiers, chat templates,
and draft models for speculative decoding live here.

Add a new family by adding an entry to REGISTRY — no other code changes needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Chat template definitions
# ---------------------------------------------------------------------------
# Each template is a callable: (system_prompt, history) -> str
# history = [{"role": "user"|"assistant", "content": str}, ...]

def _chatml(system: str, history: list) -> str:
    """ChatML format used by Qwen, Mistral-instruct variants, etc."""
    parts = []
    if system:
        parts.append(f"<|im_start|>system\n{system}<|im_end|>")
    for msg in history:
        role = msg["role"]
        content = msg["content"]
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


def _llama3(system: str, history: list) -> str:
    """Llama 3 instruct format."""
    bos = "<|begin_of_text|>"
    parts = [bos]
    if system:
        parts.append(
            f"<|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>"
        )
    for msg in history:
        role = msg["role"]
        content = msg["content"]
        parts.append(
            f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
        )
    parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
    return "".join(parts)


def _gemma(system: str, history: list) -> str:
    """Gemma instruct format."""
    parts = []
    for i, msg in enumerate(history):
        if msg["role"] == "user":
            content = msg["content"]
            if i == 0 and system:
                content = f"{system}\n\n{content}"
            parts.append(f"<start_of_turn>user\n{content}<end_of_turn>")
        else:
            parts.append(f"<start_of_turn>model\n{msg['content']}<end_of_turn>")
    parts.append("<start_of_turn>model\n")
    return "\n".join(parts)


def _phi(system: str, history: list) -> str:
    """Phi-3 / Phi-4 instruct format."""
    parts = []
    if system:
        parts.append(f"<|system|>\n{system}<|end|>")
    for msg in history:
        tag = "<|user|>" if msg["role"] == "user" else "<|assistant|>"
        parts.append(f"{tag}\n{msg['content']}<|end|>")
    parts.append("<|assistant|>\n")
    return "\n".join(parts)


CHAT_TEMPLATES = {
    "chatml":  _chatml,
    "llama3":  _llama3,
    "gemma":   _gemma,
    "phi":     _phi,
}


# ---------------------------------------------------------------------------
# Model size entry
# ---------------------------------------------------------------------------

@dataclass
class SizeEntry:
    label: str           # "7B"
    params_b: float      # 7.0
    # Override quant per size if needed (None = use hardware default)
    quant_override: Optional[str] = None
    # Typical layer count (for GPU offload calculation)
    n_layers: int = 32


# ---------------------------------------------------------------------------
# Family entry
# ---------------------------------------------------------------------------

@dataclass
class FamilyEntry:
    key: str                          # internal lookup key, e.g. "qwen"
    display: str                      # human name, e.g. "Qwen2.5"
    sizes: List[SizeEntry]
    chat_template: str                # key into CHAT_TEMPLATES
    default_system: str               # default system prompt

    # GGUF source (bartowski / TheBloke / etc.)
    gguf_repo_template: str           # "{hf_user}/{name}-{size}-Instruct-GGUF"
    gguf_file_template: str           # "{name}-{size}-Instruct-{quant}.gguf"

    # HuggingFace base repo (for vLLM / fp16)
    hf_repo_template: str             # "Qwen/{name}-{size}-Instruct"

    # MLX community repo (for Apple Silicon)
    mlx_repo_template: Optional[str] = None  # "mlx-community/{name}-{size}-Instruct-4bit"

    # Draft model for speculative decoding
    draft_gguf_repo: Optional[str] = None
    draft_gguf_file: Optional[str] = None

    # Aliases that resolve to this family
    aliases: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

REGISTRY: Dict[str, FamilyEntry] = {

    "qwen": FamilyEntry(
        key="qwen",
        display="Qwen2.5",
        aliases=["qwen2", "qwen25", "qwen2.5"],
        chat_template="chatml",
        default_system="You are Qwen, a helpful assistant.",
        sizes=[
            SizeEntry("0.5B", 0.5,  n_layers=28),
            SizeEntry("1.5B", 1.5,  n_layers=28),
            SizeEntry("3B",   3.0,  n_layers=36),
            SizeEntry("7B",   7.0,  n_layers=32),
            SizeEntry("14B",  14.0, n_layers=48),
            SizeEntry("32B",  32.0, n_layers=64),
            SizeEntry("72B",  72.0, n_layers=80),
        ],
        gguf_repo_template="bartowski/Qwen2.5-{size}-Instruct-GGUF",
        gguf_file_template="Qwen2.5-{size}-Instruct-{quant}.gguf",
        hf_repo_template="Qwen/Qwen2.5-{size}-Instruct",
        mlx_repo_template="mlx-community/Qwen2.5-{size}-Instruct-4bit",
        draft_gguf_repo="bartowski/Qwen2.5-0.5B-Instruct-GGUF",
        draft_gguf_file="Qwen2.5-0.5B-Instruct-Q4_K_M.gguf",
    ),

    "llama3": FamilyEntry(
        key="llama3",
        display="Llama-3.2",
        aliases=["llama3.2", "llama-3.2"],
        chat_template="llama3",
        default_system="You are a helpful, honest, and harmless assistant.",
        sizes=[
            SizeEntry("1B",  1.0,  n_layers=16),
            SizeEntry("3B",  3.0,  n_layers=28),
            SizeEntry("8B",  8.0,  n_layers=32),
            SizeEntry("11B", 11.0, n_layers=40),
        ],
        gguf_repo_template="bartowski/Llama-3.2-{size}-Instruct-GGUF",
        gguf_file_template="Llama-3.2-{size}-Instruct-{quant}.gguf",
        hf_repo_template="meta-llama/Llama-3.2-{size}-Instruct",
        mlx_repo_template="mlx-community/Llama-3.2-{size}-Instruct-4bit",
        draft_gguf_repo="bartowski/Llama-3.2-1B-Instruct-GGUF",
        draft_gguf_file="Llama-3.2-1B-Instruct-Q4_K_M.gguf",
    ),

    "llama3.1": FamilyEntry(
        key="llama3.1",
        display="Llama-3.1",
        aliases=["llama3.1", "llama-3.1-instruct"],
        chat_template="llama3",
        default_system="You are a helpful, honest, and harmless assistant.",
        sizes=[
            SizeEntry("8B",  8.0,  n_layers=32),
            SizeEntry("70B", 70.0, n_layers=80),
        ],
        gguf_repo_template="bartowski/Meta-Llama-3.1-{size}-Instruct-GGUF",
        gguf_file_template="Meta-Llama-3.1-{size}-Instruct-{quant}.gguf",
        hf_repo_template="meta-llama/Meta-Llama-3.1-{size}-Instruct",
        mlx_repo_template="mlx-community/Meta-Llama-3.1-{size}-Instruct-4bit",
        draft_gguf_repo="bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        draft_gguf_file="Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
    ),

    "mistral": FamilyEntry(
        key="mistral",
        display="Mistral-7B",
        aliases=["mistral7b", "mistralai"],
        chat_template="chatml",
        default_system="You are a helpful assistant.",
        sizes=[
            SizeEntry("7B",   7.0,  n_layers=32),
            SizeEntry("8x7B", 56.0, n_layers=32),   # Mixtral MoE
        ],
        gguf_repo_template="bartowski/Mistral-7B-Instruct-v0.3-GGUF",
        gguf_file_template="Mistral-7B-Instruct-v0.3-{quant}.gguf",
        hf_repo_template="mistralai/Mistral-7B-Instruct-v0.3",
        mlx_repo_template="mlx-community/Mistral-7B-Instruct-v0.3-4bit",
        draft_gguf_repo=None,   # No good 1B Mistral draft available
        draft_gguf_file=None,
    ),

    "phi4": FamilyEntry(
        key="phi4",
        display="Phi-4",
        aliases=["phi", "phi-4", "microsoft-phi"],
        chat_template="phi",
        default_system="You are a helpful AI assistant.",
        sizes=[
            SizeEntry("14B", 14.0, n_layers=40),
        ],
        gguf_repo_template="bartowski/phi-4-GGUF",
        gguf_file_template="phi-4-{quant}.gguf",
        hf_repo_template="microsoft/phi-4",
        mlx_repo_template="mlx-community/phi-4-4bit",
        draft_gguf_repo=None,
        draft_gguf_file=None,
    ),

    "gemma3": FamilyEntry(
        key="gemma3",
        display="Gemma-3",
        aliases=["gemma", "google-gemma"],
        chat_template="gemma",
        default_system="",
        sizes=[
            SizeEntry("1B",  1.0,  n_layers=18),
            SizeEntry("4B",  4.0,  n_layers=34),
            SizeEntry("12B", 12.0, n_layers=46),
            SizeEntry("27B", 27.0, n_layers=62),
        ],
        gguf_repo_template="bartowski/gemma-3-{size}it-GGUF",
        gguf_file_template="gemma-3-{size}it-{quant}.gguf",
        hf_repo_template="google/gemma-3-{size}it",
        mlx_repo_template="mlx-community/gemma-3-{size}it-4bit",
        draft_gguf_repo="bartowski/gemma-3-1Bit-GGUF",
        draft_gguf_file="gemma-3-1Bit-Q4_K_M.gguf",
    ),

    "deepseek": FamilyEntry(
        key="deepseek",
        display="DeepSeek-R1",
        aliases=["deepseek-r1", "deepseek-v3"],
        chat_template="chatml",
        default_system="You are a helpful assistant. Think step by step.",
        sizes=[
            SizeEntry("1.5B", 1.5,  n_layers=28),
            SizeEntry("7B",   7.0,  n_layers=30),
            SizeEntry("8B",   8.0,  n_layers=32),
            SizeEntry("14B",  14.0, n_layers=48),
            SizeEntry("32B",  32.0, n_layers=64),
            SizeEntry("70B",  70.0, n_layers=80),
        ],
        gguf_repo_template="bartowski/DeepSeek-R1-Distill-Qwen-{size}-GGUF",
        gguf_file_template="DeepSeek-R1-Distill-Qwen-{size}-{quant}.gguf",
        hf_repo_template="deepseek-ai/DeepSeek-R1-Distill-Qwen-{size}",
        mlx_repo_template="mlx-community/DeepSeek-R1-Distill-Qwen-{size}-4bit",
        draft_gguf_repo="bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF",
        draft_gguf_file="DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf",
    ),

    "smollm": FamilyEntry(
        key="smollm",
        display="SmolLM2",
        aliases=["smol", "smollm2"],
        chat_template="chatml",
        default_system="You are a helpful AI assistant.",
        sizes=[
            SizeEntry("135M", 0.135, n_layers=12),
            SizeEntry("360M", 0.36,  n_layers=16),
            SizeEntry("1.7B", 1.7,   n_layers=24),
        ],
        gguf_repo_template="bartowski/SmolLM2-{size}-Instruct-GGUF",
        gguf_file_template="SmolLM2-{size}-Instruct-{quant}.gguf",
        hf_repo_template="HuggingFaceTB/SmolLM2-{size}-Instruct",
        mlx_repo_template=None,
        draft_gguf_repo=None,
        draft_gguf_file=None,
    ),
}

# Build reverse alias map
_ALIAS_MAP: Dict[str, str] = {}
for _key, _entry in REGISTRY.items():
    _ALIAS_MAP[_key] = _key
    for _alias in _entry.aliases:
        _ALIAS_MAP[_alias.lower().replace("-", "").replace("_", "").replace(".", "").replace(" ", "")] = _key


def resolve(hint: str) -> Optional[FamilyEntry]:
    """
    Look up a FamilyEntry by any alias, case-insensitive.
    Returns None if not found.
    """
    key = hint.lower().replace("-", "").replace("_", "").replace(".", "").replace(" ", "")
    resolved = _ALIAS_MAP.get(key)
    if resolved:
        return REGISTRY[resolved]
    # Substring fallback
    for alias_key, family_key in _ALIAS_MAP.items():
        if key in alias_key or alias_key in key:
            return REGISTRY[family_key]
    return None


def list_families() -> List[FamilyEntry]:
    return list(REGISTRY.values())
