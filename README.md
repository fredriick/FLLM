# FLLM — Universal LLM Runner

Auto-detects your hardware and runs the best local LLM you can fit, with zero manual configuration.

## How it works

```text
fllm run qwen
     │
     ├─ Detects hardware (CPU, RAM, GPU, storage, unified memory)
     ├─ Classifies tier: A (GPU) | B (unified) | C (CPU) | AirLLM (low VRAM)
     ├─ Picks largest model + optimal quantization that fits in memory
     ├─ Downloads GGUF from HuggingFace (cached in ~/.cache/fllm/)
     └─ Launches the right backend (llama.cpp / vLLM / mlx-lm / AirLLM)
```

## Install

```bash
git clone https://github.com/fredriick/FLLM.git
cd fllm

# CPU only (works everywhere)
pip install -e ".[cpu]"

# NVIDIA/AMD GPU
pip install -e ".[gpu]"

# Apple Silicon
pip install -e ".[apple]"

# AirLLM (low VRAM - run 70B on 4GB GPU)
pip install -e ".[air]"
```

## Commands

```bash
# Show hardware profile as JSON
fllm info

# List supported model families
fllm list

# Run a model (auto-everything)
fllm run qwen
fllm run llama3
fllm run mistral
fllm run phi4
fllm run gemma3

# Interactive chat instead of API server
fllm run qwen --mode interactive

# Custom port
fllm run llama3 --port 9000

# Force a tier (useful for testing)
fllm run qwen --tier C

# Prevent fallback below Q4 quality
fllm run qwen --min-quant Q4_K_M

# Use AirLLM backend (low VRAM - 70B on 4GB GPU)
fllm run llama3 --backend airllm --compression 4bit

# AirLLM with 8-bit compression
fllm run qwen --backend airllm --compression 8bit

# Use a local GGUF you already have
fllm run qwen --model-path ~/models/Qwen2.5-7B-Q4_K_M.gguf

# Verbose hardware detection logs
fllm run qwen --verbose

# Benchmark token throughput
fllm bench qwen
fllm bench llama3 --output results.json
```

## Tiers explained

| Tier | Hardware | Backend | What it does |
|------|----------|---------|--------------|
| **A** | NVIDIA GPU ≥8GB VRAM / AMD ROCm (experimental) | vLLM or llama.cpp (AMD: llama.cpp only) | Kernel fusion, continuous batching, PagedAttention |
| **B** | Apple Silicon | llama.cpp (Metal) | mlx-lm experimental (--backend mlx) |
| **C** | CPU only | llama.cpp | AVX-512/AVX2/NEON vectorised inference |
| **AirLLM** | Any GPU with ≥4GB VRAM | AirLLM | Layer-wise loading, run 70B on 4GB, 405B on 8GB |

## Fallback chain

If the selected backend isn't available, fllm automatically falls back:

```
vLLM → llama.cpp
mlx-lm → llama.cpp
AirLLM → llama.cpp
```

If the model doesn't fit in memory, fllm automatically steps down:
- Smaller model size (e.g., 70B → 8B)
- Heavier quantization (e.g., Q5 → Q3 → Q2)

To prevent quality degradation, use `--min-quant`:
```bash
fllm run qwen --min-quant Q4_K_M
```

To force a specific backend:
```bash
fllm run qwen --backend llama.cpp
```

## API server

When running in `server` mode (default), fllm exposes an **OpenAI-compatible API**:

```
http://127.0.0.1:8080/v1/chat/completions
http://127.0.0.1:8080/v1/models
```

Works with any tool that supports the OpenAI SDK:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="unused")
resp = client.chat.completions.create(
    model="local",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(resp.choices[0].message.content)
```

## Benchmark

Run `fllm bench <family>` to measure token throughput. Output includes:

- **tokens/sec** — throughput measurement
- **model** — model name and size
- **quantization** — quantization method used
- **backend** — backend name
- **hardware tier** — detected tier

Save to JSON with `--output results.json`:

```json
{
  "model": "Qwen2.5 7B Q4_K_M",
  "backend": "llama.cpp",
  "tier": "C",
  "tokens_per_second": 15.2,
  "quantization": "Q4_K_M"
}
```

## Supported models

> **Tip:** Run `fllm list` to see the most up-to-date model list.

| Key | Family | Sizes |
|-----|--------|-------|
| `qwen` | Qwen2.5 Instruct | 0.5B, 1.5B, 3B, 7B, 14B, 32B, 72B |
| `llama3` | Llama 3.2 Instruct | 1B, 3B, 8B, 11B |
| `llama3.1` | Llama 3.1 Instruct | 8B, 70B |
| `mistral` | Mistral 7B / Mixtral 8x7B | 7B, 56B |
| `phi4` | Microsoft Phi-4 | 14B |
| `gemma3` | Google Gemma 3 | 1B, 4B, 12B, 27B |

## Platform support

| Platform | Supported Backends |
|----------|-------------------|
| Linux + NVIDIA | vLLM, llama.cpp, AirLLM |
| Linux + AMD (ROCm) | llama.cpp only (experimental) |
| macOS Apple Silicon | mlx-lm, llama.cpp, AirLLM |
| macOS Intel | llama.cpp |
| Windows | llama.cpp only |

## Hardware requirements

| Tier | Minimum RAM | Recommended |
|------|------------|-------------|
| A (GPU) | 8 GB VRAM | 24+ GB VRAM |
| B (unified) | 16 GB RAM | 64 GB RAM |
| C (CPU) | 8 GB RAM | 32 GB + AVX2 |
| AirLLM | 4 GB VRAM | 8+ GB VRAM |

## Architecture

```
fllm/
├── scout.py       Hardware detection (CPU, RAM, GPU, storage)
├── selector.py    Model size + quantization picker
├── downloader.py  HuggingFace GGUF fetcher with resume + progress
├── launcher.py    Orchestrator
├── backends/
│   ├── llamacpp.py   Tier B + C, GPU offload
│   ├── vllm.py       Tier A (NVIDIA)
│   ├── mlx.py        Tier B Apple Silicon (mlx-lm)
│   └── airllm.py     Low VRAM (70B on 4GB, 405B on 8GB)
└── cli.py         Terminal interface
```
