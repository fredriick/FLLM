"""
backends/airllm.py — AirLLM backend for low-VRAM inference.

AirLLM optimizes memory usage by splitting model layers and using
block-wise quantization, allowing 70B models on 4GB VRAM.
"""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path
from typing import Optional

from ..scout import HardwareProfile
from ..selector import ModelSelection
from ..registry import FamilyEntry
from ..speculative import SpecConfig


class AirLLMBackend:
    name = "airllm"

    def __init__(
        self,
        hw: HardwareProfile,
        sel: ModelSelection,
        family: Optional[FamilyEntry] = None,
        compression: Optional[str] = None,
    ):
        self.hw = hw
        self.sel = sel
        self.family = family
        self.compression = compression or "4bit"
        self._model = None
        self._tokenizer = None

    def is_available(self) -> bool:
        try:
            import airllm
            import airllm.airllm_llama
            return True
        except ImportError:
            return False

    def install_hint(self) -> str:
        return (
            "AirLLM is not installed.\n"
            "  pip install airllm\n"
            "  # For compression (recommended):\n"
            "  pip install bitsandbytes"
        )

    def launch_server(self, model_path: Path, port: int = 8080, spec: Optional[SpecConfig] = None):
        from airllm import AutoModel
        import uvicorn
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel
        from typing import List, Dict, Any
        import torch

        repo_id = self._get_repo_id(model_path)

        print(f"\n  Loading AirLLM model: {repo_id}")
        print(f"  Compression: {self.compression}")

        try:
            self._model = AutoModel.from_pretrained(
                repo_id,
                compression=self.compression,
            )
            self._tokenizer = self._model.tokenizer
        except Exception as e:
            print(f"ERROR loading model: {e}", file=sys.stderr)
            sys.exit(1)

        app = FastAPI(title="AirLLM OpenAI-Compatible API")

        class ChatMessage(BaseModel):
            role: str
            content: str

        MAX_TOKENS_CAP = 2048

        class ChatCompletionRequest(BaseModel):
            model: str
            messages: List[ChatMessage]
            max_tokens: int = 512
            temperature: float = 0.7
            stream: bool = False

        @app.post("/v1/chat/completions")
        async def chat_completions(request: ChatCompletionRequest):
            if request.stream:
                raise HTTPException(status_code=400, detail="Streaming not supported")

            capped_max_tokens = min(request.max_tokens, MAX_TOKENS_CAP)
            prompt = self._build_prompt(request.messages)

            input_ids = None
            generation_output = None
            try:
                input_tokens = self._tokenizer(
                    [prompt],
                    return_tensors="pt",
                    return_attention_mask=False,
                    truncation=True,
                    max_length=self.sel.context_tokens,
                    padding=False,
                )

                if torch.cuda.is_available():
                    input_ids = input_tokens["input_ids"].cuda()
                else:
                    input_ids = input_tokens["input_ids"]

                generation_output = self._model.generate(
                    input_ids,
                    max_new_tokens=capped_max_tokens,
                    use_cache=False,
                    return_dict_in_generate=True,
                )

                output_text = self._tokenizer.decode(
                    generation_output.sequences[0]
                )

                response = {
                    "id": "chatcmpl-airllm",
                    "object": "chat.completion",
                    "created": 1234567890,
                    "model": request.model,
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": output_text,
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": input_ids.shape[1],
                        "completion_tokens": capped_max_tokens,
                        "total_tokens": input_ids.shape[1] + capped_max_tokens,
                    },
                }

                return response
            finally:
                del generation_output, input_ids
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        @app.get("/v1/models")
        async def list_models():
            return {
                "object": "list",
                "data": [
                    {
                        "id": self.sel.family.key,
                        "object": "model",
                        "owned_by": "airllm",
                        "permission": [],
                    }
                ],
            }

        print(f"\n  ▶  AirLLM server  →  http://127.0.0.1:{port}/v1")
        print(f"     Model: {self.sel.family.display} {self.sel.size.label}")
        print(f"     Compression: {self.compression}")
        print(f"     Context: {self.sel.context_tokens:,} tokens\n")

        uvicorn.run(app, host="127.0.0.1", port=port)

    def _get_repo_id(self, model_path: Path) -> str:
        if model_path.exists() and model_path.is_dir():
            return str(model_path)
        if self.sel.hf_repo:
            return self.sel.hf_repo
        if self.sel.gguf_repo:
            return self.sel.gguf_repo
        return f"meta-llama/Llama-2-7b-hf"

    def _build_prompt(self, messages: list) -> str:
        prompt = ""
        for msg in messages:
            if msg.role == "system":
                prompt += f"System: {msg.content}\n"
            elif msg.role == "user":
                prompt += f"User: {msg.content}\n"
            elif msg.role == "assistant":
                prompt += f"Assistant: {msg.content}\n"
        prompt += "Assistant:"
        return prompt

    def launch_interactive(self, model_path: Path, session, renderer, spec: Optional[SpecConfig] = None):
        print("AirLLM interactive mode not yet implemented.")
        print("Use --mode server instead.")
        sys.exit(1)
