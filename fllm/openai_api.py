"""
openai_api.py — OpenAI-compatible API server for FLLM.

Drop-in replacement for OpenAI SDK. Supports:
  - POST /v1/chat/completions  (streaming + non-streaming)
  - POST /v1/completions       (legacy text completions)
  - GET  /v1/models            (list available models)
  - GET  /v1/models/{model}    (model details)
  - POST /v1/embeddings        (if model supports it)

Usage:
  fllm serve <family> --port 8080
  
Then point any OpenAI SDK client at http://127.0.0.1:8080/v1:
  
  from openai import OpenAI
  client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="fllm")
  resp = client.chat.completions.create(
      model="local",
      messages=[{"role": "user", "content": "Hello!"}],
  )
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from .scout import HardwareProfile
from .selector import ModelSelection
from .metrics import MetricsTracker


def _generate_id(prefix: str = "chatcmpl") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:24]}"


def _timestamp() -> int:
    return int(time.time())


def create_openai_app(
    model_path: Path,
    hw: HardwareProfile,
    sel: ModelSelection,
    model_label: str,
    web: bool = False,
) -> Any:
    """Create a FastAPI app with full OpenAI-compatible endpoints."""
    try:
        from fastapi import FastAPI, Request, HTTPException
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import JSONResponse, StreamingResponse
    except ImportError:
        print("ERROR: fastapi not installed. pip install fastapi uvicorn", file=sys.stderr)
        sys.exit(1)

    try:
        from llama_cpp import Llama
    except ImportError:
        print("ERROR: llama-cpp-python not installed. pip install llama-cpp-python", file=sys.stderr)
        sys.exit(1)

    from .backends.llamacpp import _gpu_layers

    app = FastAPI(title="FLLM OpenAI-Compatible API", version="0.1.0")

    # CORS — allow all origins for local dev
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Load model ────────────────────────────────────────────────────────
    ngl = _gpu_layers(hw, sel)

    if os.environ.get("FLLM_CPU_ONLY", "").lower() in ("1", "true", "yes"):
        ngl = 0

    try:
        _fd = os.dup(2)
        os.dup2(os.open(os.devnull, os.O_WRONLY), 2)
        try:
            llm = Llama(
                model_path=str(model_path),
                n_ctx=sel.context_tokens,
                n_gpu_layers=999 if ngl < 0 else ngl,
                verbose=False,
                chat_format="chatml",
            )
        finally:
            os.dup2(_fd, 2)
            os.close(_fd)
    except (ValueError, RuntimeError):
        print("  ⚠  GPU init failed, retrying CPU-only …", file=sys.stderr)
        llm = Llama(
            model_path=str(model_path),
            n_ctx=sel.context_tokens,
            n_gpu_layers=0,
            verbose=False,
            chat_format="chatml",
        )

    model_id = sel.family.key
    created_at = _timestamp()

    # ── Metrics tracker ───────────────────────────────────────────────────
    metrics_dir = Path.home() / ".cache" / "fllm" / "metrics" / model_id
    tracker = MetricsTracker(
        model_label=model_label,
        persist_dir=metrics_dir,
    )

    # ── GET /v1/models ────────────────────────────────────────────────────

    @app.get("/v1/models")
    async def list_models():
        return {
            "object": "list",
            "data": [
                {
                    "id": model_id,
                    "object": "model",
                    "created": created_at,
                    "owned_by": "fllm",
                    "permission": [],
                    "root": model_id,
                    "parent": None,
                }
            ],
        }

    @app.get("/v1/models/{model_name}")
    async def get_model(model_name: str):
        # Accept any model name — we only have one loaded
        return {
            "id": model_id,
            "object": "model",
            "created": created_at,
            "owned_by": "fllm",
            "permission": [],
            "root": model_id,
            "parent": None,
        }

    # ── POST /v1/chat/completions ─────────────────────────────────────────

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        body = await request.json()
        messages = body.get("messages", [])
        stream = body.get("stream", False)
        max_tokens = body.get("max_tokens") or body.get("max_completion_tokens") or 2048
        temperature = body.get("temperature", 0.7)
        top_p = body.get("top_p", 1.0)
        stop = body.get("stop")
        presence_penalty = body.get("presence_penalty", 0.0)
        frequency_penalty = body.get("frequency_penalty", 0.0)
        user_model = body.get("model", model_id)

        # Build kwargs for llama-cpp-python's chat completion
        kwargs: dict = dict(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            stream=stream,
        )
        if stop:
            kwargs["stop"] = stop if isinstance(stop, list) else [stop]

        if stream:
            return StreamingResponse(
                _stream_chat(llm, kwargs, user_model, tracker),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        # Non-streaming — track metrics
        with tracker.track_request() as req:
            try:
                result = llm.create_chat_completion(**kwargs)
            except Exception as e:
                req.record(success=False, error=str(e))
                raise HTTPException(status_code=500, detail=str(e))

            usage = result.get("usage", {})
            req.record(
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                stream=False,
            )

        # Ensure proper OpenAI format
        completion_id = _generate_id()
        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": _timestamp(),
            "model": user_model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": result["choices"][0]["message"]["content"],
                    },
                    "finish_reason": result["choices"][0].get("finish_reason", "stop"),
                }
            ],
            "usage": result.get("usage", {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }),
        }

    # ── POST /v1/completions (legacy) ─────────────────────────────────────

    @app.post("/v1/completions")
    async def completions(request: Request):
        body = await request.json()
        prompt = body.get("prompt", "")
        stream = body.get("stream", False)
        max_tokens = body.get("max_tokens", 256)
        temperature = body.get("temperature", 0.7)
        top_p = body.get("top_p", 1.0)
        stop = body.get("stop")
        user_model = body.get("model", model_id)

        kwargs: dict = dict(
            prompt=prompt if isinstance(prompt, str) else prompt[0],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=stream,
        )
        if stop:
            kwargs["stop"] = stop if isinstance(stop, list) else [stop]

        if stream:
            return StreamingResponse(
                _stream_completion(llm, kwargs, user_model, tracker),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )

        # Non-streaming — track metrics
        with tracker.track_request() as req:
            try:
                result = llm.create_completion(**kwargs)
            except Exception as e:
                req.record(success=False, error=str(e))
                raise HTTPException(status_code=500, detail=str(e))

            usage = result.get("usage", {})
            req.record(
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                stream=False,
            )

        completion_id = _generate_id("cmpl")
        return {
            "id": completion_id,
            "object": "text_completion",
            "created": _timestamp(),
            "model": user_model,
            "choices": [
                {
                    "index": 0,
                    "text": result["choices"][0]["text"],
                    "finish_reason": result["choices"][0].get("finish_reason", "stop"),
                    "logprobs": None,
                }
            ],
            "usage": result.get("usage", {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }),
        }

    # ── POST /v1/embeddings ───────────────────────────────────────────────

    @app.post("/v1/embeddings")
    async def embeddings(request: Request):
        body = await request.json()
        inp = body.get("input", "")
        user_model = body.get("model", model_id)

        if isinstance(inp, str):
            inputs = [inp]
        elif isinstance(inp, list):
            inputs = inp
        else:
            raise HTTPException(status_code=400, detail="input must be string or list")

        try:
            data = []
            for i, text in enumerate(inputs):
                emb = llm.embed(text)
                data.append({
                    "object": "embedding",
                    "index": i,
                    "embedding": emb,
                })
            total_tokens = sum(len(t.split()) for t in inputs)  # rough estimate
            return {
                "object": "list",
                "data": data,
                "model": user_model,
                "usage": {
                    "prompt_tokens": total_tokens,
                    "total_tokens": total_tokens,
                },
            }
        except Exception as e:
            raise HTTPException(
                status_code=501,
                detail=f"Embeddings not supported by this model: {e}",
            )

    # ── Metrics endpoints ─────────────────────────────────────────────────

    @app.get("/v1/metrics")
    async def get_metrics():
        """Return aggregate usage metrics."""
        return tracker.summary_dict()

    @app.get("/v1/metrics/recent")
    async def get_recent_metrics(n: int = 10):
        """Return the last N request metrics."""
        return {"requests": tracker.recent(n)}

    @app.post("/v1/metrics/reset")
    async def reset_metrics():
        """Reset all metrics counters."""
        tracker.reset()
        return {"status": "ok", "message": "Metrics reset"}

    @app.post("/v1/metrics/save")
    async def save_metrics():
        """Persist current metrics to disk."""
        tracker.persist()
        return {"status": "ok", "path": str(metrics_dir)}

    # ── Health ────────────────────────────────────────────────────────────

    @app.get("/health")
    async def health():
        summary = tracker.summary_dict()
        return {
            "status": "ok",
            "model": model_label,
            "total_requests": summary["total_requests"],
            "uptime_seconds": summary["uptime_seconds"],
        }

    # ── API key bypass middleware ──────────────────────────────────────────
    # Accept any API key (or none) — local usage doesn't need auth

    @app.middleware("http")
    async def bypass_api_key(request: Request, call_next):
        # Strip Authorization header validation — accept anything
        return await call_next(request)

    # ── Web UI ────────────────────────────────────────────────────────────

    if web:
        from .webui import mount_webui
        mount_webui(app, model_label)

    return app


# ── Streaming helpers ─────────────────────────────────────────────────────

async def _stream_chat(llm, kwargs, model_name: str, tracker: MetricsTracker):
    """Yield SSE chunks for chat completions, with metrics tracking."""
    completion_id = _generate_id()
    created = _timestamp()
    completion_tokens = 0

    with tracker.track_request() as req:
        try:
            stream = llm.create_chat_completion(**kwargs)
            first_token = True
            for chunk in stream:
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                finish_reason = chunk.get("choices", [{}])[0].get("finish_reason")

                if first_token and delta.get("content"):
                    req.record_first_token()
                    first_token = False

                if delta.get("content"):
                    # Rough token estimate: ~4 chars per token
                    completion_tokens += max(1, len(delta["content"]) // 4)

                sse_data = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_name,
                    "choices": [
                        {
                            "index": 0,
                            "delta": delta,
                            "finish_reason": finish_reason,
                        }
                    ],
                }
                yield f"data: {json.dumps(sse_data)}\n\n"

            # Estimate prompt tokens from messages
            prompt_text = " ".join(
                m.get("content", "") for m in kwargs.get("messages", [])
            )
            prompt_tokens = max(1, len(prompt_text) // 4)

            req.record(
                prompt_tokens=prompt_tokens,
                completion_tokens=max(1, completion_tokens),
                stream=True,
            )

            yield "data: [DONE]\n\n"
        except Exception as e:
            req.record(success=False, error=str(e), stream=True)
            error_data = {"error": {"message": str(e), "type": "server_error"}}
            yield f"data: {json.dumps(error_data)}\n\n"
            yield "data: [DONE]\n\n"


async def _stream_completion(llm, kwargs, model_name: str, tracker: MetricsTracker):
    """Yield SSE chunks for legacy completions, with metrics tracking."""
    completion_id = _generate_id("cmpl")
    created = _timestamp()
    completion_tokens = 0

    with tracker.track_request() as req:
        try:
            stream = llm.create_completion(**kwargs)
            first_token = True
            for chunk in stream:
                text = chunk.get("choices", [{}])[0].get("text", "")
                finish_reason = chunk.get("choices", [{}])[0].get("finish_reason")

                if first_token and text:
                    req.record_first_token()
                    first_token = False

                if text:
                    completion_tokens += max(1, len(text) // 4)

                sse_data = {
                    "id": completion_id,
                    "object": "text_completion",
                    "created": created,
                    "model": model_name,
                    "choices": [
                        {
                            "index": 0,
                            "text": text,
                            "finish_reason": finish_reason,
                            "logprobs": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(sse_data)}\n\n"

            # Estimate prompt tokens
            prompt = kwargs.get("prompt", "")
            prompt_tokens = max(1, len(prompt) // 4)

            req.record(
                prompt_tokens=prompt_tokens,
                completion_tokens=max(1, completion_tokens),
                stream=True,
            )

            yield "data: [DONE]\n\n"
        except Exception as e:
            req.record(success=False, error=str(e), stream=True)
            error_data = {"error": {"message": str(e), "type": "server_error"}}
            yield f"data: {json.dumps(error_data)}\n\n"
            yield "data: [DONE]\n\n"
