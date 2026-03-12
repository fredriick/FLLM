"""
remote.py — Remote inference client for FLLM.

Connect to a remote FLLM server (or any OpenAI-compatible endpoint)
and run interactive chat or one-shot queries.

Usage:
  fllm remote http://192.168.1.100:8080                # interactive chat
  fllm remote http://192.168.1.100:8080 --ask "Hello"  # one-shot query
  fllm remote http://192.168.1.100:8080 --stream        # streaming mode
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request
import urllib.error
from typing import Dict, Iterator, List, Optional


# ---------------------------------------------------------------------------
# Remote client
# ---------------------------------------------------------------------------

class RemoteClient:
    """Client for connecting to a remote FLLM / OpenAI-compatible server."""

    def __init__(
        self,
        base_url: str,
        api_key: str = "fllm",
        model: Optional[str] = None,
        timeout: int = 120,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self._server_model: Optional[str] = None

    # ── Connection check ──────────────────────────────────────────────────

    def check_connection(self) -> Dict:
        """Check if the remote server is reachable. Returns health info."""
        try:
            data = self._get("/health")
            return data
        except Exception:
            # Try /v1/models as fallback
            try:
                data = self._get("/v1/models")
                return {"status": "ok", "models": data}
            except Exception as e:
                from .errors import FLLMError
                raise FLLMError(
                    f"Cannot connect to {self.base_url}",
                    hint=(
                        "Make sure the server is running:\n"
                        f"  fllm serve <model> --port <port>\n"
                        f"  Target: {self.base_url}"
                    ),
                )

    def get_model(self) -> str:
        """Get the model name from the server."""
        if self.model:
            return self.model

        if self._server_model:
            return self._server_model

        try:
            data = self._get("/v1/models")
            models = data.get("data", [])
            if models:
                self._server_model = models[0].get("id", "unknown")
                return self._server_model
        except Exception:
            pass

        return "remote"

    def get_metrics(self) -> Optional[Dict]:
        """Fetch metrics from the remote server."""
        try:
            return self._get("/v1/metrics")
        except Exception:
            return None

    # ── Chat completion ───────────────────────────────────────────────────

    def chat(
        self,
        messages: List[Dict],
        max_tokens: int = 2048,
        temperature: float = 0.7,
        stream: bool = False,
    ) -> str:
        """Send a chat completion request. Returns the response text."""
        body = {
            "model": self.get_model(),
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
        }

        if stream:
            return self._stream_chat(body)

        data = self._post("/v1/chat/completions", body)
        choices = data.get("choices", [])
        if choices:
            return choices[0].get("message", {}).get("content", "")
        return ""

    def _stream_chat(self, body: Dict) -> str:
        """Stream a chat completion and print tokens as they arrive."""
        url = f"{self.base_url}/v1/chat/completions"
        req_data = json.dumps(body).encode("utf-8")

        req = urllib.request.Request(
            url,
            data=req_data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )

        full_text = ""
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                buffer = ""
                while True:
                    chunk = resp.read(1)
                    if not chunk:
                        break
                    buffer += chunk.decode("utf-8", errors="replace")

                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        line = line.strip()

                        if not line or not line.startswith("data: "):
                            continue

                        data_str = line[6:]
                        if data_str == "[DONE]":
                            break

                        try:
                            data = json.loads(data_str)
                            delta = data.get("choices", [{}])[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                print(content, end="", flush=True)
                                full_text += content
                        except json.JSONDecodeError:
                            pass

        except urllib.error.URLError as e:
            from .errors import FLLMError
            raise FLLMError(
                f"Connection lost: {e}",
                hint=f"Server may have disconnected: {self.base_url}",
            )

        print()  # newline after streaming
        return full_text

    # ── HTTP helpers ──────────────────────────────────────────────────────

    def _get(self, path: str) -> Dict:
        url = f"{self.base_url}{path}"
        req = urllib.request.Request(
            url,
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read())

    def _post(self, path: str, body: Dict) -> Dict:
        url = f"{self.base_url}{path}"
        req_data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=req_data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read())


# ---------------------------------------------------------------------------
# Interactive remote chat
# ---------------------------------------------------------------------------

def run_remote_chat(
    client: RemoteClient,
    system_prompt: Optional[str] = None,
    stream: bool = True,
):
    """Run an interactive chat session with a remote FLLM server."""
    model = client.get_model()

    print(f"\n{'─' * 55}")
    print(f"  Remote Chat — {client.base_url}")
    print(f"  Model: {model}")
    if system_prompt:
        print(f"  System: {system_prompt[:50]}{'...' if len(system_prompt) > 50 else ''}")
    print(f"  Type /quit to exit, /clear to reset history")
    print(f"{'─' * 55}\n")

    messages: List[Dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    while True:
        try:
            user_input = input("  You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Bye!\n")
            break

        if not user_input:
            continue

        if user_input.lower() in ("/quit", "/exit", "/q"):
            print("  Bye!\n")
            break

        if user_input.lower() == "/clear":
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            print("  History cleared.\n")
            continue

        if user_input.lower() == "/metrics":
            metrics = client.get_metrics()
            if metrics:
                _print_remote_metrics(metrics)
            else:
                print("  Metrics not available on this server.\n")
            continue

        if user_input.lower() == "/info":
            health = client.check_connection()
            print(f"  Server: {json.dumps(health, indent=2)}\n")
            continue

        messages.append({"role": "user", "content": user_input})

        print(f"  AI: ", end="", flush=True)
        try:
            response = client.chat(
                messages=messages,
                stream=stream,
            )
            if not stream:
                print(response)

            messages.append({"role": "assistant", "content": response})
            print()
        except Exception as e:
            print(f"\n  ✗  Error: {e}\n")
            # Remove the failed user message
            messages.pop()


# ---------------------------------------------------------------------------
# One-shot query
# ---------------------------------------------------------------------------

def run_remote_query(
    client: RemoteClient,
    prompt: str,
    system_prompt: Optional[str] = None,
    stream: bool = False,
) -> str:
    """Send a single query and return/print the response."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    response = client.chat(messages=messages, stream=stream)
    if not stream:
        print(response)

    return response


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_remote_metrics(metrics: Dict):
    """Print remote server metrics."""
    print(f"\n  Remote Server Metrics:")
    print(f"  {'─' * 40}")
    for key in ["model_label", "uptime_seconds", "total_requests",
                 "total_tokens", "mean_tps", "peak_tps"]:
        val = metrics.get(key, "—")
        label = key.replace("_", " ").title()
        print(f"  {label:<24} {val}")
    print(f"  {'─' * 40}\n")
