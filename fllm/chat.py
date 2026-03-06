"""
chat.py — Chat session manager.

Handles:
  - Chat template rendering for all supported families
  - Multi-turn history management
  - System prompt injection
  - Token budget tracking (warns when approaching context limit)
  - Session save/load (JSONL)
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterator, List, Optional

from .registry import FamilyEntry, CHAT_TEMPLATES


# ---------------------------------------------------------------------------
# Message / History
# ---------------------------------------------------------------------------

@dataclass
class Message:
    role: str      # "user" | "assistant" | "system"
    content: str
    timestamp: float = field(default_factory=time.time)
    tokens: Optional[int] = None   # filled in if backend reports it


@dataclass
class ChatSession:
    family_key: str
    model_label: str
    system_prompt: str
    context_limit: int
    messages: List[Message] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    # ── Serialisation ─────────────────────────────────────────────────────

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "family_key": self.family_key,
                "model_label": self.model_label,
                "system_prompt": self.system_prompt,
                "context_limit": self.context_limit,
                "created_at": self.created_at,
                "messages": [asdict(m) for m in self.messages],
            }, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "ChatSession":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        sess = cls(
            family_key=data["family_key"],
            model_label=data["model_label"],
            system_prompt=data["system_prompt"],
            context_limit=data["context_limit"],
            created_at=data.get("created_at", time.time()),
        )
        sess.messages = [Message(**m) for m in data.get("messages", [])]
        return sess

    # ── History helpers ───────────────────────────────────────────────────

    MAX_HISTORY = 100

    def add(self, role: str, content: str, tokens: Optional[int] = None):
        self.messages.append(Message(role=role, content=content, tokens=tokens))
        if len(self.messages) > self.MAX_HISTORY:
            # Keep system messages, trim oldest non-system messages
            system_msgs = [m for m in self.messages if m.role == "system"]
            non_system = [m for m in self.messages if m.role != "system"]
            keep = self.MAX_HISTORY - len(system_msgs)
            self.messages = system_msgs + non_system[-keep:]

    def history_for_template(self) -> List[dict]:
        """Return messages (excluding system) as plain dicts for template rendering."""
        return [
            {"role": m.role, "content": m.content}
            for m in self.messages
            if m.role != "system"
        ]

    def estimated_tokens(self) -> int:
        """Rough token count: ~4 chars per token."""
        total = len(self.system_prompt) // 4
        for m in self.messages:
            total += len(m.content) // 4
        return total

    def context_usage_pct(self) -> float:
        return self.estimated_tokens() / self.context_limit * 100


# ---------------------------------------------------------------------------
# Template renderer
# ---------------------------------------------------------------------------

class TemplateRenderer:
    def __init__(self, family: FamilyEntry):
        self.family = family
        self._fn: Callable = CHAT_TEMPLATES.get(family.chat_template, CHAT_TEMPLATES["chatml"])

    def render(self, session: ChatSession) -> str:
        return self._fn(session.system_prompt, session.history_for_template())

    def render_messages(self, system: str, messages: list) -> str:
        return self._fn(system, messages)


# ---------------------------------------------------------------------------
# Interactive chat loop
# ---------------------------------------------------------------------------

class InteractiveChat:
    """
    Cross-backend interactive chat session.
    Wraps a generate_fn callable so it works with llama.cpp, vLLM, or mlx-lm.

    generate_fn signature: (prompt: str) -> str
    """

    COMMANDS = {
        "/help":   "Show this help",
        "/save":   "Save session to disk",
        "/load":   "Load a previous session",
        "/clear":  "Clear conversation history",
        "/system": "Change system prompt",
        "/tokens": "Show estimated token usage",
        "/exit":   "Quit",
    }

    def __init__(
        self,
        session: ChatSession,
        renderer: TemplateRenderer,
        generate_fn: Callable[[str], str],
        session_dir: Optional[Path] = None,
        cleanup_fn: Optional[Callable] = None,
    ):
        self.session = session
        self.renderer = renderer
        self.generate = generate_fn
        self.session_dir = session_dir or Path.home() / ".cache" / "fllm" / "sessions"
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self._cleanup_fn = cleanup_fn

    def run(self):
        self._print_header()
        try:
            while True:
                try:
                    raw = input("\nYou: ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\n\nSession ended.")
                    self._auto_save()
                    break

                if not raw:
                    continue

                if raw.startswith("/"):
                    if self._handle_command(raw):
                        break
                    continue

                # Normal message
                self.session.add("user", raw)
                prompt = self.renderer.render(self.session)

                # Token budget warning
                pct = self.session.context_usage_pct()
                if pct > 80:
                    print(f"\n  ⚠  Context {pct:.0f}% full — consider /clear", file=sys.stderr)

                # Generate
                print("\nAssistant: ", end="", flush=True)
                t0 = time.time()
                try:
                    reply = self.generate(prompt)
                except Exception as e:
                    print(f"\n  ✗ Generation error: {e}", file=sys.stderr)
                    self.session.messages.pop()   # Remove the user message
                    continue

                elapsed = time.time() - t0
                # Estimate tokens per second (rough)
                reply_tokens = len(reply) // 4
                tps = reply_tokens / elapsed if elapsed > 0 else 0

                print(reply)
                print(f"\n  [{reply_tokens}t  {tps:.1f} tok/s  {elapsed:.2f}s]", file=sys.stderr)

                self.session.add("assistant", reply, tokens=reply_tokens)
        finally:
            if self._cleanup_fn:
                self._cleanup_fn()

    # ── Commands ─────────────────────────────────────────────────────────────

    def _handle_command(self, cmd: str) -> bool:
        """Returns True if the loop should exit."""
        parts = cmd.split(maxsplit=1)
        verb = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        if verb in ("/exit", "/quit"):
            self._auto_save()
            print("Bye!")
            return True

        elif verb == "/help":
            print("\nCommands:")
            for c, desc in self.COMMANDS.items():
                print(f"  {c:10s}  {desc}")

        elif verb == "/tokens":
            est = self.session.estimated_tokens()
            pct = self.session.context_usage_pct()
            print(f"  Estimated tokens: {est:,} / {self.session.context_limit:,} ({pct:.1f}%)")

        elif verb == "/clear":
            self.session.messages.clear()
            print("  History cleared.")

        elif verb == "/system":
            if arg:
                self.session.system_prompt = arg
                print(f"  System prompt updated.")
            else:
                print(f"  Current system prompt: {self.session.system_prompt!r}")

        elif verb == "/save":
            name = arg or datetime.now().strftime("%Y%m%d_%H%M%S")
            path = self.session_dir / f"{name}.json"
            self.session.save(path)
            print(f"  Saved → {path}")

        elif verb == "/load":
            if not arg:
                self._list_sessions()
            else:
                path = self.session_dir / f"{arg}.json"
                if not path.exists():
                    path = Path(arg)
                if path.exists():
                    self.session = ChatSession.load(path)
                    print(f"  Loaded {path.name}  ({len(self.session.messages)} messages)")
                else:
                    print(f"  Session not found: {arg}")

        else:
            print(f"  Unknown command: {verb}  (try /help)")

        return False

    def _auto_save(self):
        name = datetime.now().strftime("autosave_%Y%m%d_%H%M%S")
        path = self.session_dir / f"{name}.json"
        self.session.save(path)
        print(f"  Session auto-saved → {path}", file=sys.stderr)

    def _list_sessions(self):
        sessions = sorted(self.session_dir.glob("*.json"))
        if not sessions:
            print("  No saved sessions found.")
        else:
            print("  Saved sessions:")
            for p in sessions[-10:]:   # Show last 10
                print(f"    {p.stem}")

    def _print_header(self):
        print(f"\n{'─' * 55}")
        print(f"  Model   : {self.session.model_label}")
        print(f"  Template: {self.renderer.family.chat_template}")
        print(f"  Context : {self.session.context_limit:,} tokens")
        print(f"  System  : {self.session.system_prompt[:60]!r}{'...' if len(self.session.system_prompt) > 60 else ''}")
        print(f"{'─' * 55}")
        print("  Type /help for commands, /exit to quit.\n")
