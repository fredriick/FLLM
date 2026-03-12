"""
templates.py — Library of system prompt templates for FLLM.

Built-in templates for common use cases. Users can also add
custom templates in ~/.fllm/templates/ as .txt files.

Usage:
  fllm run qwen --mode interactive --template coding
  fllm run llama3 --mode interactive --template creative
  fllm templates                    # list all available
  fllm templates show coding        # print a template
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Built-in templates
# ---------------------------------------------------------------------------

BUILTIN_TEMPLATES: Dict[str, Tuple[str, str]] = {
    # key: (description, system_prompt)

    "default": (
        "General-purpose helpful assistant",
        "You are a helpful, accurate, and concise assistant.",
    ),

    "coding": (
        "Expert programmer and code reviewer",
        (
            "You are an expert software engineer. Write clean, efficient, "
            "well-documented code. When reviewing code, identify bugs, suggest "
            "improvements, and explain your reasoning. Use best practices and "
            "modern patterns. If asked to write code, include brief comments "
            "and handle edge cases."
        ),
    ),

    "creative": (
        "Creative writer and storyteller",
        (
            "You are a creative writer with a vivid imagination. Write engaging, "
            "original content with strong narrative voice. Use rich descriptions, "
            "compelling characters, and interesting plot structures. Adapt your "
            "style to match the requested genre or tone."
        ),
    ),

    "academic": (
        "Research assistant and academic writer",
        (
            "You are an academic research assistant. Provide well-structured, "
            "evidence-based responses. Cite relevant concepts and theories. "
            "Use precise academic language. When discussing research, consider "
            "methodology, limitations, and implications. Format responses with "
            "clear sections and logical flow."
        ),
    ),

    "tutor": (
        "Patient teacher and explainer",
        (
            "You are a patient and encouraging tutor. Explain concepts step by step, "
            "starting from fundamentals. Use analogies and examples to make complex "
            "ideas accessible. Check understanding by asking follow-up questions. "
            "Adapt your explanation level to the student's knowledge."
        ),
    ),

    "math": (
        "Mathematics and problem-solving expert",
        (
            "You are a mathematics expert. Solve problems step by step, showing "
            "all work clearly. Explain the reasoning behind each step. Use proper "
            "mathematical notation. When appropriate, verify your answer and "
            "discuss alternative approaches."
        ),
    ),

    "analyst": (
        "Data analyst and critical thinker",
        (
            "You are a data analyst and critical thinker. Analyze information "
            "systematically, identify patterns, and draw evidence-based conclusions. "
            "Present findings clearly with supporting data. Consider multiple "
            "perspectives and potential biases. Quantify claims when possible."
        ),
    ),

    "devops": (
        "DevOps and infrastructure engineer",
        (
            "You are a senior DevOps engineer. Help with infrastructure, CI/CD, "
            "containerization, cloud services, monitoring, and automation. "
            "Prioritize security, reliability, and scalability. Provide concrete "
            "commands and configuration examples. Explain trade-offs between "
            "different approaches."
        ),
    ),

    "debug": (
        "Debugging specialist",
        (
            "You are a debugging specialist. When given code or error messages, "
            "systematically identify the root cause. Explain why the bug occurs, "
            "provide a fix, and suggest how to prevent similar issues. Consider "
            "edge cases and race conditions. Be thorough but concise."
        ),
    ),

    "summarize": (
        "Concise summarizer",
        (
            "You are a summarization expert. Provide clear, concise summaries "
            "that capture the key points. Use bullet points for clarity. "
            "Prioritize the most important information. Keep summaries brief "
            "unless asked for detail."
        ),
    ),

    "translate": (
        "Multilingual translator",
        (
            "You are a professional translator. Translate text accurately while "
            "preserving meaning, tone, and cultural nuance. If the source or "
            "target language is ambiguous, ask for clarification. Provide brief "
            "notes on any cultural context or untranslatable expressions."
        ),
    ),

    "shell": (
        "Shell command and scripting expert",
        (
            "You are a shell scripting expert. Help with bash, zsh, and CLI tools. "
            "Provide working commands with explanations. Warn about destructive "
            "operations. Prefer portable POSIX-compatible solutions when possible. "
            "Include error handling in scripts."
        ),
    ),

    "concise": (
        "Minimal, direct responses",
        (
            "Be extremely concise. Give direct answers without unnecessary "
            "explanation. Use short sentences. Skip pleasantries. Only elaborate "
            "if explicitly asked."
        ),
    ),

    "socratic": (
        "Teaches through questions",
        (
            "You are a Socratic teacher. Instead of giving direct answers, guide "
            "the user to discover the answer themselves through thoughtful questions. "
            "Break complex problems into smaller parts. Acknowledge good reasoning "
            "and gently redirect incorrect thinking."
        ),
    ),
}


# ---------------------------------------------------------------------------
# Custom templates from disk
# ---------------------------------------------------------------------------

def _custom_templates_dir() -> Path:
    """Return the custom templates directory."""
    import os
    config_dir = Path(os.environ.get("FLLM_CONFIG_DIR", Path.home() / ".fllm"))
    return config_dir / "templates"


def _load_custom_templates() -> Dict[str, Tuple[str, str]]:
    """Load custom templates from ~/.fllm/templates/*.txt"""
    templates_dir = _custom_templates_dir()
    if not templates_dir.exists():
        return {}

    custom = {}
    for f in sorted(templates_dir.glob("*.txt")):
        key = f.stem
        content = f.read_text().strip()

        # First line can be a description (prefixed with #)
        lines = content.split("\n", 1)
        if lines[0].startswith("#"):
            desc = lines[0].lstrip("# ").strip()
            prompt = lines[1].strip() if len(lines) > 1 else ""
        else:
            desc = f"Custom template: {key}"
            prompt = content

        custom[key] = (desc, prompt)

    return custom


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_all_templates() -> Dict[str, Tuple[str, str]]:
    """Return all templates (builtin + custom). Custom overrides builtin."""
    merged = dict(BUILTIN_TEMPLATES)
    merged.update(_load_custom_templates())
    return merged


def get_template(name: str) -> Optional[str]:
    """Get a template's system prompt by name. Returns None if not found."""
    all_templates = get_all_templates()
    entry = all_templates.get(name)
    return entry[1] if entry else None


def list_template_names() -> List[str]:
    """Return sorted list of all template names."""
    return sorted(get_all_templates().keys())


def resolve_system_prompt(
    template_name: Optional[str] = None,
    system_prompt: Optional[str] = None,
    model_default: Optional[str] = None,
) -> str:
    """
    Resolve the system prompt with priority:
      1. Explicit --system flag
      2. --template name
      3. Model's default system prompt
      4. Built-in 'default' template
    """
    if system_prompt:
        return system_prompt

    if template_name:
        prompt = get_template(template_name)
        if prompt is not None:
            return prompt
        # Template not found — warn and fall through
        import sys
        print(f"  ⚠  Template '{template_name}' not found, using model default.",
              file=sys.stderr)

    if model_default:
        return model_default

    return BUILTIN_TEMPLATES["default"][1]


# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------

def print_templates():
    """Print all available templates."""
    all_t = get_all_templates()
    custom = _load_custom_templates()
    custom_keys = set(custom.keys())

    print(f"\n  Prompt Templates ({len(all_t)} available)\n")
    print(f"  {'Name':<16} {'Type':<10} {'Description'}")
    print(f"  {'─' * 16} {'─' * 10} {'─' * 44}")

    for key in sorted(all_t.keys()):
        desc, _ = all_t[key]
        ttype = "custom" if key in custom_keys else "builtin"
        print(f"  {key:<16} {ttype:<10} {desc}")

    print(f"\n  Usage:")
    print(f"    fllm run qwen --mode interactive --template coding")
    print(f"    fllm templates show <name>")
    print(f"\n  Custom templates: {_custom_templates_dir()}")
    print(f"    Create a .txt file (first line starting with # = description)\n")


def print_template_detail(name: str):
    """Print a single template's full content."""
    all_t = get_all_templates()
    entry = all_t.get(name)

    if not entry:
        print(f"\n  Template '{name}' not found.\n")
        print(f"  Available: {', '.join(sorted(all_t.keys()))}\n")
        return

    desc, prompt = entry
    print(f"\n  Template: {name}")
    print(f"  Description: {desc}")
    print(f"  {'─' * 55}")
    # Wrap prompt text nicely
    for line in prompt.split(". "):
        line = line.strip()
        if line and not line.endswith("."):
            line += "."
        if line:
            print(f"  {line}")
    print(f"  {'─' * 55}\n")
