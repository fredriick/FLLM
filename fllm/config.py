"""
config.py — Configuration file support for FLLM.

Loads defaults from ~/.fllm/config.yaml so users don't have to
repeat flags on every invocation.

Config file format:
  # ~/.fllm/config.yaml
  default_model: qwen
  port: 8080
  mode: interactive
  backend: llama.cpp
  tier: B
  context: 4096
  cache_dir: ~/.cache/fllm
  compression: null
  no_spec: false
  verbose: false
  web: false

  system_prompt: "You are a helpful assistant."

  # Per-model overrides
  models:
    deepseek:
      context: 8192
      system_prompt: "You are a coding assistant."
    llama3:
      tier: A
      backend: vllm

Usage:
  from fllm.config import load_config, apply_config_defaults

  config = load_config()
  args = apply_config_defaults(args, config)
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# Config paths
# ---------------------------------------------------------------------------

def config_dir() -> Path:
    """Return the FLLM config directory (~/.fllm)."""
    return Path(os.environ.get("FLLM_CONFIG_DIR", Path.home() / ".fllm"))


def config_path() -> Path:
    """Return the path to the config file."""
    return config_dir() / "config.yaml"


# ---------------------------------------------------------------------------
# Load / save
# ---------------------------------------------------------------------------

def load_config(path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load config from YAML file. Returns empty dict if not found.
    Falls back to config_path() if no path given.
    """
    p = path or config_path()
    if not p.exists():
        return {}

    try:
        import yaml
    except ImportError:
        # PyYAML not installed — try simple parser
        return _load_simple(p)

    with open(p) as f:
        data = yaml.safe_load(f)

    return data if isinstance(data, dict) else {}


def save_config(data: Dict[str, Any], path: Optional[Path] = None):
    """Save config dict to YAML file."""
    p = path or config_path()
    p.parent.mkdir(parents=True, exist_ok=True)

    try:
        import yaml
        with open(p, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    except ImportError:
        # Fallback: write simple key: value format
        _save_simple(data, p)


def init_config(path: Optional[Path] = None) -> Path:
    """Create a default config file if it doesn't exist. Returns the path."""
    p = path or config_path()
    if p.exists():
        return p

    default = {
        "default_model": None,
        "port": 8080,
        "mode": "server",
        "backend": None,
        "tier": None,
        "context": None,
        "cache_dir": str(Path.home() / ".cache" / "fllm"),
        "compression": None,
        "no_spec": False,
        "verbose": False,
        "web": False,
        "system_prompt": None,
        "models": {},
    }

    save_config(default, p)
    return p


# ---------------------------------------------------------------------------
# Apply config to argparse args
# ---------------------------------------------------------------------------

# Maps config keys to argparse attribute names
_CONFIG_TO_ARG = {
    "port": "port",
    "mode": "mode",
    "backend": "backend",
    "tier": "tier",
    "context": "context",
    "cache_dir": "cache_dir",
    "compression": "compression",
    "no_spec": "no_spec",
    "verbose": "verbose",
    "web": "web",
    "system_prompt": "system_prompt",
}


def apply_config_defaults(
    args: argparse.Namespace,
    config: Dict[str, Any],
    model_key: Optional[str] = None,
) -> argparse.Namespace:
    """
    Apply config file defaults to argparse args.
    CLI flags always take priority over config values.
    Per-model overrides take priority over global config.
    """
    if not config:
        return args

    # Build merged config: global < per-model
    merged = dict(config)
    if model_key and "models" in config and isinstance(config["models"], dict):
        model_overrides = config["models"].get(model_key, {})
        if isinstance(model_overrides, dict):
            merged.update(model_overrides)

    for config_key, arg_name in _CONFIG_TO_ARG.items():
        config_value = merged.get(config_key)
        if config_value is None:
            continue

        current = getattr(args, arg_name, None)

        # Only apply config default if the CLI arg wasn't explicitly set
        if _is_default(args, arg_name, current):
            # Convert types as needed
            if arg_name == "cache_dir" and isinstance(config_value, str):
                config_value = Path(config_value).expanduser()
            if arg_name == "no_spec" and isinstance(config_value, str):
                config_value = config_value.lower() in ("true", "1", "yes")
            if arg_name == "verbose" and isinstance(config_value, str):
                config_value = config_value.lower() in ("true", "1", "yes")
            if arg_name == "web" and isinstance(config_value, str):
                config_value = config_value.lower() in ("true", "1", "yes")
            if arg_name == "context" and config_value is not None:
                config_value = int(config_value)
            if arg_name == "port" and config_value is not None:
                config_value = int(config_value)

            setattr(args, arg_name, config_value)

    return args


def get_default_model(config: Dict[str, Any]) -> Optional[str]:
    """Get the default model from config, if set."""
    return config.get("default_model")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_default(args: argparse.Namespace, name: str, value) -> bool:
    """
    Heuristic: consider a value as "default" (not explicitly set by user)
    if it matches the argparse default.
    """
    # These are the argparse defaults we know about
    defaults = {
        "port": 8080,
        "mode": "server",
        "backend": None,
        "tier": None,
        "context": None,
        "cache_dir": None,
        "compression": None,
        "no_spec": False,
        "verbose": False,
        "web": False,
        "system_prompt": None,
    }
    return value == defaults.get(name, None)


def _load_simple(path: Path) -> Dict[str, Any]:
    """Simple YAML-like parser for when PyYAML isn't installed."""
    result: Dict[str, Any] = {}
    current_section: Optional[str] = None
    current_model: Optional[str] = None

    for line in path.read_text().splitlines():
        stripped = line.strip()

        # Skip comments and empty lines
        if not stripped or stripped.startswith("#"):
            continue

        # Check indentation level
        indent = len(line) - len(line.lstrip())

        if indent == 0 and ":" in stripped:
            key, _, val = stripped.partition(":")
            key = key.strip()
            val = val.strip()

            if key == "models":
                current_section = "models"
                result["models"] = {}
                continue

            current_section = None
            current_model = None
            result[key] = _parse_value(val)

        elif indent >= 2 and current_section == "models":
            if indent == 2 and stripped.endswith(":"):
                current_model = stripped[:-1].strip()
                result["models"][current_model] = {}
            elif indent >= 4 and current_model and ":" in stripped:
                key, _, val = stripped.partition(":")
                result["models"][current_model][key.strip()] = _parse_value(val.strip())

    return result


def _parse_value(val: str) -> Any:
    """Parse a simple YAML value."""
    if not val or val == "null" or val == "~":
        return None
    if val.lower() == "true":
        return True
    if val.lower() == "false":
        return False
    try:
        return int(val)
    except ValueError:
        pass
    try:
        return float(val)
    except ValueError:
        pass
    # Strip quotes
    if (val.startswith('"') and val.endswith('"')) or \
       (val.startswith("'") and val.endswith("'")):
        return val[1:-1]
    return val


def _save_simple(data: Dict[str, Any], path: Path):
    """Write config in simple YAML-like format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    models = data.pop("models", {})

    for key, val in data.items():
        if val is None:
            lines.append(f"{key}: null")
        elif isinstance(val, bool):
            lines.append(f"{key}: {'true' if val else 'false'}")
        elif isinstance(val, str):
            lines.append(f'{key}: "{val}"')
        else:
            lines.append(f"{key}: {val}")

    if models:
        lines.append("models:")
        for model_key, model_cfg in models.items():
            lines.append(f"  {model_key}:")
            for k, v in model_cfg.items():
                if v is None:
                    lines.append(f"    {k}: null")
                elif isinstance(v, bool):
                    lines.append(f"    {k}: {'true' if v else 'false'}")
                elif isinstance(v, str):
                    lines.append(f'    {k}: "{v}"')
                else:
                    lines.append(f"    {k}: {v}")

    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------

def print_config(config: Dict[str, Any]):
    """Pretty-print the current config."""
    if not config:
        print("\n  No config file found.")
        print(f"  Run 'fllm config init' to create one at {config_path()}\n")
        return

    print(f"\n  Config: {config_path()}\n")

    models = config.get("models", {})
    for key, val in config.items():
        if key == "models":
            continue
        display = val if val is not None else "(not set)"
        print(f"  {key:<20} {display}")

    if models:
        print(f"\n  Per-model overrides:")
        for model_key, overrides in models.items():
            parts = ", ".join(f"{k}={v}" for k, v in overrides.items() if v is not None)
            print(f"    {model_key:<16} {parts}")

    print()
