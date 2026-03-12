"""
errors.py — Structured error handling for FLLM.

Provides:
  - Custom exception hierarchy for common failure modes
  - Consistent error formatting with actionable hints
  - CLI error wrapper that catches exceptions gracefully
"""

from __future__ import annotations

import sys
import traceback
from typing import Optional


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------

class FLLMError(Exception):
    """Base exception for all FLLM errors."""
    exit_code: int = 1
    hint: Optional[str] = None

    def __init__(self, message: str, hint: Optional[str] = None):
        super().__init__(message)
        if hint:
            self.hint = hint


class ModelNotFoundError(FLLMError):
    """Model family not recognized."""
    def __init__(self, family: str, available: Optional[list] = None):
        avail_str = ""
        if available:
            avail_str = f"\n  Available: {', '.join(available)}"
        super().__init__(
            f"Unknown model family: '{family}'",
            hint=f"Run 'fllm models' to see supported models.{avail_str}",
        )


class ModelTooLargeError(FLLMError):
    """Model doesn't fit in available memory."""
    def __init__(self, family: str, required_gb: float, available_gb: float):
        super().__init__(
            f"Model '{family}' requires ~{required_gb:.1f} GB but only "
            f"{available_gb:.1f} GB available.",
            hint=(
                "Try a smaller model, lower quantization, or reduce context:\n"
                "  fllm scan                    # see what fits\n"
                "  fllm run <model> --context 2048  # reduce context"
            ),
        )


class BackendNotAvailableError(FLLMError):
    """Required backend is not installed."""
    def __init__(self, backend: str, install_hint: str = ""):
        super().__init__(
            f"Backend '{backend}' is not available.",
            hint=install_hint or f"Install the required backend and try again.",
        )


class DownloadError(FLLMError):
    """Model download failed."""
    def __init__(self, message: str, repo: str = ""):
        hint = "Check your internet connection and try again."
        if repo:
            hint += f"\n  Repository: {repo}"
        super().__init__(message, hint=hint)


class AuthenticationError(FLLMError):
    """HuggingFace authentication required."""
    def __init__(self, repo: str = ""):
        super().__init__(
            "This model requires authentication.",
            hint=(
                "Set your HuggingFace token:\n"
                "  export HF_TOKEN=hf_...\n"
                "  # or\n"
                "  huggingface-cli login"
                + (f"\n  Repository: {repo}" if repo else "")
            ),
        )


class ConfigError(FLLMError):
    """Configuration file error."""
    def __init__(self, message: str):
        super().__init__(
            message,
            hint=f"Check your config file: fllm config show",
        )


class GPUInitError(FLLMError):
    """GPU initialization failed."""
    def __init__(self, message: str = ""):
        super().__init__(
            message or "GPU initialization failed.",
            hint=(
                "Try running in CPU-only mode:\n"
                "  export FLLM_CPU_ONLY=1\n"
                "  # or use a different backend:\n"
                "  fllm run <model> --backend llama.cpp"
            ),
        )


class DependencyError(FLLMError):
    """Required Python package not installed."""
    def __init__(self, package: str, install_cmd: str = ""):
        cmd = install_cmd or f"pip install {package}"
        super().__init__(
            f"Required package '{package}' is not installed.",
            hint=f"Install it with:\n  {cmd}",
        )


class ServerError(FLLMError):
    """Server startup or runtime error."""
    def __init__(self, message: str, port: int = 0):
        hint = "Check if another process is using the port."
        if port:
            hint += f"\n  Port: {port}\n  Try: fllm serve <model> --port {port + 1}"
        super().__init__(message, hint=hint)


# ---------------------------------------------------------------------------
# Error formatting
# ---------------------------------------------------------------------------

def format_error(error: Exception, verbose: bool = False) -> str:
    """Format an exception into a user-friendly error message."""
    lines = []

    if isinstance(error, FLLMError):
        lines.append(f"  ✗  {error}")
        if error.hint:
            lines.append("")
            for hint_line in error.hint.split("\n"):
                lines.append(f"  {hint_line}")
    elif isinstance(error, KeyboardInterrupt):
        lines.append("  Interrupted.")
    elif isinstance(error, MemoryError):
        lines.append("  ✗  Out of memory!")
        lines.append("")
        lines.append("  Try a smaller model or reduce context length:")
        lines.append("    fllm scan")
        lines.append("    fllm run <model> --context 2048")
    elif isinstance(error, ConnectionError):
        lines.append(f"  ✗  Connection error: {error}")
        lines.append("")
        lines.append("  Check your internet connection and try again.")
    elif isinstance(error, PermissionError):
        lines.append(f"  ✗  Permission denied: {error}")
        lines.append("")
        lines.append("  Check file permissions or try with sudo.")
    elif isinstance(error, FileNotFoundError):
        lines.append(f"  ✗  File not found: {error}")
    elif isinstance(error, OSError) and "Address already in use" in str(error):
        lines.append(f"  ✗  Port already in use.")
        lines.append("")
        lines.append("  Try a different port: fllm serve <model> --port 8081")
    else:
        lines.append(f"  ✗  Unexpected error: {type(error).__name__}: {error}")

    if verbose and not isinstance(error, (FLLMError, KeyboardInterrupt)):
        lines.append("")
        lines.append("  Traceback:")
        for tb_line in traceback.format_exception(type(error), error, error.__traceback__):
            for sub_line in tb_line.rstrip().split("\n"):
                lines.append(f"    {sub_line}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI error wrapper
# ---------------------------------------------------------------------------

def cli_error_handler(func, verbose: bool = False):
    """
    Wrap a CLI command function with graceful error handling.
    Returns an exit code (0 for success, non-zero for errors).
    """
    try:
        func()
        return 0
    except KeyboardInterrupt:
        print("\n  Interrupted.\n")
        return 130
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else 0
    except FLLMError as e:
        print(f"\n{format_error(e, verbose=verbose)}\n")
        return e.exit_code
    except MemoryError as e:
        print(f"\n{format_error(e, verbose=verbose)}\n")
        return 1
    except Exception as e:
        print(f"\n{format_error(e, verbose=verbose)}\n")
        if not verbose:
            print("  Run with --verbose for full traceback.\n")
        return 1
