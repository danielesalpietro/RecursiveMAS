"""Thread-local token counter shared across all inference modules.

Each module's main() calls reset() at the start and get() at the end
to emit a [tokens] line that serve.py can parse.
"""
import threading

_local = threading.local()


def reset() -> None:
    _local.prompt = 0
    _local.generated = 0


def add(prompt_n: int, generated_n: int) -> None:
    _local.prompt = getattr(_local, "prompt", 0) + prompt_n
    _local.generated = getattr(_local, "generated", 0) + generated_n


def get() -> tuple:
    p = getattr(_local, "prompt", 0)
    g = getattr(_local, "generated", 0)
    return p, g
