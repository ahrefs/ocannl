#!/usr/bin/env python3
"""The one rule every benchmark driver uses to find the benchmark venv's interpreter.

Each measurement box keeps one shared venv (e.g. ``~/.venvs/ocannl-bench``) that a checkout
reaches either through a ``benchmarks/.venv`` symlink or through ``BENCH_VENV_PY``. A driver that
spelled its own ``.venv/bin/python`` honoured only the symlink, and only on POSIX, so the rule
lives here rather than in each driver.
"""

import os
from pathlib import Path

HERE = Path(__file__).resolve().parent


def venv_python(bench_dir=HERE, environ=None):
    """``BENCH_VENV_PY`` if set, else the venv's Windows interpreter if present, else the POSIX one.

    ``BENCH_VENV_PY`` is for environments where ``benchmarks/.venv`` is unusable (a shared venv
    outside the checkout, or deep worktree paths hitting Windows MAX_PATH during torch install).
    """
    environ = os.environ if environ is None else environ
    windows = Path(bench_dir) / ".venv/Scripts/python.exe"
    return Path(
        environ.get("BENCH_VENV_PY", windows if windows.exists() else Path(bench_dir) / ".venv/bin/python")
    )
