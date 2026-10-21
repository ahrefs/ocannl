#!/usr/bin/env python3
"""The one rule every benchmark driver uses to find the benchmark venv's interpreter.

Each measurement box keeps one shared venv, ``~/.venvs/ocannl-bench``, that a checkout reaches
through ``BENCH_VENV_PY``, through a ``benchmarks/.venv`` symlink, or -- in a fresh worktree, which
has neither -- by finding the shared venv itself (gh-ocannl-1433). A driver that spelled its own
``.venv/bin/python`` honoured only the symlink, and only on POSIX, so the rule lives here rather
than in each driver.

Run as a script it prints ``<interpreter>\\t<route>``, the pair ``resolve`` returns, which is how
``check-box.sh`` reports which route resolved.
"""

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# The box's shared venv, which `install-linux.sh --bench-venv` provisions on a Linux box and
# benchmarks/README.md#setup has built by hand on macOS; check-box.sh's --venv defaults to it.
SHARED_VENV = Path.home() / ".venvs/ocannl-bench"


def _interpreter(venv):
    """The venv's interpreter if it exists -- the Windows layout first -- else ``None``."""
    for rel in ("Scripts/python.exe", "bin/python"):
        if (Path(venv) / rel).exists():
            return Path(venv) / rel
    return None


def resolve(bench_dir=HERE, environ=None, shared=SHARED_VENV):
    """``(interpreter, route)``: the first of these that applies, and which one it was.

    1. ``BENCH_VENV_PY``, whenever it is non-empty (route ``BENCH_VENV_PY``) -- for environments
       where ``benchmarks/.venv`` is unusable (deep worktree paths hitting Windows MAX_PATH during
       torch install) or a deliberate other venv. An empty one is unset: it names no interpreter.
    2. ``benchmarks/.venv``'s interpreter, Windows layout first, when it exists (route
       ``benchmarks/.venv``).
    3. The shared venv's interpreter, when it exists (route ``shared venv <dir>``): a fresh
       worktree has no ``.venv`` link, and the box's venv is right there. ``shared=None`` leaves
       this step out.
    4. Otherwise ``benchmarks/.venv/bin/python``, which does not exist (route ``none``): an
       unprovisioned checkout names the interpreter it is missing, so the spawn error says where
       the venv was expected.
    """
    environ = os.environ if environ is None else environ
    if environ.get("BENCH_VENV_PY"):
        return Path(environ["BENCH_VENV_PY"]), "BENCH_VENV_PY"
    local = _interpreter(Path(bench_dir) / ".venv")
    if local is not None:
        return local, "benchmarks/.venv"
    if shared is not None:
        found = _interpreter(shared)
        if found is not None:
            return found, f"shared venv {shared}"
    return Path(bench_dir) / ".venv/bin/python", "none"


def venv_python(bench_dir=HERE, environ=None, shared=SHARED_VENV):
    """The interpreter ``resolve`` finds (see there for the order)."""
    return resolve(bench_dir, environ, shared)[0]


if __name__ == "__main__":
    if len(sys.argv) != 1:
        sys.exit("usage: bench_venv.py  (prints <interpreter>\\t<route> for this checkout)")
    found, route = resolve()
    print(f"{found}\t{route}")
