#!/usr/bin/env python3
"""Launcher for benchmarks/runners/ocannl/bench_bf16_crown.exe (gh-ocannl-2372).

Rejects configured artifact prefixes BEFORE starting the OCaml process (so Utils' startup cleanup
cannot touch any shared build_files/ or log_files/ directory) and executes the runner in a fresh
private working directory through benchmarks/cells.py.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if os.environ.get("OCANNL_BUILD_FILES_PREFIX"):
        sys.stderr.write(
            "gh2372_bf16_crown: unset OCANNL_BUILD_FILES_PREFIX (configured prefixes cannot be "
            "assumed process-private before OCANNL startup cleanup)\n"
        )
        return 2
    for a in args:
        if a.startswith("--ocannl_build_files_prefix") or a.startswith("--build_files_prefix"):
            sys.stderr.write(
                f"gh2372_bf16_crown: refuse {a!r} (leave build_files_prefix at its default)\n"
            )
            return 2
    if len(args) < 2:
        sys.stderr.write(
            "usage: gh2372_bf16_crown.py <repo-root> <backend metal|hip> [bench_bf16_crown args...]\n"
        )
        return 2
    root = Path(args[0]).resolve()
    backend = args[1]
    if backend not in ("metal", "hip"):
        sys.stderr.write(f"gh2372_bf16_crown: backend must be metal or hip, got {backend!r}\n")
        return 2
    exe = root / "_build/default/benchmarks/runners/ocannl/bench_bf16_crown.exe"
    if not exe.is_file():
        sys.stderr.write(f"gh2372_bf16_crown: missing executable {exe}; build it with dune first\n")
        return 2
    with tempfile.TemporaryDirectory(prefix="gh2372_crown.") as tmp:
        cell_dir = Path(tmp) / "cell"
        proc = subprocess.run(
            [
                sys.executable,
                str(HERE / "cells.py"),
                "cell",
                "--backend",
                backend,
                "--dir",
                str(cell_dir),
                "--timeout",
                "900",
                "--",
                str(exe),
                *args[2:],
            ]
        )
        stdout = cell_dir / "stdout"
        if stdout.is_file():
            sys.stdout.write(stdout.read_text())
        return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
