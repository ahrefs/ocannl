#!/usr/bin/env python3
"""Two revisions, interleaved: the A/B driver over the shared cell runner (gh-ocannl-1443).

    python3 benchmarks/ab.py BASE HEAD --target bin/narrow_gebp_bench.exe --backend cc \\
        --out DIR --rounds 12 -- f16 512 20

builds the dune TARGET once at each revision, each in a detached worktree of this repository under
DIR, and then measures the two executables as the two arms of one `bin_cells.py` run: the same
hermetic cells, the same configuration-trace check, the same supervision, visited in
`Bench_design`'s order (for two arms, AB BA AB BA ...), paired round by round with the exact sign
test, and written as one table through display_safe. The first revision is the reference of the
ratios. Everything after `--` is the bench's common arguments; every bin_cells option the two arms
share passes through (`--rounds`, `--reader record`, `--metric`, `--env`, `--box`, `--cell-timeout`), and
`--args-a`/`--args-b` give either arm arguments of its own (the same `NAME=VALUE ... ARGS` form as
bin_cells' `--arm`).

DIR must be outside the checkout and empty. It receives `ab.json` (the revisions asked for, the
commits they resolved to, each build's executable and sha256), `build-<label>.log` per build,
`bin/<label>.exe` (the measured copies), and `cells/` -- the bin_cells run, whose `report.md` is
also printed, with each arm's commit beside its executable's sha256. Each worktree is removed once
its executable is copied out (`--keep-worktrees` keeps them under `DIR/src-<label>`). A revision is
anything `git rev-parse` resolves to a commit; a treatment that is not a commit yet is committed on
a branch first, so the report names what was measured.

The exit status is bin_cells': 0 when every cell is clean, 3 when the only faults are cells
refusing their measurement for host contention, 1 otherwise; a revision that does not resolve or a
build that fails exits before any cell runs (2 and 1).

Boundary: the build is the commit's, from scratch in its own worktree (no shared `_build`), with
DUNE_* and OCAMLPARAM cleared; what the box's toolchain and opam switch are is the operator's, and
the same for both builds.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import bin_cells
import cells

DRIVER = "ab"
ROOT = cells.ROOT
LABELS = ("a", "b")


def resolve_revision(repo, rev):
    """The commit `rev` names in `repo`, or None."""
    probe = subprocess.run(["git", "rev-parse", "--verify", "--quiet", f"{rev}^{{commit}}"],
                           cwd=repo, capture_output=True, text=True)
    return probe.stdout.strip() if probe.returncode == 0 and probe.stdout.strip() else None


def git_supervised(repo, argv, log):
    """A mutating git command (`worktree add`/`remove`), supervised like every other child."""
    with open(log, "ab") as f:
        return cells.run_in_group(["git", *argv], repo, cells.hermetic_env(os.environ)[0], 600, f, subprocess.STDOUT)


def build_revision(repo, label, commit, target, out, keep, jobs):
    """Build `target` at `commit` in a detached worktree; the copied executable's path."""
    src = out / f"src-{label}"
    log = out / f"build-{label}.log"
    status = git_supervised(repo, ["worktree", "add", "--detach", str(src), commit], log)
    if status != 0:
        sys.exit(f"{DRIVER}: git worktree add for {label} ({commit}) failed ({status}); see {log}")
    try:
        built = cells.build_target(src, target, log, DRIVER, jobs=jobs)
        exe = out / "bin" / f"{label}.exe"
        exe.parent.mkdir(exist_ok=True)
        shutil.copy2(built, exe)
    finally:
        if not keep:
            git_supervised(repo, ["worktree", "remove", "--force", str(src)], log)
    return exe


def build_parser():
    parser = argparse.ArgumentParser(
        prog="ab.py", description=__doc__.split("\n\n", 1)[0],
        usage="%(prog)s REV_A REV_B --target DUNE_TARGET --backend B --out DIR [--rounds R] "
              "[--args-a ARGS] [--args-b ARGS] [--keep-worktrees] [BIN_CELLS-OPTIONS...] "
              "[-- BENCH-ARGS...]")
    parser.add_argument("rev_a", help="the reference revision")
    parser.add_argument("rev_b", help="the treatment revision")
    parser.add_argument("--target", required=True,
                        help="the dune target to build at each revision, e.g. bin/schedule_bench.exe")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--args-a", default="", help="arm a's own arguments (bin_cells --arm form)")
    parser.add_argument("--args-b", default="", help="arm b's own arguments (bin_cells --arm form)")
    parser.add_argument("--keep-worktrees", action="store_true",
                        help="keep each revision's worktree (and its _build) under DIR")
    parser.add_argument("--build-jobs", type=int, help="dune's -j for each build")
    parser.add_argument("--repo", type=Path, default=ROOT,
                        help="the repository the revisions are read from (default: the checkout "
                             "this script lives in)")
    # bin_cells' options the two arms share, forwarded as given.
    parser.add_argument("--backend", required=True, choices=cells.BACKENDS)
    parser.add_argument("--rounds")
    parser.add_argument("--cell-timeout")
    parser.add_argument("--reader")
    parser.add_argument("--metric", action="append", default=[])
    parser.add_argument("--env", action="append", default=[])
    parser.add_argument("--box")
    parser.add_argument("--skip-box-check", action="store_true")
    return parser


def forwarded(args):
    """The bin_cells options of `args`, as bin_cells' argv."""
    argv = ["--backend", args.backend]
    for opt in ("rounds", "cell_timeout", "reader", "box"):
        if getattr(args, opt) is not None:
            argv += [f"--{opt.replace('_', '-')}", getattr(args, opt)]
    for opt in ("metric", "env"):
        for value in getattr(args, opt):
            argv += [f"--{opt}", value]
    return argv + (["--skip-box-check"] if args.skip_box_check else [])


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    split = argv.index("--") if "--" in argv else len(argv)
    args = build_parser().parse_args(argv[:split])
    args.repo = args.repo.resolve()
    common = argv[split:]
    # The arms' own arguments go through bin_cells' parser first, so a malformed one or a stated
    # library setting is refused before anything is built.
    arm_specs = [f"{label}={spec}" for label, spec in zip(LABELS, (args.args_a, args.args_b))]
    for spec in arm_specs:
        try:
            cells.parse_arm(spec)
        except ValueError as e:
            sys.exit(f"{DRIVER}: {e}")
    commits = {}
    for label, rev in zip(LABELS, (args.rev_a, args.rev_b)):
        commit = resolve_revision(args.repo, rev)
        if commit is None:
            print(f"{DRIVER}: {rev!r} names no commit of {args.repo}", file=sys.stderr)
            return 2
        commits[label] = commit
    out = cells.fresh_out_dir(args.out, DRIVER)
    out.mkdir(parents=True, exist_ok=True)
    cells.install_cancellation(DRIVER)
    record = {"repository": str(args.repo.resolve()),
              "revisions": dict(zip(LABELS, (args.rev_a, args.rev_b))), "commits": commits,
              "target": args.target, "executables": {}}
    exes = {}
    for label in LABELS:
        exes[label] = build_revision(args.repo, label, commits[label], args.target, out,
                                     args.keep_worktrees, args.build_jobs)
        record["executables"][label] = {"path": str(exes[label]),
                                        "sha256": cells.sha256(exes[label])}
        (out / "ab.json").write_text(json.dumps(record, indent=1) + "\n")
    cell_argv = [*forwarded(args), "--out", str(out / "cells"),
                 *(f"--arm-exe={label}={exes[label]}" for label in LABELS),
                 *(f"--arm={spec}" for spec in arm_specs), *common]
    return bin_cells.main(cell_argv, arm_commits=commits)


if __name__ == "__main__":
    sys.exit(main())
