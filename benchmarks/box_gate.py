"""The box gate for the Python benchmark drivers -- the twin of `box-gate.sh`, whose header is the
contract: a driver that loads fixtures runs `check-box.sh` (gh-ocannl-1062) on the box it measures
on and the fixtures it loads, before its first measurement, and refuses to start on a GAP with the
check's own output above the refusal.

    box_gate.add_arguments(ap)                     # --box BOX, --skip-box-check
    ...
    box_gate.require("gh1181_cells.py", args, HERE / "fixtures", names, run=<managed runner>)

`--box` names the measurement box (an ID `fixtures/DIGESTS.txt` declares; `check-box.sh` refuses
any other), or the fleet's name for one, which the parsed value holds as the box's ID. `--skip-box-check` measures without the check, for a deliberate partial run, and says
so on the run's output. Neither, at `require`, exits 2; a GAP exits 1; any other status of the
check (a usage or harness error: nothing was judged) exits 2.
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CHECK_BOX = HERE / "check-box.sh"


def canonical_box(name):
    """The measurement box a fleet name stands for (`rog-nv-linux` -> `rog-nv`), else `name`.

    check-box.sh's table answers, so a driver records the box's ID in its rows rather than the
    fleet's name for it (gh-ocannl-1433); a check-box.sh that cannot answer leaves the name as
    given, for the check itself to judge."""
    import subprocess

    try:
        done = subprocess.run(
            ["bash", str(CHECK_BOX), "--canonical-box", name],
            capture_output=True, text=True, timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return name
    out = done.stdout.strip()
    return out if done.returncode == 0 and out and "\n" not in out else name


def add_arguments(ap):
    """The gate's two options, on a driver's (sub)parser. `--box` holds the box's ID."""
    ap.add_argument(
        "--box",
        metavar="BOX",
        type=canonical_box,
        help="the measurement box this run measures on (an ID fixtures/DIGESTS.txt's "
        "measurement-boxes declares); benchmarks/check-box.sh checks it and the run's fixtures "
        "before the first cell, and a GAP refuses the run",
    )
    ap.add_argument(
        "--skip-box-check",
        action="store_true",
        help="measure without benchmarks/check-box.sh -- for a deliberate partial run",
    )


def require(driver, args, fixture_dir, names, *, run):
    """Return when the box is READY for `names` (workload stems read from `fixture_dir`), or the
    run was told to skip the check; otherwise exit, after the check's own output.

    `run(cmd)` runs the check, output not captured, and returns its exit status. It is the
    driver's own managed runner -- the check leads a process group of its own, collected whole if
    the driver is cancelled -- because the check's device item is a GPU process that a bare
    `subprocess.run` would leave running past a SIGTERM; so it has no default. Anything but an
    int (a runner's timeout verdict) is a check that judged nothing.
    """
    names = list(names)
    if not names:
        sys.exit(f"{driver}: the box gate was given no fixture to check")
    if args.skip_box_check:
        print(
            f"{driver}: --skip-box-check: measuring without benchmarks/check-box.sh "
            "(a deliberate partial run)",
            flush=True,
        )
        return
    if not args.box:
        print(
            f"{driver}: --box BOX is required to measure: the measurement box "
            "benchmarks/check-box.sh checks before the first cell (fixtures/DIGESTS.txt's "
            "measurement-boxes); --skip-box-check for a deliberate partial run",
            file=sys.stderr,
            flush=True,
        )
        sys.exit(2)
    cmd = ["bash", str(CHECK_BOX), "--box", args.box, "--fixture-dir", str(fixture_dir)]
    for name in names:
        cmd += ["--fixture", name]
    sys.stdout.flush()
    sys.stderr.flush()
    rc = run(cmd)
    if rc == 0:
        return
    if rc == 1:
        print(
            f"{driver}: refusing to start: {args.box} is not ready for this run -- each GAP above "
            "names its repair; --skip-box-check is for a deliberate partial run",
            file=sys.stderr,
            flush=True,
        )
        sys.exit(1)
    print(
        f"{driver}: refusing to start: benchmarks/check-box.sh judged nothing (exit {rc})",
        file=sys.stderr,
        flush=True,
    )
    sys.exit(2)
