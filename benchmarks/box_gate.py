"""The box gate for the Python benchmark drivers -- the twin of `box-gate.sh`, whose header is the
contract: a driver that loads fixtures runs `check-box.sh` (gh-ocannl-1062) on the box it measures
on and the fixtures it loads, before its first measurement, and refuses to start on a GAP with the
check's own output above the refusal.

    box_gate.add_arguments(ap)                     # --box BOX, --skip-box-check
    ...
    box_gate.require("gh1181_cells.py", args, HERE / "fixtures", names)

`--box` names the measurement box (an ID `fixtures/DIGESTS.txt` declares; `check-box.sh` refuses
any other). `--skip-box-check` measures without the check, for a deliberate partial run, and says
so on the run's output. Neither, at `require`, exits 2; a GAP exits 1; any other status of the
check (a usage or harness error: nothing was judged) exits 2.
"""

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CHECK_BOX = HERE / "check-box.sh"


def add_arguments(ap):
    """The gate's two options, on a driver's (sub)parser."""
    ap.add_argument(
        "--box",
        metavar="BOX",
        help="the measurement box this run measures on (an ID fixtures/DIGESTS.txt's "
        "measurement-boxes declares); benchmarks/check-box.sh checks it and the run's fixtures "
        "before the first cell, and a GAP refuses the run",
    )
    ap.add_argument(
        "--skip-box-check",
        action="store_true",
        help="measure without benchmarks/check-box.sh -- for a deliberate partial run",
    )


def require(driver, args, fixture_dir, names, run=subprocess.run):
    """Return when the box is READY for `names` (workload stems read from `fixture_dir`), or the
    run was told to skip the check; otherwise exit, after the check's own output.

    `run` spawns the check (`subprocess.run`'s calling convention, output not captured): a driver
    that runs its helpers in their own process group passes its runner for that.
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
    rc = run(cmd).returncode
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
