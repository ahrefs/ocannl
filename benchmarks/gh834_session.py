"""Session completeness for gh834_cells.sh, read from the result line as JSON (stdlib only).

Usage: python3 gh834_session.py OUT_FILE

Reads the last line of OUT_FILE that starts with "{" (the runner's result line) and prints one
diagnostic per problem, exiting 1 if there was any and 0 if the session is complete. The driver
prefixes each line with its step name.

The line is parsed rather than scanned: the arms' field order and nesting are the runner's to
choose (benchmarks/runners/ocannl/bench_json.ml), and a regex that stops at an arm's first closing
brace reads a nested object placed before a scalar as a missing scalar -- the failure that made
gh-ocannl-1263's per-arm timing matrix move to the arm's last field. Every field the checks rest on
is required to be present with its type, so a renamed or dropped field fails the session rather
than reading as a pass.
"""
import math
from pathlib import Path
import sys

# gh834_cells.sh runs this under `python3 -I`, which leaves the script's own directory off sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import strict_json  # noqa: E402

# The two placement arms a tuned gpt2_mini session runs (Train.tune_placements).
REQUIRED_ARMS = ("A", "B")

# An arm's refused-window counts: every refusal, then the two causes named apart from contention.
REFUSAL_COUNTS = ("timings_contended", "timings_unbatched", "timings_unreadable")


def is_count(v):
    return isinstance(v, int) and not isinstance(v, bool) and v >= 0


def problems(path):
    try:
        with open(path, encoding="utf-8") as f:
            lines = [line for line in f.read().splitlines() if line.startswith("{")]
    except OSError as e:
        return ["INCOMPLETE SESSION: cannot read the result: %s" % e]
    if not lines:
        return ["INCOMPLETE SESSION: no result line"]
    try:
        # JSON has no NaN or Infinity, but json.loads accepts both by default (and reads 1e999 as
        # infinity); a runner that spells a non-finite number either way has emitted a line no other
        # consumer can read.
        record = strict_json.loads(lines[-1])
    except ValueError as e:
        return ["INCOMPLETE SESSION: the result line is not JSON (%s)" % e]
    tune = record.get("tune") if isinstance(record, dict) else None
    arms = tune.get("arms") if isinstance(tune, dict) else None
    if not isinstance(arms, list) or not all(isinstance(a, dict) for a in arms):
        return ["INCOMPLETE SESSION: the result line carries no tune arms"]
    out = []
    for name in REQUIRED_ARMS:
        found = [a for a in arms if a.get("arm") == name]
        if len(found) != 1 or found[0].get("state") != "searched":
            # Absent, ambiguous, or did not search: "searched":true at the top covers a died search.
            detail = "" if len(found) <= 1 else " (it appears %d times)" % len(found)
            out.append("INCOMPLETE SESSION: arm %s did not complete a search%s" % (name, detail))
        # A search can complete having timed nothing (every candidate declined; the GPU serial
        # baseline is never dispatched): such an arm holds no evidence about timing cost.
        best = found[0].get("best_ms") if len(found) == 1 else None
        # strict_json refuses a non-finite number, so isfinite only guards what a refactor lets in.
        if not isinstance(best, (int, float)) or isinstance(best, bool) or not math.isfinite(best):
            out.append("INCOMPLETE SESSION: arm %s timed no candidate" % name)
    # The remaining checks span every arm on the line, not only the required two.
    for arm in arms:
        name = arm.get("arm", "?")
        for field in ("terminal_failure",) + REFUSAL_COUNTS:
            if field not in arm:
                out.append("INCOMPLETE SESSION: arm %s carries no %s" % (name, field))
        bad = [f for f in REFUSAL_COUNTS if f in arm and not is_count(arm[f])]
        for field in bad:
            out.append("INCOMPLETE SESSION: arm %s has a non-count %s" % (name, field))
    # Whatever shape a terminal failure takes, it is not null.
    if any(a.get("terminal_failure") is not None for a in arms if "terminal_failure" in a):
        out.append("INCOMPLETE SESSION: an arm carries a terminal failure")
    # Per arm, timings_contended counts every refused window, timings_unbatched the ones refused
    # because queued calibration measured no batch within its target (gh-ocannl-1098), and
    # timings_unreadable the ones refused for a degenerate clock reading (gh-ocannl-1517); the two
    # are disjoint, and the rest is contention. Each leaves the session incomplete; they are named
    # apart because they call for different follow-ups: the readings behind a no-batch refusal
    # cannot tell a queue threshold from a stall on every probe (one that repeats on an idle rerun
    # is the threshold), and a clock that resolved nothing is not host load at all.
    counted = [a for a in arms if all(is_count(a.get(f)) for f in REFUSAL_COUNTS)]
    if any(a["timings_contended"] > a["timings_unbatched"] + a["timings_unreadable"] for a in counted):
        out.append("CONTENDED SESSION: timing windows were refused for host contention")
    if any(a["timings_unbatched"] > 0 for a in counted):
        out.append(
            "UNBATCHED SESSION: queued calibration measured no batch within the target"
            " (a queue threshold if it repeats on an idle rerun, else a stall)"
        )
    if any(a["timings_unreadable"] > 0 for a in counted):
        out.append(
            "UNREADABLE SESSION: timing windows were refused for a degenerate clock reading"
            " (a clock that resolved nothing, not host load)"
        )
    return out


def main(argv):
    if len(argv) != 2:
        print(__doc__.split("\n\n")[1], file=sys.stderr)
        return 2
    found = problems(argv[1])
    for line in found:
        print(line)
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
