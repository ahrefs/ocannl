"""The two-pass protocol check of gh728_cells.sh's tuned bench_gpt legs (stdlib only).

Usage: python3 gh728_protocol.py OUT PASS
  OUT   a leg's output file (gpt_<profile>_<pass>.out): the runner's stdout plus the driver's
        `exit: N` line.
  PASS  `search` or `replay`: what the leg was run as.

The replay's kernel table is what the measurement quotes, so the protocol is verified off the
runner's own result line rather than assumed from the exit status: the first pass must have
searched, and the second must have replayed every arm and searched none -- a replay over an
incomplete cache searches the missing arms and would publish a search-loaded table. Neither pass
may carry an arm that neither searched nor replayed (a pre-search failure ships the untuned
default), nor a search that refused a timing window (its winner was crowned from an incomplete
comparison).

Prints one `   protocol: ...` line (marked `-- NOT A <PASS> PASS` when it fails) and exits 0 when the
leg is the pass it was run as, 1 otherwise; a file with no single readable result line exits 1
with the reason.
"""
import sys

import bench_output


def result_line(path):
    """The leg's one result line, parsed; SystemExit naming the defect otherwise."""
    with open(path) as f:
        lines = bench_output.result_lines(f.read())
    if len(lines) != 1:
        sys.exit(f"   protocol: {path}: expected one result line, found {len(lines)}")
    try:
        return bench_output.only_result_record("\n".join(lines))
    except ValueError as e:  # NaN, Infinity or 1e999 included: a runner writes null there
        sys.exit(f"   protocol: {path}: a result line is not strict JSON ({e})")


def verdict(row, want):
    """`(ok, line)`: whether `row` is a `want` pass, and the protocol line saying what it was."""
    tune = row.get("tune") or {}
    searched, n_search, n_replay = row.get("searched"), tune.get("searches"), tune.get("replays")
    n_none = tune.get("no_searches")
    refused = sum((a.get("timings_contended") or 0) for a in tune.get("arms") or [])
    if want == "search":
        passed = searched is True and (n_search or 0) > 0
    else:
        passed = searched is False and n_search == 0 and (n_replay or 0) > 0
    ok = n_none == 0 and refused == 0 and passed
    line = (f"   protocol: {want} pass searched={searched} searches={n_search} replays={n_replay}"
            f" no_searches={n_none} refused_windows={refused}")
    return ok, line + ("" if ok else "  -- NOT A " + want.upper() + " PASS")


def main(argv):
    if len(argv) != 2 or argv[1] not in ("search", "replay"):
        print(__doc__, file=sys.stderr)
        return 2
    path, want = argv
    ok, line = verdict(result_line(path), want)
    print(line)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
