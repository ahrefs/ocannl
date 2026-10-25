"""The correctness gate of gh612_cells.sh's `parity` subcommand (stdlib only).

Usage: python3 gh612_parity.py OUT_ROOT MAX_ULP EXPECT_RUNS EXPECT_CELLS EXPECT_STEPS
  OUT_ROOT      the driver's results tree: cells at <OUT_ROOT>/<label>/r<rep>.
  MAX_ULP       PARITY_MAX_ULP, the failure threshold in f32 ulp.
  EXPECT_RUNS   how many SEARCH cells the gate must have covered; 0 checks no count.
  EXPECT_CELLS  the exact label/rep set (space-separated), or empty to check no set.
  EXPECT_STEPS  the loss vectors' sample count; empty for the default 8.

Computed rather than eyeballed. This exists because the claim it replaces ("bit-identical losses")
was produced by an ad-hoc script that rounded to 4 decimals before comparing: at full serialized
precision the runs are NOT identical, they agree to a few f32 ulp -- which is the right result for
reassociation across different kernel schedules, and a different claim. Compares every cell's loss
vector at the precision bench_gpt actually serializes.

Exits 0 when the gate passes, 1 when it fails (each reason on stderr behind `PARITY GATE FAILED:`,
or one `FAILED:` line for a record the gate cannot read); the per-step agreement goes to stdout.
"""
import collections
import glob
import math
import os
import sys

import bench_output


def ulp32(x):
    return 2.0 ** ((math.frexp(x)[1] - 1) - 23)


def isfin(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def loss_sequences(root):
    """{loss tuple: [cell tags]} over every search.out and pass-2 replay2.out under `root`."""
    seqs = collections.defaultdict(list)
    # BOTH passes. Every quoted tuned p50 comes from a pass-2 process, and `replay` validates only
    # its exit status, a timing fragment and the cache-hit count -- so a replay that reconstructed
    # the computation incorrectly while the cold search was fine would ship an unverified timing,
    # and scanning search.out alone would never see it.
    for kind in ("search.out", "replay2.out"):
        for f in sorted(glob.glob(os.path.join(root, "*", "r*", kind))):
            # belt and braces alongside the staging-dir naming: never count a staging artifact as
            # a cell
            if os.sep + ".staging" in f:
                continue
            try:
                d = bench_output.result_record(open(f).read())
            except bench_output.NonFinite as e:
                # A runner writes null for a number JSON cannot spell (gh-ocannl-676); NaN, Infinity
                # or 1e999 here is a regressed emitter, and a finding rather than a traceback.
                sys.exit(f"FAILED: {f}: the result line is not strict JSON ({e})")
            if d is None or "losses" not in d:
                continue
            tag = "/".join(f.split(os.sep)[-3:-1]) + ("" if kind == "search.out" else " [pass2]")
            seqs[tuple(d["losses"])].append(tag)
    return seqs


def parity(root, maxulp, expect, want_cells, steps):
    """Print the agreement table and return the gate's failures (empty when it passes)."""
    seqs = loss_sequences(root)
    tot = sum(len(v) for v in seqs.values())
    if not tot:
        sys.exit(f"no search.out with losses under {root}")
    p2 = sum(1 for ks in seqs.values() for c in ks if c.endswith("[pass2]"))
    print(f"{tot} records ({tot-p2} search + {p2} pass-2 replay), {len(seqs)} distinct loss sequences")
    for L, ks in sorted(seqs.items(), key=lambda kv: -len(kv[1])):
        print(f"  n={len(ks):2}  {L[0]!r} ...  e.g. {ks[0]}")
    # NaN/inf must be rejected BEFORE any span arithmetic: `max(0, nan)` is 0 in Python, so a run
    # whose losses are all NaN would leave `worst` at zero and print PASSED -- a correctness gate
    # certifying the precise outcome it exists to catch. A diverged loss now arrives as JSON `null`
    # rather than as a line that failed to parse at all (gh-ocannl-676), so the test is None-aware
    # and the exit is immediate: the span arithmetic below would raise a TypeError on it, which is
    # a traceback where a finding belongs.
    nonfinite = sorted({c for L, ks in seqs.items() for c in ks if not all(isfin(v) for v in L)})
    if nonfinite:
        sys.exit("FAILED: non-finite losses (null/NaN/inf) in: " + " ".join(nonfinite))
    worst = 0.0
    print("per-step agreement across ALL runs:")
    for i in range(len(next(iter(seqs)))):
        vals = sorted({L[i] for L in seqs})
        span = max(vals) - min(vals)
        u = span / ulp32(vals[0])
        worst = max(worst, u)
        print(f"  step {i}: {len(vals)} distinct, span {span:.3e} = {u:5.1f} f32 ulp "
              f"(rel {span/vals[0]:.2e})")
    print(f"WORST: {worst:.0f} f32 ulp (threshold {maxulp:.0f}). NOT bit-identity -- state it as"
          "\n       agreement to within that many ulp.")
    bad = []
    if worst > maxulp:
        bad.append(f"loss divergence {worst:.0f} ulp exceeds PARITY_MAX_ULP={maxulp:.0f}")
    # EXPECT_RUNS counts SEARCH cells; pass-2 records are extra coverage, not extra cells.
    searched = sum(1 for ks in seqs.values() for c in ks if not c.endswith("[pass2]"))
    if expect and searched != expect:
        bad.append(f"covered {searched} search runs, expected EXPECT_RUNS={expect}")
    # A count is not a set: with a reusable OUT_ROOT a stale or unrelated cell can stand in for a
    # missing required one and still total EXPECT_RUNS. EXPECT_CELLS pins the exact label/rep set.
    if want_cells:
        have = {c for ks in seqs.values() for c in ks if not c.endswith("[pass2]")}
        missing = sorted(want_cells - have)
        extra = sorted(have - want_cells)
        if missing:
            bad.append(f"missing required cells: {' '.join(missing)}")
        if extra:
            bad.append(f"unexpected cells present (stale OUT_ROOT?): {' '.join(extra)}")
        # The PASS-2 side needs the same treatment, and skipping it silently is the failure mode:
        # `replays` accepts a replay2.out on step_ms alone, and a record without `losses` is simply
        # not seen by the loop above -- so the gate could pass with a timed artifact never
        # output-verified.
        havep2 = {c[: -len(" [pass2]")] for ks in seqs.values() for c in ks if c.endswith("[pass2]")}
        m2 = sorted(want_cells - havep2)
        x2 = sorted(havep2 - want_cells)
        if m2:
            bad.append(f"pass-2 replays missing loss vectors (timed but unverified): {' '.join(m2)}")
        if x2:
            bad.append(f"unexpected pass-2 replays (stale OUT_ROOT?): {' '.join(x2)}")
    lens = {len(L) for L in seqs}
    if len(lens) != 1:
        bad.append(f"loss vectors have differing lengths {sorted(lens)}")
    # Consistent lengths are not enough: 26 records of `"losses":[]` share a length, skip the
    # comparison loop entirely, leave worst at 0 and would print PASSED having checked no executed
    # value at all. Require the expected sample count so a truncated or empty vector fails instead
    # of certifying.
    got = lens.pop() if len(lens) == 1 else None
    if got is not None and steps and got != steps:
        bad.append(f"loss vectors have {got} samples, expected EXPECT_STEPS={steps}")
    if got == 0:
        bad.append("loss vectors are EMPTY: the gate would compare nothing")
    # `tot` counts records, and a search plus ITS OWN pass-2 replay is two records of one cached
    # winner -- a codegen error shared by both is then compared only with itself. Require two
    # independent searches.
    if searched < 2:
        bad.append(f"only {searched} independent search cell(s): nothing to compare against")
    return bad


def main(argv):
    if len(argv) != 5:
        print(__doc__, file=sys.stderr)
        return 2
    root, maxulp, expect, cells, steps = argv
    bad = parity(root, float(maxulp), int(expect), set(cells.split()), int(steps) if steps else 8)
    if bad:
        for b in bad:
            print("PARITY GATE FAILED: " + b, file=sys.stderr)
        return 1
    print("parity gate PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
