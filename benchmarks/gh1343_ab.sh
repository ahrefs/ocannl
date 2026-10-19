#!/bin/bash
# gh-ocannl-1343 measurement driver: paired before/after timing of the mlp_bn_names training step
# (test/training/bench_mlp_bn_step.ml) through the UNTUNED default CPU pipeline. "after" is the
# bench built at ROOT's HEAD; "before" is the same tree with only the scheduler sources
# (arrayjit/lib/schedule.ml and .mli, the files gh-ocannl-1343 changes) taken from BASE, built in a
# throwaway worktree -- so the two binaries differ in the scheduler alone, and the bench program,
# workload and every other library are identical. Protocol: interleaved rounds, the pair's order
# reversed on every other round (gh-ocannl-1099).
#
# Usage:
#   benchmarks/gh1343_ab.sh build   ROOT BASE BIN
#   benchmarks/gh1343_ab.sh measure ROOT BIN OUT ROUNDS STEPS
#   benchmarks/gh1343_ab.sh summary OUT
#
#   ROOT    the checkout to build (its HEAD) and to run from (its benchmarks/ocannl_config is the
#           nearest config on the upward search).
#   BASE    the revision the "before" scheduler sources come from.
#   BIN     directory receiving after.exe and before.exe; `build` creates it, refused unless empty.
#           A measurement never builds, so it can run inside a timing-only window.
#   OUT     results directory, created, refused unless empty: OUT/driver.log records the run, each
#           invocation writes OUT/<side>_r<round>.out (and .err, .status); `summary` prints the
#           per-side medians and the median of the paired ratios before/after (and `measure` runs
#           it at the end).
#   ROUNDS  a positive even count: round r runs (after, before) for even r, (before, after) for odd.
#   STEPS   timed steps per invocation (after 20 warmup steps).
#
# Exit: 0 every cell valid; 1 a cell failed or the summary is incomplete; 2 usage.
# Hermetic: every OCANNL_* variable is unset and the backend is on argv (--ocannl_backend=cc).
set -u
while read -r v; do unset "$v"; done < <(env | sed -n 's/^\(OCANNL_[A-Z0-9_]*\)=.*/\1/p')
usage() { sed -n '/^# Usage/,/^# Exit/p' "$0" | sed '$d' >&2; exit 2; }
BENCH=test/training/bench_mlp_bn_step.exe

summary() { # OUT
  python3 - "$1" <<'EOF'
import glob, os, re, statistics, sys
out = sys.argv[1]
log = os.path.join(out, 'driver.log')
m = re.search(r' rounds=(\d+)', open(log).read()) if os.path.exists(log) else None
if not m:
    print('summary: driver.log names no rounds=N: cannot tell a complete run from a partial one')
    sys.exit(1)
rounds = int(m.group(1))
line_re = re.compile(r'^mlp_bn_step: .* segments=(\d+) grid_segments=(\d+) .* median_ms=(\S+) ')
cells, bad = {}, []
for r in range(rounds):
    for side in ('after', 'before'):
        base = os.path.join(out, f'{side}_r{r}')
        status = open(base + '.status').read().strip() if os.path.exists(base + '.status') else '?'
        text = open(base + '.out').read() if os.path.exists(base + '.out') else ''
        lm = next((line_re.match(ln) for ln in text.splitlines() if line_re.match(ln)), None)
        if status != '0' or lm is None:
            bad.append(f'{side} round {r}: exit {status}, timing line {"found" if lm else "missing"}')
            continue
        cells.setdefault(side, {})[r] = (int(lm.group(1)), int(lm.group(2)), float(lm.group(3)))
for side in ('before', 'after'):
    vals = cells.get(side, {})
    if vals:
        segs = sorted({(s, g) for s, g, _ in vals.values()})
        meds = [ms for _, _, ms in vals.values()]
        print(f'{side}: segments/grid_segments {segs}, median of round medians {statistics.median(meds):.4f} ms'
              f' (min {min(meds):.4f}, max {max(meds):.4f}, rounds {len(meds)})')
pairs = [cells['before'][r][2] / cells['after'][r][2]
         for r in range(rounds) if r in cells.get('before', {}) and r in cells.get('after', {})]
if pairs:
    print(f'speedup before/after: median of {len(pairs)} paired ratios {statistics.median(pairs):.3f}x'
          f' (min {min(pairs):.3f}, max {max(pairs):.3f})')
for b in bad:
    print('FAILED CELL: ' + b)
sys.exit(1 if bad or len(pairs) != rounds else 0)
EOF
}

case "${1:-}" in
  build)
    [ $# -eq 4 ] || usage
    ROOT=$(cd "$2" && pwd -P) || exit 2
    BASE=$3 BIN=$4
    if [ -e "$BIN" ] && [ -n "$(ls -A "$BIN")" ]; then echo "gh1343_ab: $BIN is not empty" >&2; exit 2; fi
    mkdir -p "$BIN" && BIN=$(cd "$BIN" && pwd -P) || exit 2
    git -C "$ROOT" rev-parse --verify --quiet "$BASE^{commit}" >/dev/null \
      || { echo "gh1343_ab: no commit $BASE in $ROOT" >&2; exit 2; }
    (cd "$ROOT" && dune build "./$BENCH") || exit 1
    cp "$ROOT/_build/default/$BENCH" "$BIN/after.exe" || exit 1
    WT=$(mktemp -d "${TMPDIR:-/tmp}/gh1343-before.XXXXXX") || exit 1
    rmdir "$WT"
    git -C "$ROOT" worktree add --detach "$WT" HEAD >/dev/null || exit 1
    trap 'git -C "$ROOT" worktree remove --force "$WT" >/dev/null 2>&1' EXIT
    git -C "$WT" checkout "$BASE" -- arrayjit/lib/schedule.ml arrayjit/lib/schedule.mli || exit 1
    # A nested worktree builds its parent without a dune-workspace of its own (AGENTS.md).
    [ -e "$WT/dune-workspace" ] || printf '(lang dune 3.20)\n' > "$WT/dune-workspace"
    (cd "$WT" && dune build "./$BENCH") || exit 1
    cp "$WT/_build/default/$BENCH" "$BIN/before.exe" || exit 1
    {
      echo "after:  $(git -C "$ROOT" rev-parse HEAD)"
      echo "before: $(git -C "$ROOT" rev-parse HEAD) with arrayjit/lib/schedule.ml{,i} at $(git -C "$ROOT" rev-parse "$BASE")"
    } > "$BIN/provenance.txt"
    cat "$BIN/provenance.txt"
    ;;
  measure)
    [ $# -eq 6 ] || usage
    ROOT=$(cd "$2" && pwd -P) || exit 2
    BIN=$3 OUT=$4 ROUNDS=$5 STEPS=$6
    case "$ROUNDS" in ''|*[!0-9]*) usage;; esac
    case "$STEPS" in ''|*[!0-9]*) usage;; esac
    { [ "$ROUNDS" -gt 0 ] && [ $((ROUNDS % 2)) -eq 0 ] && [ "$STEPS" -gt 0 ]; } || usage
    [ -x "$BIN/after.exe" ] && [ -x "$BIN/before.exe" ] || { echo "gh1343_ab: run build first" >&2; exit 2; }
    if [ -e "$OUT" ] && [ -n "$(ls -A "$OUT")" ]; then echo "gh1343_ab: $OUT is not empty" >&2; exit 2; fi
    mkdir -p "$OUT" && OUT=$(cd "$OUT" && pwd -P) && BIN=$(cd "$BIN" && pwd -P) || exit 2
    {
      echo "gh1343_ab: host=$(hostname) date=$(date -u +%Y-%m-%dT%H:%M:%SZ) rounds=$ROUNDS steps=$STEPS"
      cat "$BIN/provenance.txt" 2>/dev/null
    } > "$OUT/driver.log"
    cat "$OUT/driver.log"
    cd "$ROOT/benchmarks" || exit 2
    for ((r = 0; r < ROUNDS; r++)); do
      if [ $((r % 2)) -eq 0 ]; then order="after before"; else order="before after"; fi
      for side in $order; do
        "$BIN/$side.exe" "$STEPS" --ocannl_backend=cc > "$OUT/${side}_r$r.out" 2> "$OUT/${side}_r$r.err"
        echo $? > "$OUT/${side}_r$r.status"
        echo "round $r $side: $(cat "$OUT/${side}_r$r.out")" | tee -a "$OUT/driver.log"
      done
    done
    summary "$OUT" | tee -a "$OUT/driver.log"
    exit "${PIPESTATUS[0]}"
    ;;
  summary)
    [ $# -eq 2 ] || usage
    summary "$2"
    ;;
  *) usage ;;
esac
