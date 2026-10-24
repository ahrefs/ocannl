#!/bin/bash
# gh-ocannl-1557 measurement driver: does each placement arm's shipped step keep the time its
# search crowned it at? Runs the tuned cifar_resnet cell on cc four times -- the search's scratch
# lineage on the `Train.init_params` draw (BENCH_TUNE_SCRATCH_WEIGHTS=init, how the runners searched
# before gh-ocannl-1557) and on the fixture's weights (the default), each with arm A and arm B
# forced (`tune_ship_arm`) -- and prints, per cell, the shipped arm's crown (`best_ms`) against the
# timed steps' p10 and p50. One fresh schedule cache per weight set: its arm-A cell searches both
# arms, and its arm-B cell replays the stored crowns when the search's timings were clean (a
# contended search stores nothing, so that cell searches again).
#
# Usage:
#   benchmarks/gh1557_cells.sh (--box BOX | --skip-box-check) FIXTURE OUT BUDGET [WEIGHTS...]
#
#   --box    before the first cell, benchmarks/check-box.sh checks BOX and the fixture, and a GAP
#            refuses the run (benchmarks/box-gate.sh; --skip-box-check is for a deliberate partial
#            run). The checkout is the one this script is in; bench_conv.exe must already be built
#            there (`dune build ./benchmarks/runners/ocannl/bench_conv.exe`): it never builds.
#   FIXTURE  cifar_resnet.safetensors (the gate checks it against fixtures/DIGESTS.txt).
#   OUT      results directory, created, refused unless empty: OUT/<weights>-<arm>.out (the result
#            line) and .err per cell, OUT/cache-<weights>/ the schedule caches, OUT/summary.txt.
#   BUDGET   autotune_budget_seconds for every search (each arm is its own search).
#   WEIGHTS  init and/or fixture (default: both, init first).
#
# Exit: 0 every cell produced a result line; 1 a cell failed; 2 usage.
# Hermetic: every OCANNL_* and BENCH_* variable is unset and the backend is on argv
# (--ocannl_backend=cc), and so is every OpenMP runtime control (OMP_*, GOMP_*, KMP_*).
set -u
. "$(dirname "$0")/box-gate.sh"
box_gate_args "$@"; shift "$box_gate_shift"
usage() { sed -n '/^# Usage/,/^# Exit/p' "$0" | sed '$d' >&2; exit 2; }
[ $# -ge 3 ] || usage
fixture=$1 out=$2 budget=$3
shift 3
case $budget in '' | *[!0-9]*) echo "gh1557_cells: BUDGET must be a whole number of seconds" >&2; exit 2 ;; esac
weights=("$@")
[ ${#weights[@]} -gt 0 ] || weights=(init fixture)
for w in "${weights[@]}"; do
  case $w in init | fixture) ;; *) echo "gh1557_cells: unknown weights: $w" >&2; usage ;; esac
done
root=$(cd "$(dirname "$0")/.." && pwd -P) || exit 2
exe=$root/_build/default/benchmarks/runners/ocannl/bench_conv.exe
[ -x "$exe" ] || { echo "gh1557_cells: $exe is not built" >&2; exit 2; }
# The gate first: an absent fixture is the check's GAP, with its repair, not a usage error here.
fixture_dir=$(cd "$(dirname "$fixture")" 2>/dev/null && pwd -P) || fixture_dir=$(dirname "$fixture")
fixture=$fixture_dir/$(basename "$fixture")
box_gate gh1557_cells "$fixture_dir" "$(basename "$fixture" .safetensors)"
[ -f "$fixture" ] || { echo "gh1557_cells: no fixture at $fixture" >&2; exit 2; }
mkdir -p "$out" || exit 2
out=$(cd "$out" && pwd -P) || exit 2
# A fresh directory only: a leftover cache would make a search a replay.
[ -z "$(ls -A "$out")" ] || { echo "gh1557_cells: $out is not empty" >&2; exit 2; }
while read -r v; do unset "$v"; done < <(env | sed -nE 's/^((OCANNL|BENCH|OMP|GOMP|KMP)_[A-Za-z0-9_]*)=.*/\1/p')
cd "$root/benchmarks" || exit 2
rc=0
for w in "${weights[@]}"; do
  for arm in a b; do
    echo "gh1557_cells: weights=$w arm=$arm start $(date +%T)" >&2
    BENCH_FIXTURE=$fixture BENCH_TUNE=1 BENCH_TUNE_SCRATCH_WEIGHTS=$w "$exe" --ocannl_backend=cc \
      --ocannl_tune_ship_arm="$arm" --ocannl_autotune_cache_dir="$out/cache-$w" \
      --ocannl_autotune_budget_seconds="$budget" >"$out/$w-$arm.out" 2>"$out/$w-$arm.err" || rc=1
  done
done
python3 - "$root/benchmarks" "$out" "${weights[@]}" <<'EOF' | tee "$out/summary.txt"
import os, sys
sys.path.insert(0, sys.argv[1])
import strict_json
out = sys.argv[2]
for w in sys.argv[3:]:
    for arm in "ab":
        path = os.path.join(out, f"{w}-{arm}.out")
        try:
            d = strict_json.last_result(open(path).read())
        except (OSError, ValueError) as e:
            print(f"weights={w} arm={arm.upper()}: no result line ({e})")
            continue
        if d is None:
            print(f"weights={w} arm={arm.upper()}: no result line")
            continue
        shipped = d["tune"]["shipped"]
        a = {x["arm"]: x for x in d["tune"]["arms"]}[shipped]
        s = d["step_ms"]
        print(f"weights={w} shipped={shipped} state={a['state']} crown={a['best_ms']:.2f} "
              f"p10={s['p10']:.2f} p50={s['p50']:.2f} p10/crown={s['p10'] / a['best_ms']:.3f} "
              f"p50/crown={s['p50'] / a['best_ms']:.3f} contended={a['timings_contended']} "
              f"label={a['best_label']}")
EOF
exit $rc
