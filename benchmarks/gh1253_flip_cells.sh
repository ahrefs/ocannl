#!/bin/bash
# gh-ocannl-1253 stage-2 measurement driver: does the `auto` seeding gate seed the same segments as
# threshold 32 on gpt2_mini_train, or do better? Per fixture (b8, b256, b1_s128) and gate (32, auto), the
# untuned training step of bench_gpt, in ABBA order (32, auto, auto, 32), reporting the seeding
# census the untuned compile logs and the step time.
#
# Usage (from the checkout under test):
#   benchmarks/gh1253_flip_cells.sh (--box BOX | --skip-box-check) BACKEND OUT [CAP]
#
#   --box BOX  the measurement box (benchmarks/box-gate.sh; check-box.sh checks it and the
#              fixtures in FIXTURE_DIR before OUT is made, and bin_cells.py checks each fixture's
#              again before its cells).
#   BACKEND    cuda | hip, pinned on every command line.
#   OUT        results directory, created, and refused unless empty: OUT/driver.log is everything
#              this script prints, OUT/<fixture>/ that fixture's bin_cells.py run (report.md,
#              cells.jsonl, manifest.json, cells/<NNN>-<gate>/ with each cell's stdout -- the
#              runner's result line -- and stderr, the census line included), and OUT/summary.txt
#              one line per cell: fixture, gate, pass, median step ms, the seeded-plan derivation
#              time of each untuned compile (the cost tune repeats on a trajectory-matched warm
#              hit), and the census (the seeded segments and their geometries).
#   CAP        wall cap in seconds per cell (default 1800).
#   FIXTURE_DIR (environment) where gpt2_mini_train{,_b256,_b1_s128}.safetensors are read from (default
#              this box's main checkout store, ~/ocannl-staging/benchmarks/fixtures: a fresh
#              worktree holds only DIGESTS.txt).
#
# The cells are bin_cells.py's (`--reader record`, gh-ocannl-1607): per fixture one run whose two
# arms, g32 and auto, are the gate under test (--ocannl_gpu_default_sketch_min_reduction), over two
# rounds of Bench_design's order -- (g32, auto), (auto, g32): the ABBA above, a cell's pass being its
# round plus one. Each cell is hermetic (no OCANNL_*, BENCH_* or OpenMP variable inherited, no config
# file read), supervised under CAP, and configuration-checked: its resolved-configuration records
# must be its command line, which pins the untuned regime it measures (no search, model_default
# off, BENCH_TUNE=0) beside the gate. A cell that is not clean in its run fails the driver.

set -euo pipefail

. "$(dirname "$0")/box-gate.sh"
box_gate_args "$@"
shift "$box_gate_shift"

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 (--box BOX | --skip-box-check) BACKEND OUT [CAP]" >&2
  exit 2
fi
backend=$1
out=$2
cap=${3:-1800}
case "$backend" in
cuda | hip) ;;
*)
  echo "$0: BACKEND must be cuda or hip, got $backend" >&2
  exit 2
  ;;
esac
root=$(cd "$(dirname "$0")/.." && pwd)
fixture_dir=${FIXTURE_DIR:-$HOME/ocannl-staging/benchmarks/fixtures}
# gh-ocannl-1442 adds the batch-1 fixture, where the ranked gate's small contractions sit;
# FIXTURES (environment, space-separated) narrows or reorders the set.
read -r -a fixtures <<<"${FIXTURES:-gpt2_mini_train gpt2_mini_train_b256 gpt2_mini_train_b1_s128}"
box_gate gh1253flip "$fixture_dir" "${fixtures[@]}"
gate=(--skip-box-check)
if [[ $box_gate_skip == 0 ]]; then gate=(--box "$box_gate_box"); fi

if [[ -e "$out" ]] && [[ -n "$(ls -A "$out" 2>/dev/null)" ]]; then
  echo "$0: $out exists and is not empty" >&2
  exit 2
fi
mkdir -p "$out"
out=$(cd "$out" && pwd)
exec > >(tee -a "$out/driver.log") 2>&1

cd "$root"
echo "gh1253_flip_cells: revision $(git rev-parse HEAD) backend $backend host $(hostname) at $(date -u +%FT%TZ)"
git status --porcelain --untracked-files=no | sed 's/^/  dirty: /'
dune build benchmarks/runners/ocannl/bench_gpt.exe
gpt=$root/_build/default/benchmarks/runners/ocannl/bench_gpt.exe

failed=0
for fixture in "${fixtures[@]}"; do
  echo "== fixture $fixture"
  rc=0
  python3 "$root/benchmarks/bin_cells.py" --exe "$gpt" --backend "$backend" --out "$out/$fixture" \
    "${gate[@]}" --reader record --rounds 2 --cell-timeout "$cap" \
    --env "BENCH_FIXTURE=$fixture_dir/$fixture.safetensors" --env BENCH_TUNE=0 \
    --env BENCH_MATERIALIZE=0 --env BENCH_DEBUG=0 \
    --arm g32=--ocannl_gpu_default_sketch_min_reduction=32 \
    --arm auto=--ocannl_gpu_default_sketch_min_reduction=auto \
    -- --ocannl_autotune_search=false --ocannl_model_default_schedule=false \
    --ocannl_autotune_log=true >/dev/null || rc=$?
  echo "   bin_cells exit $rc (report: $fixture/report.md)"
  if [[ $rc -eq 125 ]]; then
    # A survivor holds the device every later cell would be timed against.
    echo "gh1253_flip_cells: a process outlived its cleanup; stopping"
    exit 125
  fi
  if [[ $rc -ne 0 ]]; then failed=$((failed + 1)); fi
done

# One summary line per cell, in the order the cells ran; a cell its run did not find clean fails.
python3 - "$root/benchmarks" "$out" "${fixtures[@]}" <<'EOF' | tee -a "$out/summary.txt" || failed=$((failed + 1))
import json, sys
sys.path.insert(0, sys.argv[1])
import bench_output
import cells
from display_safe import display_value, log_line
out, bad = sys.argv[2], 0
gates = {"g32": "32", "auto": "auto"}
for fixture in sys.argv[3:]:
    run = f"{out}/{fixture}"
    try:
        manifest, records = cells.read_run(run)
    except (OSError, ValueError) as e:
        print(log_line(f"{display_value(fixture)}: no complete bin_cells run ({display_value(str(e))})"))
        bad += 1
        continue
    planned = sum(len(order) for order in manifest["orders"])
    if len(records) != planned:
        print(f"{display_value(fixture)}: {len(records)} of {planned} planned cells ran")
        bad += 1
    for r in records:
        head = f"{display_value(fixture)} gate={gates.get(r['arm'], r['arm'])} pass={r['round'] + 1}"
        if not r["ok"]:
            print(log_line(f"{head} FAILED: {display_value('; '.join(r['problems']))}"))
            bad += 1
            continue
        err = cells.cell_stream(run, r, "stderr")
        # Each field is optional (a cell may seed nothing, or log no derivation): a missing one
        # reads `none logged`/`?`.
        derive, census = bench_output.seeded_default(err)
        # The median as the result line states it (bin_cells' reader already held it to a time).
        ms = json.dumps(bench_output.field(
            bench_output.result_record(cells.cell_stream(run, r, "stdout")), "step_ms.p50"))
        print(log_line(f"{head} step_p50_ms={ms} "
                       f"plan_derivation=[{','.join(derive) or 'none logged'}] "
                       f"census=[{census[-1] if census else 'none logged'}]"))
sys.exit(1 if bad else 0)
EOF

echo "gh1253_flip_cells: $failed failed fixture run(s) or summary"
exit $((failed > 0 ? 1 : 0))
