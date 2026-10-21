#!/bin/bash
# gh-ocannl-1253 stage-2 measurement driver: does the `auto` seeding gate seed the same segments as
# threshold 32 on gpt2_mini_train, or do better? Per fixture (b8, b256, b1_s128) and gate (32, auto), the
# untuned training step of bench_gpt, in ABBA order (32, auto, auto, 32), reporting the seeding
# census the untuned compile logs and the step time.
#
# Usage (from the checkout under test):
#   benchmarks/gh1253_flip_cells.sh (--box BOX | --skip-box-check) BACKEND OUT [CAP]
#
#   --box BOX  the measurement box (benchmarks/box-gate.sh; check-box.sh checks it and the two
#              fixtures in FIXTURE_DIR before OUT is made).
#   BACKEND    cuda | hip, pinned on every command line.
#   OUT        results directory, created, and refused unless empty: OUT/driver.log is everything
#              this script prints, each cell writes OUT/<cell>.out (the runner's result line) and
#              OUT/<cell>.err (its log, the census line included), and OUT/summary.txt one line per
#              cell: fixture, gate, pass, median step ms, the seeded-plan derivation time of each untuned
#              compile (the cost tune repeats on a trajectory-matched warm hit), and the census (the
#              seeded segments and their geometries).
#   CAP        wall cap in seconds per cell (default 1800).
#   FIXTURE_DIR (environment) where gpt2_mini_train{,_b256,_b1_s128}.safetensors are read from (default
#              this box's main checkout store, ~/ocannl-staging/benchmarks/fixtures: a fresh
#              worktree holds only DIGESTS.txt).
#
# No ambient configuration reaches a cell: every OCANNL_* variable is dropped, the config file is
# not read, and each cell pins the untuned regime it measures (no search, model_default off, the
# gate under test).

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

if [[ -e "$out" ]] && [[ -n "$(ls -A "$out" 2>/dev/null)" ]]; then
  echo "$0: $out exists and is not empty" >&2
  exit 2
fi
mkdir -p "$out"
out=$(cd "$out" && pwd)
exec > >(tee -a "$out/driver.log") 2>&1

while IFS= read -r var; do unset "$var"; done < <(compgen -e | grep '^OCANNL_' || true)
export OCANNL_NO_CONFIG_FILE=true

cd "$root"
echo "gh1253_flip_cells: revision $(git rev-parse HEAD) backend $backend host $(hostname) at $(date -u +%FT%TZ)"
git status --porcelain --untracked-files=no | sed 's/^/  dirty: /'
dune build benchmarks/runners/ocannl/bench_gpt.exe
gpt=$root/_build/default/benchmarks/runners/ocannl/bench_gpt.exe

failed=0
cell() {
  local fixture=$1 gate=$2 pass=$3 name="$1-gate$2-pass$3" census ms derive
  echo "== cell $name"
  if BENCH_FIXTURE="$fixture_dir/$fixture.safetensors" BENCH_TUNE=0 BENCH_MATERIALIZE=0 \
    BENCH_DEBUG=0 perl -e 'alarm shift; exec @ARGV' "$cap" "$gpt" \
    "--ocannl_backend=$backend" --ocannl_autotune_search=false \
    --ocannl_model_default_schedule=false --ocannl_autotune_log=true \
    "--ocannl_gpu_default_sketch_min_reduction=$gate" \
    >"$out/$name.out" 2>"$out/$name.err"; then
    # Each field is optional (a cell may seed nothing, or log no derivation): a missing one reads
    # `none logged`/`?` in the summary instead of ending the driver under `set -e -o pipefail`.
    census=$({ grep -o 'seeded_default: [0-9]* of [0-9]* segments seeded.*' "$out/$name.err" ||
      true; } | tail -1)
    # The runner's step_ms is a percentile object; the summary carries its median, read off the
    # parsed result line (strict_json.py names on stderr why a cell has none).
    ms=$(python3 "$root/benchmarks/strict_json.py" field "$out/$name.out" step_ms.p50) || ms=""
    derive=$({ grep -o 'seeded_default: plan derived in [0-9.]* s' "$out/$name.err" || true; } |
      sed 's/.* in //' | paste -sd, -)
    echo "$fixture gate=$gate pass=$pass step_p50_ms=${ms:-?} plan_derivation=[${derive:-none logged}] census=[${census:-none logged}]" |
      tee -a "$out/summary.txt"
  else
    echo "   FAILED (exit $?)"
    failed=$((failed + 1))
  fi
}

for fixture in "${fixtures[@]}"; do
  cell "$fixture" 32 1
  cell "$fixture" auto 1
  cell "$fixture" auto 2
  cell "$fixture" 32 2
done

echo "gh1253_flip_cells: $failed failed cell(s)"
exit $((failed > 0 ? 1 : 0))
