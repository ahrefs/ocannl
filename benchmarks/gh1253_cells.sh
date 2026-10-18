#!/bin/bash
# gh-ocannl-1253 measurement driver: the cost model's ranking-score capacity, calibrated and checked
# on one box, so the PR's predicted-vs-measured table quotes invocations rather than restating them.
#
# Usage (from the checkout under test): benchmarks/gh1253_cells.sh BACKEND OUT [CAP]
#
#   BACKEND  cuda | hip -- pinned on every command line (no ambient configuration is trusted).
#   OUT      results directory, created, and refused unless empty: OUT/driver.log records what this
#            script prints, each cell writes OUT/<cell>.out and OUT/<cell>.err.
#   CAP      wall cap in seconds per cell (default 900); a capped cell counts as failed.
#
# Cells, all on explicitly initialized standalone fixtures (host arrays built in the runner; no
# safetensors fixture, so no box fixture gate applies):
#   bandwidth          tools/calibrate_bandwidth.exe: the STREAM kernels, rows into
#                      OUT/calibration.tsv; its fitted model_peak_memory_bandwidth is passed to every
#                      later cell, so the model's milliseconds are this box's rather than the
#                      backend's class ceiling.
#   calibrate          tools/calibrate_cache_capacity.exe: the untiled weight-gradient contraction
#                      (256 x 1024 output) swept over reduction extents 256..32768 on the plain
#                      default pipeline; prints the fitted model_cache_capacity, which is passed to
#                      the wgrad cells.
#   wgrad_b<B>_untiled bench_wgrad at batch B in {8, 256} (gpt2_mini_train's l*_ffn_w2 gradient)
#                      with seeding and the reduction sink off -- the plain default plan, whose
#                      `wgrad-model: plan=default` rows predict it.
#   wgrad_b<B>_seeded  the same with the seeded default at threshold 32 -- the seeded plan, whose
#                      `wgrad-model: plan=seeded` rows predict it.
# The shipped-kernel rows (`bench: kernel ... w: wg_w.grad`) are the measured side of the table.

set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 BACKEND OUT [CAP]" >&2
  exit 2
fi
backend=$1
out=$2
cap=${3:-900}
case "$backend" in
cuda | hip) ;;
*)
  echo "$0: BACKEND must be cuda or hip, got $backend" >&2
  exit 2
  ;;
esac
if [[ -e "$out" ]] && [[ -n "$(ls -A "$out" 2>/dev/null)" ]]; then
  echo "$0: $out exists and is not empty" >&2
  exit 2
fi
mkdir -p "$out"
exec > >(tee -a "$out/driver.log") 2>&1

root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"
echo "gh1253_cells: revision $(git rev-parse HEAD) backend $backend host $(hostname) at $(date -u +%FT%TZ)"
git status --porcelain --untracked-files=no | sed 's/^/  dirty: /'

dune build tools/calibrate_bandwidth.exe tools/calibrate_cache_capacity.exe \
  benchmarks/runners/ocannl/bench_wgrad.exe
bandwidth=$root/_build/default/tools/calibrate_bandwidth.exe
calibrate=$root/_build/default/tools/calibrate_cache_capacity.exe
wgrad=$root/_build/default/benchmarks/runners/ocannl/bench_wgrad.exe

failed=0
cell() {
  local name=$1
  shift
  echo "== cell $name: $*"
  if perl -e 'alarm shift; exec @ARGV' "$cap" "$@" >"$out/$name.out" 2>"$out/$name.err"; then
    echo "   ok"
  else
    echo "   FAILED (exit $?)"
    failed=$((failed + 1))
  fi
  cat "$out/$name.out"
}

# A fitted constant read back from a cell's config-pasteable line; empty when the cell printed none
# (its failure is already counted).
fitted() { sed -n "s/^$1=//p" "$2" | tail -1; }

cell bandwidth "$bandwidth" "--ocannl_backend=$backend" \
  "--ocannl_autotune_calibration_file=$out/calibration.tsv"
model=()
bw=$(fitted model_peak_memory_bandwidth "$out/bandwidth.out")
if [[ -n "$bw" ]]; then model+=("--ocannl_model_peak_memory_bandwidth=$bw"); fi
echo "   model flags after bandwidth: ${model[*]:-none}"
cell calibrate "$calibrate" --repeats=10 "--ocannl_backend=$backend" ${model[@]+"${model[@]}"}
cap=$(fitted model_cache_capacity "$out/calibrate.out")
if [[ -n "$cap" ]]; then model+=("--ocannl_model_cache_capacity=$cap"); fi
echo "   model flags for the wgrad cells: ${model[*]:-none}"
for batch in 8 256; do
  dims="$batch,128,256,1024"
  cell "wgrad_b${batch}_untiled" env BENCH_WGRAD_DIMS="$dims" BENCH_REPEATS=20 "$wgrad" \
    "--ocannl_backend=$backend" --ocannl_gpu_default_sketch_min_reduction=0 \
    --ocannl_gpu_default_reduction_sink=false ${model[@]+"${model[@]}"}
  cell "wgrad_b${batch}_seeded" env BENCH_WGRAD_DIMS="$dims" BENCH_REPEATS=20 "$wgrad" \
    "--ocannl_backend=$backend" --ocannl_gpu_default_sketch_min_reduction=32 \
    ${model[@]+"${model[@]}"}
done

echo "gh1253_cells: $failed failed cell(s)"
exit $((failed > 0 ? 1 : 0))
