#!/bin/bash
# gh-ocannl-1253 measurement driver: the cost model's ranking-score capacity, calibrated and checked
# on one box, so the PR's predicted-vs-measured table quotes invocations rather than restating them.
# gh-ocannl-1442 extends it with the score's own rates (the issue rate and the DRAM rate) and with
# ranking-agreement cells, which time every whole-routine candidate at a site and compare the
# score's order with the measured one.
#
# Usage (from the checkout under test): benchmarks/gh1253_cells.sh BACKEND OUT [CAP]
#
#   BACKEND  cuda | hip -- pinned on every command line; no ambient OCANNL_* variable or config
#            file reaches a cell (see below).
#   OUT      results directory, created, and refused unless empty: OUT/driver.log records what this
#            script prints, each cell writes OUT/<cell>.out and OUT/<cell>.err.
#   CAP      wall cap in seconds per cell (default 900); a capped cell counts as failed.
#
# Cells, all on explicitly initialized standalone inputs (host arrays built in the runner: no
# recorded workload file is read, so the box gate's workload check has nothing to check):
#   bandwidth          tools/calibrate_bandwidth.exe: the STREAM kernels, rows into
#                      OUT/calibration.tsv; its fitted bandwidth is passed to every later cell as
#                      model_dram_bandwidth, the ranking score's DRAM rate (gh-ocannl-1442), so the
#                      model's milliseconds are this box's while the bound keeps the backend's class
#                      ceiling, as it ships.
#   calibrate          tools/calibrate_cache_capacity.exe: the untiled weight-gradient contraction
#                      (256 x 1024 output) swept over reduction extents 256..65536 on the plain
#                      default pipeline, split against the calibrated DRAM bandwidth; prints the
#                      fitted model_cache_capacity (none when no point reaches DRAM) and
#                      model_cache_bandwidth (the below-capacity plateau), passed to the wgrad
#                      cells.
#   issue              tools/calibrate_issue_rate.exe: the seeded sketch of a 512 x 512 contraction
#                      (not the validation cells' 256 x 1024) swept over reduction extents; prints
#                      the fitted model_issue_rate, passed to every later cell, and the intercept
#                      the score does not price.
#   wgrad_b<B>_untiled bench_wgrad at batch B in {1, 8, 256} (gpt2_mini_train's l*_ffn_w2 gradient)
#                      with seeding, the split and the reduction sink off -- the plain default
#                      plan, whose
#                      `wgrad-model: plan=default` rows predict it.
#   wgrad_b<B>_seeded  the same with the seeded default at threshold 32, the split and the sink off
#                      -- the one-kernel seeded sketch, whose `wgrad-model: plan=seeded` rows predict
#                      it; the cell fails rather than time a plan that seeded nothing or fell back
#                      (BENCH_WGRAD_REQUIRE_SEEDED).
#   rank_<site>_*      per site of SITES (default: the wgrad layer at b8 and b256, and the
#                      gh-ocannl-1190 fp8 matmul), bench_rank_site: `time` tunes the site with seed
#                      candidates only, every one timed (autotune_keep_fraction=1), rows into
#                      OUT/rank_<site>.tsv; `fitted` and `ceilings` list model_explain's ranking
#                      score of each candidate under the fitted rates and with both rates off (the
#                      capacity and cache tier as fitted either way); benchmarks/gh1442_agreement.py
#                      joins them into OUT/rank_<site>_agreement.txt.
# RANK_ONLY=1 (gh-ocannl-1278) runs the agreement cells alone, under the shipped per-class
# constants: no calibration or wgrad cell runs, and the `fitted` arm is named `shipped` (the
# defaults, no model_* flag), beside `ceilings`. With it, SITES may name bench_rank_site's
# gpt2_mini projection sites (qkv:b,s,h,e,k and outproj:b,s,j,h,e), whose explain lines list the
# coalesced branch's leaves the tuner times beside the family.
# Both wgrad arms pin BENCH_TUNE=0 and model_default_schedule=false, so an ambient profile or
# environment cannot swap the measured plans for tuned or model-picked ones.
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
out=$(cd "$out" && pwd) # absolute: the cells run from the checkout's root
exec > >(tee -a "$out/driver.log") 2>&1

# No ambient configuration reaches a cell: every OCANNL_* variable is dropped and the checkout's
# config file is not read, so each cell runs the defaults plus exactly the flags written below
# (an inherited OCANNL_SCHEDULE_FISSION=false, say, would make the seeded arm decline silently).
while IFS= read -r var; do unset "$var"; done < <(compgen -e | grep '^OCANNL_' || true)
export OCANNL_NO_CONFIG_FILE=true

root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"
echo "gh1253_cells: revision $(git rev-parse HEAD) backend $backend host $(hostname) at $(date -u +%FT%TZ)"
git status --porcelain --untracked-files=no | sed 's/^/  dirty: /'

dune build tools/calibrate_bandwidth.exe tools/calibrate_cache_capacity.exe \
  tools/calibrate_issue_rate.exe benchmarks/runners/ocannl/bench_wgrad.exe \
  benchmarks/runners/ocannl/bench_rank_site.exe
bandwidth=$root/_build/default/tools/calibrate_bandwidth.exe
calibrate=$root/_build/default/tools/calibrate_cache_capacity.exe
issue=$root/_build/default/tools/calibrate_issue_rate.exe
wgrad=$root/_build/default/benchmarks/runners/ocannl/bench_wgrad.exe
rank_site=$root/_build/default/benchmarks/runners/ocannl/bench_rank_site.exe

failed=0
last_ok=false
cell() {
  local name=$1
  shift
  echo "== cell $name: $*"
  if perl -e 'alarm shift; exec @ARGV' "$cap" "$@" >"$out/$name.out" 2>"$out/$name.err"; then
    echo "   ok"
    last_ok=true
  else
    echo "   FAILED (exit $?)"
    failed=$((failed + 1))
    last_ok=false
  fi
  cat "$out/$name.out"
}

# A fitted constant read back from a cell's config-pasteable line -- only from a cell that
# succeeded: a fit its own tool rejected (a non-monotone split exits 1 after printing it) must not
# reach the later cells. Empty otherwise; the failure is already counted.
fitted() { if $last_ok; then sed -n "s/^$1=//p" "$2" | tail -1; fi; }

model=()
tiers=()
fitted_arm=fitted
if [[ ${RANK_ONLY:-0} == 1 ]]; then
  fitted_arm=shipped
  echo "   RANK_ONLY: agreement cells only, under the shipped constants"
else
  cell bandwidth "$bandwidth" "--ocannl_backend=$backend" \
    "--ocannl_autotune_calibration_file=$out/calibration.tsv"
  model=()
  bw=$(fitted model_peak_memory_bandwidth "$out/bandwidth.out")
  if [[ -n "$bw" ]]; then model+=("--ocannl_model_dram_bandwidth=$bw"); fi
  echo "   model flags after bandwidth: ${model[*]:-none}"
  cell calibrate "$calibrate" --repeats=10 "--ocannl_backend=$backend" ${model[@]+"${model[@]}"}
  capacity=$(fitted model_cache_capacity "$out/calibrate.out")
  if [[ -n "$capacity" ]]; then
    model+=("--ocannl_model_cache_capacity=$capacity")
  elif $last_ok; then
    # The sweep never reached DRAM: the capacity is only known to exceed every swept working set,
    # which covers the wgrad cells' (the same contraction, at most the same extent). No DRAM tier is
    # then the reading for them -- not the backend's advisory, which the sweep contradicts.
    model+=("--ocannl_model_cache_capacity=none")
  fi
  cache_bw=$(fitted model_cache_bandwidth "$out/calibrate.out")
  if [[ -n "$cache_bw" ]]; then model+=("--ocannl_model_cache_bandwidth=$cache_bw"); fi
  # The capacity and cache tier as fitted, without the rates: the ceilings arm of the agreement cells.
  tiers=(${model[@]+"${model[@]}"})
  cell issue "$issue" --repeats=10 "--ocannl_backend=$backend" ${model[@]+"${model[@]}"}
  rate=$(fitted model_issue_rate "$out/issue.out")
  if [[ -n "$rate" ]]; then model+=("--ocannl_model_issue_rate=$rate"); fi
  echo "   model flags for the wgrad cells: ${model[*]:-none}"
  for batch in 1 8 256; do
    dims="$batch,128,256,1024"
    cell "wgrad_b${batch}_untiled" env BENCH_TUNE=0 BENCH_WGRAD_REQUIRE_SEEDED=0 BENCH_WGRAD_DIMS="$dims" \
      BENCH_REPEATS=20 "$wgrad" \
      "--ocannl_backend=$backend" --ocannl_model_default_schedule=false \
      --ocannl_gpu_default_sketch_min_reduction=0 --ocannl_gpu_default_split_reduce_min_serial=0 \
      --ocannl_gpu_default_reduction_sink=false ${model[@]+"${model[@]}"}
    cell "wgrad_b${batch}_seeded" env BENCH_TUNE=0 BENCH_WGRAD_REQUIRE_SEEDED=1 BENCH_WGRAD_DIMS="$dims" \
      BENCH_REPEATS=20 "$wgrad" \
      "--ocannl_backend=$backend" --ocannl_model_default_schedule=false \
      --ocannl_gpu_default_sketch_min_reduction=32 --ocannl_gpu_default_split_reduce_min_serial=0 \
      --ocannl_gpu_default_reduction_sink=false ${model[@]+"${model[@]}"}
  done
fi

# Ranking agreement (gh-ocannl-1442). The ceilings arm drops the DRAM rate from the tiers too.
# Every arm pins the seeding gate at threshold 32, so the seeded incumbent -- model_explain's
# `seeded`, the tuner's F_seeded row -- is one plan whatever rates an arm scores under.
ceilings=()
for flag in ${tiers[@]+"${tiers[@]}"}; do
  case "$flag" in --ocannl_model_dram_bandwidth=*) ;; *) ceilings+=("$flag") ;; esac
done
ceilings+=(--ocannl_model_issue_rate=none --ocannl_model_dram_bandwidth=none)
for site in ${SITES:-wgrad:8,128,256,1024 wgrad:256,128,256,1024 fp8mm:8192,32,4096}; do
  kind=${site%%:*}
  dims=${site#*:}
  name="${kind}_${dims//,/x}"
  cell "rank_${name}_time" env BENCH_TUNE=1 BENCH_SITE="$kind" BENCH_SITE_DIMS="$dims" "$rank_site" \
    "--ocannl_backend=$backend" --ocannl_gpu_default_sketch_min_reduction=32 \
    --ocannl_autotune_keep_fraction=1 \
    "--ocannl_autotune_calibration_file=$out/rank_${name}.tsv" ${model[@]+"${model[@]}"}
  timed=$last_ok
  cell "rank_${name}_${fitted_arm}" env BENCH_SITE="$kind" BENCH_SITE_DIMS="$dims" "$rank_site" \
    "--ocannl_backend=$backend" --ocannl_gpu_default_sketch_min_reduction=32 \
    ${model[@]+"${model[@]}"}
  cell "rank_${name}_ceilings" env BENCH_SITE="$kind" BENCH_SITE_DIMS="$dims" "$rank_site" \
    "--ocannl_backend=$backend" --ocannl_gpu_default_sketch_min_reduction=32 "${ceilings[@]}"
  if $timed; then
    python3 "$root/benchmarks/gh1442_agreement.py" "$out/rank_${name}.tsv" \
      "ceilings=$out/rank_${name}_ceilings.out" \
      "${fitted_arm}=$out/rank_${name}_${fitted_arm}.out" |
      tee "$out/rank_${name}_agreement.txt" || failed=$((failed + 1))
  fi
done

echo "gh1253_cells: $failed failed cell(s)"
exit $((failed > 0 ? 1 : 0))
