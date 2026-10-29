#!/bin/bash
# gh-ocannl-1253 measurement driver: the cost model's ranking-score capacity, calibrated and checked
# on one box, so the PR's predicted-vs-measured table quotes invocations rather than restating them.
# gh-ocannl-1442 extends it with the score's own rates (the issue rate and the DRAM rate) and with
# ranking-agreement cells, which time every whole-routine candidate at a site and compare the
# score's order with the measured one.
#
# Usage (from the checkout under test): benchmarks/gh1253_cells.sh BACKEND OUT [CAP]
#
#   BACKEND  cuda | hip, or metal with RANK_ONLY=1 -- pinned on every command line; no ambient
#            OCANNL_* variable or config file reaches a cell (see below).
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
#                      (not the validation cells' 256 x 1024) swept over reduction extents, and
#                      (gh-ocannl-1479) the launch sweep of pinned geometries over output sizes;
#                      prints the fitted model_issue_rate (full-fill when the launch fit holds),
#                      model_fill_threads and model_multiprocessors, passed to every later cell,
#                      and the intercept the score does not price.
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
# BENCH_PICK=1 (inherited by the cells) makes each listing cell also print model_default's actual
# pick, which the agreement script joins to its measured row against the seeded plan's
# (gh-ocannl-1477). PICK_PROFILES (with RANK_ONLY=1, say 'performance approximate') adds, per site,
# a pick-only cell rank_<site>_pick_<profile> under --ocannl_profile=<profile> with the shipped
# seeding gate (no threshold pin): what that profile ships, joined to the same measured rows.
# gh-ocannl-1479, the launch fill: every listing cell gains a `nofill` arm, the same constants with
# model_fill_threads=none, so one timing TSV gives the agreement before and after the fill term.
# gh-ocannl-1495: and `nomma` (model_mma_rate and model_device_gather_rate none: the score before
# them) and `mmaonly` (the gather rate none) arms; at an fp8mm site the agreement also fits the two
# rates from the `fitted`/`shipped` arm's explain lines (gh1442_agreement.py --fit-mma).
# gh-ocannl-1600: and `notxn` (model_transaction_rate none) and `noocc`
# (model_workgroup_bytes_per_multiprocessor none) arms -- the score without each of its two terms,
# so one timing TSV gives each term's effect -- and TIME_REPEATS=N (default 1) times every candidate
# N times into the same TSV, so benchmarks/gh1600_txn_fit.py, run on the `notxn` listing at the
# shipped rate (TXN_RATE=R, the backend's advisory; unset: no transaction-term check), judges the
# rate on medians against the measured spread (p90/p10 of the fastest admitted candidate);
# NO_COALESCED=1 for a backend that does not admit the coalesced layouts (CUDA).
# FIT_FILL=1 with RANK_ONLY=1 runs the issue cell first (MULTIPROCESSORS=P passes --multiprocessors,
# needed only where the device reports none: a non-AGX Metal GPU) and passes its fitted constants to the listing and pick
# cells, which otherwise run the shipped ones. PICK_GATE0=1 adds pick cells with the seeding off
# (gpu_default_sketch_min_reduction=0), with and without the fill (gate0, gate0nofill):
# model_default against the plain default, no anchor. A pick
# whose configuration seeds a plan the timing cell did not time (a profile's auto gate deciding
# otherwise) gets a timing cell of its own under that profile, appended to the same TSV.
# EXPLAIN_ONLY=<run dir> re-scores a past run's timed sites at this checkout's revision, timing
# no candidate (a correctness batch, not a measurement): per site of SITES only the `shipped` and
# `ceilings` listing cells run (the RANK_ONLY constants, BENCH_TUNE=0 pinned), and
# gh1442_agreement.py joins them to <run dir>/rank_<site>.tsv -- that run's rows, read and never
# written. BENCH_PICK=1 adds each arm's pick line, whose pick_ms/untuned_ms are the cell's own
# in-process timing (advisory: a rerun moves them, nothing else). Running it at two revisions
# against one run dir is the before/after agreement check of a ranking-score change, on the same
# measured rows.
# It refuses before building anything when a site's TSV is missing or empty, and when a timing
# knob is set (FIT_FILL, TIME_REPEATS, PICK_PROFILES, PICK_GATE0, TXN_RATE); it ends with each
# site's `joined=` lines, prefixed by site.
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
sites=${SITES:-wgrad:8,128,256,1024 wgrad:256,128,256,1024 fp8mm:8192,32,4096}
site_name() {
  local kind=${1%%:*} dims=${1#*:}
  echo "${kind}_${dims//,/x}"
}
rows_dir=""
if [[ -n "${EXPLAIN_ONLY:-}" ]]; then
  for knob in FIT_FILL PICK_GATE0; do
    if [[ ${!knob:-0} != 0 ]]; then
      echo "$0: EXPLAIN_ONLY times nothing, but $knob=${!knob} asks for cells it does not run" >&2
      exit 2
    fi
  done
  for knob in PICK_PROFILES TXN_RATE; do
    if [[ -n "${!knob:-}" ]]; then
      echo "$0: EXPLAIN_ONLY times nothing, but $knob=${!knob} asks for cells it does not run" >&2
      exit 2
    fi
  done
  if [[ ${TIME_REPEATS:-1} != 1 ]]; then
    echo "$0: EXPLAIN_ONLY times nothing, but TIME_REPEATS=$TIME_REPEATS asks for cells it does not run" >&2
    exit 2
  fi
  if [[ ! -d "$EXPLAIN_ONLY" ]]; then
    echo "$0: EXPLAIN_ONLY=$EXPLAIN_ONLY is not a directory" >&2
    exit 2
  fi
  rows_dir=$(cd "$EXPLAIN_ONLY" && pwd) # absolute: the cells run from the checkout's root
  missing=0
  for site in $sites; do
    tsv="$rows_dir/rank_$(site_name "$site").tsv"
    if [[ ! -s "$tsv" ]]; then
      echo "$0: EXPLAIN_ONLY: no timing rows for site $site: $tsv is missing or empty" >&2
      missing=$((missing + 1))
    fi
  done
  if [[ $missing -gt 0 ]]; then exit 2; fi
fi
case "$backend" in
cuda | hip) ;;
metal)
  if [[ ${RANK_ONLY:-0} != 1 && -z "$rows_dir" ]]; then
    echo "$0: BACKEND metal needs RANK_ONLY=1 or EXPLAIN_ONLY (the calibration cells are CUDA/HIP only)" >&2
    exit 2
  fi
  ;;
*)
  echo "$0: BACKEND must be cuda, hip or metal, got $backend" >&2
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

if [[ -n "$rows_dir" ]]; then
  echo "   EXPLAIN_ONLY: the shipped and ceilings listings joined to the rows in $rows_dir"
  if [[ -s "$rows_dir/driver.log" ]]; then sed -n '1s/^/   rows from: /p' "$rows_dir/driver.log"; fi
  dune build benchmarks/runners/ocannl/bench_rank_site.exe
else
  dune build tools/calibrate_bandwidth.exe tools/calibrate_cache_capacity.exe \
    tools/calibrate_issue_rate.exe benchmarks/runners/ocannl/bench_wgrad.exe \
    benchmarks/runners/ocannl/bench_rank_site.exe
fi
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
# The issue cell's fitted constants: the rate always, the launch fill's two when its fit held.
issue_flags() {
  local rate fill mp
  rate=$(fitted model_issue_rate "$out/issue.out")
  fill=$(fitted model_fill_threads "$out/issue.out")
  mp=$(fitted model_multiprocessors "$out/issue.out")
  if [[ -n "$rate" ]]; then model+=("--ocannl_model_issue_rate=$rate"); fi
  if [[ -n "$fill" ]]; then model+=("--ocannl_model_fill_threads=$fill"); fi
  if [[ -n "$mp" ]]; then model+=("--ocannl_model_multiprocessors=$mp"); fi
}
mp_flag=()
if [[ -n "${MULTIPROCESSORS:-}" ]]; then mp_flag=("--multiprocessors=$MULTIPROCESSORS"); fi
if [[ -n "$rows_dir" ]]; then
  fitted_arm=shipped
elif [[ ${RANK_ONLY:-0} == 1 ]]; then
  fitted_arm=shipped
  echo "   RANK_ONLY: agreement cells only, under the shipped constants"
  if [[ ${FIT_FILL:-0} == 1 ]]; then
    fitted_arm=fitted
    cell issue "$issue" --repeats=10 "--ocannl_backend=$backend" ${mp_flag[@]+"${mp_flag[@]}"}
    issue_flags
    echo "   FIT_FILL: listing and pick cells under ${model[*]:-the shipped constants}"
  fi
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
  cell issue "$issue" --repeats=10 "--ocannl_backend=$backend" ${mp_flag[@]+"${mp_flag[@]}"} \
    ${model[@]+"${model[@]}"}
  issue_flags
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
ceilings+=(--ocannl_model_issue_rate=none --ocannl_model_dram_bandwidth=none
  --ocannl_model_fill_threads=none --ocannl_model_mma_rate=none
  --ocannl_model_device_gather_rate=none)
for site in $sites; do
  kind=${site%%:*}
  dims=${site#*:}
  name=$(site_name "$site")
  if [[ -n "$rows_dir" ]]; then
    cell "rank_${name}_${fitted_arm}" env BENCH_TUNE=0 BENCH_SITE="$kind" BENCH_SITE_DIMS="$dims" "$rank_site" \
      "--ocannl_backend=$backend" --ocannl_gpu_default_sketch_min_reduction=32
    cell "rank_${name}_ceilings" env BENCH_TUNE=0 BENCH_SITE="$kind" BENCH_SITE_DIMS="$dims" "$rank_site" \
      "--ocannl_backend=$backend" --ocannl_gpu_default_sketch_min_reduction=32 "${ceilings[@]}"
    fit=()
    if [[ $kind == fp8mm ]]; then fit=(--fit-mma "$out/rank_${name}_${fitted_arm}.out"); fi
    python3 "$root/benchmarks/gh1442_agreement.py" "$rows_dir/rank_${name}.tsv" \
      "ceilings=$out/rank_${name}_ceilings.out" "${fitted_arm}=$out/rank_${name}_${fitted_arm}.out" \
      ${fit[@]+"${fit[@]}"} | tee "$out/rank_${name}_agreement.txt" || failed=$((failed + 1))
    continue
  fi
  timed=true
  for rep in $(seq 1 "${TIME_REPEATS:-1}"); do
    suffix=""
    if [[ $rep -gt 1 ]]; then suffix="_r$rep"; fi
    cell "rank_${name}_time$suffix" env BENCH_TUNE=1 BENCH_SITE="$kind" BENCH_SITE_DIMS="$dims" "$rank_site" \
      "--ocannl_backend=$backend" --ocannl_gpu_default_sketch_min_reduction=32 \
      --ocannl_autotune_keep_fraction=1 \
      "--ocannl_autotune_calibration_file=$out/rank_${name}.tsv" ${model[@]+"${model[@]}"}
    if ! $last_ok; then timed=false; fi
  done
  cell "rank_${name}_${fitted_arm}" env BENCH_SITE="$kind" BENCH_SITE_DIMS="$dims" "$rank_site" \
    "--ocannl_backend=$backend" --ocannl_gpu_default_sketch_min_reduction=32 \
    ${model[@]+"${model[@]}"}
  cell "rank_${name}_ceilings" env BENCH_SITE="$kind" BENCH_SITE_DIMS="$dims" "$rank_site" \
    "--ocannl_backend=$backend" --ocannl_gpu_default_sketch_min_reduction=32 "${ceilings[@]}"
  # The first statement of a key wins, so the fill's own flags are dropped, not overridden.
  nofill=()
  for flag in ${model[@]+"${model[@]}"}; do
    case "$flag" in --ocannl_model_fill_threads=* | --ocannl_model_multiprocessors=*) ;; *) nofill+=("$flag") ;; esac
  done
  cell "rank_${name}_nofill" env BENCH_SITE="$kind" BENCH_SITE_DIMS="$dims" "$rank_site" \
    "--ocannl_backend=$backend" --ocannl_gpu_default_sketch_min_reduction=32 \
    ${nofill[@]+"${nofill[@]}"} --ocannl_model_fill_threads=none
  # gh-ocannl-1495: the same constants without its two terms (the score before it) and with the
  # tile-MMA rate alone, so one timing TSV gives the agreement of each step.
  nomma=()
  for flag in ${model[@]+"${model[@]}"}; do
    case "$flag" in --ocannl_model_mma_rate=* | --ocannl_model_device_gather_rate=*) ;; *) nomma+=("$flag") ;; esac
  done
  cell "rank_${name}_nomma" env BENCH_SITE="$kind" BENCH_SITE_DIMS="$dims" "$rank_site" \
    "--ocannl_backend=$backend" --ocannl_gpu_default_sketch_min_reduction=32 \
    ${nomma[@]+"${nomma[@]}"} --ocannl_model_mma_rate=none --ocannl_model_device_gather_rate=none
  cell "rank_${name}_mmaonly" env BENCH_SITE="$kind" BENCH_SITE_DIMS="$dims" "$rank_site" \
    "--ocannl_backend=$backend" --ocannl_gpu_default_sketch_min_reduction=32 \
    ${nomma[@]+"${nomma[@]}"} --ocannl_model_device_gather_rate=none
  # gh-ocannl-1600: the same constants without the transaction term, and without the occupancy
  # factor (which also leaves the coalesced layouts out of model_default).
  cell "rank_${name}_notxn" env BENCH_SITE="$kind" BENCH_SITE_DIMS="$dims" "$rank_site" \
    "--ocannl_backend=$backend" --ocannl_gpu_default_sketch_min_reduction=32 \
    ${model[@]+"${model[@]}"} --ocannl_model_transaction_rate=none
  cell "rank_${name}_noocc" env BENCH_SITE="$kind" BENCH_SITE_DIMS="$dims" "$rank_site" \
    "--ocannl_backend=$backend" --ocannl_gpu_default_sketch_min_reduction=32 \
    ${model[@]+"${model[@]}"} --ocannl_model_workgroup_bytes_per_multiprocessor=none
  picks=()
  if [[ ${RANK_ONLY:-0} == 1 ]]; then
    pick_cells=()
    for profile in ${PICK_PROFILES:-}; do pick_cells+=("$profile:--ocannl_profile=$profile"); done
    if [[ ${PICK_GATE0:-0} == 1 ]]; then
      # The gate-0 pick with and without the fill: what nothing-seeded model_default ships.
      pick_cells+=("gate0:--ocannl_gpu_default_sketch_min_reduction=0")
      pick_cells+=("gate0nofill:--ocannl_gpu_default_sketch_min_reduction=0")
    fi
    for spec in ${pick_cells[@]+"${pick_cells[@]}"}; do
      pname=${spec%%:*}
      pflag=${spec#*:}
      pout="$out/rank_${name}_pick_${pname}.out"
      pflags=(${model[@]+"${model[@]}"})
      if [[ $pname == *nofill ]]; then pflags=(${nofill[@]+"${nofill[@]}"} --ocannl_model_fill_threads=none); fi
      cell "rank_${name}_pick_${pname}" env BENCH_PICK=1 BENCH_SITE="$kind" BENCH_SITE_DIMS="$dims" \
        "$rank_site" "--ocannl_backend=$backend" "$pflag" ${pflags[@]+"${pflags[@]}"}
      if $last_ok; then
        picks+=(--pick "${pname}=$pout")
        # The plan this configuration seeds, timed here unless the timing cell already did.
        seeded=$(sed -n 's/.*seeded_label="\([^"]*\)".*/\1/p' "$pout" | tail -1)
        if $timed && [[ -n "$seeded" && "$seeded" != none ]] &&
          ! grep -qF "$seeded" "$out/rank_${name}.tsv"; then
          cell "rank_${name}_time_${pname}" env BENCH_TUNE=1 BENCH_SITE="$kind" BENCH_SITE_DIMS="$dims" \
            "$rank_site" "--ocannl_backend=$backend" "$pflag" --ocannl_autotune_keep_fraction=1 \
            "--ocannl_autotune_calibration_file=$out/rank_${name}.tsv" ${pflags[@]+"${pflags[@]}"}
        fi
      fi
    done
  fi
  if $timed; then
    # gh-ocannl-1495: the tile-MMA and device-gather rates, fitted on a site with tensorized
    # leaves (the fp8 matmul).
    fit=()
    if [[ $kind == fp8mm ]]; then fit=(--fit-mma "$out/rank_${name}_${fitted_arm}.out"); fi
    python3 "$root/benchmarks/gh1442_agreement.py" "$out/rank_${name}.tsv" \
      "ceilings=$out/rank_${name}_ceilings.out" "nofill=$out/rank_${name}_nofill.out" \
      "nomma=$out/rank_${name}_nomma.out" "mmaonly=$out/rank_${name}_mmaonly.out" \
      "noocc=$out/rank_${name}_noocc.out" "notxn=$out/rank_${name}_notxn.out" \
      "${fitted_arm}=$out/rank_${name}_${fitted_arm}.out" ${fit[@]+"${fit[@]}"} \
      ${picks[@]+"${picks[@]}"} |
      tee "$out/rank_${name}_agreement.txt" || failed=$((failed + 1))
    if [[ -n "${TXN_RATE:-}" ]]; then
      # gh-ocannl-1600: the shipped transaction rate, judged on the medians of this TSV (held out
      # from the cells it was fitted on) against its measured spread.
      nocoal=()
      if [[ ${NO_COALESCED:-0} == 1 ]]; then nocoal=(--no-coalesced); fi
      python3 "$root/benchmarks/gh1600_txn_fit.py" --rate "$TXN_RATE" --show ${nocoal[@]+"${nocoal[@]}"} \
        "$name=$out/rank_${name}.tsv,$out/rank_${name}_notxn.out" |
        tee "$out/rank_${name}_txn.txt" || failed=$((failed + 1))
    fi
  fi
done

if [[ -n "$rows_dir" ]]; then
  echo "gh1253_cells: agreement against $rows_dir"
  for site in $sites; do
    name=$(site_name "$site")
    sed -n "s/^\([a-z]*: joined=\)/$name \1/p" "$out/rank_${name}_agreement.txt" 2>/dev/null || true
  done
fi
echo "gh1253_cells: $failed failed cell(s)"
exit $((failed > 0 ? 1 : 0))
