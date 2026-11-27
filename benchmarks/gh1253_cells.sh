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
#            OCANNL_*, BENCH_* or OpenMP variable or config file reaches a cell (see below).
#   OUT      results directory, created, and refused unless empty: OUT/driver.log records what this
#            script prints, each cell writes OUT/<cell>.out and OUT/<cell>.err, and runs from
#            OUT/<cell>/, which keeps its cell.json record.
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
# CAPACITY_ONLY=1 (gh-ocannl-1687) runs the bandwidth and calibrate cells alone and stops: the
# DRAM rate and the cache capacity, and nothing else. It is the only mode BACKEND cc takes: cc's
# advisory model_cache_capacity is that cell's fit, on the plain default pipeline, whose Grid
# iterations the score reads as resident blocks (ahrefs/ocannl#1610). On cc the calibrate cell
# splits on its sweep's own rate contrast, over finer extents, not against the STREAM rate: the
# untiled contraction is issue-bound there, below DRAM bandwidth even when cached.
# RANK_ONLY=1 (gh-ocannl-1278) runs the agreement cells alone, under the shipped per-class
# constants: no calibration or wgrad cell runs, and the `fitted` arm is named `shipped` (the
# defaults, no model_* flag), beside `ceilings`. With it, SITES may name bench_rank_site's
# gpt2_mini projection sites (qkv:b,s,h,e,k and outproj:b,s,j,h,e), whose explain lines list the
# coalesced branch's leaves the tuner times beside the family.
# COMPARE_AUTO_GATE=1 compares the shipped auto gate with auto-legacy (the pre-gh-ocannl-2279
# score) on CUDA or HIP, at each site of SITES. Both arms use one bench_rank_site binary and
# the shipped leaf settings. This mode takes OUT, CAP, SITES and TIME_REPEATS (default 1).
# SITES is a space-separated list of distinct wgrad, fp8mm, qkv or outproj sites with positive
# decimal dimensions. It checks all sites and conflicting mode, calibration, pick and fit options
# before building.
# Each pair runs in alternating order across repeats, with one fresh TSV per arm per repeat.
# OUT/rank_<site>_gate_r<N>.txt joins each arm's seeding census to its own exact timed plan.
# A deduplicated preset joins by schedule digest and reports the candidate label that was timed.
# The join requires clean resolved configuration records that differ only in the gate and
# TSV destination. Missing evidence fails that pair; when a timing row is missing, the join prints
# each arm's timing-admission reasons from `rank-site: tune` beside the missing-sample count
# (gh-ocannl-2376). Inspect failed output in OUT and retry in an empty directory.
# Where the backend advises no operand-delivery rate, both modes resolve to the same gate and
# the pair report marks them as identical.
# BENCH_PICK=1 (stated to every cell) makes each listing cell also print model_default's actual
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
# rate on stock medians (from .tsv.stock when a profile timing cell appended, with the stock timing
# censuses passed via --census for preset-digest attribution) against the measured spread (p90/p10
# of the fastest admitted candidate);
# NO_COALESCED=1 for a backend that does not admit the coalesced layouts.
# gh-ocannl-1821: and a `nostg` arm (model_staged_copy_rate none: no staged-copy term, a continuous
# launch fill, and no coalesced layouts admitted through the copy rate) -- the fill-only reference
# the shipped arm is judged against; its kernel lines' stg= are what gh1600_txn_fit.py
# --staged-rate re-scores. With BENCH_PICK=1 each arm prints model_default's pick, so `nostg`'s is
# the pick before the term.
# gh-ocannl-1639: and a `noilp` arm (model_fill_chains none: the thread-only launch fill, no
# per-thread ILP credit) -- the reference the shipped arm's chain credit is judged against.
# FILL_CHAINS=L states the ILP cap to the fitted/shipped arm (and to EXPLAIN_ONLY's shipped arm,
# which is how a cap is fitted: one EXPLAIN_ONLY run per candidate L against one timed run's rows).
# gh-ocannl-2262: and `noshl` (model_shared_load_rate none: no register-reuse term) and `nolane`
# (model_fill_lane_width none: the thread-only launch fill) arms -- each the shipped constants with
# one term unset, the references the shipped arm's two terms are judged against.
# SHARED_LOAD_RATE=R and FILL_LANE_WIDTH=W state them to the fitted/shipped arm (and to
# EXPLAIN_ONLY's shipped arm, as FILL_CHAINS does); its kernel lines' shl= (the filled excess
# shared loads, listed whether or not a rate prices them) are what gh1600_txn_fit.py
# --shared-rate re-scores.
# gh-ocannl-2280: and `noidl` (model_staged_idle_rate none: no partition-guarded staged-idle term).
# STAGED_COPY_RATE=R and STAGED_IDLE_RATE=R state them to the fitted/shipped arm (and to
# EXPLAIN_ONLY's shipped arm); its kernel lines' idl= (the filled refused workgroup bytes, listed
# whether or not a rate prices them) are what gh1600_txn_fit.py --idle-rate re-scores.
# FIT_FILL=1 with RANK_ONLY=1 runs the issue cell first (MULTIPROCESSORS=P passes --multiprocessors,
# needed only where the device reports none: a non-AGX Metal GPU) and passes its fitted constants to the listing and pick
# cells, which otherwise run the shipped ones. PICK_GATE0=1 adds pick cells with the seeding off
# (gpu_default_sketch_min_reduction=0), with and without the fill (gate0, gate0nofill):
# model_default against the plain default, no anchor. A pick
# whose configuration seeds a plan the timing cell did not time (a profile's auto gate deciding
# otherwise) gets a timing cell of its own under that profile, appended to the same TSV. Before
# that append the stock rows are saved as rank_<site>.tsv.stock and the main TSV is marked as
# requiring the snapshot. Keep both files for replay.
# EXPLAIN_ONLY=<run dir> re-scores a past run's timed sites at this checkout's revision, timing
# no candidate (a correctness batch, not a measurement): per site of SITES only the `shipped` and
# `ceilings` listing cells run (the RANK_ONLY constants, BENCH_TUNE=0 pinned), and
# gh1442_agreement.py joins them to <run dir>/rank_<site>.tsv -- that run's rows, read and never
# written. Preset attribution comes from the stock _time.err and _time_rN.err censuses when
# retained, with the stock rows from .tsv.stock when present.
# BENCH_PICK=1 adds each arm's pick line, whose pick_ms/untuned_ms are the cell's own
# in-process timing (advisory: a rerun moves them, nothing else). Running it at two revisions
# against one run dir is the before/after agreement check of a ranking-score change, on the same
# measured rows.
# It refuses before building anything when a site's TSV is missing or empty or a marked TSV lacks
# its .tsv.stock (stock_snapshot_missing). It also refuses a timing knob (FIT_FILL, TIME_REPEATS,
# PICK_PROFILES, PICK_GATE0, TXN_RATE). It ends with each site's `joined=` lines, prefixed by site.
# Both wgrad arms pin BENCH_TUNE=0 and model_default_schedule=false, so an ambient profile or
# environment cannot swap the measured plans for tuned or model-picked ones.
# Every cell is one of the shared runner's (benchmarks/cells.py `cell`, gh-ocannl-1607): hermetic
# (no OCANNL_*, BENCH_* or OpenMP variable inherited, no config file read; a runner mode is stated
# per cell), supervised under CAP (a capped group is killed whole; one not proven gone stops the
# driver, exit 125), run from a fresh directory of its own rather than the checkout's root, and
# configuration-checked: its resolved-configuration records must be its command line, and a cell
# that ran under anything else failed.
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
stock_snapshot_marker='# gh1253_cells: stock snapshot required'
site_name() {
  local kind=${1%%:*} dims=${1#*:}
  echo "${kind}_${dims//,/x}"
}
gate_usage() {
  echo "$0: COMPARE_AUTO_GATE: $*" >&2
  exit 2
}
compare_gate=${COMPARE_AUTO_GATE:-0}
case "$compare_gate" in 0 | 1) ;; *) gate_usage "expected 0 or 1";; esac
if [[ $compare_gate == 1 ]]; then
  case "$backend" in cuda | hip) ;; *) gate_usage "BACKEND must be cuda or hip";; esac
  for knob in RANK_ONLY CAPACITY_ONLY FIT_FILL PICK_GATE0 BENCH_PICK NO_COALESCED; do
    [[ ${!knob:-0} == 0 ]] || gate_usage "$knob conflicts with this mode"
  done
  for knob in EXPLAIN_ONLY PICK_PROFILES TXN_RATE MULTIPROCESSORS FILL_CHAINS SHARED_LOAD_RATE FILL_LANE_WIDTH STAGED_COPY_RATE STAGED_IDLE_RATE; do
    [[ -z ${!knob:-} ]] || gate_usage "$knob conflicts with this mode"
  done
  gate_repeats=${TIME_REPEATS:-1}
  integer='^[1-9][0-9]*$'
  positive_decimal='^[0-9]+([.][0-9]+)?$'
  [[ $gate_repeats =~ $integer ]] || gate_usage "TIME_REPEATS must be a positive integer"
  [[ $cap =~ $positive_decimal && $cap == *[1-9]* ]] || gate_usage "CAP must be positive seconds"
  [[ $sites != *$'\n'* ]] || gate_usage "SITES must fit on one line"
  read -r -a gate_sites <<< "$sites"
  seen=' '
  for site in ${gate_sites[@]+"${gate_sites[@]}"}; do
    case "${site%%:*}" in
      wgrad) arity=3 ;;
      fp8mm) arity=2 ;;
      qkv | outproj) arity=4 ;;
      *) gate_usage "unknown site '$site'" ;;
    esac
    shape="^[^:]+:[1-9][0-9]*(,[1-9][0-9]*){$arity}$"
    [[ $site =~ $shape ]] || gate_usage "invalid dimensions for '$site'"
    [[ $seen != *" $site "* ]] || gate_usage "duplicate site '$site'"
    seen+="$site "
  done
  [[ $seen != ' ' ]] || gate_usage "SITES must name at least one site"
fi
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
    elif [[ ! -f "$tsv.stock" ]] && grep -qxF "$stock_snapshot_marker" "$tsv"; then
      echo "$0: EXPLAIN_ONLY: stock_snapshot_missing: $tsv.stock is required by $tsv" >&2
      missing=$((missing + 1))
    fi
  done
  if [[ $missing -gt 0 ]]; then exit 2; fi
fi
if [[ ${CAPACITY_ONLY:-0} == 1 && ( ${RANK_ONLY:-0} == 1 || -n "$rows_dir" ) ]]; then
  echo "$0: CAPACITY_ONLY runs the calibration cells alone; it takes neither RANK_ONLY nor EXPLAIN_ONLY" >&2
  exit 2
fi
case "$backend" in
cuda | hip) ;;
cc)
  if [[ ${CAPACITY_ONLY:-0} != 1 ]]; then
    echo "$0: BACKEND cc needs CAPACITY_ONLY=1 (its seeding, wgrad and agreement cells are GPU-shaped)" >&2
    exit 2
  fi
  ;;
metal)
  if [[ ${RANK_ONLY:-0} != 1 && -z "$rows_dir" ]]; then
    echo "$0: BACKEND metal needs RANK_ONLY=1 or EXPLAIN_ONLY (the calibration cells are CUDA/HIP only)" >&2
    exit 2
  fi
  ;;
*)
  echo "$0: BACKEND must be cuda, hip, metal or cc, got $backend" >&2
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

# No ambient configuration reaches a cell: the shared runner drops every OCANNL_* variable and pins
# no config file, so each cell runs the defaults plus exactly the flags written below (an inherited
# OCANNL_SCHEDULE_FISSION=false, say, would make the seeded arm decline silently). BENCH_PICK, the
# one runner mode this driver takes from its caller, is stated to every cell, as it was inherited.
pick_env=()
if [[ -n "${BENCH_PICK:-}" ]]; then pick_env=("BENCH_PICK=$BENCH_PICK"); fi

root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"
echo "gh1253_cells: revision $(git rev-parse HEAD) backend $backend host $(hostname) at $(date -u +%FT%TZ)"
git status --porcelain --untracked-files=no | sed 's/^/  dirty: /'

if [[ $compare_gate == 1 ]]; then
  dune build -j 4 benchmarks/runners/ocannl/bench_rank_site.exe
elif [[ -n "$rows_dir" ]]; then
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
# [cell NAME [env] [NAME=VALUE]... EXE ARGS...]: one cell of the shared runner, its stated runner
# modes the leading NAME=VALUE words.
cell() {
  local name=$1 rc=0
  shift
  if [[ $1 == env ]]; then shift; fi
  echo "== cell $name: $*"
  python3 "$root/benchmarks/cells.py" cell --backend "$backend" --dir "$out/$name" \
    --stdout "$out/$name.out" --stderr "$out/$name.err" --timeout "$cap" -- \
    ${pick_env[@]+"${pick_env[@]}"} "$@" || rc=$?
  if [[ $rc -eq 0 ]]; then
    echo "   ok"
    last_ok=true
  else
    echo "   FAILED (exit $rc)"
    if [[ $rc -eq 125 ]]; then
      echo "gh1253_cells: a cell's process group outlived its cleanup; stopping"
      exit 125
    fi
    failed=$((failed + 1))
    last_ok=false
  fi
  cat "$out/$name.out"
}

if [[ $compare_gate == 1 ]]; then
  for site in "${gate_sites[@]}"; do
    name=$(site_name "$site")
    for rep in $(seq 1 "$gate_repeats"); do
      pair="rank_${name}_gate_r$rep"
      arms=(auto auto-legacy)
      if ((rep % 2 == 0)); then arms=(auto-legacy auto); fi
      pair_ok=true
      for gate in "${arms[@]}"; do
        arm="${pair}_$gate"
        cell "$arm" env BENCH_TUNE=1 BENCH_SITE="${site%%:*}" BENCH_SITE_DIMS="${site#*:}" "$rank_site" \
          "--ocannl_gpu_default_sketch_min_reduction=$gate" --ocannl_autotune_log_seeding=true \
          --ocannl_autotune_keep_fraction=1 "--ocannl_autotune_calibration_file=$out/$arm.tsv"
        if ! $last_ok; then pair_ok=false; fi
      done
      if $pair_ok; then
        python3 "$root/benchmarks/gh1253_gate_compare.py" "$backend" "$out/$pair" |
          tee "$out/$pair.txt" || failed=$((failed + 1))
      fi
    done
  done
  echo "gh1253_cells: $failed failed gate cell(s) or join(s)"
  exit $((failed > 0 ? 1 : 0))
fi

# A fitted constant read back from a cell's config-pasteable line -- only from a cell that
# succeeded: a fit its own tool rejected (a non-monotone split exits 1 after printing it) must not
# reach the later cells. Empty otherwise; the failure is already counted.
fitted() { if $last_ok; then sed -n "s/^$1=//p" "$2" | tail -1; fi; }

# gh-ocannl-1639: the ILP cap under fit, stated to the arms that carry the model flags.
ilp_flag=()
if [[ -n "${FILL_CHAINS:-}" ]]; then ilp_flag=("--ocannl_model_fill_chains=$FILL_CHAINS"); fi
# gh-ocannl-2262: the register-reuse rate and the lane group under fit, carried the same way.
if [[ -n "${SHARED_LOAD_RATE:-}" ]]; then ilp_flag+=("--ocannl_model_shared_load_rate=$SHARED_LOAD_RATE"); fi
if [[ -n "${FILL_LANE_WIDTH:-}" ]]; then ilp_flag+=("--ocannl_model_fill_lane_width=$FILL_LANE_WIDTH"); fi
# gh-ocannl-2280: the staged-copy and staged-idle rates under fit, carried the same way.
if [[ -n "${STAGED_COPY_RATE:-}" ]]; then ilp_flag+=("--ocannl_model_staged_copy_rate=$STAGED_COPY_RATE"); fi
if [[ -n "${STAGED_IDLE_RATE:-}" ]]; then ilp_flag+=("--ocannl_model_staged_idle_rate=$STAGED_IDLE_RATE"); fi
model=(${ilp_flag[@]+"${ilp_flag[@]}"})
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
    # The cap under fit reaches the calibration too: its fit refuses a cap its sweep cannot identify.
    cell issue "$issue" --repeats=10 ${mp_flag[@]+"${mp_flag[@]}"} ${ilp_flag[@]+"${ilp_flag[@]}"}
    issue_flags
    echo "   FIT_FILL: listing and pick cells under ${model[*]:-the shipped constants}"
  fi
else
  cell bandwidth "$bandwidth" \
    "--ocannl_autotune_calibration_file=$out/calibration.tsv"
  model=(${ilp_flag[@]+"${ilp_flag[@]}"})
  bw=$(fitted model_peak_memory_bandwidth "$out/bandwidth.out")
  if [[ -n "$bw" ]]; then model+=("--ocannl_model_dram_bandwidth=$bw"); fi
  echo "   model flags after bandwidth: ${model[*]:-none}"
  calibrate_flags=(${model[@]+"${model[@]}"})
  if [[ $backend == cc ]]; then
    # gh-ocannl-1687: on cc the untiled contraction is issue-bound -- its requested rate sits far
    # below the STREAM rate on both sides of the knee (4-16 against 119 GB/s on minix's Zen 5),
    # so a split against the DRAM rate reads every point as spilled. The fit splits on the
    # sweep's own contrast instead (no DRAM rate), at two points per octave around the knee where
    # x (k x 1024 floats) outgrows the last-level cache.
    calibrate_flags=(--ocannl_model_dram_bandwidth=none
      --ks=256,512,1024,2048,3072,4096,6144,8192,12288,16384,24576,32768)
  fi
  cell calibrate "$calibrate" --repeats=10 ${calibrate_flags[@]+"${calibrate_flags[@]}"}
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
  if [[ ${CAPACITY_ONLY:-0} == 1 ]]; then
    echo "gh1253_cells: CAPACITY_ONLY: fitted ${model[*]:-nothing}; $failed cell(s) failed"
    if [[ $failed -gt 0 ]]; then exit 1; fi
    exit 0
  fi
  # The capacity and cache tier as fitted, without the rates: the ceilings arm of the agreement cells.
  tiers=(${model[@]+"${model[@]}"})
  cell issue "$issue" --repeats=10 ${mp_flag[@]+"${mp_flag[@]}"} \
    ${model[@]+"${model[@]}"}
  issue_flags
  echo "   model flags for the wgrad cells: ${model[*]:-none}"
  for batch in 1 8 256; do
    dims="$batch,128,256,1024"
    cell "wgrad_b${batch}_untiled" env BENCH_TUNE=0 BENCH_WGRAD_REQUIRE_SEEDED=0 BENCH_WGRAD_DIMS="$dims" "$wgrad" \
      --ocannl_model_default_schedule=false \
      --ocannl_gpu_default_sketch_min_reduction=0 --ocannl_gpu_default_split_reduce_min_serial=0 \
      --ocannl_gpu_default_reduction_sink=false ${model[@]+"${model[@]}"}
    cell "wgrad_b${batch}_seeded" env BENCH_TUNE=0 BENCH_WGRAD_REQUIRE_SEEDED=1 BENCH_WGRAD_DIMS="$dims" "$wgrad" \
      --ocannl_model_default_schedule=false \
      --ocannl_gpu_default_sketch_min_reduction=32 --ocannl_gpu_default_split_reduce_min_serial=0 \
      --ocannl_gpu_default_reduction_sink=false ${model[@]+"${model[@]}"}
  done
fi

# Ranking agreement (gh-ocannl-1442). The ceilings arm drops the DRAM rate from the tiers too.
# The stock listings pin the seeding gate at threshold 32. The timing census identifies the
# stock incumbent, including the configured preset when seeding makes no changes.
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
      --ocannl_gpu_default_sketch_min_reduction=32 ${ilp_flag[@]+"${ilp_flag[@]}"}
    cell "rank_${name}_ceilings" env BENCH_TUNE=0 BENCH_SITE="$kind" BENCH_SITE_DIMS="$dims" "$rank_site" \
      --ocannl_gpu_default_sketch_min_reduction=32 "${ceilings[@]}"
    fit=()
    if [[ $kind == fp8mm ]]; then fit=(--fit-mma "$out/rank_${name}_${fitted_arm}.out"); fi
    timing_census=()
    for log in "$rows_dir/rank_${name}_time.err" "$rows_dir/rank_${name}_time_r"[0-9]*.err; do
      if [[ -f "$log" ]]; then timing_census+=(--census "$log"); fi
    done
    rows="$rows_dir/rank_${name}.tsv"
    stock_tsv=()
    if [[ -f "$rows.stock" ]]; then stock_tsv=(--stock-tsv "$rows.stock"); fi
    python3 "$root/benchmarks/gh1442_agreement.py" "$rows" \
      "ceilings=$out/rank_${name}_ceilings.out" "${fitted_arm}=$out/rank_${name}_${fitted_arm}.out" \
      ${stock_tsv[@]+"${stock_tsv[@]}"} ${timing_census[@]+"${timing_census[@]}"} ${fit[@]+"${fit[@]}"} |
      tee "$out/rank_${name}_agreement.txt" || failed=$((failed + 1))
    continue
  fi
  timed=true
  rows="$out/rank_${name}.tsv"
  stock_rows="$rows"
  timing_census=()
  txn_census=()
  for rep in $(seq 1 "${TIME_REPEATS:-1}"); do
    suffix=""
    if [[ $rep -gt 1 ]]; then suffix="_r$rep"; fi
    cell "rank_${name}_time$suffix" env BENCH_TUNE=1 BENCH_SITE="$kind" BENCH_SITE_DIMS="$dims" "$rank_site" \
      --ocannl_gpu_default_sketch_min_reduction=32 \
      --ocannl_autotune_keep_fraction=1 --ocannl_autotune_log_seeding=true \
      "--ocannl_autotune_calibration_file=$rows" ${model[@]+"${model[@]}"}
    timing_census+=(--census "$out/rank_${name}_time$suffix.err")
    txn_census+=(--census "$name=$out/rank_${name}_time$suffix.err")
    if ! $last_ok; then timed=false; fi
  done
  cell "rank_${name}_${fitted_arm}" env BENCH_SITE="$kind" BENCH_SITE_DIMS="$dims" "$rank_site" \
    --ocannl_gpu_default_sketch_min_reduction=32 \
    ${model[@]+"${model[@]}"}
  cell "rank_${name}_ceilings" env BENCH_SITE="$kind" BENCH_SITE_DIMS="$dims" "$rank_site" \
    --ocannl_gpu_default_sketch_min_reduction=32 "${ceilings[@]}"
  # The first statement of a key wins, so the fill's own flags are dropped, not overridden.
  nofill=()
  for flag in ${model[@]+"${model[@]}"}; do
    case "$flag" in --ocannl_model_fill_threads=* | --ocannl_model_multiprocessors=*) ;; *) nofill+=("$flag") ;; esac
  done
  cell "rank_${name}_nofill" env BENCH_SITE="$kind" BENCH_SITE_DIMS="$dims" "$rank_site" \
    --ocannl_gpu_default_sketch_min_reduction=32 \
    ${nofill[@]+"${nofill[@]}"} --ocannl_model_fill_threads=none
  # gh-ocannl-1495: the same constants without its two terms (the score before it) and with the
  # tile-MMA rate alone, so one timing TSV gives the agreement of each step.
  nomma=()
  for flag in ${model[@]+"${model[@]}"}; do
    case "$flag" in --ocannl_model_mma_rate=* | --ocannl_model_device_gather_rate=*) ;; *) nomma+=("$flag") ;; esac
  done
  cell "rank_${name}_nomma" env BENCH_SITE="$kind" BENCH_SITE_DIMS="$dims" "$rank_site" \
    --ocannl_gpu_default_sketch_min_reduction=32 \
    ${nomma[@]+"${nomma[@]}"} --ocannl_model_mma_rate=none --ocannl_model_device_gather_rate=none
  cell "rank_${name}_mmaonly" env BENCH_SITE="$kind" BENCH_SITE_DIMS="$dims" "$rank_site" \
    --ocannl_gpu_default_sketch_min_reduction=32 \
    ${nomma[@]+"${nomma[@]}"} --ocannl_model_device_gather_rate=none
  # gh-ocannl-1600: the same constants without the transaction term, and without the occupancy
  # factor (which also leaves the coalesced layouts out of model_default).
  cell "rank_${name}_notxn" env BENCH_SITE="$kind" BENCH_SITE_DIMS="$dims" "$rank_site" \
    --ocannl_gpu_default_sketch_min_reduction=32 \
    ${model[@]+"${model[@]}"} --ocannl_model_transaction_rate=none
  cell "rank_${name}_noocc" env BENCH_SITE="$kind" BENCH_SITE_DIMS="$dims" "$rank_site" \
    --ocannl_gpu_default_sketch_min_reduction=32 \
    ${model[@]+"${model[@]}"} --ocannl_model_workgroup_bytes_per_multiprocessor=none
  # gh-ocannl-1821/2280: the same constants without the staged-copy term (and its launch tail),
  # and without the partition-guarded staged-idle term (first statement of a key wins).
  nostg=()
  noidl=()
  for flag in ${model[@]+"${model[@]}"}; do
    case "$flag" in --ocannl_model_staged_copy_rate=*) ;; *) nostg+=("$flag") ;; esac
    case "$flag" in --ocannl_model_staged_idle_rate=*) ;; *) noidl+=("$flag") ;; esac
  done
  cell "rank_${name}_nostg" env BENCH_SITE="$kind" BENCH_SITE_DIMS="$dims" "$rank_site" \
    --ocannl_gpu_default_sketch_min_reduction=32 \
    ${nostg[@]+"${nostg[@]}"} --ocannl_model_staged_copy_rate=none
  cell "rank_${name}_noidl" env BENCH_SITE="$kind" BENCH_SITE_DIMS="$dims" "$rank_site" \
    --ocannl_gpu_default_sketch_min_reduction=32 \
    ${noidl[@]+"${noidl[@]}"} --ocannl_model_staged_idle_rate=none
  # gh-ocannl-1639: the same constants with the thread-only launch fill (no ILP cap stated).
  noilp=()
  for flag in ${model[@]+"${model[@]}"}; do
    case "$flag" in --ocannl_model_fill_chains=*) ;; *) noilp+=("$flag") ;; esac
  done
  cell "rank_${name}_noilp" env BENCH_SITE="$kind" BENCH_SITE_DIMS="$dims" "$rank_site" \
    --ocannl_gpu_default_sketch_min_reduction=32 \
    ${noilp[@]+"${noilp[@]}"} --ocannl_model_fill_chains=none
  # gh-ocannl-2262: the same constants without the register-reuse term, and with the thread-only
  # launch fill's lane count (the first statement of a key wins, so a stated one is dropped).
  noshl=()
  nolane=()
  for flag in ${model[@]+"${model[@]}"}; do
    case "$flag" in --ocannl_model_shared_load_rate=*) ;; *) noshl+=("$flag") ;; esac
    case "$flag" in --ocannl_model_fill_lane_width=*) ;; *) nolane+=("$flag") ;; esac
  done
  cell "rank_${name}_noshl" env BENCH_SITE="$kind" BENCH_SITE_DIMS="$dims" "$rank_site" \
    --ocannl_gpu_default_sketch_min_reduction=32 \
    ${noshl[@]+"${noshl[@]}"} --ocannl_model_shared_load_rate=none
  cell "rank_${name}_nolane" env BENCH_SITE="$kind" BENCH_SITE_DIMS="$dims" "$rank_site" \
    --ocannl_gpu_default_sketch_min_reduction=32 \
    ${nolane[@]+"${nolane[@]}"} --ocannl_model_fill_lane_width=none
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
        "$rank_site" "$pflag" ${pflags[@]+"${pflags[@]}"}
      if $last_ok; then
        picks+=(--pick "${pname}=$pout")
        # The plan this configuration seeds, timed here unless the timing cell already did.
        # Read by bench_output's one reader of the rank-site lines (gh-ocannl-1767): the literal
        # decoded, an escaped quote in it kept; a line it refuses (named on stderr) leaves it empty.
        seeded=$(python3 "$root/benchmarks/bench_output.py" rank-site-field "$pout" pick seeded_label) || seeded=
        if $timed && [[ -n "$seeded" && "$seeded" != none ]] &&
          ! grep -qF "$seeded" "$rows"; then
          if [[ $stock_rows == "$rows" ]]; then
            stock_rows="$rows.stock"
            if [[ -f "$rows" ]]; then cp "$rows" "$stock_rows"; else : > "$stock_rows"; fi
            printf '%s\n' "$stock_snapshot_marker" >> "$rows"
          fi
          cell "rank_${name}_time_${pname}" env BENCH_TUNE=1 BENCH_SITE="$kind" BENCH_SITE_DIMS="$dims" \
            "$rank_site" "$pflag" --ocannl_autotune_keep_fraction=1 \
            "--ocannl_autotune_calibration_file=$rows" ${pflags[@]+"${pflags[@]}"}
        fi
      fi
    done
  fi
  if $timed; then
    # gh-ocannl-1495: the tile-MMA and device-gather rates, fitted on a site with tensorized
    # leaves (the fp8 matmul).
    fit=()
    if [[ $kind == fp8mm ]]; then fit=(--fit-mma "$out/rank_${name}_${fitted_arm}.out"); fi
    python3 "$root/benchmarks/gh1442_agreement.py" "$rows" \
      "ceilings=$out/rank_${name}_ceilings.out" "nofill=$out/rank_${name}_nofill.out" \
      "nomma=$out/rank_${name}_nomma.out" "mmaonly=$out/rank_${name}_mmaonly.out" \
      "noocc=$out/rank_${name}_noocc.out" "notxn=$out/rank_${name}_notxn.out" \
      "nostg=$out/rank_${name}_nostg.out" "noidl=$out/rank_${name}_noidl.out" \
      "noilp=$out/rank_${name}_noilp.out" \
      "noshl=$out/rank_${name}_noshl.out" "nolane=$out/rank_${name}_nolane.out" \
      "${fitted_arm}=$out/rank_${name}_${fitted_arm}.out" --stock-tsv "$stock_rows" \
      ${timing_census[@]+"${timing_census[@]}"} \
      ${fit[@]+"${fit[@]}"} \
      ${picks[@]+"${picks[@]}"} |
      tee "$out/rank_${name}_agreement.txt" || failed=$((failed + 1))
    if [[ -n "${TXN_RATE:-}" ]]; then
      # gh-ocannl-1600: the shipped transaction rate, judged on the stock medians (held out from
      # the cells it was fitted on) against the stock measured spread; uses .tsv.stock when mixed
      # and the stock timing censuses for preset-digest attribution.
      nocoal=()
      if [[ ${NO_COALESCED:-0} == 1 ]]; then nocoal=(--no-coalesced); fi
      python3 "$root/benchmarks/gh1600_txn_fit.py" --rate "$TXN_RATE" --show ${nocoal[@]+"${nocoal[@]}"} \
        ${txn_census[@]+"${txn_census[@]}"} \
        "$name=$stock_rows,$out/rank_${name}_notxn.out" |
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
