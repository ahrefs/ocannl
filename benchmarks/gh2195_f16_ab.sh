#!/bin/bash
# gh-ocannl-2195 (CUDA, HIP) and gh-ocannl-2372 (Metal): throughput A/B of a GPU backend's two f16
# accumulator arms -- fp16_arithmetic=false (Fp16_wide: f32 reduction accumulators, the
# f32-accumulate uniform-f16 tensor-unit arms behind a converted f16 boundary) against
# fp16_arithmetic=true (Fp16_narrow: storage-width f16 accumulators and the f16-accumulate
# tensor-unit arms) -- the cost side of resolving auto wide, whose accuracy side is
# test/operations/f16_accum_ab. The arms are named `false` and `true`, not `auto`: auto resolved
# narrow on CUDA and HIP before gh-ocannl-2195 and resolves wide there since, while explicit values
# name the same arm on both sides of any flip, so the arms mean the same at every revision this
# driver can measure (below). Which arm `auto` is at the measured revision is REPORTED from the
# generated sources (the identity proof below).
#
# Usage: gh2195_f16_ab.sh [--box BOX | --skip-box-check] [--repeats N]
#                         [--training-fixture PATH [--tuned-training] | --untuned-fixture PATH]
#                         <repo-root> <backend cuda|hip|metal> <memory-label> [rounds] [sizes...]
#   e.g. gh2195_f16_ab.sh ~/ocannl cuda "GDDR7 (RTX 5070 Ti Laptop, sm_120)" 8 1024 2048
#        gh2195_f16_ab.sh --box minix --untuned-fixture ~/ocannl-staging/benchmarks/fixtures/gpt2_mini_train.safetensors \
#          ~/ocannl hip "unified (gfx1151)" 8 1024 2048
#        gh2195_f16_ab.sh --box m4-max --repeats 50 --tuned-training \
#          --training-fixture ~/ocannl-staging/benchmarks/fixtures/gpt2_mini_train.safetensors \
#          ~/ocannl metal "unified (M4 Max)" 12 1024 2048
#
# [rounds] must be even and positive (default 8); [sizes] must be multiples of 64 (regtile's
# block; the mma variants need 32) or those variants are skipped and the run is invalid (default
# 1024 2048). The cells are bin_cells.py's (gh-ocannl-1607): per size one run of four arms -- the
# two fp16_arithmetic arms (`false`, `true`) at half and at f32 (the control below), named
# <arm>-<prec> -- visited round by round in Bench_design's spread mirrored rotation, so every arm's
# position histogram is equal at a multiple of four rounds, and every pair of arms exchanges
# positions within each pair of rounds. Each cell is hermetic (no OCANNL_*, BENCH_* or OpenMP
# variable inherited, no config file read), supervised, and configuration-checked: its
# resolved-configuration records must be its command line, so the library must carry the stream
# (gh-ocannl-1566): measure an older revision with the driver of its own time.
#
# --repeats is the positive number of schedule_bench repetitions per GEMM cell (default 10).
# HAND-SCHEDULED ROWS. Each treatment cell is bin/schedule_bench at m = n = k = <size> with f16
# operands and output. All five variants are the treatment: the tensorized mma_pd1/mma_pd2 (staged
# Tile_mma; the policy picks their arm -- on CUDA sm_80+ the f32-accumulate m16n8k16 / converted
# wmma arm against wmma's f16-accumulate fragments, on HIP the f32 against the f16 WMMA
# accumulator, on Metal the f32-accumulate simdgroup_float8x8 with per-simdgroup scratch against
# simdgroup_half8x8) and the serial legs parallel/smem/regtile, whose accumulator residency the
# policy moves too (the backend's accum_prec: f32 or half). No schedule cache is involved: every
# variant is a hand-written schedule compiled through [~lowered_transform].
#
# TRAINING ROW (only with --training-fixture or its alias --untuned-fixture). In untuned mode
# (default), runs bench_gpt's untuned training step (BENCH_TUNE=0, so Bench_harness.compile_default
# = Autotune.untuned_compile; no search, model_default off) on the given recorded fixture -- never
# a regenerated one -- under BENCH_PRECISION=f16 (f16 activations and gradients, f32 masters and
# loss head, dynamic loss scaling), one bin_cells --reader record run of the same four arms
# (<arm>-half at BENCH_PRECISION=f16, <arm>-single at f32), metric step_ms.p50, with
# BENCH_DOMINANT_KERNEL=0. The fixture loads behind the box gate, so the row needs --box (or
# --skip-box-check). Its source proof writes bench_gpt's generated sources per arm and precision:
# the f32 sources must be byte-identical across arms (the drift control) and the f16 ones must
# differ (the treatment reaches the step).
#
# TUNED TRAINING ROW (--tuned-training, gh-ocannl-2372, shared with gh1051_bf16_ab.sh). Replaces
# the training row's f16 cells with bench_gpt's seed-only search (BENCH_TUNE=1) followed by
# fresh-process replays. Each f16 policy searches once in an initially empty cache of its own, then
# only replays that cache. The two f32 cells keep the same untuned step under both explicit flags,
# so their paired ratio measures session drift without search noise. The source proof compares the
# shipped replay's generated sources at f16 and the untuned sources at f32. Validated by
# gh1051_cells.py at precision `half`.
#
# DRIFT CONTROL (the A/B protocol in docs/agent-notes/training-and-performance.md): next to every
# treatment cell runs a CONTROL cell, the same bench at f32 under the same arm's fp16_arithmetic
# flag. The f16 policy cannot reach an f32 kernel, and before any timing the driver PROVES it: one
# untimed control run per arm writes its generated source (.cu on CUDA, .hip on HIP, .metal on
# Metal; the untimed oracle's plain default compile included), and the two sources must be
# byte-identical or the driver stops. The control's wide/narrow ratio is then the session drift the
# treatment ratios are read against. The proof's twin at f16 shows the treatment REACHES codegen:
# one untimed f16 run per arm, whose sources must differ (and whether the oracle's plain default
# kernel differs, plus which arm `auto` matches, is reported).
#
# VALIDATION: benchmarks/gh1051_cells.py's mode "ab" at PREC half (shared with gh1051_bf16_ab.sh at
# bfloat16; its HALF section) reads each size's bin_cells run and checks every cell's verdict and
# whole-output checksums, plus every tensorized f16 cell's MMA census; any failure fails the run.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
. "$HERE/box-gate.sh"
box_gate_args "$@"
shift "$box_gate_shift"
USAGE="usage: $0 [--box BOX | --skip-box-check] [--repeats N] [--search-timeout S] [--training-fixture PATH [--tuned-training] | --untuned-fixture PATH] <repo-root> <backend cuda|hip|metal> <memory-label> [rounds] [sizes...]"
REPEATS=10
SEARCH_TIMEOUT=7200
FIXTURE=
TUNED=0
while [ $# -gt 0 ]; do
  case $1 in
    --repeats)
      [ $# -ge 2 ] || { echo "$USAGE"; exit 2; }
      REPEATS=$2
      if [[ "$REPEATS" = *[!0-9]* || "$REPEATS" != *[1-9]* ]]; then
        echo "repeats must be a positive integer, got '$REPEATS'"
        exit 2
      fi
      shift 2 ;;
    --search-timeout)
      [ $# -ge 2 ] || { echo "$USAGE"; exit 2; }
      SEARCH_TIMEOUT=$2
      if [[ "$SEARCH_TIMEOUT" = *[!0-9]* || "$SEARCH_TIMEOUT" != *[1-9]* ]]; then
        echo "search-timeout must be a positive integer, got '$SEARCH_TIMEOUT'"
        exit 2
      fi
      shift 2 ;;
    --training-fixture | --untuned-fixture)
      [ $# -ge 2 ] && [ -n "$2" ] || { echo "$USAGE"; exit 2; }
      FIXTURE=$2
      shift 2 ;;
    --tuned-training) TUNED=1; shift ;;
    *) break ;;
  esac
done
if [ -n "$FIXTURE" ]; then
  [ -f "$FIXTURE" ] || { echo "no such fixture: $FIXTURE"; exit 2; }
  FIXTURE=$(cd "$(dirname "$FIXTURE")" && pwd)/$(basename "$FIXTURE")
  case $FIXTURE in *.safetensors) ;; *) echo "the training fixture must be a .safetensors file, got $FIXTURE"; exit 2 ;; esac
  mode=$(python3 -c 'import struct, sys
sys.path.insert(0, sys.argv[1])
import strict_json
f = open(sys.argv[2], "rb"); n = struct.unpack("<Q", f.read(8))[0]
print(strict_json.loads(f.read(n).decode()).get("__metadata__", {}).get("mode", ""))' \
    "$HERE" "$FIXTURE" 2>/dev/null)
  [ "$mode" = train ] || { echo "the training fixture must be a training one (metadata mode=train), got mode='$mode': $FIXTURE"; exit 2; }
elif [ "$TUNED" = 1 ]; then
  echo "--tuned-training requires --training-fixture"
  exit 2
fi
[ $# -ge 3 ] || { echo "$USAGE"; exit 2; }
ROOT=$(cd "$1" && pwd) || { echo "no such repo root: $1"; exit 2; }
BACKEND=$2
case $BACKEND in
  cuda) EXT=cu ;;
  hip) EXT=hip ;;
  metal) EXT=metal ;;
  *) echo "backend must be cuda, hip, or metal, got '$BACKEND'"; exit 2 ;;
esac
LABEL=$3
ROUNDS=${4:-8}
shift $(($# < 4 ? $# : 4))
SIZES=("$@")
[ ${#SIZES[@]} -gt 0 ] || SIZES=(1024 2048)
for size in "${SIZES[@]}"; do
  case $size in '' | *[!0-9]* | 0) echo "sizes must be positive integers, got '$size'"; exit 2 ;; esac
done
case $ROUNDS in '' | *[!0-9]*) echo "rounds must be a positive even integer, got '$ROUNDS'"; exit 2 ;; esac
if ((ROUNDS == 0 || ROUNDS % 2 != 0)); then
  echo "rounds must be a positive even integer (every round is mirrored by its partner), got $ROUNDS"
  exit 2
fi
GATE=()
if [ -n "$FIXTURE" ]; then
  # Before the first measurement: the training row loads the fixture.
  box_gate gh2195 "$(dirname "$FIXTURE")" "$(basename "$FIXTURE" .safetensors)"
  if [ "$box_gate_skip" = 1 ]; then GATE=(--skip-box-check); else GATE=(--box "$box_gate_box"); fi
fi
cd "$ROOT" || exit 1
dune build bin/schedule_bench.exe 2>/dev/null || { echo "BUILD FAILED"; exit 1; }
BENCH="$ROOT/_build/default/bin/schedule_bench.exe"
if [ -n "$FIXTURE" ]; then
  dune build benchmarks/runners/ocannl/bench_gpt.exe 2>/dev/null || { echo "BUILD FAILED (bench_gpt)"; exit 1; }
fi
GPT="$ROOT/_build/default/benchmarks/runners/ocannl/bench_gpt.exe"
OUT=$(mktemp -d "${TMPDIR:-/tmp}/gh2195_ab.XXXXXX") || exit 1
# The search and replay cells have different working directories.
OUT=$(cd "$OUT" && pwd -P) || exit 1
echo "gh2195 f16 A/B: backend=$BACKEND label=$LABEL rounds=$ROUNDS repeats=$REPEATS sizes=${SIZES[*]} sha=$(git -C "$ROOT" rev-parse HEAD)"
echo "host=$(hostname) raw=$OUT training-fixture=${FIXTURE:-none}"
git -C "$ROOT" status --porcelain --untracked-files=no | sed 's/^/  dirty: /'
if [ "$BACKEND" = cuda ]; then
  nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv,noheader 2>/dev/null |
    sed 's/^/device: /'
elif [ "$BACKEND" = hip ]; then
  rocminfo 2>/dev/null | grep -E "Marketing Name|^ *Name: *gfx" | sed 's/^ */device: /' | sort -u
else
  # The Mac and its OS: a Metal reading names both (gh-ocannl-2150).
  system_profiler SPDisplaysDataType 2>/dev/null | grep -E "Chipset Model|Total Number of Cores" |
    sed 's/^ */device: /'
  echo "os: macOS $(sw_vers -productVersion 2>/dev/null) ($(sw_vers -buildVersion 2>/dev/null))"
fi
FAIL=0

# One untimed identity run of EXE [ARGS...] under fp16_arithmetic=ARM at storage PREC (a label),
# its generated sources copied to $OUT/identity_<TAG>_<PREC>_<ARM>. Each run is a cell of the
# shared runner, whose debug files land under its own directory. ENV words (NAME=VALUE) lead EXE.
identity_run() {
  local tag=$1 prec=$2 arm=$3 default_prec=$4
  shift 4
  local prefix="gh2195_${tag}_${prec}_$arm" dir="$OUT/identity_run_${tag}_${prec}_$arm"
  local expect=(--expect fp16_arithmetic="$arm")
  [ -n "$default_prec" ] && expect+=(--expect default_prec="$default_prec")
  python3 "$HERE/cells.py" cell --backend "$BACKEND" --dir "$dir" --timeout 1800 "${expect[@]}" -- \
    "$@" --ocannl_fp16_arithmetic="$arm" --ocannl_output_debug_files_in_build_directory=true \
    --ocannl_build_files_prefix="$prefix"
  local rc=$?
  # A group that outlived its cleanup holds the device every later cell would be timed against.
  [ "$rc" -eq 125 ] && { echo "identity run ($tag, $prec, $arm): a process outlived its cleanup; stopping"; exit 125; }
  # An f16 run may differ from its oracle by rounding (exit 0); a nonzero exit is a fault.
  [ "$rc" -eq 0 ] || { echo "identity run ($tag, $prec, $arm) exited $rc"; FAIL=1; }
  mkdir -p "$OUT/identity_${tag}_${prec}_$arm"
  cp "$dir/build_files/$prefix"/*."$EXT" "$OUT/identity_${tag}_${prec}_$arm/" 2>/dev/null
}

n_sources() { find "$1" -name "*.$EXT" 2>/dev/null | wc -l | tr -d ' '; }

compare_sources() {
  local na nb
  na=$(n_sources "$1") nb=$(n_sources "$2")
  [ "$na" -gt 0 ] && [ "$nb" -gt 0 ] || { echo error; return; }
  diff -r "$1" "$2" >/dev/null 2>&1
  case $? in 0) echo same ;; 1) echo differ ;; *) echo error ;; esac
}

prove() {
  local tag=$1 what=$2 d="$OUT/identity_$1"
  local n_src n_half n_diff v
  n_src=$(n_sources "${d}_single_false")
  v=$(compare_sources "${d}_single_false" "${d}_single_true")
  if [ "$v" != same ]; then
    echo "DRIFT CONTROL INVALID ($what): the f32 control's generated .$EXT across arms: $v ($n_src and $(n_sources "${d}_single_true") files)"
    exit 1
  fi
  echo "drift control ($what): $n_src generated f32 kernels byte-identical across arms"
  n_half=$(n_sources "${d}_half_false")
  v=$(compare_sources "${d}_half_false" "${d}_half_true")
  if [ "$v" != differ ]; then
    echo "TREATMENT UNREACHED ($what): the f16 generated .$EXT across arms: $v ($n_half and $(n_sources "${d}_half_true") files)"
    exit 1
  fi
  n_diff=$(diff -rq "${d}_half_false" "${d}_half_true" | wc -l | tr -d ' ')
  if [ "$TUNED" = 1 ] && [ "$tag" = train ]; then
    echo "treatment reach ($what): $n_diff differing f16 source paths across $n_half wide and $(n_sources "${d}_half_true") narrow sources"
  else
    echo "treatment reach ($what): $n_diff of $n_half generated f16 sources differ across arms"
  fi
}

# The GEMM cells' proof, plus oracle comparison and which arm `auto` is at this revision.
for prec in single half; do
  for arm in false true; do
    identity_run gemm "$prec" "$arm" "$prec" \
      "$BENCH" "${SIZES[0]}" "$REPEATS" "${SIZES[0]}" "${SIZES[0]}" 0 --ocannl_default_prec="$prec"
  done
done
prove gemm schedule_bench
ref_false=$(find "$OUT/identity_gemm_half_false" -name "*mm_reference*.$EXT" | sort)
if [ -z "$ref_false" ]; then
  echo "oracle (plain default compile, f16): no mm_reference source found to compare"
else
  for f in $ref_false; do
    if cmp -s "$f" "$OUT/identity_gemm_half_true/$(basename "$f")"; then
      echo "oracle (plain default compile, f16): $(basename "$f") identical across arms"
    else
      echo "oracle (plain default compile, f16): $(basename "$f") differs across arms"
    fi
  done
fi
identity_run gemm half auto half \
  "$BENCH" "${SIZES[0]}" "$REPEATS" "${SIZES[0]}" "${SIZES[0]}" 0 --ocannl_default_prec=half
AUTO="$OUT/identity_gemm_half_auto"
if [ "$(compare_sources "$AUTO" "$OUT/identity_gemm_half_false")" = same ]; then
  echo "auto (schedule_bench, f16): byte-identical to the wide arm (false) -- auto resolves wide here"
elif [ "$(compare_sources "$AUTO" "$OUT/identity_gemm_half_true")" = same ]; then
  echo "auto (schedule_bench, f16): byte-identical to the narrow arm (true) -- auto resolves narrow here"
else
  echo "auto (schedule_bench, f16): matches neither arm's sources ($(n_sources "$AUTO") files written)"
fi

# Run the GEMM timing cells before any multi-hour cold training search so GEMM readings are
# reported immediately.
for size in "${SIZES[@]}"; do
  arms=()
  for arm in false true; do
    for prec in half single; do
      arms+=(--arm "$arm-$prec=--ocannl_fp16_arithmetic=$arm --ocannl_default_prec=$prec")
    done
  done
  python3 "$HERE/bin_cells.py" --exe "$BENCH" --backend "$BACKEND" --out "$OUT/n$size" \
    --rounds "$ROUNDS" "${arms[@]}" -- "$size" "$REPEATS" "$size" "$size" 0 >/dev/null
  rc=$?
  [ "$rc" -eq 125 ] && { echo "n=$size: a process outlived its cleanup; stopping"; exit 125; }
  [ "$rc" -eq 0 ] || { echo "n=$size: bin_cells exited $rc (see $OUT/n$size/report.md)"; FAIL=1; }
done
python3 "$HERE/gh1051_cells.py" ab "$OUT" "$LABEL" half || FAIL=1
[ -n "$FIXTURE" ] || exit $FAIL
if [ "$TUNED" = 1 ] && [ "$FAIL" -ne 0 ]; then
  echo "stopping before cold tuned training searches: GEMM validation already failed"
  exit "$FAIL"
fi

# The training row's cold search (when tuned) and source proof, before training timing.
TRAIN_ENV=("BENCH_FIXTURE=$FIXTURE" BENCH_MATERIALIZE=0 BENCH_DEBUG=0 BENCH_DOMINANT_KERNEL=0)
TRAIN_ARGS=(--ocannl_model_default_schedule=false
  --ocannl_gpu_default_seed_narrow_storage=false --ocannl_autotune_log=true)
TRAIN_MODE=untuned
if [ "$TUNED" = 1 ]; then
  TRAIN_MODE=tuned
  # Both placement arms of each policy search cold. The replay pins search off so a missing
  # cache entry cannot silently run a new search in a timed process.
  for arm in false true; do
    python3 "$HERE/cells.py" cell --backend "$BACKEND" --dir "$OUT/search_$arm" --timeout "$SEARCH_TIMEOUT" \
      --expect fp16_arithmetic="$arm" --expect autotune_search=true -- \
      "${TRAIN_ENV[@]}" BENCH_TUNE=1 BENCH_PRECISION=f16 "$GPT" "${TRAIN_ARGS[@]}" \
      --ocannl_fp16_arithmetic="$arm" --ocannl_autotune_search=true \
      --ocannl_autotune_cache_dir="$OUT/cache-$arm" || exit $?
    python3 "$HERE/gh1051_cells.py" tuned-search "$OUT/search_$arm" half || exit $?
  done
fi
for prec in single half; do
  bench_prec=f32
  [ "$prec" = half ] && bench_prec=f16
  for arm in false true; do
    tune=0
    replay=()
    if [ "$TUNED" = 1 ] && [ "$prec" = half ]; then
      tune=1
      replay=(--ocannl_autotune_cache_dir="$OUT/cache-$arm")
    fi
    identity_run train "$prec" "$arm" "" "${TRAIN_ENV[@]}" BENCH_TUNE="$tune" \
      "BENCH_PRECISION=$bench_prec" "$GPT" "${TRAIN_ARGS[@]}" --ocannl_autotune_search=false \
      ${replay[@]+"${replay[@]}"}
  done
done
prove train bench_gpt
if [ "$TUNED" = 1 ]; then
  # Also prove fp16_arithmetic reaches bench_gpt codegen holding the schedule fixed (untuned),
  # so a schedule difference between two independent cold searches cannot masquerade as policy reach.
  for arm in false true; do
    identity_run train_fixed half "$arm" "" "${TRAIN_ENV[@]}" BENCH_TUNE=0 \
      BENCH_PRECISION=f16 "$GPT" "${TRAIN_ARGS[@]}" --ocannl_autotune_search=false
  done
  n_fix_w=$(n_sources "$OUT/identity_train_fixed_half_false")
  n_fix_n=$(n_sources "$OUT/identity_train_fixed_half_true")
  v_fix=$(compare_sources "$OUT/identity_train_fixed_half_false" "$OUT/identity_train_fixed_half_true")
  if [ "$v_fix" != differ ]; then
    echo "TREATMENT UNREACHED (bench_gpt fixed-schedule): the f16 generated .$EXT across arms: $v_fix ($n_fix_w and $n_fix_n files)"
    exit 1
  fi
  n_fix_diff=$(diff -rq "$OUT/identity_train_fixed_half_false" "$OUT/identity_train_fixed_half_true" | wc -l | tr -d ' ')
  echo "treatment reach (bench_gpt fixed-schedule): $n_fix_diff of $n_fix_w generated f16 sources differ across arms"
  [ "$FAIL" -eq 0 ] || exit "$FAIL"
  python3 "$HERE/gh1051_cells.py" tuned-proof "$OUT" "$LABEL" half || exit $?
fi

# The training row: one bin_cells run (not under n*/, which gh1051_cells.py reads as sizes).
TRAIN="$OUT/$TRAIN_MODE"
envs=()
for e in "${TRAIN_ENV[@]}"; do envs+=(--env "$e"); done
tune=0
cache_false= cache_true=
runner=("$HERE/bin_cells.py")
if [ "$TUNED" = 1 ]; then
  tune=1
  runner=("$HERE/gh1051_cells.py" tuned-cells "$OUT" --prec half)
  # bin_cells shell-splits each arm's arguments, including paths from the operator's TMPDIR.
  quoted_out=$(python3 -c 'import shlex, sys; print(shlex.quote(sys.argv[1]))' "$OUT") || exit 1
  cache_false=" --ocannl_autotune_cache_dir=${quoted_out}/cache-false"
  cache_true=" --ocannl_autotune_cache_dir=${quoted_out}/cache-true"
fi
python3 "${runner[@]}" --exe "$GPT" --backend "$BACKEND" --out "$TRAIN" ${GATE[@]+"${GATE[@]}"} \
  --reader record --rounds "$ROUNDS" --cell-timeout 1800 "${envs[@]}" \
  --arm "false-half=BENCH_TUNE=$tune BENCH_PRECISION=f16 --ocannl_fp16_arithmetic=false$cache_false" \
  --arm "true-half=BENCH_TUNE=$tune BENCH_PRECISION=f16 --ocannl_fp16_arithmetic=true$cache_true" \
  --arm "false-single=BENCH_TUNE=0 BENCH_PRECISION=f32 --ocannl_fp16_arithmetic=false" \
  --arm "true-single=BENCH_TUNE=0 BENCH_PRECISION=f32 --ocannl_fp16_arithmetic=true" \
  -- "${TRAIN_ARGS[@]}" --ocannl_autotune_search=false >/dev/null
rc=$?
[ "$rc" -eq 125 ] && { echo "training: a process outlived its cleanup; stopping"; exit 125; }
[ "$rc" -eq 0 ] || { echo "training: bin_cells exited $rc (see $TRAIN/report.md)"; FAIL=1; }
python3 "$HERE/gh1051_cells.py" "$TRAIN_MODE" "$TRAIN" "$LABEL" "$(basename "$FIXTURE" .safetensors)" half ||
  FAIL=1
exit $FAIL
