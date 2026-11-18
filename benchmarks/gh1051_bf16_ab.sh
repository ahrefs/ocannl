#!/bin/bash
# gh-ocannl-1051 (HIP) and gh-ocannl-2150 (Metal): throughput A/B of a GPU backend's two bf16
# accumulator arms: bf16_arithmetic=false (Bf16_wide, f32 accumulators) against
# bf16_arithmetic=true (Bf16_narrow: the bf16-accumulate tensor-unit arm -- gfx11's bf16 WMMA on
# HIP, the simdgroup_bfloat8x8 accumulator on Metal -- and storage-width serial accumulators). It
# is the measurement behind resolving auto wide (HIP since gh-ocannl-1051; Metal since
# gh-ocannl-2029, decided on accuracy and timed here). The arms are named `false` and `true`, not
# `auto`: auto resolved narrow on both backends once and resolves wide on both now, while the
# explicit values name the same arm on both sides of those changes, so the arms mean the same at
# every revision this driver can measure (below). Which arm `auto` is at the measured revision is
# REPORTED from the generated sources (the identity proof below), so a reading of "wide" as "the
# default" is checked rather than assumed.
#
# Usage: gh1051_bf16_ab.sh [--box BOX | --skip-box-check] [--repeats N]
#                          [--training-fixture PATH [--tuned-training]]
#                          <repo-root> <backend hip|metal> <memory-label> [rounds] [sizes...]
#   e.g. gh1051_bf16_ab.sh ~/ocannl hip "unified (gfx1151)" 8 1024 2048
#        gh1051_bf16_ab.sh --box mac-studio --repeats 50 --tuned-training \
#          --training-fixture ~/ocannl-staging/benchmarks/fixtures/gpt2_mini_train.safetensors \
#          ~/ocannl metal "unified (M4 Max)" 12 1024 2048
#
# [rounds] must be even and positive (default 8). The cells are bin_cells.py's (gh-ocannl-1607): per
# size one run of four arms -- the two bf16_arithmetic arms (`false`, `true`) at bf16 and at f32
# (the control below), named <arm>-<prec> -- visited round by round in Bench_design's spread
# mirrored rotation, so every arm's position histogram is equal at a multiple of four rounds, and
# every pair of arms exchanges positions within each pair of rounds. Each cell is hermetic (no
# OCANNL_*, BENCH_* or OpenMP variable inherited, no config file read), supervised, and
# configuration-checked: its resolved-configuration records must be its command line, so the
# library must carry the stream (gh-ocannl-1566): measure an older revision with the driver of its
# own time.
#
# --repeats is the positive number of schedule_bench repetitions per GEMM cell (default 10).
# Each treatment cell is bin/schedule_bench at m = n = k = <size> with bf16 operands and output. All
# five variants are the treatment, not only the tensorized mma_pd1/mma_pd2: the policy also moves
# the serial legs' (parallel, smem, regtile) accumulator (the backend's accum_prec). No schedule
# cache is involved: every variant is a hand-written schedule compiled through [~lowered_transform].
#
# TRAINING ROW (only with --training-fixture; gh-ocannl-2150). The end-to-end cost: bench_gpt's
# untuned training step (BENCH_TUNE=0, so Bench_harness.compile_default = Autotune.untuned_compile;
# no search, model_default off) on the given recorded fixture -- never a regenerated one -- under
# BENCH_PRECISION=bf16 (bf16 activations and gradients, f32 masters and loss head), one bin_cells
# --reader record run of the same four arms (<arm>-bfloat16 at BENCH_PRECISION=bf16,
# <arm>-single at f32), metric step_ms.p50 (each step synced, which is what training pays on
# Metal), with the dominant-kernel instrument off (BENCH_DOMINANT_KERNEL=0: it times every kernel
# once more and would be most of each cell's wall time). The fixture loads behind the box gate, so
# the row needs --box (or --skip-box-check). The row has a source proof of its own, as the GEMM
# cells do (below): one untimed run per arm and precision writes bench_gpt's generated sources, the
# f32 ones must be byte-identical across the arms (the row's drift control) and the bf16 ones must
# differ (the treatment reaches the step). Its validation is gh1051_cells.py's mode "untuned",
# shared with gh2195_f16_ab.sh.
#
# TUNED TRAINING ROW (--tuned-training, gh-ocannl-2321). Replaces the optional untuned row's bf16
# cells with bench_gpt's seed-only search (BENCH_TUNE=1) followed by fresh-process replays. Each
# bf16 policy searches once in an initially empty cache of its own, then only replays that cache.
# The two f32 cells keep the same untuned step under both explicit flags, so their paired ratio
# measures session drift without search noise. The source proof compares the shipped replay's
# generated sources at bf16 and the untuned sources at f32. The bf16 comparison can include
# schedule differences because each policy chooses its own crown. Only the one JSON result
# record from each search, proof and timing cell is read for tuned provenance. Candidate logs
# are evidence for investigation, outside the check. A failed cold search, mismatched source
# proof, or missing verified replay invalidates the run. Each cold search can take up to 30
# minutes. Restart with a fresh invocation after a failure so every crown comes from this run.
#
# DRIFT CONTROL (the A/B protocol in docs/agent-notes/training-and-performance.md): next to every
# treatment cell runs a CONTROL cell, the same bench at f32 under the same arm's bf16_arithmetic
# flag. The bf16 policy cannot reach an f32 kernel, and before any timing the driver PROVES it: one
# untimed control run per arm writes its generated source (.hip on HIP, .metal on Metal), and the
# two sources must be byte-identical or the driver stops. The control's paired wide/narrow ratio is
# then the session drift the treatment ratios are read against. The proof's twin at bf16 shows the
# treatment REACHES codegen: one untimed bf16 run per arm, whose sources must differ; a third, under
# `auto`, is compared with both and the arm it matches is reported.
#
# VALIDATION: a timing counts only from a cell whose output is validated, which the bench's exit
# status cannot do for bf16 (its partial sums round, and the narrow tensorized arm's error is
# gross by design on gfx11). benchmarks/gh1051_cells.py, shared with gh1051_boundary_split.sh and
# gh2195_f16_ab.sh, reads each size's bin_cells run and checks every cell's verdict there and its
# whole-output checksums (determinism across rounds, a structural bound against the exact f32
# control, identical checksums from the narrow arm's two tensorized pipelinings, every tensorized
# cell's census reporting intrinsics) and prints the table, its wide/narrow ratios paired by round;
# any failure fails the run. Its METAL section says what changes on Apple GPUs.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
. "$HERE/box-gate.sh"
box_gate_args "$@"
shift "$box_gate_shift"
USAGE="usage: $0 [--box BOX | --skip-box-check] [--repeats N] [--training-fixture PATH [--tuned-training]] <repo-root> <backend hip|metal> <memory-label> [rounds] [sizes...]"
REPEATS=10
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
    --training-fixture)
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
  # bench_gpt picks its step from the fixture's metadata: an inference fixture (mode=infer) would
  # time a forward-only step in the training row.
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
  hip) EXT=hip ;;
  metal) EXT=metal ;;
  *) echo "backend must be hip or metal, got '$BACKEND'"; exit 2 ;;
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
  box_gate gh1051 "$(dirname "$FIXTURE")" "$(basename "$FIXTURE" .safetensors)"
  if [ "$box_gate_skip" = 1 ]; then GATE=(--skip-box-check); else GATE=(--box "$box_gate_box"); fi
fi
cd "$ROOT" || exit 1
dune build bin/schedule_bench.exe 2>/dev/null || { echo "BUILD FAILED"; exit 1; }
BENCH="$ROOT/_build/default/bin/schedule_bench.exe"
if [ -n "$FIXTURE" ]; then
  dune build benchmarks/runners/ocannl/bench_gpt.exe 2>/dev/null || { echo "BUILD FAILED (bench_gpt)"; exit 1; }
fi
GPT="$ROOT/_build/default/benchmarks/runners/ocannl/bench_gpt.exe"
OUT=$(mktemp -d "${TMPDIR:-/tmp}/gh1051_ab.XXXXXX")
echo "gh1051 bf16 A/B: backend=$BACKEND label=$LABEL rounds=$ROUNDS repeats=$REPEATS sizes=${SIZES[*]} sha=$(git -C "$ROOT" rev-parse HEAD)"
echo "host=$(hostname) raw=$OUT training-fixture=${FIXTURE:-none}"
git -C "$ROOT" status --porcelain --untracked-files=no | sed 's/^/  dirty: /'
if [ "$BACKEND" = hip ]; then
  rocminfo 2>/dev/null | grep -E "Marketing Name|^ *Name: *gfx" | sed 's/^ */device: /' | sort -u
else
  # The Mac and its OS: a Metal reading names both (gh-ocannl-2150).
  system_profiler SPDisplaysDataType 2>/dev/null | grep -E "Chipset Model|Total Number of Cores" |
    sed 's/^ */device: /'
  echo "os: macOS $(sw_vers -productVersion 2>/dev/null) ($(sw_vers -buildVersion 2>/dev/null))"
fi
FAIL=0

# One untimed identity run of EXE [ARGS...] under bf16_arithmetic=ARM at storage PREC (a label),
# its generated sources copied to $OUT/identity_<TAG>_<PREC>_<ARM>. Each run is a cell of the shared
# runner, whose debug files land under its own directory. ENV words (NAME=VALUE) lead EXE.
identity_run() {
  local tag=$1 prec=$2 arm=$3 default_prec=$4
  shift 4
  local prefix="gh1051_${tag}_${prec}_$arm" dir="$OUT/identity_run_${tag}_${prec}_$arm"
  local expect=(--expect bf16_arithmetic="$arm")
  [ -n "$default_prec" ] && expect+=(--expect default_prec="$default_prec")
  python3 "$HERE/cells.py" cell --backend "$BACKEND" --dir "$dir" --timeout 1800 "${expect[@]}" -- \
    "$@" --ocannl_bf16_arithmetic="$arm" --ocannl_output_debug_files_in_build_directory=true \
    --ocannl_build_files_prefix="$prefix"
  local rc=$?
  # A group that outlived its cleanup holds the device every later cell would be timed against.
  [ "$rc" -eq 125 ] && { echo "identity run ($tag, $prec, $arm): a process outlived its cleanup; stopping"; exit 125; }
  # A bf16 run may differ from its oracle by rounding (exit 0); a nonzero exit is a fault.
  [ "$rc" -eq 0 ] || { echo "identity run ($tag, $prec, $arm) exited $rc"; FAIL=1; }
  mkdir -p "$OUT/identity_${tag}_${prec}_$arm"
  cp "$dir/build_files/$prefix"/*."$EXT" "$OUT/identity_${tag}_${prec}_$arm/" 2>/dev/null
}

# The source collection of one identity run, as a file count: 0 when the run wrote none or the copy
# failed, which no comparison below may read as evidence either way.
n_sources() { find "$1" -name "*.$EXT" 2>/dev/null | wc -l | tr -d ' '; }

# diff -r's verdict on two source collections: same (0), differ (1), or error (2, or both sides
# empty) -- an I/O error is never read as "differ".
compare_sources() {
  local na nb
  na=$(n_sources "$1") nb=$(n_sources "$2")
  [ "$na" -gt 0 ] && [ "$nb" -gt 0 ] || { echo error; return; }
  diff -r "$1" "$2" >/dev/null 2>&1
  case $? in 0) echo same ;; 1) echo differ ;; *) echo error ;; esac
}

# The proof for one bench TAG: its f32 sources identical across the arms (the drift control), its
# bf16 sources different (the treatment reaches codegen). Each needs both arms' sources written and
# compared without error. Stops the driver when either fails.
prove() {
  local tag=$1 what=$2 d="$OUT/identity_$1"
  local n_src n_bf n_diff v
  n_src=$(n_sources "${d}_single_false")
  v=$(compare_sources "${d}_single_false" "${d}_single_true")
  if [ "$v" != same ]; then
    echo "DRIFT CONTROL INVALID ($what): the f32 control's generated .$EXT across arms: $v ($n_src and $(n_sources "${d}_single_true") files)"
    exit 1
  fi
  echo "drift control ($what): $n_src generated f32 kernels byte-identical across arms"
  n_bf=$(n_sources "${d}_bfloat16_false")
  v=$(compare_sources "${d}_bfloat16_false" "${d}_bfloat16_true")
  if [ "$v" != differ ]; then
    echo "TREATMENT UNREACHED ($what): the bf16 generated .$EXT across arms: $v ($n_bf and $(n_sources "${d}_bfloat16_true") files)"
    exit 1
  fi
  n_diff=$(diff -rq "${d}_bfloat16_false" "${d}_bfloat16_true" | wc -l | tr -d ' ')
  if [ "$TUNED" = 1 ] && [ "$tag" = train ]; then
    echo "treatment reach ($what): $n_diff differing bf16 source paths across $n_bf wide and $(n_sources "${d}_bfloat16_true") narrow sources"
  else
    echo "treatment reach ($what): $n_diff of $n_bf generated bf16 sources differ across arms"
  fi
}

# The GEMM cells' proof, plus which arm `auto` is at this revision (reported, not required).
for prec in single bfloat16; do
  for arm in false true; do
    identity_run gemm "$prec" "$arm" "$prec" \
      "$BENCH" "${SIZES[0]}" "$REPEATS" "${SIZES[0]}" "${SIZES[0]}" 0 --ocannl_default_prec="$prec"
  done
done
prove gemm schedule_bench
identity_run gemm bfloat16 auto bfloat16 \
  "$BENCH" "${SIZES[0]}" "$REPEATS" "${SIZES[0]}" "${SIZES[0]}" 0 --ocannl_default_prec=bfloat16
AUTO="$OUT/identity_gemm_bfloat16_auto"
if [ "$(compare_sources "$AUTO" "$OUT/identity_gemm_bfloat16_false")" = same ]; then
  echo "auto (schedule_bench, bf16): byte-identical to the wide arm (false) -- auto resolves wide here"
elif [ "$(compare_sources "$AUTO" "$OUT/identity_gemm_bfloat16_true")" = same ]; then
  echo "auto (schedule_bench, bf16): byte-identical to the narrow arm (true) -- auto resolves narrow here"
else
  echo "auto (schedule_bench, bf16): matches neither arm's sources ($(n_sources "$AUTO") files written)"
fi

# The training row's proof, before any timing.
TRAIN_ENV=()
if [ -n "$FIXTURE" ]; then
  TRAIN_ENV=("BENCH_FIXTURE=$FIXTURE" BENCH_MATERIALIZE=0 BENCH_DEBUG=0 BENCH_DOMINANT_KERNEL=0)
  TRAIN_ARGS=(--ocannl_model_default_schedule=false
    --ocannl_gpu_default_seed_narrow_storage=false --ocannl_autotune_log=true)
  TRAIN_MODE=untuned
  if [ "$TUNED" = 1 ]; then
    TRAIN_MODE=tuned
    # Both placement arms of each policy search cold. The replay pins search off so a missing
    # cache entry cannot silently run a new search in a timed process.
    for arm in false true; do
      python3 "$HERE/cells.py" cell --backend "$BACKEND" --dir "$OUT/search_$arm" --timeout 1800 \
        --expect bf16_arithmetic="$arm" --expect autotune_search=true -- \
        "${TRAIN_ENV[@]}" BENCH_TUNE=1 BENCH_PRECISION=bf16 "$GPT" "${TRAIN_ARGS[@]}" \
        --ocannl_bf16_arithmetic="$arm" --ocannl_autotune_search=true \
        --ocannl_autotune_cache_dir="$OUT/cache-$arm" || exit $?
    done
  fi
  for prec in single bfloat16; do
    bench_prec=f32
    [ "$prec" = bfloat16 ] && bench_prec=bf16
    for arm in false true; do
      tune=0
      replay=()
      if [ "$TUNED" = 1 ] && [ "$prec" = bfloat16 ]; then
        tune=1
        replay=(--ocannl_autotune_cache_dir="$OUT/cache-$arm")
      fi
      identity_run train "$prec" "$arm" "" "${TRAIN_ENV[@]}" BENCH_TUNE="$tune" \
        "BENCH_PRECISION=$bench_prec" "$GPT" "${TRAIN_ARGS[@]}" --ocannl_autotune_search=false \
        ${replay[@]+"${replay[@]}"}
    done
  done
  prove train bench_gpt
  [ "$FAIL" -eq 0 ] || exit "$FAIL"
  if [ "$TUNED" = 1 ]; then
    python3 "$HERE/gh1051_cells.py" tuned-proof "$OUT" "$LABEL" || exit $?
  fi
fi

for size in "${SIZES[@]}"; do
  arms=()
  for arm in false true; do
    for prec in bfloat16 single; do
      arms+=(--arm "$arm-$prec=--ocannl_bf16_arithmetic=$arm --ocannl_default_prec=$prec")
    done
  done
  python3 "$HERE/bin_cells.py" --exe "$BENCH" --backend "$BACKEND" --out "$OUT/n$size" \
    --rounds "$ROUNDS" "${arms[@]}" -- "$size" "$REPEATS" "$size" "$size" 0 >/dev/null
  rc=$?
  [ "$rc" -eq 125 ] && { echo "n=$size: a process outlived its cleanup; stopping"; exit 125; }
  [ "$rc" -eq 0 ] || { echo "n=$size: bin_cells exited $rc (see $OUT/n$size/report.md)"; FAIL=1; }
done
python3 "$HERE/gh1051_cells.py" ab "$OUT" "$LABEL" || FAIL=1
[ -n "$FIXTURE" ] || exit $FAIL

# The training row: one bin_cells run (not under n*/, which gh1051_cells.py reads as sizes).
TRAIN="$OUT/$TRAIN_MODE"
envs=()
for e in "${TRAIN_ENV[@]}"; do envs+=(--env "$e"); done
tune=0
cache_false= cache_true=
if [ "$TUNED" = 1 ]; then
  tune=1
  # bin_cells shell-splits each arm's arguments, including paths from the operator's TMPDIR.
  quoted_out=$(python3 -c 'import shlex, sys; print(shlex.quote(sys.argv[1]))' "$OUT") || exit 1
  cache_false=" --ocannl_autotune_cache_dir=${quoted_out}/cache-false"
  cache_true=" --ocannl_autotune_cache_dir=${quoted_out}/cache-true"
fi
python3 "$HERE/bin_cells.py" --exe "$GPT" --backend "$BACKEND" --out "$TRAIN" ${GATE[@]+"${GATE[@]}"} \
  --reader record --rounds "$ROUNDS" --cell-timeout 1800 "${envs[@]}" \
  --arm "false-bfloat16=BENCH_TUNE=$tune BENCH_PRECISION=bf16 --ocannl_bf16_arithmetic=false$cache_false" \
  --arm "true-bfloat16=BENCH_TUNE=$tune BENCH_PRECISION=bf16 --ocannl_bf16_arithmetic=true$cache_true" \
  --arm "false-single=BENCH_TUNE=0 BENCH_PRECISION=f32 --ocannl_bf16_arithmetic=false" \
  --arm "true-single=BENCH_TUNE=0 BENCH_PRECISION=f32 --ocannl_bf16_arithmetic=true" \
  -- "${TRAIN_ARGS[@]}" --ocannl_autotune_search=false >/dev/null
rc=$?
[ "$rc" -eq 125 ] && { echo "training: a process outlived its cleanup; stopping"; exit 125; }
[ "$rc" -eq 0 ] || { echo "training: bin_cells exited $rc (see $TRAIN/report.md)"; FAIL=1; }
python3 "$HERE/gh1051_cells.py" "$TRAIN_MODE" "$TRAIN" "$LABEL" "$(basename "$FIXTURE" .safetensors)" bfloat16 ||
  FAIL=1
exit $FAIL
