#!/bin/bash
# gh-ocannl-1051: GEMM throughput A/B of HIP's two bf16 accumulator arms: bf16_arithmetic=false
# (Bf16_wide, f32 accumulators) against bf16_arithmetic=true (Bf16_narrow: gfx11's bf16-accumulate
# WMMA and storage-width serial accumulators) -- the measurement behind resolving auto wide on HIP.
# The narrow arm is named `true`, not `auto`: before gh-ocannl-1051 auto resolved narrow on HIP and
# since then it resolves wide, while `true` names the narrow arm on both sides of that change, so
# this reproduces the measurement at any revision.
#
# Usage: gh1051_bf16_ab.sh <repo-root> <memory-label> [rounds] [sizes...]
#   e.g. gh1051_bf16_ab.sh ~/ocannl "unified (gfx1151)" 6 1024 2048
#
# [rounds] must be even and positive (default 6): round r runs the arms in the order
# (false, true) for even r and (true, false) for odd r, so only complete ABBA blocks give each arm
# every position equally often.
#
# Each treatment cell is bin/schedule_bench at m = n = k = <size> with bf16 operands and output. All
# five variants are the treatment, not only the tensorized mma_pd1/mma_pd2: the policy also moves
# the serial legs' accumulator (HIP's accum_prec). No schedule cache is involved: every variant is
# a hand-written schedule compiled through [~lowered_transform].
#
# DRIFT CONTROL (the A/B protocol in docs/agent-notes/training-and-performance.md): next to every
# treatment cell runs a CONTROL cell, the same bench at f32 under the same arm's bf16_arithmetic
# flag. The bf16 policy cannot reach an f32 kernel, and before any timing the driver PROVES it: one
# untimed control run per arm writes its generated HIP source, and the two sources must be
# byte-identical or the driver stops. The control's wide/narrow ratio is then the session drift the
# treatment ratios are read against.
#
# VALIDATION: a timing counts only from a cell whose output is validated, which the bench's exit
# status cannot do for bf16 (its partial sums round, and the narrow tensorized arm's error is gross
# by design). benchmarks/gh1051_cells.py, shared with gh1051_boundary_split.sh, checks every
# cell's status and whole-output checksums (determinism across rounds, a structural bound against
# the exact f32 control, identical checksums from the narrow arm's two tensorized pipelinings,
# every tensorized cell's census reporting intrinsics) and
# prints the table; any failure fails the run.
#
# Hermetic like gh514_cells.sh: every OCANNL_* variable is unset, every treatment is on argv, and the
# bench runs from benchmarks/, whose ocannl_config is the nearest on the upward search.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
[ $# -ge 2 ] || { echo "usage: $0 <repo-root> <memory-label> [rounds] [sizes...]"; exit 2; }
ROOT=$(cd "$1" && pwd) || { echo "no such repo root: $1"; exit 2; }
LABEL=$2
ROUNDS=${3:-6}
shift $(($# < 3 ? $# : 3))
SIZES=("$@")
[ ${#SIZES[@]} -gt 0 ] || SIZES=(1024 2048)
for size in "${SIZES[@]}"; do
  case $size in '' | *[!0-9]* | 0) echo "sizes must be positive integers, got '$size'"; exit 2 ;; esac
done
case $ROUNDS in '' | *[!0-9]*) echo "rounds must be a positive even integer, got '$ROUNDS'"; exit 2 ;; esac
if ((ROUNDS == 0 || ROUNDS % 2 != 0)); then
  echo "rounds must be a positive even integer (complete ABBA blocks), got $ROUNDS"
  exit 2
fi
while read -r v; do unset "$v"; done < <(env | sed -n 's/^\(OCANNL_[A-Z0-9_]*\)=.*/\1/p')
cd "$ROOT" || exit 1
dune build bin/schedule_bench.exe 2>/dev/null || { echo "BUILD FAILED"; exit 1; }
BENCH="$ROOT/_build/default/bin/schedule_bench.exe"
cd "$ROOT/benchmarks" || exit 1
OUT=$(mktemp -d "${TMPDIR:-/tmp}/gh1051_ab.XXXXXX")
echo "gh1051 bf16 A/B: label=$LABEL rounds=$ROUNDS sizes=${SIZES[*]} sha=$(git -C "$ROOT" rev-parse HEAD)"
echo "host=$(hostname) raw=$OUT"
rocminfo 2>/dev/null | grep -E "Marketing Name|^ *Name: *gfx" | sed 's/^ */device: /' | sort -u
FAIL=0
bench() { # size arm prec log [extra args...]
  local size=$1 arm=$2 prec=$3 log=$4
  shift 4
  "$BENCH" "$size" 10 "$size" "$size" 0 --ocannl_backend=hip --ocannl_default_prec="$prec" \
    --ocannl_bf16_arithmetic="$arm" "$@" >"$log" 2>"$log.err"
}
# The control's identity proof, untimed: generated HIP source per arm, byte-compared.
for arm in false true; do
  prefix="gh1051_ctl_$$_$arm"
  bench "${SIZES[0]}" "$arm" single "$OUT/identity_$arm.log" \
    --ocannl_output_debug_files_in_build_directory=true --ocannl_build_files_prefix="$prefix" ||
    { echo "control identity run ($arm) exited $?"; FAIL=1; }
  mkdir -p "$OUT/identity_$arm"
  cp build_files/"$prefix"/*.hip "$OUT/identity_$arm/" 2>/dev/null
  rm -rf build_files/"$prefix" log_files/"$prefix"
done
n_src=$(find "$OUT/identity_false" -name '*.hip' | wc -l)
if [ "$n_src" -eq 0 ] || ! diff -r "$OUT/identity_false" "$OUT/identity_true" >/dev/null; then
  echo "DRIFT CONTROL INVALID: the f32 control's generated HIP differs across arms (or none was written: $n_src files)"
  exit 1
fi
echo "drift control: $n_src generated f32 kernels byte-identical across arms"
for size in "${SIZES[@]}"; do
  for ((r = 0; r < ROUNDS; r++)); do
    if ((r % 2 == 0)); then order=(false true); else order=(true false); fi
    for arm in "${order[@]}"; do
      for prec in bfloat16 single; do
        log="$OUT/n${size}_r${r}_${arm}_${prec}.log"
        bench "$size" "$arm" "$prec" "$log"
        st=$?
        echo "$st" >"$log.status"
        echo "n=$size round=$r arm=$arm prec=$prec exit=$st"
      done
    done
  done
done
python3 "$HERE/gh1051_cells.py" ab "$OUT" "$LABEL" || FAIL=1
exit $FAIL
