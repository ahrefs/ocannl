#!/bin/bash
# gh-ocannl-1051: GEMM throughput A/B of bf16_arithmetic=false (Bf16_wide, f32 accumulators) against
# bf16_arithmetic=auto on one HIP box, the measurement that decides whether auto resolves wide on
# HIP.
#
# Usage: gh1051_bf16_ab.sh <repo-root> <memory-label> [rounds] [sizes...]
#   e.g. gh1051_bf16_ab.sh ~/ocannl "unified (gfx1151)" 6 1024 2048
#
# [rounds] must be even and positive (default 6): round r runs the arms in the order
# (false, auto) for even r and (auto, false) for odd r, so only complete ABBA blocks give each arm
# every position equally often.
#
# Each treatment cell is bin/schedule_bench at m = n = k = <size> with bf16 operands and output. All
# five variants are the treatment, not only the tensorized mma_pd1/mma_pd2: the policy also moves
# the serial legs' accumulator (HIP's accum_prec), and the flip moves both. No schedule cache is
# involved: every variant is a hand-written schedule compiled through [~lowered_transform].
#
# DRIFT CONTROL (the A/B protocol in docs/agent-notes/training-and-performance.md): next to every
# treatment cell runs a CONTROL cell, the same bench at f32 under the same arm's bf16_arithmetic
# flag. The bf16 policy cannot reach an f32 kernel, and before any timing the driver PROVES it: one
# untimed control run per arm writes its generated HIP source, and the two sources must be
# byte-identical or the driver stops. The control's wide/auto ratio is then the session drift the
# treatment ratios are read against.
#
# Exit statuses are validated per cell, not merely printed. schedule_bench exits 1 on a result that
# differs from its unscheduled oracle; its operands are exact in f32, tf32 and f16 but a k-term bf16
# accumulation is not, so a bf16 cell may exit 1 on rounding -- accepted only when the log carries
# the bench's "WRONG RESULT" verdict and no FAILED variant. Any other nonzero status, or a
# nonzero f32 control, fails the run, as does a missing timing line or a summarizer failure.
#
# Hermetic like gh514_cells.sh: every OCANNL_* variable is unset, every treatment is on argv, and the
# bench runs from benchmarks/, whose ocannl_config is the nearest on the upward search.
set -u
[ $# -ge 2 ] || { echo "usage: $0 <repo-root> <memory-label> [rounds] [sizes...]"; exit 2; }
ROOT=$(cd "$1" && pwd) || { echo "no such repo root: $1"; exit 2; }
LABEL=$2
ROUNDS=${3:-6}
shift $(($# < 3 ? $# : 3))
SIZES=("$@")
[ ${#SIZES[@]} -gt 0 ] || SIZES=(1024 2048)
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
for arm in false auto; do
  prefix="gh1051_ctl_$$_$arm"
  bench "${SIZES[0]}" "$arm" single "$OUT/identity_$arm.log" \
    --ocannl_output_debug_files_in_build_directory=true --ocannl_build_files_prefix="$prefix" ||
    { echo "control identity run ($arm) exited $?"; FAIL=1; }
  mkdir -p "$OUT/identity_$arm"
  cp build_files/"$prefix"/*.hip "$OUT/identity_$arm/" 2>/dev/null
  rm -rf build_files/"$prefix" log_files/"$prefix"
done
n_src=$(find "$OUT/identity_false" -name '*.hip' | wc -l)
if [ "$n_src" -eq 0 ] || ! diff -r "$OUT/identity_false" "$OUT/identity_auto" >/dev/null; then
  echo "DRIFT CONTROL INVALID: the f32 control's generated HIP differs across arms (or none was written: $n_src files)"
  exit 1
fi
echo "drift control: $n_src generated f32 kernels byte-identical across arms"
for size in "${SIZES[@]}"; do
  for ((r = 0; r < ROUNDS; r++)); do
    if ((r % 2 == 0)); then order=(false auto); else order=(auto false); fi
    for arm in "${order[@]}"; do
      for prec in bfloat16 single; do
        log="$OUT/n${size}_r${r}_${arm}_${prec}.log"
        bench "$size" "$arm" "$prec" "$log"
        st=$?
        echo "n=$size round=$r arm=$arm prec=$prec exit=$st"
        if grep -q "FAILED" "$log"; then
          echo "  FAILED variant in $log"; FAIL=1
        elif ((st != 0)) && ! { [ "$prec" = bfloat16 ] && ((st == 1)) && grep -q "^WRONG RESULT" "$log"; }; then
          echo "  unexpected exit $st"; FAIL=1
        fi
        for v in parallel smem regtile mma_pd1 mma_pd2; do
          grep -q "^$v  *[0-9.]* ms" "$log" || { echo "  MISSING timing for $v"; FAIL=1; }
        done
      done
    done
  done
done
python3 - "$OUT" "$LABEL" <<'PY' || { echo "SUMMARY FAILED"; FAIL=1; }
import glob, os, re, statistics, sys
out, label = sys.argv[1], sys.argv[2]
cells = {}
for f in glob.glob(os.path.join(out, "n*_r*_*_*.log")):
    m = re.match(r"n(\d+)_r(\d+)_(\w+?)_(bfloat16|single)\.log$", os.path.basename(f))
    size, arm, prec = int(m.group(1)), m.group(3), m.group(4)
    for line in open(f):
        t = re.match(r"^(\w+)\s+([0-9.]+) ms\s+([0-9.]+) GFLOP/s.*\[(.*)\]\s*$", line)
        if t:
            cells.setdefault((size, t.group(1), prec, arm), []).append((float(t.group(2)), t.group(4)))
variants = ["parallel", "smem", "regtile", "mma_pd1", "mma_pd2"]
fmt = lambda xs: f"{statistics.median(x for x, _ in xs):.3f} ({min(x for x, _ in xs):.3f}-{max(x for x, _ in xs):.3f})"
med = lambda xs: statistics.median(x for x, _ in xs)
print("\n| memory | n | variant | auto ms (median, min-max) | wide ms (median, min-max) | wide/auto | f32 control wide/auto | census (wide) |")
print("| --- | --- | --- | --- | --- | --- | --- | --- |")
for size in sorted({k[0] for k in cells}):
    for v in variants:
        a, w = cells[(size, v, "bfloat16", "auto")], cells[(size, v, "bfloat16", "false")]
        ca, cw = cells[(size, v, "single", "auto")], cells[(size, v, "single", "false")]
        print(f"| {label} | {size} | {v} | {fmt(a)} | {fmt(w)} | {med(w) / med(a):.3f} | "
              f"{med(cw) / med(ca):.3f} | {w[0][1]} |")
PY
exit $FAIL
