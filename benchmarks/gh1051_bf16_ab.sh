#!/bin/bash
# gh-ocannl-1051: GEMM throughput A/B of bf16_arithmetic=false (Bf16_wide, f32 accumulators) against
# bf16_arithmetic=auto on one HIP box, the measurement that decides whether auto resolves wide on
# HIP.
#
# Usage: gh1051_bf16_ab.sh <repo-root> <memory-label> [rounds] [sizes...]
#   e.g. gh1051_bf16_ab.sh ~/ocannl "unified (gfx1151)" 5 1024 2048
#
# Each cell is bin/schedule_bench at m = n = k = <size> with bf16 operands and output, run once per
# arm per round. All five variants are the treatment, not only the tensorized mma_pd1/mma_pd2: the
# policy also moves the serial legs' accumulator (HIP's accum_prec), and the flip moves both. The
# arms alternate run by run, in ABBA order across rounds, so neither owns a position in the session
# (docs/agent-notes/training-and-performance.md, the A/B protocol). No schedule cache is involved:
# every variant is a hand-written schedule compiled through [~lowered_transform], never tuned.
#
# schedule_bench EXITS 1 on every bf16 cell and that is expected here: its operands are exact in
# f32, tf32 and f16, but a k-term bf16 accumulation is not, so its "WRONG RESULT" guard (written for
# exact precisions) fires on rounding. The exit status is recorded per cell, not treated as a
# failure; only a missing timing line is. The DIFFERS lines are kept in the raw log because they
# are the accuracy half of the question (auto's bf16-accumulate WMMA drifts far more than the wide
# arm's single narrowing).
#
# Hermetic like gh514_cells.sh: every OCANNL_* variable is unset, every treatment is on argv, and the
# bench runs from benchmarks/, whose ocannl_config is the nearest on the upward search.
set -u
ROOT=$1; LABEL=$2; ROUNDS=${3:-5}; shift 3 2>/dev/null || shift $#
SIZES=("$@")
[ ${#SIZES[@]} -gt 0 ] || SIZES=(1024 2048)
while read -r v; do unset "$v"; done < <(env | sed -n 's/^\(OCANNL_[A-Z0-9_]*\)=.*/\1/p')
cd "$ROOT" || exit 1
dune build bin/schedule_bench.exe 2>/dev/null || { echo "BUILD FAILED"; exit 1; }
BENCH="$ROOT/_build/default/bin/schedule_bench.exe"
cd benchmarks || exit 1
OUT=$(mktemp -d "${TMPDIR:-/tmp}/gh1051_ab.XXXXXX")
echo "gh1051 bf16 A/B: label=$LABEL rounds=$ROUNDS sizes=${SIZES[*]} sha=$(git -C "$ROOT" rev-parse HEAD)"
echo "host=$(hostname) raw=$OUT"
rocminfo 2>/dev/null | grep -E "Marketing Name|^ *Name: *gfx" | sed 's/^ */device: /' | sort -u
MISSING=0
for size in "${SIZES[@]}"; do
  for ((r = 0; r < ROUNDS; r++)); do
    if ((r % 2 == 0)); then order=(false auto); else order=(auto false); fi
    for arm in "${order[@]}"; do
      log="$OUT/n${size}_r${r}_${arm}.log"
      "$BENCH" "$size" 10 "$size" "$size" 0 --ocannl_backend=hip --ocannl_default_prec=bfloat16 \
        --ocannl_bf16_arithmetic="$arm" >"$log" 2>"$log.err"
      echo "n=$size round=$r arm=$arm exit=$?"
      for v in parallel smem regtile mma_pd1 mma_pd2; do
        grep -q "^$v  *[0-9.]* ms" "$log" || { echo "  MISSING timing for $v"; MISSING=1; }
      done
    done
  done
done
python3 - "$OUT" "$LABEL" <<'PY'
import glob, os, re, statistics, sys
out, label = sys.argv[1], sys.argv[2]
cells = {}
for f in glob.glob(os.path.join(out, "n*_r*_*.log")):
    m = re.match(r"n(\d+)_r(\d+)_(\w+)\.log$", os.path.basename(f))
    size, arm = int(m.group(1)), m.group(3)
    for line in open(f):
        t = re.match(r"^(\w+)\s+([0-9.]+) ms\s+([0-9.]+) GFLOP/s.*\[(.*)\]\s*$", line)
        if t:
            cells.setdefault((size, t.group(1), arm), []).append((float(t.group(2)), t.group(4)))
variants = ["parallel", "smem", "regtile", "mma_pd1", "mma_pd2"]
print(f"\n| memory | n | variant | auto ms (median, min-max) | wide ms (median, min-max) | wide/auto | census (wide) |")
print("| --- | --- | --- | --- | --- | --- | --- |")
for size in sorted({k[0] for k in cells}):
    for v in variants:
        a, w = cells.get((size, v, "auto"), []), cells.get((size, v, "false"), [])
        if not a or not w:
            print(f"| {label} | {size} | {v} | missing | missing | | |")
            continue
        am, wm = statistics.median(x for x, _ in a), statistics.median(x for x, _ in w)
        fmt = lambda xs, med: f"{med:.3f} ({min(x for x, _ in xs):.3f}-{max(x for x, _ in xs):.3f})"
        print(f"| {label} | {size} | {v} | {fmt(a, am)} | {fmt(w, wm)} | {wm / am:.3f} | {w[0][1]} |")
PY
exit $MISSING
