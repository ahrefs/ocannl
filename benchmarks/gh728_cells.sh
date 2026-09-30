#!/bin/bash
# gh-ocannl-728 Move 0: does coalescing the q/k/v projections' head axis into the column tile pay
# today? One driver, one box, every leg of the measurement the GO/NO-GO comment quotes -- so the
# comment names this invocation rather than transcribing commands.
#
# Usage: gh728_cells.sh <backend> <out-dir> <gpt2_mini.safetensors> [repeats batches]
# [repeats batches] (default 200 8, both even) size the bench's timing; anything smaller is a smoke
# run of the driver itself, not a measurement.
#
# Legs, in order (each writes <leg>.out / <leg>.err under <out-dir>, then an `exit: N` line):
#   psb_exact_fwd   bin/projection_shape_bench <repeats> <batches> qkv fwd both --with-mma (exact)
#   psb_exact_rev   the same seeds in the reversed rotation, no search          (order control)
#   psb_approx_fwd  psb_exact_fwd under --ocannl_profile=approximate            (tf32 tensorized seeds
#                   on CUDA; HIP has no f32 tile shape, so there it is a replicate of the scalar legs)
#   gpt_default     bench_gpt, untuned, with the per-kernel table              (the shipped default)
#   gpt_exact_search / gpt_exact_replay       bench_gpt tuned: the searching pass, then a fresh
#   gpt_approx_search / gpt_approx_replay     process replaying its winner (the replay's per-kernel
#                                             table is the one quoted; a search pass's launches are
#                                             inflated by its accumulated modules)
# Every search starts from an EMPTY schedule cache (bench_gpt: a fresh autotune_cache_dir per
# profile under <out-dir>; projection_shape_bench: ~cache_dir:"" by construction) under the default
# timing objective, which the bench's search header prints (autotune_timing=queued).
#
# Hermetic: OCANNL_* and BENCH_* are unset, every treatment is on argv, the runners start from
# benchmarks/ (the nearest ocannl_config; bin/ has the cwd trap). Exits nonzero when any leg did.
set -u
[ $# -eq 3 ] || [ $# -eq 5 ] ||
  { echo "usage: $0 <backend> <out-dir> <gpt2_mini.safetensors> [repeats batches]" >&2; exit 2; }
BACKEND=$1
REPEATS=${4:-200}
BATCHES=${5:-8}
for n in "$REPEATS" "$BATCHES"; do
  case $n in '' | *[!0-9]*) echo "gh728_cells: repeats and batches must be integers, got '$n'" >&2; exit 2 ;; esac
done
case $BACKEND in cuda | hip | metal | cc) ;; *) echo "gh728_cells: unknown backend '$BACKEND'" >&2; exit 2 ;; esac
mkdir -p "$2" || { echo "gh728_cells: cannot create $2" >&2; exit 2; }
OUT=$(cd "$2" && pwd -P) || OUT=""
[ -n "$OUT" ] && [ -d "$OUT" ] || { echo "gh728_cells: not a usable directory: $2" >&2; exit 2; }
# A fresh directory only: a leftover cache would make a "search" a replay, and a leftover log would
# be read as this run's.
[ -z "$(ls -A "$OUT")" ] || { echo "gh728_cells: $OUT is not empty" >&2; exit 2; }
FIXTURE=$(cd "$(dirname "$3")" 2>/dev/null && pwd -P)/$(basename "$3")
[ -f "$FIXTURE" ] || { echo "gh728_cells: no fixture at $3" >&2; exit 2; }
TREE=$(cd "$(dirname "$0")/.." && pwd -P) || exit 2
while read -r v; do unset "$v"; done < <(env | sed -n 's/^\(OCANNL_[A-Z0-9_]*\)=.*/\1/p')
while read -r v; do unset "$v"; done < <(env | sed -n 's/^\(BENCH_[A-Z0-9_]*\)=.*/\1/p')

(cd "$TREE" && dune build bin/projection_shape_bench.exe benchmarks/runners/ocannl/bench_gpt.exe) ||
  { echo "gh728_cells: build failed" >&2; exit 1; }
PSB=$TREE/_build/default/bin/projection_shape_bench.exe
GPT=$TREE/_build/default/benchmarks/runners/ocannl/bench_gpt.exe
{
  echo "gh728_cells: backend=$BACKEND host=$(hostname) commit=$(git -C "$TREE" rev-parse HEAD)$(git -C "$TREE" diff --quiet HEAD -- || echo +uncommitted) repeats=$REPEATS batches=$BATCHES"
  echo "fixture: $FIXTURE sha256=$( (sha256sum "$FIXTURE" 2>/dev/null || shasum -a 256 "$FIXTURE") | cut -d' ' -f1)"
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | sed 's/^/device: /'
  rocminfo 2>/dev/null | grep -E "Marketing Name|^ *Name: *gfx" | sed 's/^ */device: /' | sort -u
} | tee "$OUT/manifest.txt"

FAILED=0
leg() { # name cmd...
  local name=$1 rc
  shift
  echo "== $name: $*" | tee -a "$OUT/manifest.txt"
  (cd "$TREE/benchmarks" && "$@") >"$OUT/$name.out" 2>"$OUT/$name.err"
  rc=$?
  echo "exit: $rc" >>"$OUT/$name.out"
  echo "   exit: $rc" | tee -a "$OUT/manifest.txt"
  [ "$rc" -eq 0 ] || FAILED=1
}
PIN=(--ocannl_backend="$BACKEND")
APPROX=(--ocannl_profile=approximate)

leg psb_exact_fwd "$PSB" "$REPEATS" "$BATCHES" qkv fwd both --with-mma "${PIN[@]}"
leg psb_exact_rev "$PSB" "$REPEATS" "$BATCHES" qkv rev seeds --with-mma "${PIN[@]}"
leg psb_approx_fwd "$PSB" "$REPEATS" "$BATCHES" qkv fwd both --with-mma "${APPROX[@]}" "${PIN[@]}"

leg gpt_default env BENCH_FIXTURE="$FIXTURE" BENCH_KERNEL_TABLE=1 "$GPT" "${PIN[@]}"
for profile in exact approx; do
  extra=()
  [ "$profile" = approx ] && extra=("${APPROX[@]}")
  cache=$OUT/cache_$profile
  mkdir -p "$cache"
  for pass in search replay; do
    leg "gpt_${profile}_$pass" env BENCH_FIXTURE="$FIXTURE" BENCH_TUNE=1 BENCH_KERNEL_TABLE=1 \
      "$GPT" ${extra[@]+"${extra[@]}"} --ocannl_autotune_cache_dir="$cache" "${PIN[@]}"
  done
done
echo "gh728_cells: done, failed=$FAILED" | tee -a "$OUT/manifest.txt"
exit "$FAILED"
