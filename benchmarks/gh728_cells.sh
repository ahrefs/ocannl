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
#   psb_exact_rev   the same, in the reversed rotation                          (order control)
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
# Positive and even, as the bench itself requires: zero batches time nothing (and the gate would
# have no contenders), zero repeats divide by zero; odd counts break the mirrored visiting order.
for n in "$REPEATS" "$BATCHES"; do
  case $n in '' | *[!0-9]*) echo "gh728_cells: repeats and batches must be integers, got '$n'" >&2; exit 2 ;; esac
  ((10#$n > 0 && 10#$n % 2 == 0)) || { echo "gh728_cells: repeats and batches must be positive and even, got $n" >&2; exit 2; }
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
# The OpenMP runtime's controls shape the cc backend's parallel Grid kernels (team size, placement):
# cleared, so every cc leg runs the runtime's own default regime.
while read -r v; do unset "$v"; done < <(env | sed -nE 's/^((OMP|GOMP|KMP)_[A-Z0-9_]*)=.*/\1/p')

# The commit the binaries are built from, marked when the tree carries anything that commit does
# not: a modified tracked file, or an untracked non-ignored one (a stray root `dune` or
# `dune-workspace` is a build input too). Read BEFORE the build and the legs, so nothing this run
# writes can mark it; <out-dir> is refused inside the tree for the same reason.
case $OUT/ in "$TREE"/*) echo "gh728_cells: <out-dir> must be outside the checkout $TREE" >&2; exit 2 ;; esac
COMMIT=$(git -C "$TREE" rev-parse HEAD) || exit 2
[ -z "$(git -C "$TREE" status --porcelain --untracked-files=all)" ] || COMMIT="$COMMIT+uncommitted"
(cd "$TREE" && dune build bin/projection_shape_bench.exe benchmarks/runners/ocannl/bench_gpt.exe) ||
  { echo "gh728_cells: build failed" >&2; exit 1; }
PSB=$TREE/_build/default/bin/projection_shape_bench.exe
GPT=$TREE/_build/default/benchmarks/runners/ocannl/bench_gpt.exe
# The workload's bytes must be SOME recorded origin's (benchmarks/fixtures/DIGESTS.txt): numbers
# on unrecorded bytes compare with nothing, and the raw hash alone cannot say so (entries may be
# content digests). The verdict line, naming whose bytes these are, goes into the manifest.
FIXTURE_VERDICT=$(cd "$TREE/benchmarks" && python3 fixture_digest.py --check "$FIXTURE") &&
  case $FIXTURE_VERDICT in *" — MATCH"*) ;; *) false ;; esac ||
  { echo "gh728_cells: fixture is not a recorded origin's bytes: ${FIXTURE_VERDICT:-no verdict}" >&2; exit 2; }
# Which device the numbers are OF, selected by the backend under test and refused when empty: a
# manifest that names no device cannot be read or reproduced.
case $BACKEND in
  cuda) DEVICE=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null) ;;
  hip) DEVICE=$(rocminfo 2>/dev/null | grep -E "Marketing Name|^ *Name: *gfx" | sed 's/^ *//' | sort -u) ;;
  metal) DEVICE=$(system_profiler SPDisplaysDataType 2>/dev/null | grep -E "Chipset Model|Total Number of Cores" | sed 's/^ *//') ;;
  cc) DEVICE=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || lscpu 2>/dev/null | grep -E "^Model name") ;;
esac
[ -n "$DEVICE" ] || { echo "gh728_cells: could not identify the $BACKEND device" >&2; exit 2; }
{
  echo "gh728_cells: backend=$BACKEND host=$(hostname) commit=$COMMIT repeats=$REPEATS batches=$BATCHES"
  echo "fixture: $FIXTURE: $FIXTURE_VERDICT"
  printf '%s\n' "$DEVICE" | sed 's/^/device: /'
  # Device selection is the box's configuration rather than a treatment: recorded, not cleared.
  env | grep -E '^(CUDA|HIP|ROCR)_VISIBLE_DEVICES=' | sed 's/^/device selection: /'
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
# The reversed rotation runs the same mode as the forward leg -- searches included -- so the two
# finalists rounds are timed in the same (search-loaded) process state and differ in visiting order
# only; the second set of searches is also a replicate of what the tuner crowns.
leg psb_exact_rev "$PSB" "$REPEATS" "$BATCHES" qkv rev both --with-mma "${PIN[@]}"
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
    # The two-pass protocol is what the replay's kernel table is quoted for, so it is verified off
    # the runner's own result line rather than assumed from the exit status: the first pass must
    # have searched, and the second must have replayed every arm and searched none -- a replay over
    # an incomplete cache searches the missing arms and would publish a search-loaded table.
    (
      set -o pipefail
      python3 - "$OUT/gpt_${profile}_$pass.out" "$pass" <<'VERIFY' 2>&1 | tee -a "$OUT/manifest.txt"
import json, sys
path, want = sys.argv[1], sys.argv[2]
rows = [json.loads(l) for l in open(path) if l.startswith('{"framework"')]
if len(rows) != 1:
    sys.exit(f"   protocol: {path}: expected one result line, found {len(rows)}")
tune = rows[0].get("tune") or {}
searched, n_search, n_replay = rows[0].get("searched"), tune.get("searches"), tune.get("replays")
n_none = tune.get("no_searches")
# An arm that neither searched nor replayed (a pre-search failure) ships the untuned default:
# neither pass may carry one.
ok = n_none == 0 and ((searched is True and (n_search or 0) > 0) if want == "search" else (
    searched is False and n_search == 0 and (n_replay or 0) > 0))
line = (f"   protocol: {want} pass searched={searched} searches={n_search} replays={n_replay}"
        f" no_searches={n_none}")
print(line + ("" if ok else "  -- NOT A " + want.upper() + " PASS"))
sys.exit(0 if ok else 1)
VERIFY
    ) || FAILED=1
  done
done
echo "gh728_cells: done, failed=$FAILED" | tee -a "$OUT/manifest.txt"
exit "$FAILED"
