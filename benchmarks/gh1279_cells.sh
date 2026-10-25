#!/bin/bash
# gh-ocannl-1279: the attention out projection's contraction-coalesced seeds (the (h, e) pair merged
# into one k, tiles up to 256 deep) against its best uncoalesced seeds, on one box. One driver, so
# the closing comment names this invocation rather than transcribing commands.
#
# Usage: gh1279_cells.sh <out-dir> <backend> [repeats batches]
#   [repeats batches] (default 200 8, both even) size bin/projection_shape_bench's timing; anything
#   smaller is a smoke run of the driver, not a measurement.
#
# Legs, in order (each writes <leg>.out / <leg>.err under <out-dir>, then an `exit: N` line):
#   psb_fwd   bin/projection_shape_bench <repeats> <batches> outproj fwd both
#             every scalar seed of the out projection's sites (gpt2_mini's b8 x 128 rows and its
#             merged 1024 rows, j = 256, (h, e) = (8, 32), plus a 16-row twin at the split-reduce
#             eligibility bound), uncoalesced and contraction-coalesced (coal-k), timed in
#             interleaved geometry rounds, then a full search per site (empty cache), then one
#             finalists round holding each site's best uncoalesced seed, best coal-k seed, the
#             search's crown and the untuned default
#   psb_rev   the same in the reversed rotation, seeds only                    (order control)
# Each leg's `gh1279:` lines are echoed: per site the two classes' median-best finalists with their
# k-tile, staged workgroup memory, p10/p50/p90 batch times and shipped kernels, the throughput
# ratio, and (psb_fwd) the search's crown with its wall-clock cost on the `crowned:` line. Every
# candidate is checked cell by cell against a host oracle on inputs whose sums are exact in f32.
#
# Hermetic: OCANNL_* and BENCH_* are unset, the treatment is on argv, the runner starts from
# benchmarks/ (the nearest ocannl_config; bin/ has the cwd trap). No fixture is read. Exits 1 when
# any leg failed, 3 when the only faults are legs whose timings were all refused for host contention
# (the bench's own exit 3: the box's verdict, not the experiment's), 0 otherwise.
set -u
[ $# -eq 2 ] || [ $# -eq 4 ] || { sed -n '6,8p' "$0"; exit 2; }
OUT=$1 BACKEND=$2
REPEATS=${3:-200}
BATCHES=${4:-8}
for n in "$REPEATS" "$BATCHES"; do
  case $n in '' | *[!0-9]*) echo "gh1279_cells: repeats and batches must be integers, got '$n'" >&2; exit 2 ;; esac
  ((10#$n > 0 && 10#$n % 2 == 0)) || { echo "gh1279_cells: repeats and batches must be positive and even, got $n" >&2; exit 2; }
done
case $BACKEND in cuda | hip | metal) ;; *) echo "gh1279_cells: a GPU backend (cuda, hip, metal), got '$BACKEND'" >&2; exit 2 ;; esac
while read -r v; do unset "$v"; done < <(env | sed -n 's/^\(OCANNL_[A-Z0-9_]*\)=.*/\1/p')
while read -r v; do unset "$v"; done < <(env | sed -n 's/^\(BENCH_[A-Z0-9_]*\)=.*/\1/p')
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
mkdir -p "$OUT" || exit 1
OUT=$(CDPATH= cd -- "$OUT" && pwd)
[ -z "$(ls -A "$OUT")" ] || { echo "gh1279_cells: $OUT is not empty" >&2; exit 2; }
COMMIT=$(git -C "$ROOT" rev-parse HEAD) || exit 1
[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=all)" ] || COMMIT="$COMMIT+uncommitted"
echo "gh1279_cells: commit $COMMIT, backend $BACKEND, repeats $REPEATS, batches $BATCHES" | tee "$OUT/manifest"
cd "$ROOT" || exit 1
dune build bin/projection_shape_bench.exe 2>/dev/null || { echo "BUILD FAILED"; exit 1; }
cd benchmarks || exit 1
EXE=../_build/default/bin/projection_shape_bench.exe
rc=0
leg() { # NAME ORDER MODE
  "$EXE" "$REPEATS" "$BATCHES" outproj "$2" "$3" --ocannl_backend="$BACKEND" \
    >"$OUT/$1.out" 2>"$OUT/$1.err"
  local r=$?
  echo "exit: $r" >>"$OUT/$1.out"
  echo "leg $1 ($2 $3): exit: $r"
  grep -E '^gh1279:|crowned:' "$OUT/$1.out"
  # A failure outranks a refusal, whichever leg came last.
  if [ "$r" -ne 0 ] && [ "$rc" -ne 1 ]; then rc=$r; fi
  return 0
}
leg psb_fwd fwd both
leg psb_rev rev seeds
exit $rc
