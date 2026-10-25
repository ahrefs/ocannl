#!/bin/bash
# gh-ocannl-1343 measurement driver: paired before/after timing of the mlp_bn_names training step
# (test/training/bench_mlp_bn_step.ml) through the UNTUNED default CPU pipeline. "after" is ROOT's
# HEAD; "before" is HEAD with only the scheduler sources (arrayjit/lib/schedule.ml and .mli, the
# files gh-ocannl-1343 changes) taken from BASE -- a commit this driver writes with git plumbing (a
# scratch index, write-tree, commit-tree), touching no branch, index or working tree -- so the two
# binaries differ in the scheduler alone, the bench program, workload and every other library being
# identical, and the report names both commits. The two revisions are built and measured by
# benchmarks/ab.py (gh-ocannl-1443): each built once in a worktree of its own, then the two
# executables as the arms of one bin_cells run -- hermetic (no OCANNL_*/BENCH_*/OpenMP variable: on
# Linux the pool-parallel loops run on OpenMP, and an inherited OMP_NUM_THREADS=1 would quietly
# serialize exactly what is measured), backend pinned to cc, no config file, the configuration
# records read back, supervised, interleaved AB BA ..., paired round by round with the exact sign
# test.
#
# Usage:
#   benchmarks/gh1343_ab.sh ROOT BASE OUT ROUNDS STEPS [AB-OPTION...]
#
#   ROOT    the checkout whose HEAD is "after"; refused with uncommitted tracked changes, which no
#           commit would carry.
#   BASE    the revision the "before" scheduler sources come from.
#   OUT     ab.py's output directory: outside ROOT, created, refused unless empty. OUT/cells/report.md
#           is the result (also printed).
#   ROUNDS  a positive even count.
#   STEPS   timed steps per cell (after 20 warmup steps).
#   AB-OPTION  passed to ab.py, e.g. --build-jobs N, --keep-worktrees, --cell-timeout S.
#
# Exit: ab.py's -- 0 every cell clean; 1 a cell or a build failed; 2 usage; 3 the only faults are
# cells refusing their measurement for host contention.
set -u
usage() { sed -n '/^# Usage/,/^# Exit/p' "$0" | sed '$d' >&2; exit 2; }
HERE=$(cd "$(dirname "$0")" && pwd -P)
SCHEDULER=(arrayjit/lib/schedule.ml arrayjit/lib/schedule.mli)

before_commit() { # ROOT HEAD BASE: prints HEAD's tree with BASE's scheduler sources, as a commit
  local idx rc
  idx=$(mktemp -d) || return 1
  (
    export GIT_INDEX_FILE=$idx/index
    git -C "$1" read-tree "$2" || exit 1
    for f in "${SCHEDULER[@]}"; do
      entry=$(git -C "$1" ls-tree "$3" -- "$f") || exit 1
      [ -n "$entry" ] || { echo "gh1343: $f is not in $3" >&2; exit 1; }
      read -r mode _ sha _ <<<"$entry"
      git -C "$1" update-index --cacheinfo "$mode,$sha,$f" || exit 1
    done
    tree=$(git -C "$1" write-tree) || exit 1
    git -C "$1" commit-tree "$tree" -p "$2" -m "gh1343_ab: $2 with the scheduler sources of $3"
  )
  rc=$?
  rm -rf "$idx"
  return $rc
}

[ $# -ge 5 ] || usage
root=$(cd "$1" && pwd -P) || exit 2
base=$2 out=$3 rounds=$4 steps=$5
shift 5
case $steps in '' | *[!0-9]*) echo "gh1343: STEPS must be a positive integer, got '$steps'" >&2; exit 2 ;; esac
[ -z "$(git -C "$root" status --porcelain --untracked-files=no)" ] ||
  { echo "gh1343: $root has uncommitted tracked changes; commit them first" >&2; exit 2; }
head=$(git -C "$root" rev-parse --verify HEAD) || exit 2
base_commit=$(git -C "$root" rev-parse --verify --quiet "$base^{commit}") ||
  { echo "gh1343: $base names no commit of $root" >&2; exit 2; }
before=$(before_commit "$root" "$head" "$base_commit") || exit 1
echo "gh1343: before=$before (HEAD $head, scheduler from $base_commit), after=$head" >&2
exec python3 "$HERE/ab.py" "$before" "$head" --repo "$root" \
  --target test/training/bench_mlp_bn_step.exe --backend cc --out "$out" --rounds "$rounds" "$@" \
  -- "$steps"
