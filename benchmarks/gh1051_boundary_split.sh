#!/bin/bash
# gh-ocannl-1051 residual: how much of Bf16_wide's tensorized cost on HIP is gh-ocannl-1064's
# coordinate-table d boundary rather than the f32-accumulate WMMA rate? Times bin/schedule_bench
# under --ocannl_bf16_arithmetic=false at two revisions that differ by exactly the gh-1064 commit:
# BEFORE (the staging-fragment x[i] copy) and AFTER (the table-addressed boundary).
#
# Usage: gh1051_boundary_split.sh <root-before> <root-after> <memory-label> [rounds] [sizes...]
# [rounds] must be positive and even (complete ABBA blocks), default 6.
#
# The treatment reaches only Tile_mma emission (mma_pd1/mma_pd2). parallel/smem/regtile are the
# in-process drift controls, and the driver PROVES they are: one untimed run per revision writes the
# generated HIP, and those three variants' sources must be byte-identical across revisions, while
# the mma sources must differ (before: __mma_dstage; after: ocannl_wmma_rc16) -- else it stops.
# Every cell is validated by benchmarks/gh1051_cells.py (shared with gh1051_bf16_ab.sh): its exit
# status, and its whole-output checksums -- identical across rounds, bitwise identical across the
# two revisions (the change touches only how the wide d boundary is addressed, with the same
# conversions), and the mma cells within a structural bound of the same run's serial wide one.
# Hermetic: OCANNL_* unset, treatments on argv, run from benchmarks/.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
[ $# -ge 3 ] || { echo "usage: $0 <root-before> <root-after> <memory-label> [rounds] [sizes...]"; exit 2; }
BEFORE=$(cd "$1" && pwd) || exit 2
AFTER=$(cd "$2" && pwd) || exit 2
LABEL=$3
ROUNDS=${4:-6}
shift $(($# < 4 ? $# : 4))
SIZES=("$@")
[ ${#SIZES[@]} -gt 0 ] || SIZES=(1024 2048)
case $ROUNDS in '' | *[!0-9]*) echo "bad rounds '$ROUNDS'"; exit 2 ;; esac
((ROUNDS > 0 && ROUNDS % 2 == 0)) || { echo "rounds must be positive and even, got $ROUNDS"; exit 2; }
while read -r v; do unset "$v"; done < <(env | sed -n 's/^\(OCANNL_[A-Z0-9_]*\)=.*/\1/p')
for r in "$BEFORE" "$AFTER"; do
  (cd "$r" && dune build bin/schedule_bench.exe 2>/dev/null) || { echo "BUILD FAILED in $r"; exit 1; }
done
declare -A ROOT=([before]=$BEFORE [after]=$AFTER)
OUT=$(mktemp -d "${TMPDIR:-/tmp}/gh1051_split.XXXXXX")
echo "gh1051 boundary split: label=$LABEL rounds=$ROUNDS sizes=${SIZES[*]}"
echo "before=$(git -C "$BEFORE" rev-parse HEAD) after=$(git -C "$AFTER" rev-parse HEAD) host=$(hostname) raw=$OUT"
rocminfo 2>/dev/null | grep -E "Marketing Name|^ *Name: *gfx" | sed 's/^ */device: /' | sort -u
bench() { # rev size log [extra...]
  local rev=$1 size=$2 log=$3
  shift 3
  (cd "${ROOT[$rev]}/benchmarks" &&
    "${ROOT[$rev]}/_build/default/bin/schedule_bench.exe" "$size" 10 "$size" "$size" 0 \
      --ocannl_backend=hip --ocannl_default_prec=bfloat16 --ocannl_bf16_arithmetic=false "$@") \
    >"$log" 2>"$log.err"
}
FAIL=0
for rev in before after; do
  prefix="gh1051_split_$$_$rev"
  bench "$rev" "${SIZES[0]}" "$OUT/identity_$rev.log" \
    --ocannl_output_debug_files_in_build_directory=true --ocannl_build_files_prefix="$prefix"
  mkdir -p "$OUT/identity_$rev"
  cp "${ROOT[$rev]}"/benchmarks/build_files/"$prefix"/*.hip "$OUT/identity_$rev/" 2>/dev/null
  rm -rf "${ROOT[$rev]}"/benchmarks/build_files/"$prefix" "${ROOT[$rev]}"/benchmarks/log_files/"$prefix"
done
for v in parallel smem regtile; do
  f=$(cd "$OUT/identity_before" && ls | grep "^mm_${v}\b.*\.hip$" | head -1)
  if [ -z "$f" ] || ! cmp -s "$OUT/identity_before/$f" "$OUT/identity_after/$f"; then
    echo "CONTROL INVALID: generated $v source (${f:-none}) differs across revisions"; exit 1
  fi
done
for v in mma_pd1 mma_pd2; do
  f=$(cd "$OUT/identity_before" && ls | grep "^mm_${v}.*\.hip$" | head -1)
  grep -q "__mma_dstage" "$OUT/identity_before/$f" && grep -q "ocannl_wmma_rc16" "$OUT/identity_after/$f" &&
    ! grep -q "ocannl_wmma_rc16" "$OUT/identity_before/$f" && ! grep -q "__mma_dstage" "$OUT/identity_after/$f" ||
    { echo "TREATMENT INVALID: $v sources do not show staging-copy before / table after"; exit 1; }
done
echo "controls: parallel/smem/regtile HIP byte-identical across revisions; mma_pd1/2 staging-copy before, table after"
for size in "${SIZES[@]}"; do
  for ((r = 0; r < ROUNDS; r++)); do
    if ((r % 2 == 0)); then order=(before after); else order=(after before); fi
    for rev in "${order[@]}"; do
      log="$OUT/n${size}_r${r}_${rev}.log"
      bench "$rev" "$size" "$log"
      st=$?
      echo "$st" >"$log.status"
      echo "n=$size round=$r rev=$rev exit=$st"
    done
  done
done
python3 "$HERE/gh1051_cells.py" split "$OUT" "$LABEL" || FAIL=1
exit $FAIL
