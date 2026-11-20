#!/usr/bin/env bash
# Compare small bounded-outcome TSV pairs through the public CLI and check the report and status.
set -u
. "$(cd "$(dirname "$0")/../scripts" && pwd)/harness-support.sh"
harness_args "$@"
harness_require python3
harness_scratch test-compare-bounded-outcomes
HERE=$(cd "$(dirname "$0")" && pwd)
TOOL=$HERE/compare-bounded-outcomes.py
base=$TMP/base.tsv
head=$TMP/head.tsv
out=$TMP/out
accepted='v=2 of default u=1 of bcast_if_1 k=3 of a'
basis_changed='v=2 of a u=1 of bcast_if_1 k=3 of a'
size_changed='v=3 of default u=1 of bcast_if_1 k=3 of a'
unsized='v=2 of default u=1 of bcast_if_1 k=unsized'

row() { printf '%s\t%s\t%s\n' "$@"; }
compare() {
  rc=0
  python3 "$TOOL" "$@" >"$out" 2>&1 || rc=$?
}
detail() { printf '%s\t%s\t%s\t%s\t%s' "$@"; }
has_line() { grep -qxF -- "$1" "$out"; }

if harness_leg 'matching keys compare equally after reordering'; then
  row 'same ; store' '0,1' "$unsized" 'same ; store' 'batch' refused >"$base"
  row 'same ; store' 'batch' refused 'same ; store' '0,1' "$unsized" >"$head"
  compare "$base" "$head"
  if [ "$rc" = 0 ] && has_line 'head: 1 accepted stores, 1 accepted ordered runs, 0 accepted batches'; then
    report 0 'row order is immaterial'
  else report 1 'row order is immaterial' "$(cat "$out")"; fi
fi

row 'gain' '0,1' refused 'gain' 'batch' refused 'loss' '1,0' "$accepted" \
  'basis' '0,1' "$accepted" 'size' '1,0' "$accepted" 'unsized' 'batch' "$accepted" >"$base"
row 'unsized' 'batch' "$unsized" 'basis' '0,1' "$basis_changed" 'size' '1,0' "$size_changed" \
  'loss' '1,0' refused 'gain' 'batch' "$accepted" 'gain' '0,1' "$accepted" >"$head"
expected=(--expect-refused-to-accepted 2 --expect-accepted-to-refused 1 --expect-changed-answer 3)

if harness_leg 'all transition types report their keys and answers'; then
  compare "$base" "$head"
  if [ "$rc" = 1 ] &&
    has_line "$(detail refused-to-accepted gain '0,1' refused "$accepted")" &&
    has_line "$(detail accepted-to-refused loss '1,0' "$accepted" refused)" &&
    has_line "$(detail changed-answer basis '0,1' "$accepted" "$basis_changed")" &&
    has_line "$(detail changed-answer size '1,0' "$accepted" "$size_changed")" &&
    has_line "$(detail changed-answer unsized batch "$accepted" "$unsized")" &&
    has_line 'refused-to-accepted: 2 (1 ordered, 1 batched), expected 0' &&
    has_line 'changed-answer: 3 (2 ordered, 1 batched), expected 0'; then
    report 0 'unexpected transitions fail with the old and new outcomes'
  else report 1 'unexpected transitions fail with the old and new outcomes' "$(cat "$out")"; fi
fi

if harness_leg 'expected counts must match exactly'; then
  compare "$base" "$head" "${expected[@]}"
  report "$rc" 'exact transition counts pass' "$(cat "$out")"
  compare "$base" "$head" --expect-refused-to-accepted 2 --expect-accepted-to-refused 1
  if [ "$rc" = 1 ] && has_line 'changed-answer: 3 (2 ordered, 1 batched), expected 0'; then
    report 0 'allowing acceptance transitions still checks the answers'
  else report 1 'allowing acceptance transitions still checks the answers' "$(cat "$out")"; fi
  compare "$base" "$head" --expect-refused-to-accepted 3 --expect-accepted-to-refused 1 --expect-changed-answer 3
  if [ "$rc" = 1 ] && has_line 'refused-to-accepted: 2 (1 ordered, 1 batched), expected 3'; then
    report 0 'too few expected gains fail'
  else report 1 'too few expected gains fail' "$(cat "$out")"; fi
fi

if harness_leg 'keys missing from either side fail despite matched counts'; then
  cp "$base" "$TMP/base-missing.tsv"
  cp "$head" "$TMP/head-missing.tsv"
  row 'base-only' 'batch' refused >>"$TMP/base-missing.tsv"
  row 'head-only' '0,1' refused >>"$TMP/head-missing.tsv"
  compare "$TMP/base-missing.tsv" "$TMP/head-missing.tsv" "${expected[@]}"
  if [ "$rc" = 1 ] &&
    has_line "$(detail missing-from-head base-only batch refused -)" &&
    has_line "$(detail missing-from-base head-only '0,1' - refused)"; then
    report 0 'both sides report the exact missing keys'
  else report 1 'both sides report the exact missing keys' "$(cat "$out")"; fi
fi

if harness_leg 'invalid dumps and arguments are errors'; then
  row duplicate batch refused duplicate batch refused >"$TMP/duplicate.tsv"
  row invalid '1, 0' refused >"$TMP/order.tsv"
  row invalid '0,0' refused >"$TMP/repeat-order.tsv"
  row invalid '0,2' refused >"$TMP/gap-order.tsv"
  row invalid batch 'possibly accepted' >"$TMP/outcome.tsv"
  printf 'invalid\tbatch\trefused\textra\n' >"$TMP/columns.tsv"
  : >"$TMP/empty.tsv"
  rejects() { # PATTERN BASE HEAD [ARGS...]
    local diagnostic=$1
    shift
    if harness_rejected 2 "$diagnostic" "$out" python3 "$TOOL" "$@"; then
      report 0 "$diagnostic"
    else report 1 "$diagnostic" "$(cat "$out")"; fi
  }
  rejects "duplicate.tsv:2: duplicate key" "$TMP/duplicate.tsv" "$head"
  rejects "duplicate.tsv:2: duplicate key" "$base" "$TMP/duplicate.tsv"
  rejects "invalid store or order" "$base" "$TMP/order.tsv"
  rejects "repeat-order.tsv:1: invalid store or order" "$base" "$TMP/repeat-order.tsv"
  rejects "gap-order.tsv:1: invalid store or order" "$base" "$TMP/gap-order.tsv"
  rejects "invalid outcome" "$TMP/outcome.tsv" "$head"
  rejects "expected three tab-separated fields" "$TMP/columns.tsv" "$head"
  rejects "empty dump" "$base" "$TMP/empty.tsv"
  rejects "expected counts must be nonnegative" "$base" "$head" --expect-changed-answer -1
fi

finish
