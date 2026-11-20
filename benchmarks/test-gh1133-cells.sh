#!/usr/bin/env bash
# Driver-level controls for benchmarks/gh1133_cells.sh's exit status: an
# invocation with no `step` step (a prep or dry run) owes no step-time matrix
# and exits 0 when its steps pass, while one that asks for `step` and gets no
# result line still exits 1. The summary's own rendering is pinned by
# test/operations/gh1133_summary.py; this harness pins the driver's half, that
# it tells the summary which kind of run it was. It also pins the startup line
# stating what the default KERNEL_TABLE=1 costs (the driver header's "What
# KERNEL_TABLE=1 costs"): on stderr for a run of more than three step/kprof
# cells, absent with KERNEL_TABLE=0, and naming the runner's own default
# instrument budget.
# Usage: benchmarks/test-gh1133-cells.sh [--keep|--help]
#
# The driver runs from a copy in a scratch tree beside the shipping summary
# module, on a fresh OUT per leg; it reaches no dune, runner or fixture, since
# the legs request only `summary` and a `step` that its build/provenance
# precondition refuses. Each oracle has a mutated twin it must reject.
set -u
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
. "$here/../scripts/harness-support.sh"
harness_args "$@"
harness_require python3
harness_scratch test-gh1133-cells
SRC=$here/gh1133_cells.sh
tree=$TMP/tree
mkdir -p "$tree/benchmarks" "$TMP/base" "$TMP/runs"
cp "$here/gh1133_summary.py" "$here/bench_output.py" "$here/strict_json.py" "$here/display_safe.py" "$here/box-gate.sh" "$tree/benchmarks/" || exit 2

# [drive SUBJECT RUN STEP...]: run SUBJECT as the scratch tree's driver with
# OUT=runs/RUN/out; its stdout (the driver log) goes to runs/RUN/stdout. The
# box gate is skipped: the scratch tree is no measurement box, and the gate's
# own legs are benchmarks/test-box-gate.sh's.
drive() {
  local subject=$1 run=$2
  shift 2
  mkdir -p "$TMP/runs/$run"
  cp "$subject" "$tree/benchmarks/gh1133_cells.sh" || return 2
  got=0
  bash "$tree/benchmarks/gh1133_cells.sh" --skip-box-check "$TMP/runs/$run/out" "$TMP/base" 60 "$@" \
    >"$TMP/runs/$run/stdout" 2>&1 || got=$?
}
prep_run_passes() { # SUBJECT RUN
  drive "$1" "$2" summary
  [ "$got" = 0 ] && grep -q 'No step stage requested' "$TMP/runs/$2/stdout" \
    && ! grep -q 'NO MEASUREMENT' "$TMP/runs/$2/stdout"
}
empty_step_stage_fails() { # SUBJECT RUN
  drive "$1" "$2" step summary
  [ "$got" = 1 ] && grep -q 'NO MEASUREMENT' "$TMP/runs/$2/out/summary.md" \
    && ! grep -q 'No step stage requested' "$TMP/runs/$2/stdout"
}

if prep_run_passes "$SRC" prep; then report 0 'a run without a step stage exits 0 when its steps pass'
else report 1 'a run without a step stage exits 0 when its steps pass' "see $TMP/runs/prep (--keep)"; fi
if empty_step_stage_fails "$SRC" step; then report 0 'a step stage with no result line exits 1'
else report 1 'a step stage with no result line exits 1' "see $TMP/runs/step (--keep)"; fi

# The table's cost line: the default table over FIXTURES x TREATMENTS x REPEATS
# = 2 x 2 x 3 = 12 step cells is announced at startup -- the first line, on
# stderr, before the driver log (which the driver prints at exit) -- and is in
# the log too; KERNEL_TABLE=0 announces nothing.
table_note_shown() { # SUBJECT RUN
  local note='^gh1133: KERNEL_TABLE=1 re-times every shipped kernel of each of 12 step/kprof cells, up to the runner.s 600 s instrument budget a cell: allow up to 120 min'
  # KERNEL_TABLE empty reads as unset, so an exported KERNEL_TABLE=0 cannot turn the default off.
  KERNEL_TABLE= FIXTURES="fa fb" TREATMENTS="base keep" REPEATS=3 drive "$1" "$2" step
  head -n 1 "$TMP/runs/$2/stdout" | grep -q "$note" && grep -q "$note" "$TMP/runs/$2/out/driver.log"
}
step_only_quiet() { # SUBJECT RUN
  KERNEL_TABLE=0 FIXTURES="fa fb" TREATMENTS="base keep" REPEATS=3 drive "$1" "$2" step
  grep -q 'kernel table: 0;' "$TMP/runs/$2/stdout" && ! grep -q 'KERNEL_TABLE=1 re-times' "$TMP/runs/$2/stdout"
}
if table_note_shown "$SRC" table; then report 0 'KERNEL_TABLE=1 states its cost at startup'
else report 1 'KERNEL_TABLE=1 states its cost at startup' "see $TMP/runs/table (--keep)"; fi
if step_only_quiet "$SRC" notable; then report 0 'KERNEL_TABLE=0 states no table cost'
else report 1 'KERNEL_TABLE=0 states no table cost' "see $TMP/runs/notable (--keep)"; fi
# The 600 s the line quotes is the runner's default budget, which the driver
# cannot override (it clears BENCH_*).
budget=$(sed -n 's/^let default_dominant_kernel_budget_s = \([0-9]*\)\.$/\1/p' "$here/runners/ocannl/bench_harness.ml")
if [ "$budget" = 600 ]; then report 0 "the cost line's 600 s is the runner's default instrument budget"
else report 1 "the cost line's 600 s is the runner's default instrument budget" "bench_harness.ml says '${budget}'"; fi
quiet=$(mutant quiet-table '/^  echo "\$table_note" >&2$/ { next } { print }') || exit 2
cmp -s "$SRC" "$quiet" && { echo "mutant quiet-table changed nothing" >&2; exit 2; }
expect_rejected 'the table cost goes unannounced' "$quiet" table_note_shown 'kernel table: 1;'

# The driver as it was: every summary owes the matrix.
owes=$(mutant owes-matrix '/matrix=\(--no-step-stage\)/ { next } { print }') || exit 2
cmp -s "$SRC" "$owes" && { echo "mutant owes-matrix changed nothing" >&2; exit 2; }
expect_rejected 'every summary owes a matrix' "$owes" prep_run_passes 'NO MEASUREMENT'
# A driver that never tells the summary a step stage ran.
never=$(mutant never-step '/\[ "\$step_stage" = 1 \] \|\| matrix=/ { print "      matrix=(--no-step-stage)"; next } { print }') || exit 2
cmp -s "$SRC" "$never" && { echo "mutant never-step changed nothing" >&2; exit 2; }
expect_rejected 'the step stage goes unreported' "$never" empty_step_stage_fails 'No step stage requested'
finish
