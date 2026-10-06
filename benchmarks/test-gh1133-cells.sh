#!/usr/bin/env bash
# Driver-level controls for benchmarks/gh1133_cells.sh's exit status: an
# invocation with no `step` step (a prep or dry run) owes no step-time matrix
# and exits 0 when its steps pass, while one that asks for `step` and gets no
# result line still exits 1. The summary's own rendering is pinned by
# test/operations/gh1133_summary.py; this harness pins the driver's half, that
# it tells the summary which kind of run it was.
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
cp "$here/gh1133_summary.py" "$tree/benchmarks/" || exit 2

# [drive SUBJECT RUN STEP...]: run SUBJECT as the scratch tree's driver with
# OUT=runs/RUN/out; its stdout (the driver log) goes to runs/RUN/stdout.
drive() {
  local subject=$1 run=$2
  shift 2
  mkdir -p "$TMP/runs/$run"
  cp "$subject" "$tree/benchmarks/gh1133_cells.sh" || return 2
  got=0
  bash "$tree/benchmarks/gh1133_cells.sh" "$TMP/runs/$run/out" "$TMP/base" 60 "$@" \
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

# The driver as it was: every summary owes the matrix.
owes=$(mutant owes-matrix '/matrix=\(--no-step-stage\)/ { next } { print }') || exit 2
cmp -s "$SRC" "$owes" && { echo "mutant owes-matrix changed nothing" >&2; exit 2; }
expect_rejected 'every summary owes a matrix' "$owes" prep_run_passes 'NO MEASUREMENT'
# A driver that never tells the summary a step stage ran.
never=$(mutant never-step '/\[ "\$step_stage" = 1 \] \|\| matrix=/ { print "      matrix=(--no-step-stage)"; next } { print }') || exit 2
cmp -s "$SRC" "$never" && { echo "mutant never-step changed nothing" >&2; exit 2; }
expect_rejected 'the step stage goes unreported' "$never" empty_step_stage_fails 'No step stage requested'
finish
