#!/usr/bin/env bash

# Integration coverage for tools/sweep.sh's execution-accounting contract. The
# fake opam keeps the test small; real git/worktree operations exercise the
# history migration and the reused-worktree path that made cached GPU passes
# ambiguous in the first place.

set -euo pipefail

# Most assertions below are deliberately quiet shell predicates. If one fails
# under errexit, name the exact site before cleanup removes its evidence; the
# expected-error controls temporarily disable errexit and therefore stay quiet.
on_error() {
  local rc=$1 line=$2 command=$3 name
  case $- in
    *e*) printf 'sweep_harness: line %s failed (exit %s): %s\n' "$line" "$rc" "$command" >&2 ;;
    *) return "$rc" ;;
  esac
  # And what the run under test actually said. Nearly every assertion here is a
  # quiet predicate over a CAPTURED string that no file holds, so without this a
  # failure reaches CI as a bare `grep -q` and reproducing it is the only way to
  # learn what the sweep printed (gh-ocannl-893 was diagnosed that way). Named
  # indirectly rather than by a per-capture dump, so a capture added later is
  # covered by adding its name here and nothing else.
  for name in incremental forced slow_forced coverage hostile complete_fail \
    environment_executed partial_matrix singleton_fail repeated_backend_fail \
    repeated_backend_pass mixed_scope_fail mixed_scope_cleared historical_matrix \
    local_identity_error unsafe_identity_error only_typo_error matrix_error state_first state_same \
    state_other_ref state_green state_unjudged state_regression state_after_fix state_moved \
    capped capped_target serial_red serial_clean serial_control; do
    [ -n "${!name:-}" ] || continue
    printf -- '--- %s ---\n%s\n' "$name" "${!name}" >&2
  done
  return "$rc"
}
trap 'on_error "$?" "$LINENO" "$BASH_COMMAND"' ERR

# `! cmd` is exempt from errexit -- bash does not exit on a command whose value
# is being inverted -- so a negative assertion spelled that way can never fail
# this harness. All eight of them were inert, which is why the ambient-backend
# leak below was reported three assertions past the site that saw it first
# (gh-ocannl-893). Routed through a function, the command errexit weighs is the
# CALL, and the ERR trap above names its line. Spelled with `if` rather than the
# inversion it replaces, and saying what matched: from inside a function the
# trap's `$BASH_COMMAND` is the body, which would name no pattern at all.
absent() {
  if grep -q "$@"; then
    printf 'sweep_harness: unexpected match for %s\n' "$*" >&2
    return 1
  fi
}

# The fixture's inputs come from this file and nowhere else. This harness runs
# as a test action INSIDE a sweep unit, so anything the launching sweep exports
# is in scope here; the SWEEP_TEST_ names are what the fake opam below reads, and
# an inherited one would rewrite a unit's log without appearing at any call site.
# The nested sweep's own variables are neutralized at `run_sweep_args`, which is
# where its environment is built.
unset SWEEP_TEST_CALLS SWEEP_TEST_WAIT_PREFIX SWEEP_TEST_OPAM_RC \
  SWEEP_TEST_OPAM_OUT SWEEP_TEST_OPAM_OUT_CC SWEEP_TEST_OPAM_OUT_MULTIDEV_CC \
  SWEEP_TEST_OPAM_OUT_METAL SWEEP_TEST_LOCAL_BOX SWEEP_TEST_JOBS \
  SWEEP_TEST_OPAM_SERIAL_RED SWEEP_TEST_OPAM_OUT_SERIAL

sweep=$1
aggregate=$2
verdict_probe=$(cd "$(dirname "$3")" && pwd)/$(basename "$3")
rendered_metal_options=$4
rendered_hip_options=$5
rendered_nvrtc_options=$6
tmp=$(mktemp -d "${TMPDIR:-/tmp}/ocannl-sweep-test.XXXXXX")
holder_pid=
wait_prefix=
cleanup() {
  [ -n "$wait_prefix" ] && touch "$wait_prefix.release"
  if [ -n "$holder_pid" ]; then
    kill "$holder_pid" 2>/dev/null || true
    wait "$holder_pid" 2>/dev/null || true
  fi
  rm -rf "$tmp"
}
trap cleanup EXIT

origin=$tmp/origin.git
main=$tmp/main
state=$tmp/state
fake_bin=$tmp/bin
calls=$tmp/opam.calls
mkdir -p "$state/logs" "$fake_bin"

git init -q --bare "$origin"
git init -q -b master "$main"
git -C "$main" config user.name sweep-test
git -C "$main" config user.email sweep-test@example.invalid
mkdir -p "$main/benchmarks/fixtures"
printf '# measurement-boxes: m4-max minix rog-nv\n' >"$main/benchmarks/fixtures/DIGESTS.txt"
printf 'fixture\n' >"$main/fixture"
mkdir -p "$main/test"
printf 'initial golden\n' >"$main/test/unit.cc_expected.ml"
printf 'unrelated fixture\n' >"$main/test/noise.expected"
printf 'pre-diff golden\n' >"$main/test/pre_diff_expected.ml"
printf 'let%%expect_test _ = print_endline "old" [%%expect {| old |}]\n' \
  >"$main/test/inline_expect.ml"
printf '(rule\n (alias runtest-state-probe)\n (deps unit.cc_expected.ml noise.expected)\n (action (diff "unit.%%{read:../config/ocannl_backend.txt}_expected.ml" unit.actual)))\n(rule\n (alias runtest-pre-diff-probe)\n (deps pre_diff_expected.ml)\n (action (progn (run crashing.exe) (diff pre_diff_expected.ml pre_diff.actual))))\n' \
  >"$main/test/dune"
git -C "$main" add fixture benchmarks/fixtures/DIGESTS.txt test/dune \
  test/unit.cc_expected.ml test/noise.expected \
  test/pre_diff_expected.ml test/inline_expect.ml
git -C "$main" commit -qm fixture
fixture_sha=$(git -C "$main" rev-parse HEAD)
git -C "$main" remote add origin "$origin"
git -C "$main" push -q -u origin master

cat >"$fake_bin/opam" <<'EOF'
#!/bin/sh
printf '%s\n' "$*" >>"$SWEEP_TEST_CALLS"
# A serial rerun (gh-ocannl-945) -- the sweep's `-j 1` call for ONE stanza, its
# alias the last argument -- answers on its own: red exactly when that alias is
# listed in SWEEP_TEST_OPAM_SERIAL_RED, with SWEEP_TEST_OPAM_OUT_SERIAL as its
# failure text, so a fixture can hold one stanza red while another clears.
case " $* " in
  *" -j 1 "*)
    # The last positional parameter: `${*##* }` is not it -- pattern removal on
    # `$*` applies to each parameter separately, so it yields the whole line.
    for last; do :; done
    case " ${SWEEP_TEST_OPAM_SERIAL_RED:-} " in
      *" $last "*)
        [ -n "${SWEEP_TEST_OPAM_OUT_SERIAL:-}" ] && printf '%s\n' "$SWEEP_TEST_OPAM_OUT_SERIAL"
        exit 1
        ;;
      *) exit 0 ;;
    esac
    ;;
esac
# Stands in for what a test run writes to the unit's log. The common output
# drives failure-fingerprint coverage; the per-backend outputs let the skip
# aggregation controls distinguish an intersection from a union without GPUs.
[ -n "${SWEEP_TEST_OPAM_OUT:-}" ] && printf '%s\n' "$SWEEP_TEST_OPAM_OUT"
case ${OCANNL_BACKEND:-} in
  cc) [ -n "${SWEEP_TEST_OPAM_OUT_CC:-}" ] && printf '%s\n' "$SWEEP_TEST_OPAM_OUT_CC" ;;
  multidev_cc)
    [ -n "${SWEEP_TEST_OPAM_OUT_MULTIDEV_CC:-}" ] &&
      printf '%s\n' "$SWEEP_TEST_OPAM_OUT_MULTIDEV_CC"
    ;;
  # A backend no aggregation control selects, so that the hermeticity control
  # can hand the nested sweep an AMBIENT backend belonging to neither unit and
  # still be answered. Without an arm here a leaked `metal` produces nothing and
  # the control passes for the wrong reason.
  metal) [ -n "${SWEEP_TEST_OPAM_OUT_METAL:-}" ] && printf '%s\n' "$SWEEP_TEST_OPAM_OUT_METAL" ;;
esac
if [ -n "${SWEEP_TEST_WAIT_PREFIX:-}" ]; then
  : >"$SWEEP_TEST_WAIT_PREFIX.ready"
  waited=0
  while [ ! -e "$SWEEP_TEST_WAIT_PREFIX.release" ]; do
    sleep 0.05
    waited=$((waited + 1))
    [ "$waited" -lt 200 ] || exit 99
  done
fi
exit "${SWEEP_TEST_OPAM_RC:-0}"
EOF
chmod +x "$fake_bin/opam"

# Exercise the exact previous schema: old evidence is retained, but marked
# unknown rather than being upgraded retroactively to executed coverage.
printf 'when\tmachine\tbackend\tref\toutcome\tseconds\ttarget\tslow\tlog\n' >"$state/history.tsv"
printf '20260820T000000Z\tlocal\tcc\tdeadbee\tpass\t1\t<all>\t0\t-\n' >>"$state/history.tsv"

run_sweep_args() {
  # The nested sweep's environment, constructed in full rather than added to.
  # `-u` is the load-bearing half: `tools/sweep.sh` runs a unit's tests as
  # `OCANNL_BACKEND=<backend> opam exec -- dune build ...`, so when the sweep
  # runs this harness the backend it selected is exported into it -- and the
  # nested sweep's own forced-clean leg carries no backend of its own, so the
  # fake opam answered the launcher's and wrote one unit's skip records into
  # another unit's log, reading a union as an intersection (gh-ocannl-893). The
  # sweep's caps go with it: an ambient one would silently rewrite the budgets
  # the cancellation controls below depend on. The hostile-ambient control after
  # the coverage assertions is what keeps this from rotting back.
  #
  # Quoted, unlike the assignment prefix this replaces: these are `env`'s
  # ARGUMENTS now, so the multi-line fixture logs would otherwise be split into
  # words and `env` would try to run one of them as the command.
  env -u OCANNL_BACKEND -u OCANNL_TOOL_SWEEP_CAP -u OCANNL_TOOL_SWEEP_CONTEXT_CAP \
    -u OCANNL_TOOL_SWEEP_LOCAL_BOX \
    "HOME=$tmp/home" \
    "PATH=$fake_bin:$PATH" \
    "SWEEP_TEST_CALLS=$calls" \
    "SWEEP_TEST_WAIT_PREFIX=${SWEEP_TEST_WAIT_PREFIX:-}" \
    "SWEEP_TEST_OPAM_RC=${SWEEP_TEST_OPAM_RC:-0}" \
    "SWEEP_TEST_OPAM_OUT=${SWEEP_TEST_OPAM_OUT:-}" \
    "SWEEP_TEST_OPAM_OUT_CC=${SWEEP_TEST_OPAM_OUT_CC:-}" \
    "SWEEP_TEST_OPAM_OUT_MULTIDEV_CC=${SWEEP_TEST_OPAM_OUT_MULTIDEV_CC:-}" \
    "SWEEP_TEST_OPAM_OUT_METAL=${SWEEP_TEST_OPAM_OUT_METAL:-}" \
    "SWEEP_TEST_OPAM_SERIAL_RED=${SWEEP_TEST_OPAM_SERIAL_RED:-}" \
    "SWEEP_TEST_OPAM_OUT_SERIAL=${SWEEP_TEST_OPAM_OUT_SERIAL:-}" \
    "OCANNL_TOOL_SWEEP_LOCAL_BOX=${SWEEP_TEST_LOCAL_BOX-m4-max}" \
    "OCANNL_TOOL_SWEEP_JOBS=${SWEEP_TEST_JOBS:-}" \
    "OCANNL_TOOL_SWEEP_REPO=$main" \
    "OCANNL_TOOL_SWEEP_STATE=$state" \
    "$sweep" "$@"
}

run_sweep_backend() {
  local backend=$1
  shift
  run_sweep_args --only "$backend" "$@"
}

run_sweep() { run_sweep_backend cc "$@"; }

incremental=$(run_sweep)
forced=$(run_sweep --force)
slow_forced=$(run_sweep --slow --force)

grep -q 'm4-max/cc: incremental-pass .*execution=incremental' <<<"$incremental"
grep -q 'm4-max/cc: pass .*execution=forced' <<<"$forced"

expected_header='when	machine	backend	ref	outcome	seconds	target	slow	log	execution'
[ "$(head -1 "$state/history.tsv")" = "$expected_header" ]
[ "$(awk -F '\t' 'NR == 2 { print $5 ":" $10 }' "$state/history.tsv")" = 'legacy-pass:unknown' ]
[ "$(awk -F '\t' 'NR == 3 { print $5 ":" $10 }' "$state/history.tsv")" = 'incremental-pass:incremental' ]
[ "$(awk -F '\t' 'NR == 4 { print $5 ":" $10 }' "$state/history.tsv")" = 'pass:forced' ]
[ "$(awk -F '\t' 'NR == 5 { print $5 ":" $8 ":" $10 }' "$state/history.tsv")" = 'pass:1:forced' ]

# A full-suite unit builds @runtest and @train together in one dune call
# (test/training/dune says why the tier exists); only a narrow --target run
# still spells `dune runtest <target>`.
[ "$(sed -n '1p' "$calls")" = 'exec -- dune build @runtest @train' ]
[ "$(sed -n '2p' "$calls")" = 'exec -- dune clean' ]
[ "$(sed -n '3p' "$calls")" = 'exec -- dune build --force @runtest @train' ]
[ "$(sed -n '4p' "$calls")" = 'exec -- dune clean' ]
[ "$(sed -n '5p' "$calls")" = 'exec -- dune build --force @runtest @train' ]
[ "$(sed -n '6p' "$calls")" = 'exec -- dune build --force @slow' ]

# Per-unit state distinguishes a standing red from two transitions that need an
# operator's attention: red after green, and red after the failing golden was
# edited. It also compares against the previous FAILURE across an intervening
# green, so a moving fingerprint is reported as nondeterminism rather than
# hidden by yesterday's verdict. Every absent assertion is a negative control:
# a sweep that shouts on the standing-red cases defeats the signal this state
# exists to add.
state_failure='File "test/dune", lines 1-4, characters 0-0:
1 | (rule
2 |  (alias runtest-state-probe)
......
FAILED: fixture state failure.
diff --git a/_build/default/test/unit.cc_expected.ml b/_build/default/test/unit.actual'
state_first=$(SWEEP_TEST_OPAM_RC=1 SWEEP_TEST_OPAM_OUT=$state_failure \
  run_sweep_args --target state-probe)
absent 'REGRESSION OR FIX DID NOT TAKE' <<<"$state_first"
absent 'fingerprint moved since the previous failure' <<<"$state_first"

# A passing diagnostic run of an explicitly requested ref is a separate
# experiment. Without REF in the cursor key it becomes origin/master's green
# predecessor and makes the unchanged standing failure below look regressive.
state_other_ref=$(run_sweep_args --ref "$fixture_sha" --target state-probe)
grep -q 'm4-max/cc: incremental-pass' <<<"$state_other_ref"

state_same=$(SWEEP_TEST_OPAM_RC=1 SWEEP_TEST_OPAM_OUT=$state_failure \
  run_sweep_args --target state-probe)
absent 'REGRESSION OR FIX DID NOT TAKE' <<<"$state_same"
absent 'fingerprint moved since the previous failure' <<<"$state_same"

# A run-then-diff progn that crashes in its producer never ran the diff. Its
# stanza contains a source-controlled expected operand, but without a unified
# diff header that operand is not proven to have failed and must not enter the
# cursor's golden provenance.
pre_diff_failure='File "test/dune", lines 5-8, characters 0-0:
5 | (rule
6 |  (alias runtest-pre-diff-probe)
......
Error: crashing.exe exited 2 before diff'
pre_diff_first=$(SWEEP_TEST_OPAM_RC=1 SWEEP_TEST_OPAM_OUT=$pre_diff_failure \
  run_sweep_args --target pre-diff-probe)
absent 'REGRESSION OR FIX DID NOT TAKE' <<<"$pre_diff_first"
printf 'changed without reaching diff\n' >"$main/test/pre_diff_expected.ml"
git -C "$main" add test/pre_diff_expected.ml
git -C "$main" commit -qm 'change expectation behind crashing producer'
git -C "$main" push -q origin master
pre_diff_second=$(SWEEP_TEST_OPAM_RC=1 SWEEP_TEST_OPAM_OUT=$pre_diff_failure \
  run_sweep_args --target pre-diff-probe)
absent 'REGRESSION OR FIX DID NOT TAKE' <<<"$pre_diff_second"

# Inline ppx_expect promotion compares the checked-in source directly with an
# _build .corrected file. That resolved unified-diff header is proof of the
# failed baseline even though the first operand is neither under _build nor
# named *.expected.
inline_failure='File "test/inline_expect.ml", line 1, characters 0-0:
diff --git a/test/inline_expect.ml b/_build/default/test/inline_expect.ml.corrected'
inline_first=$(SWEEP_TEST_OPAM_RC=1 SWEEP_TEST_OPAM_OUT=$inline_failure \
  run_sweep_args --target inline-expect-probe)
absent 'REGRESSION OR FIX DID NOT TAKE' <<<"$inline_first"
printf 'let%%expect_test _ = print_endline "new" [%%expect {| stale |}]\n' \
  >"$main/test/inline_expect.ml"
git -C "$main" add test/inline_expect.ml
git -C "$main" commit -qm 'attempt inline expectation fix'
git -C "$main" push -q origin master
inline_fix_sha=$(git -C "$main" rev-parse HEAD)
inline_second=$(SWEEP_TEST_OPAM_RC=1 SWEEP_TEST_OPAM_OUT=$inline_failure \
  run_sweep_args --target inline-expect-probe)
grep -q "m4-max/cc: REGRESSION OR FIX DID NOT TAKE -- test/inline_expect.ml last changed at $(printf '%s' "$inline_fix_sha" | cut -c1-8) (previous failing copy: $(printf '%s' "$fixture_sha" | cut -c1-8))" \
  <<<"$inline_second"

# An expected fixture merely listed in the failing stanza's deps is not the
# failed diff input. The old all-token extraction records it and makes this
# unrelated edit look like a failed fix on the next identical red.
printf 'changed unrelated fixture\n' >"$main/test/noise.expected"
git -C "$main" add test/noise.expected
git -C "$main" commit -qm 'change unrelated expected dependency'
git -C "$main" push -q origin master
state_after_noise=$(SWEEP_TEST_OPAM_RC=1 SWEEP_TEST_OPAM_OUT=$state_failure \
  run_sweep_args --target state-probe)
absent 'REGRESSION OR FIX DID NOT TAKE' <<<"$state_after_noise"
absent 'fingerprint moved since the previous failure' <<<"$state_after_noise"

state_green=$(run_sweep_args --target state-probe)
grep -q 'm4-max/cc: incremental-pass' <<<"$state_green"
# A timeout judged nothing and must not erase that green predecessor. This is
# the non-coverage shape that would otherwise make a real regression disappear.
state_unjudged=$(SWEEP_TEST_OPAM_RC=142 SWEEP_TEST_OPAM_OUT='fixture timeout' \
  run_sweep_args --target state-probe)
grep -q 'm4-max/cc: timeout' <<<"$state_unjudged"
state_regression=$(SWEEP_TEST_OPAM_RC=1 SWEEP_TEST_OPAM_OUT=$state_failure \
  run_sweep_args --target state-probe)
grep -q 'm4-max/cc: REGRESSION OR FIX DID NOT TAKE -- previous verdict was incremental-pass' \
  <<<"$state_regression"
absent 'fingerprint moved since the previous failure' <<<"$state_regression"

# Land the exact kind of attempted fix #897 was about. The next sweep resolves
# the new origin/master, finds that the currently failing golden's last-touch
# commit moved, and prints both that commit and the previous failing copy's.
printf 'attempted fix\n' >"$main/test/unit.cc_expected.ml"
git -C "$main" add test/unit.cc_expected.ml
git -C "$main" commit -qm 'attempted golden fix'
git -C "$main" push -q origin master
fix_sha=$(git -C "$main" rev-parse HEAD)
state_after_fix=$(SWEEP_TEST_OPAM_RC=1 SWEEP_TEST_OPAM_OUT=$state_failure \
  run_sweep_args --target state-probe)
grep -q "m4-max/cc: REGRESSION OR FIX DID NOT TAKE -- test/unit.cc_expected.ml last changed at $(printf '%s' "$fix_sha" | cut -c1-8) (previous failing copy: $(printf '%s' "$fixture_sha" | cut -c1-8))" \
  <<<"$state_after_fix"
absent 'fingerprint moved since the previous failure' <<<"$state_after_fix"

moved_failure="$state_failure
Error: a different fixture state failure."
state_moved=$(SWEEP_TEST_OPAM_RC=1 SWEEP_TEST_OPAM_OUT=$moved_failure \
  run_sweep_args --target state-probe)
grep -q 'm4-max/cc: fingerprint moved since the previous failure at ' <<<"$state_moved"
absent 'REGRESSION OR FIX DID NOT TAKE' <<<"$state_moved"

unit_state=$(grep -l "$(printf '^last_verdict\tfail$')" \
  "$state"/unit-state/*state-probe*.state | head -1)
[ -n "$unit_state" ] && [ -f "$unit_state" ]
grep -q '^last_verdict.fail$' "$unit_state"
grep -q "^golden.$fix_sha.test/unit.cc_expected.ml$" "$unit_state"

# Two complete forced units expose only the INTERSECTION of their skip sets.
# The three absent backends keep this a potential finding rather than a failure:
# one of them may have evaluated the common claim. A per-backend-only marker
# must not leak into the report merely because it occurred somewhere. A skip
# whose gate belongs to the environment occurs in both logs too. These two
# backends share one box, so they prove no cross-box fact and environment
# aggregation stays explicitly insufficient.
common=$'SKIPPED on fixture (vacuous): common unevaluated claim\nOCANNL_TOOL_VERDICT_SKIP\tbackend\tfixture.exe\tcommon unevaluated claim'
cc_only=$'SKIPPED on fixture (vacuous): cc-only unevaluated claim\nOCANNL_TOOL_VERDICT_SKIP\tbackend\tfixture.exe\tcc-only unevaluated claim'
multidev_only=$'SKIPPED on fixture (vacuous): multidev-only unevaluated claim\nOCANNL_TOOL_VERDICT_SKIP\tbackend\tfixture.exe\tmultidev-only unevaluated claim'
environment=$'SKIPPED on fixture gate (vacuous): environment-gated claim\nOCANNL_TOOL_VERDICT_SKIP\tenvironment\tfixture.exe\tenvironment-gated claim'
outside=$'SKIPPED on external matrix (vacuous): independently-covered claim\nOCANNL_TOOL_VERDICT_SKIP\toutside-sweep\tfixture.exe\tindependently-covered claim'
cc_unit_log=$common$'\n'$cc_only$'\n'$environment$'\n'$outside
multidev_unit_log=$common$'\n'$multidev_only$'\n'$environment$'\n'$outside
coverage=$(SWEEP_TEST_OPAM_OUT_CC=$cc_unit_log \
  SWEEP_TEST_OPAM_OUT_MULTIDEV_CC=$multidev_unit_log \
  run_sweep_args --force --only cc --only multidev_cc)
coverage_report=$(sed -n 's/^skip coverage: .* -- //p' <<<"$coverage" | tail -1)
[ -f "$coverage_report" ]
grep -q '^status: partial (2 of 5 known backends completed)$' "$coverage_report"
grep -q '^missing backends: metal, cuda, hip$' "$coverage_report"
grep -q '^POTENTIAL: skipped on every completed backend: fixture.exe: common unevaluated claim$' \
  "$coverage_report"
absent 'cc-only unevaluated claim' "$coverage_report"
absent 'multidev-only unevaluated claim' "$coverage_report"
absent 'environment-gated claim' "$coverage_report"
absent 'independently-covered claim' "$coverage_report"
grep -q '^completed boxes: m4-max$' "$coverage_report"
grep -q '^missing boxes: minix, rog-nv$' "$coverage_report"
grep -q '^environment status: insufficient (1 of 3 declared boxes completed; need at least 2 unless the matrix is complete)$' \
  "$coverage_report"
grep -q '^environment result: NOT AGGREGATED$' "$coverage_report"

# The verdict and each finding must reach the sweep's OWN summary, not only the
# report file: the scheduled routine's notification path quotes sweep output,
# and a zero-coverage claim that lives only behind the report path is one no
# human reads (gh-ocannl-792). Indented, so the `skip coverage:` pointer line
# stays the one line the path is extracted from -- which the extraction above
# already proved. Per-backend-only and environment-gated claims must not reach
# the summary either, for the same reason they stay out of the report.
grep -q '^  result: POTENTIAL -- 1 claim(s) skipped on every completed backend; absent backends remain unknown$' \
  <<<"$coverage"
grep -q '^  POTENTIAL: skipped on every completed backend: fixture.exe: common unevaluated claim$' \
  <<<"$coverage"
absent 'cc-only unevaluated claim' <<<"$coverage"
absent 'multidev-only unevaluated claim' <<<"$coverage"
absent 'environment-gated claim' <<<"$coverage"
absent 'independently-covered claim' <<<"$coverage"

# The same fixture with a hostile backend in the AMBIENT environment. This is
# the harness's own running condition: the sweep runs a unit's tests as
# `OCANNL_BACKEND=<backend> opam exec -- dune build ...`, so the launcher's
# choice of backend is exported into every test action of that unit, this one
# included. It used to reach the nested sweep, whose forced-clean leg names no
# backend of its own -- so the fake opam answered the AMBIENT one and wrote the
# cc unit's skip records into the multidev_cc unit's log, turning the
# intersection this exists to take into a union (gh-ocannl-893). The ambient
# value names a backend NEITHER selected unit runs, carrying a claim of its own:
# a leak then appears in both units' logs and grows the intersection, so what is
# pinned is that the nested sweep answers only the backends the SWEEP selected,
# not merely that these two particular ones survive.
#
# The aggregation is compared rather than restated: the clean run's findings are
# already pinned to their exact text above, and their extraction is an errexit
# assignment that fails loudly on no match -- so an empty pair cannot agree
# vacuously, which is the failure mode a comparison invites.
leaked=$'SKIPPED on fixture (vacuous): leaked-ambient claim\nOCANNL_TOOL_VERDICT_SKIP\tbackend\tfixture.exe\tleaked-ambient claim'
coverage_findings=$(grep -E '^  (result|FAIL|POTENTIAL): ' <<<"$coverage")
hostile=$(OCANNL_BACKEND=metal \
  SWEEP_TEST_OPAM_OUT_CC=$cc_unit_log \
  SWEEP_TEST_OPAM_OUT_MULTIDEV_CC=$multidev_unit_log \
  SWEEP_TEST_OPAM_OUT_METAL=$leaked \
  run_sweep_args --force --only cc --only multidev_cc)
[ "$(grep -E '^  (result|FAIL|POTENTIAL): ' <<<"$hostile")" = "$coverage_findings" ]

# A single-backend forced run cannot aggregate, and its summary says so through
# the same channel rather than staying silent about the report it wrote.
grep -q '^  result: NOT AGGREGATED$' <<<"$forced"

# The pure aggregator's complete-matrix control is the escalation seam the real
# sweep reaches only when both remote GPU boxes and all local units pass. All
# five logs sharing both claims is exit 1 with backend and environment FAIL
# lines. The environment matrix comes from DIGESTS in the nested sweep above;
# these arguments pin the aggregator's independent contract.
aggregate_args=()
for box in m4-max minix rog-nv; do aggregate_args+=(--known-box "$box"); done
for backend in cc multidev_cc metal cuda hip; do
  log=$tmp/$backend.log
  "$verdict_probe" "$backend" >"$log" 2>&1
  aggregate_args+=(--known "$backend")
  case $backend in
    cc | multidev_cc | metal) box=m4-max ;;
    cuda) box=rog-nv ;;
    hip) box=minix ;;
  esac
  aggregate_args+=(--run "$backend" "$box" "$log")
done
set +e
complete_fail=$("$aggregate" "${aggregate_args[@]}" 2>&1)
complete_fail_rc=$?
set -e
[ "$complete_fail_rc" -eq 1 ]
grep -q '^status: complete (5 of 5 known backends completed)$' <<<"$complete_fail"
grep -q '^FAIL: skipped on every known backend: verdict_skip_probe.exe: common unevaluated claim$' \
  <<<"$complete_fail"
grep -q '^environment status: complete (3 of 3 declared boxes completed)$' <<<"$complete_fail"
grep -q '^environment result: FAIL -- 1 claim(s) skipped on every declared box$' \
  <<<"$complete_fail"
grep -q '^FAIL: skipped on every declared box: verdict_skip_probe.exe: common environment-gated claim$' \
  <<<"$complete_fail"

# Executing the environment-gated leg in ONE complete log removes it from the
# intersection even though the same box/backend claim remains skipped. The
# aggregator still exits 1 for that independent backend failure; its environment
# result must pass and must not carry the all-box finding.
"$verdict_probe" hip execute-environment >"$tmp/hip.log" 2>&1
set +e
environment_executed=$("$aggregate" "${aggregate_args[@]}" 2>&1)
environment_executed_rc=$?
set -e
[ "$environment_executed_rc" -eq 1 ]
grep -q '^result: FAIL -- 1 claim(s) skipped on every known backend$' \
  <<<"$environment_executed"
grep -q '^environment result: PASS -- no claim was skipped on every declared box$' \
  <<<"$environment_executed"
absent 'FAIL: skipped on every declared box:' <<<"$environment_executed"

printf 'this backend evaluated the common claim\n' >"$tmp/hip.log"
complete_pass=$("$aggregate" "${aggregate_args[@]}")
grep -q '^result: PASS -- no claim was skipped on every known backend$' <<<"$complete_pass"
grep -q '^environment result: PASS -- no claim was skipped on every declared box$' \
  <<<"$complete_pass"

# Two of three boxes make an all-observed environment skip POTENTIAL, never a
# FAIL: the absent box may execute it. This also proves completeness is counted
# by distinct box rather than by the number of logs (m4-max contributes three
# in the complete case above).
"$verdict_probe" cc >"$tmp/cc.log" 2>&1
"$verdict_probe" hip >"$tmp/hip.log" 2>&1
partial_matrix=$("$aggregate" \
  --known cc --known multidev_cc --known metal --known cuda --known hip \
  --known-box m4-max --known-box minix --known-box rog-nv \
  --run cc m4-max "$tmp/cc.log" --run hip minix "$tmp/hip.log")
grep -q '^environment status: partial (2 of 3 declared boxes completed)$' <<<"$partial_matrix"
grep -q '^environment result: POTENTIAL -- 1 claim(s) skipped on every completed box; absent boxes remain unknown$' \
  <<<"$partial_matrix"
grep -q '^POTENTIAL: skipped on every completed box: verdict_skip_probe.exe: common environment-gated claim$' \
  <<<"$partial_matrix"

# One logical leg can have different reasons for being unevaluated: this is the
# real autotune_mma_companion shape, backend-scoped on cc/multidev_cc and
# configuration-scoped on default-config CUDA/HIP. With Metal red and therefore
# absent, every declared box is represented but no successful unit executed the
# claim. Filtering by scope first falsely reported PASS; claim-and-box evidence
# must report the complete environment matrix as FAIL.
"$verdict_probe" cc environment-as-backend >"$tmp/mixed-cc.log" 2>&1
"$verdict_probe" multidev_cc environment-as-backend >"$tmp/mixed-multidev.log" 2>&1
"$verdict_probe" cuda >"$tmp/mixed-cuda.log" 2>&1
"$verdict_probe" hip >"$tmp/mixed-hip.log" 2>&1
set +e
mixed_scope_fail=$("$aggregate" \
  --known cc --known multidev_cc --known metal --known cuda --known hip \
  --known-box m4-max --known-box minix --known-box rog-nv \
  --run cc m4-max "$tmp/mixed-cc.log" \
  --run multidev_cc m4-max "$tmp/mixed-multidev.log" \
  --run cuda rog-nv "$tmp/mixed-cuda.log" --run hip minix "$tmp/mixed-hip.log" 2>&1)
mixed_scope_fail_rc=$?
set -e
[ "$mixed_scope_fail_rc" -eq 1 ]
grep -q '^environment result: FAIL -- 1 claim(s) skipped on every declared box$' \
  <<<"$mixed_scope_fail"
grep -q '^FAIL: skipped on every declared box: verdict_skip_probe.exe: common environment-gated claim$' \
  <<<"$mixed_scope_fail"

# The successful Metal leg is the execution that must clear the same mixed-scope
# claim. Its other backend-scoped fixture claim remains independent.
"$verdict_probe" metal execute-environment >"$tmp/mixed-metal.log" 2>&1
set +e
mixed_scope_cleared=$("$aggregate" \
  --known cc --known multidev_cc --known metal --known cuda --known hip \
  --known-box m4-max --known-box minix --known-box rog-nv \
  --run cc m4-max "$tmp/mixed-cc.log" \
  --run multidev_cc m4-max "$tmp/mixed-multidev.log" \
  --run metal m4-max "$tmp/mixed-metal.log" \
  --run cuda rog-nv "$tmp/mixed-cuda.log" --run hip minix "$tmp/mixed-hip.log" 2>&1)
mixed_scope_cleared_rc=$?
set -e
[ "$mixed_scope_cleared_rc" -eq 1 ]
grep -q '^environment result: PASS -- no claim was skipped on every declared box$' \
  <<<"$mixed_scope_cleared"
absent 'FAIL: skipped on every declared box:' <<<"$mixed_scope_cleared"

# A backend may run on more than one declared box. It counts once toward backend
# completeness, but each box remains independent environment evidence.
"$verdict_probe" cc >"$tmp/repeated-m4.log" 2>&1
"$verdict_probe" cc >"$tmp/repeated-minix.log" 2>&1
set +e
repeated_backend_fail=$("$aggregate" \
  --known cc --known metal --known-box m4-max --known-box minix \
  --run cc m4-max "$tmp/repeated-m4.log" --run cc minix "$tmp/repeated-minix.log" 2>&1)
repeated_backend_fail_rc=$?
set -e
[ "$repeated_backend_fail_rc" -eq 1 ]
grep -q '^completed backends: cc$' <<<"$repeated_backend_fail"
grep -q '^status: insufficient (1 of 2 known backends completed; need at least 2)$' \
  <<<"$repeated_backend_fail"
grep -q '^environment status: complete (2 of 2 declared boxes completed)$' \
  <<<"$repeated_backend_fail"
grep -q '^environment result: FAIL -- 1 claim(s) skipped on every declared box$' \
  <<<"$repeated_backend_fail"
"$verdict_probe" cc execute-environment >"$tmp/repeated-minix.log" 2>&1
repeated_backend_pass=$("$aggregate" \
  --known cc --known metal --known-box m4-max --known-box minix \
  --run cc m4-max "$tmp/repeated-m4.log" --run cc minix "$tmp/repeated-minix.log")
grep -q '^environment result: PASS -- no claim was skipped on every declared box$' \
  <<<"$repeated_backend_pass"

# Completeness outranks the partial-matrix observation floor. A valid singleton
# declaration must still turn its one box's skip into FAIL, and execution in
# that same one log must turn it into PASS.
"$verdict_probe" cc >"$tmp/cc.log" 2>&1
set +e
singleton_fail=$("$aggregate" \
  --known cc --known metal --known-box m4-max \
  --run cc m4-max "$tmp/cc.log" 2>&1)
singleton_fail_rc=$?
set -e
[ "$singleton_fail_rc" -eq 1 ]
grep -q '^environment status: complete (1 of 1 declared boxes completed)$' <<<"$singleton_fail"
grep -q '^FAIL: skipped on every declared box: verdict_skip_probe.exe: common environment-gated claim$' \
  <<<"$singleton_fail"
"$verdict_probe" cc execute-environment >"$tmp/cc.log" 2>&1
singleton_pass=$("$aggregate" \
  --known cc --known metal --known-box m4-max \
  --run cc m4-max "$tmp/cc.log")
grep -q '^environment result: PASS -- no claim was skipped on every declared box$' \
  <<<"$singleton_pass"

# Equal human labels in two DIFFERENT executables are different test legs. Copy
# the real probe under another basename so this control reaches the production
# identity emission rather than restating its record format in the fixture.
other_probe=$tmp/other_skip_probe.exe
cp "$verdict_probe" "$other_probe"
"$verdict_probe" cc >"$tmp/identity-cc.log" 2>&1
"$other_probe" metal >"$tmp/identity-metal.log" 2>&1
identity_clear=$("$aggregate" \
  --known cc --known metal \
  --known-box m4-max \
  --run cc m4-max "$tmp/identity-cc.log" --run metal m4-max "$tmp/identity-metal.log")
grep -q '^result: PASS -- no claim was skipped on every known backend$' <<<"$identity_clear"

# A historical target from before the declaration keeps its backend answer and
# says that environment aggregation is unavailable. Treating observed row/log
# origins as a declared matrix would turn this into invented completeness.
legacy_matrix=$("$aggregate" \
  --known cc --known metal \
  --run cc m4-max "$tmp/identity-cc.log" --run metal m4-max "$tmp/identity-metal.log")
grep -q '^result: PASS -- no claim was skipped on every known backend$' <<<"$legacy_matrix"
grep -q '^environment status: unavailable (target declares no measurement-box matrix)$' \
  <<<"$legacy_matrix"
grep -q '^environment result: NOT AGGREGATED$' <<<"$legacy_matrix"

# Zero successful units is routine when every selected backend is unavailable
# or red. This runs under macOS's stock Bash 3.2 in the local suite and pins the
# nounset-safe branch before any empty-array expansion.
empty=$("$aggregate" --known cc --known metal)
grep -q '^completed backends: <none>$' <<<"$empty"
grep -q '^result: NOT AGGREGATED$' <<<"$empty"

# Evidence-processing errors are harness failures (exit 2), never an empty set
# that can read as CLEAR/PASS. Fault-inject sort, the last command of the
# extraction pipeline, so pipefail must reach the explicit error conversion.
fail_bin=$tmp/fail-bin
mkdir -p "$fail_bin"
cat >"$fail_bin/sort" <<'EOF'
#!/bin/sh
exit 7
EOF
chmod +x "$fail_bin/sort"
set +e
extract_error=$(PATH=$fail_bin:$PATH "$aggregate" \
  --known cc --known metal \
  --known-box m4-max \
  --run cc m4-max "$tmp/identity-cc.log" --run metal m4-max "$tmp/identity-metal.log" 2>&1)
extract_error_rc=$?
set -e
[ "$extract_error_rc" -eq 2 ]
grep -q '^aggregate-skips: cannot extract compatible skip records from ' <<<"$extract_error"

# A supported `sweep.sh --ref` may target a commit from before Verdict emitted
# machine records. Its legacy human line is evidence of a skip, not evidence of
# execution; a human/machine count mismatch must make the whole log incompatible.
printf 'SKIPPED on cc (vacuous): common unevaluated claim\n' >"$tmp/legacy-cc.log"
set +e
legacy_error=$("$aggregate" \
  --known cc --known metal \
  --known-box m4-max \
  --run cc m4-max "$tmp/legacy-cc.log" --run metal m4-max "$tmp/identity-metal.log" 2>&1)
legacy_error_rc=$?
set -e
[ "$legacy_error_rc" -eq 2 ]
grep -q '^aggregate-skips: cannot extract compatible skip records from ' <<<"$legacy_error"

# A successful analysis whose destination stops accepting bytes is still a
# harness failure. A read-only descriptor makes the first report write fail
# deterministically without relying on a device Dune's sandbox may deny; the
# explicit success exit below it must not erase that error.
: >"$tmp/read-only-report"
exec 8<"$tmp/read-only-report"
set +e
"$aggregate" "${aggregate_args[@]}" >&8 2>"$tmp/report-write.err"
report_write_rc=$?
set -e
exec 8<&-
[ "$report_write_rc" -eq 2 ]
grep -q '^aggregate-skips: cannot write report$' "$tmp/report-write.err"

# Hold one run after it owns the worktree lock, then replace its history with
# the old schema. A competing launch must refuse at the lock without migrating
# that file; this deterministically pins migration behind serialization.
wait_prefix=$tmp/migration-lock
SWEEP_TEST_WAIT_PREFIX=$wait_prefix run_sweep >"$tmp/holder.out" 2>"$tmp/holder.err" &
holder_pid=$!
for _ in {1..200}; do
  [ -e "$wait_prefix.ready" ] && break
  sleep 0.05
done
[ -e "$wait_prefix.ready" ]
printf 'when\tmachine\tbackend\tref\toutcome\tseconds\ttarget\tslow\tlog\n' >"$state/history.tsv"
printf '20260820T000000Z\tlocal\tcc\tdeadbee\tpass\t1\t<all>\t0\t-\n' >>"$state/history.tsv"
set +e
SWEEP_TEST_WAIT_PREFIX= run_sweep >"$tmp/competitor.out" 2>"$tmp/competitor.err"
competitor_rc=$?
set -e
[ "$competitor_rc" -eq 2 ]
[ "$(head -1 "$state/history.tsv")" = "$(printf 'when\tmachine\tbackend\tref\toutcome\tseconds\ttarget\tslow\tlog')" ]
touch "$wait_prefix.release"
wait "$holder_pid"
holder_pid=

# A red GPU unit must carry its RTC context (gh-ocannl-784): the flags the
# kernels were compiled under and which toolkit did it, beside the failure rather
# than nowhere. `metal` is the case this can pin without hardware and without a
# reachable remote -- it is a LOCAL unit, and its context block is built only from
# `command -v` guards plus the pure option-builder alias, so on a machine with no
# macOS tooling it still prints the production property sequence and proves the
# block was emitted, reached the log, and was carried into the fingerprint. The
# cuda and hip arms differ from it only in which discovery commands they guard.
#
# A red unit, not a green one: this is diagnosis, and emitting it on a pass would
# run dune a second time on every sweep.
# The failure text the fake dune writes. Its remaining lines are the shapes
# `cuda_to_ptx` and `hip_to_code` append to their compiler's message when a
# compile fails, and `compile_metal_source` appends to a Metal failure -- as
# opposed to the pure policy vectors the context block prints. `fingerprint` is
# backend-blind, so the local metal unit pins all three lines' extraction;
# producing them is separately covered on their hardware boxes. All three
# vectors come from their production renderers through the OCaml driver, so
# this fixture cannot drift from any source of truth (gh-ocannl-881,
# gh-ocannl-849); the nvrtc one carries the driver's sentinel include and
# architecture slots, which is what makes its line unmistakable below.
case $rendered_nvrtc_options in
  *sentinel*) ;;
  *)
    printf 'sweep_harness: nvrtc vector lost its sentinel: %s\n' "$rendered_nvrtc_options" >&2
    false
    ;;
esac
# The metal vector has no sentinel to lose: this arm pins the production
# property sequence itself, so there is nothing artificial in it to look for.
# What it must not silently become is EMPTY -- the whole-line check below would
# then hold against a bare `metal options: ` prefix, and its controls with it.
case $rendered_metal_options in
  '')
    printf 'sweep_harness: metal vector rendered empty\n' >&2
    false
    ;;
esac
rtc_failure='Fatal error: exception nvrtc_compile_program k.cu: nvrtc: error: no
nvrtc options: '"$rendered_nvrtc_options"
rtc_failure=$rtc_failure$'\nhiprtc options: '"$rendered_hip_options"
rtc_failure=$rtc_failure$'\nmetal options: '"$rendered_metal_options"
SWEEP_TEST_OPAM_RC=1 SWEEP_TEST_OPAM_OUT=$rtc_failure \
  run_sweep_backend metal >"$tmp/metal.out" 2>&1
grep -q 'm4-max/metal: fail' "$tmp/metal.out"
# The recorded outcome is the column the collection must not be able to corrupt.
# Folded into the unit, the diagnostic shares the unit's CAP, and a red suite that
# ran the cap down would be filed as `timeout` -- coverage lost -- instead of as
# the failure it was (Codex P2 on PR #510). The structural guarantee is that
# collection happens strictly after `record`; this pins the column it protects.
[ "$(awk -F '\t' '$3 == "metal" { print $5 }' "$state/history.tsv" | tail -1)" = fail ]
metal_log=$(awk -F '\t' '$3 == "metal" { print $9 }' "$state/history.tsv" | tail -1)
[ -n "$metal_log" ] && [ -f "$metal_log" ]
grep -q '^=== rtc-context (metal) ===$' "$metal_log"
grep -q '^=== end rtc-context ===$' "$metal_log"
grep -q 'rtc option policy from arrayjit/test/runtest-test_metal_compile_options' "$metal_log"
# The fingerprint is what a caller diffs against yesterday's, so the block has to
# reach it and not merely the log.
grep -q '^=== rtc-context (metal) ===$' "${metal_log%.log}.fingerprint"
# And so does the effective vector of a failed compile, which reaches the log as
# an ordinary line of the exception message: it begins neither at an error site
# nor at `Error`/`Fatal error`/`Exception`, so before its own selector existed it
# stopped at the log and never reached the file callers diff.
grep -Fxq "nvrtc options: $rendered_nvrtc_options" "${metal_log%.log}.fingerprint"
# Negative control for that extraction: the vector is required WHOLE, as one
# line. A selector that stopped at the first space, or a fingerprint that held
# the line under another prefix, would still satisfy a substring match; both
# the truncated vector and a deliberately altered one must fail the whole-line
# check the positive assertion relies on.
truncated_nvrtc_options=${rendered_nvrtc_options% *}
[ "$truncated_nvrtc_options" != "$rendered_nvrtc_options" ]
absent -Fx "nvrtc options: $truncated_nvrtc_options" "${metal_log%.log}.fingerprint"
altered_nvrtc_options=${rendered_nvrtc_options/sentinel/altered}
[ "$altered_nvrtc_options" != "$rendered_nvrtc_options" ]
absent -Fx "nvrtc options: $altered_nvrtc_options" "${metal_log%.log}.fingerprint"
grep -Fxq "hiprtc options: $rendered_hip_options" "${metal_log%.log}.fingerprint"
grep -Fxq "metal options: $rendered_metal_options" "${metal_log%.log}.fingerprint"
# And the same whole-line controls, for the same reason. With no sentinel to
# corrupt, the alteration rewrites the first property's VALUE -- a spelling no
# renderer output can produce -- rather than a slot the fixture invented.
truncated_metal_options=${rendered_metal_options% *}
[ "$truncated_metal_options" != "$rendered_metal_options" ]
absent -Fx "metal options: $truncated_metal_options" "${metal_log%.log}.fingerprint"
altered_metal_options=${rendered_metal_options/=/=altered-}
[ "$altered_metal_options" != "$rendered_metal_options" ]
absent -Fx "metal options: $altered_metal_options" "${metal_log%.log}.fingerprint"

# And a GREEN unit must NOT pay for it -- the same backend, so the only thing
# that differs is the outcome. The log path is derived from the sweep's timestamp
# and may well be the one above, rewritten: that is fine and is itself part of the
# check, since the assertions on the failing run have already read it.
run_sweep_backend metal >"$tmp/metal_pass.out" 2>&1
grep -q 'm4-max/metal: incremental-pass' "$tmp/metal_pass.out"
metal_pass_log=$(awk -F '\t' '$3 == "metal" { print $9 }' "$state/history.tsv" | tail -1)
[ -f "$metal_pass_log" ]
absent 'rtc-context' "$metal_pass_log"

# Both of dune's location spellings must reach the fingerprint, and a dune
# location must reduce to the stanza it names. `lines N-M` is what a stanza
# whose action exited non-zero produces -- how every explicit-rule test in this
# repository fails -- and matching only the singular `line N` left such a unit
# with an EMPTY fingerprint, which compares equal to any other empty one: the
# consumer that diffs against the previous non-pass run filed a red suite as
# unchanged and said nothing. The excerpt below is a real dune stanza-error
# shape, elision marker included.
#
# The identifier is NOT reliably a bare word on its keyword's line, so the
# shapes this repository's dune files actually use are all present below: bare,
# quoted, wrapped so the keyword ends one line and its value begins the next,
# and nested as `(alias (name x))`. Reading only the same-line bare form leaves
# the others falling back to the shifting span -- the very thing this
# normalization exists to avoid -- while looking like it works, because the
# fallback is silent.
dune_failure='File "test/operations/dune", lines 4683-4700, characters 0-533:
4683 | (rule
4684 |  ; ocannl-backend: none -- a comment, not a name
4685 |  (alias runtest-fixture_stanza)
......
4700 |    %{dep:fixture.exe})))
File "test/operations/dune", lines 273-280, characters 0-100:
 273 | (rule
 274 |  (target "backend-0-0.log.actual")
 275 |  (package neural_nets_lib)
File "test/operations/dune", lines 669-676, characters 0-100:
 669 | (rule
 670 |  (targets
 671 |   zero_out_local_decl-unoptimized.ll.actual
 672 |   zero_out_local_decl.extension.actual)
File "test/operations/dune", lines 72-80, characters 0-100:
  72 | (alias
  73 |  (name slow)
  74 |  (deps
File "test/operations/dune", lines 990-999, characters 0-100:
 990 | (rule
......
 999 |    %{dep:whatever.exe})))
File "test/operations/fixture.expected", line 1, characters 0-0:
FAILED: 1 check did not hold.'
SWEEP_TEST_OPAM_RC=1 SWEEP_TEST_OPAM_OUT=$dune_failure \
  run_sweep >"$tmp/dune_fail.out" 2>&1
grep -q 'm4-max/cc: fail' "$tmp/dune_fail.out"
dune_fail_fp=$(awk -F '\t' '$3 == "cc" { print $9 }' "$state/history.tsv" | tail -1)
dune_fail_fp=${dune_fail_fp%.log}.fingerprint
# The stanza name, not the span: line numbers in a dune file shift under any
# edit to that file, so a fingerprint keyed on them reports wholesale change
# when an unrelated stanza is inserted above -- overstating the very thing the
# diff is asked to measure.
grep -q '^File "test/operations/dune", alias runtest-fixture_stanza$' "$dune_fail_fp"
absent '4683' "$dune_fail_fp"
# A comment preceding the stanza field must not be mistaken for its name.
absent 'ocannl-backend' "$dune_fail_fp"
# A quoted identifier, and one dune wrapped onto the line after its keyword.
grep -q '^File "test/operations/dune", target "backend-0-0.log.actual"$' "$dune_fail_fp"
grep -q '^File "test/operations/dune", targets zero_out_local_decl-unoptimized.ll.actual$' \
  "$dune_fail_fp"
# `(alias (name x))` is named by the nested field, not by the outer keyword:
# a keyword left pending must be abandoned when the value turns out to be a form.
grep -q '^File "test/operations/dune", name slow$' "$dune_fail_fp"
absent 'alias name' "$dune_fail_fp"
# The one honest fallback: dune elided everything identifying, so the span is
# all there is. Silence here would be a stanza mis-attributed to a neighbour.
grep -q '^File "test/operations/dune", lines 990-999$' "$dune_fail_fp"
# A non-dune location keeps its line number, which is stable and is what a
# reader needs there.
grep -q '^File "test/operations/fixture.expected", line 1$' "$dune_fail_fp"

# A non-pass whose log yields nothing extractable is its own condition, not a
# fingerprint of zero failures. Left empty it compares equal to the previous
# empty one, so the diffing consumer reports no change; the sentinel makes the
# file differ from a real fingerprint in either direction, and the summary
# carries it to the human, the channel the scheduled routine actually quotes.
SWEEP_TEST_OPAM_RC=1 SWEEP_TEST_OPAM_OUT='a red suite that named no error site' \
  run_sweep >"$tmp/blank_fail.out" 2>&1
grep -q 'm4-max/cc: fail' "$tmp/blank_fail.out"
grep -q '^  m4-max/cc: (no fingerprintable diagnostics -- read the log) -- ' \
  "$tmp/blank_fail.out"
blank_fail_fp=$(awk -F '\t' '$3 == "cc" { print $9 }' "$state/history.tsv" | tail -1)
blank_fail_fp=${blank_fail_fp%.log}.fingerprint
[ -s "$blank_fail_fp" ]
grep -q '^(no fingerprintable diagnostics -- read the log)$' "$blank_fail_fp"

# The launcher must bind the physical local host to its stable fleet ID. With
# that identity absent the sweep refuses before it can write a mislabeled row;
# a hard-coded m4-max local unit would make this control pass incorrectly.
set +e
local_identity_error=$(SWEEP_TEST_LOCAL_BOX= run_sweep 2>&1)
local_identity_error_rc=$?
set -e
[ "$local_identity_error_rc" -eq 2 ]
grep -q "^sweep: set OCANNL_TOOL_SWEEP_LOCAL_BOX to this host's portable measurement-box ID$" \
  <<<"$local_identity_error"

# A path separator in the local ID must be rejected before it becomes part of a
# per-unit log path. The DIGESTS parser applies the same portable-ID grammar to
# declared and recorded origins.
set +e
unsafe_identity_error=$(SWEEP_TEST_LOCAL_BOX='m4/max' run_sweep 2>&1)
unsafe_identity_error_rc=$?
set -e
[ "$unsafe_identity_error_rc" -eq 2 ]
grep -q "^sweep: set OCANNL_TOOL_SWEEP_LOCAL_BOX to this host's portable measurement-box ID$" \
  <<<"$unsafe_identity_error"

# An --only typo is refused, with the backends that would have been accepted:
# a selector that matches nothing would otherwise record nothing and exit 0
# (sweep.sh says so at the check). Every positive run of this harness goes
# through `--only cc`, the first backend the table lists -- the exact value the
# check once refused on the ubuntu CI leg (gh-ocannl-949) -- so this is the
# negative half of that control, not a restatement of it.
set +e
only_typo_error=$(run_sweep_backend cudaa 2>&1)
only_typo_error_rc=$?
set -e
[ "$only_typo_error_rc" -eq 2 ]
grep -q "^sweep: unknown backend 'cudaa'; known: cc multidev_cc metal cuda hip" <<<"$only_typo_error"

# A unit with a dune job cap -- the per-unit table names minix/hip, a unit this
# harness cannot reach, so the run-wide override stands in -- compiles at full
# width under `@check` and only then runs its tests under `-j`. The compile
# leg's status is dropped on purpose (test_cmd says why), so the verdict must
# still come from the capped call, and the cap must not reach the uncapped
# units: the run before this one recorded no `@check` and no `-j`.
capped=$(SWEEP_TEST_JOBS=2 run_sweep --force)
grep -q 'm4-max/cc: pass .*execution=forced' <<<"$capped"
[ "$(tail -3 "$calls" | sed -n '1p')" = 'exec -- dune clean' ]
[ "$(tail -3 "$calls" | sed -n '2p')" = 'exec -- dune build @check' ]
[ "$(tail -3 "$calls" | sed -n '3p')" = 'exec -- dune build -j 2 --force @runtest @train' ]
absent 'dune build \(-j\|@check\)' <<<"$(tail -4 "$calls" | sed -n '1p')"
# A capped --target run keeps its narrow meaning: the cap reaches the test call,
# and no workspace-wide `@check` runs ahead of it to eat the unit's deadline on
# code the target never reaches.
capped_target=$(SWEEP_TEST_JOBS=2 run_sweep_args --target state-probe)
[ "$(tail -1 "$calls")" = 'exec -- dune runtest -j 2 state-probe' ]
absent '@check' <<<"$(tail -2 "$calls" | sed -n '1p')"

# An environment-red unit -- a red whose log carries a runtime-refusal signature
# from sweep.sh's ENVIRONMENT_REFUSALS table -- reruns its failing stanzas one at
# a time at `-j 1` and records which stayed red (gh-ocannl-945). The fixture is
# the 2026-09-05 minix/hip shape: one stanza refused at hip_init, one that
# crashed after (no signature of its own, still a red stanza), and an inline
# expectation located in a source file, which names no stanza and must be
# reported unmapped rather than approximated by a directory-wide alias. The
# fake opam holds the first stanza red on its own and clears the second. One
# unit (`--only cc`): every local unit would rerun the same fixture, and the
# remote ones would reach for ssh.
environment_failure='File "test/dune", line 2, characters 7-28:
2 |  (alias runtest-serial-probe)
Fatal error: exception hip_init:
HIP_ERROR_INVALID_DEVICE
File "test/dune", lines 5-8, characters 0-0:
5 | (rule
6 |  (alias runtest-pre-diff-probe)
......
Command got signal SEGV.
File "test/inline_expect.ml", line 1, characters 0-0:
Error: inline expectation differs'
serial_red=$(SWEEP_TEST_OPAM_RC=1 SWEEP_TEST_OPAM_OUT=$environment_failure \
  SWEEP_TEST_OPAM_SERIAL_RED='@test/runtest-serial-probe' \
  SWEEP_TEST_OPAM_OUT_SERIAL='File "test/dune", line 2, characters 7-28:
2 |  (alias runtest-serial-probe)
Error: the claim itself' run_sweep_backend cc --target serial-probe)
grep -q 'm4-max/cc: fail ' <<<"$serial_red"
# One dune call per stanza, so each has its own verdict; sorted, after the unit.
[ "$(tail -3 "$calls" | sed -n '1p')" = 'exec -- dune runtest serial-probe' ]
[ "$(tail -3 "$calls" | sed -n '2p')" = 'exec -- dune build -j 1 @test/runtest-pre-diff-probe' ]
[ "$(tail -3 "$calls" | sed -n '3p')" = 'exec -- dune build -j 1 @test/runtest-serial-probe' ]
# The verdict reaches all three channels: the summary, the log, the fingerprint.
grep -q 'm4-max/cc: environment-red, 2 stanzas rerun at -j 1' <<<"$serial_red"
grep -q 'm4-max/cc: serial rerun: still red: @test/runtest-serial-probe$' <<<"$serial_red"
serial_log=$(awk -F '\t' '$3 == "cc" { print $9 }' "$state/history.tsv" | tail -1)
grep -q '^serial rerun: still red: @test/runtest-serial-probe$' "$serial_log"
grep -q '^serial rerun: unmapped: \[File "test/inline_expect.ml", line 1\]$' "$serial_log"
absent '^serial rerun: all clean' "$serial_log"
grep -q '^Error: the claim itself$' "$serial_log"
grep -q '^serial rerun: still red: @test/runtest-serial-probe$' "${serial_log%.log}.fingerprint"
grep -q '^serial rerun: unmapped: ' "${serial_log%.log}.fingerprint"
# The rerun's verdict must not replace the unit's: the row still says fail.
[ "$(awk -F '\t' '$3 == "cc" { print $5 }' "$state/history.tsv" | tail -1)" = fail ]

# The same red with every stanza clean on its own: the digest carries the
# difference, so the unit-state cursor reports the fingerprint as moved.
serial_clean=$(SWEEP_TEST_OPAM_RC=1 SWEEP_TEST_OPAM_OUT=$environment_failure \
  run_sweep_backend cc --target serial-probe)
grep -q 'm4-max/cc: serial rerun: all clean$' <<<"$serial_clean"
grep -q 'm4-max/cc: fingerprint moved since the previous failure at ' <<<"$serial_clean"
serial_clean_log=$(awk -F '\t' '$3 == "cc" { print $9 }' "$state/history.tsv" | tail -1)
grep -q '^serial rerun: all clean$' "${serial_clean_log%.log}.fingerprint"
absent 'still red' "${serial_clean_log%.log}.fingerprint"

# Negative control: a red whose failures are the tests' own gets no second run.
serial_control=$(SWEEP_TEST_OPAM_RC=1 SWEEP_TEST_OPAM_OUT=$state_failure \
  run_sweep_backend cc --target serial-probe)
grep -q 'm4-max/cc: fail ' <<<"$serial_control"
[ "$(tail -1 "$calls")" = 'exec -- dune runtest serial-probe' ]
absent 'serial rerun' <<<"$serial_control"
serial_control_log=$(awk -F '\t' '$3 == "cc" { print $9 }' "$state/history.tsv" | tail -1)
absent 'serial rerun' "${serial_control_log%.log}.fingerprint"

# A historical target may declare fewer boxes than today's execution map. The
# extra local unit still proves backend facts, but cannot be counted as a member
# of that target's environment matrix.
printf '# measurement-boxes: minix rog-nv\n' >"$main/benchmarks/fixtures/DIGESTS.txt"
git -C "$main" add benchmarks/fixtures/DIGESTS.txt
git -C "$main" commit -qm 'historical two-box matrix'
git -C "$main" push -q origin master
historical_matrix=$(run_sweep --force)
historical_report=$(sed -n 's/^skip coverage: .* -- //p' <<<"$historical_matrix" | tail -1)
grep -q '^completed boxes: <none>$' "$historical_report"
grep -q '^missing boxes: minix, rog-nv$' "$historical_report"
grep -q '^environment status: insufficient (0 of 2 declared boxes completed; need at least 2 unless the matrix is complete)$' \
  "$historical_report"
grep -q '^environment result: NOT AGGREGATED$' "$historical_report"

# Negative control for the one-list contract: changing only the declaration to
# add a box with no execution unit makes the sweep refuse before claiming any
# matrix result. A second hard-coded box census in sweep.sh would stay green.
printf '# measurement-boxes: m4-max minix rog-nv spare\n' \
  >"$main/benchmarks/fixtures/DIGESTS.txt"
git -C "$main" add benchmarks/fixtures/DIGESTS.txt
git -C "$main" commit -qm 'add unscheduled declared box'
git -C "$main" push -q origin master
set +e
matrix_error=$(run_sweep 2>&1)
matrix_error_rc=$?
set -e
[ "$matrix_error_rc" -eq 2 ]
grep -q "^sweep: declared measurement box 'spare' has no sweep unit$" <<<"$matrix_error"

printf 'sweep execution accounting, RTC context, fingerprinting and skip aggregation: PASS\n'
