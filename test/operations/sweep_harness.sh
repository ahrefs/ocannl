#!/usr/bin/env bash

# Integration coverage for tools/sweep.sh's execution-accounting contract. The
# fake opam keeps the test small; real git/worktree operations exercise the
# history migration and the reused-worktree path that made cached GPU passes
# ambiguous in the first place.

set -euo pipefail
. "$(dirname "$0")/../../scripts/harness-support.sh"

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
    capped capped_target remote_opt_in serial_red serial_clean serial_two_inline \
    serial_many_inline serial_control lanes lane_stop_seed lane_stopped \
    aggregator_missing stamp_advance after_cancel; do
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
  SWEEP_TEST_OPAM_SERIAL_RED SWEEP_TEST_OPAM_OUT_SERIAL SWEEP_TEST_SSH_CALLS \
  SWEEP_TEST_SSH_MODE SWEEP_TEST_OWN_GROUP SWEEP_TEST_WAIT_TICKS

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
ssh_calls=$tmp/ssh.calls
# Every fixture wait in this file -- the fake opam's hold, the fake ssh's
# release and hang, and the harness's own readiness checks -- is bounded by
# this many 50ms ticks. Each wait ends as soon as its condition holds, so the
# bound is paid only by a run that is already failing; it is generous so that a
# loaded CI runner starting a nested sweep slowly is not mistaken for one.
wait_ticks=2400
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
# A fixture that arms the wait prefix also gets two witnesses of the lanes
# contract (sweep.sh's run_lane): this process's pid, which the cancellation
# controls check is gone once the sweep returns, and an exclusivity marker a
# second concurrent call on the local worktree cannot take -- local units run
# one at a time, so an overlap is a red unit rather than a silent pass. The
# marker is released on TERM too, which is how a cancelled unit ends.
if [ -n "${SWEEP_TEST_WAIT_PREFIX:-}" ]; then
  printf '%s\n' "$$" >>"$SWEEP_TEST_WAIT_PREFIX.opam-pids"
  if ! mkdir "$SWEEP_TEST_WAIT_PREFIX.busy" 2>/dev/null; then
    printf 'overlapping local unit: %s\n' "$*" >>"$SWEEP_TEST_WAIT_PREFIX.overlap"
    exit 98
  fi
  trap 'rmdir "$SWEEP_TEST_WAIT_PREFIX.busy"' EXIT
  trap 'exit 143' TERM
fi
[ -n "${SWEEP_TEST_OPAM_OUT:-}" ] && printf '%s\n' "$SWEEP_TEST_OPAM_OUT"
case ${OCANNL_BACKEND:-} in
  cc) [ -n "${SWEEP_TEST_OPAM_OUT_CC:-}" ] && printf '%s\n' "$SWEEP_TEST_OPAM_OUT_CC" ;;
  # A backend no nested local sweep runs (multidev_cc's unit is on minix), so
  # that the hermeticity control can hand the nested sweep an AMBIENT backend
  # belonging to neither unit and still be answered. Without an arm here a
  # leaked `multidev_cc` produces nothing and the control passes for the wrong
  # reason.
  multidev_cc)
    [ -n "${SWEEP_TEST_OPAM_OUT_MULTIDEV_CC:-}" ] &&
      printf '%s\n' "$SWEEP_TEST_OPAM_OUT_MULTIDEV_CC"
    ;;
  metal) [ -n "${SWEEP_TEST_OPAM_OUT_METAL:-}" ] && printf '%s\n' "$SWEEP_TEST_OPAM_OUT_METAL" ;;
esac
if [ -n "${SWEEP_TEST_WAIT_PREFIX:-}" ]; then
  : >"$SWEEP_TEST_WAIT_PREFIX.ready"
  waited=0
  while [ ! -e "$SWEEP_TEST_WAIT_PREFIX.release" ]; do
    sleep 0.05
    waited=$((waited + 1))
    [ "$waited" -lt "$SWEEP_TEST_WAIT_TICKS" ] || exit 99
  done
fi
exit "${SWEEP_TEST_OPAM_RC:-0}"
EOF
chmod +x "$fake_bin/opam"

# The harness never reaches a real sweep box. The default below selects only
# cc, while tests that mean to exercise remote selection opt in explicitly and
# hit this recorder. A failed ssh is an ordinary unreachable remote to the
# sweep, so the assertions on this file are the part that makes accidental
# contact fail the harness.
#
# Two opt-in modes stand in for a live remote lane without reaching one:
# `release` waits for the local unit to report ready and releases it, which a
# serial loop -- where the local unit must finish before any remote unit starts
# -- can never do; `hang` answers the reachability probe and then keeps its
# connection busy until killed, so a cancellation control has a remote lane
# in flight. Both stay bounded, and both still end as an unreachable box.
cat >"$fake_bin/ssh" <<'EOF'
#!/bin/sh
printf '%s\n' "$*" >>"$SWEEP_TEST_SSH_CALLS"
case ${SWEEP_TEST_SSH_MODE:-} in
  release)
    waited=0
    while [ ! -e "$SWEEP_TEST_WAIT_PREFIX.ready" ]; do
      sleep 0.05
      waited=$((waited + 1))
      [ "$waited" -lt "$SWEEP_TEST_WAIT_TICKS" ] || exit 1
    done
    : >"$SWEEP_TEST_WAIT_PREFIX.release"
    ;;
  hang)
    case $* in *'printf %s "$HOME"'*) printf '%s' "$HOME"; exit 0 ;; esac
    printf '%s\n' "$$" >>"$SWEEP_TEST_WAIT_PREFIX.ssh-pids"
    : >"$SWEEP_TEST_WAIT_PREFIX.ssh-running"
    waited=0
    while [ "$waited" -lt "$SWEEP_TEST_WAIT_TICKS" ]; do
      sleep 0.05
      waited=$((waited + 1))
    done
    ;;
esac
exit 1
EOF
chmod +x "$fake_bin/ssh"

# Exercise the exact previous schema: old evidence is retained, but marked
# unknown rather than being upgraded retroactively to executed coverage.
printf 'when\tmachine\tbackend\tref\toutcome\tseconds\ttarget\tslow\tlog\n' >"$state/history.tsv"
printf '20260820T000000Z\tlocal\tcc\tdeadbee\tpass\t1\t<all>\t0\t-\n' >>"$state/history.tsv"

run_sweep_args() {
  local arg has_only=0 args=("$@")
  for arg in "${args[@]}"; do
    [ "$arg" = --only ] && has_only=1
  done
  [ "$has_only" -eq 1 ] || args=(--only cc "${args[@]}")
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
  local environment=(-u OCANNL_BACKEND -u OCANNL_TOOL_SWEEP_CAP -u OCANNL_TOOL_SWEEP_CONTEXT_CAP \
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
    "SWEEP_TEST_SSH_CALLS=$ssh_calls" \
    "SWEEP_TEST_SSH_MODE=${SWEEP_TEST_SSH_MODE:-}" \
    "SWEEP_TEST_WAIT_TICKS=$wait_ticks" \
    "OCANNL_TOOL_SWEEP_LOCAL_BOX=${SWEEP_TEST_LOCAL_BOX-m4-max}" \
    "OCANNL_TOOL_SWEEP_JOBS=${SWEEP_TEST_JOBS:-}" \
    "OCANNL_TOOL_SWEEP_REPO=$main" \
    "OCANNL_TOOL_SWEEP_STATE=$state")
  # The cancellation controls need the sweep's own pid, and a process group
  # that holds nothing but the sweep: `exec` makes the caller's `$!` name the
  # sweep itself, and perl's setpgrp gives it a group of its own to signal.
  if [ -n "${SWEEP_TEST_OWN_GROUP:-}" ]; then
    exec perl -e 'setpgrp(0, 0); exec @ARGV or exit 127' -- \
      env "${environment[@]}" "$sweep" "${args[@]}"
  fi
  env "${environment[@]}" "$sweep" "${args[@]}"
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
# aggregation stays explicitly insufficient. (cc and metal: multidev_cc runs on
# minix, which this harness never reaches.)
common=$'SKIPPED on fixture (vacuous): common unevaluated claim\nOCANNL_TOOL_VERDICT_SKIP\tbackend\tfixture.exe\tcommon unevaluated claim'
cc_only=$'SKIPPED on fixture (vacuous): cc-only unevaluated claim\nOCANNL_TOOL_VERDICT_SKIP\tbackend\tfixture.exe\tcc-only unevaluated claim'
metal_only=$'SKIPPED on fixture (vacuous): metal-only unevaluated claim\nOCANNL_TOOL_VERDICT_SKIP\tbackend\tfixture.exe\tmetal-only unevaluated claim'
environment=$'SKIPPED on fixture gate (vacuous): environment-gated claim\nOCANNL_TOOL_VERDICT_SKIP\tenvironment\tfixture.exe\tenvironment-gated claim'
outside=$'SKIPPED on external matrix (vacuous): independently-covered claim\nOCANNL_TOOL_VERDICT_SKIP\toutside-sweep\tfixture.exe\tindependently-covered claim'
cc_unit_log=$common$'\n'$cc_only$'\n'$environment$'\n'$outside
metal_unit_log=$common$'\n'$metal_only$'\n'$environment$'\n'$outside
coverage=$(SWEEP_TEST_OPAM_OUT_CC=$cc_unit_log \
  SWEEP_TEST_OPAM_OUT_METAL=$metal_unit_log \
  run_sweep_args --force --only cc --only metal)
coverage_report=$(sed -n 's/^skip coverage: .* -- //p' <<<"$coverage" | tail -1)
[ -f "$coverage_report" ]
grep -q '^status: partial (2 of 5 known backends completed)$' "$coverage_report"
grep -q '^missing backends: cuda, hip, multidev_cc$' "$coverage_report"
grep -q '^POTENTIAL: skipped on every completed backend: fixture.exe: common unevaluated claim$' \
  "$coverage_report"
absent 'cc-only unevaluated claim' "$coverage_report"
absent 'metal-only unevaluated claim' "$coverage_report"
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
absent 'metal-only unevaluated claim' <<<"$coverage"
absent 'environment-gated claim' <<<"$coverage"
absent 'independently-covered claim' <<<"$coverage"

# The same fixture with a hostile backend in the AMBIENT environment. This is
# the harness's own running condition: the sweep runs a unit's tests as
# `OCANNL_BACKEND=<backend> opam exec -- dune build ...`, so the launcher's
# choice of backend is exported into every test action of that unit, this one
# included. It used to reach the nested sweep, whose forced-clean leg names no
# backend of its own -- so the fake opam answered the AMBIENT one and wrote the
# cc unit's skip records into the multidev_cc unit's log (both were local
# then), turning the intersection this exists to take into a union
# (gh-ocannl-893). The ambient
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
hostile=$(OCANNL_BACKEND=multidev_cc \
  SWEEP_TEST_OPAM_OUT_CC=$cc_unit_log \
  SWEEP_TEST_OPAM_OUT_METAL=$metal_unit_log \
  SWEEP_TEST_OPAM_OUT_MULTIDEV_CC=$leaked \
  run_sweep_args --force --only cc --only metal)
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
    cc | metal) box=m4-max ;;
    cuda) box=rog-nv ;;
    hip | multidev_cc) box=minix ;;
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
# by distinct box rather than by the number of logs (m4-max and minix each
# contribute two in the complete case above).
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
  --run multidev_cc minix "$tmp/mixed-multidev.log" \
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
  --run multidev_cc minix "$tmp/mixed-multidev.log" \
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
for ((waited = 0; waited < wait_ticks; waited++)); do
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
assert_metal_options() {
  if grep -Fxq "metal options: $1" "$2"; then return 0; fi
  echo 'metal fingerprint differs from rendered options' >&2
  return 1
}
assert_metal_options "$rendered_metal_options" "${metal_log%.log}.fingerprint"
printf 'metal options: %s-MUTANT\n' "$rendered_metal_options" >"$tmp/metal-mutant.fingerprint"
harness_rejected 1 '^metal fingerprint differs from rendered options$' \
  "$tmp/metal-mutant-rejection" assert_metal_options \
  "$rendered_metal_options" "$tmp/metal-mutant.fingerprint"
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
grep -q "^sweep: unknown backend 'cudaa'; known: cc metal cuda hip multidev_cc" <<<"$only_typo_error"

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

# Every unqualified --target fixture above is scoped to cc by run_sweep_args;
# the recorder must still be empty after state-probe, pre-diff-probe and the
# capped target. Without that default these cases walk cuda and hip, coupling
# the harness to whether the real lab aliases happen to be awake.
[ ! -s "$ssh_calls" ]
# Explicit remote selection remains possible and reaches only the fake ssh.
# This is the opposing control for the default rather than a blanket removal
# of the remote units from nested sweeps.
remote_opt_in=$(run_sweep_args --only cuda --target state-probe)
grep -q 'rog-nv/cuda: skip (unreachable)' <<<"$remote_opt_in"
grep -q 'rog-nv-wsl' "$ssh_calls"
absent 'minix-amd-wsl' "$ssh_calls"

# An environment-red unit -- a red whose log carries a runtime-refusal signature
# from sweep.sh's ENVIRONMENT_REFUSALS table -- reruns its failing stanzas one at
# a time at `-j 1` and records which stayed red (gh-ocannl-945). The fixture is
# the 2026-09-05 minix/hip shape: one stanza refused at hip_init, one that
# crashed after (no signature of its own, still a red stanza), a multi-name
# tests stanza, and an inline expectation located in a source file. Every name
# in the tests stanza gets its own generated alias; the one inline site gets a
# bounded directory fallback. The fake opam holds the first stanza red on its
# own and clears the others. One unit (`--only cc`): every local unit would
# rerun the same fixture, and the remote ones would reach for ssh.
environment_failure='File "test/dune", line 2, characters 7-28:
2 |  (alias runtest-serial-probe)
Fatal error: exception hip_init:
HIP_ERROR_INVALID_DEVICE
File "test/dune", lines 5-8, characters 0-0:
5 | (rule
6 |  (alias runtest-pre-diff-probe)
......
Command got signal SEGV.
File "test/dune", lines 9-12, characters 0-0:
9 | (tests
10 |  (names "serial-alpha"
11 |   serial-beta))
12 |  (libraries fixture))
File "test/inline_expect.ml", line 1, characters 0-0:
Error: inline expectation differs
diff --git a/test/inline_expect.ml b/_build/default/test/inline_expect.ml.corrected'
serial_red=$(SWEEP_TEST_OPAM_RC=1 SWEEP_TEST_OPAM_OUT=$environment_failure \
  SWEEP_TEST_OPAM_SERIAL_RED='@test/runtest-serial-probe' \
  SWEEP_TEST_OPAM_OUT_SERIAL='File "test/dune", line 2, characters 7-28:
2 |  (alias runtest-serial-probe)
Error: the claim itself' run_sweep_backend cc --target serial-probe)
grep -q 'm4-max/cc: fail ' <<<"$serial_red"
# One dune call per stanza, so each has its own verdict; sorted, after the unit.
[ "$(tail -6 "$calls" | sed -n '1p')" = 'exec -- dune runtest serial-probe' ]
[ "$(tail -6 "$calls" | sed -n '2p')" = 'exec -- dune build -j 1 @test/runtest-pre-diff-probe' ]
[ "$(tail -6 "$calls" | sed -n '3p')" = 'exec -- dune build -j 1 @test/runtest-serial-probe' ]
[ "$(tail -6 "$calls" | sed -n '4p')" = 'exec -- dune build -j 1 @test/runtest-serial-alpha' ]
[ "$(tail -6 "$calls" | sed -n '5p')" = 'exec -- dune build -j 1 @test/runtest-serial-beta' ]
[ "$(tail -6 "$calls" | sed -n '6p')" = 'exec -- dune build -j 1 @test/runtest' ]
# The verdict reaches all three channels: the summary, the log, the fingerprint.
grep -q 'm4-max/cc: environment-red, 4 stanzas and 1 directory fallback rerun at -j 1' \
  <<<"$serial_red"
grep -q 'm4-max/cc: serial rerun: still red: @test/runtest-serial-probe$' <<<"$serial_red"
serial_log=$(awk -F '\t' '$3 == "cc" { print $9 }' "$state/history.tsv" | tail -1)
grep -q '^serial rerun: still red: @test/runtest-serial-probe$' "$serial_log"
grep -q '^serial rerun: directory fallback (1 inline site): @test/runtest$' "$serial_log"
absent '^serial rerun: unmapped:' "$serial_log"
absent '^serial rerun: all clean' "$serial_log"
grep -q '^Error: the claim itself$' "$serial_log"
grep -q '^serial rerun: still red: @test/runtest-serial-probe$' "${serial_log%.log}.fingerprint"
grep -q '^serial rerun: directory fallback (1 inline site): @test/runtest$' \
  "${serial_log%.log}.fingerprint"
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

# Two inline sites are still a bounded fallback. They share one directory, so
# the directory alias runs once; the site count is the cap, not the number of
# distinct aliases it happens to produce.
two_inline_failure='Fatal error: exception hip_init:
HIP_ERROR_INVALID_DEVICE
File "test/inline_one.ml", line 1, characters 0-0:
Error: first inline expectation differs
diff --git a/test/inline_one.ml b/_build/default/test/inline_one.ml.corrected
File "test/inline_two.ml", line 1, characters 0-0:
Error: second inline expectation differs
diff --git a/test/inline_two.ml b/_build/default/test/inline_two.ml.corrected'
serial_two_inline=$(SWEEP_TEST_OPAM_RC=1 SWEEP_TEST_OPAM_OUT=$two_inline_failure \
  run_sweep_backend cc --target two-inline-probe)
[ "$(tail -2 "$calls" | sed -n '1p')" = 'exec -- dune runtest two-inline-probe' ]
[ "$(tail -2 "$calls" | sed -n '2p')" = 'exec -- dune build -j 1 @test/runtest' ]
grep -q 'm4-max/cc: environment-red, 0 stanzas and 1 directory fallback rerun at -j 1' \
  <<<"$serial_two_inline"
two_inline_log=$(awk -F '\t' '$3 == "cc" { print $9 }' "$state/history.tsv" | tail -1)
grep -q '^serial rerun: directory fallback (2 inline sites): @test/runtest$' \
  "$two_inline_log"
absent '^serial rerun: unmapped:' "$two_inline_log"

# Three inline sites cross the bound: report each one and do not turn the retry
# into a directory-wide serial suite. This is the opposing control for the two
# cases above, and catches either an off-by-one cap or a cap accidentally
# applied to deduplicated directory aliases.
three_inline_failure="$two_inline_failure
File \"test/inline_three.ml\", line 1, characters 0-0:
Error: third inline expectation differs
diff --git a/test/inline_three.ml b/_build/default/test/inline_three.ml.corrected
File \"test/compile_error.ml\", line 1, characters 0-0:
Error: this source site has no expectation correction"
serial_many_inline=$(SWEEP_TEST_OPAM_RC=1 SWEEP_TEST_OPAM_OUT=$three_inline_failure \
  run_sweep_backend cc --target three-inline-probe)
[ "$(tail -1 "$calls")" = 'exec -- dune runtest three-inline-probe' ]
grep -q 'm4-max/cc: environment-red, 0 stanzas and 0 directory fallbacks rerun at -j 1' \
  <<<"$serial_many_inline"
many_inline_log=$(awk -F '\t' '$3 == "cc" { print $9 }' "$state/history.tsv" | tail -1)
grep -q '^serial rerun: nothing to rerun -- no site names a stanza$' "$many_inline_log"
grep -q '^serial rerun: unmapped: \[File "test/compile_error.ml", line 1\] \[File "test/inline_one.ml", line 1\] \[File "test/inline_three.ml", line 1\] \[File "test/inline_two.ml", line 1\]$' \
  "$many_inline_log"
absent '^serial rerun: directory fallback' "$many_inline_log"

# Negative control: a red whose failures are the tests' own gets no second run.
serial_control=$(SWEEP_TEST_OPAM_RC=1 SWEEP_TEST_OPAM_OUT=$state_failure \
  run_sweep_backend cc --target serial-probe)
grep -q 'm4-max/cc: fail ' <<<"$serial_control"
[ "$(tail -1 "$calls")" = 'exec -- dune runtest serial-probe' ]
absent 'serial rerun' <<<"$serial_control"
serial_control_log=$(awk -F '\t' '$3 == "cc" { print $9 }' "$state/history.tsv" | tail -1)
absent 'serial rerun' "${serial_control_log%.log}.fingerprint"

# Lanes (gh-ocannl-976): one per machine, concurrent across machines and
# sequential within one. The local cc unit holds until a REMOTE unit's probe
# releases it, which a serial loop -- every local unit ahead of every remote one
# -- can never do: there cc would time out red and the remote probes would
# arrive after the fact. The local metal unit follows cc in the same lane, and
# the fake opam's exclusivity marker turns any overlap of the two into a red
# unit. The lanes line is derived from the table's machine column.
lanes=$(SWEEP_TEST_WAIT_PREFIX=$tmp/lanes SWEEP_TEST_SSH_MODE=release \
  run_sweep_args --only cc --only metal --only cuda --only hip --only multidev_cc \
  --target lane-probe)
grep -q '^lanes:  m4-max(cc,metal)  rog-nv(cuda)  minix(hip,multidev_cc)$' <<<"$lanes"
grep -q '^  m4-max/cc: incremental-pass ' <<<"$lanes"
grep -q '^  m4-max/metal: incremental-pass ' <<<"$lanes"
grep -q '^  rog-nv/cuda: skip (unreachable)$' <<<"$lanes"
grep -q '^  minix/hip: skip (unreachable)$' <<<"$lanes"
grep -q '^  minix/multidev_cc: skip (unreachable)$' <<<"$lanes"
[ ! -e "$tmp/lanes.overlap" ]
# The rows are those of a serial run in everything but their order: one per
# unit, each under its own machine.
[ "$(awk -F '\t' '$7 == "lane-probe" { print $2 "/" $3 ":" $5 }' "$state/history.tsv" | sort)" = \
  "$(printf '%s\n' m4-max/cc:incremental-pass m4-max/metal:incremental-pass \
    minix/hip:skip minix/multidev_cc:skip rog-nv/cuda:skip | sort)" ]

# The run record (gh-ocannl-977). A complete run's record names the exit kind,
# every selected unit's outcome with its lane's completion, and today's whole
# backend->box map -- the last one covering backends this run did not select, so
# a consumer can age a backend's staleness against the box that owns it now.
lanes_record=$(sed -n 's/^run:  *//p' <<<"$lanes")
[ -f "$lanes_record" ]
[ "$(head -1 "$lanes_record")" = "$(printf 'schema\t1')" ]
[ "$(awk -F '\t' '$1 == "run" { print $8 }' "$lanes_record")" = complete ]
[ "$(awk -F '\t' '$1 == "run" { print $5 "\t" $6 }' "$lanes_record")" = \
  "$(printf 'lane-probe\t0')" ]
[ "$(awk -F '\t' '$1 == "unit" { print $2 "/" $3 ":" $4 ":" $5 }' "$lanes_record" | sort)" = \
  "$(printf '%s\n' m4-max/cc:incremental-pass:0 m4-max/metal:incremental-pass:0 \
    minix/hip:skip:0 minix/multidev_cc:skip:0 rog-nv/cuda:skip:0 | sort)" ]
[ "$(awk -F '\t' '$1 == "backend" { print $2 ":" $3 }' "$lanes_record")" = \
  "$(printf '%s\n' cc:m4-max metal:m4-max cuda:rog-nv hip:minix multidev_cc:minix)" ]
# The log column points at the unit's own log, so a consumer needs no second
# rule for reconstructing the path a row's diagnostics live in.
[ -f "$(awk -F '\t' '$1 == "unit" && $3 == "cc" { print $6 }' "$lanes_record")" ]

# A lane that stops before finishing (the exit-2 shape the consumer could not
# tell from a startup refusal): the record is written anyway, the rows the lane
# DID write are in it, the units it never reached are `no-row`, and both are
# marked as belonging to a lane that stopped -- while the other lane, which
# finished, is not. Provoked by corrupting the unit-state file of the first unit
# of the local lane, which kills that lane strictly AFTER its history row: the
# seeding run writes that file rather than the harness recomputing its keyed
# name, which would only restate what unit_state_path already says.
lane_stop_seed=$(run_sweep_args --only cc --only metal --only cuda --target lane-stop-probe)
grep -q '^  m4-max/cc: incremental-pass ' <<<"$lane_stop_seed"
lane_stop_state=$(ls "$state"/unit-state/m4-max-cc-lane-stop-probe-*.state)
[ -f "$lane_stop_state" ]
printf 'schema\t9\n' >"$lane_stop_state"
set +e
lane_stopped=$(run_sweep_args --only cc --only metal --only cuda --target lane-stop-probe 2>&1)
lane_stopped_rc=$?
set -e
[ "$lane_stopped_rc" -eq 2 ]
grep -q '^sweep: lane(s) stopped before finishing: m4-max (exit 2)$' <<<"$lane_stopped"
lane_stopped_record=$(sed -n 's/^run:  *//p' <<<"$lane_stopped")
[ -f "$lane_stopped_record" ]
[ "$(awk -F '\t' '$1 == "run" { print $8 }' "$lane_stopped_record")" = lane-stopped ]
[ "$(awk -F '\t' '$1 == "unit" { print $2 "/" $3 ":" $4 ":" $5 }' "$lane_stopped_record" | sort)" = \
  "$(printf '%s\n' m4-max/cc:incremental-pass:1 m4-max/metal:no-row:1 rog-nv/cuda:skip:0 | sort)" ]

# A post-lane harness failure rewrites the exit kind rather than leaving a record
# that claims `complete` over an exit 2. Provoked by running a copy of the sweep
# from a directory that holds no aggregate-skips.sh -- the aggregator is resolved
# beside the script -- on a forced unscoped run, which is the only shape that
# aggregates. The units all recorded, so their rows stay real; it is the run-level
# kind that has to tell the truth about how the process ended.
# Everything the script resolves beside itself is SYMLINKED in and only
# aggregate-skips.sh is withheld, so the run reaches the aggregation step instead
# of refusing at startup over some other sibling. Symlinked as a set rather than
# copied by name: sweep.sh reaches for its neighbours (box-jobs.sh, and the
# benchmarks fixture parser one level up) and a fixture naming them individually
# goes red the next time one is added -- which is exactly how this one first
# failed, on a base that had grown one.
lonely_tools=$(cd "$(dirname "$sweep")" && pwd)
mkdir -p "$tmp/lonely/tools" "$tmp/lonely/benchmarks"
for lonely_sibling in "$lonely_tools"/*; do
  if [ "$(basename "$lonely_sibling")" = aggregate-skips.sh ]; then continue; fi
  ln -s "$lonely_sibling" "$tmp/lonely/tools/$(basename "$lonely_sibling")"
done
ln -s "$lonely_tools/../benchmarks/fixture_digest.py" "$tmp/lonely/benchmarks/fixture_digest.py"
[ ! -e "$tmp/lonely/tools/aggregate-skips.sh" ]
sweep_with_aggregator=$sweep
sweep=$tmp/lonely/tools/sweep.sh
set +e
aggregator_missing=$(run_sweep --force 2>&1)
aggregator_missing_rc=$?
set -e
sweep=$sweep_with_aggregator
[ "$aggregator_missing_rc" -eq 2 ]
grep -q '^sweep: skip aggregator is not executable: ' <<<"$aggregator_missing"
aggregator_missing_record=$(sed -n 's/^run:  *//p' <<<"$aggregator_missing")
[ "$(awk -F '\t' '$1 == "run" { print $8 }' "$aggregator_missing_record")" = post-run-failed ]
[ "$(awk -F '\t' '$1 == "unit" { print $2 "/" $3 ":" $4 ":" $5 }' "$aggregator_missing_record")" = \
  m4-max/cc:pass:0 ]

# The stamp names every per-run artifact, and two invocations a second apart can
# otherwise choose the same one -- a cancelled run releases the lock inside the
# second it ends, so its retry can land on top of its logs, fingerprints and
# record. Seeding the next few seconds' stamps as taken forces the advance
# whichever second the sweep starts in, rather than relying on it being slow.
seeded_stamps=$(perl -MPOSIX -e \
  'print join(" ", map { strftime("%Y%m%dT%H%M%SZ", gmtime(time + $_)) } 0 .. 5)')
for seeded in $seeded_stamps; do : >"$state/logs/$seeded-seed.log"; done
stamp_advance=$(run_sweep_args --target stamp-probe)
advanced_record=$(sed -n 's/^run:  *//p' <<<"$stamp_advance")
advanced_stamp=$(basename "$advanced_record" -run.tsv)
for seeded in $seeded_stamps; do
  if [ "$advanced_stamp" = "$seeded" ]; then
    printf 'sweep_harness: run stamp %s collided with a seeded artifact\n' "$advanced_stamp" >&2
    exit 1
  fi
done
# And the advance is what produced that: the run's own history row carries the
# same advanced stamp, so the row and its artifacts still name one run.
[ -n "$(awk -F '\t' -v s="$advanced_stamp" '$1 == s && $7 == "stamp-probe"' "$state/history.tsv")" ]
rm -f "$state"/logs/*-seed.log

# Cancelling a sweep stops EVERY lane: here the local lane's unit is held in its
# test leg and the rog-nv lane's in its preparation ssh, both under supervisors.
# TERM to the sweep's pid must be relayed through each lane to its supervisor,
# and a TERM to the process group reaches them directly; either way the sweep
# returns only after every unit process is gone, which the pids and the lock
# (taken again by the follow-up run) witness.
cancel_sweep() { # pid|group
  local how=$1 prefix=$tmp/cancel-$1 pid rc
  wait_prefix=$prefix
  SWEEP_TEST_OWN_GROUP=1 SWEEP_TEST_WAIT_PREFIX=$prefix SWEEP_TEST_SSH_MODE=hang \
    run_sweep_args --only cc --only cuda --target cancel-probe \
    >"$prefix.out" 2>"$prefix.err" &
  pid=$!
  holder_pid=$pid
  waited=0
  until [ -e "$prefix.ready" ] && [ -e "$prefix.ssh-running" ]; do
    [ "$waited" -lt "$wait_ticks" ] || break
    sleep 0.05
    waited=$((waited + 1))
  done
  [ -e "$prefix.ready" ] && [ -e "$prefix.ssh-running" ]
  case $how in
    pid) kill -TERM "$pid" ;;
    group) kill -TERM -- "-$pid" ;;
  esac
  set +e
  wait "$pid"
  rc=$?
  set -e
  holder_pid=
  wait_prefix=
  if [ "$rc" -ne 143 ]; then
    printf 'sweep_harness: sweep cancelled by %s TERM exited %s\n' "$how" "$rc" >&2
    cat "$prefix.out" "$prefix.err" >&2
    return 1
  fi
  while IFS= read -r pid; do
    if kill -0 "$pid" 2>/dev/null; then
      printf 'sweep_harness: unit process %s outlived the cancelled sweep\n' "$pid" >&2
      return 1
    fi
  done < <(cat "$prefix.opam-pids" "$prefix.ssh-pids")
  [ ! -e "$prefix.busy" ]
  # A cancelled run ended, so it owes a record too: the rows its lanes wrote
  # before the signal are real, and the exit kind says why the rest are missing.
  # Located through the sweep's OWN `run:` line, not by scanning the log
  # directory: a cancelled run ends before the summary block, so that locator is
  # the only thing standing between an operator and its record.
  cancel_record=$(sed -n 's/^run:  *//p' "$prefix.out")
  [ -f "$cancel_record" ]
  [ "$(awk -F '\t' '$1 == "run" { print $8 }' "$cancel_record")" = cancelled ]
}
# Called directly, not captured: errexit does not reach inside a command
# substitution, and the assertions are in the function.
cancel_sweep pid
cancel_sweep group
after_cancel=$(run_sweep_args --target cancel-probe)
grep -q '^  m4-max/cc: incremental-pass ' <<<"$after_cancel"

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
records_before=$(ls "$state"/logs/*-run.tsv | wc -l)
set +e
matrix_error=$(run_sweep 2>&1)
matrix_error_rc=$?
set -e
[ "$matrix_error_rc" -eq 2 ]
grep -q "^sweep: declared measurement box 'spare' has no sweep unit$" <<<"$matrix_error"
# A startup refusal swept nothing and writes NO record: that absence is the
# signal, and it is what distinguishes this exit 2 from a lane-stopped one.
[ "$(ls "$state"/logs/*-run.tsv | wc -l)" -eq "$records_before" ]

printf 'sweep execution accounting, RTC context, fingerprinting, run record and skip aggregation: PASS\n'
