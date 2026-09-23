#!/usr/bin/env bash

# Integration coverage for tools/sweep.sh's execution-accounting contract. The
# fake opam keeps the test small; real git/worktree operations exercise the
# history migration and the reused-worktree path that made cached GPU passes
# ambiguous in the first place.

# -E (errtrace) so the ERR trap below also fires inside helper functions,
# command substitutions and subshells: without it a failing predicate inside a
# helper such as `dest_refused` ends the harness without naming any line.
set -Eeuo pipefail
. "$(dirname "$0")/../../scripts/harness-support.sh"

# Most assertions below are deliberately quiet shell predicates. If one fails
# under errexit, name the exact site before cleanup removes its evidence; the
# expected-error controls temporarily disable errexit and therefore stay quiet.
# One predicate per statement, never `[ A ] && [ B ]`: errexit exempts a failing
# LEFT operand of an `&&` list, so the pair passes silently in exactly the case
# the first half exists to catch -- and an assertion that cannot fail is worse
# than none, because the case around it reads as covered.
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
    capped capped_target remote_opt_in dest_wsl dest_linux dest_missing dest_local_only \
    dest_bogus dest_no_kind_of dest_half dest_override dest_override_wins dest_bad_override \
    dest_inherited_kind_of serial_red serial_clean serial_two_inline \
    serial_many_inline serial_control lanes lane_stop_seed lane_stopped \
    aggregator_missing stamp_advance dxg_clean dxg_red dxg_collection dxg_unavailable \
    dxg_many dxg_bounds dxg_no_trigger native_quiet_block native_red_a native_red_b \
    native_of_dxg dxg_of_native native_collection native_unit_linux native_unit_wsl \
    native_unit_cpu native_abort_run native_other_abort hold_lock_ok \
    tuf_asleep tuf_no_wake_lab tuf_up tuf_unreachable tuf_inhibited tuf_sleep_fails \
    tuf_unguarded tuf_unguarded_wsl tuf_self_refusal tuf_cancelled guard_held guard_refused \
    guard_absent \
    after_cancel; do
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
  SWEEP_TEST_SSH_MODE SWEEP_TEST_OWN_GROUP SWEEP_TEST_WAIT_TICKS \
  SWEEP_TEST_HOSTS SWEEP_TEST_DEST_ROG SWEEP_TEST_DEST_MINIX \
  SWEEP_TEST_KERNEL_LINES SWEEP_TEST_BOOT_ID SWEEP_TEST_DEST_TUF SWEEP_TEST_WAKE_LAB \
  SWEEP_TEST_WAKE_LAB_CALLS SWEEP_TEST_TUF_STATUS SWEEP_TEST_TUF_SLEEP SWEEP_TEST_HOLD_DENIED

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
wake_lab_calls=$tmp/wake-lab.calls
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
# A third, `window`, answers a remote unit's reachability probe, its guest-identity probe and its
# kernel-window collection -- the last with SWEEP_TEST_KERNEL_LINES as the box's kernel log for the
# window -- and refuses everything else, so the unit ends at its preparation as `error`, which is a
# path that collects the window (gh-ocannl-1034). What the window QUERY was is left in the call log.
cat >"$fake_bin/ssh" <<'EOF'
#!/bin/sh
printf '%s\n' "$*" >>"$SWEEP_TEST_SSH_CALLS"
case ${SWEEP_TEST_SSH_MODE:-} in
  window)
    # A far side whose supervisor could not take the sleep guard (no polkit grant) says so on the
    # unit's stderr before the unit runs; the fixture says it for the leg it then refuses.
    case $* in
      *"-- --hold '"*)
        [ -n "${SWEEP_TEST_HOLD_DENIED:-}" ] &&
          echo 'sweep-hold: WARNING: running WITHOUT a sleep guard, so nothing at the OS level stops a suspend under this run -- fixture-inhibit refused: Access denied' >&2
        ;;
    esac
    case $* in
      *window-bounds*)
        now=$(date +%s)
        printf 'window-bounds %s %s\n' "$((now - 5))" "$((now + 1))"
        printf 'window-boot %s\n' "$SWEEP_TEST_BOOT_ID"
        [ -n "${SWEEP_TEST_KERNEL_LINES:-}" ] && printf '%s\n' "$SWEEP_TEST_KERNEL_LINES"
        exit 0
        ;;
      *'"$HOME"'*) printf '%s\n%s\n%s\n' "$HOME" "$(date +%s)" "$SWEEP_TEST_BOOT_ID"; exit 0 ;;
      *boot_id*) printf '%s\n' "$SWEEP_TEST_BOOT_ID"; exit 0 ;;
    esac
    ;;
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
    # The reachability probe, answered rather than hung: it is NOT under a
    # cancellation-visible supervisor, so hanging here would wait out its own 60s
    # cap and the control would never reach the preparation call whose relay and
    # reap it exists to test. Matched on `$HOME` alone, so that what the probe
    # asks for besides it -- the box's clock, since gh-ocannl-979 -- can change
    # without silently turning this control into a 60-second sleep.
    case $* in
      *'"$HOME"'*) printf '%s\n%s\n' "$HOME" "$(date +%s)"; exit 0 ;;
    esac
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

# The lab's power script, as the sweep uses it for its one gated box (gh-ocannl-1035): `status tuf`,
# answered as the real script prints a box that is asleep (the default) or up at its Linux, and
# `sleep tuf`, which -- as the real one does before any power verb -- first takes the box's LANE lock
# and is refused while another holder has it. So a lane that asked for its own box's sleep while
# still holding its reservation is refused by itself here, exactly as it would be in the lab. The
# sleep's other outcomes are the real script's lines: done, refused by a block inhibitor, failed.
cat >"$fake_bin/wake-lab.sh" <<'EOF'
#!/bin/sh
printf '%s\n' "$*" >>"$SWEEP_TEST_WAKE_LAB_CALLS"
case $1 in
  status)
    echo 'box    router-active   reached OS and ssh endpoint'
    case ${SWEEP_TEST_TUF_STATUS:-down} in
      up) echo 'tuf    router-active=?  os=linux  linux=UP  sleep-blocks=0' ;;
      *) echo 'tuf    router-active=?  os=--  linux=--' ;;
    esac
    echo
    echo 'sleep-blocks counts a native Linux box logind block inhibitors on sleep (listed under it):'
    ;;
  sleep)
    mkdir -p "$WAKE_LAB_LOCK_DIR"
    if ! perl -e 'use Fcntl ":flock"; open(my $h, ">>", $ARGV[0]) or exit 1;
        exit(flock($h, LOCK_EX | LOCK_NB) ? 0 : 1)' "$WAKE_LAB_LOCK_DIR/$2.lock"; then
      echo "  sleep REFUSED on $2: $(head -1 "$WAKE_LAB_LOCK_DIR/$2.lock")"
      echo "sleep REFUSED on: $2 (a lab lock is held; wait for the holder, or --force to take the box anyway)"
      exit 1
    fi
    echo "$2: sleep"
    case ${SWEEP_TEST_TUF_SLEEP:-ok} in
      ok) echo 'confirming...'; echo "$2=DOWN (07:00:00)" ;;
      inhibited)
        echo '  Operation inhibited by "fleet-worker" (PID 4242 "python3", user lukstafi), reason is "a correctness slot".'
        echo "  sleep REFUSED on $2 by a block inhibitor (a run there holds it; see status)"
        exit 1
        ;;
      *) echo "  sleep FAILED on $2 (command exited 5)"; exit 1 ;;
    esac
    ;;
  *) exit 2 ;;
esac
EOF
chmod +x "$fake_bin/wake-lab.sh"

# Stand-ins for the site's wake-lab host table (gh-ocannl-1030), which is site data outside the
# repository: the sweep sources it and asks `kind_of <box>` which boot -- native Ubuntu (`linux`) or
# the WSL guest (`wsl`) -- each GPU box's destination is. The nested sweep is pointed at one of
# these through WAKE_LAB_HOSTS (run_sweep_args pins it), the native one by default because that is
# the lab's boot today. Each defines the table's other functions too, so that only the property a
# case names differs from a real table.
write_hosts() { # path kind-of-body
  cat >"$1" <<HOSTS
mac_of() { case "\$1" in rog|minix) echo 00:00:00:00:00:00 ;; *) return 1 ;; esac; }
eth_mac_of() { mac_of "\$1"; }
ip_of() { case "\$1" in rog|minix) echo 192.0.2.1 ;; *) return 1 ;; esac; }
echo 'a site table that prints while sourced'
$2
HOSTS
}
write_hosts "$tmp/hosts-linux.sh" 'kind_of() { case "$1" in rog|minix) echo linux ;; *) return 1 ;; esac; }'
write_hosts "$tmp/hosts-wsl.sh" 'kind_of() { case "$1" in rog|minix) echo wsl ;; *) return 1 ;; esac; }'
# A kind wake-lab itself refuses, as a box booted into Windows would read if someone wrote it down.
write_hosts "$tmp/hosts-bogus.sh" 'kind_of() { case "$1" in rog|minix) echo win ;; *) return 1 ;; esac; }'
# A table from before kind_of existed.
write_hosts "$tmp/hosts-no-kind-of.sh" ''
# One box known, the other not: the sweep must refuse the RUN, not sweep the half it can reach.
write_hosts "$tmp/hosts-half.sh" 'kind_of() { case "$1" in rog) echo linux ;; *) return 1 ;; esac; }'

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
  # The lab lock directory is PINNED rather than merely left to $HOME. The nested sweep reserves
  # each remote lane's box for real, and this harness's remote lanes are fake -- so a run that
  # reached the true lab's locks could wait out OCANNL_TOOL_SWEEP_LAB_LOCK_WAIT on a reservation a
  # genuine sweep was holding, or skip its lanes for that reason, instead of producing the
  # `skip (unreachable)` rows the cases assert. $HOME above already covers the DEFAULT path, but an
  # ambient WAKE_LAB_LOCK_DIR -- the supported way to move that directory, and what the ludics-lite
  # suite sets -- overrides the default and would be inherited straight through. Its wait budget is
  # unset for the reason the caps above are: an ambient one would rewrite a budget these cases
  # depend on.
  #
  # Quoted, unlike the assignment prefix this replaces: these are `env`'s
  # ARGUMENTS now, so the multi-line fixture logs would otherwise be split into
  # words and `env` would try to run one of them as the command.
  #
  # The site host table and the per-box destination overrides are pinned for the same reason as the
  # lock directory (gh-ocannl-1030): an ambient WAKE_LAB_HOSTS or OCANNL_TOOL_SWEEP_DEST_* -- both
  # supported knobs -- would otherwise decide which alias the fake remote lanes are asked for, and
  # the destination cases below assert exactly that.
  local environment=(-u OCANNL_BACKEND -u OCANNL_TOOL_SWEEP_CAP -u OCANNL_TOOL_SWEEP_CONTEXT_CAP \
    -u OCANNL_TOOL_SWEEP_LOCAL_BOX -u OCANNL_TOOL_SWEEP_LAB_LOCK_WAIT \
    "HOME=$tmp/home" \
    "WAKE_LAB_LOCK_DIR=$tmp/lab-locks" \
    "WAKE_LAB_HOSTS=${SWEEP_TEST_HOSTS:-$tmp/hosts-linux.sh}" \
    "OCANNL_TOOL_SWEEP_DEST_ROG=${SWEEP_TEST_DEST_ROG:-}" \
    "OCANNL_TOOL_SWEEP_DEST_MINIX=${SWEEP_TEST_DEST_MINIX:-}" \
    "OCANNL_TOOL_SWEEP_DEST_TUF=${SWEEP_TEST_DEST_TUF:-}" \
    "OCANNL_TOOL_SWEEP_WAKE_LAB=${SWEEP_TEST_WAKE_LAB-$fake_bin/wake-lab.sh}" \
    "SWEEP_TEST_WAKE_LAB_CALLS=$wake_lab_calls" \
    "SWEEP_TEST_TUF_STATUS=${SWEEP_TEST_TUF_STATUS:-down}" \
    "SWEEP_TEST_TUF_SLEEP=${SWEEP_TEST_TUF_SLEEP:-ok}" \
    "SWEEP_TEST_HOLD_DENIED=${SWEEP_TEST_HOLD_DENIED:-}" \
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
    "SWEEP_TEST_KERNEL_LINES=${SWEEP_TEST_KERNEL_LINES:-}" \
    "SWEEP_TEST_BOOT_ID=${SWEEP_TEST_BOOT_ID:-fixture-boot}" \
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
[ -n "$unit_state" ]
[ -f "$unit_state" ]
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
[ -n "$metal_log" ]
[ -f "$metal_log" ]
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
grep -q 'rog-nv-linux' "$ssh_calls"
absent 'minix-amd' "$ssh_calls"
absent -- '-wsl' "$ssh_calls"

# The GPU destinations follow the site's boot kind (gh-ocannl-1030). rog and minix are dual-boot,
# only the booted system answers, and the sweep once hard-coded the `-wsl` aliases -- so from the
# first native-Ubuntu boot every GPU unit and multidev_cc recorded `skip (unreachable)`, the same
# row a sleeping box writes. Each case below starts from an empty recorder, and each asks the fake
# ssh for exactly the aliases its table names and no other.
#
# WSL boot: both boxes through their WSL guests.
: >"$ssh_calls"
dest_wsl=$(SWEEP_TEST_HOSTS=$tmp/hosts-wsl.sh \
  run_sweep_args --only cuda --only hip --only multidev_cc --target dest-wsl-probe)
grep -q '^destinations: rog-nv=rog-nv-wsl minix=minix-amd-wsl tuf=tuf-amd-linux$' <<<"$dest_wsl"
# tuf is single-boot, so the WSL table leaves it where it is -- and asleep (the fake's default), it
# is a gate that dials nothing, which is why no `-linux` alias reaches the recorder below.
grep -q '^  tuf/hip: gate (tuf not up: router-active=? os=-- linux=--)$' <<<"$dest_wsl"
grep -q '^  rog-nv/cuda: skip (unreachable)$' <<<"$dest_wsl"
grep -q '^  minix/hip: skip (unreachable)$' <<<"$dest_wsl"
grep -q '^  minix/multidev_cc: skip (unreachable)$' <<<"$dest_wsl"
grep -q ' rog-nv-wsl ' "$ssh_calls"
grep -q ' minix-amd-wsl ' "$ssh_calls"
absent -- '-linux' "$ssh_calls"
# Native boot (the default table): both boxes through native Ubuntu, and each lane still reserves
# ITS box -- the lock is the box's, whichever system it booted, so a `-linux` alias must name the
# same lock file the `-wsl` one did. The lane locks earlier cases left behind are removed first, so
# the files' presence afterwards is this run's evidence.
: >"$ssh_calls"
rm -f "$tmp/lab-locks/rog.lock" "$tmp/lab-locks/minix.lock"
dest_linux=$(run_sweep_args --only cuda --only hip --only multidev_cc --target dest-linux-probe)
grep -q '^destinations: rog-nv=rog-nv-linux minix=minix-amd-linux tuf=tuf-amd-linux$' <<<"$dest_linux"
grep -q '^  rog-nv/cuda: skip (unreachable)$' <<<"$dest_linux"
grep -q '^  minix/hip: skip (unreachable)$' <<<"$dest_linux"
grep -q '^  minix/multidev_cc: skip (unreachable)$' <<<"$dest_linux"
grep -q ' rog-nv-linux ' "$ssh_calls"
grep -q ' minix-amd-linux ' "$ssh_calls"
absent -- '-wsl' "$ssh_calls"
[ -e "$tmp/lab-locks/rog.lock" ]
[ -e "$tmp/lab-locks/minix.lock" ]
grep -q '^ocannl sweep ' "$tmp/lab-locks/rog.lock"
grep -q '^ocannl sweep ' "$tmp/lab-locks/minix.lock"
# An unreadable kind refuses the RUN at startup -- exit 2, nothing contacted, no history row and
# no run record -- rather than guessing an alias and filing the guess's failure as a sleeping box.
# The refusal says which table, which box, and what to set instead.
# Every check says what it saw and RETURNS rather than failing in place: errtrace would fire the
# trap on a bare predicate in here, but it names the predicate (and, on macOS's bash 3.2, only the
# function's first line), not which run it was checking; a nonzero return is caught at the call
# site, which names the run.
dest_refused() { # rc output target -- asserts the startup-refusal shape for that run
  if [ "$1" -ne 2 ]; then
    printf 'sweep_harness: %s exited %s, not the startup refusal 2:\n%s\n' "$3" "$1" "$2" >&2
    return 1
  fi
  if [ -s "$ssh_calls" ]; then
    printf 'sweep_harness: %s contacted ssh before refusing:\n' "$3" >&2
    cat "$ssh_calls" >&2
    return 1
  fi
  absent "$3" "$state/history.tsv" || return 1
  absent '^run: ' <<<"$2" || return 1
  absent '^lanes:' <<<"$2" || return 1
}
: >"$ssh_calls"
set +e
dest_missing=$(SWEEP_TEST_HOSTS=$tmp/no-such-hosts.sh \
  run_sweep_args --only cuda --target dest-missing-probe 2>&1)
dest_missing_rc=$?
set -e
dest_refused "$dest_missing_rc" "$dest_missing" dest-missing-probe
grep -qF "sweep: cannot read the site host table $tmp/no-such-hosts.sh for rog's boot kind (set WAKE_LAB_HOSTS, or name the destination with OCANNL_TOOL_SWEEP_DEST_ROG)" \
  <<<"$dest_missing"
grep -q '^sweep: no ssh destination for rog-nv/cuda; refusing to guess one$' <<<"$dest_missing"
# ...and only a SELECTED remote unit needs one: a local-only run on a host with no site table (CI,
# a developer's machine) runs as before. The opposing control for the refusal above.
dest_local_only=$(SWEEP_TEST_HOSTS=$tmp/no-such-hosts.sh run_sweep_args --target dest-local-probe)
grep -q '^  m4-max/cc: incremental-pass ' <<<"$dest_local_only"
absent '^destinations:' <<<"$dest_local_only"
[ ! -s "$ssh_calls" ]
# A kind the sweep has no alias for.
set +e
dest_bogus=$(SWEEP_TEST_HOSTS=$tmp/hosts-bogus.sh \
  run_sweep_args --only hip --target dest-bogus-probe 2>&1)
dest_bogus_rc=$?
set -e
dest_refused "$dest_bogus_rc" "$dest_bogus" dest-bogus-probe
grep -qF "sweep: the site host table $tmp/hosts-bogus.sh gives no usable boot kind for minix (kind_of minix: 'win'; expected linux or wsl)" \
  <<<"$dest_bogus"
grep -q '^sweep: no ssh destination for minix/hip; refusing to guess one$' <<<"$dest_bogus"
# A table with no kind_of at all.
set +e
dest_no_kind_of=$(SWEEP_TEST_HOSTS=$tmp/hosts-no-kind-of.sh \
  run_sweep_args --only cuda --target dest-no-kind-of-probe 2>&1)
dest_no_kind_of_rc=$?
set -e
dest_refused "$dest_no_kind_of_rc" "$dest_no_kind_of" dest-no-kind-of-probe
grep -qF "gives no usable boot kind for rog (kind_of rog: '<none>'; expected linux or wsl)" \
  <<<"$dest_no_kind_of"
# ...even when the launcher exports a kind_of of its own: bash imports it into the sweep, and only
# the TABLE's function may answer. `env` passes the exported function through to the nested sweep.
kind_of() { echo linux; }
export -f kind_of
set +e
dest_inherited_kind_of=$(SWEEP_TEST_HOSTS=$tmp/hosts-no-kind-of.sh \
  run_sweep_args --only cuda --target dest-inherited-kind-of-probe 2>&1)
dest_inherited_kind_of_rc=$?
set -e
unset -f kind_of
dest_refused "$dest_inherited_kind_of_rc" "$dest_inherited_kind_of" dest-inherited-kind-of-probe
grep -qF "gives no usable boot kind for rog (kind_of rog: '<none>'; expected linux or wsl)" \
  <<<"$dest_inherited_kind_of"
# One box known and the other not: rog's lane must not have started, or the run would have swept
# the half it could reach and reported the rest in a row that reads like a sleeping box.
set +e
dest_half=$(SWEEP_TEST_HOSTS=$tmp/hosts-half.sh \
  run_sweep_args --only cuda --only hip --target dest-half-probe 2>&1)
dest_half_rc=$?
set -e
dest_refused "$dest_half_rc" "$dest_half" dest-half-probe
grep -q '^sweep: no ssh destination for minix/hip; refusing to guess one$' <<<"$dest_half"
# The per-run override names a destination outright, and needs no table for that box...
dest_override=$(SWEEP_TEST_HOSTS=$tmp/no-such-hosts.sh SWEEP_TEST_DEST_ROG=rog-nv-linux \
  run_sweep_args --only cuda --target dest-override-probe)
grep -q '^destinations: rog-nv=rog-nv-linux$' <<<"$dest_override"
grep -q '^  rog-nv/cuda: skip (unreachable)$' <<<"$dest_override"
grep -q ' rog-nv-linux ' "$ssh_calls"
# ...and wins over one that says otherwise, for its box only.
: >"$ssh_calls"
dest_override_wins=$(SWEEP_TEST_HOSTS=$tmp/hosts-wsl.sh SWEEP_TEST_DEST_ROG=rog-nv-linux \
  run_sweep_args --only cuda --only hip --target dest-override-wins-probe)
grep -q '^destinations: rog-nv=rog-nv-linux minix=minix-amd-wsl tuf=tuf-amd-linux$' <<<"$dest_override_wins"
absent ' rog-nv-wsl ' "$ssh_calls"
absent ' minix-amd-linux ' "$ssh_calls"
# An override must be one of ITS box's two canonical aliases, and anything else is refused: an
# alias of another box (the lane would reserve the wrong lock and leave this one open to a
# restart mid-unit), an option-shaped word, a `user@` or a second `@` (the lock is derived from
# the same string ssh parses, and every `@` spelling makes those readings disagree), and a custom
# alias, whose boot kind the `-wsl` PATH test downstream could not see.
for dest_bad in minix-amd-linux -oProxyCommand=true alice@rog-nv-linux \
  a@rog-nv-linux@elsewhere rog-lab; do
  : >"$ssh_calls"
  set +e
  dest_bad_override=$(SWEEP_TEST_DEST_ROG=$dest_bad \
    run_sweep_args --only cuda --target dest-bad-override-probe 2>&1)
  dest_bad_override_rc=$?
  set -e
  dest_refused "$dest_bad_override_rc" "$dest_bad_override" dest-bad-override-probe
  grep -qF "sweep: OCANNL_TOOL_SWEEP_DEST_ROG='$dest_bad' is not one of rog's aliases (rog-nv-linux or rog-nv-wsl)" \
    <<<"$dest_bad_override"
done

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

# The dxg window filter (gh-ocannl-979), driven directly: tools/kernel-window.sh is
# sourced by the sweep and by this harness for exactly that reason (after
# tools/box-jobs.sh, whose transport names pick a unit's window kind). The input is
# the 2026-09-15 evidence as the two boxes recorded it -- minix's benign boot
# lines at four different errnos, and one real lost message, which the bridge
# reports as a TRIPLE.
. "$(cd "$(dirname "$sweep")" && pwd)/box-jobs.sh"
. "$(cd "$(dirname "$sweep")" && pwd)/kernel-window.sh"
dxg_benign='Sep 15 09:00:01 box kernel: misc dxg: dxgk: dxgkio_is_feature_enabled: Ioctl failed: -22
Sep 15 09:00:01 box kernel: misc dxg: dxgk: dxgkio_query_adapter_info: Ioctl failed: -22
Sep 15 09:00:01 box kernel: misc dxg: dxgk: dxgkio_query_adapter_info: Ioctl failed: -2
Sep 15 09:00:01 box kernel: misc dxg: dxgk: dxgkio_query_adapter_info: Ioctl failed: -11
Sep 15 09:00:01 box kernel: misc dxg: dxgk: dxgkio_query_adapter_info: Ioctl failed: -1'
dxg_burst='Sep 15 09:05:00 box kernel: misc dxg: dxgk: dxgvmb_send_sync_msg: vmbus_sendpacket failed: fffffff5
Sep 15 09:05:00 box kernel: misc dxg: dxgk: create_existing_sysmem: failed set existing pages: fffffff5
Sep 15 09:05:00 box kernel: misc dxg: dxgk: dxgkio_create_allocation: Ioctl failed: -11'
dxg_unrelated='Sep 15 09:05:01 box kernel: hv_balloon: Max. dynamic memory size: 32 GB'

# Benign-only, at every errno the boxes have shown: nothing kept, no burst. An
# errno-keyed filter would have kept four of these five.
dxg_clean=$(printf '%s\n%s\n' "$dxg_benign" "$dxg_unrelated" |
  dxg_window_summary 20260915T090000Z 20260915T091000Z)
grep -q '^=== dxg window 20260915T090000Z..20260915T091000Z (utc) ===$' <<<"$dxg_clean"
grep -q '^=== dxg window: 0 vmbus_sendpacket failures ===$' <<<"$dxg_clean"
absent 'dxgkio_query_adapter_info' <<<"$dxg_clean"
absent 'dxgkio_is_feature_enabled' <<<"$dxg_clean"
absent 'hv_balloon' <<<"$dxg_clean"

# One lost message reported as three lines counts ONCE: keying on the fffffff5
# status instead would count the vmbus line and the sysmem line as two bursts.
# All three lines are kept and shown -- a new signature is what this exists to
# surface -- but only the vmbus one is counted.
dxg_red=$(printf '%s\n%s\n%s\n' "$dxg_benign" "$dxg_burst" "$dxg_unrelated" |
  dxg_window_summary 20260915T090000Z 20260915T091000Z)
grep -q '^=== dxg window: 1 vmbus_sendpacket failures ===$' <<<"$dxg_red"
[ "$(grep 'misc dxg' <<<"$dxg_red" | grep -cv '^dxg signature: ')" -eq 3 ]
grep -q 'create_existing_sysmem: failed set existing pages: fffffff5' <<<"$dxg_red"
absent 'dxgkio_query_adapter_info' <<<"$dxg_red"
# and the block carries the distinct signatures, which is what a fingerprint reads
[ "$(grep -c '^dxg signature: ' <<<"$dxg_red")" -eq 3 ]

# Two lost messages are two bursts, and an empty window is zero rather than one
# (the `grep -c` of an empty line).
[ "$(printf '%s\n%s\n' "$dxg_burst" "$dxg_burst" |
  dxg_window_summary A B | sed -n 's/^=== dxg window: \([0-9]*\) .*/\1/p')" = 2 ]
[ "$(printf '' | dxg_window_summary A B |
  sed -n 's/^=== dxg window: \([0-9]*\) .*/\1/p')" = 0 ]

# And the count the rest of the sweep reads back out of a unit's log is the one
# the filter wrote: the block goes into a log, `window_count` takes it out.
# Evidence is read from the collector's SIDECAR beside a unit's log, never from
# the log: a log holds whatever the unit's tests printed.
printf '%s\n' "$dxg_red" >"$(window_sidecar "$tmp/dxg-probe.log")"
[ "$(window_count "$tmp/dxg-probe.log")" = 1 ]
[ -z "$(window_count "$tmp/absent.log")" ]
# A block sitting in the LOG is not evidence -- that is the whole point of the
# sidecar -- so it neither counts nor makes the unit red.
printf '%s\n' "$dxg_red" >"$tmp/log-only.log"
[ -z "$(window_count "$tmp/log-only.log")" ]
if window_red "$tmp/log-only.log"; then
  printf 'sweep_harness: a dxg block in a unit log was read as evidence\n' >&2
  exit 1
fi

# A guest REPLACED during the window (2026-09-16). The journal query spans boots on purpose, so a
# window covering a VM death collects both -- and a new VM that came up clean contributes no dxg
# lines at all, which is how both GPU units of sweep 20260916T074913Z recorded `0
# vmbus_sendpacket failures` over machines that had ceased to exist under them. That clean reading
# was then cited as evidence against the VM having been the problem. The replacement has to take
# the verdict, because the verdict is what every other reader keys on.
dxg_replaced=$(printf '%s\n%s\n' "$dxg_benign" "$dxg_unrelated" |
  dxg_window_summary 20260915T090000Z 20260915T091000Z replaced)
grep -q '^=== dxg window: vm-replaced vmbus_sendpacket failures ===$' <<<"$dxg_replaced"
grep -q 'the guest was REPLACED during this window' <<<"$dxg_replaced"
# ...and a window that WOULD have read as a clean zero is exactly the case that must not.
absent '=== dxg window: 0 vmbus_sendpacket failures ===' <<<"$dxg_replaced"
# The lines and the count above it are still there to read: the replacement adds a finding, it does
# not throw the window away.
dxg_replaced_red=$(printf '%s\n%s\n' "$dxg_benign" "$dxg_burst" |
  dxg_window_summary 20260915T090000Z 20260915T091000Z replaced)
grep -q 'vmbus_sendpacket failed: fffffff5' <<<"$dxg_replaced_red"

# A replaced guest is environment-red: the machine the unit ran on is gone, so whatever the unit
# reported is not a judgement about the code, which is what the serial rerun exists to establish.
printf '%s\n' "$dxg_replaced" >"$(window_sidecar "$tmp/dxg-replaced.log")"
[ "$(window_count "$tmp/dxg-replaced.log")" = vm-replaced ]
window_red "$tmp/dxg-replaced.log"
# ...and it is a stable fingerprint line, so a unit red for it twice does not read as `fingerprint
# moved` while a unit that starts or stops losing its guest does.
window_fingerprint_lines "$tmp/dxg-replaced.log" | grep -q '^dxg window: guest replaced mid-window$'

# A failed collection over a guest KNOWN to have been replaced keeps the replacement verdict. The
# two coincide often -- a VM that has just been destroyed and recreated is exactly the one whose
# journal query fails -- so letting `unavailable` win would drop the stronger evidence in the case
# it was collected for.
dxg_lost=$(window_unavailable dxg 20260915T090000Z 20260915T091000Z "kernel log unreadable" replaced)
grep -q '^=== dxg window: vm-replaced vmbus_sendpacket failures ===$' <<<"$dxg_lost"
grep -q 'the guest was REPLACED during this window' <<<"$dxg_lost"
# ...while a failed collection that establishes nothing about the guest still says so.
dxg_lost_plain=$(window_unavailable dxg 20260915T090000Z 20260915T091000Z "kernel log unreadable")
grep -q "^=== dxg window: $WINDOW_UNAVAILABLE vmbus_sendpacket failures ===\$" <<<"$dxg_lost_plain"
absent 'REPLACED' <<<"$dxg_lost_plain"
# ...and so does one whose guest was known to have SURVIVED: `unavailable` is the right verdict
# there, and only `replaced` may override it.
dxg_lost_same=$(window_unavailable dxg A B "kernel log unreadable" same)
grep -q "^=== dxg window: $WINDOW_UNAVAILABLE vmbus_sendpacket failures ===\$" <<<"$dxg_lost_same"
# The stronger verdict survives the round trip through the sidecar, so the record and the rerun
# trigger read it rather than the collection failure.
printf '%s\n' "$dxg_lost" >"$(window_sidecar "$tmp/dxg-lost.log")"
[ "$(window_count "$tmp/dxg-lost.log")" = vm-replaced ]
window_red "$tmp/dxg-lost.log"

# The replaced-guest predicate the `error` path reads, which is a different question from
# "does this unit earn a rerun": an error never reached dune, so serial_rerun has no stanza to run.
window_guest_replaced "$tmp/dxg-replaced.log"
printf '%s\n' "$dxg_red" >"$(window_sidecar "$tmp/dxg-burst-only.log")"
if window_guest_replaced "$tmp/dxg-burst-only.log"; then
  printf 'sweep_harness: a plain burst was read as a replaced guest\n' >&2
  exit 1
fi
if window_guest_replaced "$tmp/absent.log"; then
  printf 'sweep_harness: a unit with no window was read as a replaced guest\n' >&2
  exit 1
fi

# `same` is the ordinary case and changes nothing; `unknown` is the pre-existing state under a
# name, for a box whose kernel publishes no boot id -- it must not become an alarm.
dxg_same=$(printf '%s\n' "$dxg_unrelated" |
  dxg_window_summary 20260915T090000Z 20260915T091000Z same)
grep -q '^=== dxg window: 0 vmbus_sendpacket failures ===$' <<<"$dxg_same"
absent 'REPLACED' <<<"$dxg_same"
dxg_unknown=$(printf '%s\n' "$dxg_unrelated" |
  dxg_window_summary 20260915T090000Z 20260915T091000Z unknown)
grep -q '^=== dxg window: 0 vmbus_sendpacket failures ===$' <<<"$dxg_unknown"
grep -q 'boot id could not be read' <<<"$dxg_unknown"
# The two-argument form is what the collector used before the boot check existed and what this
# harness calls everywhere above: it must keep meaning "nothing known about the guest".
dxg_legacy=$(printf '%s\n' "$dxg_unrelated" | dxg_window_summary A B)
grep -q '^=== dxg window: 0 vmbus_sendpacket failures ===$' <<<"$dxg_legacy"
absent 'REPLACED' <<<"$dxg_legacy"

# The remote reports the guest it is running as, on the same round trip as the bounds -- there is
# no second ssh to lose, and nothing else in the answer can tell one VM from its replacement.
window_cmd dxg 1757980800 | grep -q 'window-boot'
window_cmd dxg 1757980800 | grep -q '/proc/sys/kernel/random/boot_id'

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
grep -q '^lanes:  m4-max(cc,metal)  rog-nv(cuda)  minix(hip,multidev_cc)  tuf(hip)$' <<<"$lanes"
grep -q '^  m4-max/cc: incremental-pass ' <<<"$lanes"
grep -q '^  m4-max/metal: incremental-pass ' <<<"$lanes"
grep -q '^  rog-nv/cuda: skip (unreachable)$' <<<"$lanes"
grep -q '^  minix/hip: skip (unreachable)$' <<<"$lanes"
grep -q '^  minix/multidev_cc: skip (unreachable)$' <<<"$lanes"
[ ! -e "$tmp/lanes.overlap" ]

# The same ambient-leak control as the hostile backend above, for the lab lock directory, which is
# inherited the same way and fails far more quietly. A nested sweep that reached the REAL directory
# would reserve boxes a genuine sweep may be holding: its fake remote lanes would then wait out the
# lock budget and report `skip (box ... reserved by ...)` instead of the `skip (unreachable)` rows
# just asserted. $HOME above already moves the DEFAULT path, but an ambient WAKE_LAB_LOCK_DIR --
# the supported way to move that directory, and what the ludics-lite suite sets for its own
# hermeticity -- overrides the default, so two suites that are each hermetic alone would contend
# once run together.
#
# It has to select REMOTE units to prove anything: a lane reserves a box only when it has a host,
# so the same control over the two local units reserves nothing and passes whatever leaks. The
# ambient directory holds a genuinely HELD flock for both boxes, so a leak stops those lanes dead.
lab_leak=$tmp/ambient-lab-locks
mkdir -p "$lab_leak"
printf 'a genuine sweep (pid 1)\n' >"$lab_leak/rog.lock"
printf 'a genuine sweep (pid 1)\n' >"$lab_leak/minix.lock"
exec 6>>"$lab_leak/rog.lock"
exec 5>>"$lab_leak/minix.lock"
perl -e 'use Fcntl ":flock"; exit(flock(STDIN, LOCK_EX | LOCK_NB) ? 0 : 1)' <&6
perl -e 'use Fcntl ":flock"; exit(flock(STDIN, LOCK_EX | LOCK_NB) ? 0 : 1)' <&5
# Its OWN target: the history rows of this run are keyed by it, and reusing the lanes probe's
# target would add a second set of rows under that name and break the row assertion below.
lab_hostile=$(WAKE_LAB_LOCK_DIR=$lab_leak \
  SWEEP_TEST_WAIT_PREFIX=$tmp/lab-lanes SWEEP_TEST_SSH_MODE=release \
  run_sweep_args --only cc --only metal --only cuda --only hip --only multidev_cc \
  --target lab-lock-probe)
grep -q '^  rog-nv/cuda: skip (unreachable)$' <<<"$lab_hostile"
grep -q '^  minix/hip: skip (unreachable)$' <<<"$lab_hostile"
grep -q '^  minix/multidev_cc: skip (unreachable)$' <<<"$lab_hostile"
absent 'reserved by' <<<"$lab_hostile"
exec 6>&- 5>&-

# The OTHER lock file wake-lab keeps per box, and the lane must ignore it. `<box>.hold.lock` says
# "this box's VM must not be destroyed" -- what `wake-lab.sh --hold` takes and its Windows-side
# holder carries -- and it is deliberately not the lane's business: a hold keeps a VM alive FOR a
# lane, so a lane that waited on it would be waiting on its own caller. That is not hypothetical.
# Until 2026-09-18 both claims shared `<box>.lock`, and the cross-machine sweep routine -- which
# holds rog and minix with `--hold` and then runs this script in the same session -- reserved the
# boxes against its own sweep: every remote lane waited out LAB_LOCK_WAIT and skipped, and three of
# the five backends that routine is the only gate for got zero coverage (run 20260918T050903Z,
# ludics-lite#224). So: both hold locks genuinely HELD, and the lanes must still run to the point
# of probing their boxes -- `skip (unreachable)`, never `skip (box ... reserved by ...)`.
# The fixture goes in the lock directory `run_sweep_args` PINS, not one of its own: that helper
# builds the nested sweep's environment in full, so a `WAKE_LAB_LOCK_DIR=` prefix here would be
# discarded and the sweep would never see these files -- the case would then pass for the ordinary
# unreachable-box reason and pin nothing.
hold_locks=$tmp/lab-locks
mkdir -p "$hold_locks"
printf 'wake-lab --hold (pid 1, since 20260918T050814Z)\n' >"$hold_locks/rog.hold.lock"
printf 'wake-lab --hold (pid 1, since 20260918T050814Z)\n' >"$hold_locks/minix.hold.lock"
exec 6>>"$hold_locks/rog.hold.lock"
exec 5>>"$hold_locks/minix.hold.lock"
perl -e 'use Fcntl ":flock"; exit(flock(STDIN, LOCK_EX | LOCK_NB) ? 0 : 1)' <&6
perl -e 'use Fcntl ":flock"; exit(flock(STDIN, LOCK_EX | LOCK_NB) ? 0 : 1)' <&5
# Earlier cases in this file already reserved these boxes in this directory, so their lane locks
# are lying about. Remove them, or their mere presence afterwards would be evidence of nothing.
rm -f "$hold_locks/rog.lock" "$hold_locks/minix.lock"
hold_lock_ok=$(SWEEP_TEST_WAIT_PREFIX=$tmp/hold-lanes SWEEP_TEST_SSH_MODE=release \
  run_sweep_args --only cc --only metal --only cuda --only hip --only multidev_cc \
  --target hold-lock-probe)
grep -q '^  rog-nv/cuda: skip (unreachable)$' <<<"$hold_lock_ok"
grep -q '^  minix/hip: skip (unreachable)$' <<<"$hold_lock_ok"
grep -q '^  minix/multidev_cc: skip (unreachable)$' <<<"$hold_lock_ok"
absent 'reserved by' <<<"$hold_lock_ok"
# ...and the lane really did take its OWN lock while the hold lock was held, rather than reaching
# some other directory: the lane lock file is there, beside the hold lock nobody asked it about.
# One statement each: under errexit a failing LEFT side of an `&&` list is exempt, so the pair
# written as one AND-list would pass silently in exactly the case it exists to catch.
[ -e "$hold_locks/rog.lock" ]
[ -e "$hold_locks/minix.lock" ]
exec 6>&- 5>&-

# The rows are those of a serial run in everything but their order: one per
# unit, each under its own machine.
[ "$(awk -F '\t' '$7 == "lane-probe" { print $2 "/" $3 ":" $5 }' "$state/history.tsv" | sort)" = \
  "$(printf '%s\n' m4-max/cc:incremental-pass m4-max/metal:incremental-pass \
    minix/hip:skip minix/multidev_cc:skip rog-nv/cuda:skip tuf/hip:gate | sort)" ]

# The run record (gh-ocannl-977). A complete run's record names the exit kind,
# every selected unit's outcome with its lane's completion, and today's whole
# backend->box map -- the last one covering backends this run did not select, so
# a consumer can age a backend's staleness against the box that owns it now.
lanes_record=$(sed -n 's/^run:  *//p' <<<"$lanes")
[ -f "$lanes_record" ]
[ "$(head -1 "$lanes_record")" = "$(printf 'schema\t5')" ]
[ "$(awk -F '\t' '$1 == "run" { print $8 }' "$lanes_record")" = complete ]
[ "$(awk -F '\t' '$1 == "run" { print $5 "\t" $6 }' "$lanes_record")" = \
  "$(printf 'lane-probe\t0')" ]
[ "$(awk -F '\t' '$1 == "unit" { print $2 "/" $3 ":" $4 ":" $5 }' "$lanes_record" | sort)" = \
  "$(printf '%s\n' m4-max/cc:incremental-pass:0 m4-max/metal:incremental-pass:0 \
    minix/hip:skip:0 minix/multidev_cc:skip:0 rog-nv/cuda:skip:0 tuf/hip:gate:0 | sort)" ]
[ "$(awk -F '\t' '$1 == "backend" { print $2 ":" $3 }' "$lanes_record")" = \
  "$(printf '%s\n' cc:m4-max metal:m4-max cuda:rog-nv hip:minix multidev_cc:minix hip:tuf)" ]
# ...and the memory model each unit exercised: hip twice, once on each side of the unified/discrete
# line, which is what the second hip unit is for.
[ "$(awk -F '\t' '$1 == "unit" { print $2 "/" $3 ":" $11 }' "$lanes_record" | sort)" = \
  "$(printf '%s\n' m4-max/cc:- m4-max/metal:unified/apple minix/hip:unified/gfx1151 \
    minix/multidev_cc:- rog-nv/cuda:discrete/sm_120 tuf/hip:discrete/gfx1102 | sort)" ]
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

# The collection reads the journal ACROSS boots. `journalctl -k` is the same
# kernel match plus an implied `-b`, which truncates the answer at the current
# boot -- and a window spanning a VM death is the one case this collection exists
# for. Measured on minix (2026-09-15, three boots): `-k` returned 123 of the
# window's vmbus lines against 365 for the match, and the same implied `-b`
# reported zero dxg lines for rog-nv's 2026-09-13 window, which holds 255 and that
# unit's burst. Pinned on the emitted command because no fixture has a journal:
# this is the trap, not the spelling.
dxg_collection=$(window_cmd dxg 1757894400)
absent 'journalctl -k' <<<"$dxg_collection"
# The journal is selected only if it produces an ENTRY. A host with journalctl and
# no readable kernel journal exits 0 AND prints `-- No entries --` on stdout, so
# neither a status test nor a nonempty test falls back, and the burst still in the
# kernel ring is recorded as zero.
grep -q 'grep -qv "\^--"' <<<"$dxg_collection"
# Both ends bounded, on both branches: an event after the unit finished must not be
# attributed to it, and `dmesg -T` alone returns the whole current-boot ring, so an
# EARLIER unit's burst would buy this one a rerun it did not earn.
grep -q 'journalctl -q _TRANSPORT=kernel --since @\$window_start --until @\$window_end' \
  <<<"$dxg_collection"
grep -q 'dmesg -T --since @\$window_start --until @\$window_end' <<<"$dxg_collection"
# BOTH bounds belong to the clock that timestamps the log: the start is the
# instant the reachability probe read on that box when the unit began there, and
# the end is that box's clock at collection time. The controller and a WSL VM do
# not share a wall clock (these boxes resynchronise after host resumes), so a
# controller instant carried across -- or a start reconstructed by subtracting a
# locally measured duration, which assumes the remote clock advanced continuously
# meanwhile -- would put a terminal burst outside the window and lose the rerun.
# Evaluated by running the emitted command here, which is local shell and `date`.
# `|| true` on the capture, not on the command: the emitted command DELIBERATELY
# ends nonzero where it could not read a kernel log (this host has no journalctl
# and a dmesg that rejects the bounds), because collect_dxg_window reads that
# status as `unavailable` rather than as a clean window. Only the bounds line,
# printed before either branch runs, is under test here.
# Evaluated against FAKE kernel tools, never the host's own: on a box with a
# readable persistent journal -- either sweep box, and CI's Linux runner -- the
# emitted query would stream every kernel entry since the fixture's start epoch,
# which is a year-and-growing of them, to test two numbers printed before the
# query runs. The fakes also make the branch taken here independent of whatever
# the host happens to have.
cat >"$fake_bin/journalctl" <<'EOF'
#!/bin/sh
exit 0
EOF
cat >"$fake_bin/dmesg" <<'EOF'
#!/bin/sh
exit 0
EOF
chmod +x "$fake_bin/journalctl" "$fake_bin/dmesg"
dxg_now=$(date +%s)
dxg_bounds=$(PATH=$fake_bin:$PATH eval "$dxg_collection" 2>/dev/null |
  sed -n 's/^window-bounds \([0-9]*\) \([0-9]*\)$/\1 \2/p;/^window-bounds /q') || true
[ -n "$dxg_bounds" ]
# The start is the instant passed in, untouched -- not recomputed from anything.
[ "${dxg_bounds%% *}" = 1757894400 ]
# The end is the box's own clock plus the one second of slack that keeps a burst
# timestamped inside the collection's own second from falling outside the bound.
[ "$(( ${dxg_bounds##* } - dxg_now ))" -ge 1 ]
[ "$(( ${dxg_bounds##* } - dxg_now ))" -le 3 ]

# A window bigger than the shown cap still surfaces every distinct signature: the
# RAW lines are capped (496 in one minix boot, and the log is for reading) but the
# signature list is not, because a new kernel signature appearing at line 300 that
# neither shows nor moves the fingerprint is precisely the blindness this feature
# exists to remove.
dxg_many=$({ i=0; while [ "$i" -lt 60 ]; do printf '%s\n' "$dxg_burst"; i=$((i + 1)); done
  printf 'Sep 15 09:09:09 box kernel: misc dxg: dxgk: dxgkio_late_signature: Ioctl failed: -7\n'; } |
  dxg_window_summary 20260915T090000Z 20260915T091000Z)
[ "$(grep 'misc dxg' <<<"$dxg_many" | grep -cv '^dxg signature: ')" -eq 40 ]
grep -q '^(181 lines in the window; the first 40 are shown)$' <<<"$dxg_many"
grep -q '^dxg signature: misc dxg: dxgk: dxgkio_late_signature: Ioctl failed: -7$' <<<"$dxg_many"
[ "$(grep -c '^dxg signature: ' <<<"$dxg_many")" -eq 4 ]
grep -q '^=== dxg window: 60 vmbus_sendpacket failures ===$' <<<"$dxg_many"
# And that late signature moves the fingerprint, which is the point of keeping it.
printf '%s\n' "$dxg_many" >"$(window_sidecar "$tmp/dxg-many.log")"
printf '%s\n' "$({ i=0; while [ "$i" -lt 60 ]; do printf '%s\n' "$dxg_burst"; i=$((i + 1)); done; } |
  dxg_window_summary 20260915T090000Z 20260915T091000Z)" >"$(window_sidecar "$tmp/dxg-many-plain.log")"
[ "$(window_fingerprint_lines "$tmp/dxg-many.log")" != \
  "$(window_fingerprint_lines "$tmp/dxg-many-plain.log")" ]


# A collection that did not happen is not a clean window. `unavailable` is
# distinguishable from `0` everywhere it travels -- the block, the burst reader,
# and the run record -- because "nobody read the box" and "the bridge was fine"
# mean opposite things, and the second would lose the rerun.
dxg_unavailable=$(window_unavailable dxg 20260915T090000Z 20260915T091000Z 'ssh exit 255')
grep -q '^collection failed: ssh exit 255$' <<<"$dxg_unavailable"
grep -q '^=== dxg window: unavailable vmbus_sendpacket failures ===$' <<<"$dxg_unavailable"
printf '%s\n' "$dxg_unavailable" >"$(window_sidecar "$tmp/dxg-unavailable.log")"
[ "$(window_count "$tmp/dxg-unavailable.log")" = unavailable ]

# What the FINGERPRINT gets is the block's stable half. A fingerprint is compared
# bytewise against the previous failure's, and a standing environment red repeats:
# the window instants, the kernel timestamps and the count all differ between two
# equally broken runs (161 and 123 on minix within one hour), so the verbatim
# block would report `fingerprint moved` every time and cost the suppression that
# keeps sweep output actionable.
printf '%s\n' "$dxg_red" >"$(window_sidecar "$tmp/dxg-red-a.log")"
printf '%s\n' "$(printf '%s\n%s\n%s\n' "$dxg_benign" "$dxg_burst" "$dxg_burst" |
  dxg_window_summary 20260915T230000Z 20260915T234500Z)" >"$(window_sidecar "$tmp/dxg-red-b.log")"
[ "$(window_count "$tmp/dxg-red-a.log")" = 1 ]
[ "$(window_count "$tmp/dxg-red-b.log")" = 2 ]
# Different windows, different counts, same signatures: the fingerprint halves
# must be identical, or a standing red is reported as moving every run.
[ "$(window_fingerprint_lines "$tmp/dxg-red-a.log")" = \
  "$(window_fingerprint_lines "$tmp/dxg-red-b.log")" ]
grep -q '^dxg window: burst present$' <<<"$(window_fingerprint_lines "$tmp/dxg-red-a.log")"
absent '20260915T090000Z' <<<"$(window_fingerprint_lines "$tmp/dxg-red-a.log")"
# But a bridge that stops failing, or fails in a NEW way, still moves it.
printf '%s\n' "$dxg_clean" >"$(window_sidecar "$tmp/dxg-clean.log")"
[ "$(window_fingerprint_lines "$tmp/dxg-clean.log")" != \
  "$(window_fingerprint_lines "$tmp/dxg-red-a.log")" ]
grep -q '^dxg window: no burst$' <<<"$(window_fingerprint_lines "$tmp/dxg-clean.log")"
grep -q '^dxg window: collection unavailable$' \
  <<<"$(window_fingerprint_lines "$tmp/dxg-unavailable.log")"
printf '%s\n' "$(printf '%s\nSep 15 09:05:02 box kernel: misc dxg: dxgk: dxgkio_destroy_allocation: Ioctl failed: -9\n' \
  "$dxg_burst" | dxg_window_summary 20260915T090000Z 20260915T091000Z)" >"$(window_sidecar "$tmp/dxg-new-sig.log")"
[ "$(window_fingerprint_lines "$tmp/dxg-new-sig.log")" != \
  "$(window_fingerprint_lines "$tmp/dxg-red-a.log")" ]

# The kernel-evidence trigger (gh-ocannl-979), at the seam that decides it and in
# the one end-to-end direction a fixture can reach. A POSITIVE count only: a clean
# window is a finding the other way, an unavailable one establishes nothing in
# either direction, and absent evidence -- a local unit, or one that never ran --
# is not red. Without the negative legs the trigger would be "any unit that
# collected a window", which is not a trigger at all.
window_red "$tmp/dxg-probe.log"
for quiet in dxg-clean dxg-unavailable absent; do
  if window_red "$tmp/$quiet.log"; then
    printf 'sweep_harness: %s was read as environment-red\n' "$quiet" >&2
    exit 1
  fi
done

# End to end: a LOCAL unit whose test leg printed a complete burst block into its
# log gets no rerun, no dxg fingerprint lines and no record window. This harness
# itself dumps such fixtures on failure and runs as a test action inside a sweep
# unit, so the log of a local cc unit really can contain one; reading evidence out
# of the log would mark a unit that never touched /dev/dxg environment-red, buy it
# the expensive serial rerun, and report a bridge failure in its record row.
dxg_window_block=$(printf '%s\n%s\n' "$dxg_benign" "$dxg_burst" |
  dxg_window_summary 20260915T090000Z 20260915T091000Z)
dxg_no_trigger=$(SWEEP_TEST_OPAM_RC=1 \
  SWEEP_TEST_OPAM_OUT="$state_failure
$dxg_window_block" \
  run_sweep_backend cc --target state-probe)
grep -q 'm4-max/cc: fail ' <<<"$dxg_no_trigger"
absent 'serial rerun' <<<"$dxg_no_trigger"
absent 'environment-red' <<<"$dxg_no_trigger"
dxg_no_trigger_log=$(awk -F '\t' '$3 == "cc" { print $9 }' "$state/history.tsv" | tail -1)
grep -q '^=== dxg window: 1 vmbus_sendpacket failures ===$' "$dxg_no_trigger_log"
[ -z "$(window_count "$dxg_no_trigger_log")" ]
[ ! -e "$(window_sidecar "$dxg_no_trigger_log")" ]
absent '^dxg window: ' "${dxg_no_trigger_log%.log}.fingerprint"
absent '^dxg signature: ' "${dxg_no_trigger_log%.log}.fingerprint"
dxg_no_trigger_record=$(sed -n 's/^run:  *//p' <<<"$dxg_no_trigger")
[ "$(awk -F '\t' '$1 == "unit" && $3 == "cc" { print $7 "\t" $8 "\t" $9 "\t" $10 }' \
  "$dxg_no_trigger_record")" = "$(printf -- '-\t-\t-\t-')" ]

# What a collected window DOES put in the record and the fingerprint, driven
# through the same readers the sweep uses, with the sidecar a collection writes.
dxg_trigger_log=$tmp/collected-unit.log
printf 'fixture unit log\n' >"$dxg_trigger_log"
printf '%s\n' "$dxg_window_block" >"$(window_sidecar "$dxg_trigger_log")"
window_red "$dxg_trigger_log"
[ "$(window_count "$dxg_trigger_log")" = 1 ]
[ "$(window_bounds "$dxg_trigger_log")" = '20260915T090000Z 20260915T091000Z' ]
grep -q '^dxg window: burst present$' <<<"$(window_fingerprint_lines "$dxg_trigger_log")"
# A failed collection reaches the record as `unavailable` -- never as `-`, which
# is the unit nobody tried to read -- keeping the window it knows.
[ "$(window_bounds "$tmp/dxg-unavailable.log")" = '20260915T090000Z 20260915T091000Z' ]
[ "$(window_count "$tmp/dxg-unavailable.log")" = unavailable ]
# And where it failed before the box could report any bounds -- an unreachable
# probe, a box whose `date` said nothing -- the bounds are `-` and the count is
# still `unavailable`, which is what distinguishes it in the record from a unit
# that has no window because none was ever collected.
window_unavailable dxg - - 'no clock reading from rog-nv' \
  >"$(window_sidecar "$tmp/dxg-noclock.log")"
[ "$(window_bounds "$tmp/dxg-noclock.log")" = '- -' ]
[ "$(window_count "$tmp/dxg-noclock.log")" = unavailable ]
if window_red "$tmp/dxg-noclock.log"; then
  printf 'sweep_harness: a collection with no bounds was read as environment-red\n' >&2
  exit 1
fi

# ---- The native window (gh-ocannl-1034). A native boot has no bridge: its refusals are the GPU
# drivers' own, and the unit's TRANSPORT picks which window it reads -- the seam first. The kinds are
# the transports box_jobs_dest_transport names, so a destination is classified by its suffix alone,
# including a box the sweep's table does not have yet.
[ "$(window_kind_of rog-nv-wsl)" = dxg ]
[ "$(window_kind_of minix-amd-wsl)" = dxg ]
[ "$(window_kind_of rog-nv-linux)" = native ]
[ "$(window_kind_of minix-amd-linux)" = native ]
[ "$(window_kind_of tuf-amd-linux)" = native ]
# ...and one that names neither boot has NO window, which is `-` in the record, not `unavailable`.
[ -z "$(window_kind_of gpu-box)" ]
[ -z "$(window_kind_of '')" ]

# Real kernel lines, read from the native boots' journals (`journalctl _TRANSPORT=kernel`,
# 2026-09-23): minix's SDMA refusal pair from gh-ocannl-1029's ladder, and a quiet window's worth of
# what else a native kernel logs -- the nvme queue census and the amdgpu ring setup at every resume,
# a workqueue warning, rog-nv's NIC announcing its `XID 641` at boot, and the NVRM lines rog-nv
# logged in a driver upgrade's last minutes. The Xid event is the NVIDIA driver's documented form;
# no sweep box has logged one.
native_sdma='Sep 23 11:12:35 minix-amd-linux kernel: amdgpu 0000:c5:00.0: No more SDMA queue to allocate (8 total queues)'
native_dqm='Sep 23 11:12:35 minix-amd-linux kernel: amdgpu: process pid 85562 DQM create queue type 1 failed. ret -12'
native_xid='Sep 23 14:02:11 rog-nv-linux kernel: NVRM: Xid (PCI:0000:02:00): 79, pid=41234, name=tensor_puzzles.exe, GPU has fallen off the bus.'
native_quiet='Sep 23 10:54:54 minix-amd-linux kernel: nvme nvme0: 16/0/0 default/read/poll queues
Sep 23 10:54:54 minix-amd-linux kernel: amdgpu 0000:c5:00.0: ring sdma0 uses VM inv eng 12 on hub 0
Sep 23 21:23:55 minix-amd-linux kernel: workqueue: inode_switch_wbs_work_fn hogged CPU for >10000us 4 times, consider switching to WQ_UNBOUND
Sep 22 22:48:59 rog-nv-linux kernel: r8169 0000:82:00.0 eth0: RTL8125B, 48:21:0b:7c:5a:14, XID 641, IRQ 188
Sep 22 17:28:01 rog-nv-linux kernel: NVRM: VM: invalid mmap context'

# A quiet native window is a CLEAN one: zero, a positive finding -- not `unavailable`, which is what
# every native GPU unit recorded while it collected the dxg window. The drivers' own lines are kept
# and shown, the rest of the kernel log is not, and the words that merely LOOK like a refusal -- a
# ring named `sdma0`, a NIC's `XID` -- count for nothing.
native_quiet_block=$(printf '%s\n' "$native_quiet" |
  native_window_summary 20260923T111000Z 20260923T113000Z)
grep -q '^=== native window 20260923T111000Z..20260923T113000Z (utc) ===$' <<<"$native_quiet_block"
grep -q '^=== native window: 0 GPU queue refusals ===$' <<<"$native_quiet_block"
grep -q '^native signature: amdgpu 0000:c5:00.0: ring sdma0 uses VM inv eng 12 on hub 0$' \
  <<<"$native_quiet_block"
grep -q '^native signature: NVRM: VM: invalid mmap context$' <<<"$native_quiet_block"
absent 'nvme0' <<<"$native_quiet_block"
absent 'XID 641' <<<"$native_quiet_block"
absent 'workqueue' <<<"$native_quiet_block"
printf '%s\n' "$native_quiet_block" >"$(window_sidecar "$tmp/native-quiet.log")"
[ "$(window_count "$tmp/native-quiet.log")" = 0 ]
[ "$(window_kind "$tmp/native-quiet.log")" = native ]
grep -q '^native window: no refusal$' <<<"$(window_fingerprint_lines "$tmp/native-quiet.log")"
if window_red "$tmp/native-quiet.log"; then
  printf 'sweep_harness: a native window with no refusal was read as environment-red\n' >&2
  exit 1
fi

# Each refusal signature makes the unit environment-red. The SDMA pair is ONE refused queue -- the
# reason line, then KFD's refusal -- and counts once; either line alone still counts, so a driver
# that logs only one of them keeps the rerun. An Xid is one event.
native_count() { # kernel lines on stdin -> the count a native window records for them
  native_window_summary A B | sed -n 's/^=== native window: \([0-9]*\) .*/\1/p'
}
[ "$(printf '%s\n%s\n%s\n' "$native_quiet" "$native_sdma" "$native_dqm" | native_count)" = 1 ]
[ "$(printf '%s\n' "$native_dqm" | native_count)" = 1 ]
[ "$(printf '%s\n' "$native_sdma" | native_count)" = 1 ]
[ "$(printf '%s\n%s\n%s\n%s\n' "$native_sdma" "$native_dqm" "$native_sdma" "$native_dqm" |
  native_count)" = 2 ]
[ "$(printf '%s\n%s\n' "$native_quiet" "$native_xid" | native_count)" = 1 ]
[ "$(printf '%s\n%s\n%s\n' "$native_sdma" "$native_dqm" "$native_xid" | native_count)" = 2 ]
[ "$(printf '' | native_count)" = 0 ]
for native_case in sdma dqm xid; do
  native_line=native_$native_case
  printf '%s\n' "$(printf '%s\n%s\n' "$native_quiet" "${!native_line}" |
    native_window_summary 20260923T111000Z 20260923T113000Z)" \
    >"$(window_sidecar "$tmp/native-$native_case.log")"
  window_red "$tmp/native-$native_case.log"
  grep -q '^native window: refusal present$' \
    <<<"$(window_fingerprint_lines "$tmp/native-$native_case.log")"
done

# The fingerprint's half is stable across two equally refused runs: the window, the timestamps, the
# count and the PIDS all differ, and a pid in a signature would report `fingerprint moved` on every
# repeat of a standing SDMA red.
native_red_a=$(printf '%s\n%s\n' "$native_sdma" "$native_dqm" |
  native_window_summary 20260923T111000Z 20260923T113000Z)
native_red_b=$(printf '%s\n%s\n%s\n%s\n' "${native_sdma/11:12:35/15:40:02}" \
  "${native_dqm/85562/90210}" "$native_sdma" "${native_dqm/85562/90377}" |
  native_window_summary 20260923T153000Z 20260923T160000Z)
printf '%s\n' "$native_red_a" >"$(window_sidecar "$tmp/native-red-a.log")"
printf '%s\n' "$native_red_b" >"$(window_sidecar "$tmp/native-red-b.log")"
[ "$(window_count "$tmp/native-red-a.log")" = 1 ]
[ "$(window_count "$tmp/native-red-b.log")" = 2 ]
[ "$(window_fingerprint_lines "$tmp/native-red-a.log")" = \
  "$(window_fingerprint_lines "$tmp/native-red-b.log")" ]
grep -q '^native signature: amdgpu: process pid N DQM create queue type 1 failed. ret -12$' \
  <<<"$native_red_a"
absent '^native signature: .*85562' <<<"$native_red_a"
grep -q '^native signature: NVRM: Xid (PCI:0000:02:00): 79, pid=N, name=tensor_puzzles.exe, GPU has fallen off the bus.$' \
  <<<"$(printf '%s\n' "$native_xid" | native_window_summary A B)"
# `dmesg -T` stamps a line differently from the journal, and the signature is the same either way.
grep -q '^native signature: amdgpu: process pid N DQM create queue type 1 failed. ret -12$' \
  <<<"$(printf '[Wed Sep 23 11:12:35 2026] amdgpu: process pid 85562 DQM create queue type 1 failed. ret -12\n' |
    native_window_summary A B)"
# ...while a clean window and a refused one still differ.
[ "$(window_fingerprint_lines "$tmp/native-quiet.log")" != \
  "$(window_fingerprint_lines "$tmp/native-red-a.log")" ]

# The two signature sets are separate: a native window does not count a dxg burst -- nor keep it,
# since a bridge line cannot come from a native boot -- and a dxg window does not count an SDMA
# refusal.
native_of_dxg=$(printf '%s\n%s\n' "$dxg_benign" "$dxg_burst" | native_window_summary A B)
grep -q '^=== native window: 0 GPU queue refusals ===$' <<<"$native_of_dxg"
absent 'misc dxg' <<<"$native_of_dxg"
dxg_of_native=$(printf '%s\n%s\n%s\n' "$native_sdma" "$native_dqm" "$native_xid" |
  dxg_window_summary A B)
grep -q '^=== dxg window: 0 vmbus_sendpacket failures ===$' <<<"$dxg_of_native"
absent 'amdgpu' <<<"$dxg_of_native"

# A native collection that failed, and a native box that rebooted mid-unit, keep the verdicts the
# dxg kind has: `unavailable` establishes nothing, and `vm-replaced` -- the kernel's boot id, not the
# bridge's -- is environment-red.
window_unavailable native 20260923T111000Z 20260923T113000Z 'ssh exit 255' \
  >"$(window_sidecar "$tmp/native-unavailable.log")"
[ "$(window_count "$tmp/native-unavailable.log")" = unavailable ]
[ "$(window_kind "$tmp/native-unavailable.log")" = native ]
grep -q '^native window: collection unavailable$' \
  <<<"$(window_fingerprint_lines "$tmp/native-unavailable.log")"
if window_red "$tmp/native-unavailable.log"; then
  printf 'sweep_harness: an unavailable native window was read as environment-red\n' >&2
  exit 1
fi
printf '%s\n' "$native_quiet" | native_window_summary A B replaced \
  >"$(window_sidecar "$tmp/native-replaced.log")"
window_red "$tmp/native-replaced.log"
window_guest_replaced "$tmp/native-replaced.log"

# The no-window case: a unit with no sidecar has no kind, no bounds and no count, and the record
# writes `-` for all four. `unavailable` would claim a collection was tried.
printf 'fixture unit log\n' >"$tmp/no-window.log"
[ -z "$(window_kind "$tmp/no-window.log")" ]
[ -z "$(window_count "$tmp/no-window.log")" ]
[ -z "$(window_bounds "$tmp/no-window.log")" ]
[ -z "$(window_fingerprint_lines "$tmp/no-window.log")" ]

# The native QUERY. A native box restricts dmesg, so the dxg probe's fall-through for a window with
# no kernel entry -- the normal native window -- lands on a dmesg that refuses, and the collection
# fails: that is how every native GPU unit came to record `unavailable`. The native query judges the
# journal on the CURRENT BOOT instead, and then answers for the window whatever it holds. Run here
# against fakes shaped like a native box: a journal with this boot's kernel lines and none in the
# window, and a dmesg that refuses.
native_collection=$(window_cmd native 1758625800)
absent 'journalctl -k' <<<"$native_collection"
grep -q 'journalctl -q _TRANSPORT=kernel -b -n 1 --no-pager' <<<"$native_collection"
grep -q 'journalctl -q _TRANSPORT=kernel --since @\$window_start --until @\$window_end' \
  <<<"$native_collection"
grep -q 'dmesg -T --since @\$window_start --until @\$window_end' <<<"$native_collection"
native_box=$tmp/native-box-bin
mkdir -p "$native_box"
cat >"$native_box/journalctl" <<'EOF'
#!/bin/sh
case " $* " in
  *" -b "*) echo 'Sep 23 10:54:54 minix-amd-linux kernel: Linux version 6.17.0' ;;
esac
exit 0
EOF
cat >"$native_box/dmesg" <<'EOF'
#!/bin/sh
echo 'dmesg: read kernel buffer failed: Operation not permitted' >&2
exit 1
EOF
chmod +x "$native_box/journalctl" "$native_box/dmesg"
native_rc=0
PATH=$native_box:$PATH eval "$native_collection" >/dev/null 2>&1 || native_rc=$?
[ "$native_rc" -eq 0 ]
# ...where the dxg query, on the same box, fails -- the regression this replaces.
dxg_on_native_rc=0
PATH=$native_box:$PATH eval "$(window_cmd dxg 1758625800)" >/dev/null 2>&1 || dxg_on_native_rc=$?
[ "$dxg_on_native_rc" -ne 0 ]
# And a native box whose journal has no readable kernel log still falls back to dmesg, and a dmesg
# that refuses there fails the collection -- `unavailable`, never a clean zero.
cat >"$native_box/journalctl" <<'EOF'
#!/bin/sh
echo '-- No entries --'
exit 0
EOF
native_rc=0
PATH=$native_box:$PATH eval "$native_collection" >/dev/null 2>&1 || native_rc=$?
[ "$native_rc" -ne 0 ]

# End to end, through the sweep's own dispatch: a remote unit reached through a `-linux` alias
# collects the NATIVE window and records its kind, one reached through `-wsl` the dxg window, and a
# remote CPU unit none. The fake box answers the probes and the window query and refuses the
# preparation, so each unit ends as `error` -- a path that collects its window. Both kinds are fed
# the SAME kernel log, holding one refusal of each kind: each counts only its own.
native_kernel=$(printf '%s\n%s\n%s\n%s\n%s\n' "$native_quiet" "$dxg_benign" "$dxg_burst" \
  "$native_sdma" "$native_dqm")
: >"$ssh_calls"
native_unit_linux=$(SWEEP_TEST_SSH_MODE=window SWEEP_TEST_KERNEL_LINES=$native_kernel \
  run_sweep_args --only hip --target native-window-probe)
grep -q '^destinations: minix=minix-amd-linux tuf=tuf-amd-linux$' <<<"$native_unit_linux"
grep -q '^  minix/hip: error (cannot pin minix-amd-linux' <<<"$native_unit_linux"
native_unit_record=$(sed -n 's/^run:  *//p' <<<"$native_unit_linux")
[ "$(awk -F '\t' '$1 == "unit" && $2 == "minix" && $3 == "hip" { print $9 "\t" $10 }' "$native_unit_record")" = \
  "$(printf '1\tnative')" ]
native_unit_log=$(awk -F '\t' '$1 == "unit" && $2 == "minix" && $3 == "hip" { print $6 }' "$native_unit_record")
[ "$(window_kind "$native_unit_log")" = native ]
grep -q '^native signature: amdgpu: process pid N DQM create queue type 1 failed. ret -12$' \
  "$(window_sidecar "$native_unit_log")"
absent 'misc dxg' "$(window_sidecar "$native_unit_log")"
grep -q '^native window: refusal present$' "${native_unit_log%.log}.fingerprint"
grep -q -- '-b -n 1' "$ssh_calls"
# A native boot's work legs carry the sleep guard (gh-ocannl-1035): the preparation this fixture
# refuses went out as the supervisor's `--hold`, named for the run, the unit and the leg.
grep -qE -- "-- --hold 'ocannl sweep [0-9]{8}T[0-9]{6}Z minix/hip prep' 600 sh -c " "$ssh_calls"
: >"$ssh_calls"
native_unit_wsl=$(SWEEP_TEST_SSH_MODE=window SWEEP_TEST_KERNEL_LINES=$native_kernel \
  SWEEP_TEST_HOSTS=$tmp/hosts-wsl.sh run_sweep_args --only hip --target native-window-probe)
grep -q '^destinations: minix=minix-amd-wsl tuf=tuf-amd-linux$' <<<"$native_unit_wsl"
native_wsl_record=$(sed -n 's/^run:  *//p' <<<"$native_unit_wsl")
[ "$(awk -F '\t' '$1 == "unit" && $2 == "minix" && $3 == "hip" { print $9 "\t" $10 }' "$native_wsl_record")" = \
  "$(printf '1\tdxg')" ]
native_wsl_log=$(awk -F '\t' '$1 == "unit" && $2 == "minix" && $3 == "hip" { print $6 }' "$native_wsl_record")
absent 'amdgpu' "$(window_sidecar "$native_wsl_log")"
absent -- '-b -n 1' "$ssh_calls"
# ...and a WSL boot's do not: a guest's inhibitor cannot stop its Windows host sleeping, and the
# Windows-side holder is what keeps that lane alive. The opposing control for the native case.
absent -- "-- --hold '" "$ssh_calls"
# The remote CPU unit on the same native box: no window, so `-` in all four fields, and no window
# query sent at all.
: >"$ssh_calls"
native_unit_cpu=$(SWEEP_TEST_SSH_MODE=window SWEEP_TEST_KERNEL_LINES=$native_kernel \
  run_sweep_args --only multidev_cc --target native-window-probe)
grep -q '^  minix/multidev_cc: error ' <<<"$native_unit_cpu"
native_cpu_record=$(sed -n 's/^run:  *//p' <<<"$native_unit_cpu")
[ "$(awk -F '\t' '$1 == "unit" && $3 == "multidev_cc" { print $7 "\t" $8 "\t" $9 "\t" $10 }' \
  "$native_cpu_record")" = "$(printf -- '-\t-\t-\t-')" ]
absent 'window-bounds' "$ssh_calls"

# The ROCr abort a refused SDMA queue ends in lands in the unit's LOG, and makes the unit
# environment-red there too -- the half that survives a window that could not be read. Driven
# through a local unit's log with a failing fake, since this arm reads the log; the
# `Fatal error: exception` arm is covered by the serial-rerun cases above.
native_abort='schedule_conv_gemm.exe: ./runtime/hsa-runtime/core/runtime/amd_gpu_agent.cpp:2003: virtual void rocr::AMD::GpuAgent::ReleaseQueueMainScratch(rocr::AMD::ScratchInfo&): Assertion `scratch.main_queue_base'"'"' failed.'
native_abort_run=$(SWEEP_TEST_OPAM_RC=1 SWEEP_TEST_OPAM_OUT="$state_failure
$native_abort" run_sweep_backend cc --target state-probe)
grep -q 'm4-max/cc: fail ' <<<"$native_abort_run"
grep -q 'serial rerun' <<<"$native_abort_run"
# ...but an unrelated assertion in the same runtime is a test's own failure, not the environment's.
native_other_abort=$(SWEEP_TEST_OPAM_RC=1 SWEEP_TEST_OPAM_OUT="$state_failure
${native_abort/ReleaseQueueMainScratch/AcquireQueueScratch}" run_sweep_backend cc --target state-probe)
grep -q 'm4-max/cc: fail ' <<<"$native_other_abort"
absent 'serial rerun' <<<"$native_other_abort"


# ---- The gated tuf lane and the sleep guard (gh-ocannl-1035). tuf is the fleet's discrete-memory
# hip box, a Wi-Fi laptop nothing the caller runs can wake: its lane runs only when `wake-lab.sh
# status tuf` reaches its Linux, records `gate` -- never `skip`, never `error` -- when it does not,
# and ends by asking wake-lab to sleep the box again. The fake wake-lab above answers both verbs.
#
# Asleep (the fake's default): a gate for its unit, with the status reading as the reason, and
# nothing else -- no ssh to the box, no reservation, no sleep. minix's hip unit beside it is the
# opposing control: an ungated box's unit is dialled as always.
: >"$ssh_calls"
: >"$wake_lab_calls"
rm -f "$tmp/lab-locks/tuf.lock"
tuf_asleep=$(SWEEP_TEST_SSH_MODE=window run_sweep_args --only hip --target tuf-asleep-probe)
grep -q '^lanes:  minix(hip)  tuf(hip)$' <<<"$tuf_asleep"
grep -q '^  tuf/hip: gate (tuf not up: router-active=? os=-- linux=--)$' <<<"$tuf_asleep"
grep -q '^  minix/hip: error (cannot pin minix-amd-linux' <<<"$tuf_asleep"
absent 'tuf-amd-linux' "$ssh_calls"
grep -q 'minix-amd-linux' "$ssh_calls"
[ "$(cat "$wake_lab_calls")" = 'status tuf' ]
[ ! -e "$tmp/lab-locks/tuf.lock" ]
tuf_asleep_record=$(sed -n 's/^run:  *//p' <<<"$tuf_asleep")
[ "$(awk -F '\t' '$1 == "run" { print $8 }' "$tuf_asleep_record")" = complete ]
[ "$(awk -F '\t' '$1 == "unit" && $2 == "tuf" { print $3 ":" $4 ":" $6 ":" $10 ":" $11 }' \
  "$tuf_asleep_record")" = 'hip:gate:-:-:discrete/gfx1102' ]
[ "$(awk -F '\t' '$2 == "tuf" && $7 == "tuf-asleep-probe" { print $5 ":" $6 }' "$state/history.tsv")" = \
  'gate:0' ]
# A host with no wake-lab.sh cannot ask, and says that instead of guessing the box is up.
: >"$wake_lab_calls"
tuf_no_wake_lab=$(SWEEP_TEST_WAKE_LAB=$tmp/no-such-wake-lab.sh run_sweep_args --only hip \
  --target tuf-asleep-probe)
grep -qF "  tuf/hip: gate (tuf not asked: no wake-lab.sh at $tmp/no-such-wake-lab.sh)" \
  <<<"$tuf_no_wake_lab"
[ ! -s "$wake_lab_calls" ]

# Up: the unit is dialled at tuf's only alias, its window is the native kind by that alias alone,
# its record carries the memory model, every work leg carries the guard -- and when the lane is
# done the box is put back to sleep, which the fake grants only once the lane has let go of its
# own reservation.
: >"$ssh_calls"
: >"$wake_lab_calls"
tuf_up=$(SWEEP_TEST_TUF_STATUS=up SWEEP_TEST_SSH_MODE=window SWEEP_TEST_KERNEL_LINES=$native_quiet \
  run_sweep_args --only hip --target tuf-up-probe)
grep -q '^  tuf/hip: error (cannot pin tuf-amd-linux' <<<"$tuf_up"
grep -q '^  tuf: put back to sleep (wake-lab.sh sleep tuf)$' <<<"$tuf_up"
# The sleep comes after the lane's unit, never between units.
[ "$(grep -n '^  tuf/hip: ' <<<"$tuf_up" | head -1 | cut -d: -f1)" -lt \
  "$(grep -n '^  tuf: put back' <<<"$tuf_up" | cut -d: -f1)" ]
[ "$(cat "$wake_lab_calls")" = "$(printf 'status tuf\nsleep tuf')" ]
grep -q '^ocannl sweep ' "$tmp/lab-locks/tuf.lock"
grep -qE -- "-- --hold 'ocannl sweep [0-9]{8}T[0-9]{6}Z tuf/hip prep' 600 sh -c " "$ssh_calls"
tuf_up_record=$(sed -n 's/^run:  *//p' <<<"$tuf_up")
[ "$(awk -F '\t' '$1 == "unit" && $2 == "tuf" { print $3 ":" $4 ":" $10 ":" $11 }' \
  "$tuf_up_record")" = 'hip:error:native:discrete/gfx1102' ]
# The fake's refusal is real, so the grant above means something: with the box's lane lock held,
# as the lane held it until just before asking, the same call is refused.
exec 6>>"$tmp/lab-locks/tuf.lock"
perl -e 'use Fcntl ":flock"; exit(flock(STDIN, LOCK_EX | LOCK_NB) ? 0 : 1)' <&6
set +e
tuf_self_refusal=$(SWEEP_TEST_WAKE_LAB_CALLS=$wake_lab_calls WAKE_LAB_LOCK_DIR=$tmp/lab-locks \
  "$fake_bin/wake-lab.sh" sleep tuf)
tuf_self_refusal_rc=$?
set -e
exec 6>&-
[ "$tuf_self_refusal_rc" -eq 1 ]
grep -q '^  sleep REFUSED on tuf: ' <<<"$tuf_self_refusal"

# Up at the status check but not answering the unit (its Wi-Fi dropped, it slept again): still a
# gate, not the `skip` that asks why a wake failed -- and no sleep, since nothing reached it.
: >"$wake_lab_calls"
tuf_unreachable=$(SWEEP_TEST_TUF_STATUS=up run_sweep_args --only hip --target tuf-up-probe)
grep -q '^  tuf/hip: gate (unreachable)$' <<<"$tuf_unreachable"
grep -q '^  minix/hip: skip (unreachable)$' <<<"$tuf_unreachable"
absent '^  tuf: ' <<<"$tuf_unreachable"
[ "$(cat "$wake_lab_calls")" = 'status tuf' ]

# A sleep the box refuses -- a block inhibitor, another run there -- is the interlock working: the
# box stays up, the holder is named, and the run is as complete as it was.
tuf_inhibited=$(SWEEP_TEST_TUF_STATUS=up SWEEP_TEST_TUF_SLEEP=inhibited SWEEP_TEST_SSH_MODE=window \
  run_sweep_args --only hip --target tuf-up-probe)
grep -qF '  tuf: left awake, not a failure -- sleep REFUSED on tuf by a block inhibitor (a run there holds it; see status) (Operation inhibited by "fleet-worker" (PID 4242 "python3", user lukstafi), reason is "a correctness slot".)' \
  <<<"$tuf_inhibited"
[ "$(awk -F '\t' '$1 == "run" { print $8 }' "$(sed -n 's/^run:  *//p' <<<"$tuf_inhibited")")" = \
  complete ]
# ...and one that fails outright is a WARNING for the operator, still not a sweep failure.
tuf_sleep_fails=$(SWEEP_TEST_TUF_STATUS=up SWEEP_TEST_TUF_SLEEP=fail SWEEP_TEST_SSH_MODE=window \
  run_sweep_args --only hip --target tuf-up-probe)
grep -qF '  tuf: WARNING -- wake-lab.sh sleep tuf exited 1: sleep FAILED on tuf (command exited 5)' \
  <<<"$tuf_sleep_fails"
[ "$(awk -F '\t' '$1 == "run" { print $8 }' "$(sed -n 's/^run:  *//p' <<<"$tuf_sleep_fails")")" = \
  complete ]

# A native unit whose far side could not take the guard (no polkit grant) ran anyway, and says so
# on its summary line -- for each native unit, and for no WSL one, whose legs never ask.
tuf_unguarded=$(SWEEP_TEST_TUF_STATUS=up SWEEP_TEST_HOLD_DENIED=1 SWEEP_TEST_SSH_MODE=window \
  run_sweep_args --only hip --target tuf-up-probe)
grep -qF '  tuf/hip: ran WITHOUT a sleep guard -- fixture-inhibit refused: Access denied' \
  <<<"$tuf_unguarded"
grep -qF '  minix/hip: ran WITHOUT a sleep guard -- fixture-inhibit refused: Access denied' \
  <<<"$tuf_unguarded"
tuf_unguarded_wsl=$(SWEEP_TEST_HOLD_DENIED=1 SWEEP_TEST_SSH_MODE=window \
  SWEEP_TEST_HOSTS=$tmp/hosts-wsl.sh run_sweep_args --only hip --target tuf-up-probe)
absent 'sleep guard' <<<"$tuf_unguarded_wsl"


# A lane that reached tuf and was then CANCELLED still asks for tuf's sleep -- detached, so the
# group TERM that cancelled the run does not take the request with it, and with the lane's box
# reservation already released, so the fake's lock check grants it. Both lanes are held in their
# preparation ssh (the `hang` fake) and the whole group is signalled, as a scheduler cancel does.
tuf_cancel_prefix=$tmp/tuf-cancel
wait_prefix=$tuf_cancel_prefix
: >"$wake_lab_calls"
SWEEP_TEST_OWN_GROUP=1 SWEEP_TEST_WAIT_PREFIX=$tuf_cancel_prefix SWEEP_TEST_SSH_MODE=hang \
  SWEEP_TEST_TUF_STATUS=up run_sweep_args --only hip --target tuf-cancel-probe \
  >"$tuf_cancel_prefix.out" 2>"$tuf_cancel_prefix.err" &
tuf_cancel_pid=$!
holder_pid=$tuf_cancel_pid
waited=0
until [ "$(wc -l <"$tuf_cancel_prefix.ssh-pids" 2>/dev/null || echo 0)" -ge 2 ]; do
  [ "$waited" -lt "$wait_ticks" ] || break
  sleep 0.05
  waited=$((waited + 1))
done
[ "$(wc -l <"$tuf_cancel_prefix.ssh-pids")" -ge 2 ]
kill -TERM -- "-$tuf_cancel_pid"
set +e
wait "$tuf_cancel_pid"
tuf_cancel_rc=$?
set -e
holder_pid=
wait_prefix=
[ "$tuf_cancel_rc" -eq 143 ]
waited=0
until grep -q '^sleep tuf$' "$wake_lab_calls"; do
  [ "$waited" -lt "$wait_ticks" ] || break
  sleep 0.05
  waited=$((waited + 1))
done
tuf_cancelled=$(cat "$tuf_cancel_prefix.out" "$tuf_cancel_prefix.err")
[ "$(cat "$wake_lab_calls")" = "$(printf 'status tuf\nsleep tuf')" ]
# The request's durable record is its own log beside the run's, named for the run's stamp; the
# lane's summary line saying so is best-effort on this path (a cancelled top level does not wait to
# publish it), so the log is what is asserted.
tuf_cancel_stamp=$(sed -n 's/^sweep \([0-9TZ]*\) .*/\1/p' "$tuf_cancel_prefix.out")
tuf_cancel_sleep_log=$state/logs/$tuf_cancel_stamp-tuf-sleep.log
waited=0
until grep -q '^tuf=DOWN' "$tuf_cancel_sleep_log" 2>/dev/null; do
  [ "$waited" -lt "$wait_ticks" ] || break
  sleep 0.05
  waited=$((waited + 1))
done
grep -q '^tuf=DOWN' "$tuf_cancel_sleep_log"

# The guard itself: the far-side supervisor's `--hold`, run here against fake inhibitors. The
# program is the sweep's own, extracted as tools/test-test-run.sh extracts unit_jobs (the sweep
# cannot be sourced), and asserted to have matched.
supervisor=$tmp/supervisor.pl
sed -n "/^capped_perl='/,/^'\$/p" "$sweep" | sed '1s/^capped_perl=.//;$d' >"$supervisor"
grep -q 'sub hold_sleep' "$supervisor"
guard_bin=$tmp/guard-bin
mkdir -p "$guard_bin" "$tmp/guard-empty"
cat >"$guard_bin/systemd-inhibit" <<EOF
#!/bin/sh
printf '%s\n' "\$*" >"$tmp/inhibit.args"
echo "\$\$" >"$tmp/inhibit.pid"
while [ "\$1" != -- ]; do shift; done
shift
exec "\$@"
EOF
chmod +x "$guard_bin/systemd-inhibit"
# Held: the inhibitor is taken before the unit starts and is alive while it runs (the unit checks),
# with the logind fields the lab reads; the unit's own exit status is the supervisor's; and the
# inhibitor ends with the unit's tree, not after some timeout.
set +e
guard_held=$(PATH=$guard_bin:$PATH perl "$supervisor" --hold 'ocannl sweep S tuf/hip suite' 30 \
  sh -c "kill -0 \$(cat '$tmp/inhibit.pid') && echo inhibitor-alive-during-unit; exit 7" 2>&1)
guard_held_rc=$?
set -e
[ "$guard_held_rc" -eq 7 ]
grep -q '^inhibitor-alive-during-unit$' <<<"$guard_held"
grep -q '^sweep-hold: sleep:idle block inhibitor held for: ocannl sweep S tuf/hip suite$' <<<"$guard_held"
[ "$(cat "$tmp/inhibit.args")" = \
  '--what=sleep:idle --mode=block --who=ocannl-sweep --why=ocannl sweep S tuf/hip suite -- sh -c echo HELD; exec cat >/dev/null' ]
guard_waited=0
while kill -0 "$(cat "$tmp/inhibit.pid")" 2>/dev/null; do
  [ "$guard_waited" -lt "$wait_ticks" ] || break
  sleep 0.05
  guard_waited=$((guard_waited + 1))
done
if kill -0 "$(cat "$tmp/inhibit.pid")" 2>/dev/null; then
  printf 'sweep_harness: the sleep inhibitor outlived its unit\n' >&2
  exit 1
fi
# Refused (the polkit grant missing): loud, and the unit still runs with its own status.
cat >"$guard_bin/systemd-inhibit" <<'EOF'
#!/bin/sh
echo 'Failed to inhibit: Access denied' >&2
exit 1
EOF
set +e
guard_refused=$(PATH=$guard_bin:$PATH perl "$supervisor" --hold why 30 sh -c 'echo unit-ran; exit 3' 2>&1)
guard_refused_rc=$?
set -e
[ "$guard_refused_rc" -eq 3 ]
grep -q '^unit-ran$' <<<"$guard_refused"
grep -q '^sweep-hold: WARNING: running WITHOUT a sleep guard, .* -- .*/systemd-inhibit refused: Failed to inhibit: Access denied$' \
  <<<"$guard_refused"
# No systemd-inhibit at all: the same, said differently. A PATH holding nothing, so a host that has
# one (a Linux CI runner) cannot answer for the fixture.
set +e
guard_absent=$(PATH=$tmp/guard-empty "$(command -v perl)" "$supervisor" --hold why 30 \
  /bin/sh -c 'echo unit-ran; exit 4' 2>&1)
guard_absent_rc=$?
set -e
[ "$guard_absent_rc" -eq 4 ]
grep -q '^sweep-hold: WARNING: running WITHOUT a sleep guard, .* -- no systemd-inhibit on PATH$' \
  <<<"$guard_absent"
# And without `--hold` -- every local unit, every WSL leg -- nothing is asked for at all, with an
# inhibitor right there on PATH.
rm -f "$tmp/inhibit.args"
cat >"$guard_bin/systemd-inhibit" <<EOF
#!/bin/sh
printf '%s\n' "\$*" >"$tmp/inhibit.args"
exit 1
EOF
set +e
guard_none=$(PATH=$guard_bin:$PATH perl "$supervisor" 30 sh -c 'exit 5' 2>&1)
guard_none_rc=$?
set -e
[ "$guard_none_rc" -eq 5 ]
[ -z "$guard_none" ]
[ ! -e "$tmp/inhibit.args" ]

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
  # One per statement, per the rule at the top: this is the assertion that the loop above ended
  # because both files appeared rather than because it ran out of ticks, and as an `&&` list a
  # missing `.ready` -- the left operand -- was exempt from errexit, so a sweep that never got
  # ready was cancelled anyway and whatever the cancel then observed was read as the real thing.
  [ -e "$prefix.ready" ]
  [ -e "$prefix.ssh-running" ]
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

printf 'sweep execution accounting, RTC context, dxg window evidence, fingerprinting, run record and skip aggregation: PASS\n'
