#!/usr/bin/env bash
# Shared child waits: preserve status, recheck returning traps, and bound Bash's
# wait-entry signal race under fork/exit load, against the former bare wait.
# Usage: scripts/test-process-group.sh [--only REGEX] [--keep] [--list-legs]
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
. "$HERE/harness-support.sh"
. "$HERE/process-group.sh"
harness_args "$@"
harness_scratch process-group
harness_require perl python3 /bin/bash
race_driver= driver_cancelled=0
cleanup_fixture() {
  local live_jobs
  if [ -n "$race_driver" ]; then
    # The driver may have ended just before the wait returned. Signal only
    # while this shell still owns a running/stopped job with that pid.
    live_jobs=$(jobs -pr; jobs -ps)
    case $'\n'"$live_jobs"$'\n' in
      *$'\n'"$race_driver"$'\n'*) kill -TERM "$race_driver" 2>/dev/null || : ;;
    esac
    wait_child "$race_driver" 2>/dev/null || :
    race_driver=
  fi
}

driver_signal() {
  driver_cancelled=$1
  [ -z "$race_driver" ] || exit "$driver_cancelled"
}
driver_run() { # LOG ARGS...; status in driver_rc
  local log=$1
  shift
  # Latch cancellation until the fork has published the pid cleanup owns.
  driver_cancelled=0
  trap 'driver_signal 130' INT
  trap 'driver_signal 143' TERM
  python3 "$HERE/../test/support/wait_signal_race.py" "$@" >"$log" 2>&1 &
  race_driver=$!
  [ "$driver_cancelled" = 0 ] || exit "$driver_cancelled"
  wait_child "$race_driver"
  driver_rc=$?
  race_driver=
  trap 'exit 130' INT
  trap 'exit 143' TERM
  [ "$driver_cancelled" = 0 ] || exit "$driver_cancelled"
}

status_leg() {
  /bin/bash -c '
    . "$1"
    (exit 37) & child=$!
    wait_child "$child"; rc=$?
    [ "$rc" = 37 ] || exit 1
    # A trap returning normally must not replace the eventual child status.
    trap : TERM
    (sleep .2; exit 43) & child=$!
    perl -e "select undef,undef,undef,.05; kill 15, \$ARGV[0]" "$$" & sender=$!
    wait_child "$child"; rc=$?
    wait "$sender"
    [ "$rc" = 43 ]
  ' wait-status "$HERE/process-group.sh" >"$TMP/status" 2>&1
  report "$?" 'child status survives normal exit and a returning signal trap' "$(cat "$TMP/status")"
}
harness_leg_run '1: child status survives a returning trap' status_leg

driver_available() {
  local platform
  platform=$(python3 -c 'import os; print(os.name)') || { harness_abandon 'cannot inspect the stress driver platform'; return 1; }
  if [ "$platform" != posix ]; then
    skip "$1" 'Python has no POSIX FIFOs/process groups on this host'
    return 1
  fi
}
race_leg() {
  driver_available 'wait-entry stress driver' || return
  driver_run "$TMP/race" "$HERE/process-group.sh" "${OCANNL_TOOL_SIGNAL_RACE_READY:-}"
  local rc=$driver_rc
  cat "$TMP/race"
  report "$rc" 'shared wait handles TERM promptly under load; unbounded wait loses the oracle' "$(cat "$TMP/race")"
}
harness_leg_run '2: wait-entry signal race under load and negative control' race_leg
cancel_leg() {
  driver_available 'driver cancellation fixture' || return
  driver_run "$TMP/cancellation" --check-cancellation "$HERE/test-process-group.sh"
  cat "$TMP/cancellation"
  report "$driver_rc" 'harness cancellation forwards TERM and reaps the driver; foreground twin fails' "$(cat "$TMP/cancellation")"
}
harness_leg_run '3: cancellation reaches and reaps the stress driver' cancel_leg
finish
