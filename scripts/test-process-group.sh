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
    live_jobs=$(trap '' INT TERM HUP; jobs -pr; jobs -ps)
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

write_status_probe() {
  cat >"$TMP/status-probe.sh" <<'PROBE'
set -u
. "$1"
if [ "$2" = bare ]; then wait_child() { wait "$1"; }; fi
child= blocked=0
cleanup() {
  local live
  if [ -n "$child" ]; then
    live=$(trap '' INT TERM HUP; builtin jobs -pr; builtin jobs -ps)
    case $'\n'"$live"$'\n' in
      *$'\n'"$child"$'\n'*) kill -KILL "$child" 2>/dev/null || : ;;
    esac
    builtin wait "$child" 2>/dev/null || :
  fi
}
trap cleanup EXIT
wait() {
  local live
  live=$(trap '' INT TERM HUP; builtin jobs -pr; builtin jobs -ps)
  case $'\n'"$live"$'\n' in
    *$'\n'"$1"$'\n'*)
      blocked=1
      echo 'refused blocking wait on a live job' >&2
      return 99 ;;
  esac
  builtin wait "$@"
}
check_child() {
  local expected=$1 rc
  wait_child "$child"; rc=$?
  [ "$blocked" = 0 ] && [ "$rc" = "$expected" ] || exit 1
  child=
}
if [ "$2" = query ]; then
  # Ignore TERM at fork, so the child is immune even before it runs its first
  # command. The parent's returning trap relays TERM to that live child.
  trap '' TERM
  (sleep .3; exit 41) & child=$!
  relayed=0
  trap 'relayed=1; kill -TERM "$child" 2>/dev/null || :' TERM
  # A file survives command substitution: inject once, at the query itself.
  query_marker=$3
  jobs() {
    if [ ! -e "$query_marker" ]; then
      : >"$query_marker"
      kill -TERM 0
    fi
    builtin jobs "$@"
  }
  check_child 41
  [ "$relayed" = 1 ] && [ -e "$query_marker" ]
  exit $?
fi
(exit 37) & child=$!
check_child 37
# Long enough that the shadowed wait deterministically sees a live job if
# wait_child regresses to a bare wait; timing never decides this verdict.
(sleep .3; exit 41) & child=$!
check_child 41
trap : TERM
(sleep .2; exit 43) & child=$!
perl -e 'select undef,undef,undef,.05; kill 15, $ARGV[0]' "$$" & sender=$!
check_child 43
builtin wait "$sender"
perl -e '$SIG{TERM}="DEFAULT"; kill 15, $$; exit 99' & child=$!
check_child 143
wait_child "$$" 2>/dev/null; rc=$?
[ "$rc" = 127 ] && [ "$blocked" = 0 ]
PROBE
}
status_leg() {
  write_status_probe
  /bin/bash "$TMP/status-probe.sh" "$HERE/process-group.sh" shared >"$TMP/status" 2>&1
  report "$?" 'only completed jobs reach wait; child status is 37/41/43/143, non-child is 127' "$(cat "$TMP/status")"
  local rc=0
  /bin/bash "$TMP/status-probe.sh" "$HERE/process-group.sh" bare >"$TMP/status-bare" 2>&1 || rc=$?
  if [ "$rc" = 1 ] && grep -qF 'refused blocking wait on a live job' "$TMP/status-bare"; then
    report 0 'structural negative control: a bare wait on a live child is refused'
  else
    report 1 'structural negative control: a bare wait on a live child is refused' "exit $rc; $(cat "$TMP/status-bare")"
  fi
}
harness_leg_run '1: child status survives a returning trap' status_leg

query_leg() {
  write_status_probe
  python3 - "$HERE/process-group.sh" "$TMP/unmasked.sh" <<'MUTANT'
from pathlib import Path
import sys
source = Path(sys.argv[1]).read_text()
old = "wait_jobs=$(trap '' INT TERM HUP; jobs -pr; jobs -ps)"
if source.count(old) != 1:
    raise SystemExit('expected exactly one masked wait_child query')
Path(sys.argv[2]).write_text(source.replace(old, 'wait_jobs=$(jobs -pr; jobs -ps)'))
MUTANT
  [ "$?" = 0 ] || { harness_abandon 'cannot build unmasked-query mutant'; return; }
  local mode helper rc
  for mode in shared unmasked; do
    helper=$HERE/process-group.sh
    [ "$mode" != unmasked ] || helper=$TMP/unmasked.sh
    # Own the group before injecting kill -TERM 0; never signal the harness.
    # Use this harness's Bash, covering /bin/bash 3.2 and CI's PATH Bash.
    python3 - "$BASH" "$TMP/status-probe.sh" "$helper" "$TMP/query-$mode.marker" >"$TMP/query-$mode" 2>&1 <<'DRIVER'
import subprocess
import sys
try:
    result = subprocess.run([sys.argv[1], sys.argv[2], sys.argv[3], 'query', sys.argv[4]],
                            start_new_session=True, timeout=10)
except subprocess.TimeoutExpired:
    raise SystemExit('query probe exceeded its 10-second bound')
raise SystemExit(result.returncode)
DRIVER
    rc=$?
    if [ "$mode" = shared ]; then
      report "$rc" 'masked job-table query survives group TERM and preserves child status' "$(cat "$TMP/query-$mode")"
    elif [ "$rc" = 1 ] && grep -qF 'refused blocking wait on a live job' "$TMP/query-$mode"; then
      report 0 'unmasked-query negative control: blocking wait on a live child is refused'
    else
      report 1 'unmasked-query negative control: blocking wait on a live child is refused' "exit $rc; $(cat "$TMP/query-$mode")"
    fi
  done
}
harness_leg_run '1b: group TERM inside the job-table query' query_leg

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
