#!/usr/bin/env bash

# Run the formatter gate while also rejecting the invalid-odoc warnings that
# ocamlformat reports without changing its successful exit status.
#
# The default command passes `--force`: ocamlformat's warning is printed only
# when its action RUNS, and after a `dune fmt` every already-formatted file's
# action is up to date, so a plain `dune build @fmt` replays nothing and passes
# a tree that CI's fresh checkout rejects (gh-ocannl-1155). The last line of
# output is always a `fmt-check:` verdict naming the exit status, so a reader
# who piped the output through `tail` still sees whether it passed.
#
# The default command then builds the repository scans, `@test/operations/scans`
# (gh-ocannl-1321): this script is the documented last step before every push,
# so the scans run on every push without an edit first having to match a
# description of their reach -- a description that kept trailing it. They take
# seconds on a warm build, and the time is printed. Run from the repository
# root. OCANNL_TOOL_FMT_CHECK_SCANS=0 skips them, and says so: CI's `fmt` job,
# whose switch holds no project dependency (the build matrix runs the scans), or
# a formatting-only check. A command given as arguments runs alone, the scans
# not built: that is the harness's form, and a caller's own formatter. The
# script takes no options: an argument starting with `-` is refused with the
# usage, never run as the command (`fmt-check.sh -j 4` used to exit 127).
#
# On a fleet box the default command takes one of the box's run-time
# correctness slots for both builds, as tools/test-run.sh does (gh-ocannl-1519,
# gh-ocannl-1004; take_fleet_slot below), so the documented last step before a
# push never builds while a measurement holds the box.

set -uo pipefail

# Set only by this script, for the copy of itself it runs inside the fleet slot
# (take_fleet_slot): the file that says it started and, once it has one, its
# verdict's status.
slot_record=${OCANNL_TOOL_FMT_CHECK_IN_SLOT-}
unset OCANNL_TOOL_FMT_CHECK_IN_SLOT
[ -z "$slot_record" ] || : >"$slot_record" || exit 2

# Credentials never reach dune, which records every spawned process's environment in the
# worktree's `_build/trace.csexp` (gh-ocannl-1280): the deny-list tools/test-run.sh applies, to
# this script's environment and -- for the default command -- again inside the one `opam exec`
# builds, whose switch updates could set one.
credential_env=$(dirname "$0")/credential-env.sh
[ -r "$credential_env" ] || { echo "fmt-check: FAILED (exit 2): cannot read $credential_env"; exit 2; }
# shellcheck source=credential-env.sh
. "$credential_env"
eval "$(credential_env_scrub_text)" || {
  echo "fmt-check: FAILED (exit 2): cannot remove credential variables:$credential_env_left"
  exit 2
}

verdict() { # STATUS [REASON]
  [ -z "$slot_record" ] || echo "$1" >"$slot_record"
  if [ "$1" -eq 0 ]; then
    echo "fmt-check: PASSED${2:+ ($2)}"
  else
    echo "fmt-check: FAILED (exit $1)${2:+: $2}"
  fi
  exit "$1"
}

# Builds a command line running "$@" inside the switch, with the credentials
# removed again after `opam exec` applied the switch's environment.
in_switch() {
  scrubbed_command=(opam exec -- sh -c "{ $(credential_env_scrub_text); } || exit 2; exec \"\$@\"" \
    fmt-check "$@")
}

usage() {
  cat <<'EOF'
usage: tools/fmt-check.sh                     dune build @fmt --force, then the repository scans
       tools/fmt-check.sh <command> [<arg>...]  <command> as the formatter, alone
Run from the repository root. There are no options; OCANNL_TOOL_FMT_CHECK_SCANS=0 skips the scans.
EOF
}

# The fleet's run-time correctness slot, taken as tools/test-run.sh takes it
# (plan_slot there, gh-ocannl-1004): the first of tools/fleet-worker-candidates.sh's
# fleet-worker.sh that answers `execution slot --probe` runs this script again,
# under `execution slot --wait <OCANNL_TOOL_SLOT_WAIT, 600> --cpu`. `--cpu`
# because neither build holds a device: `@test/operations/scans` is on
# `Slot_kind.config_free_aliases` (test/support/slot_kind.ml), whose closure the
# `slot_kind_live` scan keeps free of any backend, so `tools/test-run.sh plan
# build @test/operations/scans` resolves "no backend"; `@fmt` links nothing. A
# caller's own enclosing `execution slot` is reused, not doubled (the fleet's
# nested-slot rule, which the probe proves the deployed version has). No probe
# answer -- CI, a machine outside the fleet, OCANNL_TOOL_FLEET_WORKER=none --
# runs the builds directly, silently. One slot covers both builds; the copy
# under it reports through $slot_record, so a slot that never ran it (refused,
# or the registry unreachable) is told apart from a check that failed: exit 75,
# `SLOT REFUSED`, as tools/test-run.sh reports it, and nothing ran.
take_fleet_slot() {
  local candidates fw= cand tag box slots tokens wait status recorded self
  candidates=$(dirname "$0")/fleet-worker-candidates.sh
  [ -r "$candidates" ] || verdict 2 "cannot read $candidates"
  # shellcheck source=fleet-worker-candidates.sh
  . "$candidates"
  slot_dir=$(mktemp -d "${TMPDIR:-/tmp}/ocannl-fmt-check-slot.XXXXXX") || verdict 2 "cannot make a scratch directory"
  trap 'rm -rf "$slot_dir"' EXIT
  while IFS= read -r cand; do
    [ -x "$cand" ] || continue
    # Bounded, and answered through a file, as tools/test-run.sh asks it.
    perl -e 'alarm shift; exec @ARGV or exit 127' 30 "$cand" execution slot --probe \
      </dev/null >"$slot_dir/probe" 2>/dev/null || continue
    read -r tag _ _ box slots tokens _ <"$slot_dir/probe"
    [ "$tag" = EXECUTION ] && [ -n "$tokens" ] && { fw=$cand; break; }
  done < <(fleet_worker_candidates)
  if [ -z "$fw" ]; then
    rm -rf "$slot_dir"
    trap - EXIT
    return 0
  fi
  wait=${OCANNL_TOOL_SLOT_WAIT:-600}
  case $wait in '' | *[!0-9]*) wait=600 ;; esac
  # CDPATH cleared: a `cd` it resolves prints the directory, which would join the path.
  self=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)/$(basename -- "$0") || verdict 2 "cannot resolve $0"
  echo "fmt-check: taking one of $box's $slots fleet correctness slots for both builds, as --cpu (gh-ocannl-1519); waiting at most ${wait}s, an enclosing slot reused"
  OCANNL_TOOL_FMT_CHECK_IN_SLOT=$slot_dir/record "$fw" execution slot --wait "$wait" --cpu -- \
    "${BASH:-bash}" "$self"
  status=$?
  [ -e "$slot_dir/record" ] ||
    verdict 75 "SLOT REFUSED: fleet-worker.sh exited $status without running the check; nothing ran"
  recorded=
  read -r recorded <"$slot_dir/record"
  case $recorded in
    '' | *[!0-9]*)
      [ "$status" -ne 0 ] || status=1
      verdict "$status" "the check under the fleet slot ended without a verdict"
      ;;
  esac
  # The check's own verdict line was its last, so its status is the exit status, whatever
  # fleet-worker.sh's own (a failure of its supervision after the check is not the check's);
  # nothing is printed after it.
  exit "$recorded"
}

case ${1-} in
  -*)
    usage
    verdict 2 "'$1' is not an option: fmt-check.sh takes none"
    ;;
esac

scans=
if [ "$#" -eq 0 ]; then
  case ${OCANNL_TOOL_FMT_CHECK_SCANS-1} in
    1) scans=yes ;;
    0) scans=skipped ;;
    *) verdict 2 "OCANNL_TOOL_FMT_CHECK_SCANS is 0 (skip the scans) or 1 (the default), not '${OCANNL_TOOL_FMT_CHECK_SCANS}'" ;;
  esac
  [ -n "$slot_record" ] || take_fleet_slot
  in_switch dune build @fmt --force
  set -- "${scrubbed_command[@]}"
fi

fmt_log=$(mktemp "${TMPDIR:-/tmp}/ocannl-fmt-check.XXXXXX") || exit 2
trap 'rm -f "$fmt_log"' EXIT

"$@" >"$fmt_log" 2>&1
fmt_status=$?
if ! cat "$fmt_log"; then
  verdict 2 "could not replay formatter output"
fi

# A real formatter failure owns the verdict, including statuses other than 1.
if [ "$fmt_status" -ne 0 ]; then
  verdict "$fmt_status" "the formatter command failed"
fi

if grep -Fq "Warning: Invalid documentation comment:" "$fmt_log"; then
  verdict 1 "invalid documentation comments are forbidden"
fi

case $scans in
  skipped) verdict 0 "repository scans skipped: OCANNL_TOOL_FMT_CHECK_SCANS=0" ;;
  yes) ;;
  *) verdict 0 ;;
esac

# Unpiped, so dune's own diff and promotion hints reach the reader as they are.
echo "fmt-check: building the repository scans (@test/operations/scans)"
scans_started=$SECONDS
in_switch dune build @test/operations/scans
"${scrubbed_command[@]}"
scans_status=$?
echo "fmt-check: the repository scans took $((SECONDS - scans_started)) s"
if [ "$scans_status" -ne 0 ]; then
  verdict "$scans_status" "the repository scans failed (dune build @test/operations/scans)"
fi

verdict 0
