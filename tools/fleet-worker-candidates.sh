#!/usr/bin/env bash
# Where this host's fleet-worker.sh might be -- the lukstafi/ludics-lite issue-wave skill's front
# end to the fleet's execution registry and run-time slots -- and the one way OCANNL's tools ask
# it whether it answers. Sourced by tools/test-run.sh, tools/fmt-check.sh and
# tools/test-harnesses.sh, which each take one of the box's correctness slots through it
# (gh-ocannl-1004, gh-ocannl-1519, gh-ocannl-1576), and by tools/sweep.sh, which asks it whether an
# exclusive measurement holds a box before running a unit there (gh-ocannl-1097) and, through
# tools/gpu-tokens.sh, asks a GPU box's own copy who holds its tokens (gh-ocannl-1177). Every one
# of them finds its fleet-worker.sh with ONE probe, fleet_slot_probe below (gh-ocannl-1841), and
# the two that run themselves again inside the slot do it with ONE helper, fleet_slot_rerun.
# Sourced, never executed; sourcing defines functions only.
#
# One list for all, so that the knob means the same thing to each: OCANNL_TOOL_FLEET_WORKER names
# another fleet-worker.sh (a harness's fake), `none` turns the fleet off, and unset tries the two
# skill trees in turn -- they are deployed independently, and one of them may predate a verb the
# other has. A caller takes the first candidate that answers `execution slot --probe` (one line,
# `EXECUTION SLOT PROBE <box> <slots> <gpu tokens>`, no lock and no registry read), which is
# the capability check and, on a fleet box, names the box as the fleet's registry does.
#
# A home directory as the argument lists the candidates under it instead of this host's: the sweep
# asks a GPU box's own fleet-worker.sh about its tokens (tools/gpu-tokens.sh), so it hands the box's
# home here, and the one list serves both sides of the ssh.
fleet_worker_candidates() { # [home]; prints the fleet-worker.sh candidates, one per line
  case ${OCANNL_TOOL_FLEET_WORKER-} in
    none) ;;
    '')
      printf '%s\n' "${1:-$HOME}/.claude/skills/issue-wave/scripts/fleet-worker.sh" \
        "${1:-$HOME}/.codex/skills/issue-wave/scripts/fleet-worker.sh"
      ;;
    *) printf '%s\n' "$OCANNL_TOOL_FLEET_WORKER" ;;
  esac
}

# The one probe (gh-ocannl-1576, gh-ocannl-1841): each candidate in turn until one answers
# `execution slot --probe`. The two skill trees are deployed independently, and one of them may
# predate the probe while the other has it; a fleet-worker.sh from before the fleet's nested-slot
# rule has no `--probe` and is never used, so a caller's own enclosing `execution slot` is reused
# by the slot this one takes, never doubled.
#
# THE ANSWER'S BOUNDARY (a fail-closed allowlist): a candidate answers only by exiting 0 with a
# first line whose first three words are exactly `EXECUTION SLOT PROBE`, followed by a box name and
# two positive whole numbers without a leading zero, the slots and the GPU tokens; any words after
# those (`measurement <id>`, inside a measurement hold) are ignored. Anything else -- a non-zero
# exit, a refusal line, a line from an older verb, an empty answer -- is not an answer, and the next
# candidate is asked. That is the whole grammar, so a reader of the probe never re-parses its line.
#
# Each ask is bounded, 30 s: by RUNNER when one is given, a command prefix called as
# `RUNNER... <seconds> <command>...` (tools/sweep.sh passes its run_capped, which a cancellation
# reaches), and by perl's alarm otherwise. It is answered through ANSWER (a scratch path the caller
# owns, removed again here), so a descendant the probe leaves behind cannot hold a pipe's read open;
# and fd 9 -- tools/test-run.sh's worktree lock -- is closed for it. On an answer, returns 0 with
# fleet_slot_fw (the fleet-worker.sh that answered), fleet_slot_box, fleet_slot_slots and
# fleet_slot_tokens set; otherwise -- CI, a machine outside the fleet, OCANNL_TOOL_FLEET_WORKER=none
# -- returns 1 with all four empty, and the caller goes without the fleet (each says how).
#
# The names are lower-case, outside the fleet's FLEET_* namespace: a fleet host's environment
# EXPORTS FLEET_LOCAL_BOX (and FLEET_BOXES, FLEET_ANCHOR), and an assignment keeps the export, so a
# global of such a name cleared here would reach the probed fleet-worker.sh as an empty box name.
#
# tools/gpu-tokens.sh runs these same functions on a GPU box, as text (`declare -f`) under the
# box's bash, so they stay plain bash: no helper beyond these, perl and the candidates' own verbs.
fleet_slot_fw= fleet_slot_box= fleet_slot_slots= fleet_slot_tokens=
fleet_slot_alarm() { # seconds command... -- the default bound
  perl -e 'alarm shift; exec @ARGV or exit 127' "$@"
}
fleet_slot_count() { # word -- 0 for a positive whole number without a leading zero
  case $1 in '' | 0* | *[!0-9]*) return 1 ;; esac
}
fleet_slot_probe() { # ANSWER [RUNNER...]
  local answer=$1 fw tag w2 w3 box slots tokens
  shift
  [ "$#" -gt 0 ] || set -- fleet_slot_alarm
  fleet_slot_fw= fleet_slot_box= fleet_slot_slots= fleet_slot_tokens=
  while IFS= read -r fw; do
    [ -x "$fw" ] || continue
    "$@" 30 "$fw" execution slot --probe </dev/null >"$answer" 2>/dev/null 9>&- || continue
    tag= w2= w3= box= slots= tokens=
    read -r tag w2 w3 box slots tokens _ <"$answer"
    if [ "$tag $w2 $w3" = 'EXECUTION SLOT PROBE' ] && [ -n "$box" ] &&
      fleet_slot_count "$slots" && fleet_slot_count "$tokens"; then
      fleet_slot_fw=$fw fleet_slot_box=$box fleet_slot_slots=$slots fleet_slot_tokens=$tokens
      break
    fi
  done < <(fleet_worker_candidates)
  rm -f "$answer"
  [ -n "$fleet_slot_fw" ]
}

# The one rerun-yourself-in-the-slot (gh-ocannl-1841), for a tool whose whole run is one batch that
# holds no device: tools/fmt-check.sh (gh-ocannl-1519) and tools/test-harnesses.sh
# (gh-ocannl-1576). tools/test-run.sh takes its slot around dune instead (plan_slot there), and
# decides `--cpu` per batch.
#
# Through the fleet-worker.sh fleet_slot_probe finds, COMMAND (the caller again, by an absolute
# path) runs under `execution slot --wait <OCANNL_TOOL_SLOT_WAIT, 600> --cpu`, with the caller's
# environment variable VAR naming DIR/record. The copy under the slot must clear VAR at once (so
# nothing it runs inherits it), truncate the file it names before anything else, and write its exit
# status there as it ends. A caller's own enclosing `execution slot` is reused, not doubled (the
# fleet's nested-slot rule, which the probe proves the deployed version has). DIR is a scratch
# directory the caller owns and removes; the probe answers through DIR/probe.
#
# No probe answer -- CI, a machine outside the fleet, OCANNL_TOOL_FLEET_WORKER=none -- returns 0 at
# once, silently, and the caller runs its work directly. Otherwise this never returns: it exits
# with the copy's recorded status, whatever fleet-worker.sh's own after it (a failure of its
# supervision after the run is not the run's); with 75, after `<PREFIX>: FAILED (exit 75): SLOT
# REFUSED: ...`, when no record exists -- the slot never ran the copy (refused, or the registry
# unreachable), so nothing ran; and with fleet-worker.sh's status (1 if that was 0), after `<PREFIX>:
# FAILED (exit <n>): <WHAT> under the fleet slot ended without a status`, when the record is there
# but empty -- the copy was killed. Those lines go to stdout and are the last thing printed, so
# a tool whose last line is its verdict keeps that form.
fleet_slot_rerun() { # VAR DIR PREFIX WHAT -- COMMAND...
  local var=$1 dir=$2 prefix=$3 what=$4 wait rc recorded
  shift 4
  [ "${1-}" != -- ] || shift
  fleet_slot_probe "$dir/probe" || return 0
  wait=${OCANNL_TOOL_SLOT_WAIT:-600}
  case $wait in '' | *[!0-9]*) wait=600 ;; esac
  echo "$prefix: taking one of $fleet_slot_box's $fleet_slot_slots fleet correctness slots for $what, as --cpu; waiting at most ${wait}s, an enclosing slot reused"
  rc=0
  env "$var=$dir/record" "$fleet_slot_fw" execution slot --wait "$wait" --cpu -- "$@" || rc=$?
  if [ ! -e "$dir/record" ]; then
    echo "$prefix: FAILED (exit 75): SLOT REFUSED: fleet-worker.sh exited $rc without running $what; nothing ran"
    exit 75
  fi
  recorded=
  read -r recorded <"$dir/record"
  case $recorded in
    '' | *[!0-9]*)
      [ "$rc" -ne 0 ] || rc=1
      echo "$prefix: FAILED (exit $rc): $what under the fleet slot ended without a status"
      exit "$rc"
      ;;
  esac
  exit "$recorded"
}
