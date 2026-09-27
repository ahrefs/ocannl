#!/usr/bin/env bash
# Where this host's fleet-worker.sh might be -- the lukstafi/ludics-lite issue-wave skill's front
# end to the fleet's execution registry and run-time slots. Sourced by tools/test-run.sh, which
# takes one of the box's correctness slots through it (gh-ocannl-1004), and by tools/sweep.sh,
# which asks it whether an exclusive measurement holds a box before running a unit there
# (gh-ocannl-1097). Sourced, never executed.
#
# One list for both, so that the knob means the same thing to each: OCANNL_TOOL_FLEET_WORKER names
# another fleet-worker.sh (a harness's fake), `none` turns the fleet off, and unset tries the two
# skill trees in turn -- they are deployed independently, and one of them may predate a verb the
# other has. A caller takes the first candidate that answers `execution slot --probe` (one line,
# `EXECUTION SLOT PROBE <box> <slots> <gpu tokens>`, no lock and no registry read), which is
# the capability check and, on a fleet box, names the box as the fleet's registry does.

fleet_worker_candidates() { # prints the fleet-worker.sh candidates, one per line
  case ${OCANNL_TOOL_FLEET_WORKER-} in
    none) ;;
    '')
      printf '%s\n' "$HOME/.claude/skills/issue-wave/scripts/fleet-worker.sh" \
        "$HOME/.codex/skills/issue-wave/scripts/fleet-worker.sh"
      ;;
    *) printf '%s\n' "$OCANNL_TOOL_FLEET_WORKER" ;;
  esac
}
