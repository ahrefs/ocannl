#!/usr/bin/env bash
# Where this host's fleet-worker.sh might be -- the lukstafi/ludics-lite issue-wave skill's front
# end to the fleet's execution registry and run-time slots. Sourced by tools/test-run.sh,
# tools/fmt-check.sh and tools/test-harnesses.sh, which each take one of the box's correctness
# slots through it (gh-ocannl-1004, gh-ocannl-1519, gh-ocannl-1576) and find it with ONE probe,
# fleet_slot_probe below, and by tools/sweep.sh, which asks it whether an exclusive measurement
# holds a box before running a unit there (gh-ocannl-1097). Sourced, never executed.
#
# One list for both, so that the knob means the same thing to each: OCANNL_TOOL_FLEET_WORKER names
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

# The slot-takers' one probe (gh-ocannl-1576): each candidate in turn until one answers
# `execution slot --probe`. The two skill trees are deployed independently, and one of them may
# predate the probe while the other has it; a fleet-worker.sh from before the fleet's nested-slot
# rule has no `--probe` and is never used, so a caller's own enclosing `execution slot` is reused
# by the slot this one takes, never doubled. Each ask is bounded, so a wedged fleet-worker.sh costs
# the caller 30 s and no slot; it is answered through ANSWER (a scratch path the caller owns,
# removed again here), so a descendant the probe leaves behind cannot hold a pipe's read open; and
# fd 9 -- tools/test-run.sh's worktree lock -- is closed for it. On an answer, returns 0 with
# fleet_slot_fw (the fleet-worker.sh to take the slot through), fleet_slot_box, fleet_slot_slots
# and fleet_slot_tokens set; otherwise -- CI, a machine outside the fleet,
# OCANNL_TOOL_FLEET_WORKER=none -- returns 1 with all four empty, and the caller runs its work
# directly, silently.
fleet_slot_fw= fleet_slot_box= fleet_slot_slots= fleet_slot_tokens=
fleet_slot_probe() { # ANSWER
  local fw tag box slots tokens
  fleet_slot_fw= fleet_slot_box= fleet_slot_slots= fleet_slot_tokens=
  while IFS= read -r fw; do
    [ -x "$fw" ] || continue
    perl -e 'alarm shift; exec @ARGV or exit 127' 30 "$fw" execution slot --probe \
      </dev/null >"$1" 2>/dev/null 9>&- || continue
    tag= box= slots= tokens=
    read -r tag _ _ box slots tokens _ <"$1"
    if [ "$tag" = EXECUTION ] && [ -n "$tokens" ]; then
      fleet_slot_fw=$fw fleet_slot_box=$box fleet_slot_slots=$slots fleet_slot_tokens=$tokens
      break
    fi
  done < <(fleet_worker_candidates)
  rm -f "$1"
  [ -n "$fleet_slot_fw" ]
}
