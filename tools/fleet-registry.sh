#!/usr/bin/env bash
# Sweep fleet registry reader, sourced by tools/sweep.sh.
# The caller provides fleet_worker_candidates, ask_capped, run_capped and the
# lab-map.sh lookups; per-unit reads use its LANE_DIR. No probe runs on sourcing.

# ------------------------------------------------------ the fleet's execution reservations
# The lab locks say "a lane runs on this box" and "keep this VM alive"; neither says "this box is
# timing something". That is the fleet's execution registry (lukstafi/ludics-lite's issue-wave
# skill, references/executions.md): a coordinator RESERVES a box for a run, and a `measurement`
# reservation is exclusive -- the registry refuses every other reservation on its host while one is
# outstanding, and every correctness slot there refuses with it. The sweep takes neither, so an
# exclusive measurement was invisible to it: on 2026-09-27 the rog lane ran `dune clean` and a cuda
# `@slow` suite into the middle of a 4-7 h tuned measurement there (gh-ocannl-1097), the search froze
# on its next candidate, and every timing it took from then on was contaminated.
#
# So before EACH unit, not just a lane's first (minix's second unit starts long after its first,
# and a measurement reserved in between is no less exclusive), the lane asks the registry whether
# an outstanding `measurement` names its box, and on one records `skip (box <box> under an exclusive
# measurement: <request_id> (<state> on <host>))` instead of running. A skip, like a box another
# lane holds: nothing was tested and nothing failed, and the run record's skip coverage is already
# the channel for a backend that went untested. A correctness reservation defers nothing: the
# fleet's policy lets correctness runs share a box (its run-time slots bound them), and a standing
# one lasts a worker's whole life, so a sweep that stood aside for those would rarely run at all.
#
# The other direction is the registry's. The sweep owns no registry record, so a measurement
# reserved WHILE a unit runs is refused by the fleet's side: a `measurement` reserve, run or
# dispatch in fleet-execution.py refuses a box whose wake-lab LANE lock is held, naming the holder,
# and holds that lock SHARED while it writes its record (lukstafi/ludics-lite#445, since
# lukstafi/ludics-lite#451). A remote lane takes the lock EXCLUSIVE before it reads the registry
# for any unit, so for it the race is closed both ways: a measurement that got the lock first has
# its record written before this read can happen, and one that comes after finds the lock held and
# is refused. The local lane takes no lab lock (run_lane: it has no host, and the lock is about a
# box's VM), so nothing refuses a measurement reserved on this host while one of its units runs;
# there the check stays one read before each unit.
#
# A record names its box by an ssh identity, and a box has one per endpoint. A remote lane's names
# are every alias on its box's row of wake-lab.sh's endpoint map (lab_map), the boots the sweep
# never addresses included: a measurement booked on a dual-boot box's Windows side, for a
# verification reboot, holds the box as surely as one on its Linux, and tuf's row lists a `-win`
# and a `-wsl` its lane never dials. The row is the whole answer -- the map is the one table, and a
# run with no map has no remote lane to ask for. The local lane's name is the one `execution slot
# --probe` gives this host -- the fleet's `mac-studio`, not the `m4-max` measurement-box ID the
# history rows carry.
#
# The reader is the registry's own, `fleet-worker.sh execution list --active --compact` -- the
# supervision read executions.md documents, which asks the anchor over ssh from anywhere else --
# through the first fleet-worker candidate that answers the probe, as tools/test-run.sh chooses the
# one it takes its slot through. No candidate answering means this host is outside the fleet, and
# the header says the registry was NOT CONSULTED. A registry that cannot be read for one unit fails
# OPEN and loud: the unit runs, under a WARNING line. An outage must not cost a day of the only
# coverage five backends have; a measurement overlapped by a sweep can be run again, and the
# fleet's measurement guidance already has its owner check the box's activity before timing.
#
# The names are the sweep's own, outside the fleet's FLEET_* namespace: a fleet host's environment
# EXPORTS FLEET_LOCAL_BOX (and FLEET_BOXES, FLEET_ANCHOR), and an assignment keeps the export, so a
# global of that name cleared here reaches the probed fleet-worker.sh as an empty box name and the
# probe dies -- the registry silently NOT CONSULTED on the very host that runs the sweep.
SWEEP_FLEET_FW=     # the fleet-worker.sh that answered the probe, empty for none
SWEEP_FLEET_BOX=    # this host's name in the fleet, as that probe gave it
SWEEP_FLEET_STATUS= # the header's line
sweep_fleet_probe() {
  local fw probe tag box tokens
  while IFS= read -r fw; do
    [ -x "$fw" ] || continue
    ask_capped probe 30 "$fw" execution slot --probe || continue
    read -r tag _ _ box _ tokens _ <<<"$probe"
    if [ "$tag" = EXECUTION ] && [ -n "$box" ] && [ -n "$tokens" ]; then
      SWEEP_FLEET_FW=$fw
      SWEEP_FLEET_BOX=$box
      SWEEP_FLEET_STATUS="consulted before each unit through $fw (this host is $box)"
      return 0
    fi
  done < <(fleet_worker_candidates)
  if [ "${OCANNL_TOOL_FLEET_WORKER-}" = none ]; then
    SWEEP_FLEET_STATUS="NOT CONSULTED -- OCANNL_TOOL_FLEET_WORKER=none"
  else
    SWEEP_FLEET_STATUS="NOT CONSULTED -- no fleet-worker.sh answered 'execution slot --probe' ($(fleet_worker_candidates | tr '\n' ' ' | sed 's/ $//'))"
  fi
}

# The registry names that are this lane's box, space-separated; empty when there is nothing to ask.
sweep_fleet_lane_names() { # ssh-destination (empty for the local lane)
  if [ -z "$1" ]; then
    printf '%s' "$SWEEP_FLEET_BOX"
    return
  fi
  lab_row "$(lab_box_of "$1")"
}

# 0 with MEASUREMENT_HOLDERS set when an outstanding measurement names one of the names; 1 when none
# does; 2 with REGISTRY_REASON set when the registry could not be read. Through run_capped and a
# file, not a command substitution, so a cancellation reaches the reader (see remote_guest_id).
sweep_fleet_under_measurement() { # names...
  local out rc
  MEASUREMENT_HOLDERS=
  REGISTRY_REASON=
  out=$(mktemp "$LANE_DIR/registry.XXXXXX") || {
    REGISTRY_REASON="cannot create a scratch file under $LANE_DIR"
    return 2
  }
  run_capped 120 "$SWEEP_FLEET_FW" execution list --active --compact >"$out" 2>"$out.err" </dev/null
  rc=$?
  if [ "$rc" -ne 0 ]; then
    REGISTRY_REASON="execution list exited $rc: $(tail -1 "$out.err" 2>/dev/null | tr -d '\000-\037' | cut -c1-200)"
    rm -f "$out" "$out.err"
    return 2
  fi
  MEASUREMENT_HOLDERS=$(perl -MJSON::PP -e '
    my %names = map { $_ => 1 } @ARGV;
    local $/;
    my $list = eval { JSON::PP->new->decode(<STDIN>) };
    exit 3 unless ref $list eq "ARRAY";
    my @held;
    for my $r (@$list) {
      exit 3 unless ref $r eq "HASH" && ref $r->{request} eq "HASH";
      my $q = $r->{request};
      next unless ($q->{kind} // "") eq "measurement" && $names{$q->{execution_host} // ""};
      push @held, sprintf("%s (%s on %s)", $r->{request_id} // "?", $r->{state} // "?",
        $q->{execution_host});
    }
    print join("; ", @held);' "$@" <"$out")
  rc=$?
  rm -f "$out" "$out.err"
  if [ "$rc" -ne 0 ]; then
    REGISTRY_REASON="execution list printed no registry this could read"
    return 2
  fi
  [ -n "$MEASUREMENT_HOLDERS" ]
}

