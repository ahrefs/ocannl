#!/usr/bin/env bash
# Shared process liveness and signal-responsive child waits for harnesses and tools.
#
# A signal probe alone counts zombies as alive on Linux.  Narrow that answer
# with process state where the host publishes it, while retaining the signal as
# a necessary condition so a state census can never make an unreachable group
# look reachable.  Callers may use this snapshot to shorten a grace period or
# choose diagnostic wording; they must not use it to veto cleanup signals.

group_alive() { # <pgid>; exits 0 iff some member is not a zombie
  local pgid=$1 f line states found
  # 0 and negatives are kill specials (the caller's own group, or broadcast),
  # so only a positive decimal integer is a process-group id at all.
  case $pgid in '' | *[!0-9]* | 0) return 1 ;; esac
  kill -0 -- "-$pgid" 2>/dev/null || return 1
  if [ -r /proc/self/stat ]; then
    found=0
    for f in /proc/[0-9]*/stat; do
      # Grouped, not `read ... 2>/dev/null`: the shell reports a failed
      # redirection before the command's own stderr redirection applies.
      { read -r line <"$f"; } 2>/dev/null || continue
      # `pid (comm) state ppid pgrp ...`; comm may itself contain ") ".
      line=${line##*) }
      # shellcheck disable=SC2086
      set -- $line
      [ "${3:-}" = "$pgid" ] || continue
      found=1
      [ "$1" = Z ] || return 0
    done
    # A published group containing only zombies is dead.  If procfs published
    # no member despite the successful signal probe, fall through: a platform
    # with a different stat layout did not answer the question.
    [ "$found" = 1 ] && return 1
  fi
  if states=$(ps -A -o pgid=,stat= 2>/dev/null) && [ -n "$states" ]; then
    printf '%s\n' "$states" |
      awk -v g="$pgid" '$1 == g && $2 !~ /^[Zz]/ { alive = 1 } END { exit !alive }'
    return $?
  fi
  # The signal probe above succeeded.  Where neither state reader answers,
  # conservatively over-report the group as alive.
  return 0
}

# The same narrowing for one process: a killed process whose parent has died
# waits as a zombie until a reaper collects it, which under load (or under a
# PID 1 that reaps nothing) is after the caller looks (gh-ocannl-1383).
pid_alive() { # <pid>; exits 0 iff the process exists and is not a zombie
  local pid=$1 line state
  case $pid in '' | *[!0-9]* | 0) return 1 ;; esac
  kill -0 "$pid" 2>/dev/null || return 1
  if [ -r /proc/self/stat ]; then
    # Unreadable here means it was reaped between the probe and the read.
    { read -r line <"/proc/$pid/stat"; } 2>/dev/null || return 1
    line=${line##*) }
    # shellcheck disable=SC2086
    set -- $line
    [ "${1:-}" != Z ]
    return $?
  fi
  state=$(ps -o stat= -p "$pid" 2>/dev/null) || state=
  state=${state//[[:space:]]/}
  case $state in
    [Zz]*) return 1 ;;
    ?*) return 0 ;;
  esac
  # No row: reaped since the probe, or no state reader answered, which as above
  # over-reports it as alive.
  kill -0 "$pid" 2>/dev/null
}

# Let a trapped signal run within a second even when Bash holds it in front
# of a blocking wait (gh-ocannl-1504). The foreground tick ends sooner when
# PID exits; Bash reaps it before running the trap, so a trap that exits cannot
# leave a tick behind in the caller's process group. A background tick plus
# `wait $!` bounded the trap too, but leaked that tick on cancellation
# (gh-ocannl-1914's sweep cancellation leg).
# Close the run's descriptors: lock fd 9 and repeat's witness fds 6 and 8.
wait_tick() { # PID
  # Apply redirections only in the exec child, so a trap keeps the caller's
  # streams and lock descriptors even if it lands during tick setup.
  (exec perl -e 'my $n = 0; select undef, undef, undef, 0.05 while kill(0, $ARGV[0]) && ++$n < 20' "$1" \
    </dev/null >/dev/null 2>&1 6>&- 8>&- 9>&-)
}

# Wait for an unreaped direct child, preserving its exit status. The job table
# belongs to this shell, so an exited child's recycled pid cannot extend the
# wait. Select running and stopped jobs: Bash 3.2 can retain completed jobs
# in `jobs -p`, and a stopped job's eventual wait would still block.
# Callers' traps may relay cancellation and return; check again after each tick
# rather than treating an interrupted wait's status as the child's verdict.
wait_child() { # PID
  local wait_jobs
  while :; do
    wait_jobs=$(jobs -pr; jobs -ps)
    case $'\n'"$wait_jobs"$'\n' in
      *$'\n'"$1"$'\n'*) wait_tick "$1" || : ;;
      *) break ;;
    esac
  done
  wait "$1"
}
