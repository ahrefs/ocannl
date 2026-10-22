#!/usr/bin/env bash
# Shared process-group liveness predicate for the shell harnesses and tools.
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
