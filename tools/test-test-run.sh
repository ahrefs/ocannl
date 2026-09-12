#!/usr/bin/env bash
# Tests for the shared `group_alive` used by tools/test-run.sh -- the predicate
# its two callers ask "is anything in this process group still running?" --
# and for the two sentences `stop` prints about a surviving
# process group, which are what that predicate is for (gh-ocannl-742).
#
#   tools/test-test-run.sh          # run every leg
#   tools/test-test-run.sh --keep   # keep the temp dir for inspection
#
# It is the sibling of scripts/test-setup-ocaml-env.sh, whose leg 1 (f) tests
# the same predicate in the SessionStart hook. Neither is a dune test: they
# spawn, STOP and kill process groups, which is a poor fit for `dune runtest`.
# The Ubuntu CI leg runs both directly so Linux decides the kernel-dependent
# zombie-group control (gh-ocannl-795).
#
# It tests the WORKING-TREE copy: `group_alive` is extracted from the shared
# scripts/process-group.sh; `ps_token`, `proc_identity_matches`, `proc_alive`
# come from tools/test-run.sh. The `stop` legs drive that same tool as a
# subprocess. Each extraction is asserted structurally before use, so a sed
# that matched nothing cannot leave every leg passing without testing anything.
#
# Legs:
#   1. extraction -- the functions came out of the shipping script.
#   2. a genuinely live group reads ALIVE (the other side of leg 4: without
#      this, a group_alive that answered "dead" always would pass leg 4).
#   3. a group with no members at all reads DEAD, and so does the bare signal.
#   4. a group holding nothing but a ZOMBIE reads dead, where the bare
#      `kill -0 -- -PGID` this replaced reads it as alive. That misreading is
#      what made `stop` able to announce an orphaned group holding only corpses.
#   5. a pgid that is not a positive decimal integer is refused -- 0 and
#      negatives are kill specials (caller's own group, broadcast).
#   6. `stop` on a group whose leader IGNORES TERM says so and escalates --
#      and the escalation kills the whole group. Every fixture group holds two
#      processes, or a leader-only kill would pass for a group kill.
#   7. `stop` on a group whose leader exits on TERM says the TERM went out and
#      asks for a re-run, rather than claiming the group ignored it.
#   8. `stop` on a run recorded by the version that kept its lock BESIDE the
#      worktree (no `runs` root on record) still attributes a leftover holding
#      that in-tree lock to the run, reaps it, and says so.
#   9. `repeat` forces and preserves three identical dune runs while holding the
#      worktree lock for every iteration.
#  10. stdout drift is a distinct red result, with pairwise diff artifacts.
#  11. stderr-only drift is reported separately and remains a green diagnostic.
#  12. a red dune iteration keeps its nonzero exit code even when repeatable.
#  13. `--alone` serializes dune with `-j 1` on every iteration.
#  14. an active repeat is `last`, and `stop last` cancels the whole set after
#      the current iteration rather than launching the remaining ones.
#  15. cancellation during post-loop comparison still publishes a verdict.
#  16. `wait`'s default bounded deadline covers every repeat iteration.
#  17. repeat cancellation state and traps precede publication as `last`.
#  18. cancellation is checked on both sides of supervisor launch.
#  19. cancelling a later iteration preserves an earlier test failure.
#  20. a supervisor killed without its Dune group cannot overlap the next run.
#  21. the dead supervisor pid is cleared before orphan-group reaping.
#  22. orphan cleanup gates on reachability and revalidates identity for KILL.
#  23. a zombie retains its recorded identity while remaining non-live.
#  24. a setsid descendant cannot escape into the next repeat build context.
#  25. repeat isolation flags precede `dune exec`'s argument separator.
#  26. an option written AFTER the dune arguments is refused, having run
#      nothing -- the misplacement dune would otherwise digest as a red run.
#  27. the same option in its documented place is consumed, not forwarded.
#  28. past dune's own `--` the same word is a program argument, untouched.
#  29. an invocation dune's own parser refuses -- unknown option, unknown
#      subcommand, malformed operand -- digests as INVOCATION REFUSED quoting
#      dune's complaint: `run`/`wait` exit 2 over a RECORDED exit 1, `status`
#      keeps its publication 0, and dune was invoked exactly once.
#  30. controls: a red run (`Error:`/`File` lines, exit 1) still digests as
#      FAIL with exit 1 -- including one whose output opens with a `dune:`
#      line, and one that prints a complete nested dune refusal and then
#      goes on (the shape must be alone in the log, not merely first).
#  31. the `--cap` guard still refuses BEFORE dune is spawned: the two exit-2
#      refusals are told apart by whether dune was ever invoked.
#  32. `repeat` stops after a refused first iteration and exits 2 under the
#      same verdict, while a merely red iteration is still repeated in full.
#  33. read-only paths and lock-status: absent state, invalid input, physical
#      paths, recorded and legacy paths, held/released and uninspectable locks.

set -u

KEEP=0
for arg in "$@"; do
  case "$arg" in
    --keep) KEEP=1 ;;
    # The whole leading comment block, however long it grows: a pinned line
    # range silently truncates --help the first time a leg is added.
    -h|--help) sed -n '2,${/^#/!q;p;}' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "test-test-run.sh: unknown argument '$arg'" >&2; exit 2 ;;
  esac
done

HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="$HERE/test-run.sh"
GROUP_SRC="$HERE/../scripts/process-group.sh"
HOOK_SRC="$HERE/../scripts/setup-ocaml-env.sh"
[ -f "$SRC" ] || { echo "no $SRC" >&2; exit 2; }
[ -f "$GROUP_SRC" ] || { echo "no $GROUP_SRC" >&2; exit 2; }
[ -f "$HOOK_SRC" ] || { echo "no $HOOK_SRC" >&2; exit 2; }

failures=0
report() { # report RC LABEL [DETAIL]
  if [ "$1" -eq 0 ]; then
    printf 'PASS  %s\n' "$2"
  else
    failures=$((failures + 1))
    printf 'FAIL  %s\n' "$2"
    [ $# -ge 3 ] && printf '      %s\n' "$3"
  fi
  return 0
}
skipped=0
skip() { # skip LABEL REASON -- a leg this system cannot decide, not a failure
  skipped=$((skipped + 1))
  printf 'SKIP  %s\n      %s\n' "$1" "$2"
  return 0
}

# The harness needs the two facts about a process that `group_alive` needs, read
# INDEPENDENTLY of it: its state and its process group. Neither is available the
# same way everywhere -- a Git Bash/MSYS `ps` takes no `-o` at all, which the
# Windows CI job depends on and which would otherwise leave every state probe
# here reading "gone". So: /proc where it answers (MSYS has one), `ps` where it
# does not, and the prerequisite check below refuses to let a leg run on a
# system where neither does -- an unreadable state must skip a leg, never pass
# or fail one (Codex review round 1, P2).
pstate() { # <pid> -> its one-letter state, empty where this system will not say
  local line
  if [ -r "/proc/$1/stat" ]; then
    # Grouped, not `read ... 2>/dev/null`: the shell reports a failed
    # redirection before the command's own stderr redirection applies, and
    # these readers are asked about pids that are expected to be gone.
    { read -r line <"/proc/$1/stat"; } 2>/dev/null || return 0
    line=${line##*) }               # comm may itself hold ") "
    # shellcheck disable=SC2086
    set -- $line                    # `state ppid pgrp ...`
    printf '%s' "${1:-}"
    return 0
  fi
  ps -o state= -p "$1" 2>/dev/null | tr -d ' ' | cut -c1
}
ppgid() { # <pid> -> its process group, empty where this system will not say
  local line
  if [ -r "/proc/$1/stat" ]; then
    { read -r line <"/proc/$1/stat"; } 2>/dev/null || return 0
    line=${line##*) }
    # shellcheck disable=SC2086
    set -- $line
    printf '%s' "${3:-}"
    return 0
  fi
  ps -o pgid= -p "$1" 2>/dev/null | tr -d ' '
}
# Probed against a process known to be alive and to have a group -- this one.
# An empty answer here means the reader is absent, which is a different fact
# from a process being gone, and the two are indistinguishable at a leg.
have_state=1; [ -n "$(pstate $$)" ] || have_state=0
have_pgid=1;  [ -n "$(ppgid $$)" ]  || have_pgid=0

echo "testing $SRC and $GROUP_SRC"
printf '  digest %s\n' "$( (cksum <"$SRC") 2>/dev/null || echo '?')"
printf '  group digest %s\n' "$( (cksum <"$GROUP_SRC") 2>/dev/null || echo '?')"
printf '  state reader: %s; pgid reader: %s\n' \
  "$([ "$have_state" = 1 ] && echo present || echo ABSENT)" \
  "$([ "$have_pgid" = 1 ] && echo present || echo ABSENT)"

# Checked, not assumed: nothing here uses `set -e`, so a `mktemp` that fails
# would leave TMP empty and `rm -rf "$TMP"` would be handed the ROOT.
zparent=""   # leg 4's self-stopping zombie maker; cleanup must resume it
livepid=""   # leg 2's live group leader
leader=""    # the stop legs' current group leader
member=""    # and the second process it put in that group
member_token=""  # ... and its start token, since it is not this shell's child
repeat_pid="" # leg 14's active repeat coordinator
legacy_holder="" # leg 8's holder of the in-tree lock
escape_pid="" # leg 24's session-escaped descendant
escape_release=""
TMP="$(mktemp -d "${TMPDIR:-/tmp}/test-run-test.XXXXXX" 2>/dev/null)" || TMP=""
if [ -z "$TMP" ] || [ ! -d "$TMP" ]; then
  echo "could not create a temporary directory under ${TMPDIR:-/tmp}" >&2
  exit 2
fi
cleanup() {
  # Leg 4's zombie maker STOPS ITSELF and is resumed at the end of the leg.
  # Interrupted in between, nothing else would ever resume it: it would be
  # reparented to PID 1 still stopped, still holding its zombie child. Killing
  # it is not the answer either -- that orphans the zombie onto a PID 1 that, in
  # the very environment this leg is about, never reaps. Resume it so it reaps
  # its own child, and only insist if it will not go.
  local waited
  if [ -n "${zparent:-}" ] && kill -0 "$zparent" 2>/dev/null; then
    kill -CONT "$zparent" 2>/dev/null
    for waited in 1 2 3 4 5 6 7 8 9 10; do
      kill -0 "$zparent" 2>/dev/null || break
      sleep 0.2
    done
    kill -KILL "$zparent" 2>/dev/null
    wait "$zparent" 2>/dev/null
  fi
  if [ -n "${livepid:-}" ] && kill -0 "$livepid" 2>/dev/null; then
    kill -KILL -- "-$livepid" 2>/dev/null
    kill -KILL "$livepid" 2>/dev/null
    wait "$livepid" 2>/dev/null
  fi
  # The stop legs' leader ignores TERM by design, so only KILL removes it --
  # and the group with it, since a leg interrupted before its `stop` leaves the
  # whole fixture group behind.
  if [ -n "${leader:-}" ] && kill -0 "$leader" 2>/dev/null; then
    kill -KILL -- "-$leader" 2>/dev/null
    kill -KILL "$leader" 2>/dev/null
    wait "$leader" 2>/dev/null
  fi
  # The group's second member is not this shell's child, so it is named
  # directly as well: a group kill that failed is exactly the case where it
  # would otherwise be left behind. Identity-checked -- see kill_member.
  if [ -n "${member:-}" ]; then kill_member; fi
  if [ -n "${repeat_pid:-}" ] && kill -0 "$repeat_pid" 2>/dev/null; then
    kill -TERM "$repeat_pid" 2>/dev/null
    wait "$repeat_pid" 2>/dev/null
  fi
  if [ -n "${legacy_holder:-}" ] && kill -0 "$legacy_holder" 2>/dev/null; then
    kill -KILL "$legacy_holder" 2>/dev/null
    wait "$legacy_holder" 2>/dev/null
  fi
  [ -z "${escape_release:-}" ] || touch "$escape_release"
  if [ -n "${escape_pid:-}" ] && kill -0 "$escape_pid" 2>/dev/null; then
    kill -KILL "$escape_pid" 2>/dev/null
  fi
  if [ "$KEEP" = 1 ]; then
    echo "kept $TMP"
  elif [ -n "$TMP" ] && [ -d "$TMP" ] && [ "$TMP" != "/" ]; then
    rm -rf "$TMP"
  fi
  return 0
}
trap cleanup EXIT
# Without these, a TERM or a Ctrl-C kills the shell outright and the EXIT trap
# never runs -- which is how an interrupted run could leave a stopped zombie
# maker behind, or a fixture process group outlive the harness that forked it.
# Exiting from the handler is what gets EXIT to fire.
#
# The INT arm covers the way this is actually interrupted, a Ctrl-C at a
# terminal. It is inert when the harness is itself a BACKGROUND job of a
# non-interactive shell -- such a child inherits SIGINT ignored, and a signal
# ignored on entry cannot be re-trapped -- so a scripted test of the cleanup
# path has to signal TERM to see anything happen.
trap 'exit 130' INT
trap 'exit 143' TERM

# ---------------------------------------------------------------------------
# Leg 1: extraction
# ---------------------------------------------------------------------------
# Checked structurally -- opens with the header, closes with the brace, has a
# body -- rather than by grepping for one line of it: this guard must still hold
# while the function under test is being mutated to see a leg go red. The header
# is matched as a PREFIX, since it carries a trailing comment.
#
# `ps_token` is extracted alongside `group_alive` because the stop legs below
# have to FORGE a run directory's leader token, and a token spelled differently
# from the one the shipping script recomputes would leave every one of them
# falling through to "nothing left to signal" -- passing no leg, but testing
# neither sentence either.
for fn in group_alive ps_token proc_identity_matches proc_alive; do
  if [ "$fn" = group_alive ]; then fn_src="$GROUP_SRC"; else fn_src="$SRC"; fi
  g_min=10
  sed -n "/^$fn() {/,/^}/p" "$fn_src" >"$TMP/$fn.sh"
  g_lines="$(wc -l <"$TMP/$fn.sh" | tr -d ' ')"
  g_head="$(head -n1 "$TMP/$fn.sh")"
  case $g_head in "$fn() {"*) g_ok=1 ;; *) g_ok=0 ;; esac
  if [ "$g_ok" = 0 ] \
     || [ "$(tail -n1 "$TMP/$fn.sh")" != "}" ] || [ "$g_lines" -lt "$g_min" ]; then
    report 1 "$fn: extracted" "sed did not capture the function body from $fn_src"
  else
    report 0 "$fn: extracted ($g_lines lines)"
    # shellcheck disable=SC1090
    . "$TMP/$fn.sh"
  fi
done

# The decision behind this issue is one definition, not two copies tested in
# parallel. Count definitions across the shared helper and both production
# callers so either copy coming back fails the harness that CI runs.
group_definition_count() { awk '/^group_alive\(\)/ { n++ } END { print n + 0 }' "$@"; }
one_group_definition() { [ "$(group_definition_count "$@")" = 1 ]; }
if one_group_definition "$GROUP_SRC" "$SRC" "$HOOK_SRC"; then
  report 0 "group_alive has one shared production definition"
else
  report 1 "group_alive has one shared production definition" \
    "found $(group_definition_count "$GROUP_SRC" "$SRC" "$HOOK_SRC") definitions across $GROUP_SRC, $SRC and $HOOK_SRC"
fi
# Synthetic duplicate: the clean production tree alone cannot distinguish a
# working uniqueness check from one that always answers yes.
cp "$GROUP_SRC" "$TMP/duplicate-process-group.sh"
if one_group_definition "$GROUP_SRC" "$SRC" "$HOOK_SRC" "$TMP/duplicate-process-group.sh"; then
  report 1 "the shared-definition check rejects a duplicate" \
    "the production definition plus a copied definition still read as unique"
else
  report 0 "the shared-definition check rejects a duplicate"
fi

# ---------------------------------------------------------------------------
# Leg 2: a live group reads alive
# ---------------------------------------------------------------------------
# `set -m` puts the child in a process group of its own, so its pid IS a pgid
# holding exactly one live process -- the shape all four call sites signal.
# That the child really leads its own group has to be CHECKED (a shell whose
# setpgrp does not take would otherwise have this leg quietly testing the
# harness's own group), so both legs need the pgid reader.
live_label="a group holding a running process reads as alive"
empty_label="a group with no members left reads as dead"
if [ "$have_pgid" = 0 ]; then
  skip "$live_label" "no way to read a process's group on this system"
  skip "$empty_label" "no way to read a process's group on this system"
else
  set -m
  sleep 30 >/dev/null 2>&1 </dev/null &
  livepid=$!
  set +m
  lpgid="$(ppgid "$livepid")"
  if [ "$lpgid" != "$livepid" ]; then
    report 1 "$live_label" \
      "the child did not lead its own group (pgid '${lpgid:-gone}' vs pid $livepid)"
  elif ! kill -0 -- "-$livepid" 2>/dev/null; then
    report 1 "$live_label" "the group is not even signal-reachable; the leg tested nothing"
  elif group_alive "$livepid"; then
    report 0 "$live_label"
  else
    report 1 "$live_label" "group_alive said dead for a group with a live sleep in it"
  fi

  # -------------------------------------------------------------------------
  # Leg 3: an empty group reads dead
  # -------------------------------------------------------------------------
  kill -KILL -- "-$livepid" 2>/dev/null
  wait "$livepid" 2>/dev/null   # reaped here, so the group really is empty
  livepid=""
  if kill -0 -- "-$lpgid" 2>/dev/null; then
    report 1 "$empty_label" "pgid $lpgid is still signal-reachable after the kill and reap"
  elif group_alive "$lpgid"; then
    report 1 "$empty_label" "group_alive said alive for an empty group"
  else
    report 0 "$empty_label"
  fi
fi

# ---------------------------------------------------------------------------
# Leg 4: a zombie-only group reads dead
# ---------------------------------------------------------------------------
# The negative control for the whole change: `kill -0` must still say YES here,
# or the leg would pass without exercising the difference. Under an init that
# reaps, such a corpse is transient and the bare probe merely lost a race with
# it; under one that does not -- the ordinary container case -- it is PERMANENT,
# so waiting it out was never the fix.
#
# Making a zombie that is reliably still a zombie when looked at, and that leaves
# nothing behind afterwards, takes care. The parent puts the child in a process
# group of ITS OWN (so the group holds the zombie and nothing else), then STOPs
# itself before the child exits: a stopped shell runs no SIGCHLD handler, so it
# cannot reap, and the child stays a zombie for as long as the leg needs.
#
# All of that needs a state reader, and needs it INDEPENDENT of the function
# under test: without one, "is it a zombie yet" reads the same as "it is gone",
# the leg would sit out its whole retry budget and then judge `group_alive` on a
# premise it never established -- and the cleanup assertion at the end would
# read an empty state as "reaped" and pass on a system that cannot see the
# corpse at all. So the whole leg, zombie maker included, is skipped there.
zlabel="a group holding nothing but a zombie reads as dead"
clabel="the state reader alone rejects a zombie-only group (signal probe forced to say alive)"
rlabel="the zombie leg reaps its own zombie, leaving no process-table entry"
ilabel="a zombie retains identity without being reported alive"
if [ "$have_state" = 0 ]; then
  skip "$zlabel" "no way to read a process's state on this system"
  skip "$clabel" "no way to read a process's state on this system"
  skip "$rlabel" "the zombie leg did not run, so it left nothing to reap"
  skip "$ilabel" "the zombie leg did not run, so identity was not testable"
else
zpidfile="$TMP/zombie.pid"; rm -f "$zpidfile"
bash -c 'set -m
         sleep 0.5 >/dev/null 2>&1 </dev/null &
         echo $! >"$1"
         set +m
         kill -STOP $$
         wait' _ "$zpidfile" >/dev/null 2>&1 </dev/null &
zparent=$!
zpid=""
for _ in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
  [ -s "$zpidfile" ] && zpid="$(cat "$zpidfile")" && break
  sleep 0.1
done
# Wait for real zombiehood rather than guessing at it, through `pstate`, which
# is this harness's own reader and shares no code with `group_alive`.
zstate=""
if [ -n "$zpid" ]; then
  for _ in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30; do
    zstate="$(pstate "$zpid")"
    case "$zstate" in Z*) break ;; esac
    sleep 0.1
  done
fi
if [ -z "$zpid" ]; then
  report 1 "$zlabel" "could not start the zombie maker"
  report 1 "$clabel" "could not start the zombie maker"
else
  case "$zstate" in
    Z*)
      # Whether the BARE probe over-reports here is a property of the kernel:
      # Linux (and every container on it) counts a zombie as a group member and
      # says alive, which is the bug; Darwin's killpg answers ESRCH once the
      # group holds only corpses, so the bare probe happens to agree there. The
      # claim itself holds on both, and the control below runs on both.
      if kill -0 -- "-$zpid" 2>/dev/null; then
        zwhere="the bare \`kill -0 -- -$zpid\` says ALIVE here -- the misreading this fixes"
      else
        zwhere="this kernel's killpg already answers dead for a zombie-only group"
      fi
      if group_alive "$zpid"; then
        report 1 "$zlabel" \
          "group_alive counted a zombie as work -- stop can report a phantom orphaned group"
      else
        report 0 "$zlabel ($zwhere)"
      fi
      printf '%s\n' "$zpid" >"$TMP/zombie-identity.pid"
      ps_token "$zpid" >"$TMP/zombie-identity.token"
      if proc_identity_matches "$TMP/zombie-identity.pid" "$TMP/zombie-identity.token" \
         && ! proc_alive "$TMP/zombie-identity.pid" "$TMP/zombie-identity.token"; then
        report 0 "$ilabel"
      else
        report 1 "$ilabel" \
          "the zombie either lost its recorded identity or was incorrectly reported live"
      fi
      # The negative control, run everywhere: shadow `kill` so the signal probe
      # inside group_alive succeeds, which is exactly what the bare check did on
      # the kernel where this was reproduced. The state reader must still refuse
      # the group -- otherwise this platform's pass above was the gate's doing
      # alone and the ladder itself is untested here.
      (
        kill() { case "$*" in -0*) return 0 ;; *) command kill "$@" ;; esac; }
        if ! kill -0 -- "-$zpid"; then
          exit 3   # the shadow did not take; the control tested nothing
        fi
        group_alive "$zpid" && exit 1
        exit 0
      )
      case $? in
        0) report 0 "$clabel" ;;
        3) report 1 "$clabel" 'the `kill` shadow did not take effect' ;;
        *) report 1 "$clabel" \
             "with the signal probe forced alive, group_alive called a zombie-only group alive" ;;
      esac
      ;;
    *)
      report 1 "$zlabel" \
        "the child never reached state Z (saw '${zstate:-gone}'), so the leg tested nothing"
      report 1 "$clabel" \
        "the child never reached state Z (saw '${zstate:-gone}'), so the leg tested nothing"
      report 1 "$ilabel" \
        "the child never reached state Z (saw '${zstate:-gone}'), so identity was not tested"
      ;;
  esac
fi
# Let the parent reap its own child rather than orphaning the zombie, then check
# we really left nothing behind.
kill -CONT "$zparent" 2>/dev/null
wait "$zparent" 2>/dev/null
zparent=""
zleft=""
if [ -n "$zpid" ]; then
  for _ in 1 2 3 4 5 6 7 8 9 10; do
    zleft="$(pstate "$zpid")"
    [ -z "$zleft" ] && break
    sleep 0.1
  done
fi
if [ -z "$zleft" ]; then
  report 0 "$rlabel"
else
  report 1 "$rlabel" "pid $zpid is still present as '$zleft'"
fi
fi

# ---------------------------------------------------------------------------
# Leg 5: only a positive decimal integer is a pgid
# ---------------------------------------------------------------------------
# A corrupted or forged pgid file must never reach kill: `kill -0 -- -0` targets
# the CALLER's own group (which is alive, so the answer would be a confident
# yes) and a negative reading broadcasts.
bad=""
for candidate in "" "0" "12x" "-1" "abc" " 7"; do
  if group_alive "$candidate"; then bad="$bad '$candidate'"; fi
done
if [ -z "$bad" ]; then
  report 0 "a pgid that is not a positive decimal integer is refused"
else
  report 1 "a pgid that is not a positive decimal integer is refused" \
    "accepted:$bad"
fi

# ---------------------------------------------------------------------------
# Legs 6-7: the two sentences `stop` prints about a surviving process group
# ---------------------------------------------------------------------------
# They are driven for real: `tools/test-run.sh stop last` against a FORGED run
# directory -- the metadata a launch records (cmd, cap, wt, log, pgid, gtoken)
# written by hand around a process group this harness controls -- and the answer
# read from what stop actually printed.
#
# Nothing of the ambient run history is touched: OCANNL_TOOL_TEST_RUNS is
# pointed at this run's temp dir, so the `last` pointer these legs move is the
# fixture's own and dies with the temp dir. `last` rather than an explicit run
# directory, because that is the spelling an operator uses and it exercises the
# pointer too; the public `paths last` query supplies its filename without
# reconstructing the shipping script's worktree-key implementation.
#
# The fixture deliberately records no pid/ptoken and leaves no `exit` file: a
# run with either is owned or finished, and the group branch is the one reached
# by a run whose supervisor is gone while its process group is not.
l_ignored="stop: a group whose leader ignores TERM is reported with the unreaped-exits caveat"
l_killed="stop: that escalation kills the whole group, not just its leader"
l_took="stop: a group whose leader takes the TERM is reported as TERMed, not as ignoring it"

STOP_RUNS="$TMP/runs"
STOP_WT="$TMP/wt"
mkdir -p "$STOP_RUNS" "$STOP_WT"
STOP_RUNS=$(cd "$STOP_RUNS" && pwd -P)
STOP_WT=$(cd "$STOP_WT" && pwd -P)
stop_last=$(OCANNL_TOOL_TEST_RUNS="$STOP_RUNS" "$SRC" paths last) || exit 1

# A leg that cannot establish its premise must skip, not pass: without a pgid
# reader the fixture could name THIS shell's group and stop would signal the
# harness itself, and without a start token group_verified refuses the fixture
# and every leg reads the same "nothing left to signal".
stop_skip=""
if [ "$have_pgid" = 0 ]; then
  stop_skip="no way to read a process's group, so a fixture group cannot be told from this shell's own"
elif [ -z "$(ps_token $$)" ]; then
  stop_skip="this system records no start token, so a forged leader cannot be identity-verified"
fi

mk_fixture() { # <tag> <pgid> <last pointer path>; 0 iff the run is reachable as `last`
  local d="$STOP_RUNS/19700101-000000-$1"
  mkdir -p "$d" "$STOP_WT" || return 1
  # cmd and cap are what resolve_run demands before it will trust a directory
  # enough to signal anything named in it.
  printf 'runtest (test-test-run.sh fixture %s)\n' "$1" >"$d/cmd"
  printf '0\n' >"$d/cap"
  printf '%s\n' "$STOP_WT" >"$d/wt"
  # `runs` names the state root the lock would live under; without it the
  # script reads the run as one of the version that kept its lock in the tree.
  printf '%s\n' "$STOP_RUNS" >"$d/runs"
  : >"$d/log"
  printf '%s\n' "$2" >"$d/pgid"
  ps_token "$2" >"$d/gtoken"
  # group_verified refuses an empty token outright, so it is checked here
  # rather than left to reappear as a wording failure three legs later.
  [ -s "$d/gtoken" ] || return 1
  printf '%s\n' "$d" >"$3" || return 1
  return 0
}

start_leader() { # <marker> <bash -c body>; sets $leader (and its $member)
  local m=$1 body=$2 p g i
  leader=""; member=""; member_token=""
  rm -f "$m"
  # `set -m` puts the child in its own process group -- the shape stop signals,
  # and the only shape it is safe to hand a group kill.
  set -m
  bash -c "$body" _ "$m" >/dev/null 2>&1 </dev/null &
  p=$!
  set +m
  # Recorded BEFORE the checks below rather than after them: job control has
  # just put this child out of reach of any signal aimed at the harness, so an
  # INT arriving in the window between the fork and the recording would leave
  # the EXIT cleanup with nothing to kill and the fixture group running past
  # the run (Codex round 1, P2). Recording early is safe in the case the checks
  # are about to reject: a group kill aimed at a pid that never led a group
  # names a group that does not exist.
  leader=$p
  # The body writes the marker -- carrying the pid of the SECOND member it put
  # in the group -- after installing its TERM disposition, so this wait is what
  # makes the leg's premise true rather than merely likely.
  for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
    [ -s "$m" ] && break
    sleep 0.1
  done
  g="$(ppgid "$p")"
  member="$(tr -dc '0-9' <"$m" 2>/dev/null)"
  member_token="$(ps_token "$member" 2>/dev/null)"
  # The second member is checked into the group, not assumed into it: a fixture
  # whose extra process ended up somewhere else would leave leg 6 judging the
  # escalation on the leader alone again, which is the hole it exists to close.
  if [ ! -s "$m" ] || [ "$g" != "$p" ] || [ -z "$member" ] ||
     [ "$(ppgid "$member")" != "$p" ]; then
    end_leader
    return 1
  fi
  return 0
}
kill_member() { # KILL the group's second member -- only if it is still IT
  # Unlike the leader, the member is a GRANDchild: nothing here holds it as a
  # zombie, so once `stop` has killed it, init reaps it and its pid is free to
  # be recycled. An unconditional numeric kill from the EXIT trap could then
  # land on an unrelated process on a busy host (Codex round 2, P2). The gate
  # is the same one the shipping script uses for every pid it did not just
  # fork: the recorded start token has to still be the one that pid answers
  # with. (lstart's one-second resolution is the known floor there, which is
  # why test-run.sh prefers /proc starttime where it exists.)
  [ -n "${member:-}" ] && [ -n "${member_token:-}" ] || return 0
  [ "$(ps_token "$member" 2>/dev/null)" = "$member_token" ] || return 0
  kill -KILL "$member" 2>/dev/null
  return 0
}
end_leader() { # whatever the leg concluded, the whole fixture group goes
  if [ -n "${leader:-}" ]; then
    kill -KILL -- "-$leader" 2>/dev/null
    kill -KILL "$leader" 2>/dev/null
  fi
  # Named separately as well as reached through the group: this is the
  # harness's backstop for the very defect leg 6 now tests for, an escalation
  # that reached only the leader. The leader needs no such check -- it is this
  # shell's own child, held as a zombie until the `wait` below, so its pid
  # cannot be recycled underneath us.
  kill_member
  if [ -n "${leader:-}" ]; then wait "$leader" 2>/dev/null; fi
  leader=""; member=""; member_token=""
  return 0
}
# Each fixture group holds TWO processes, because a group holding only its
# leader cannot tell a group kill from a leader kill: with one member, the
# incorrect `kill -KILL "$pg"` passes every leg here while real dune children
# would survive it (Codex round 1, P2). The member is a plain background
# `sleep` the leader forks before exec'ing its own, so it is in the group and
# inherits the leader's TERM disposition -- SIG_IGN survives both fork and
# exec, so an IGNORING fixture is TERM-proof as a whole, and a TAKING one dies
# as a whole. The marker doubles as the member's pid, so the harness can track
# a process that is not its own child.
body_ignores='trap "" TERM; sleep 600 & echo $! >"$1"; exec sleep 600'
body_takes='sleep 600 & echo $! >"$1"; exec sleep 600'
stop_out=""; stop_rc=""; stop_pg=""; stop_member=""; stop_err=""; stop_diag=""
stop_probe() { # <tag> <leader body> <script> <last pointer path>
  stop_out=""; stop_rc=""; stop_pg=""; stop_member=""; stop_err=""; stop_diag=""
  if ! start_leader "$TMP/$1.marker" "$2"; then
    stop_err="could not start a two-process group leader for the '$1' fixture"
    return 1
  fi
  # Kept past end_leader, which clears the live handles: the escalation claim
  # is about processes that are supposed to be gone by the time it is asked.
  stop_pg=$leader
  stop_member=$member
  if ! mk_fixture "$1" "$stop_pg" "$4"; then
    stop_err="could not build the '$1' fixture run directory under $STOP_RUNS"
    return 1
  fi
  # stdout and stderr kept APART. The sentence is stdout, and it is matched
  # whole; stderr carries diagnostics that are not part of the answer and can
  # appear for reasons that have nothing to do with the wording -- a /proc
  # entry vanishing under group_alive's scan being the one that actually bit
  # (Codex round 2, P2). Merging the two made a passing stop fail an exact
  # match. It is kept and shown on failure rather than discarded, since a
  # failing leg is exactly when it is worth reading.
  stop_out="$(OCANNL_TOOL_TEST_RUNS="$STOP_RUNS" "$3" stop last 2>"$TMP/$1.stderr")"
  stop_rc=$?
  stop_diag="$(cat "$TMP/$1.stderr" 2>/dev/null)"
  return 0
}
said() { # <label> <the whole line stop must have printed>
  # The WHOLE output, not a substring of it, and a clean exit with it: these
  # two legs exist to keep the two sentences apart, and a containment test
  # would pass both for a stop that printed two of them at once (Codex
  # round 1, P2).
  if [ "${stop_rc:-1}" = 0 ] && [ "$stop_out" = "$2" ]; then
    report 0 "$1"
  else
    report 1 "$1" \
      "expected exactly \"$2\"; stop (exit ${stop_rc:-?}) printed: ${stop_out:-<nothing>}"
    [ -n "${stop_diag:-}" ] && printf '      on stderr: %s\n' "$stop_diag"
  fi
}

if [ -n "$stop_skip" ]; then
  skip "$l_ignored" "$stop_skip"
  skip "$l_killed" "$stop_skip"
  skip "$l_took" "$stop_skip"
else
  # -------------------------------------------------------------------------
  # Leg 6: the leader ignores TERM
  # -------------------------------------------------------------------------
  if stop_probe ignores "$body_ignores" "$SRC" "$stop_last"; then
    said "$l_ignored" \
      "orphaned process group $stop_pg survived TERM (possibly only as unreaped exited processes); escalated to KILL"
    # The sentence is a claim about what stop DID, so the doing is checked too:
    # an escalation that only announced itself would leave the group holding the
    # worktree lock, which is the whole reason stop reaches for KILL here.
    if [ "$have_state" = 0 ]; then
      skip "$l_killed" "no way to read a process's state on this system"
    else
      # Both members, so a KILL that reached only the leader fails here.
      k_left=""
      for _ in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
        k_left=""
        for k_pid in $stop_pg $stop_member; do
          case "$(pstate "$k_pid")" in '' | Z*) ;; *) k_left="$k_left $k_pid" ;; esac
        done
        [ -z "$k_left" ] && break
        sleep 0.1
      done
      if [ -z "$k_left" ]; then
        report 0 "$l_killed"
      else
        report 1 "$l_killed" \
          "still running two seconds after the escalation:$k_left (group $stop_pg)"
      fi
    fi
  else
    report 1 "$l_ignored" "$stop_err"
    report 1 "$l_killed" "$stop_err"
  fi
  end_leader

  # -------------------------------------------------------------------------
  # Leg 7: the leader takes the TERM
  # -------------------------------------------------------------------------
  # The difference from leg 6 is one `trap` in the leader and nothing else, so
  # a stop that reported both the same way would fail exactly one of them.
  if stop_probe takes "$body_takes" "$SRC" "$stop_last"; then
    said "$l_took" "sent TERM to the orphaned process group $stop_pg; re-run stop to confirm"
  else
    report 1 "$l_took" "$stop_err"
  fi
  end_leader

  # -------------------------------------------------------------------------
  # Leg 8: a legacy run's leftover is reaped through its in-tree lock
  # -------------------------------------------------------------------------
  # The version before gh-ocannl-606 kept the lock and its owner pointer
  # BESIDE the worktree, and a run directory it recorded has no `runs` root.
  # Its leftovers -- a descendant dune holding the inherited lock fd -- hold
  # THAT lock, so `stop` must read the run under those paths or it reports
  # nothing to reap while the lock stays held. The fixture: a run directory
  # with no `runs`, no pid and no pgid (only the lock can attribute
  # anything), the in-tree owner pointer naming it, and a perl holding an
  # flock on the in-tree lock file, which stop's census must find and TERM.
  l_legacy="stop: a legacy run's leftover holding the in-tree lock is attributed and reaped"
  legacy_dir=$STOP_RUNS/19700101-000000-legacy
  mkdir -p "$legacy_dir" "$STOP_WT"
  printf 'runtest (test-test-run.sh fixture legacy)\n' >"$legacy_dir/cmd"
  printf '0\n' >"$legacy_dir/cap"
  printf '%s\n' "$STOP_WT" >"$legacy_dir/wt"
  : >"$legacy_dir/log"
  printf '%s\n' "$legacy_dir" >"$STOP_WT/.test-run.lock.owner"
  printf '%s\n' "$legacy_dir" >"$stop_last"
  rm -f "$TMP/legacy.pid"
  perl -e 'use Fcntl ":flock";
           open(my $fh, ">>", $ARGV[0]) or exit 1;
           flock($fh, LOCK_EX | LOCK_NB) or exit 1;
           $| = 1; print "$$\n"; sleep 600' \
    "$STOP_WT/.test-run.lock" >"$TMP/legacy.pid" 2>/dev/null </dev/null &
  legacy_holder=$!
  for _ in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
    [ -s "$TMP/legacy.pid" ] && break
    sleep 0.1
  done
  # The premise, checked: the lock really is held before stop is asked.
  legacy_lock_out=$(OCANNL_TOOL_TEST_RUNS="$STOP_RUNS" "$SRC" lock-status last)
  legacy_lock_rc=$?
  if [ ! -s "$TMP/legacy.pid" ] || [ "$legacy_lock_rc" != 3 ] || [ "$legacy_lock_out" != held ]; then
    report 1 "$l_legacy" "the fixture holder did not take the in-tree lock"
  else
    stop_out="$(OCANNL_TOOL_TEST_RUNS="$STOP_RUNS" "$SRC" stop last 2>"$TMP/legacy.stderr")"
    stop_rc=$?
    stop_diag="$(cat "$TMP/legacy.stderr" 2>/dev/null)"
    said "$l_legacy" \
      "run is dead without a verdict, but its leftover processes held the worktree lock; reaped them"
    # The claim is about what stop DID: the holder must be gone and the lock free.
    for _ in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
      kill -0 "$legacy_holder" 2>/dev/null || break
      sleep 0.1
    done
    if kill -0 "$legacy_holder" 2>/dev/null; then
      report 1 "$l_legacy (the holder is gone and the lock is free)" \
        "pid $legacy_holder still holds $STOP_WT/.test-run.lock after stop"
    elif OCANNL_TOOL_TEST_RUNS="$STOP_RUNS" "$SRC" lock-status last >"$TMP/legacy-lock-after" \
         && [ "$(cat "$TMP/legacy-lock-after")" = idle ]; then
      report 0 "$l_legacy (the holder is gone and the lock is free)"
    else
      report 1 "$l_legacy (the holder is gone and the lock is free)" \
        "$STOP_WT/.test-run.lock is still locked after stop"
    fi
  fi
  if kill -0 "$legacy_holder" 2>/dev/null; then kill -KILL "$legacy_holder" 2>/dev/null; fi
  wait "$legacy_holder" 2>/dev/null
  legacy_holder=""
  rm -f "$STOP_WT/.test-run.lock" "$STOP_WT/.test-run.lock.owner"
fi

# ---------------------------------------------------------------------------
# Legs 9-17: repeat mode's output, lifecycle and exit-code contract
# ---------------------------------------------------------------------------
repeat_root=$TMP/repeat-repo
repeat_bin=$TMP/repeat-bin
mkdir -p "$repeat_root/tools" "$repeat_root/scripts" "$repeat_bin"
cp "$SRC" "$repeat_root/tools/test-run.sh"
cp "$GROUP_SRC" "$repeat_root/scripts/process-group.sh"
chmod +x "$repeat_root/tools/test-run.sh"
# Read-only query contract: no state store, no first run, and no lock file yet.
# The fixture root (not the caller's cwd) owns all omitted-RUN queries.
query_runs=$TMP/query-state/missing/../runs
query_expected=$(cd "$TMP" && pwd -P)/query-state/runs
query() {
  OCANNL_TOOL_TEST_RUNS=$query_runs "$repeat_root/tools/test-run.sh" "$@"
}
query_rc=0
query_out=$(query paths runs) || query_rc=$?
query_wt=$(query paths worktree)
query_lock=$(query paths lock)
query_owner=$(query paths owner)
query_last=$(query paths last)
query_idle=$(query lock-status) || query_rc=$?
if [ "$query_rc" = 0 ] && [ "$query_out" = "$query_expected" ] \
   && [ "$query_wt" = "$(cd "$repeat_root" && pwd -P)" ] \
   && [ "$query_idle" = idle ] && [ ! -e "$TMP/query-state" ] \
   && [ -n "$query_lock" ] && [ -n "$query_owner" ] && [ -n "$query_last" ]; then
  report 0 "queries: absent state resolves physically without creating files"
else
  report 1 "queries: absent state resolves physically without creating files" \
    "rc=$query_rc runs=$query_out worktree=$query_wt lock=$query_lock idle=$query_idle"
fi
query_bad=
for args in 'paths' 'paths bogus' 'paths lock last extra' 'paths run' \
            'paths run missing' 'paths lock missing' 'lock-status last' \
            'lock-status last extra'; do
  query_out=$(query $args 2>"$TMP/query-error")
  query_rc=$?
  if [ "$query_rc" != 2 ] || [ -n "$query_out" ] || [ ! -s "$TMP/query-error" ]; then
    query_bad="$args: rc=$query_rc output=$query_out"; break
  fi
done
if [ -z "$query_bad" ] && [ ! -e "$TMP/query-state" ]; then
  report 0 "queries: malformed arguments and missing runs fail without state mutation"
else
  report 1 "queries: malformed arguments and missing runs fail without state mutation" "$query_bad"
fi

# Physical normalization must agree with launch even when the caller uses a
# symlinked script or state root. MSYS may copy rather than create a symlink.
ln -s "$repeat_root" "$TMP/query-repo-link" 2>/dev/null
ln -s "$repeat_root" "$TMP/query-state-link" 2>/dev/null
if [ -L "$TMP/query-repo-link" ] && [ -L "$TMP/query-state-link" ]; then
  query_out=$(OCANNL_TOOL_TEST_RUNS="$TMP/query-state-link/nonexistent/runs" \
    "$TMP/query-repo-link/tools/test-run.sh" paths runs)
  query_rc=$?
  if [ "$query_rc" = 0 ] && [ "$query_out" = "$query_wt/nonexistent/runs" ] \
     && [ ! -e "$repeat_root/nonexistent" ]; then
    report 0 "queries: symlink spellings identify the same physical worktree and state root"
  else
    report 1 "queries: symlink spellings identify the same physical worktree and state root" \
      "rc=$query_rc runs=$query_out expected=$query_wt/nonexistent/runs"
  fi
else
  skip "queries: symlink spellings identify the same physical worktree and state root" \
    "native symlinks unavailable"
fi

# The pointer returned by paths last must feed the SAME resolver as status and
# stop; the published fixture has deliberately no live pid and no verdict.
mkdir -p "$query_expected/fixture"
query_runs=$query_expected
query_dir=$query_expected/fixture
printf 'fixture\n' >"$query_dir/cmd"
printf '0\n' >"$query_dir/cap"
printf '%s\n' "$query_wt" >"$query_dir/wt"
printf '%s\n' "$query_runs" >"$query_dir/runs"
printf '%s\n' "$query_dir" >"$query_last"
query_bad=
for ref in last fixture "$query_dir"; do
  [ "$(query paths run "$ref")" = "$query_dir" ] \
    && [ "$(query paths lock "$ref")" = "$query_lock" ] \
    && [ "$(query paths owner "$ref")" = "$query_owner" ] \
    && [ "$(query paths worktree "$ref")" = "$query_wt" ] \
    && [ "$(query paths runs "$ref")" = "$query_runs" ] \
    && [ "$(query paths last "$ref")" = "$query_last" ] \
    || query_bad="$ref did not resolve the published fixture"
done
query_out=$(query lock-status last); query_rc=$?
if [ -z "$query_bad" ] && [ "$query_rc" = 0 ] && [ "$query_out" = idle ] \
   && [ ! -e "$query_lock" ] && [ ! -e "$query_owner" ]; then
  report 0 "queries: last, bare id and absolute run share recorded paths without creating locks"
else
  report 1 "queries: last, bare id and absolute run share recorded paths without creating locks" "$query_bad"
fi

# A held lock needs no process metadata to be reported held. Run the holder
# synchronously; Perl closes its nonstandard descriptors when execing the query.
query_out=$(perl -MFcntl=:flock -e '
  open my $fh, ">>", $ARGV[0] or die $!;
  flock($fh, LOCK_EX | LOCK_NB) or die $!;
  system @ARGV[1..$#ARGV];
  exit($? >> 8);
' "$query_lock" env OCANNL_TOOL_TEST_RUNS="$query_runs" \
  "$repeat_root/tools/test-run.sh" lock-status last)
query_rc=$?
query_after=$(query lock-status); query_after_rc=$?
if [ "$query_rc" = 3 ] && [ "$query_out" = held ] \
   && [ "$query_after_rc" = 0 ] && [ "$query_after" = idle ] \
   && [ ! -s "$query_lock" ]; then
  report 0 "queries: occupied and released locks have distinct statuses without writes"
else
  report 1 "queries: occupied and released locks have distinct statuses without writes" \
    "held=$query_rc:$query_out released=$query_after_rc:$query_after"
fi
rm "$query_lock"
mkdir "$query_lock"
query_out=$(query lock-status 2>"$TMP/query-error"); query_rc=$?
if [ "$query_rc" = 2 ] && [ -z "$query_out" ] && [ -d "$query_lock" ]; then
  report 0 "queries: an uninspectable lock is an error, never idle"
else
  report 1 "queries: an uninspectable lock is an error, never idle" "rc=$query_rc output=$query_out"
fi
rmdir "$query_lock"
# Legacy paths come from the recorded worktree, not the querying script's root.
rm "$query_dir/runs"
printf '%s\n' "$STOP_WT" >"$query_dir/wt"
if [ "$(query paths lock last)" = "$STOP_WT/.test-run.lock" ] \
   && [ "$(query paths owner last)" = "$STOP_WT/.test-run.lock.owner" ]; then
  report 0 "queries: legacy runs retain their recorded in-tree lock paths"
else
  report 1 "queries: legacy runs retain their recorded in-tree lock paths"
fi

cat >"$repeat_bin/dune" <<'EOF'
#!/usr/bin/env bash
set -u
# Close the inherited lock descriptor before probing through a fresh open.
# Acquiring here would prove repeat released its one set-wide lock too early.
# Ask the production read-only API; a held answer must come from the actual
# lock, not a glob over implementation-private filenames. A separate absent
# lock control below prevents a constant "held" query from blessing this leg.
lock_probe() {
  local answer rc
  answer=$(tools/test-run.sh lock-status 9>&-)
  rc=$?
  if [ "$rc" != 3 ] || [ "$answer" != held ]; then
    echo "repeat fixture expected held lock, got $rc: $answer" >&2
    exit 91
  fi
}
lock_probe
# Repeat establishes a fresh context with `dune clean` before each measured
# invocation. The fixture keeps setup out of the iteration count and streams.
if [ "${1:-}" = clean ]; then
  printf 'clean %s\n' "$*" >>"$REPEAT_TEST_CALLS"
  exit 0
fi
n=0
[ ! -f "$REPEAT_TEST_COUNTER" ] || n=$(cat "$REPEAT_TEST_COUNTER")
n=$((n + 1))
printf '%s\n' "$n" >"$REPEAT_TEST_COUNTER"
printf '%s\n' "$*" >>"$REPEAT_TEST_CALLS"
if [ -n "${REPEAT_TEST_WAIT_PREFIX:-}" ] \
   && { [ -z "${REPEAT_TEST_WAIT_AT:-}" ] || [ "$REPEAT_TEST_WAIT_AT" = "$n" ]; }; then
  : >"$REPEAT_TEST_WAIT_PREFIX.ready"
  while [ ! -e "$REPEAT_TEST_WAIT_PREFIX.release" ]; do sleep 0.05; done
fi
case $REPEAT_TEST_MODE in
  stable) printf 'stable stdout\n'; printf 'stable stderr\n' >&2 ;;
  stdout) printf 'stdout %s\n' "$n"; printf 'stable stderr\n' >&2 ;;
  stderr) printf 'stable stdout\n'; printf 'stderr %s\n' "$n" >&2 ;;
  fail) printf 'stable stdout\n'; printf 'stable failure\n' >&2; exit 7 ;;
  fail_first)
    if [ "$n" = 1 ]; then printf 'first failure\n' >&2; exit 7; fi
    printf 'later stdout\n'; printf 'later stderr\n' >&2
    ;;
  orphan_first)
    if [ "$n" = 1 ]; then
      printf '%s\n' "$$" >"$REPEAT_TEST_ORPHAN_PID"
      trap ': >"$REPEAT_TEST_ORPHAN_REAPED"; exit 0' TERM
      kill -KILL "$PPID"
      while :; do sleep 1; done
    fi
    [ -e "$REPEAT_TEST_ORPHAN_REAPED" ] || exit 93
    printf 'post-reap stdout\n'; printf 'post-reap stderr\n' >&2
    ;;
  session_escape)
    for arg in "$@"; do
      case $arg in
        --build-dir=*) mkdir -p "${arg#--build-dir=}" ;;
      esac
    done
    perl -MPOSIX -e '
      $SIG{HUP} = "IGNORE";
      POSIX::setsid() >= 0 or die "setsid: $!";
      open my $pf, ">", $ENV{REPEAT_TEST_ESCAPE_PID} or die;
      print $pf "$$\n"; close $pf;
      select undef, undef, undef, 0.05 until -e $ENV{REPEAT_TEST_ESCAPE_RELEASE};
    ' </dev/null >/dev/null 2>&1 &
    for _ in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
      [ -s "$REPEAT_TEST_ESCAPE_PID" ] && break
      sleep 0.05
    done
    printf 'escaped parent stdout\n'; printf 'escaped parent stderr\n' >&2
    ;;
  # dune's own command-line refusals, verbatim from dune 3.24 (cmdliner): the
  # `dune: <complaint>` / `Usage: dune ...` pair on stderr, exit 1, nothing
  # else -- the operand one wraps its complaint onto an indented continuation
  # line, which the digest must quote whole.
  usage_option)
    printf "dune: unknown option '--frobnicate'.\nUsage: dune build [OPTION]… [TARGET]…\nTry 'dune build --help' or 'dune --help' for more information.\n" >&2
    exit 1 ;;
  usage_command)
    printf "dune: unknown command 'frobnicate', must be one of 'build', 'clean', 'exec', 'runtest' or 'test'.\nUsage: dune COMMAND …\nTry 'dune --help' for more information.\n" >&2
    exit 1 ;;
  usage_operand)
    printf "dune: option '-j': invalid concurrency value, must be 'auto' or a positive\n      number\nUsage: dune build [OPTION]… [TARGET]…\nTry 'dune build --help' or 'dune --help' for more information.\n" >&2
    exit 1 ;;
  # An ordinary red run: a located error, exit 1 -- and the same with a test's
  # own output opening on a `dune:` line, which no `Usage:` line follows.
  red)
    printf 'File "test/operations/fixture.ml", line 3, characters 4-9:\nError: This expression has type int but an expression was expected of type float\n' >&2
    exit 1 ;;
  dune_prefixed_red)
    printf 'dune: is what this test prints first\n'
    printf 'File "test/operations/fixture.ml", line 3, characters 4-9:\nError: This expression has type int but an expression was expected of type float\n' >&2
    exit 1 ;;
  # A program dune DID run (`dune exec`) that shells out to dune with bad
  # arguments, prints that nested refusal first, then fails on its own: the
  # complete refusal shape, followed by evidence that something ran -- a
  # located error in one, plain output in the other.
  nested_usage_red)
    printf "dune: unknown option '--frobnicate'.\nUsage: dune build [OPTION]… [TARGET]…\nTry 'dune build --help' or 'dune --help' for more information.\n" >&2
    printf 'File "test/operations/fixture.ml", line 3, characters 4-9:\nError: This expression has type int but an expression was expected of type float\n' >&2
    exit 1 ;;
  nested_usage_output)
    printf "dune: unknown option '--frobnicate'.\nUsage: dune build [OPTION]… [TARGET]…\nTry 'dune build --help' or 'dune --help' for more information.\n" >&2
    printf 'the program went on to print this and then failed\n'
    exit 1 ;;
  *) echo "unknown repeat fixture mode: $REPEAT_TEST_MODE" >&2; exit 92 ;;
esac
EOF
chmod +x "$repeat_bin/dune"
cat >"$repeat_bin/diff" <<'EOF'
#!/usr/bin/env bash
set -u
if [ -n "${REPEAT_TEST_DIFF_WAIT_PREFIX:-}" ]; then
  : >"$REPEAT_TEST_DIFF_WAIT_PREFIX.ready"
  while [ ! -e "$REPEAT_TEST_DIFF_WAIT_PREFIX.release" ]; do /bin/sleep 0.05; done
fi
exec "$REPEAT_TEST_REAL_DIFF" "$@"
EOF
chmod +x "$repeat_bin/diff"

repeat_out= repeat_rc= repeat_dir=
await_fixture_ready() { # MARKER [ATTEMPTS] -- never assume the fixture arrived
  local marker=$1 attempts=${2:-300} attempt=0
  while [ "$attempt" -lt "$attempts" ]; do
    [ -e "$marker" ] && return 0
    sleep 0.1
    attempt=$((attempt + 1))
  done
  return 1
}
# Negative control for the timeout half of the helper's contract. The live
# fixtures below prove its success half; this absent marker must be refused.
if await_fixture_ready "$TMP/repeat-never-ready" 1; then
  echo "repeat fixture readiness accepted an absent marker" >&2
  exit 2
fi
repeat_probe() { # tag mode [repeat options/count/dune argv...]
  local tag=$1 mode=$2 runs=$TMP/repeat-runs-$1
  shift 2
  mkdir -p "$runs"
  : >"$TMP/$tag.counter"
  : >"$TMP/$tag.calls"
  REPEAT_TEST_MODE=$mode \
  REPEAT_TEST_COUNTER=$TMP/$tag.counter \
  REPEAT_TEST_CALLS=$TMP/$tag.calls \
  REPEAT_TEST_WAIT_PREFIX= \
  REPEAT_TEST_WAIT_AT= \
  REPEAT_TEST_ORPHAN_PID= \
  REPEAT_TEST_ORPHAN_REAPED= \
  REPEAT_TEST_DIFF_WAIT_PREFIX= \
  REPEAT_TEST_REAL_DIFF="$(command -v diff)" \
  OCANNL_TOOL_TEST_RUNS=$runs \
  PATH=$repeat_bin:$PATH \
    "$repeat_root/tools/test-run.sh" repeat "$@" >"$TMP/$tag.out" 2>"$TMP/$tag.err"
  repeat_rc=$?
  repeat_out=$(cat "$TMP/$tag.out")
  repeat_dir=$(OCANNL_TOOL_TEST_RUNS="$runs" "$repeat_root/tools/test-run.sh" paths run last 2>/dev/null)
}

# This is an ordering invariant, not a timing lottery: once publish_run writes
# `last`, an external stop may arrive on the next instruction. Pin all three
# prerequisites above that line so a future refactor cannot reopen the gap.
completed_line=$(grep -n '^    completed=0$' "$SRC" | cut -d: -f1)
signal_trap_line=$(grep -n "^    trap 'repeat_signal TERM' TERM$" "$SRC" | cut -d: -f1)
exit_trap_line=$(grep -n '^    trap repeat_exit EXIT$' "$SRC" | cut -d: -f1)
publish_line=$(grep -n '^    publish_run || die "cannot publish repeat run ' "$SRC" | cut -d: -f1)
if [ -n "$completed_line" ] && [ -n "$signal_trap_line" ] \
   && [ -n "$exit_trap_line" ] && [ -n "$publish_line" ] \
   && [ "$completed_line" -lt "$publish_line" ] \
   && [ "$signal_trap_line" -lt "$publish_line" ] \
   && [ "$exit_trap_line" -lt "$publish_line" ]; then
  report 0 "repeat: cancellation lifecycle is armed before publication"
else
  report 1 "repeat: cancellation lifecycle is armed before publication" \
    "completed=${completed_line:-missing} signal=${signal_trap_line:-missing} exit=${exit_trap_line:-missing} publish=${publish_line:-missing}"
fi

# A real red remains the useful verdict when a later iteration is cancelled.
# Block only iteration two, stop the managed set there, and prove both the
# retained exit 7 and the cancellation annotation.
repeat_red_cancel_runs=$TMP/repeat-runs-red-cancel
repeat_red_cancel_prefix=$TMP/repeat-red-cancel
mkdir -p "$repeat_red_cancel_runs"
: >"$TMP/repeat-red-cancel.counter"
: >"$TMP/repeat-red-cancel.calls"
REPEAT_TEST_MODE=fail_first \
REPEAT_TEST_COUNTER=$TMP/repeat-red-cancel.counter \
REPEAT_TEST_CALLS=$TMP/repeat-red-cancel.calls \
REPEAT_TEST_WAIT_PREFIX=$repeat_red_cancel_prefix \
REPEAT_TEST_WAIT_AT=2 \
REPEAT_TEST_DIFF_WAIT_PREFIX= \
REPEAT_TEST_REAL_DIFF="$(command -v diff)" \
REPEAT_TEST_ORPHAN_PID= \
REPEAT_TEST_ORPHAN_REAPED= \
OCANNL_TOOL_TEST_RUNS=$repeat_red_cancel_runs \
PATH=$repeat_bin:$PATH \
  "$repeat_root/tools/test-run.sh" repeat 3 build @cheap \
  >"$TMP/repeat-red-cancel.out" 2>"$TMP/repeat-red-cancel.err" &
repeat_pid=$!
if ! await_fixture_ready "$repeat_red_cancel_prefix.ready"; then
  report 1 "repeat: earlier failure survives later cancellation" \
    "setup timeout waiting for iteration two's readiness marker"
  exit 1
fi
red_cancel_stop=$(OCANNL_TOOL_TEST_RUNS=$repeat_red_cancel_runs \
  "$repeat_root/tools/test-run.sh" stop last 2>"$TMP/repeat-red-cancel-stop.err")
red_cancel_stop_rc=$?
touch "$repeat_red_cancel_prefix.release"
wait "$repeat_pid"
repeat_red_cancel_rc=$?
repeat_pid=
repeat_red_cancel_dir=$(OCANNL_TOOL_TEST_RUNS="$repeat_red_cancel_runs" "$repeat_root/tools/test-run.sh" paths run last 2>/dev/null)
if [ "$red_cancel_stop_rc" = 0 ] \
   && grep -q '^sent TERM to the repeat coordinator; ' <<<"$red_cancel_stop" \
   && [ "$repeat_red_cancel_rc" = 7 ] \
   && [ "$(cat "$repeat_red_cancel_dir/exit" 2>/dev/null)" = 7 ] \
   && grep -q '^repeat result: CANCELLED -- completed 2 of 3 iterations$' "$TMP/repeat-red-cancel.out"; then
  report 0 "repeat: earlier failure survives later cancellation"
else
  report 1 "repeat: earlier failure survives later cancellation" \
    "stop $red_cancel_stop_rc; repeat $repeat_red_cancel_rc: $(cat "$TMP/repeat-red-cancel.out")"
fi

# Kill the first capped supervisor from inside its Dune child, leaving that
# child/group alive. Iteration two refuses to pass unless the coordinator's
# identity-verified reap completed first.
repeat_orphan_runs=$TMP/repeat-runs-orphan
repeat_orphan_marker=$TMP/repeat-orphan-reaped
repeat_orphan_pid_file=$TMP/repeat-orphan-pid
mkdir -p "$repeat_orphan_runs"
: >"$TMP/repeat-orphan.counter"
: >"$TMP/repeat-orphan.calls"
REPEAT_TEST_MODE=orphan_first \
REPEAT_TEST_COUNTER=$TMP/repeat-orphan.counter \
REPEAT_TEST_CALLS=$TMP/repeat-orphan.calls \
REPEAT_TEST_WAIT_PREFIX= \
REPEAT_TEST_WAIT_AT= \
REPEAT_TEST_DIFF_WAIT_PREFIX= \
REPEAT_TEST_REAL_DIFF="$(command -v diff)" \
REPEAT_TEST_ORPHAN_PID=$repeat_orphan_pid_file \
REPEAT_TEST_ORPHAN_REAPED=$repeat_orphan_marker \
OCANNL_TOOL_TEST_RUNS=$repeat_orphan_runs \
PATH=$repeat_bin:$PATH \
  "$repeat_root/tools/test-run.sh" repeat 2 build @cheap \
  >"$TMP/repeat-orphan.out" 2>"$TMP/repeat-orphan.err"
repeat_orphan_rc=$?
repeat_orphan_dir=$(OCANNL_TOOL_TEST_RUNS="$repeat_orphan_runs" "$repeat_root/tools/test-run.sh" paths run last 2>/dev/null)
orphan_pid=$(cat "$repeat_orphan_pid_file" 2>/dev/null)
if [ "$repeat_orphan_rc" = 137 ] \
   && [ -e "$repeat_orphan_marker" ] \
   && [ "$(cat "$repeat_orphan_dir/iteration-2/exit" 2>/dev/null)" = 0 ] \
   && grep -q '^repeat: iteration group .* survived its supervisor; reaping before reuse$' "$TMP/repeat-orphan.out"; then
  report 0 "repeat: surviving iteration group is reaped before reuse"
else
  report 1 "repeat: surviving iteration group is reaped before reuse" \
    "exit $repeat_orphan_rc; marker=$([ -e "$repeat_orphan_marker" ] && echo yes || echo no); output: $(cat "$TMP/repeat-orphan.out")"
fi
if [ -n "$orphan_pid" ] && kill -0 "$orphan_pid" 2>/dev/null; then
  kill -KILL -- "-$orphan_pid" 2>/dev/null
  kill -KILL "$orphan_pid" 2>/dev/null
fi

# A Dune action can leave its recorded process group with setsid. The FIFO
# writer inherited from the launch must still expose that survivor and refuse
# the shared build before iteration two, even though the group reap sees no
# reachable original group. Releasing the fixture afterwards also proves the
# refusal retained (rather than deleted) its build context while it was live.
repeat_escape_runs=$TMP/repeat-runs-session-escape
escape_release=$TMP/repeat-session-escape.release
escape_pid_file=$TMP/repeat-session-escape.pid
mkdir -p "$repeat_escape_runs"
: >"$TMP/repeat-session-escape.counter"
: >"$TMP/repeat-session-escape.calls"
if REPEAT_TEST_MODE=session_escape \
   REPEAT_TEST_COUNTER=$TMP/repeat-session-escape.counter \
   REPEAT_TEST_CALLS=$TMP/repeat-session-escape.calls \
   REPEAT_TEST_WAIT_PREFIX= \
   REPEAT_TEST_WAIT_AT= \
   REPEAT_TEST_DIFF_WAIT_PREFIX= \
   REPEAT_TEST_REAL_DIFF="$(command -v diff)" \
   REPEAT_TEST_ORPHAN_PID= \
   REPEAT_TEST_ORPHAN_REAPED= \
   REPEAT_TEST_ESCAPE_PID=$escape_pid_file \
   REPEAT_TEST_ESCAPE_RELEASE=$escape_release \
   OCANNL_TOOL_TEST_RUNS=$repeat_escape_runs \
   PATH=$repeat_bin:$PATH \
     "$repeat_root/tools/test-run.sh" repeat 2 build @cheap \
     >"$TMP/repeat-session-escape.out" 2>"$TMP/repeat-session-escape.err"; then
  repeat_escape_rc=0
else
  repeat_escape_rc=$?
fi
repeat_escape_dir=$(OCANNL_TOOL_TEST_RUNS="$repeat_escape_runs" "$repeat_root/tools/test-run.sh" paths run last 2>/dev/null)
escape_pid=$(cat "$escape_pid_file" 2>/dev/null)
escape_pgid=$(ppgid "$escape_pid")
recorded_pgid=$(cat "$repeat_escape_dir/iteration-1/pgid" 2>/dev/null)
if [ "$repeat_escape_rc" = 2 ] \
   && [ -n "$escape_pid" ] && kill -0 "$escape_pid" 2>/dev/null \
   && [ -n "$escape_pgid" ] && [ "$escape_pgid" != "$recorded_pgid" ] \
   && [ -d "$repeat_escape_dir/build" ] \
   && [ ! -d "$repeat_escape_dir/iteration-2" ] \
   && grep -q 'left a session-escaped descendant; refusing to reuse' \
        "$TMP/repeat-session-escape.err"; then
  report 0 "repeat: a session-escaped descendant blocks build reuse"
else
  report 1 "repeat: a session-escaped descendant blocks build reuse" \
    "exit $repeat_escape_rc; pid=${escape_pid:-missing}; escaped-pgid=${escape_pgid:-missing}; recorded-pgid=${recorded_pgid:-missing}; stderr=$(cat "$TMP/repeat-session-escape.err")"
fi
touch "$escape_release"
for _ in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
  kill -0 "$escape_pid" 2>/dev/null || break
  sleep 0.1
done
if kill -0 "$escape_pid" 2>/dev/null; then kill -KILL "$escape_pid" 2>/dev/null; fi
escape_pid=
escape_release=

pre_launch_line=$(grep -n '^      \[ -z "$repeat_cancelled" \] || break$' "$SRC" | head -1 | cut -d: -f1)
launch_line=$(grep -n '^        perl -e "$supervisor_perl" -- "$cap" /bin/bash -c \\' "$SRC" | head -1 | cut -d: -f1)
supervisor_line=$(grep -n '^      repeat_sup=\$!$' "$SRC" | cut -d: -f1)
post_launch_line=$(grep -n '^      \[ -z "$repeat_cancelled" \] || kill "-\$repeat_cancelled" "\$repeat_sup" 2>/dev/null$' "$SRC" | cut -d: -f1)
if [ -n "$pre_launch_line" ] && [ -n "$launch_line" ] \
   && [ -n "$supervisor_line" ] && [ -n "$post_launch_line" ] \
   && [ "$pre_launch_line" -lt "$launch_line" ] \
   && [ "$launch_line" -lt "$supervisor_line" ] \
   && [ "$supervisor_line" -lt "$post_launch_line" ]; then
  report 0 "repeat: cancellation brackets supervisor launch"
else
  report 1 "repeat: cancellation brackets supervisor launch" \
    "pre=${pre_launch_line:-missing} launch=${launch_line:-missing} supervisor=${supervisor_line:-missing} post=${post_launch_line:-missing}"
fi

clear_supervisor_line=$(grep -n '^      repeat_sup=$' "$SRC" | tail -1 | cut -d: -f1)
reap_group_line=$(grep -n '^      reap_repeat_group "$iter"$' "$SRC" | cut -d: -f1)
if [ -n "$clear_supervisor_line" ] && [ -n "$reap_group_line" ] \
   && [ "$clear_supervisor_line" -lt "$reap_group_line" ]; then
  report 0 "repeat: dead supervisor pid is cleared before group reap"
else
  report 1 "repeat: dead supervisor pid is cleared before group reap" \
    "clear=${clear_supervisor_line:-missing} reap=${reap_group_line:-missing}"
fi

sed -n '/^    reap_repeat_group() {/,/^    }/p' "$SRC" >"$TMP/reap-repeat-group.sh"
sed -n '/^      if ! group_identity_matches "$iter_dir"; then$/,/^      kill -KILL/p' \
  "$TMP/reap-repeat-group.sh" >"$TMP/reap-before-kill.sh"
if grep -q 'kill -0 -- "-\$pg"' "$TMP/reap-repeat-group.sh" \
   && grep -q 'group_alive "$pg" || return 0' "$TMP/reap-repeat-group.sh" \
   && ! grep -q 'group_alive' "$TMP/reap-before-kill.sh" \
   && grep -B6 'kill -KILL -- "-\$pg"' "$TMP/reap-repeat-group.sh" \
        | grep -q 'leaderless iteration group \$pg survived TERM' \
   && grep -q 'leaderless iteration group \$pg survived KILL' "$TMP/reap-repeat-group.sh"; then
  report 0 "repeat: orphan reap fails closed after TERM and accepts post-KILL zombies"
else
  report 1 "repeat: orphan reap fails closed after TERM and accepts post-KILL zombies" \
    "$(tr '\n' ';' <"$TMP/reap-repeat-group.sh")"
fi

repeat_probe repeat-identical stable 3 build @cheap
if [ "$repeat_rc" = 0 ] && grep -q '^repeat result: IDENTICAL -- ' <<<"$repeat_out" \
   && [ "$(cat "$TMP/repeat-identical.counter")" = 3 ] \
   && [ "$(grep -c '^clean ' "$TMP/repeat-identical.calls")" = 3 ] \
   && [ "$(grep -c -- '--force' "$TMP/repeat-identical.calls")" = 3 ] \
   && [ "$(grep -c -- '--cache=disabled' "$TMP/repeat-identical.calls")" = 3 ] \
   && [ "$(grep -c -- '--build-dir=' "$TMP/repeat-identical.calls")" = 6 ] \
   && [ -n "$repeat_dir" ] \
   && [ ! -e "$repeat_dir/build" ] \
   && [ "$(find "$repeat_dir" \( -name stdout -o -name stderr \) | wc -l | tr -d ' ')" = 6 ]; then
  report 0 "repeat: identical forced runs retain every stdout/stderr"
else
  report 1 "repeat: identical forced runs retain every stdout/stderr" \
    "exit $repeat_rc; output: ${repeat_out:-<nothing>}; stderr: $(cat "$TMP/repeat-identical.err")"
fi

repeat_probe repeat-stdout stdout 3 build @cheap
if [ "$repeat_rc" = 1 ] && grep -q '^repeat result: DIFFERING -- ' <<<"$repeat_out" \
   && [ -s "$repeat_dir/diffs/1-2.stdout" ]; then
  report 0 "repeat: stdout drift is red and pairwise-diffed"
else
  report 1 "repeat: stdout drift is red and pairwise-diffed" \
    "exit $repeat_rc; output: ${repeat_out:-<nothing>}"
fi

repeat_probe repeat-stderr stderr 3 build @cheap
if [ "$repeat_rc" = 0 ] && grep -q '^repeat result: STDERR-ONLY -- ' <<<"$repeat_out" \
   && [ -s "$repeat_dir/diffs/1-2.stderr" ]; then
  report 0 "repeat: stderr-only drift is distinct and diagnostic-green"
else
  report 1 "repeat: stderr-only drift is distinct and diagnostic-green" \
    "exit $repeat_rc; output: ${repeat_out:-<nothing>}"
fi

repeat_probe repeat-red fail 2 build @cheap
if [ "$repeat_rc" = 7 ] && grep -q '^repeat result: IDENTICAL -- ' <<<"$repeat_out"; then
  report 0 "repeat: a repeatable red dune leg keeps its exit code"
else
  report 1 "repeat: a repeatable red dune leg keeps its exit code" \
    "expected exit 7; got $repeat_rc; output: ${repeat_out:-<nothing>}"
fi

repeat_probe repeat-alone stable --alone 2 build @cheap
if [ "$repeat_rc" = 0 ] && [ "$(grep -c -- '-j 1' "$TMP/repeat-alone.calls")" = 2 ] \
   && grep -q 'iteration 1/2 -- dune (alone, -j 1)' <<<"$repeat_out"; then
  report 0 "repeat: --alone serializes every dune iteration"
else
  report 1 "repeat: --alone serializes every dune iteration" \
    "exit $repeat_rc; calls: $(tr '\n' ';' <"$TMP/repeat-alone.calls"); output: ${repeat_out:-<nothing>}"
fi

# The first `--` belongs to Dune, not to repeat. The trailing --force is a
# program argument on purpose: this fails if isolation flags are appended at
# the end or if the splice mistakes a later option-looking argument for Dune's.
repeat_probe repeat-separator stable 2 exec ./prog.exe -- alpha --force
if [ "$repeat_rc" = 0 ] \
   && [ "$(grep -c '^exec ./prog.exe --force --cache=disabled --build-dir=.* -- alpha --force$' \
          "$TMP/repeat-separator.calls")" = 2 ]; then
  report 0 "repeat: isolation flags precede Dune's argument separator"
else
  report 1 "repeat: isolation flags precede Dune's argument separator" \
    "exit $repeat_rc; calls: $(tr '\n' ';' <"$TMP/repeat-separator.calls")"
fi

# An active repeat must replace `last`, and stop must signal the OUTER
# coordinator so it records cancellation and refuses to start iteration two.
repeat_stop_runs=$TMP/repeat-runs-stop
repeat_stop_prefix=$TMP/repeat-stop
mkdir -p "$repeat_stop_runs"
: >"$TMP/repeat-stop.counter"
: >"$TMP/repeat-stop.calls"
REPEAT_TEST_MODE=stable \
REPEAT_TEST_COUNTER=$TMP/repeat-stop.counter \
REPEAT_TEST_CALLS=$TMP/repeat-stop.calls \
REPEAT_TEST_WAIT_PREFIX=$repeat_stop_prefix \
OCANNL_TOOL_TEST_RUNS=$repeat_stop_runs \
PATH=$repeat_bin:$PATH \
  "$repeat_root/tools/test-run.sh" repeat 3 build @cheap \
  >"$TMP/repeat-stop.out" 2>"$TMP/repeat-stop.err" &
repeat_pid=$!
if ! await_fixture_ready "$repeat_stop_prefix.ready"; then
  report 1 "repeat: last resolves active state and stop cancels the whole set" \
    "setup timeout waiting for iteration one's readiness marker"
  exit 1
fi
status_out=$(OCANNL_TOOL_TEST_RUNS=$repeat_stop_runs \
  "$repeat_root/tools/test-run.sh" status last 2>"$TMP/repeat-stop-status.err")
status_rc=$?
stop_out=$(OCANNL_TOOL_TEST_RUNS=$repeat_stop_runs \
  "$repeat_root/tools/test-run.sh" stop last 2>"$TMP/repeat-stop-stop.err")
stop_rc=$?
touch "$repeat_stop_prefix.release"
wait "$repeat_pid"
repeat_stop_rc=$?
repeat_pid=
if [ "$status_rc" = 3 ] && grep -q '^running: ' <<<"$status_out" \
   && [ "$stop_rc" = 0 ] && grep -q '^sent TERM to the repeat coordinator; ' <<<"$stop_out" \
   && [ "$repeat_stop_rc" = 143 ] \
   && [ "$(cat "$TMP/repeat-stop.counter")" = 1 ] \
   && grep -q '^repeat result: CANCELLED -- completed 1 of 3 iterations$' "$TMP/repeat-stop.out"; then
  report 0 "repeat: last resolves active state and stop cancels the whole set"
else
  report 1 "repeat: last resolves active state and stop cancels the whole set" \
    "status $status_rc: ${status_out:-<nothing>}; stop $stop_rc: ${stop_out:-<nothing>}; repeat $repeat_stop_rc: $(cat "$TMP/repeat-stop.out")"
fi

# Keep the signal traps armed after the iteration loop: this fixture gives the
# repeat its own process group, blocks inside the first pairwise diff, then
# sends TERM to the WHOLE group. The diff dies from that same signal; the exit
# finalizer must still turn the coordinator's trapped cancellation into an
# atomic verdict.
repeat_finalize_runs=$TMP/repeat-runs-finalize
repeat_finalize_prefix=$TMP/repeat-finalize
mkdir -p "$repeat_finalize_runs"
: >"$TMP/repeat-finalize.counter"
: >"$TMP/repeat-finalize.calls"
REPEAT_TEST_MODE=stdout \
REPEAT_TEST_COUNTER=$TMP/repeat-finalize.counter \
REPEAT_TEST_CALLS=$TMP/repeat-finalize.calls \
REPEAT_TEST_WAIT_PREFIX= \
REPEAT_TEST_DIFF_WAIT_PREFIX=$repeat_finalize_prefix \
REPEAT_TEST_REAL_DIFF="$(command -v diff)" \
OCANNL_TOOL_TEST_RUNS=$repeat_finalize_runs \
PATH=$repeat_bin:$PATH \
  perl -MPOSIX -e 'POSIX::setpgid(0, 0); exec @ARGV' \
  "$repeat_root/tools/test-run.sh" repeat 2 build @cheap \
  >"$TMP/repeat-finalize.out" 2>"$TMP/repeat-finalize.err" &
repeat_pid=$!
if ! await_fixture_ready "$repeat_finalize_prefix.ready"; then
  report 1 "repeat: cancellation during finalization still publishes a verdict" \
    "setup timeout waiting for the comparison readiness marker"
  exit 1
fi
kill -TERM -- "-$repeat_pid" 2>"$TMP/repeat-finalize-kill.err"
finalize_kill_rc=$?
wait "$repeat_pid"
repeat_finalize_rc=$?
repeat_pid=
repeat_finalize_dir=$(OCANNL_TOOL_TEST_RUNS="$repeat_finalize_runs" "$repeat_root/tools/test-run.sh" paths run last 2>/dev/null)
if [ "$finalize_kill_rc" = 0 ] \
   && [ "$repeat_finalize_rc" = 143 ] \
   && [ -n "$repeat_finalize_dir" ] \
   && [ "$(cat "$repeat_finalize_dir/exit" 2>/dev/null)" = 143 ] \
   && grep -q '^repeat result: CANCELLED -- completed 2 of 2 iterations$' "$TMP/repeat-finalize.out"; then
  report 0 "repeat: cancellation during finalization still publishes a verdict"
else
  report 1 "repeat: cancellation during finalization still publishes a verdict" \
    "group kill $finalize_kill_rc: $(cat "$TMP/repeat-finalize-kill.err"); repeat $repeat_finalize_rc: $(cat "$TMP/repeat-finalize.out")"
fi

# The stored cap belongs to ONE iteration. Drive `wait` against a fabricated
# live coordinator and a no-op sleep until its default deadline expires: cap 2
# across 3 iterations must report 126 seconds, not the old single-cap 122.
wait_runs=$TMP/repeat-runs-wait-budget
wait_dir=$wait_runs/20000101T000000Z-1
mkdir -p "$wait_dir" "$TMP/wait-bin"
wait_dir_real=$(cd "$wait_dir" && pwd -P)
printf 'build @cheap\n' >"$wait_dir/cmd"
printf '2\n' >"$wait_dir/cap"
printf 'repeat\n' >"$wait_dir/mode"
printf '3\n' >"$wait_dir/repeats"
printf '%s\n' "$$" >"$wait_dir/pid"
ps_token "$$" >"$wait_dir/ptoken"
: >"$wait_dir/log"
cat >"$TMP/wait-bin/sleep" <<'EOF'
#!/bin/sh
printf '%s\n' "$1" >>"$REPEAT_TEST_SLEEP_CALLS"
EOF
chmod +x "$TMP/wait-bin/sleep"
: >"$TMP/repeat-wait-sleeps"
wait_out=$(REPEAT_TEST_SLEEP_CALLS=$TMP/repeat-wait-sleeps \
  OCANNL_TOOL_TEST_RUNS=$wait_runs PATH=$TMP/wait-bin:$PATH \
  "$repeat_root/tools/test-run.sh" wait "$wait_dir" 2>"$TMP/repeat-wait.err")
wait_rc=$?
if [ "$wait_rc" = 124 ] \
   && grep -q "^wait timed out after 126s: $wait_dir_real$" <<<"$wait_out" \
   && [ "$(wc -l <"$TMP/repeat-wait-sleeps" | tr -d ' ')" = 26 ]; then
  report 0 "repeat: wait default covers every per-iteration cap"
else
  report 1 "repeat: wait default covers every per-iteration cap" \
    "exit $wait_rc; sleeps $(tr '\n' ',' <"$TMP/repeat-wait-sleeps"); output: ${wait_out:-<nothing>}"
fi

# ---------------------------------------------------------------------------
# Legs 26-28: this script's own options are refused on the wrong side of dune
# ---------------------------------------------------------------------------
# `run build @alias --cap 900` used to forward `--cap 900` to dune, which exits
# 1 on the unknown option having built nothing; the digest then reads `FAIL
# (exit 1)` with no matching error lines -- a verdict indistinguishable from a
# failing test, and a session acting on it debugs code that never ran. So the
# refusal is asserted together with the CALLS file being empty: "runs nothing"
# is the half that makes the exit code trustworthy.
argv_probe() { # tag subcommand [argv...] -- drives the tool against the fixture dune
  # `argv_mode` selects the fixture dune's behaviour (default: a green run);
  # `argv_runs` reuses an earlier probe's run store, so `status`/`wait last`
  # can read back the run an earlier `run` recorded there.
  local tag=$1 runs=${argv_runs:-$TMP/argv-runs-$1}
  shift
  mkdir -p "$runs"
  : >"$TMP/$tag.counter"
  : >"$TMP/$tag.calls"
  REPEAT_TEST_MODE=${argv_mode:-stable} \
  REPEAT_TEST_COUNTER=$TMP/$tag.counter \
  REPEAT_TEST_CALLS=$TMP/$tag.calls \
  REPEAT_TEST_WAIT_PREFIX= \
  REPEAT_TEST_WAIT_AT= \
  REPEAT_TEST_ORPHAN_PID= \
  REPEAT_TEST_ORPHAN_REAPED= \
  REPEAT_TEST_DIFF_WAIT_PREFIX= \
  REPEAT_TEST_REAL_DIFF="$(command -v diff)" \
  OCANNL_TOOL_TEST_RUNS=$runs \
  PATH=$repeat_bin:$PATH \
    "$repeat_root/tools/test-run.sh" "$@" >"$TMP/$tag.out" 2>"$TMP/$tag.err"
  argv_rc=$?
  argv_out=$(cat "$TMP/$tag.out")
  argv_err=$(cat "$TMP/$tag.err")
  argv_calls=$(cat "$TMP/$tag.calls")
  argv_dir=$(OCANNL_TOOL_TEST_RUNS="$runs" "$repeat_root/tools/test-run.sh" paths run last 2>/dev/null)
}
argv_rc= argv_out= argv_err= argv_calls= argv_dir= argv_mode= argv_runs=

# Every subcommand that takes options, and both spellings of a value option:
# the guard sits on one code path, but a future refactor that splits it must
# not be able to leave one entry point forwarding.
misplaced_label="an option after the dune arguments is refused before dune runs"
misplaced_detail=
for probe in "run:run build @cheap --cap 900" \
             "start:start build @cheap --cap=900" \
             "repeat:repeat 2 build @cheap --alone"; do
  argv_probe "misplaced-${probe%%:*}" ${probe#*:}
  # Exit 2 is the usage code; the message must name the option AND the order,
  # or the reader is left where the unknown-option error left them.
  case $argv_rc in
    2) ;;
    *) misplaced_detail="${probe#*:}: exit $argv_rc (want 2): $argv_err" ; break ;;
  esac
  case $argv_err in
    *"belongs before the dune arguments"*"tools/test-run.sh"*) ;;
    *) misplaced_detail="${probe#*:}: unhelpful refusal: $argv_err" ; break ;;
  esac
  if [ -n "$argv_calls" ]; then
    misplaced_detail="${probe#*:}: dune was invoked anyway: $argv_calls"
    break
  fi
done
if [ -z "$misplaced_detail" ]; then
  report 0 "$misplaced_label"
else
  report 1 "$misplaced_label" "$misplaced_detail"
fi

# The control for the leg above: the refusal must be about PLACEMENT, not about
# the word. A guard that also rejected the documented form would pass leg 26.
argv_probe cap-placed run --cap 900 build @cheap
if [ "$argv_rc" = 0 ] && [ "$argv_calls" = "build @cheap" ]; then
  report 0 "a correctly placed --cap is consumed and never reaches dune"
else
  report 1 "a correctly placed --cap is consumed and never reaches dune" \
    "exit $argv_rc; calls: ${argv_calls:-<none>}; stderr: $argv_err"
fi

# Past dune's own separator the word is an executable's argument -- the caller's
# business, and the one place `--cap` after the target can be meant.
argv_probe cap-after-sep run exec ./prog.exe -- --cap 900
if [ "$argv_rc" = 0 ] && [ "$argv_calls" = "exec ./prog.exe -- --cap 900" ]; then
  report 0 "an option past dune's -- reaches the program untouched"
else
  report 1 "an option past dune's -- reaches the program untouched" \
    "exit $argv_rc; calls: ${argv_calls:-<none>}; stderr: $argv_err"
fi

# ---------------------------------------------------------------------------
# Legs 29-32: an invocation dune's own parser refused is a verdict of its own
# ---------------------------------------------------------------------------
# The guard above knows two words. Everything else dune's CLI refuses -- an
# unknown option, an unknown subcommand, a malformed operand -- prints
# `dune: <complaint>` then `Usage: dune ...`, exits 1 and runs nothing, and the
# digest used to read that 1 as `FAIL (exit 1)` with no error lines: the verdict
# of a red suite, at the moment the reader decides between reading failures and
# fixing the command line (gh-ocannl-944). Each shape is driven through the
# real `run` path against the fixture dune emitting dune's exact stderr, and
# the assertions are the halves of the contract: the distinct verdict quoting
# dune's own complaint (the wrapped operand complaint WHOLE, continuation line
# included), `run`'s status 2 -- the usage code, so a caller branches without
# reading the log -- over a RECORDED status that stays dune's 1, the same two
# readings from `wait` (2) and `status` (0: it reports publication), and dune
# invoked exactly once -- what tells this refusal from the guard's, which
# shares its exit code.
refused_verdict="verdict: INVOCATION REFUSED (dune rejected the arguments; nothing ran) (exit 2)"
refused_label="a dune-refused invocation is INVOCATION REFUSED: run/wait exit 2, recorded exit 1"
refused_detail=
for probe in "usage_option|build @cheap --frobnicate" \
             "usage_command|frobnicate" \
             "usage_operand|build -j x"; do
  mode=${probe%%|*}
  case $mode in
    usage_option) want1="dune: unknown option '--frobnicate'." want2= ;;
    usage_command) want1="dune: unknown command 'frobnicate', must be one of" want2= ;;
    usage_operand) want1="dune: option '-j': invalid concurrency value" want2="        number" ;;
  esac
  argv_mode=$mode argv_probe "refused-$mode" run ${probe#*|}
  refused_runs=$TMP/argv-runs-refused-$mode
  case $argv_rc in
    2) ;;
    *) refused_detail="$mode: run exited $argv_rc (want 2); stdout: $argv_out"; break ;;
  esac
  case $argv_out in
    *"$refused_verdict"*"dune said:"*"  $want1"*"$want2"*) ;;
    *) refused_detail="$mode: verdict or quoted complaint missing: $argv_out"; break ;;
  esac
  case $argv_out in
    *"no Error/File lines matched"* | *"fingerprint:"*)
      refused_detail="$mode: the red-run digest still printed: $argv_out"; break ;;
  esac
  if [ "$argv_calls" != "${probe#*|}" ]; then
    refused_detail="$mode: dune calls (want exactly the argv): ${argv_calls:-<none>}"; break
  fi
  if [ -z "$argv_dir" ] || [ "$(cat "$argv_dir/exit" 2>/dev/null)" != 1 ] \
     || [ "$(tail -n 1 "$argv_dir/log" 2>/dev/null)" != "exit: 1" ]; then
    refused_detail="$mode: recorded status is not dune's 1: $(cat "$argv_dir/exit" 2>/dev/null; tail -n 1 "$argv_dir/log" 2>/dev/null)"
    break
  fi
  argv_runs=$refused_runs argv_probe "refused-$mode-status" status last
  if [ "$argv_rc" != 0 ] || [ "$argv_calls" != "" ]; then
    refused_detail="$mode: status last exited $argv_rc (want 0: publication, not the verdict); calls: $argv_calls"; break
  fi
  case $argv_out in
    *"$refused_verdict"*) ;;
    *) refused_detail="$mode: status last lost the verdict: $argv_out"; break ;;
  esac
  argv_runs=$refused_runs argv_probe "refused-$mode-wait" wait last
  if [ "$argv_rc" != 2 ] || [ "$argv_calls" != "" ]; then
    refused_detail="$mode: wait last exited $argv_rc (want 2); calls: $argv_calls"; break
  fi
  case $argv_out in
    *"$refused_verdict"*"  $want1"*) ;;
    *) refused_detail="$mode: wait last lost the verdict: $argv_out"; break ;;
  esac
done
argv_mode= argv_runs=
if [ -z "$refused_detail" ]; then
  report 0 "$refused_label"
else
  report 1 "$refused_label" "$refused_detail"
fi

# The controls: a run that FAILED still says so, with dune's status and the
# fingerprint -- a recogniser that fired on `dune:` alone would turn a test
# whose output opens on that word into "nothing ran", and one that stopped
# reading at `Usage:` would do the same to a `dune exec` program that printed
# a nested dune refusal and then failed (Codex review round 1, P2): both are
# the inverse misreading, so the shape must also be ALONE in the log.
red_label="a red run still digests as FAIL (exit 1) with its fingerprint"
red_detail=
for mode in red dune_prefixed_red nested_usage_red nested_usage_output; do
  argv_mode=$mode argv_probe "red-$mode" run build @cheap
  case $argv_rc in
    1) ;;
    *) red_detail="$mode: run exited $argv_rc (want 1); stdout: $argv_out"; break ;;
  esac
  case $argv_out in
    *"verdict: FAIL (exit 1)"*) ;;
    *) red_detail="$mode: FAIL verdict missing: $argv_out"; break ;;
  esac
  case $argv_out in
    *"INVOCATION REFUSED"*) red_detail="$mode: a red run read as refused: $argv_out"; break ;;
  esac
  if [ "$mode" = nested_usage_output ]; then
    # No located error to fingerprint: the digest falls through to the log
    # tail, which is the trailing evidence itself.
    case $argv_out in
      *"no Error/File lines matched"*"the program went on to print this"*) ;;
      *) red_detail="$mode: log tail missing: $argv_out"; break ;;
    esac
    argv_runs=$TMP/argv-runs-red-$mode argv_probe "red-$mode-wait" wait last
    if [ "$argv_rc" != 1 ]; then
      red_detail="$mode: wait last exited $argv_rc (want 1)"; break
    fi
    continue
  fi
  # The fingerprint is sorted (`sort -u`), so its two lines are pinned
  # separately rather than in an order the digest never promised.
  case $argv_out in
    *"fingerprint:"*'File "test/operations/fixture.ml", line 3'*) ;;
    *) red_detail="$mode: File fingerprint missing: $argv_out"; break ;;
  esac
  case $argv_out in
    *"fingerprint:"*"Error: This expression has type int"*) ;;
    *) red_detail="$mode: Error fingerprint missing: $argv_out"; break ;;
  esac
  argv_runs=$TMP/argv-runs-red-$mode argv_probe "red-$mode-wait" wait last
  if [ "$argv_rc" != 1 ]; then
    red_detail="$mode: wait last exited $argv_rc (want 1)"; break
  fi
done
argv_mode= argv_runs=
if [ -z "$red_detail" ]; then
  report 0 "$red_label"
else
  report 1 "$red_label" "$red_detail"
fi

# The guard from leg 26 is not made redundant by the digest: it knows the
# correct order and refuses without spawning dune. With a fixture that WOULD
# refuse, the guard's refusal is the one that arrives, and the calls file --
# empty here, one line for the digest's refusal -- is what separates the two
# exit-2 outcomes.
argv_mode=usage_option argv_probe guard-before-dune run build @cheap --cap 900
argv_mode=
if [ "$argv_rc" = 2 ] && [ -z "$argv_calls" ] \
   && case $argv_err in *"belongs before the dune arguments"*) true ;; *) false ;; esac \
   && case $argv_out in *"INVOCATION REFUSED"*) false ;; *) true ;; esac; then
  report 0 "the --cap guard still refuses before dune is spawned"
else
  report 1 "the --cap guard still refuses before dune is spawned" \
    "exit $argv_rc; calls: ${argv_calls:-<none>}; stderr: $argv_err; stdout: $argv_out"
fi

# `repeat` has its own vocabulary and the same trap: N refused iterations are
# byte-identical, so an unguarded set would report IDENTICAL over nothing.
# One iteration decides it (every iteration runs the same argv), the set stops
# there, the coordinator exits 2 over a recorded 1, and `wait` on the set
# digests it under the same verdict. The control: a merely red iteration is
# still repeated in full, with dune's status.
repeat_probe repeat-refused usage_option 3 build @cheap
repeat_refused_label="repeat stops after a refused first iteration and exits 2 over a recorded 1"
if [ "$repeat_rc" = 2 ] && [ "$(cat "$TMP/repeat-refused.counter")" = 1 ] \
   && [ "$(cat "$repeat_dir/exit" 2>/dev/null)" = 1 ] \
   && case $repeat_out in
        *"repeat result: INVOCATION REFUSED"*"dune said:"*"  dune: unknown option '--frobnicate'."*) true ;;
        *) false ;;
      esac; then
  argv_runs=$TMP/repeat-runs-repeat-refused argv_probe repeat-refused-wait wait last
  argv_runs=
  if [ "$argv_rc" = 2 ] && case $argv_out in *"$refused_verdict"*) true ;; *) false ;; esac; then
    report 0 "$repeat_refused_label"
  else
    report 1 "$repeat_refused_label" "wait last on the set: exit $argv_rc; stdout: $argv_out"
  fi
else
  report 1 "$repeat_refused_label" \
    "exit $repeat_rc; iterations $(cat "$TMP/repeat-refused.counter"); recorded $(cat "$repeat_dir/exit" 2>/dev/null); stdout: $repeat_out"
fi
repeat_probe repeat-red-error red 2 build @cheap
if [ "$repeat_rc" = 1 ] && [ "$(cat "$TMP/repeat-red-error.counter")" = 2 ] \
   && case $repeat_out in
        *"INVOCATION REFUSED"*) false ;;
        *"repeat result: IDENTICAL"*) true ;;
        *) false ;;
      esac; then
  report 0 "repeat: a red iteration is repeated in full and keeps dune's status"
else
  report 1 "repeat: a red iteration is repeated in full and keeps dune's status" \
    "exit $repeat_rc; iterations $(cat "$TMP/repeat-red-error.counter"); stdout: $repeat_out"
fi

echo
# The skip count is printed on every run, not only when it is nonzero: "all legs
# passed" over a run that decided three of them is the reading to prevent.
if [ "$failures" -eq 0 ]; then
  echo "all legs passed ($skipped skipped)"
else
  echo "$failures leg(s) failed ($skipped skipped)"
fi
exit $(( failures > 0 ? 1 : 0 ))
