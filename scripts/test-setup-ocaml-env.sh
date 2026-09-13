#!/usr/bin/env bash
# Tests for the "staleness against origin/master" section of
# scripts/setup-ocaml-env.sh (the SessionStart hook).
#
#   scripts/test-setup-ocaml-env.sh          # run every leg
#   scripts/test-setup-ocaml-env.sh --keep   # keep the temp dir for inspection
#
# Run it after editing that section. It takes about 90 seconds, nearly all of it
# spent sitting out watchdog timeouts. It is deliberately not a dune test
# because it spawns and kills process groups; the Ubuntu CI leg runs it directly
# so its Linux-specific zombie-group facts are exercised (gh-ocannl-795).
#
# It tests the WORKING-TREE copy of the hook, not the committed one: the script
# next to this file is copied into every throwaway clone. During the six review
# rounds of PR #430 an ad-hoc harness silently tested the committed script after
# cloning the repo, and a `run` helper executed its own label as a command — so
# this harness prints the source path and digest it is testing, and every hook
# run asserts it actually reached the section under test.
#
# Nothing here touches the real repository: the clones live under a `mktemp -d`
# directory, run with `env -i` and a sanitised PATH (no opam, so the hook stops
# right after the section under test), and see neither the user's global nor the
# system gitconfig.
#
# Legs:
#   1. `bounded` — the watchdog: TERM at the bound, KILL 5s later, process-group
#      kill, rc preservation, no orphans, and a group holding only a zombie
#      read as empty rather than as work.
#   2. counting — behind/ahead wording and recovery command, offline fallback,
#      ref-ambiguity, FETCH_HEAD untouched, no-origin silence.
#   3. SSH launcher gating — which program git ends up invoking and whether the
#      OpenSSH options were appended to it.

set -u

. "$(cd "$(dirname "$0")/../scripts" && pwd)/harness-support.sh"
harness_args "$@"

HERE="$(cd "$(dirname "$0")" && pwd)"
HOOK_SRC="$HERE/setup-ocaml-env.sh"
GROUP_SRC="$HERE/process-group.sh"
[ -f "$HOOK_SRC" ] || { echo "no $HOOK_SRC" >&2; exit 2; }
[ -f "$GROUP_SRC" ] || { echo "no $GROUP_SRC" >&2; exit 2; }
BASH_BIN="$(command -v bash)"


echo "testing $HOOK_SRC and $GROUP_SRC"
printf '  digest %s\n' "$( (cksum <"$HOOK_SRC") 2>/dev/null || echo '?')"
printf '  group digest %s\n' "$( (cksum <"$GROUP_SRC") 2>/dev/null || echo '?')"

# The header and the agent-note both tell people to run this as
# `scripts/test-setup-ocaml-env.sh`, which needs the bit to survive in git and
# through a fresh clone; it has been lost once already.
if [ -x "$0" ]; then
  report 0 "harness: this script is executable, as its documented invocation needs"
else
  report 1 "harness: this script is executable, as its documented invocation needs" \
    "$0 is not executable — 'scripts/test-setup-ocaml-env.sh' would fail with Permission denied"
fi

# Checked, not assumed: nothing here uses `set -e`, so a `mktemp` that fails —
# TMPDIR missing, unwritable, full — would leave TMP empty and every path below
# would resolve against the ROOT: `$TMP/bin` becomes /bin, and the symlink
# farm would be installed there. Refuse rather than continue.
zparent=""   # leg (f)'s self-stopping zombie maker; cleanup must resume it
harness_scratch "test-setup-ocaml-env"
cleanup_fixture() {
  # Leg (f)'s zombie maker STOPS ITSELF and is resumed at the end of the leg.
  # Interrupted in between, nothing else would ever resume it: it would be
  # reparented to PID 1 still stopped, still holding its zombie child. Killing
  # it is not the answer either — that orphans the zombie. Resume it so it reaps
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
  # A hook broken in the way leg 1 probes for can leave orphans behind. They
  # carry this run's pid in their duration (see D_* below), so they are
  # unambiguously ours to reap and no concurrent run is disturbed.
  # Everything below is written to survive running BEFORE leg 1 defined any of
  # it: an EXIT trap that itself fails under `set -u` is the worst place to
  # learn about ordering.
  local orphans d
  orphans=""
  if command -v sleep_pids >/dev/null 2>&1; then
    for d in "${D_TERM:-}" "${D_IGN_CHILD:-}" "${D_IGN_PARENT:-}" "${D_IGN_ALL:-}" \
             "${D_DAEMON:-}" "${D_WATCHDOG:-}" "${D_SELFTEST:-}" "${D_PROBE:-}"; do
      [ -n "$d" ] || continue
      orphans="$orphans $(sleep_pids "$d")"
    done
  fi
  # shellcheck disable=SC2086
  [ -n "${orphans// /}" ] && kill -KILL $orphans 2>/dev/null
  # Belt and braces on the same hazard: never hand `rm -rf` anything but the
  # directory this run actually made.
  if [ -n "${LAUNCH_ROOT:-}" ] && [ "${LAUNCH_ROOT#"${TMP:-/nonexistent}"}" = "$LAUNCH_ROOT" ] \
     && [ -d "$LAUNCH_ROOT" ] && [ "$KEEP" != 1 ]; then
    rm -rf "$LAUNCH_ROOT"     # only when it was made outside TMP
  fi
  return 0
}
# Without these, a TERM or a Ctrl-C kills the shell outright and the EXIT trap
# never runs — which is how an interrupted run left a stopped zombie maker
# behind. Exiting from the handler is what gets EXIT to fire.

# ---------------------------------------------------------------------------
# Leg 1: bounded
# ---------------------------------------------------------------------------
# Extract the predicate from the working-tree shared helper and the bound from
# the working-tree hook, then source that exact pair.
sed -n '/^group_alive() {/,/^}/p' "$GROUP_SRC" >"$TMP/bounded.sh"
sed -n '/^bounded() {/,/^}/p' "$HOOK_SRC" >>"$TMP/bounded.sh"
# Checked structurally — opens with the header, closes with the brace, has a
# body — rather than by grepping for one line of it: this guard exists to catch
# a sed that matched nothing, and must still hold while a leg under test is
# being mutated to see the assertion fail.
b_lines="$(wc -l <"$TMP/bounded.sh" | tr -d ' ')"
case "$(head -n1 "$TMP/bounded.sh")" in
  "group_alive() {"*) group_head_ok=1 ;;
  *) group_head_ok=0 ;;
esac
if [ "$group_head_ok" = 0 ] || ! grep -qx 'bounded() {' "$TMP/bounded.sh" \
   || [ "$(tail -n1 "$TMP/bounded.sh")" != "}" ] || [ "$b_lines" -lt 10 ]; then
  report 1 "bounded: extracted" \
    "sed did not capture group_alive from $GROUP_SRC and bounded from $HOOK_SRC"
else
  report 0 "bounded: extracted ($b_lines lines)"
fi
# shellcheck disable=SC1090
. "$TMP/bounded.sh"

# Survivors are counted by EXACT duration: matching `sleep` alone would also
# catch the watchdog's own `sleep 3` / `sleep 1`, and a substring match would
# additionally catch this harness's command line. The durations also carry this
# run's pid, so a second copy of this script (or a leftover from a crashed one)
# cannot be miscounted as this run's orphan.
#
# Not listed with `pgrep -a`: `-a` is not the same option on both platforms —
# procps prints the command line, while on macOS it means "include process
# ancestors in the match list" and prints no arguments at all. A counter that
# finds no duration to match reports zero forever, and every no-orphan assertion
# below then passes without testing anything, which is why the prerequisite
# check makes the counter count something known before use.
#
# /proc is tried first and `ps -A -o pid=,args=` second, in that order because
# Cygwin has the former but a `ps` without `-o`: a ps-only lister leaves the
# harness skipping every bounded leg on a shell the hook explicitly supports.
# The /proc read is also fork-free — `read -d ""` per NUL-separated argument
# rather than a `tr` per process.
D_TERM="91.$$"      # (a) honours TERM
D_IGN_CHILD="92.$$" # (b) the child that ignores TERM
D_IGN_PARENT="93.$$" # (b) the parent that does not
D_IGN_ALL="94.$$"   # (c) everything ignores TERM
D_DAEMON="95.$$"    # (d) the daemon left behind by an exit-0 command
D_WATCHDOG="97.$$"  # (e) the bound, i.e. the watchdog's own sleep
D_SELFTEST="96.$$"  # the prerequisite check's own known-live sleep
D_PROBE="98.$$"     # the job-control probe's sleep
sleep_pids() { # sleep_pids DURATION -> pids of live `sleep DURATION`
  local d="$1" f pid a0 a1 a2
  if [ -r /proc/self/cmdline ]; then
    for f in /proc/[0-9]*/cmdline; do
      a0=""; a1=""; a2=""
      { IFS= read -r -d '' a0; IFS= read -r -d '' a1; IFS= read -r -d '' a2; } \
        <"$f" 2>/dev/null
      [ "${a0##*/}" = sleep ] || continue   # argv[0] may carry a directory
      [ "$a1" = "$d" ] || continue
      [ -z "$a2" ] || continue              # exactly one argument
      pid="${f#/proc/}"
      printf '%s\n' "${pid%/cmdline}"
    done
    return 0
  fi
  ps -A -o pid=,args= 2>/dev/null | awk -v d="$d" '
    NF == 3 {
      cmd = $2; sub(/^.*\//, "", cmd)
      if (cmd == "sleep" && $3 == d) print $1
    }'
}
pgid_of() { # pgid_of PID -> its process group id, or nothing if unreadable
  # Same /proc-then-ps ladder as the survivor lister, for the same reason: a
  # `ps` without `-o` (Cygwin) answers nothing, and what this value guards is
  # too important to guess at.
  local pid="$1" line
  if [ -r "/proc/$pid/stat" ]; then
    # Grouped: a failed redirection is reported by the shell before the
    # command's own 2>/dev/null applies, and this is asked about pids that are
    # expected to be gone.
    { read -r line <"/proc/$pid/stat"; } 2>/dev/null || return 0
    line="${line##*) }"                 # comm may itself hold ") "
    # shellcheck disable=SC2086
    set -- $line                        # $1 state, $2 ppid, $3 pgrp
    printf '%s\n' "${3:-}"
    return 0
  fi
  ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' '
}

survivors() { # survivors DURATION -> count of live `sleep DURATION`
  sleep_pids "$1" | awk 'END { print NR + 0 }'
}
settled_survivors() { # settled_survivors DURATION -- allow for asynchronous reaping
  local i n
  n=0
  for i in 1 2 3 4 5; do
    n="$(survivors "$1")"
    [ "$n" = 0 ] && break
    sleep 0.2
  done
  printf '%s\n' "$n"
}

BOUND=3

# Not "is there a `ps`" but "does this `ps`, parsed this way, actually find a
# process we know is running" — the macOS `pgrep -a` trap was invisible exactly
# because the tool was present and the parse silently matched nothing.
sleep "$D_SELFTEST" >/dev/null 2>&1 </dev/null &
selftest_pid=$!
selftest_found="$(survivors "$D_SELFTEST")"
kill -KILL "$selftest_pid" 2>/dev/null
wait "$selftest_pid" 2>/dev/null
if [ "$selftest_found" != 1 ]; then
  report 1 "bounded: prerequisites — the survivor counter can count" \
    "counted $selftest_found live 'sleep $D_SELFTEST', expected 1; every no-orphan leg below would pass vacuously"
else
  report 0 "bounded: prerequisites — the survivor counter can count"
  # Guard: `bounded` signals the process GROUP. If `set -m` did not give the
  # child a group of its own, that group is OURS and the first leg would kill
  # this harness. Check before running any of them.
  self_pgid="$(pgid_of $$)"
  set -m
  sleep "$D_PROBE" >/dev/null 2>&1 </dev/null & probe=$!
  set +m
  probe_pgid="$(pgid_of "$probe")"
  # Fails CLOSED. This guard is the only thing standing between a broken
  # `set -m` and a `kill -- -PGID` aimed at this shell's own group, so "could
  # not tell" must not take the same branch as "told, and they differ" — which
  # is what an `[ -n "$self_pgid" ] &&` test did when BOTH were empty.
  if [ -z "$self_pgid" ] || [ -z "$probe_pgid" ]; then
    kill -KILL "$probe" 2>/dev/null
    wait "$probe" 2>/dev/null
    report 1 "bounded: job control gives the child its own process group" \
      "process group ids are unreadable here (self='$self_pgid' probe='$probe_pgid'); refusing to run legs that signal process groups"
  elif [ "$self_pgid" = "$probe_pgid" ]; then
    kill "$probe" 2>/dev/null
    report 1 "bounded: job control gives the child its own process group" \
      "pgid $probe_pgid == harness pgid; skipping the bounded legs rather than killing this shell"
  else
    kill -KILL -- -"$probe" 2>/dev/null
    wait "$probe" 2>/dev/null
    report 0 "bounded: job control gives the child its own process group"

    # (a) The command honours TERM: return at the bound, rc=143, nothing left.
    t0=$SECONDS
    bounded "$BOUND" sleep "$D_TERM" >/dev/null 2>&1; rc=$?
    el=$((SECONDS - t0)); left="$(settled_survivors "$D_TERM")"
    if [ "$rc" = 143 ] && [ "$el" -ge "$BOUND" ] && [ "$el" -le $((BOUND + 2)) ] && [ "$left" = 0 ]; then
      report 0 "bounded (a): TERM-honouring command returns at the bound, rc=143, no orphans"
    else
      report 1 "bounded (a): TERM-honouring command returns at the bound, rc=143, no orphans" \
        "rc=$rc elapsed=${el}s (want ${BOUND}s) survivors=$left"
    fi

    # (b) Parent dies on TERM, a child ignores it: the group is not empty, so
    #     `bounded` must stay until the KILL escalation rather than returning
    #     with an orphan still running.
    t0=$SECONDS
    bounded "$BOUND" bash -c '(trap "" TERM; exec sleep "$1") & sleep "$2"' _ \
      "$D_IGN_CHILD" "$D_IGN_PARENT" >/dev/null 2>&1; rc=$?
    el=$((SECONDS - t0)); left="$(settled_survivors "$D_IGN_CHILD")"; left2="$(settled_survivors "$D_IGN_PARENT")"
    if [ "$el" -ge $((BOUND + 4)) ] && [ "$el" -le $((BOUND + 7)) ] && [ "$left" = 0 ] && [ "$left2" = 0 ]; then
      report 0 "bounded (b): TERM-ignoring child holds the return until KILL, no orphans"
    else
      report 1 "bounded (b): TERM-ignoring child holds the return until KILL, no orphans" \
        "rc=$rc elapsed=${el}s (want $((BOUND + 5))s) survivors=$left/$left2"
    fi

    # (c) Everything ignores TERM: killed at bound+5, rc=137.
    t0=$SECONDS
    bounded "$BOUND" bash -c 'trap "" TERM; exec sleep "$1"' _ "$D_IGN_ALL" >/dev/null 2>&1; rc=$?
    el=$((SECONDS - t0)); left="$(settled_survivors "$D_IGN_ALL")"
    if [ "$rc" = 137 ] && [ "$el" -ge $((BOUND + 4)) ] && [ "$el" -le $((BOUND + 7)) ] && [ "$left" = 0 ]; then
      report 0 "bounded (c): TERM-ignoring command is KILLed at bound+5s, rc=137"
    else
      report 1 "bounded (c): TERM-ignoring command is KILLed at bound+5s, rc=137" \
        "rc=$rc elapsed=${el}s (want $((BOUND + 5))s) survivors=$left"
    fi

    # (d) The command exits 0 early but leaves a daemon in the group: the
    #     contract is that nothing of the group survives the return.
    t0=$SECONDS
    bounded "$BOUND" bash -c 'sleep "$1" >/dev/null 2>&1 </dev/null & exit 0' _ \
      "$D_DAEMON" >/dev/null 2>&1; rc=$?
    el=$((SECONDS - t0)); left="$(settled_survivors "$D_DAEMON")"
    if [ "$rc" = 0 ] && [ "$el" -ge "$BOUND" ] && [ "$el" -le $((BOUND + 2)) ] && [ "$left" = 0 ]; then
      report 0 "bounded (d): daemon left by an exit-0 command is cleaned up at the bound"
    else
      report 1 "bounded (d): daemon left by an exit-0 command is cleaned up at the bound" \
        "rc=$rc elapsed=${el}s (want ${BOUND}s) survivors=$left"
    fi

    # (e) A fast exit preserves the command's status and cancels the watchdog
    #     (no `sleep 97` left behind).
    t0=$SECONDS
    bounded "$D_WATCHDOG" bash -c 'exit 42' >/dev/null 2>&1; rc=$?
    el=$((SECONDS - t0)); left="$(settled_survivors "$D_WATCHDOG")"
    if [ "$rc" = 42 ] && [ "$el" -le 3 ] && [ "$left" = 0 ]; then
      report 0 "bounded (e): fast exit preserves rc and reaps the watchdog"
    else
      report 1 "bounded (e): fast exit preserves rc and reaps the watchdog" \
        "rc=$rc elapsed=${el}s (want ~0s) watchdog-sleeps=$left"
    fi

    # (f) A group holding nothing but a zombie must read as EMPTY. This is the
    #     misreading that cost the hook 30s on every failed ssh fetch: `kill -0`
    #     answers yes for a zombie exactly as for a live process. Under a PID 1
    #     that reaps, such a zombie is transient and the old check merely lost a
    #     race with it; under one that does not — the common container case — it
    #     is PERMANENT, so no amount of waiting would have cleared it. Leg (d)
    #     is the other side of the question — a group that really is occupied
    #     still waits — and ssh (15) is the end-to-end symptom.
    #
    #     Making a zombie that is reliably still a zombie when looked at, and
    #     that leaves nothing behind afterwards, takes some care. The parent
    #     puts the child in a process group of ITS OWN (so the group holds the
    #     zombie and nothing else), then STOPs itself before the child exits: a
    #     stopped shell runs no SIGCHLD handler, so it cannot reap, and the
    #     child stays a zombie for as long as we need. Killing that parent would
    #     orphan the zombie onto a PID 1 that — in the very environment this leg
    #     is for — never reaps it, leaking a process-table entry per run. So it
    #     is CONTinued instead, and reaps its own child on the way out.
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
    # Wait for real zombiehood rather than guessing at it, and read the state
    # with `ps -p`, independently of the `group_alive` under test.
    zstate=""
    if [ -n "$zpid" ]; then
      for _ in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30; do
        zstate="$(ps -o stat= -p "$zpid" 2>/dev/null | tr -d ' ')"
        case "$zstate" in Z*) break ;; esac
        sleep 0.1
      done
    fi
    if [ -z "$zpid" ]; then
      report 1 "bounded (f): a group holding nothing but a zombie reads as empty" \
        "could not start the zombie maker"
    else
      case "$zstate" in
        Z*)
          if ! kill -0 -- -"$zpid" 2>/dev/null; then
            skip "bounded (f): a group holding nothing but a zombie reads as empty" \
              "this kernel's killpg already answers dead, so the leg cannot distinguish the shared state reader from the bare signal probe"
          elif group_alive "$zpid"; then
            report 1 "bounded (f): a group holding nothing but a zombie reads as empty" \
              "group_alive counted a zombie as work — bounded would wait out the whole bound"
          else
            report 0 "bounded (f): a group holding nothing but a zombie reads as empty"
          fi
          ;;
        *)
          report 1 "bounded (f): a group holding nothing but a zombie reads as empty" \
            "the child never reached state Z (saw '${zstate:-gone}'), so the leg tested nothing"
          ;;
      esac
    fi
    # Let the parent reap its own child, rather than orphaning the zombie onto
    # a PID 1 that may never reap it; then check we really left nothing behind.
    kill -CONT "$zparent" 2>/dev/null
    wait "$zparent" 2>/dev/null
    zleft=""
    if [ -n "$zpid" ]; then
      for _ in 1 2 3 4 5 6 7 8 9 10; do
        zleft="$(ps -o stat= -p "$zpid" 2>/dev/null | tr -d ' ')"
        [ -z "$zleft" ] && break
        sleep 0.1
      done
    fi
    if [ -z "$zleft" ]; then
      report 0 "bounded (f): the leg reaps its own zombie, leaving no process-table entry"
    else
      report 1 "bounded (f): the leg reaps its own zombie, leaving no process-table entry" \
        "pid $zpid is still present as '$zleft'"
    fi
  fi
fi

# ---------------------------------------------------------------------------
# Shared machinery for the hook-invoking legs
# ---------------------------------------------------------------------------
# A PATH with no opam on it: the hook then stops right after the section under
# test ("=== stopped: opam required ==="), so no run of it can pin packages or
# otherwise disturb this machine's opam state.
# Everything the hook can REACH before that stop, not merely enough to get it
# started: `group_alive` shells out to `ps` and `awk` wherever there is no
# /proc, so a farm without them would send every hook run on this PATH down the
# bare-signal fallback and quietly stop exercising the code this change adds.
# A missing tool is fatal rather than skipped, for the same reason.
BIN="$TMP/bin"
mkdir -p "$BIN"
missing_tools=""
for tool in git env sh sleep basename dirname tr grep sed cat ps awk; do
  p="$(command -v "$tool" 2>/dev/null)" || { missing_tools="$missing_tools $tool"; continue; }
  ln -sf "$p" "$BIN/$tool"
done
if [ -n "$missing_tools" ]; then
  echo "these tools the hook can reach are not on PATH:$missing_tools" >&2; exit 2
fi
if PATH="$BIN" command -v opam >/dev/null 2>&1; then
  echo "sanitised PATH still resolves opam; refusing to run the hook" >&2; exit 2
fi

FAKEHOME="$TMP/home"; mkdir -p "$FAKEHOME"
RUN_ENV=()
RUN_PATH="$BIN"

run_hook() { # run_hook CLONE OUTFILE   (extra env comes from the RUN_ENV array)
  local clone="$1" out="$2"
  ( cd "$clone" && env -i \
      HOME="$FAKEHOME" PATH="$RUN_PATH" \
      GIT_CONFIG_NOSYSTEM=1 GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_SYSTEM=/dev/null \
      ${RUN_ENV[@]+"${RUN_ENV[@]}"} \
      "$BASH_BIN" "$clone/scripts/setup-ocaml-env.sh" ) >"$out" 2>&1
  # Every run must have reached (and passed) the section under test. Without
  # this, a hook that died earlier would make the absence assertions pass.
  if ! grep -qF 'stopped: opam required' "$out"; then
    report 1 "harness: hook run in $(basename "$clone") reached the opam stop" \
      "output was: $(tr '\n' '|' <"$out")"
  fi
  RUN_ENV=()
  RUN_PATH="$BIN"
}

has() { grep -qF -- "$2" "$1"; }

# The setup commands need the same isolation as the hook runs, not just the
# same identity: a global `protocol.file.allow=never` would fail every local
# clone below, and a global `core.hooksPath` could reject or rewrite the
# synthetic commits — either way the harness would fail somewhere upstream of
# the thing it is testing, and say so misleadingly.
# The setup commands run in a CLEAN environment, the same way the hook runs do,
# rather than with known-bad variables subtracted one at a time. Subtraction was
# tried twice and was wrong twice: first the config files alone, which left
# GIT_DIR and its kin pointing git at a real repository REGARDLESS of `-C`
# (verified: with GIT_DIR exported, `git -C other config user.name X` writes
# into the GIT_DIR repo); then `git rev-parse --local-env-vars`, which does not
# list GIT_ALLOW_PROTOCOL, so an inherited transport policy without `file` still
# failed every local clone below. The set of variables git reads is not a list
# this harness can keep, so it keeps none of them: `env -i`, plus exactly what
# the commands need. PATH must survive for `env` to find git at all.
git_q() { env -i PATH="$PATH" HOME="$FAKEHOME" TMPDIR="$TMP" \
              GIT_CONFIG_NOSYSTEM=1 GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_SYSTEM=/dev/null \
          git -c advice.detachedHead=false -c init.defaultBranch=master \
              -c user.name=test -c user.email=test@example.invalid \
              -c commit.gpgsign=false "$@"; }

# ---------------------------------------------------------------------------
# Leg 2: counting
# ---------------------------------------------------------------------------
ORIGIN="$TMP/origin"
mkdir -p "$ORIGIN"
git_q -C "$ORIGIN" init -q
for n in 1 2 3 4 5 6 7 8 9 10; do
  echo "$n" >"$ORIGIN/f"
  git_q -C "$ORIGIN" add f
  git_q -C "$ORIGIN" commit -q -m "c$n"
done
# A second branch at a different commit, for leg (f) to seed FETCH_HEAD from.
git_q -C "$ORIGIN" branch side master~7

new_clone() { # new_clone NAME -> path of a fresh clone carrying the working-tree hook
  local d="$TMP/$1"
  rm -rf "$d"
  git_q clone -q "$ORIGIN" "$d"
  git_q -C "$d" config advice.detachedHead false
  git_q -C "$d" config user.name test
  git_q -C "$d" config user.email test@example.invalid
  mkdir -p "$d/scripts"
  cp "$HOOK_SRC" "$d/scripts/setup-ocaml-env.sh"
  cp "$GROUP_SRC" "$d/scripts/process-group.sh"
  printf '%s\n' "$d"
}

# (a) behind only -> the merge --ff-only recovery.
c="$(new_clone c-behind)"
git_q -C "$c" checkout -q --detach refs/remotes/origin/master~5
run_hook "$c" "$TMP/out-behind"
if has "$TMP/out-behind" 'WARNING HEAD is 5 commit(s) behind origin/master' \
   && has "$TMP/out-behind" 'recover with: git merge --ff-only refs/remotes/origin/master'; then
  report 0 "count (a): 5 behind, 0 ahead -> WARNING + merge --ff-only recovery"
else
  report 1 "count (a): 5 behind, 0 ahead -> WARNING + merge --ff-only recovery" \
    "$(tr '\n' '|' <"$TMP/out-behind")"
fi

# (b) behind and ahead -> the rebase recovery, with the replay count.
c="$(new_clone c-diverged)"
git_q -C "$c" checkout -q --detach refs/remotes/origin/master~5
git_q -C "$c" commit -q --allow-empty -m local1
git_q -C "$c" commit -q --allow-empty -m local2
run_hook "$c" "$TMP/out-diverged"
if has "$TMP/out-diverged" 'WARNING HEAD is 5 commit(s) behind origin/master' \
   && has "$TMP/out-diverged" 'recover with: git rebase refs/remotes/origin/master  (2 local commit(s) to replay)'; then
  report 0 "count (b): 5 behind, 2 ahead -> rebase recovery naming the 2 local commits"
else
  report 1 "count (b): 5 behind, 2 ahead -> rebase recovery naming the 2 local commits" \
    "$(tr '\n' '|' <"$TMP/out-diverged")"
fi

# (c) up to date.
c="$(new_clone c-current)"
run_hook "$c" "$TMP/out-current"
if has "$TMP/out-current" '  ok    up to date with origin/master' \
   && ! has "$TMP/out-current" 'WARNING'; then
  report 0 "count (c): up to date -> 'ok    up to date with origin/master'"
else
  report 1 "count (c): up to date -> 'ok    up to date with origin/master'" \
    "$(tr '\n' '|' <"$TMP/out-current")"
fi

# (d) fetch failure still counts, against the last successful fetch. The clone
#     already holds an origin/master from cloning; the origin URL is then
#     switched to https and pointed at a dead proxy.
c="$(new_clone c-offline)"
git_q -C "$c" checkout -q --detach refs/remotes/origin/master~5
git_q -C "$c" remote set-url origin https://example.invalid/ocannl.git
RUN_ENV=(https_proxy=http://127.0.0.1:9 HTTPS_PROXY=http://127.0.0.1:9 ALL_PROXY=http://127.0.0.1:9)
run_hook "$c" "$TMP/out-offline"
if has "$TMP/out-offline" '  skip  fetching origin/master failed (offline?)' \
   && has "$TMP/out-offline" 'WARNING HEAD is 5 commit(s) behind origin/master (as of the last successful fetch)'; then
  report 0 "count (d): failed fetch -> skip, then the count as of the last successful fetch"
else
  report 1 "count (d): failed fetch -> skip, then the count as of the last successful fetch" \
    "$(tr '\n' '|' <"$TMP/out-offline")"
fi

# (e) a local branch AND a tag both called `origin/master`, at a different
#     commit, must not be read in place of the tracking ref.
c="$(new_clone c-ambiguous)"
git_q -C "$c" checkout -q --detach refs/remotes/origin/master~5
git_q -C "$c" branch origin/master refs/remotes/origin/master~3
git_q -C "$c" tag origin/master refs/remotes/origin/master~3
run_hook "$c" "$TMP/out-ambiguous"
if has "$TMP/out-ambiguous" 'WARNING HEAD is 5 commit(s) behind origin/master' \
   && ! has "$TMP/out-ambiguous" 'HEAD is 3 commit(s) behind'; then
  report 0 "count (e): a branch and a tag named origin/master do not change the count"
else
  report 1 "count (e): a branch and a tag named origin/master do not change the count" \
    "$(tr '\n' '|' <"$TMP/out-ambiguous")"
fi

# (f) the probe must leave FETCH_HEAD byte-identical (--no-write-fetch-head):
#     someone may be keeping it for a later `git merge FETCH_HEAD`.
#     FETCH_HEAD is seeded from `side`, NOT from master: seeded from master the
#     hook's own fetch would write back the very same line, and the comparison
#     could not tell "did not write" from "wrote the same thing" — which is
#     exactly how this leg first shipped, passing against a hook with
#     `--no-write-fetch-head` removed.
c="$(new_clone c-fetchhead)"
git_q -C "$c" fetch -q origin side
master_sha="$(git_q -C "$c" rev-parse refs/remotes/origin/master 2>/dev/null || true)"
seeded_sha="$(head -n1 "$c/.git/FETCH_HEAD" 2>/dev/null | cut -f1)"
if [ ! -s "$c/.git/FETCH_HEAD" ]; then
  report 1 "count (f): FETCH_HEAD is byte-identical across a run" \
    "the setup fetch wrote no FETCH_HEAD, so there is nothing to compare"
elif [ -z "$master_sha" ] || [ "$seeded_sha" = "$master_sha" ]; then
  report 1 "count (f): FETCH_HEAD is byte-identical across a run" \
    "seeded FETCH_HEAD already names master ($seeded_sha) — the leg cannot discriminate"
else
  cp "$c/.git/FETCH_HEAD" "$TMP/fetch-head.before"
  run_hook "$c" "$TMP/out-fetchhead"
  if cmp -s "$TMP/fetch-head.before" "$c/.git/FETCH_HEAD"; then
    report 0 "count (f): FETCH_HEAD is byte-identical across a run"
  else
    report 1 "count (f): FETCH_HEAD is byte-identical across a run" \
      "before: $(cat "$TMP/fetch-head.before") / after: $(cat "$c/.git/FETCH_HEAD")"
  fi
fi

# (g) no `origin` remote: the section prints nothing at all.
c="$(new_clone c-noremote)"
git_q -C "$c" remote remove origin
run_hook "$c" "$TMP/out-noremote"
if ! grep -q 'origin/master' "$TMP/out-noremote"; then
  report 0 "count (g): no origin remote -> the section prints nothing"
else
  report 1 "count (g): no origin remote -> the section prints nothing" \
    "$(tr '\n' '|' <"$TMP/out-noremote")"
fi

# ---------------------------------------------------------------------------
# Leg 3: SSH launcher gating
# ---------------------------------------------------------------------------
# A logging fake launcher over an ssh:// origin answers two questions per case:
# WHICH program git ended up invoking, and whether the OpenSSH options were
# appended to it. The options are only correct where OpenSSH is certain.
#
# The cases are launched concurrently and joined once: each has its own clone,
# log, launcher directory and output file, so nothing is shared but the machine.
# They used to have no choice — every one of them cost the full 30s bound,
# because git's ssh child was still a zombie in the process group when `bounded`
# tested it for emptiness. `bounded` now waits for the group to drain, which
# leg 1 (f) pins, and these are quick; running them together is just cheap.
# Spelled out here rather than read from the hook: a harness that took the list
# from the thing under test would follow a regression instead of catching it.
# The hook's own spelling is pinned by the leg below, so adding or removing an
# option there fails loudly instead of silently narrowing what these legs check.
SSH_OPT_BUNDLE="BatchMode=yes ConnectTimeout=10 ServerAliveInterval=5 ServerAliveCountMax=2"
ssh_opts_line=""
for opt in $SSH_OPT_BUNDLE; do ssh_opts_line="$ssh_opts_line -o $opt"; done
if grep -qF -- "ssh_opts=\"${ssh_opts_line# }\"" "$HOOK_SRC"; then
  report 0 "ssh (0): the hook appends exactly the bundle these legs assert on"
else
  report 1 "ssh (0): the hook appends exactly the bundle these legs assert on" \
    "hook has: $(grep -n 'ssh_opts=' "$HOOK_SRC" | tr -d '\n')"
fi

# The fake launchers must sit at a SHELL-SAFE path — no whitespace, and nothing
# a shell would rewrite ($ ` \ ' " ; & | etc). GIT_SSH_COMMAND and
# core.sshCommand are shell COMMAND STRINGS, so git splits them on whitespace,
# and the hook reads the first word to decide whether the program is OpenSSH.
# Quoting the path would satisfy git and defeat that read — the hook would see
# `'"'"'/tmp/ocannl` and decline to append, which is its documented, deliberate
# behaviour for a launcher it cannot identify, not something for the harness to
# work around. So with a TMPDIR containing spaces the fakes go somewhere else,
# and if there is nowhere, these legs say so once instead of failing fifteen
# times over while appearing to test launcher gating.
# Whitespace was only the first way this bites: TMPDIR='/tmp/ocannl-$x' has none
# and still breaks, because `$x` is expanded by the shell git runs the launcher
# through. So the test is a WHITELIST of characters that survive a shell
# unchanged, not a blacklist of the ones already seen to fail.
shell_safe_path() { # shell_safe_path PATH -> 0 if a shell leaves it alone
  case "$1" in
    ""|*[!A-Za-z0-9._/+-]*) return 1 ;;
    *) return 0 ;;
  esac
}
LAUNCH_ROOT="$TMP/launchers"
if ! shell_safe_path "$LAUNCH_ROOT"; then
  LAUNCH_ROOT="$(mktemp -d /tmp/ocannl-ssh-launchers.XXXXXX 2>/dev/null)" || LAUNCH_ROOT=""
  shell_safe_path "$LAUNCH_ROOT" || LAUNCH_ROOT=""
fi
if [ -z "$LAUNCH_ROOT" ]; then
  SKIP_SSH=1
  report 1 "ssh: the launcher-gating legs need a shell-safe directory for the fakes" \
    "TMPDIR holds characters a shell would rewrite and /tmp is unusable; these legs did not run"
else
  SKIP_SSH=0
fi

SSH_BASE="$TMP/ssh-base"
mkdir -p "$SSH_BASE/scripts"
cp "$GROUP_SRC" "$SSH_BASE/scripts/process-group.sh"
git_q -C "$SSH_BASE" init -q
git_q -C "$SSH_BASE" commit -q --allow-empty -m base
git_q -C "$SSH_BASE" remote add origin ssh://git@example.invalid/x.git

CASEDIR=""; CASELOG=""; LAUNCHER=""
ssh_prepare() { # ssh_prepare SLUG LAUNCHER_BASENAME -- sets CASEDIR/CASELOG/LAUNCHER
  [ "$SKIP_SSH" = 0 ] || return 0
  CASEDIR="$TMP/ssh-$1"
  CASELOG="$TMP/ssh-$1.log"
  rm -rf "$CASEDIR"
  cp -r "$SSH_BASE" "$CASEDIR"
  cp "$HOOK_SRC" "$CASEDIR/scripts/setup-ocaml-env.sh"
  : >"$CASELOG"
  mkdir -p "$LAUNCH_ROOT/$1"
  LAUNCHER="$(ssh_launcher "$1" "$2")"
}

sq() { # sq WORD -> WORD as a single-quoted shell literal, safe to embed in source
  local w="$1"
  w="${w//\'/\'\\\'\'}"
  printf "'%s'" "$w"
}

ssh_launcher() { # ssh_launcher SLUG BASENAME -> path of a fake logging into that case's log
  local p="$LAUNCH_ROOT/$1/$2"
  # Program and arguments are logged as two TAB-separated fields, not as one
  # space-joined line: a TMPDIR containing whitespace makes "$0 $*" ambiguous,
  # and the reader would take the first word of the PATH as the program.
  #
  # The log path is embedded as a single-quoted shell LITERAL, not interpolated
  # into double quotes: this is generated shell source, so a `$` or a backtick
  # in TMPDIR would otherwise be expanded when the launcher runs and every log
  # would come out empty. (The launcher's own path cannot be quoted the same
  # way — see the LAUNCH_ROOT note above for why that one is solved by choosing
  # a safe path instead.)
  { printf '#!/bin/sh\n'
    printf 'printf "%%s\\t%%s\\n" "$0" "$*" >> %s\n' "$(sq "$TMP/ssh-$1.log")"
    printf 'exit 255\n'
  } >"$p"
  chmod +x "$p"
  printf '%s\n' "$p"
}


SSH_SLUGS=(); SSH_LABELS=(); SSH_PROGS=(); SSH_OPTS=()
ssh_launch() { # ssh_launch SLUG LABEL EXPECTED_PROGRAM_BASENAME yes|no
  if [ "$SKIP_SSH" != 0 ]; then RUN_ENV=(); RUN_PATH="$BIN"; return 0; fi
  SSH_SLUGS+=("$1"); SSH_LABELS+=("$2"); SSH_PROGS+=("$3"); SSH_OPTS+=("$4")
  local clone="$TMP/ssh-$1"
  ( cd "$clone" && env -i \
      HOME="$FAKEHOME" PATH="$RUN_PATH" \
      GIT_CONFIG_NOSYSTEM=1 GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_SYSTEM=/dev/null \
      ${RUN_ENV[@]+"${RUN_ENV[@]}"} \
      "$BASH_BIN" "$clone/scripts/setup-ocaml-env.sh" ) >"$TMP/out-ssh-$1" 2>&1 &
  RUN_ENV=()
  RUN_PATH="$BIN"
}

# Options appended: GIT_SSH_COMMAND whose program is `ssh`.
ssh_prepare cmd-ssh ssh
RUN_ENV=(GIT_SSH_COMMAND="$LAUNCHER")
ssh_launch cmd-ssh "ssh (1): GIT_SSH_COMMAND named ssh -> options appended" ssh yes

# Options appended: core.sshCommand whose program is `ssh`.
ssh_prepare cfg-ssh ssh
git_q -C "$CASEDIR" config core.sshCommand "$LAUNCHER"
ssh_launch cfg-ssh "ssh (2): core.sshCommand named ssh -> options appended" ssh yes

# Options appended: nothing configured, so git's default `ssh` off PATH.
ssh_prepare path-ssh ssh
RUN_PATH="$LAUNCH_ROOT/path-ssh:$BIN"
ssh_launch path-ssh "ssh (3): PATH default ssh -> options appended" ssh yes

# Options appended: an `ssh.exe` basename is OpenSSH too (Git for Windows).
ssh_prepare cmd-sshexe ssh.exe
RUN_ENV=(GIT_SSH_COMMAND="$LAUNCHER")
ssh_launch cmd-sshexe "ssh (4): GIT_SSH_COMMAND named ssh.exe -> options appended" ssh.exe yes

# Options NOT appended: a custom wrapper name, variant unknown.
ssh_prepare cmd-wrapper my-ssh-wrapper
RUN_ENV=(GIT_SSH_COMMAND="$LAUNCHER")
ssh_launch cmd-wrapper "ssh (5): custom wrapper name -> options NOT appended" my-ssh-wrapper no

# Options NOT appended: plink and its kin.
ssh_prepare cmd-plink plink
RUN_ENV=(GIT_SSH_COMMAND="$LAUNCHER")
ssh_launch cmd-plink "ssh (6): plink -> options NOT appended" plink no

ssh_prepare cmd-tortoise tortoiseplink
RUN_ENV=(GIT_SSH_COMMAND="$LAUNCHER")
ssh_launch cmd-tortoise "ssh (7): tortoiseplink -> options NOT appended" tortoiseplink no

ssh_prepare cmd-putty putty
RUN_ENV=(GIT_SSH_COMMAND="$LAUNCHER")
ssh_launch cmd-putty "ssh (8): putty -> options NOT appended" putty no

# Options NOT appended: ssh.variant=simple, even for a launcher named ssh.
ssh_prepare cfg-simple ssh
git_q -C "$CASEDIR" config ssh.variant simple
RUN_ENV=(GIT_SSH_COMMAND="$LAUNCHER")
ssh_launch cfg-simple "ssh (9): ssh.variant=simple -> options NOT appended" ssh no

# Options NOT appended: GIT_SSH is a program, not a shell string — declined on
# the mechanism, not the name, so this fake is deliberately called `ssh`.
ssh_prepare env-gitssh ssh
RUN_ENV=(GIT_SSH="$LAUNCHER")
ssh_launch env-gitssh "ssh (10): GIT_SSH program named ssh -> options NOT appended" ssh no

# GIT_SSH_VARIANT outranks ssh.variant, and an explicit `ssh` variant is enough
# even for a custom program name.
ssh_prepare variant-override my-ssh-wrapper
git_q -C "$CASEDIR" config ssh.variant simple
RUN_ENV=(GIT_SSH_COMMAND="$LAUNCHER" GIT_SSH_VARIANT=ssh)
ssh_launch variant-override \
  "ssh (11): GIT_SSH_VARIANT=ssh outranks ssh.variant=simple -> options appended" my-ssh-wrapper yes

# Git's own precedence GIT_SSH_COMMAND > core.sshCommand > GIT_SSH survives the
# probe: three launchers with distinct names, one dropped per case. Custom names
# throughout, so the script appends nothing and cannot itself be the reason a
# given launcher won.
ssh_prepare prec-cmd ssh-a
git_q -C "$CASEDIR" config core.sshCommand "$(ssh_launcher prec-cmd ssh-b)"
RUN_ENV=(GIT_SSH_COMMAND="$LAUNCHER" GIT_SSH="$(ssh_launcher prec-cmd ssh-c)")
ssh_launch prec-cmd "ssh (12): GIT_SSH_COMMAND beats core.sshCommand and GIT_SSH" ssh-a no

ssh_prepare prec-cfg ssh-b
git_q -C "$CASEDIR" config core.sshCommand "$LAUNCHER"
RUN_ENV=(GIT_SSH="$(ssh_launcher prec-cfg ssh-c)")
ssh_launch prec-cfg "ssh (13): core.sshCommand beats GIT_SSH" ssh-b no

ssh_prepare prec-env ssh-c
RUN_ENV=(GIT_SSH="$LAUNCHER")
ssh_launch prec-env "ssh (14): GIT_SSH is used when nothing outranks it" ssh-c no

wait

i=0
while [ "$i" -lt "${#SSH_SLUGS[@]}" ]; do
  slug="${SSH_SLUGS[$i]}"; label="${SSH_LABELS[$i]}"
  want_prog="${SSH_PROGS[$i]}"; want_opts="${SSH_OPTS[$i]}"
  log="$TMP/ssh-$slug.log"; out="$TMP/out-ssh-$slug"
  i=$((i + 1))
  if ! grep -qF 'stopped: opam required' "$out"; then
    report 1 "$label" "the hook did not reach the opam stop: $(tr '\n' '|' <"$out")"
    continue
  fi
  # Git probes an unrecognised launcher with `-G` to detect its variant; that
  # line is not the fetch invocation, so it is filtered out.
  line="$(grep -vE '(^|[ 	])-G([ 	]|$)' "$log" | head -n1)"
  if [ -z "$line" ]; then
    report 1 "$label" "no ssh launcher invocation logged (raw log: $(tr '\n' '|' <"$log"))"
    continue
  fi
  prog="${line%%	*}"          # first TAB-separated field: the program
  prog="$(basename "$prog")"
  line="${line#*	}"             # the rest: the arguments as git passed them
  # Classified on the WHOLE bundle, not on BatchMode alone: an option silently
  # dropped from the hook, or one leaking onto a launcher that must not get it,
  # is exactly the regression that puts the long startup stalls back.
  seen=""; absent=""
  for opt in $SSH_OPT_BUNDLE; do
    case "$line" in
      *"-o $opt"*) seen="$seen $opt" ;;
      *) absent="$absent $opt" ;;
    esac
  done
  if [ "$want_opts" = yes ]; then opts_ok="$absent"; else opts_ok="$seen"; fi
  if [ "$prog" = "$want_prog" ] && [ -z "$opts_ok" ]; then
    report 0 "$label"
  elif [ "$prog" != "$want_prog" ]; then
    report 1 "$label" "invoked '$prog', wanted '$want_prog'; line: $line"
  elif [ "$want_opts" = yes ]; then
    report 1 "$label" "options missing:$absent; line: $line"
  else
    report 1 "$label" "options wrongly appended:$seen; line: $line"
  fi
done

# The symptom all of that is for, measured end to end: a failing ssh fetch must
# come back promptly. Run serially and timed — the cases above run concurrently,
# which is exactly what makes their wall clock unusable as evidence.
#
# Weaker than leg 1 (f) on purpose, and it does not replace it. Where the ssh
# child's zombie is reaped promptly this is a RACE: degrading `group_alive` back
# to a bare `kill -0` was measured passing here, the few microseconds of the
# function call being enough to win it, while leg (f) reddened. So a pass here
# is not evidence the group check is sound; a failure is evidence it is not, and
# on a PID 1 that does not reap — where the misreading is permanent rather than
# raced — this is the leg that reports the 30s stall as a stall.
if [ "$SKIP_SSH" = 0 ]; then
ssh_prepare timing ssh
RUN_ENV=(GIT_SSH_COMMAND="$LAUNCHER")
t0=$SECONDS
run_hook "$CASEDIR" "$TMP/out-ssh-timing"
el=$((SECONDS - t0))
if [ "$el" -le 10 ]; then
  report 0 "ssh (15): a failing ssh fetch returns well inside the 30s bound (${el}s)"
else
  report 1 "ssh (15): a failing ssh fetch returns well inside the 30s bound" \
    "took ${el}s — the group check is counting the ssh child's zombie as work again"
fi
fi

# ---------------------------------------------------------------------------
finish
