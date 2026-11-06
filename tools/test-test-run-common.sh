#!/usr/bin/env bash
# What the members of tools/test-run.sh's harness share (gh-ocannl-2002):
# sourced by each tools/test-test-run-NAME.sh, never run. The harness was one
# file of some 70 legs and about 760 s; it is six members so that a change reaches
# the members naming what it touched, and so that tools/test-harnesses.sh
# --jobs runs them beside one another: half of those seconds are fixtures
# waiting, not computing. tools/test-test-run.sh runs them from one command.
#
# This file holds what every member needs before its first leg (its options,
# the hermetic environment, the subject and the scratch directory), the
# fixtures legs of more than one member run against, and the leg every member
# closes with. A fixture only one member's legs read sits in that member.
#
# --subject FILE (or OCANNL_TOOL_HARNESS_SUBJECT=FILE; the option wins) tests
# FILE in place of tools/test-run.sh (gh-ocannl-1925): the control "this leg
# fails against the previous script" is `git show REV:tools/test-run.sh >F`
# and `--subject F`, with no copy of tools/ and scripts/. FILE is copied to
# tools/test-run.sh of a root under the scratch directory, beside the files it
# sources (stage_sourced; those come from THIS tree, so a FILE sourcing one the
# tree lacks is refused, exit 2), and that copy is the subject of every leg:
# the one each fixture root is given, the one the mutants are cut from and the
# one run directly. Everything else under test (scripts/process-group.sh, the
# sourced tools/*.sh, the member) stays the working tree's. The header of
# such a run and the line above its footer name FILE: its verdict is FILE's.
#
# Each leg's body is a function, run by `harness_leg_run 'N: what it pins
# #group' leg_N` (scripts/harness-support.sh), its name led by its leg numbers
# and closed by the one group it belongs to (gh-ocannl-1202); a member's
# header declares its groups. A body that cannot go on (an extraction that
# matched nothing, a fixture that is not what its checks assume) ends with
# `harness_abandon DETAIL; return`, failing that leg alone: never `exit`,
# which would end the member and hide every leg after it (gh-ocannl-1374).
# A shared fixture outside the legs that is not what they assume is a FAIL
# where it is built, and the run goes on; only a subject that cannot be staged
# at all (a missing script, the sourced-files scan) still stops the member. A
# selector is an extended regular expression over those names; one matching no
# leg of a member is refused there (exit 2) as nothing tested. The shared
# fixtures stay outside the gates, so a filtered run sets up and cleans up as
# a full one does. A filtered run prints FILTERED for each leg it left out,
# then accounts for the legs it ran and every check they skipped with its
# reason, and never prints the full-pass footer.
#
# The shared fixtures are STAGES, run in the one order the legs were written
# against: `shared_through NAME`, ahead of a leg, runs every stage not yet run
# up to NAME, so each leg of a member starts from what it started from when
# the harness was one file, the legs before it that now sit elsewhere aside
# (tools/harness-legs-alone.sh held every leg to that already). A member runs
# no stage past its last leg's. A new leg names the last stage it reads; a
# fixture a second member comes to need moves here as a stage, in its place
# in shared_stages.
#
# The harness is the sibling of scripts/test-setup-ocaml-env.sh, whose leg 1
# (f) tests the same predicate in the SessionStart hook. Neither is a dune
# test: they spawn, STOP and kill process groups, which is a poor fit for
# `dune runtest`. The Ubuntu CI leg runs both directly so Linux decides the
# kernel-dependent zombie-group control (gh-ocannl-795).
#
# It tests the WORKING-TREE copy: `group_alive` is extracted from the shared
# scripts/process-group.sh; `ps_token`, `proc_identity_matches`, `proc_alive`
# come from tools/test-run.sh; the dxg legs read the shipping tools/box-jobs.sh
# and tools/lab-map.sh for endpoint lookup. The `stop` legs drive that same tool as a
# subprocess. Each extraction is asserted structurally before use, so a sed
# that matched nothing cannot leave every leg passing without testing anything.
#
#  70. every member closes with it, after every leg that fakes a device: a
#      leg that names no device probe still reads the harness's absent
#      defaults, the width-cap probes answer "no device" through the shipping
#      readers, and a hip run is neither capped nor told anything
#      (gh-ocannl-1108).

set -u

. "$(cd "$(dirname "$0")/../scripts" && pwd)/harness-support.sh"
# --subject is this harness's own option: taken out of the words before the
# shared parser, which knows none of it, reads the rest as its own "$@".
subject=${OCANNL_TOOL_HARNESS_SUBJECT-} subject_set=0
harness_argv=()
while [ $# -gt 0 ]; do
  case $1 in
    --subject=*) subject=${1#--subject=} subject_set=1 ;;
    --subject)
      [ $# -ge 2 ] || { echo "${0##*/}: --subject needs a file" >&2; exit 2; }
      subject=$2 subject_set=1
      shift ;;
    --only)
      # Its value is not an option, whatever it looks like.
      harness_argv+=("$1")
      [ $# -lt 2 ] || { harness_argv+=("$2"); shift; } ;;
    *) harness_argv+=("$1") ;;
  esac
  shift
done
[ "$subject_set" = 0 ] || [ -n "$subject" ] || { echo "${0##*/}: --subject needs a file" >&2; exit 2; }
set -- ${harness_argv[@]+"${harness_argv[@]}"}
harness_args "$@"

# Hermetic against the fleet (gh-ocannl-1004): on a fleet box the shipping
# script takes a real run-time slot through the deployed fleet-worker.sh, and
# the fixture runs below must neither hold nor wait for one. The slot legs
# name their fake fleet-worker.sh explicitly.
export OCANNL_TOOL_FLEET_WORKER=none
# Hermetic against the caller's dune presentation: a DUNE_DIFF_COMMAND in the
# environment is recorded by every launch and changes what the digest may
# conclude from a log (gh-ocannl-1055); the legs that want one set it.
unset DUNE_DIFF_COMMAND
# Hermetic against the caller's backend from the first leg, not from
# dxg_probe's first `unset`: a worker's shell pinned to hip or cuda outranks
# the fixture configurations every backend leg resolves (gh-ocannl-1108); the
# legs that want one export it. The box's own devices are masked as soon as
# the scratch directory exists (hermetic_probes, below).
unset OCANNL_BACKEND
# Hermetic against a caller that allowed its own unmerged tree
# (gh-ocannl-1991): leg 104 sets the override where it means it.
unset OCANNL_TOOL_TEST_UNMERGED

HERE="$(cd "$(dirname "$0")" && pwd)"
# The member being run: the legs that run or cut a copy of their own harness name it.
SELF=$HERE/${0##*/}
# The subject as named, until the scratch directory exists to stage it in.
SRC=${subject:-$HERE/test-run.sh}
GROUP_SRC="$HERE/../scripts/process-group.sh"
HOOK_SRC="$HERE/../scripts/setup-ocaml-env.sh"
JOBS_SRC="$HERE/box-jobs.sh"
SWEEP_SRC="$HERE/sweep.sh"
[ -f "$SRC" ] && [ -r "$SRC" ] || { echo "no $SRC" >&2; exit 2; }
[ -f "$GROUP_SRC" ] || { echo "no $GROUP_SRC" >&2; exit 2; }
[ -f "$HOOK_SRC" ] || { echo "no $HOOK_SRC" >&2; exit 2; }
[ -f "$JOBS_SRC" ] || { echo "no $JOBS_SRC" >&2; exit 2; }
[ -f "$SWEEP_SRC" ] || { echo "no $SWEEP_SRC" >&2; exit 2; }


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
[ -z "$subject" ] || echo "  subject: $SRC in place of $HERE/test-run.sh"
printf '  digest %s\n' "$( (cksum <"$SRC") 2>/dev/null || echo '?')"
printf '  group digest %s\n' "$( (cksum <"$GROUP_SRC") 2>/dev/null || echo '?')"
printf '  state reader: %s; pgid reader: %s\n' \
  "$([ "$have_state" = 1 ] && echo present || echo ABSENT)" \
  "$([ "$have_pgid" = 1 ] && echo present || echo ABSENT)"
# What this harness inherited as ignored (gh-ocannl-1236): `bg-run.sh spawn`,
# `nohup` and `execution slot --bg` start it with HUP ignored, and a
# non-interactive bash cannot trap a signal ignored at its entry. The legs
# that deliver a signal reset it for their fixture, so this line explains a
# run rather than deciding one; perl reads the disposition the shell cannot.
printf '  inherited ignored signals: %s\n' \
  "$(perl -e 'my @i = grep { ($SIG{$_} // "") eq "IGNORE" } qw(HUP INT QUIT TERM);
              print @i ? "@i" : "none"' 2>/dev/null || echo '?')"

# Checked, not assumed: nothing here uses `set -e`, so a `mktemp` that fails
# would leave TMP empty and `rm -rf "$TMP"` would be handed the ROOT.
zparent=""   # leg 4's self-stopping zombie maker; cleanup must resume it
livepid=""   # leg 2's live group leader
leader=""    # the stop legs' current group leader
member=""    # and the second process it put in that group
member_token=""  # ... and its start token, since it is not this shell's child
repeat_pid="" # leg 14's active repeat coordinator
legacy_holder="" # leg 8's holder of the in-tree lock
inherit_holder="" # leg 67's holder of the inherited lock
escape_pid="" # leg 24's session-escaped descendant
escape_release=""
harness_scratch "test-test-run"

# ---------------------------------------------------------------------------
# Everything the shipping script SOURCES has to exist in a fixture root, and the
# list is derived from the script rather than kept here: a `. tools/<new>.sh`
# added to test-run.sh otherwise makes every leg die at startup, with the
# real message buried under forty failures (the Ubuntu leg of gh-ocannl-983's
# first CI round, where the mutation harness hit exactly this). The floor is the
# other half: a regex that stops matching must fail loudly, not stage nothing.
# A leg staging a root of its own abandons itself on a failure, as any leg's
# structural failure does (gh-ocannl-1374): `stage_sourced ROOT ||
# { harness_abandon "$stage_error"; return; }`. The two roots most of the legs
# share (repeat_root, life_root) are the exception, staged by stage_shared: the
# scan is the same for every root, so where it fails no later leg can stage its
# subject either, and the harness stops there with the reason (exit 2).
stage_error=
stage_sourced() { # <fixture root>; 1 with $stage_error saying why
  local dest=$1 rel staged=0
  stage_error=
  while IFS= read -r rel; do
    mkdir -p "$dest/$(dirname "$rel")" && cp "$HERE/../$rel" "$dest/$rel" ||
      { stage_error="cannot stage $rel into $dest"; return 1; }
    staged=$((staged + 1))
  done < <(sed -n 's/^\. \([A-Za-z0-9_./-]*\)$/\1/p' "$SRC")
  [ "$staged" -ge 3 ] ||
    { stage_error="only $staged sourced file(s) found in $SRC; the scan is broken"; return 1; }
}
stage_shared() { # <fixture root most later legs run against>
  stage_sourced "$1" || { echo "$stage_error" >&2; exit 2; }
}

# A subject named with --subject is staged once, here, before any leg reads
# SRC: a root of its own holding it as tools/test-run.sh beside what it
# sources, because the script works from the parent of the directory it sits
# in (the legs that run "$SRC" directly) and every fixture root is given a
# copy of SRC. From here on SRC is that copy, as it is the working tree's
# script when no subject is named.
if [ -n "$subject" ]; then
  mkdir -p "$TMP/subject/tools" && cp "$SRC" "$TMP/subject/tools/test-run.sh" &&
    chmod +x "$TMP/subject/tools/test-run.sh" ||
    { echo "cannot stage the subject $subject under $TMP" >&2; exit 2; }
  SRC=$TMP/subject/tools/test-run.sh
  stage_shared "$TMP/subject"
fi
# Hermetic against the box's devices (gh-ocannl-1108), as
# tools/test-mutation-run.sh is: the three probes the width cap reads (the dxg
# bridge, the KFD topology, the NVIDIA control device) name paths that do not
# exist, so on a WSL, minix, tuf or rog boot every leg sees the Mac's answer --
# a leg that is not about the cap would otherwise gain one (a lifecycle run on
# minix is capped at -j 4), and a cap leg's negative control would read the
# box instead of its fixture. A leg opts in to a device by exporting a fixture
# path (dxg_probe, native_probe) or prefixing one, and goes back through
# hermetic_probes, never `unset`: an unset (or empty) probe falls back to the
# real device. Leg 70 pins this from the end of the file.
probe_dxg_default=$TMP/no-such-dxg
probe_kfd_default=$TMP/no-such-kfd
probe_nvidia_default=$TMP/no-such-nvidia
hermetic_probes() {
  export OCANNL_TOOL_DXG_DEVICE=$probe_dxg_default \
    OCANNL_TOOL_KFD_TOPOLOGY=$probe_kfd_default \
    OCANNL_TOOL_NVIDIA_DEVICE=$probe_nvidia_default
}
hermetic_probes
cleanup_fixture() {
  if declare -F lifecycle_cleanup >/dev/null; then lifecycle_cleanup; fi
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
  if declare -F inherit_kill >/dev/null; then inherit_kill; fi
  if declare -F q_kill_all >/dev/null; then q_kill_all; fi
  [ -z "${escape_release:-}" ] || touch "$escape_release"
  if [ -n "${escape_pid:-}" ] && kill -0 "$escape_pid" 2>/dev/null; then
    kill -KILL "$escape_pid" 2>/dev/null
  fi
  return 0
}
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
shared_stop() {
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
stop_last=$(OCANNL_TOOL_TEST_RUNS="$STOP_RUNS" "$SRC" paths last) || stop_last=

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
  printf 'runtest (the harness fixture %s)\n' "$1" >"$d/cmd"
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
}

shared_repeat_root() {
# ---------------------------------------------------------------------------
# Legs 9-17: repeat mode's output, lifecycle and exit-code contract
# ---------------------------------------------------------------------------
# The fixture root is the shipping script plus what it sources (stage_sourced,
# defined before leg 67).
repeat_root=$TMP/repeat-repo
repeat_bin=$TMP/repeat-bin
mkdir -p "$repeat_root/tools" "$repeat_root/scripts" "$repeat_bin"
cp "$SRC" "$repeat_root/tools/test-run.sh"
stage_shared "$repeat_root"
chmod +x "$repeat_root/tools/test-run.sh"
# The two readers tools/batch-backends.sh would build, faked for every leg
# (gh-ocannl-1066): a run on a box where a backend meets a width cap, or under
# a fleet slot, resolves the batch's backends, and a build through the fixture
# dune would land in the calls every leg pins. The reader answers a test
# configuration's backend from the directory it runs in, the environment
# outranking the file as in the real resolution (FAKE_BACKEND_TEST and
# FAKE_BACKEND_ARRAYJIT stand for the files, cc by default as the tracked ones
# say; `fail` fails the read, `unset` resolves none). The reachability tool
# records the argv it was asked about and prints FAKE_REACH (by default: no
# stanza naming a backend, and one reading the configuration -- FAKE_READS --
# so the configurations count, as they do for any batch off the
# configuration-free list; leg 78 drives one that reads none, gh-ocannl-1227).
mkdir -p "$repeat_root/test/config" "$repeat_root/arrayjit/test"
# A build directory, as any worktree that has built has: without one the run's
# last phase records an empty promotion list without asking dune (gh-ocannl-1087).
mkdir -p "$repeat_root/_build"
batch_reader=$TMP/fake-read-config.sh
cat >"$batch_reader" <<'FAKE'
#!/usr/bin/env bash
case $PWD in
  */test/config) v=${FAKE_BACKEND_TEST:-cc} ;;
  */arrayjit/test) v=${FAKE_BACKEND_ARRAYJIT:-cc} ;;
  *) exit 3 ;;
esac
[ "$v" = fail ] && exit 1
[ -z "${OCANNL_BACKEND:-}" ] || v=$OCANNL_BACKEND
[ "$v" = unset ] && v=
printf '%s\n' "$v"
FAKE
batch_reach=$TMP/fake-slot-kind.sh
export FAKE_READS='reads config: it reaches t_cfg in x, which reads the configuration'
cat >"$batch_reach" <<'FAKE'
#!/usr/bin/env bash
[ -z "${FAKE_REACH_CALLS:-}" ] || printf '%s\n' "$*" >>"$FAKE_REACH_CALLS"
[ -z "${FAKE_REACH_ESCAPE:-}" ] || perl -MPOSIX -e '
  $SIG{$_} = "IGNORE" for qw(TERM INT HUP);
  POSIX::setsid() >= 0 or die "setsid: $!";
  open my $pf, ">", "$ARGV[0].tmp" or die; print $pf "$$\n"; close $pf;
  rename "$ARGV[0].tmp", $ARGV[0]; sleep 120' "$FAKE_REACH_ESCAPE" </dev/null >/dev/null 2>&1 &
[ -z "${FAKE_REACH_IGNORE_TERM:-}" ] || trap '' TERM INT HUP
[ -z "${FAKE_REACH_SLEEP:-}" ] || sleep "$FAKE_REACH_SLEEP"
[ -z "${FAKE_REACH_LEAVE:-}" ] || sleep "$FAKE_REACH_LEAVE" &
printf '%b\n' "${FAKE_REACH:-$FAKE_READS\nend}"
exit "${FAKE_REACH_EXIT:-0}"
FAKE
chmod +x "$batch_reader" "$batch_reach"
export OCANNL_TOOL_READ_CONFIG=$batch_reader OCANNL_TOOL_SLOT_KIND=$batch_reach
# Nor does any leg probe a device unless it says so (gh-ocannl-1211): the probe
# of a GPU batch would build bin/device_props through the fixture dune, and its
# `_probe` would stand between the slot and dune in every call the slot legs
# pin. Legs 75-77 name a stand-in. And no caller's bounds leak in: a raised
# per-test cap (an agent running @slow exports one) would change every run's
# watcher here and the values legs 75-77 read back.
export OCANNL_TOOL_DEVICE_PROBE=none
unset OCANNL_TOOL_PER_TEST_CAP OCANNL_TOOL_DEVICE_PROBE_CAP
}

shared_fixture_dune() {
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
# The run's last phase asks dune for its version and its promotion list
# (gh-ocannl-1087), still under the lock. The version is
# REPEAT_TEST_DUNE_VERSION (3.24.2 by default; `fail` fails). The list is the
# lines of the file REPEAT_TEST_PROMOTIONS names, printed on stdout from 3.22
# and on stderr before, as dune does, with REPEAT_TEST_PROMOTION_NOISE on the
# other stream; unset, the list fails, so every leg that sets none digests
# from the log. Neither query is counted or recorded among the calls every leg
# pins; the list's argv goes to REPEAT_TEST_PROMOTION_CALLS when that is set.
if [ "${1:-}" = --version ]; then
  [ "${REPEAT_TEST_DUNE_VERSION:-}" != fail ] || exit 1
  printf '%s\n' "${REPEAT_TEST_DUNE_VERSION:-3.24.2}"
  exit 0
fi
# These list/digest fixtures have no corrected payload; the separate real-Dune
# test-promotion-record harness tests capture and replay.
if [ "${1:-}" = promotion ] && [ "${2:-}" = diff ]; then exit 1; fi
if [ "${1:-}" = promotion ]; then
  [ -z "${REPEAT_TEST_PROMOTION_CALLS:-}" ] || printf '%s\n' "$*" >>"$REPEAT_TEST_PROMOTION_CALLS"
  [ -n "${REPEAT_TEST_PROMOTIONS:-}" ] || exit 1
  case ${REPEAT_TEST_DUNE_VERSION:-3.24.2} in
    3.2[01].* | 3.1[0-9].*)
      cat "$REPEAT_TEST_PROMOTIONS" >&2
      [ -z "${REPEAT_TEST_PROMOTION_NOISE:-}" ] || printf '%s\n' "$REPEAT_TEST_PROMOTION_NOISE"
      ;;
    *)
      cat "$REPEAT_TEST_PROMOTIONS"
      [ -z "${REPEAT_TEST_PROMOTION_NOISE:-}" ] || printf '%s\n' "$REPEAT_TEST_PROMOTION_NOISE" >&2
      ;;
  esac
  exit 0
fi
# The alias check's listing (gh-ocannl-1448), asked before the batch resolves:
# neither counted nor recorded among the calls every leg pins. Where
# REPEAT_TEST_ALIASES names a file of `DIR NAME` lines (`DIR` alone: a
# directory with no alias of its own) it answers as dune 3.24 does, on stderr
# -- each known directory's aliases with the standard `default`, `DIR:` headers
# for more than one, and nothing but `Error: Directory DIR does not exist.`
# lines once one is unknown, its `Error` tag colored as dune colors it under
# CLICOLOR_FORCE=1 (gh-ocannl-1591); unset, it prints nothing, which is not
# dune's answer, so the check leaves every alias to dune. The directories each
# query named go to REPEAT_TEST_SHOW_CALLS when that is set.
if [ "${1:-}" = show ]; then
  [ -n "${REPEAT_TEST_ALIASES:-}" ] || exit 0
  shift 2
  show_dirs=()
  for arg do case $arg in -*) ;; *) show_dirs+=("$arg") ;; esac; done
  [ -z "${REPEAT_TEST_SHOW_CALLS:-}" ] ||
    printf '%s\n' "${show_dirs[*]-}" >>"$REPEAT_TEST_SHOW_CALLS"
  show_missing=
  for d in ${show_dirs[@]+"${show_dirs[@]}"}; do
    show_error=Error
    [ "${CLICOLOR_FORCE-0}" = 0 ] || show_error=$'\e[1;31mError\e[0m'
    awk -v d="$d" '$1 == d { f = 1 } END { exit !f }' "$REPEAT_TEST_ALIASES" ||
      { echo "$show_error: Directory $d does not exist." >&2; show_missing=1; }
  done
  [ -z "$show_missing" ] || exit 0
  for d in ${show_dirs[@]+"${show_dirs[@]}"}; do
    [ "${#show_dirs[@]}" = 1 ] || echo "$d:" >&2
    { echo default; awk -v d="$d" '$1 == d && NF > 1 { print $2 }' "$REPEAT_TEST_ALIASES"; } | sort >&2
    [ "${#show_dirs[@]}" = 1 ] || echo >&2
  done
  exit 0
fi
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
# The trace options dune was handed, each observed on its own (leg 80): one
# line per `--trace-file` word to REPEAT_TEST_TRACE_ARGS when set -- its
# 1-based position, `before` or `after` dune's own `--`, and its value. Only
# the runner's own is kept out of the calls every leg pins: exactly
# `--trace-file=<the run's directory>/trace.csexp`, as the second word, right
# after the subcommand, where the launch puts it. Every other word -- a
# second copy of that one, a caller's own, a program's argument past `--` --
# stays in the calls verbatim. The synthetic trace REPEAT_TEST_TRACE names,
# when set, is copied to where the runner's own points.
pos=0 side=before
for arg in "$@"; do
  pos=$((pos + 1))
  if [ "$side" = before ] && [ "$arg" = -- ]; then side=after; continue; fi
  case $arg in
    --trace-file | --trace-file=*)
      [ -z "${REPEAT_TEST_TRACE_ARGS:-}" ] ||
        printf '%s %s %s\n' "$pos" "$side" "${arg#--trace-file=}" >>"$REPEAT_TEST_TRACE_ARGS" ;;
  esac
done
# The runner's own is told by where it points: the trace of a directory holding
# a run's metadata (`wt`, `cap`, `log`), which nothing hands dune any longer
# through the environment (gh-ocannl-1448).
own_trace() { # <word>; 0 iff it is --trace-file=<a run directory>/trace.csexp
  local d
  case $1 in --trace-file=*/trace.csexp) ;; *) return 1 ;; esac
  d=${1#--trace-file=}
  d=${d%/trace.csexp}
  [ -f "$d/wt" ] && [ -f "$d/cap" ] && [ -f "$d/log" ]
}
if [ $# -ge 2 ] && own_trace "$2"; then
  trace_to=${2#--trace-file=}
  trace_sub=$1
  shift 2
  set -- "$trace_sub" "$@"
  [ -z "${REPEAT_TEST_TRACE:-}" ] || cp "$REPEAT_TEST_TRACE" "$trace_to"
fi
# A build directory the argv names is made, as dune would (leg 81: a `fresh`
# run's own goes with the run).
if [ -n "${REPEAT_TEST_MKBUILD:-}" ]; then
  for arg in "$@"; do
    case $arg in --build-dir=*) mkdir -p "${arg#--build-dir=}/default" ;; esac
  done
fi
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
  # Leg 97: a dune that leaves a session-escaped descendant holding the
  # worktree lock (inherited fd 9, TERM ignored), says it is ready, and then
  # ignores TERM itself, so the supervisor's group KILL is the only way out.
  term_escape)
    perl -MPOSIX -e '
      $SIG{$_} = "IGNORE" for qw(TERM INT HUP);
      POSIX::setsid() >= 0 or die "setsid: $!";
      open my $pf, ">", "$ARGV[0].tmp" or die; print $pf "$$\n"; close $pf;
      rename "$ARGV[0].tmp", $ARGV[0]; sleep 120' "$REPEAT_TEST_ESCAPE_PID" </dev/null >/dev/null 2>&1 &
    trap '' TERM INT HUP
    while [ ! -s "$REPEAT_TEST_ESCAPE_PID" ]; do sleep 0.05; done
    : >"$REPEAT_TEST_ESCAPE_READY"
    while :; do sleep 1; done
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
  # A program that ran, printing a `test-run: ` line of its own and then a
  # nested refusal: output, not the launcher's prelude (Codex review round 2
  # on PR #832).
  prefixed_refusal)
    printf "test-run: a program's own line\n" >&2
    printf "dune: unknown option '--frobnicate'.\nUsage: dune build [OPTION]… [TARGET]…\n" >&2
    exit 1 ;;
  usage_command)
    printf "dune: unknown command 'frobnicate', must be one of 'build', 'clean', 'exec', 'runtest' or 'test'.\nUsage: dune COMMAND …\nTry 'dune --help' for more information.\n" >&2
    exit 1 ;;
  usage_operand)
    printf "dune: option '-j': invalid concurrency value, must be 'auto' or a positive\n      number\nUsage: dune build [OPTION]… [TARGET]…\nTry 'dune build --help' or 'dune --help' for more information.\n" >&2
    exit 1 ;;
  # gh-ocannl-1281: a test whose GPU device initialization outlived the
  # in-process deadline -- its line, at a line's start, among an ordinary red
  # run's -- and a test quoting such a line, indented, which is no death.
  wedged_init)
    printf 'File "test/operations/dune", lines 3-9, characters 0-120:\n' >&2
    printf 'OCANNL device_init_deadline: hip device initialization did not finish within 120s; the device or driver is likely wedged. Dying by SIGALRM.\n' >&2
    exit 1 ;;
  # ...and the same line followed by more output than the digest's 10MB scan
  # window: parallel actions can bury it.
  wedged_long)
    printf 'OCANNL device_init_deadline: hip device initialization did not finish within 120s; the device or driver is likely wedged. Dying by SIGALRM.\n' >&2
    perl -e 'my $l = ("y" x 99) . "\n"; print STDERR $l for 1 .. 110000'
    exit 1 ;;
  wedged_quoted)
    printf 'File "test/operations/dune", lines 3-9, characters 0-120:\n' >&2
    printf '  child | OCANNL device_init_deadline: hip device initialization did not finish within 1s\n' >&2
    exit 1 ;;
  # An ordinary red run: a located error, exit 1 -- and the same with a test's
  # own output opening on a `dune:` line, which no `Usage:` line follows.
  red)
    printf 'File "test/operations/fixture.ml", line 3, characters 4-9:\nError: This expression has type int but an expression was expected of type float\n' >&2
    exit 1 ;;
  # The same located error as dune prints it under CLICOLOR_FORCE=1.
  red_color)
    printf '\033[1mFile "test/operations/fixture.ml", line 3, characters 4-9\033[0m:\n\033[1;31mError\033[0m: This expression has type int but an expression was expected of type float\n' >&2
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
  # gh-ocannl-1055. A rule whose action failed before its diff ran: dune
  # quotes the stanza -- naming the `.corrected` file its `diff?` would read --
  # over the runner's own error, verbatim from the reported run (plus an
  # `.expected` stanza line and a test's `--- ` banner that no `+++ ` follows).
  action_no_diff)
    printf 'File "test/einsum/dune", lines 197-217, characters 0-458:\n197 | (rule\n198 |  (alias runtest-surjectivity)\n.....\n215 |     -diff-cmd\n216 |     -)\n217 |    (diff? surjectivity.ml surjectivity.ml.corrected))))\n' >&2
    printf 'File "test/operations/fixture.expected", line 1, characters 0-0:\n--- stage 2 ---\nnot a diff\n' >&2
    printf 'ppx_inline_test error: the following -only-test flags matched nothing: surjectivty.ml.\n' >&2
    exit 1 ;;
  # A golden mismatch as dune prints it (git diff), plain and colored.
  diff_hunk | diff_hunk_color)
    if [ "$REPEAT_TEST_MODE" = diff_hunk_color ]; then b=$(printf '\033[1m') r=$(printf '\033[0m')
    else b= r=; fi
    printf 'File "test/operations/fixture.expected", line 1, characters 0-0:\n' >&2
    printf '%sdiff --git a/_build/default/test/operations/fixture.expected b/_build/default/test/operations/fixture.exe.output%s\n' "$b" "$r" >&2
    printf '%s--- a/_build/default/test/operations/fixture.expected%s\n' "$b" "$r" >&2
    printf '%s+++ b/_build/default/test/operations/fixture.exe.output%s\n' "$b" "$r" >&2
    printf '@@ -1 +1 @@\n-old line\n+new line\n' >&2
    exit 1 ;;
  # The same mismatch through `--diff-command patdiff` (its doubled headers),
  # and through `--diff-command -`, which prints the location and no diff.
  patdiff_hunk)
    printf 'File "test/operations/fixture.expected", line 1, characters 0-0:\n' >&2
    printf -- '------ _build/default/test/operations/fixture.expected\n' >&2
    printf '++++++ _build/default/test/operations/fixture.exe.output\n' >&2
    printf '@|-1,1 +1,1 ====\n-|old line\n+|new line\n' >&2
    exit 1 ;;
  diff_suppressed)
    printf 'File "test/operations/fixture.expected", line 1, characters 0-0:\n' >&2
    exit 1 ;;
  # An inline-expect rule's mismatch under `--diff-command -`: the remaining
  # location names the `.ml` source, no `.expected` or `.corrected` at all.
  inline_suppressed)
    printf 'File "test/einsum/surjectivity.ml", line 12, characters 0-1:\n' >&2
    exit 1 ;;
  # A real hunk followed by more output than the digest's 10MB scan window.
  diff_then_noise)
    printf -- '--- a/_build/default/test/operations/fixture.expected\n' >&2
    printf '+++ b/_build/default/test/operations/fixture.exe.output\n' >&2
    printf '@@ -1 +1 @@\n-old line\n+new line\n' >&2
    yes 'noise from a parallel action' | head -c 10500000
    exit 1 ;;
  # A dune killed by a signal, not by its caller: the run's last phase must
  # exit with the status an exec'd dune would have had (gh-ocannl-1087).
  self_kill) kill -KILL $$ ;;
  # The per-test cap (gh-ocannl-1211). An action as dune starts one: a child
  # leading its own process group, which this dune kills on TERM as dune
  # does. It runs REPEAT_TEST_ACTION_SECS (60: a hung test) and records its
  # pid in REPEAT_TEST_ACTION_PID. The parent is perl renamed `dune` -- a
  # shebang script's process name is its interpreter's, and the watcher
  # finds tests as the children of a process named dune ($0 sets the name ps
  # reads on Linux and macOS; where it does not, leg 75 skips). With
  # REPEAT_TEST_ACTION_HOPS it runs that perl script instead, from hop 1: an
  # action that keeps its pid while exec renames it (gh-ocannl-1319).
  action)
    exec perl -e '$0 = "dune"; my ($pidf, $secs) = @ARGV;
      my $c = fork(); die "fork: $!" unless defined $c;
      if (!$c) { setpgrp(0, 0); open my $f, ">", "$pidf.tmp" or die; print $f "$$\n"; close $f;
                 rename "$pidf.tmp", $pidf;
                 my $hops = $ENV{REPEAT_TEST_ACTION_HOPS} // "";
                 my @cmd = $hops ne "" ? ($^X, $hops, 1) : ("sleep", $secs);
                 exec @cmd; exit 127 }
      $SIG{TERM} = sub { kill "KILL", -$c; exit 143 };
      waitpid($c, 0); print "action done\n"; exit 0' \
      "$REPEAT_TEST_ACTION_PID" "${REPEAT_TEST_ACTION_SECS:-60}"
    ;;
  # The names of the environment dune runs in -- and every test it starts
  # inherits -- appended to REPEAT_TEST_ENV_NAMES (leg 90, gh-ocannl-1448).
  env_names)
    env | sed -n 's/^\([A-Za-z_][A-Za-z0-9_]*\)=.*/\1/p' >>"$REPEAT_TEST_ENV_NAMES"
    printf 'stable stdout\n'; printf 'stable stderr\n' >&2 ;;
  # A dune that is itself slow, starting no action: dune is not a test.
  slow_dune)
    exec perl -e '$0 = "dune"; select undef, undef, undef, $ARGV[0]; print "slow dune done\n"' \
      "${REPEAT_TEST_ACTION_SECS:-3}"
    ;;
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
# A shared fixture that is not what the legs assume is a FAIL here, outside any
# leg, and the run goes on (gh-ocannl-1374): the legs reading it are named in
# the label, and whatever they conclude, the run is red.
if await_fixture_ready "$TMP/repeat-never-ready" 1; then
  report 1 "fixture: await_fixture_ready accepted an absent marker (every leg awaiting a fixture reads it)"
fi
# The three width-cap probes read what the leg exported, which is the
# harness-wide absent defaults (hermetic_probes) unless the leg opted in. A leg
# that sets the array fixture_launcher runs the script under it (leg 85: a
# parent that reaps no orphans); empty, the script runs directly.
fixture_launcher=()
fixture_probe() { # tag mode runs subcommand [argv...]
  local tag=$1 mode=$2 runs=$3
  shift 3
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
    ${fixture_launcher[@]+"${fixture_launcher[@]}"} \
    "$repeat_root/tools/test-run.sh" "$@" >"$TMP/$tag.out" 2>"$TMP/$tag.err"
  fixture_rc=$?
  fixture_out=$(cat "$TMP/$tag.out")
  fixture_err=$(cat "$TMP/$tag.err")
  fixture_calls=$(cat "$TMP/$tag.calls")
  fixture_dir=$(OCANNL_TOOL_TEST_RUNS="$runs" "$repeat_root/tools/test-run.sh" paths run last 2>/dev/null)
}
repeat_probe() {
  local tag=$1 mode=$2
  shift 2
  fixture_probe "$tag" "$mode" "$TMP/repeat-runs-$tag" repeat "$@"
  repeat_rc=$fixture_rc repeat_out=$fixture_out repeat_dir=$fixture_dir
}
# A background set whose fixture never became ready ends its leg (gh-ocannl-1374), released and
# cancelled first: its lock and its group must not reach the legs that run after it.
repeat_abandon() { # wait prefix of the background set in $repeat_pid
  touch "$1.release"
  [ -n "$repeat_pid" ] || return 0
  kill -TERM -- "-$repeat_pid" 2>/dev/null || kill -TERM "$repeat_pid" 2>/dev/null
  wait "$repeat_pid" 2>/dev/null
  repeat_pid=
}
}

shared_argv_probe() {
# ---------------------------------------------------------------------------
# Legs 26-28: this script's own options are refused on the wrong side of dune
# ---------------------------------------------------------------------------
# `run build @alias --cap 900` used to forward `--cap 900` to dune, which exits
# 1 on the unknown option having built nothing; the digest then reads `FAIL
# (exit 1)` with no matching error lines -- a verdict indistinguishable from a
# failing test, and a session acting on it debugs code that never ran. So the
# refusal is asserted together with the CALLS file being empty: "runs nothing"
# is the half that makes the exit code trustworthy.
argv_probe() { # tag subcommand [argv...]
  local tag=$1 runs=${argv_runs:-$TMP/argv-runs-$1}
  shift
  fixture_probe "$tag" "${argv_mode:-stable}" "$runs" "$@"
  argv_rc=$fixture_rc argv_out=$fixture_out argv_err=$fixture_err
  argv_calls=$fixture_calls argv_dir=$fixture_dir
}
argv_rc= argv_out= argv_err= argv_calls= argv_dir= argv_mode= argv_runs=
}

shared_dxg() {
# ---------------------------------------------------------------------------
# Legs 29-34: the dxg width cap (gh-ocannl-983)
# ---------------------------------------------------------------------------
# A box that reaches its GPU through WSL2's `/dev/dxg` bridge overflows the
# bridge's VM-bus ring at dune's default width, and the runtime reports the lost
# messages as device/binary/stream-creation refusals: a red suite in the very
# stanzas a real backend regression lands in. tools/sweep.sh capped its unit
# there; a manual run did not, and an hour of box time went into rediscovering
# the cap by bisection. So `run`/`start` injects it -- but ONLY where the caller
# expressed no width at all, which is what keeps "runs dune as given" true.
#
# Every leg drives the real launch path against the fixture dune and reads the
# recorded argv, so what is pinned is the command dune was actually given, not a
# decision variable. The bridge is faked through OCANNL_TOOL_DXG_DEVICE (the
# device path the table probes), which is the only reason these run on a Mac.
dxg_present=$TMP/fake-dxg-device
dxg_absent=$TMP/fake-dxg-device-absent
: >"$dxg_present"
rm -f "$dxg_absent"
# The fixture is the whole experiment: a present path that is absent, or an
# absent one that exists, would make every leg below agree with itself.
if [ ! -e "$dxg_present" ] || [ -e "$dxg_absent" ]; then
  report 1 "fixture: the dxg device paths are not what the legs assume (every leg faking a dxg bridge reads them)"
fi
dxg_probe() { # tag device backend subcommand [argv...]
  local tag=$1 device=$2 backend=$3
  shift 3
  # Exported rather than written as a prefix: bash's rules for assignments
  # before a FUNCTION call differ between modes, and a leg that silently ran
  # without its backend would pass by testing the no-op case.
  export OCANNL_TOOL_DXG_DEVICE=$device
  if [ -n "$backend" ]; then export OCANNL_BACKEND=$backend; else unset OCANNL_BACKEND; fi
  argv_probe "$tag" "$@"
  unset OCANNL_BACKEND
  hermetic_probes
}
# The sourced lab-map.sh and the lab's endpoint map as a fixture, which this
# section's last leg and leg 48 resolve sweep destinations through (see there).
LAB_MAP_SRC=$(dirname "$SWEEP_SRC")/lab-map.sh
LAB_MAP='rog rog-nv-linux rog-nv-win rog-nv-wsl rog-lan
minix minix-amd-linux minix-amd-win minix-amd-wsl minix-lan
tuf tuf-amd-linux tuf-amd-win tuf-amd-wsl'
}

shared_native() {
# ---------------------------------------------------------------------------
# Legs 48-52: the native GPU width caps (gh-ocannl-1033)
# ---------------------------------------------------------------------------
# The fleet runs several correctness batches at once on each native GPU box
# (lukstafi/ludics-lite#316 and ludics-lite#344), measured at a width per batch: -j 4 on
# minix, whose device-wide SDMA pool default-width hip batches can drain
# between them (gh-ocannl-1029), -j 8 on tuf and -j 8 on rog-nv. The widths used to hold only while
# every worker remembered them; now `run`/`start` injects them the way it
# injects the dxg cap. The pool comes from the KFD topology, faked here through
# OCANNL_TOOL_KFD_TOPOLOGY as directories of `properties` files copied from
# what the boxes report (only the keys the probe reads, plus a neighbour
# either side, so a parser keyed on line position would miss); the NVIDIA boot
# through OCANNL_TOOL_NVIDIA_DEVICE, the way the dxg legs fake the bridge.
kfd_node() { # dir simd engines queues-per-engine
  mkdir -p "$1"
  printf 'cpu_cores_count 0\nsimd_count %s\nhive_id 0\nnum_sdma_engines %s\nnum_sdma_xgmi_engines 0\nnum_sdma_queues_per_engine %s\nnum_cp_queues 8\n' \
    "$2" "$3" "$4" >"$1/properties"
}
kfd_cpu_node() { # dir
  mkdir -p "$1"
  printf 'cpu_cores_count 32\nsimd_count 0\nnum_sdma_engines 0\nnum_sdma_queues_per_engine 0\nnum_cp_queues 0\n' >"$1/properties"
}
kfd_small=$TMP/fake-kfd-minix   # gfx1151: 1 engine x 6, as minix-amd-linux reports it
kfd_large=$TMP/fake-kfd-tuf     # gfx1102: 2 engines x 6, as tuf-amd-linux reports it
kfd_nopool=$TMP/fake-kfd-nopool # a GPU node that reports no SDMA engine
kfd_absent=$TMP/fake-kfd-absent
kfd_cpu_node "$kfd_small/0"; kfd_node "$kfd_small/1" 80 1 6
kfd_cpu_node "$kfd_large/0"; kfd_node "$kfd_large/1" 64 2 6
kfd_cpu_node "$kfd_nopool/0"; kfd_node "$kfd_nopool/1" 80 0 6
rm -rf "$kfd_absent"
nv_present=$TMP/fake-nvidiactl
nv_absent=$TMP/fake-nvidiactl-absent
# The fleet's name for this box (lukstafi/ludics-lite's FLEET_LOCAL_BOX): rog,
# or another fleet box -- an NVIDIA machine that is not the fleet's rog. Every
# native probe pins one, so the harness never reads the name of the box it
# runs on; an empty one leaves FLEET_LOCAL_BOX unset, for the hostname legs,
# which pin FLEET_HOSTNAME_MAP (native_map) instead.
fleet_rog=rog-nv-linux
fleet_none=mac-studio
native_fleet=$fleet_none
native_map=
: >"$nv_present"
rm -f "$nv_absent"
if [ ! -f "$kfd_small/1/properties" ] || [ -e "$kfd_absent" ] ||
   [ ! -e "$nv_present" ] || [ -e "$nv_absent" ]; then
  report 1 "fixture: the native-cap paths are not what the legs assume (every leg faking a native device reads them)"
fi
# The pool the fixtures must read as, from the shipping probe itself: a fixture
# the parser misread would leave legs 48 and 49 agreeing with a wrong number.
native_pools=$(
  . "$JOBS_SRC"
  for t in "$kfd_small" "$kfd_large" "$kfd_nopool" "$kfd_absent"; do
    printf '%s,' "$(OCANNL_TOOL_KFD_TOPOLOGY=$t box_jobs_sdma_pool)"
  done
)
if [ "$native_pools" != "6,12,,," ]; then
  report 1 "fixture: the KFD fixtures read as pools '$native_pools', not 6,12,,, -- fix the fixture or the probe (every leg faking a native device reads them)"
fi
native_probe() { # tag dxg kfd nvidia backend subcommand [argv...]
  local tag=$1 dxg=$2 kfd=$3 nv=$4 backend=$5
  shift 5
  export OCANNL_TOOL_DXG_DEVICE=$dxg OCANNL_TOOL_KFD_TOPOLOGY=$kfd OCANNL_TOOL_NVIDIA_DEVICE=$nv
  if [ -n "$native_fleet" ]; then export FLEET_LOCAL_BOX=$native_fleet; else unset FLEET_LOCAL_BOX; fi
  if [ -n "$native_map" ]; then export FLEET_HOSTNAME_MAP=$native_map; else unset FLEET_HOSTNAME_MAP; fi
  if [ -n "$backend" ]; then export OCANNL_BACKEND=$backend; else unset OCANNL_BACKEND; fi
  argv_probe "$tag" "$@"
  unset OCANNL_BACKEND FLEET_LOCAL_BOX FLEET_HOSTNAME_MAP
  hermetic_probes
}
}

shared_synthetic_trace() {
# A synthetic dune trace (python3), for the legs that need a run's own trace to
# say which processes ran and how they ended. Each ROW is
# `PROG|EXIT|TARGET[,TARGET...]`, targets relative to BUILD/default; EXIT
# `open` leaves the process unfinished.
synthetic_trace() { # <path> <build dir> ROW...
  python3 - "$@" <<'PY'
import sys
path, build, rows = sys.argv[1], sys.argv[2], sys.argv[3:]
S, T0 = 1_000_000_000, 1_791_291_000 * 1_000_000_000
def atom(s):
    s = str(s).encode()
    return str(len(s)).encode() + b':' + s
def sexp(x):
    return b'(' + b''.join(sexp(e) if isinstance(e, list) else atom(e) for e in x) + b')'
out = [sexp(['config', 'init', str(T0), ['build_dir', build]])]
for i, row in enumerate(rows):
    prog, ex, targets = row.split('|')
    t = T0 + (i + 1) * S
    f = [['pid', str(100 + i)], ['prog', '/opt/bin/' + prog], ['dir', build + '/default'],
         ['target_files', [build + '/default/' + x for x in targets.split(',')]]]
    out.append(sexp(['process', 'start', str(t)] + f))
    if ex != 'open':
        out.append(sexp(['process', 'finish', [str(t), str(S // 10)]] + f + [['exit', ex]]))
open(path, 'wb').write(b''.join(out))
PY
}
}

shared_slot() {
# ---------------------------------------------------------------------------
# Legs 53-58: the fleet's run-time slot, taken by the runner (gh-ocannl-1004)
# ---------------------------------------------------------------------------
# A fake fleet-worker.sh stands in for lukstafi/ludics-lite's: it answers the
# probe (or, in `noprobe` mode, refuses it as a version without nested slots
# would), records the slot call, and then refuses, cannot reach its registry,
# or holds the slot and execs the command as the real one does. The fake
# readers every leg runs with (batch_reader, batch_reach) answer the batch's
# backends, so the kind decision is driven without a build.
slot_fake=$TMP/fake-fleet-worker.sh
cat >"$slot_fake" <<'FAKE'
#!/usr/bin/env bash
# The runner's own trace option (leg 80) is kept out of the slot calls these
# legs pin, by the fixture dune's rule: exactly `--trace-file=<the run's
# directory>/trace.csexp` (a directory holding a run's `wt`, `cap` and `log`),
# once, right after the subcommand that follows the dune it runs. Every other
# word stays verbatim.
own_trace() { # <word>; 0 iff it is --trace-file=<a run directory>/trace.csexp
  local d
  case $1 in --trace-file=*/trace.csexp) ;; *) return 1 ;; esac
  d=${1#--trace-file=}
  d=${d%/trace.csexp}
  [ -f "$d/wt" ] && [ -f "$d/cap" ] && [ -f "$d/log" ]
}
shown=() prev2= prev1= dropped=
for arg in "$@"; do
  if [ -z "$dropped" ] && own_trace "$arg" && [ "${prev2##*/}" = dune ]; then
    case $prev1 in build | runtest | test | exec) dropped=1; prev2=$prev1 prev1=$arg; continue ;; esac
  fi
  shown+=("$arg")
  prev2=$prev1 prev1=$arg
done
printf '%s\n' "${shown[*]}" >>"$FAKE_FW_CALLS"
if [ "$1 $2 $3" = "execution slot --probe" ]; then
  [ "${FAKE_FW_MODE:-}" = noprobe ] && { echo "fleet-worker.sh: execution slot [--wait <seconds>] ..." >&2; exit 2; }
  # Inside a live measurement hold the probe's line carries the hold's id.
  if [ "${FAKE_FW_MODE:-}" = measurement ]; then echo "EXECUTION SLOT PROBE fakebox 4 2 measurement M-7"
  else echo "EXECUTION SLOT PROBE fakebox 4 2"; fi
  exit 0
fi
shift 2
while [ "$1" != -- ]; do shift; done
shift
case ${FAKE_FW_MODE:-} in
  refuse) echo "EXECUTION SLOT REFUSED fakebox: all 2 GPU tokens (slots 1-2 of 4) busy after 0s; a batch that holds no GPU declares --cpu"; exit 1 ;;
  unreachable) echo "EXECUTION SLOT UNREACHABLE anchor: registry unread, no slot taken"; exit 4 ;;
  # The batch runs under the caller's measurement hold (lukstafi/ludics-lite#480),
  # and something inside it -- a nested runner, say -- logs a refusal of its own.
  measurement)
    echo "EXECUTION SLOT fakebox: inside measurement M-7, running under the hold: $*" >&2
    echo "EXECUTION SLOT REFUSED fakebox: a nested batch's own refusal"
    exec "$@" ;;
  # The real slot's sleep inhibitor: a process of its own group, beside the
  # command for the whole batch, and no child of dune (gh-ocannl-1211).
  holder)
    echo "EXECUTION SLOT fakebox: slot 1 of 4, GPU token 1 of 2 held for: $*" >&2
    perl -e 'setpgrp(0, 0); sleep 30' 9>&- &
    holder=$!
    "$@"
    rc=$?
    kill -KILL -- "-$holder" 2>/dev/null
    exit "$rc" ;;
esac
echo "EXECUTION SLOT fakebox: slot 1 of 4, GPU token 1 of 2 held for: $*" >&2
exec "$@"
FAKE
chmod +x "$slot_fake"
slot_probe() { # tag mode test-backend arrayjit-backend subcommand [argv...]
  local tag=$1 mode=$2 bt=$3 ba=$4
  shift 4
  : >"$TMP/$tag.fw"
  : >"$TMP/$tag.reach"
  FAKE_FW_CALLS=$TMP/$tag.fw FAKE_FW_MODE=$mode FAKE_BACKEND_TEST=$bt FAKE_BACKEND_ARRAYJIT=$ba \
  FAKE_REACH_CALLS=$TMP/$tag.reach \
  OCANNL_TOOL_FLEET_WORKER=$slot_fake argv_probe "$tag" "$@"
  slot_calls=$(grep -v -- '--probe' "$TMP/$tag.fw")
}
slot_calls=
}

shared_device_probe() {
# Leg 76: the device probe, against a stand-in (OCANNL_TOOL_DEVICE_PROBE) that
# answers, fails fast, or never answers, per backend. A GPU batch's probe runs
# inside the slot, ahead of dune; a query that does not answer within the
# bound refuses the batch as DEVICE UNHEALTHY (69) with dune never started and
# the query killed, while one that fails fast is recorded and passed over. An
# unread batch probes every GPU backend; a CPU batch probes nothing and keeps
# the slot call the other legs pin. The real probe is bin/device_props. A hang
# is judged only in a run the stand-in started in (probe_hang_run, below).
probe_fake=$TMP/fake-device-probe.sh
cat >"$probe_fake" <<'FAKE'
#!/usr/bin/env bash
b=${1#--ocannl_backend=}
# A stand-in the bound KILLs before it records anything, as a loaded box can make one
# (gh-ocannl-1366): once, where the leg left the marker.
if [ -e "$FAKE_PROBE_CALLS.unstarted" ]; then rm -f "$FAKE_PROBE_CALLS.unstarted"; exec sleep 30; fi
printf '%s %s\n' "$PWD" "$*" >>"$FAKE_PROBE_CALLS"
# A stand-in that forks its query rather than exec'ing it leaves the query behind its own exit
# unless the probe's bound reaches the whole tree (FAKE_PROBE_FORK; gh-ocannl-1252).
if [ -n "${FAKE_PROBE_FORK:-}" ]; then
  case " ${FAKE_PROBE_HANG:-} " in
    *" $b "*) sleep 30 & printf '%s\n' "$!" >"$FAKE_PROBE_CALLS.child"; wait; exit 0 ;;
  esac
fi
case " ${FAKE_PROBE_HANG:-} " in *" $b "*) printf '%s\n' "$$" >"$FAKE_PROBE_CALLS.pid"; exec sleep 30 ;; esac
case " ${FAKE_PROBE_ABSENT:-} " in *" $b "*) echo "no $b here" >&2; exit 2 ;; esac
echo "static.device[0].device_name = fake $b"
FAKE
chmod +x "$probe_fake"
# One bound, two questions: whether the query answered, and -- for a stand-in that never answers
# -- whether it had started, recording its call and pid, before the bound KILLed it. A loaded box
# can delay the stand-in's start past a second (gh-ocannl-1366), so the two are kept apart. A query
# that answers runs under probe_answer_cap, which it never comes near and so never waits out. A
# hang is judged by probe_hang_run, whose bound starts at 1s and is tripled, up to
# probe_hang_cap_max, for as long as the stand-in has not recorded its start: a run it never
# started in proves nothing about the query the leg asked for, so it is repeated rather than
# judged, with a note (none where the leg withholds the start itself: probe_hang_late). probe_cap is
# the bound the last run had, which the checks quote.
probe_answer_cap=30 probe_hang_cap_max=27 probe_cap=
probe_run() { # tag test-backend arrayjit-backend hang absent subcommand [argv...]
  local tag=$1 bt=$2 ba=$3 hang=$4 absent=$5
  shift 5
  probe_cap=${probe_run_cap:-$probe_answer_cap}
  : >"$TMP/$tag.probe"
  rm -f "$TMP/$tag.probe.pid" "$TMP/$tag.probe.child"
  FAKE_PROBE_CALLS=$TMP/$tag.probe FAKE_PROBE_HANG=$hang FAKE_PROBE_ABSENT=$absent \
  OCANNL_TOOL_DEVICE_PROBE=$probe_fake OCANNL_TOOL_DEVICE_PROBE_CAP=$probe_cap \
    slot_probe "$tag" hold "$bt" "$ba" "$@"
  probe_calls=$(cat "$TMP/$tag.probe")
}
probe_hang_run() { # start-evidence (pid|child) tag test-backend arrayjit-backend hang absent subcommand [argv...]
  local evidence=$1 cap=1
  shift
  while :; do
    probe_hang_start=$SECONDS
    probe_run_cap=$cap probe_run "$@"
    probe_hang_took=$((SECONDS - probe_hang_start))
    [ ! -s "$TMP/$1.probe.$evidence" ] || return 0
    [ "$cap" -lt "$probe_hang_cap_max" ] || return 0
    [ -n "${probe_hang_late:-}" ] ||
      printf '      note: %s: the stand-in had not started within %ss; judging a run bounded at %ss\n' \
        "$1" "$cap" "$((cap * 3))"
    cap=$((cap * 3))
  done
}
probe_calls= probe_hang_start= probe_hang_took=
}

shared_renamed_dune() {
# Leg 83: `repeat` has both bounds (gh-ocannl-1252). A repeat samples timing-dependent goldens,
# often GPU tests, and ran with neither: a wedged device held it for N times the iteration cap.
# Now a test past the per-test cap ends its iteration as TIMEOUT (142) naming the test, the test
# killed, and ENDS the set, since every later iteration would hang the same way; one inside it does
# not. The default follows `run`'s decision from the same resolution of the batch (a GPU batch:
# 1500s, announced in the log; a CPU batch: none), and an explicit cap applies to any batch. A GPU
# batch's device probe runs before each iteration, inside it: a query that does not answer ends
# the set as DEVICE UNHEALTHY (69) with dune never started -- not even its clean -- and an answer
# lets every iteration run, the device asked once each.
# Whether the fixture's renamed dune reads back as `dune` in ps, as leg 75 asks
# it: asked once, by the first of the later legs to run that needs it.
renamed_dune_readable() { # sets test_cap_named_ok=1 where it does
  local named
  [ -z "${test_cap_named_ok:-}" ] || return 0
  perl -e '$0 = "dune"; sleep 10' </dev/null >/dev/null 2>&1 &
  named=$!
  sleep 0.3
  ps -A -o pid= -o comm= 2>/dev/null | grep -Eq "^ *$named +(.*/)?dune$" && test_cap_named_ok=1
  kill "$named" 2>/dev/null
  wait "$named" 2>/dev/null
  return 0
}
}


# ---------------------------------------------------------------------------
# Leg 70: a leg that names no device still reads the fake (gh-ocannl-1108)
# ---------------------------------------------------------------------------
# Every leg above that faked a device went back through hermetic_probes. One
# that `unset` a probe instead would hand every later leg the box's real
# /dev/dxg, KFD topology or /dev/nvidiactl -- a leak no Mac or CI runner can
# show, having none of them, and on minix a lifecycle run took -j 4. So
# this leg runs last and names no probe. The shipping readers, sourced fresh,
# must name the absent defaults (the path, not merely its absence, so the
# check means the same on a box without the device) and answer "no device";
# and a GPU run through the real launch path must be neither capped nor told.
# The controls: the path check refuses a probe unset in a subshell, which is
# the leak; and an exported fixture pool caps the same hip run, so a probe
# that never reached the width code cannot pass the uncapped half.
# Outside the body: leg 86 reads it, to tell that this leg reported.
herm_label="a leg that names no device probe reads the harness's absent defaults: no device, no cap, no word"
leg_70() {
herm_detail=
herm_view() { # the shipping readers' answers: paths, then dxg:nvidia:pool
  (
    . "$JOBS_SRC"
    printf '%s|%s|%s|' "$(box_jobs_dxg_device)" "$(box_jobs_kfd_topology)" "$(box_jobs_nvidia_device)"
    box_jobs_dxg_host && printf 'dxg' || printf 'no-dxg'
    box_jobs_native_nvidia_host && printf ':nvidia' || printf ':no-nvidia'
    printf ':pool=%s\n' "$(box_jobs_sdma_pool)"
  )
}
herm_want="$probe_dxg_default|$probe_kfd_default|$probe_nvidia_default|no-dxg:no-nvidia:pool="
for p in "$probe_dxg_default" "$probe_kfd_default" "$probe_nvidia_default"; do
  [ ! -e "$p" ] || { herm_detail="the absent default $p exists"; break; }
done
if [ -z "$herm_detail" ]; then
  herm_seen=$(herm_view)
  [ "$herm_seen" = "$herm_want" ] ||
    herm_detail="the shipping readers answer '$herm_seen', not '$herm_want'"
fi
for backend in hip cuda; do
  [ -z "$herm_detail" ] || break
  export OCANNL_BACKEND=$backend
  argv_probe "herm-$backend" run build @cheap
  unset OCANNL_BACKEND
  if [ "$argv_rc" != 0 ] || [ "$argv_calls" != "build @cheap" ] ||
     grep -qi 'cap\|sdma\|nvidia\|dxg' <<<"$argv_err"; then
    herm_detail="$backend: exit $argv_rc; calls: ${argv_calls:-<none>}; stderr: ${argv_err:-<none>}"
  fi
done
# Control: the leak itself -- a probe unset -- is what the path check refuses.
if [ -z "$herm_detail" ]; then
  herm_leak=$(unset OCANNL_TOOL_KFD_TOPOLOGY; herm_view)
  [ "$herm_leak" != "$herm_want" ] ||
    herm_detail="control: with the KFD probe unset the readers still answer '$herm_leak', so the path check cannot see a leak"
fi
# Control: the probe the uncapped half relied on does reach the width -- a
# fixture pool exported the way a leg opts in caps the same hip run.
if [ -z "$herm_detail" ]; then
  export OCANNL_TOOL_KFD_TOPOLOGY=$kfd_small OCANNL_BACKEND=hip
  argv_probe herm-optin run build @cheap
  unset OCANNL_BACKEND
  hermetic_probes
  herm_cap=$(. "$JOBS_SRC"; printf '%s' "$BOX_JOBS_SDMA_SLOT_CAP")
  [ -n "$herm_cap" ] && [ "$argv_calls" = "build -j $herm_cap @cheap" ] ||
    herm_detail="control: hip on an exported small pool was not capped at -j ${herm_cap:-?}: calls: ${argv_calls:-<none>}; stderr: ${argv_err:-<none>}"
fi
if [ -z "$herm_detail" ]; then
  report 0 "$herm_label"
else
  report 1 "$herm_label" "$herm_detail"
fi
}

# The stages, in the order the legs were written against (the header): a
# name here is `shared_NAME` above.
shared_stages='stop repeat_root fixture_dune argv_probe dxg native synthetic_trace slot device_probe renamed_dune'
shared_done=0
shared_through() { # NAME: run every stage not yet run, up to and including NAME
  local stage at=0 i=0
  for stage in $shared_stages; do
    i=$((i + 1))
    [ "$stage" != "$1" ] || at=$i
  done
  [ "$at" != 0 ] || { echo "${0##*/}: no shared stage '$1' (tools/test-test-run-common.sh)" >&2; exit 2; }
  i=0
  for stage in $shared_stages; do
    i=$((i + 1))
    [ "$i" -gt "$shared_done" ] && [ "$i" -le "$at" ] || continue
    "shared_$stage"
    shared_done=$i
  done
}
# A member's last lines: leg 70, which names no device and so runs after every
# leg that faked one, then the footer.
test_run_close() {
  shared_through native
  harness_leg_run '70: a leg that names no device reads the absent defaults #hermetic' leg_70
  [ -z "$subject" ] || echo "the subject of this run was $subject, not $HERE/test-run.sh"
  finish
}
