#!/usr/bin/env bash
# `tools/test-run.sh bisect` (gh-ocannl-1225): find the commit that turned a
# set of targets red, one `git bisect run` step per commit, each step two runs
# of tools/test-run.sh `run`. Sourced by tools/test-run.sh from the repository
# root, after the helpers it reads (batch_known_backend, wt_key_of,
# probe_locks); never executed.
#
#   tools/test-run.sh bisect [--cap N] [--test-cap N] [--build TARGET]...
#                            [--no-first-parent] GOOD BAD TARGET...
#
# Finding gh-ocannl-1222's cause took a hand-rolled first-parent bisect on tuf, and
# three parts of its step script were easy to get wrong. Each is encoded here
# once:
#
#   - The exit mapping. `run` exits 1 for a failing test AND for a build that
#     fails where its trace shows no failed compile step, and a bisect must
#     count the second as untestable (git's skip, 125), never as bad. So each
#     step first runs `build @check` and every `--build TARGET` (name the
#     tests' executables, so a link failure skips too); only a green build
#     goes on to `build TARGET...`, and a test run that is BUILD FAILED (65,
#     a compile step of its own failed; gh-ocannl-1547) skips as well. And a verdict
#     that is not the suite's -- SLOT REFUSED (75), DEVICE UNHEALTHY (69), a
#     cap's TIMEOUT (142), INVOCATION REFUSED (2), a cancellation, a setup
#     error, a run whose recorded verdict or commit is not the one its
#     launcher reports -- ABORTS the bisect: the step exits 128, which `git
#     bisect run` takes as "stop", where any of them read as bad would
#     silently steer it. Only a run whose recorded exit is 0 or 1, the
#     launcher's own status equal to it (or 65 over a recorded 1: BUILD
#     FAILED), and whose recorded `head` is the commit under test is a
#     verdict. A TIMEOUT aborts too: a hang cannot be told from a
#     wedged device by the step, and the device probe catches the latter only
#     at the next step's start.
#   - The backend. OCANNL_BACKEND is required, and passed explicitly to every
#     run of every step, so a configuration a commit changes cannot move the
#     bisect to another backend half-way.
#   - Isolation. The bisect works in a git worktree of its own,
#     `<runs>/bisect-<stamp>-<pid>/tree` (detached; `dune-workspace` written,
#     as scripts/setup-ocaml-env.sh does, so dune roots there), never in the
#     caller's checkout, the sweep's or another session's; its runs take that
#     tree's own lock, and a step hands git its verdict -- after which git
#     checks the next commit out -- only once nothing those runs started
#     holds that lock (bounded; past it, the bisect aborts). Every step is
#     judged by ONE runner: a copy of this
#     script and the helpers it sources, frozen beside the tree when the
#     bisect starts (`runner/`), pointed at the tree through
#     OCANNL_TOOL_TESTRUN_TREE -- never the tested commit's own copy, which
#     may predate the slot, the caps or a fix to either, nor the caller's,
#     which a session may edit while the bisect runs. Per-step records stay
#     in the bisect directory: `steps.tsv` (label, commit, verdict, the build
#     and test runs), each phase's digest and log under `steps/`, and git's
#     `bisect-log`. The tree is removed at the end; the records are not.
#
# The slot, the width cap, the per-test cap and the device probe are the
# runner's own: each phase is a plain `run`, so a step takes the fleet slot,
# and is capped, exactly as a hand-launched run would be. `--cap` and
# `--test-cap` are passed to every phase's run (a cap per phase, not per
# bisect). First-parent by default (`git bisect start --first-parent`): on a
# merge-commit master each step is a merged state CI saw; `--no-first-parent`
# walks into the merged branches; under it, GOOD must be on BAD's first-parent
# chain, or git would test commits that do not descend from it (refused, exit
# 2). Before bisecting, both ends are run: BAD
# must fail and GOOD must pass HERE, or there is no regression in the range
# on this box to find (exit 1, nothing bisected).
#
# Exit codes: 0 the first bad commit was found (`result` names it); 1 the
# range does not bisect -- an end does not hold here, or only skipped commits
# are left between the last good and the first bad (`result` lists them);
# 2 usage or setup refused, before any step; 129/130/143 cancelled. An
# aborted bisect exits with the aborting run's status (75 SLOT REFUSED, 69,
# 142, 2, ...), or 126 when the run gave none of its own (it died without a
# verdict, or its record disagrees with its launcher). Retry an abort; read a
# 1.
#
# It runs in the foreground for as long as the bisect takes -- hours, on a
# long range -- so launch it as `run` is launched for a long suite, through
# the harness's background mode (bg-run.sh on the fleet); a TERM to it stops
# the current step's run, as `stop` would, and removes the tree.

bisect_usage='bisect [--cap N] [--test-cap N] [--build TARGET]... [--no-first-parent] GOOD BAD TARGET...'

bisect_lines() { # <file> -> the array bisect_read_out, one element per line
  local line
  bisect_read_out=()
  while IFS= read -r line; do bisect_read_out+=("$line"); done <"$1"
}

# A child in a process group of its own, waited for through signals: a
# trapped signal returns `wait` early, and the cancellation it set is acted on
# (bisect_cancel) before waiting again. Sets bisect_rc; the group is
# bisect_pgid until it is gone (bisect_drain). A cancellation already flagged
# starts nothing; one that landed while the child was being launched, before
# its pgid was known here (so bisect_cancel had nothing to signal), is
# delivered as soon as it is -- to the group, or to the child itself if it
# has not made the group yet (it is still this shell's unreaped child, so
# its pid names nothing else).
bisect_pgid= bisect_rc= bisect_cancelled= bisect_pending=
bisect_cancel_rc() { case $bisect_cancelled in INT) echo 130 ;; HUP) echo 129 ;; *) echo 143 ;; esac; }
bisect_spawn() { # command...
  bisect_pgid=
  if [ -n "$bisect_cancelled" ]; then
    bisect_rc=$(bisect_cancel_rc)
    return 0
  fi
  perl -e 'setpgrp(0, 0); exec @ARGV or exit 127' "$@" &
  bisect_pgid=$!
  if [ -n "$bisect_cancelled" ]; then
    kill -TERM -- "-$bisect_pgid" 2>/dev/null || kill -TERM "$bisect_pgid" 2>/dev/null
  fi
  wait "$bisect_pgid"
  bisect_rc=$?
  while [ -n "$bisect_pending" ]; do
    bisect_pending=
    wait "$bisect_pgid"
    bisect_rc=$?
  done
}
bisect_cancel() { # <signal name>
  bisect_cancelled=$1 bisect_pending=1
  [ -z "$bisect_pgid" ] || kill -TERM -- "-$bisect_pgid" 2>/dev/null
}
# Once the child is reaped: the rest of its group (a step's runner launcher and
# supervisor, which reap their dune on TERM) and the tree's lock, which dune's
# leftovers hold for as long as they can touch `_build`. Bounded; 1 when either
# outlived the bound, and the tree is then left in place.
bisect_drain() {
  local i=0 lock
  lock=$RUNS/lock-$(wt_key_of "$bisect_tree")
  while :; do
    # group_alive, not a bare signal probe: a group holding only zombies (an
    # init that is slow to reap them) is gone, not still draining.
    if ! { [ -n "$bisect_pgid" ] && group_alive "$bisect_pgid"; }; then
      probe_locks "$lock"
      [ $? = 3 ] || { bisect_pgid=; return 0; }
    fi
    i=$((i + 1))
    [ "$i" -le 600 ] || return 1
    sleep 0.1
  done
}

bisect_main() {
  local cap= test_cap= first_parent=1 builds=() good bad good_sha bad_sha backend arg
  local dir tree lang anc outcome result rc first line
  while [ $# -gt 0 ]; do
    case $1 in
      --cap) [ $# -ge 2 ] || die "--cap requires a value"; cap=$2; shift 2 ;;
      --test-cap) [ $# -ge 2 ] || die "--test-cap requires a value"; test_cap=$2; shift 2 ;;
      --build) [ $# -ge 2 ] || die "--build requires a target"; builds+=("$2"); shift 2 ;;
      --no-first-parent) first_parent=; shift ;;
      --) shift; break ;;
      -*) die "bisect: unknown option $1
  usage: tools/test-run.sh $bisect_usage" ;;
      *) break ;;
    esac
  done
  [ -z "$cap" ] || normalize_cap
  [ -z "$test_cap" ] || normalize_cap test_cap --test-cap
  [ $# -ge 3 ] || die "usage: tools/test-run.sh $bisect_usage"
  good=$1 bad=$2
  shift 2
  # The targets are dune's argv after `build`: this script's options among
  # them are misplaced, as for `run` (and so are the bisect's own).
  reject_misplaced_options "$@"
  for arg in "$@"; do
    case $arg in
      --) break ;;
      --build | --build=* | --no-first-parent)
        die "$arg belongs before GOOD and BAD:
  tools/test-run.sh $bisect_usage" ;;
    esac
  done
  for arg in "$@" ${builds[@]+"${builds[@]}"}; do
    case $arg in '' | *$'\n'* | *$'\r'*) die "bisect: a target must be one non-empty line" ;; esac
  done
  backend=${OCANNL_BACKEND:-}
  [ -n "$backend" ] || die "bisect passes OCANNL_BACKEND to every step explicitly, and none is set:
  name the backend under test (OCANNL_BACKEND=cc for a CPU bisect)"
  batch_known_backend "$backend" || die "bisect: OCANNL_BACKEND=$backend names no backend"
  command -v git >/dev/null 2>&1 || die "bisect needs git"
  # A hook's GIT_DIR/GIT_WORK_TREE must not redirect the bisect's git, here
  # or in the steps it runs (record_checkout does the same for one run).
  # shellcheck disable=SC2046 # a list of variable names, split on purpose
  unset $(git rev-parse --local-env-vars 2>/dev/null)
  good_sha=$(git rev-parse --verify --quiet "$good^{commit}") || die "bisect: GOOD $good is not a commit here"
  bad_sha=$(git rev-parse --verify --quiet "$bad^{commit}") || die "bisect: BAD $bad is not a commit here"
  [ "$good_sha" != "$bad_sha" ] || die "bisect: GOOD and BAD are one commit ($good_sha)"
  git merge-base --is-ancestor "$good_sha" "$bad_sha" ||
    die "bisect: GOOD $good is not an ancestor of BAD $bad"
  # A first-parent bisect walks BAD's first parents only: a GOOD reached
  # through a merge's second parent bounds nothing on that chain, and git
  # would test (and could name) commits that are not GOOD's descendants.
  if [ -n "$first_parent" ] &&
     ! git rev-list --first-parent "$bad_sha" | grep -Fxq "$good_sha"; then
    die "bisect: GOOD $good is not on BAD's first-parent chain (it was merged in through a second parent);
  give a GOOD on that chain, such as the merge that brought it in, or pass --no-first-parent"
  fi

  # Every runner the bisect starts keeps its runs here, under the path this
  # launch resolved: a relative OCANNL_TOOL_TEST_RUNS would otherwise be
  # re-read from each step's own directory, scattering the runs (some inside
  # the tree the end removes) away from the locks the bisect reads.
  export OCANNL_TOOL_TEST_RUNS=$RUNS
  # Dune roots at the outermost ancestor holding a dune-workspace (failing
  # that, a dune-project): one above the tree would build another tree.
  # Checked before anything is created; the tree's ancestors are the runs
  # directory's (resolved physically at startup) and the bisect directory.
  anc=$RUNS
  while :; do
    [ ! -e "$anc/dune-workspace" ] || die "bisect: $anc/dune-workspace would root dune above the bisect's tree"
    [ "$anc" != / ] && [ "$anc" != . ] || break
    anc=$(dirname "$anc")
  done
  dir=$RUNS/bisect-$(date -u +%Y%m%dT%H%M%SZ)-$$
  tree=$dir/tree
  # A refusal from here until the first step leaves nothing behind: no
  # registered worktree, no bisect directory.
  bisect_setup_fail() { # <message>
    git worktree remove --force "$tree" >/dev/null 2>&1
    git worktree prune >/dev/null 2>&1
    rm -rf "${dir:?}"
    if [ -n "$bisect_cancelled" ]; then
      echo "test-run: bisect: cancelled ($bisect_cancelled) during setup; nothing was left behind" >&2
      exit "$(bisect_cancel_rc)"
    fi
    die "$1"
  }
  # Cancellation is armed before the first thing the bisect creates, and read
  # after each step of the setup (a signal to this shell waits for the
  # foreground command it interrupts, `git worktree add` included), so an
  # interrupted setup is unwound like a refused one.
  trap 'bisect_cancel INT' INT
  trap 'bisect_cancel TERM' TERM
  trap 'bisect_cancel HUP' HUP
  bisect_setup_check() { [ -z "$bisect_cancelled" ] || bisect_setup_fail cancelled; }
  mkdir "$dir" || die "cannot create $dir"
  bisect_setup_check
  mkdir "$dir/steps" "$dir/runner" "$dir/runner/tools" "$dir/runner/scripts" ||
    bisect_setup_fail "cannot create $dir"
  # The runner: this script and its helpers as they are now (uncommitted
  # edits included, recorded in `runner-source`).
  cp "$runner_root"/tools/*.sh "$dir/runner/tools/" && cp "$runner_root"/scripts/*.sh "$dir/runner/scripts/" ||
    bisect_setup_fail "cannot copy the runner into $dir/runner"
  bisect_setup_check
  {
    printf '%s\n' "$backend" >"$dir/backend" &&
      printf '%s\n' "$cap" >"$dir/cap" &&
      printf '%s\n' "$test_cap" >"$dir/test-cap" &&
      printf '%s\n' "$@" >"$dir/targets" &&
      { [ ${#builds[@]} -eq 0 ] || printf '%s\n' "${builds[@]}"; } >"$dir/builds" &&
      printf '%s\n' "$good_sha" >"$dir/good" &&
      printf '%s\n' "$bad_sha" >"$dir/bad" &&
      printf '%s at %s%s\n' "$runner_root" "$(git rev-parse HEAD 2>/dev/null)" \
        "$(git status --porcelain=v1 2>/dev/null | grep -q . && echo ', with uncommitted changes')" \
        >"$dir/runner-source" &&
      : >"$dir/steps.tsv"
  } || bisect_setup_fail "cannot write the bisect's records in $dir"
  bisect_setup_check
  if ! git worktree add --detach "$tree" "$bad_sha" >"$dir/worktree.log" 2>&1; then
    sed 's/^/  /' "$dir/worktree.log" >&2
    bisect_setup_fail "cannot add the bisect's worktree $tree"
  fi
  bisect_tree=$(cd -P "$tree" && pwd -P) || bisect_setup_fail "cannot resolve $tree"
  lang=$(grep -m1 '^(lang dune ' "$tree/dune-project" 2>/dev/null)
  printf '%s\n' "${lang:-(lang dune 3.18)}" >"$tree/dune-workspace" ||
    bisect_setup_fail "cannot write $tree/dune-workspace"
  bisect_setup_check
  echo "bisect: $dir"
  echo "  range: $good_sha..$bad_sha ($([ -n "$first_parent" ] && echo first-parent || echo all parents))"
  echo "  every step: OCANNL_BACKEND=$backend; build @check${builds[*]:+ ${builds[*]}}, then build $*"
  echo "  runner: $(cat "$dir/runner-source")"

  outcome= result= rc=
  # The ends first: BAD must fail here and GOOD must pass, or the range holds
  # no regression this box can find.
  for line in "bad-end $bad_sha 1 BAD" "good-end $good_sha 0 GOOD"; do
    # shellcheck disable=SC2086 # four words, split on purpose
    set -- $line
    git -C "$tree" checkout --quiet --detach "$2" >>"$dir/worktree.log" 2>&1 ||
      { outcome=error result="cannot check out $2 in $tree"; rc=126; break; }
    bisect_spawn "${BASH:-bash}" "$dir/runner/tools/test-run.sh" _bisect_step "$dir" "$1"
    bisect_drain || :
    [ -z "$bisect_cancelled" ] || break
    case $bisect_rc in
      "$3") ;;
      0 | 1) outcome=ends rc=1
             result="the $4 end $2 $([ "$bisect_rc" = 0 ] && echo passes || echo fails) here: no regression in this range on this box"
             break ;;
      125) outcome=ends rc=1 result="the $4 end $2 does not build here (steps/$1-build.out)"; break ;;
      *) outcome=abort; break ;;
    esac
  done
  if [ -z "$outcome" ] && [ -z "$bisect_cancelled" ]; then
    if git -C "$tree" bisect start ${first_parent:+--first-parent} "$bad_sha" "$good_sha" \
         >>"$dir/git-bisect.out" 2>&1; then
      # Adjacent ends are decided by `start` itself.
      git -C "$tree" bisect log >"$dir/bisect-log" 2>/dev/null
      grep -q "^# first '\{0,1\}bad'\{0,1\} commit: " "$dir/bisect-log" || [ -n "$bisect_cancelled" ] ||
        bisect_spawn git -C "$tree" bisect run "${BASH:-bash}" "$dir/runner/tools/test-run.sh" \
          _bisect_step "$dir" >>"$dir/git-bisect.out"
      bisect_drain || :
      git -C "$tree" bisect log >"$dir/bisect-log" 2>/dev/null
    else
      outcome=error result="git bisect start refused (see git-bisect.out)" rc=126
    fi
  fi
  if [ -n "$bisect_cancelled" ]; then
    outcome=cancelled rc=$(bisect_cancel_rc)
    result="CANCELLED ($bisect_cancelled); the step it stopped left no verdict"
  elif [ -s "$dir/abort" ] || [ "$outcome" = abort ]; then
    outcome=abort
    rc=$(sed -n 's/^status //p' "$dir/abort" 2>/dev/null)
    case $rc in '' | *[!0-9]* | 0 | 1) rc=126 ;; esac
    result="ABORTED: $(sed -n 's/^why //p' "$dir/abort" 2>/dev/null)"
  elif [ -z "$outcome" ]; then
    first=$(sed -n "s/^# first '\{0,1\}bad'\{0,1\} commit: \[\([0-9a-f]\{40,64\}\)\].*/\1/p" \
      "$dir/bisect-log" 2>/dev/null | head -n 1)
    if [ -n "$first" ]; then
      outcome=found rc=0
      result="FIRST BAD COMMIT $first $(git log -1 --format=%s "$first" 2>/dev/null)"
    elif grep -q '^# only skipped commits left to test' "$dir/bisect-log" 2>/dev/null; then
      outcome=skipped rc=1
      result="only skipped commits are left; the first bad commit is one of:
$(sed -n "s/^# possible first '\{0,1\}bad'\{0,1\} commit: \[\([0-9a-f]*\)\] \(.*\)/  \1 \2/p" "$dir/bisect-log")"
    else
      outcome=error rc=126
      result="git bisect ended without a verdict (status $bisect_rc; see git-bisect.out and bisect-log)"
    fi
  fi
  # The verdict is decided; the traps stay armed until the exit, so a signal
  # landing during the cleanup below (the drain is bounded at 60s) only flags
  # itself, and the tree is still removed and the verdict still reported.
  printf '%s\n' "$result" >"$dir/result" 2>/dev/null
  # The tree goes; the records stay. Not while anything the bisect started
  # could still be building in it.
  if bisect_drain; then
    git worktree remove --force "$tree" >>"$dir/worktree.log" 2>&1 ||
      echo "test-run: bisect: could not remove the worktree $tree (see $dir/worktree.log)" >&2
  else
    echo "test-run: bisect: processes of the last step still hold $tree; it is left in place" >&2
  fi
  echo "steps (the runs' digests and logs in $dir/steps; their run directories in steps.tsv):"
  awk -F '\t' '{ printf "  %-9s %s  %s\n", $1, substr($2, 1, 12), $3 }' "$dir/steps.tsv"
  echo "result: $result"
  echo "records: $dir (steps.tsv, steps/, bisect-log, result)"
  exit "$rc"
}

# One step: the commit checked out in the bisect's tree, judged by the frozen
# runner -- this very copy, run as `<dir>/runner/tools/test-run.sh
# _bisect_step <dir> [label]`, by `git bisect run` or by bisect_main for the
# ends. Exits 0 good, 1 bad, 125 skip (the build is red), 128 abort (the
# reason in `<dir>/abort`).
bisect_step() { # <bisect dir> [label]
  local dir=$1 label=${2:-} tree sha n backend cap test_cap targets builds
  local phase_rc phase_run phase_verdict build_run= test_run= verdict
  [ -d "$dir" ] || die "_bisect_step is a bisect's own step, not a command"
  tree=$dir/tree
  # shellcheck disable=SC2046 # a list of variable names, split on purpose
  unset $(git rev-parse --local-env-vars 2>/dev/null)
  bisect_abort() { # <status> <why>
    { printf 'status %s\n' "$1"; printf 'why %s\n' "$2"; } >"$dir/abort"
    printf '%s\t%s\t%s\t%s\t%s\n' "$label" "${sha:-?}" "ABORT" "${build_run:--}" "${test_run:--}" >>"$dir/steps.tsv"
    echo "bisect: $label ${sha:-?}: ABORT -- $2" >&2
    exit 128
  }
  sha=$(git -C "$tree" rev-parse --verify --quiet HEAD) || bisect_abort 126 "cannot read the commit checked out in $tree"
  if [ -z "$label" ]; then
    n=$(grep -c '^step-' "$dir/steps.tsv" 2>/dev/null)
    label=step-$((${n:-0} + 1))
  fi
  backend=$(cat "$dir/backend") && cap=$(cat "$dir/cap") && test_cap=$(cat "$dir/test-cap") ||
    bisect_abort 126 "cannot read the bisect's records in $dir"
  bisect_lines "$dir/targets"
  targets=("${bisect_read_out[@]}")
  bisect_lines "$dir/builds"
  builds=(${bisect_read_out[@]+"${bisect_read_out[@]}"})
  # One phase: a plain `run` of the frozen runner in the bisect's tree, read
  # through what the run itself recorded. Sets phase_rc, phase_run and
  # phase_verdict (pass, fail, build-failed for a BUILD FAILED run -- exit 65
  # over dune's recorded 1, gh-ocannl-1547 -- or the digest's verdict line for
  # anything else).
  bisect_phase() { # <phase> dune-build-args...
    local phase=$1 out recorded
    shift
    out=$dir/steps/$label-$phase.out
    OCANNL_TOOL_TESTRUN_TREE=$tree OCANNL_BACKEND=$backend \
      "${BASH:-bash}" "$runner_root/tools/test-run.sh" run ${cap:+--cap "$cap"} \
      ${test_cap:+--test-cap "$test_cap"} build "$@" </dev/null >"$out" 2>&1
    phase_rc=$?
    phase_run=$(sed -n 's|^log:     \(/.*\)/log$|\1|p' "$out" | tail -n 1)
    phase_verdict=$(sed -n 's/^verdict: //p' "$out" | tail -n 1)
    [ -n "$phase_verdict" ] || phase_verdict="no verdict (exit $phase_rc; see steps/$label-$phase.out)"
    [ -z "$phase_run" ] || cp "$phase_run/log" "$dir/steps/$label-$phase.log" 2>/dev/null
    recorded=
    [ -z "$phase_run" ] || recorded=$(cat "$phase_run/exit" 2>/dev/null)
    case $phase_rc:$recorded in
      0:0 | 1:1 | 65:1) ;;
      *) return 0 ;;
    esac
    # The commit the run recorded is the one under test, or its verdict is
    # about something else.
    if [ "$(cat "$phase_run/head" 2>/dev/null)" != "$sha" ]; then
      phase_verdict="the run recorded commit $(cat "$phase_run/head" 2>/dev/null || echo '(none)'), not $sha"
      return 0
    fi
    case $phase_rc in 0) phase_verdict=pass ;; 1) phase_verdict=fail ;; 65) phase_verdict=build-failed ;; esac
  }
  # Before a verdict goes back to git, which then checks the next commit out:
  # nothing a run started may still hold the tree (a descendant that left
  # dune's group keeps the run's lock for as long as it lives). Bounded; one
  # that outlives the bound aborts the bisect rather than run under a moving
  # tree.
  bisect_settle() {
    local lock i=0
    lock=$RUNS/lock-$(wt_key_of "$(cd -P "$tree" && pwd -P)")
    while :; do
      probe_locks "$lock"
      case $? in
        0) return 0 ;;
        3) ;;
        *) bisect_abort 126 "cannot inspect the tree's lock $lock" ;;
      esac
      i=$((i + 1))
      [ "$i" -le 600 ] ||
        bisect_abort 126 "a process the runs of $sha started still holds $tree after 60s (lock $lock): the next commit cannot be checked out under it"
      sleep 0.1
    done
  }
  bisect_phase build @check ${builds[@]+"${builds[@]}"}
  build_run=${phase_run:--}
  case $phase_verdict in
    pass) bisect_settle ;;
    fail | build-failed)
      bisect_settle
      printf '%s\t%s\tskip\t%s\t-\n' "$label" "$sha" "$build_run" >>"$dir/steps.tsv"
      echo "bisect: $label $sha: skip -- the build is red (steps/$label-build.out)" >&2
      exit 125 ;;
    *) bisect_abort "$phase_rc" "the build run of $sha is not a verdict: $phase_verdict (steps/$label-build.out)" ;;
  esac
  bisect_phase test "${targets[@]}"
  test_run=${phase_run:--}
  case $phase_verdict in
    pass) verdict=good ;;
    fail) verdict=bad ;;
    # Compiled by @check yet failing to build for the targets (a link, an
    # executable @check does not build): the tests never ran, so the commit
    # is neither good nor bad.
    build-failed)
      bisect_settle
      printf '%s\t%s\tskip\t%s\t%s\n' "$label" "$sha" "$build_run" "$test_run" >>"$dir/steps.tsv"
      echo "bisect: $label $sha: skip -- the targets do not build (steps/$label-test.out)" >&2
      exit 125 ;;
    *) bisect_abort "$phase_rc" "the test run of $sha is not a verdict: $phase_verdict (steps/$label-test.out)" ;;
  esac
  bisect_settle
  printf '%s\t%s\t%s\t%s\t%s\n' "$label" "$sha" "$verdict" "$build_run" "$test_run" >>"$dir/steps.tsv"
  echo "bisect: $label $sha: $verdict (steps/$label-test.out)" >&2
  [ "$verdict" = good ] && exit 0
  exit 1
}
