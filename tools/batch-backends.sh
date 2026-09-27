#!/usr/bin/env bash
# Which backends a tools/test-run.sh batch can hold -- the one resolution both
# of the runner's per-batch decisions read (gh-ocannl-1066): the dune width the
# batch runs at (the tightest cap any of its backends meets on this box,
# tools/box-jobs.sh) and the fleet slot it takes (`--cpu` only when none of its
# backends holds a GPU; gh-ocannl-1004). Sourced by tools/test-run.sh after
# tools/box-jobs.sh, from the repository root; never executed.
#
# Why one resolution: the width used to read OCANNL_BACKEND alone, while the
# slot resolved the backend properly, so the two answered "which backend does
# this batch hold" differently. A batch whose backend came from a test
# configuration rather than the environment got the right slot and the wrong
# width -- on a native minix hip batch, dune's default width, the SDMA
# exhaustion of gh-ocannl-1029.
#
# The backends are RESOLVED, not guessed, and an answer that cannot be read is
# every backend. Three sources, each adding to the set:
#   - The dune argv: `exec` runs a program that may pick any backend, so it is
#     every backend; a command-line `--ocannl_backend=<name>` adds that name.
#   - The stanzas the argv reaches that NAME a backend (`; ocannl-backend: cuda
#     -- …`, which env_var_deps enforces on every stanza that does not read the
#     configuration): they hold it whatever the configuration says.
#     `ocannl_slot_kind` (test/config, over Test_utils.Slot_kind) lists each
#     named backend with the first stanza naming it, or answers unknown for an
#     argv it does not model.
#   - The configuration the other stanzas read: `ocannl_read_config`
#     (test/config, the same Utils resolution a test run makes: config file,
#     environment, command line), asked from each directory whose
#     `ocannl_config` a test can read -- test/config (copied by every test/*
#     directory and by bin/) and arrayjit/test (its own). No backend at all is
#     not cc: `Context.auto` then tries metal, cuda and hip, so it adds all
#     three.
# A reader that does not build, a read that fails, a name this does not know
# and an answer cut short are each every backend. So a misreading costs a
# narrower width or a wait for a GPU token, never a GPU batch at dune's default
# width or outside the tokens.
#
# The readers are built under the worktree lock the launcher holds, before the
# run is published: a few seconds warm, and their libraries are the batch's own
# anyway. OCANNL_TOOL_READ_CONFIG and OCANNL_TOOL_SLOT_KIND (both, or neither)
# name stand-ins for them, which is how tools/test-test-run.sh drives this
# without a build.

# The directories whose `ocannl_config` a test run can read, relative to the
# repository root: the shared test configuration, and arrayjit's own.
BATCH_CONFIG_DIRS="test/config arrayjit/test"

# Every backend a batch could hold, for an answer that cannot be read.
BATCH_ALL_BACKENDS="cuda hip metal cc multidev_cc"

batch_known_backend() { # <name>; 0 iff it is a backend (a deprecated alias included)
  case ${1:-} in cc | multidev_cc | sync_cc | multicore_cc | cuda | hip | metal) return 0 ;; esac
  return 1
}

batch_canonical() { # <name>; the canonical name of a deprecated alias
  case $1 in
    sync_cc) printf 'cc' ;;
    multicore_cc) printf 'multidev_cc' ;;
    *) printf '%s' "$1" ;;
  esac
}

batch_holds=    # one line per backend the batch can hold: `<backend> <why>`
batch_unknown=  # non-empty: the set could not be read, and why -- every backend
# shellcheck disable=SC2034  # read by tools/test-run.sh, which sources this file
batch_resolved= # non-empty once batch_resolve ran

batch_add() { # <backend> <why>; the first reason for a backend is kept
  local b
  b=$(batch_canonical "$1")
  case $'\n'$batch_holds in *$'\n'"$b "*) return 0 ;; esac
  batch_holds="$batch_holds$b $2"$'\n'
}

# The resolution's reasons, one line per backend, go to the run's log (where
# triage reads them) rather than the terminal, which gets the summary the width
# and slot announcements carry.
batch_say() { # <log> <line>
  printf 'test-run: batch: %s\n' "$2" >>"$1"
}

batch_summary() { # the batch's backends, for an announcement
  if [ -n "$batch_unknown" ]; then
    printf 'any backend (its backends are unread: %s)' "$batch_unknown"
  else
    batch_backends | paste -s -d ' ' - | sed 's/ /, /g'
  fi
}

# Whether this box has anything the resolution could change: a backend that
# meets a width cap here. A box with none (every macOS box, a Linux box without
# a GPU) needs the backends only for a fleet slot.
batch_box_has_hazard() {
  local b
  for b in $BATCH_ALL_BACKENDS; do
    [ -z "$(box_jobs_local_hazard "$b")" ] || return 0
  done
  return 1
}

# Runs a command in <dir>, in a process group of its own, under a deadline of
# <seconds> (0: none) that kills the whole group: a child that outlives the
# command would otherwise hold a pipe open to its own end. INT, TERM and HUP
# reaching the runner are relayed to the group, which gets five seconds to act
# before it is KILLed; and a group left behind by a command that exited is
# KILLed too, since it could hold the worktree lock with nothing over it
# (Codex review rounds 1-6 on PR #832). Exits with the command's status, 124
# on the deadline. Shared by the backends' readers and the fleet-slot probe:
# `perl -e "$BATCH_GROUP_RUNNER" <seconds> <dir> command...`, run directly so
# that `$!` of a backgrounded one is the runner itself.
# shellcheck disable=SC2016  # perl, not shell, expands these
BATCH_GROUP_RUNNER='
    use POSIX ":sys_wait_h";
    my ($left, $dir) = splice(@ARGV, 0, 2);
    defined(my $pid = fork) or exit 127;
    if (!$pid) { setpgrp(0, 0); chdir $dir or exit 127; exec @ARGV or exit 127 }
    setpgrp($pid, $pid);
    # Ending early: the signal to the group, a bounded grace for it to act,
    # then KILL to whatever of the group is left, and the leader reaped --
    # a group that ignores TERM must not outlive this runner holding the
    # worktree lock with no alarm left over it (Codex review round 3).
    my $end = sub {
      my ($sig, $code) = @_;
      $SIG{$_} = "IGNORE" for qw(ALRM INT TERM HUP);
      kill $sig, -$pid; kill $sig, $pid;
      if ($sig ne "KILL") {
        for (1 .. 50) { last if waitpid($pid, WNOHANG) != 0; select(undef, undef, undef, 0.1) }
      }
      kill "KILL", -$pid; kill "KILL", $pid;
      waitpid($pid, 0);
      exit $code;
    };
    $SIG{ALRM} = sub { $end->("KILL", 124) };
    $SIG{INT} = sub { $end->("INT", 130) };
    $SIG{TERM} = sub { $end->("TERM", 143) };
    $SIG{HUP} = sub { $end->("TERM", 129) };
    alarm $left if $left > 0;
    waitpid($pid, 0);
    my $st = $?;
    # A reader that left a background descendant behind has not finished:
    # that descendant holds the worktree lock with no alarm over it, so the
    # group goes too, whatever the leader answered (Codex review round 5).
    $SIG{$_} = "IGNORE" for qw(ALRM INT TERM HUP);
    alarm 0;
    kill "KILL", -$pid;
    exit(($st & 127) ? 128 + ($st & 127) : $st >> 8);
  '

# Runs a command under what is left of the resolution's <cap> seconds (0:
# unbounded), so the readers' build and every reader run share one budget:
# a hung reader answers "unread" -- every backend -- instead of stranding the
# launch (Codex review round 1 on PR #832). Uses batch_resolve's locals; with
# `-C <dir>` the command runs in <dir>, under BATCH_GROUP_RUNNER. The runner is
# waited on in the BACKGROUND, because bash
# defers a trap until a foreground command completes: this way a signal to
# the launcher alone runs its trap at once, and batch_cancel_hook (the
# launcher's) decides whether to end the runner (Codex review round 2).
batch_child=   # the runner in flight, for batch_abort
batch_bounded() { # [-C dir] command...
  local left=0 dir=. rc
  if [ "${1:-}" = -C ]; then dir=$2; shift 2; fi
  # A cancellation the launcher trapped before this reader started (in
  # new_run, take_lock, or an earlier reader) starts nothing more (Codex
  # review round 4 on PR #832).
  ! batch_cancel_hook || return 143
  if [ "$bcap" -ne 0 ]; then
    left=$((bcap - (SECONDS - bstart)))
    [ "$left" -gt 0 ] || return 124
  fi
  perl -e "$BATCH_GROUP_RUNNER" "$left" "$dir" "$@" &
  batch_child=$!
  while :; do
    wait "$batch_child"
    rc=$?
    # A trapped signal returns from `wait` early, with the runner still there.
    kill -0 "$batch_child" 2>/dev/null || break
    ! batch_cancel_hook || kill -TERM "$batch_child" 2>/dev/null
  done
  batch_child=
  return "$rc"
}

# Whether a signal the launcher trapped means the resolution must end: the
# launcher redefines it; standalone, nothing cancels.
batch_cancel_hook() { return 1; }

# Ends a runner in flight and waits for it -- its grace and KILL included --
# so that nothing holding the worktree lock outlives a trap that exits at once
# (Codex review round 4 on PR #832).
batch_abort() {
  [ -n "$batch_child" ] || return 0
  kill -TERM "$batch_child" 2>/dev/null
  while kill -0 "$batch_child" 2>/dev/null; do wait "$batch_child" 2>/dev/null; done
  batch_child=
  return 0
}

# Resolves the batch's backends into batch_holds / batch_unknown, and says what
# it found. <dune> builds the readers; the build and the readers together are
# bounded by <cap> seconds (0: unbounded), and dune's output is appended to
# <log>.
batch_resolve() { # <dune> <log> <cap> dune-argv...
  local dune=$1 log=$2 bcap=$3 bstart=$SECONDS a v reader reach out line rest b ended= d
  shift 3
  batch_holds= batch_unknown= batch_resolved=1
  if [ "${1:-}" = exec ]; then
    batch_unknown="dune exec runs a program that may pick its own backend"
  fi
  # The argv the reachability tool reads: the caller's, less the backend
  # flags read here, which its closed grammar would take as unmodelled
  # (Codex review round 6 on PR #832).
  local -a reach_argv=()
  for a; do
    [ -z "$batch_unknown" ] || break
    case $a in
      --ocannl[_-]backend=*) v=${a#*=} ;;
      *ocannl[_-]backend*) v= ;;
      *) reach_argv+=("$a"); continue ;;
    esac
    if batch_known_backend "$v"; then
      batch_add "$v" "the command line names it ($a)"
    else
      batch_unknown="the command line names a backend this does not read ($a)"
    fi
  done
  if [ -z "$batch_unknown" ]; then
    if [ -n "${OCANNL_TOOL_READ_CONFIG:-}" ] && [ -n "${OCANNL_TOOL_SLOT_KIND:-}" ]; then
      reader=$OCANNL_TOOL_READ_CONFIG reach=$OCANNL_TOOL_SLOT_KIND
    elif batch_bounded \
           "$dune" build ./test/config/ocannl_read_config.exe ./test/config/ocannl_slot_kind.exe \
           </dev/null >>"$log" 2>&1; then
      # Where dune just put them: DUNE_BUILD_DIR moves the build tree (as
      # tools/ci-compiler-test.sh does), and a copy left under _build/default
      # would be missing -- or stale (Codex review round 4 on PR #803).
      local build_dir=${DUNE_BUILD_DIR:-_build}
      case $build_dir in /*) ;; *) build_dir=$PWD/$build_dir ;; esac
      reader=$build_dir/default/test/config/ocannl_read_config.exe
      reach=$build_dir/default/test/config/ocannl_slot_kind.exe
    else
      batch_unknown="the backend readers did not build (the run's log has dune's output)"
    fi
  fi
  if [ -z "$batch_unknown" ]; then
    if batch_bounded "$reach" ${reach_argv[@]+"${reach_argv[@]}"} >"$log.reach" 2>/dev/null; then
      out=$(cat "$log.reach" 2>/dev/null)
    else
      # A failed read is not an answer, however complete what it printed
      # looks (Codex review round 5 on PR #832).
      batch_unknown="ocannl_slot_kind failed (exit $?)"
    fi
    rm -f "$log.reach"
  fi
  if [ -z "$batch_unknown" ]; then
    while IFS= read -r line; do
      [ -z "$ended" ] || { batch_unknown="ocannl_slot_kind answered past its end: '$line'"; break; }
      case $line in
        'names '*': '*)
          rest=${line#names }
          b=${rest%%: *}
          batch_known_backend "$b" ||
            { batch_unknown="ocannl_slot_kind named a backend this does not know: '$line'"; break; }
          batch_add "$b" "${rest#*: }"
          ;;
        end) ended=1 ;;
        'unknown: '*) batch_unknown=${line#unknown: }; break ;;
        *) batch_unknown="ocannl_slot_kind answered '${line}'"; break ;;
      esac
    done <<<"$out"
    [ -n "$batch_unknown" ] || [ -n "$ended" ] ||
      batch_unknown="ocannl_slot_kind's answer was cut short: '${out}'"
  fi
  if [ -z "$batch_unknown" ]; then
    for d in $BATCH_CONFIG_DIRS; do
      if ! batch_bounded -C "$d" "$reader" --read=backend --output=stdout >"$log.read" 2>/dev/null; then
        rm -f "$log.read"
        batch_unknown="the backend $d resolves is unreadable"
        break
      fi
      b=$(cat "$log.read" 2>/dev/null)
      rm -f "$log.read"
      if [ -z "$b" ]; then
        for v in metal cuda hip; do
          batch_add "$v" "$d resolves no backend, so Context.auto tries the GPUs first"
        done
      elif batch_known_backend "$b"; then
        if [ -n "${OCANNL_BACKEND:-}" ] && [ "$(batch_canonical "$OCANNL_BACKEND")" = "$(batch_canonical "$b")" ]; then
          batch_add "$b" "$d resolves backend=$b, from OCANNL_BACKEND"
        else
          batch_add "$b" "$d resolves backend=$b"
        fi
      else
        batch_unknown="$d resolves backend=$b, which this does not know"
        break
      fi
    done
  fi
  if [ -n "$batch_unknown" ]; then
    batch_say "$log" "its backends are unread, so it is taken to hold any: $batch_unknown"
  else
    while IFS=' ' read -r b rest; do
      [ -z "$b" ] || batch_say "$log" "holds $b: $rest"
    done <<<"$batch_holds"
  fi
  return 0
}

# The backends the width and the kind are judged over, one per line: the
# resolved ones, or every backend for an unread answer.
batch_backends() {
  local b rest
  if [ -n "$batch_unknown" ]; then
    # shellcheck disable=SC2086  # the list is word-split on purpose
    printf '%s\n' $BATCH_ALL_BACKENDS
  else
    while IFS=' ' read -r b rest; do [ -z "$b" ] || printf '%s\n' "$b"; done <<<"$batch_holds"
  fi
}

batch_kind() { # prints cpu when no backend of the batch holds a GPU, else gpu
  local b
  while IFS= read -r b; do
    box_jobs_cpu_backend "$b" || { printf 'gpu'; return 0; }
  done < <(batch_backends)
  printf 'cpu'
}

# The tightest width any of the batch's backends meets here, with the backend
# and hazard that set it (the first to reach that width). Empty when none does.
batch_width_cap= batch_width_backend= batch_width_hazard=
batch_width() {
  local b hazard c
  batch_width_cap= batch_width_backend= batch_width_hazard=
  while IFS= read -r b; do
    hazard=$(box_jobs_local_hazard "$b")
    c=$(box_jobs_hazard_cap "$hazard")
    [ -n "$c" ] || continue
    if [ -z "$batch_width_cap" ] || [ "$c" -lt "$batch_width_cap" ]; then
      batch_width_cap=$c batch_width_backend=$b batch_width_hazard=$hazard
    fi
  done < <(batch_backends)
}

batch_why() { # <backend>; why the batch holds it
  local b rest
  if [ -n "$batch_unknown" ]; then
    printf 'its backends are unread (%s)' "$batch_unknown"
    return 0
  fi
  while IFS=' ' read -r b rest; do
    [ "$b" = "$1" ] && { printf '%s' "$rest"; return 0; }
  done <<<"$batch_holds"
}
