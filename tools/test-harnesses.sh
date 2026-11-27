#!/usr/bin/env bash
# Run the shell-harness tier CI runs, from any working directory.
# Usage: tools/test-harnesses.sh [--shell|--toolchain|--promotion] [--list]
#                               [--expected-skips FILE] [--jobs N] [--timings]
#        tools/test-harnesses.sh --ci-shell [--list] [--jobs N] [--timings]
#        tools/test-harnesses.sh --ci-real-dune [--list] [--jobs N] [--timings]
#        tools/test-harnesses.sh [GROUP] --affected [--list] [--jobs N] [--timings] --since BASE
#        tools/test-harnesses.sh [GROUP] --affected [--list] [--jobs N] [--timings] --files PATH...
#        tools/test-harnesses.sh --member HARNESS [--member HARNESS]... [--list] [--jobs N] [--timings]
# With no group, run all three. --shell needs no OCaml toolchain; --toolchain
# runs the formatter controls and the scan-case type check (which links
# ppxlib) and --promotion the controls of tools that read
# Dune's own output (promotion, affected tests, the failed-rule listing), whose
# missing-toolchain legs report counted skips. CI runs both after installing
# its toolchain, and --promotion again at the declared Dune floor.
# On hosts where /bin/bash reports exactly 3.2, selected members run under
# /bin/bash instead of the PATH interpreter, including --affected and merge
# builds. Else they run under this dispatcher's Bash; no 3.2 leg is claimed.
# On macOS (uname -s = Darwin), `awk` is also pinned to `mawk` inside that same
# private PATH entry (searched on PATH via `type -P mawk`, then standard
# Homebrew paths `/opt/homebrew/bin/mawk` and `/usr/local/bin/mawk`), so macOS
# runs exercise Linux mawk semantics rather than macOS's BWK `/usr/bin/awk`
# (gh-ocannl-2414). Where `mawk` is missing on macOS, the run refuses (exit 2)
# before taking a fleet slot or starting any member, naming `brew install mawk`,
# and never falls back to BWK awk. Non-Darwin hosts are unaffected.
# HARNESS opening lines name that interpreter (and the pinned `awk` on macOS).
# A private PATH entry carries both to nested `bash`/`awk` calls and
# `#!/usr/bin/env bash` or `#!/usr/bin/env awk` subjects; ${BASH} also names
# the Bash interpreter in each member. Absolute interpreter paths intentionally
# remain explicit.
# A check needing newer Bash syntax gates on its running BASH_VERSINFO and
# declares itself not applicable; the dispatcher has no member skip list.
# --list remains interpreter-independent and runs no members.
# The aggregate verifies the checkout: it clears any ambient
# OCANNL_TOOL_HARNESS_SUBJECT before running members.
# Every selected harness runs after ordinary failures; any failure exits 1.
# HUP/INT/TERM stop immediately and preserve the interruption status.
# A member fails if its stderr has a complete English Bash missing
# command diagnostic in either form. Numbered lines read
# `SOURCE: line N: COMMAND: command not found`. Unnumbered lines read
# `SOURCE: COMMAND: command not found`, where SOURCE may be a name passed
# to Bash 3.2's `-c` and neither field contains a colon. Verbatim fixture
# text on stderr is in scope. Prose, localized diagnostics and diagnostics the
# member retains in its own fixture are outside this check. There is no
# expected-error list.
#
# --jobs N runs up to N members at once (gh-ocannl-2002), started in the
# manifest's order: a member is a program of its own with its own scratch
# directory, and much of what the slow ones spend is fixtures waiting, not
# computing -- the six members of tools/test-run.sh's harness, about 760 s one
# after another on a MacBook Pro, take 165 s beside one another on two cores'
# worth of CPU. Each member's stdout and stderr are kept separately and
# printed in that order between its HARNESS lines once it and every member
# before it in the manifest has ended. A member that ends
# interrupted (129, 130, 143) ends the run with that status as soon as it is
# seen to have ended -- no member starts after it, the members beside it are
# sent a TERM, and what ran is printed -- and so does a signal to this
# script. N is a positive integer; 1, the default, runs one member at a time
# with its output live, as before the option.
# A caller can set OCANNL_TOOL_HARNESS_TIMINGS_FILE to an absolute scratch path
# to collect a row per finished member. The dispatcher appends in manifest order,
# outside the member's output and verdict. This write is best-effort. Whole
# seconds run from invocation to observed exit, before serial relay drain or
# any wait to print an already-reaped member's output. Parallel rows are
# flushed at exit, including on handled interruption.
# --timings prints that same per-member timing table on stdout as a
# `harnesses: timings:` block (indented `  HARNESS CMD finished in Ns (exit RC)`
# rows in manifest order, plus `  HARNESS CMD interrupted after Ns (in flight)`
# if the serial dispatcher exits on a signal mid-member) right before the
# closing `harnesses:` summary or interruption exit. Without --timings, stdout
# stays timing-free so merge-build shared-red comparisons stay line-stable.
# --list runs no member and prints no timings.
# In serial, each member writes its captured stderr (and stdout under
# --ci-real-dune) directly to its per-member output file while a dispatcher-
# owned live relay streams that file to stderr (and stdout). Verdict scans
# therefore read complete capture files even if downstream consumption stalls
# or a descendant holds an output descriptor open. After ordinary completion,
# the dispatcher marks the member finished so each relay drains up to the
# member's exit size and exits, bounding that drain to ~1 s (20 ticks of
# 0.05 s) before stopping any still-running relay (TERM, 0.5 s grace, KILL),
# reaping it, and noting on stderr that live output was cut while the verdict
# read the full capture. An interrupted member (129, 130, 143) stops and reaps
# its relays at once, and the EXIT trap stops and reaps any active relays
# (including one just forked before registration) when the dispatcher exits,
# including on a signal sent only to the dispatcher's PID.
#
# --expected-skips FILE runs every member strict (OCANNL_TOOL_HARNESS_STRICT=1)
# except for the skips FILE lists, so a skip that newly appears on the
# platform FILE describes fails its member, and a listed skip that no longer
# occurs fails the tier (gh-ocannl-1236). FILE is read as a fail-closed
# allowlist of exactly three line forms, any other line refusing the run
# (exit 2) before a member starts:
#   - an empty line;
#   - a comment, `#` in the first column;
#   - `HARNESS<TAB>LABEL`: HARNESS one of the selected members as --list
#     prints it, LABEL the skip's label exactly as the member prints it after
#     `SKIP  ` -- non-empty, with no further tab and no carriage return, and
#     listed once.
# Without the option the list variables are cleared, so no ambient list
# relaxes a strict run. A leg that declares itself not applicable here
# (harness_not_applicable, scripts/harness-support.sh: its subject is absent
# from this platform, so no coverage is lost) is no skip: it prints `N/A   `,
# a strict run passes it, and no list names it (gh-ocannl-1839).
#
# --ci-shell runs the shell tier as the toolchain-free `harnesses` CI job
# does, which is the job's own command, so any Linux or macOS box reproduces
# its skip check before a push (gh-ocannl-1593): --shell --expected-skips
# tools/harness-skips-ci-shell.txt joined with
# tools/harness-skips-ci-shell-dune.txt, with that job's absences put on: dune
# hidden from every real-dune probe (OCANNL_TOOL_HARNESS_NO_DUNE=1,
# scripts/harness-support.sh's harness_real_dune), FLEET_BOXES unset and the
# fleet anchor's wake-lab.sh pointed away (OCANNL_TOOL_SWEEP_WAKE_LAB). On
# macOS the further skips tools/harness-skips-ci-shell-darwin.txt lists are
# expected too, under the same rules: coverage a macOS host loses, none now,
# and never a leg for a Linux-only facility, which declares itself not
# applicable instead (gh-ocannl-1930). The dune list is held to exactly the skips the hidden dune
# caused, told by their reason (gh-ocannl-1650): such a skip listed in another
# file, or a dune-list skip with another reason, fails the tier. It takes no
# group and no --expected-skips. Without it, an ambient
# OCANNL_TOOL_HARNESS_NO_DUNE is cleared, so no run hides dune unasked.
#
# --ci-real-dune is the `harness-real-dune` CI job's command, the other half
# (gh-ocannl-1650): with a dune installed, it runs only the legs that hold the
# dune list's checks -- each member's selector in real_dune_legs below,
# passed as --only -- strict with no skip expected, and fails the tier unless
# every check the dune list names printed PASS. A check whose leg the selector
# misses is red there, never a skip nobody runs per PR. It takes no group and
# no --expected-skips.
#
# --affected --since BASE is a shell change's pre-push, the command
# `tools/test-run.sh prepush` closes by naming: exactly the commands
# `tools/test-run.sh affected --harnesses --since BASE` prints, each member with
# the --only printed for it, in manifest order, under one slot and one footer
# instead of a launch per command. --files PATH... asks the same of those
# repository paths instead of a diff, for a caller whose paths no revision range
# states (tools/merge-build-check.sh: the paths both sides of a merge changed);
# it takes every word after it, none of them reading as an option. A group keeps
# the members of that group, for a caller bound to it; what it leaves out is
# still checked below. The selection is the change's, so it takes no
# --expected-skips and no CI form, and --since and --files belong to it alone.
# Those commands are read, never evaluated, through a fail-closed allowlist:
# each line is `HARNESS` or `HARNESS --only 'ERE'`, HARNESS a member listed once
# and ERE free of quotes, tabs and carriage returns. Any other line, or an ask
# that fails (a BASE that is no revision), refuses the run (exit 2) before a
# slot is taken or a member starts. A change reaching no harness is a result,
# not the empty selection the other forms refuse: exit 0, `harnesses: 0 run, 0
# failed`, no slot. --list prints the commands and runs nothing.
#
# On a fleet box a run takes one of the box's run-time correctness slots
# itself, as --cpu (no member holds a device), as tools/test-run.sh and
# tools/fmt-check.sh take theirs and through the same probe,
# tools/fleet-worker-candidates.sh's fleet_slot_probe (gh-ocannl-1576): this
# script runs again under `fleet-worker.sh execution slot --wait
# <OCANNL_TOOL_SLOT_WAIT, 600> --cpu`, through the rerun helper it shares with
# tools/fmt-check.sh, fleet_slot_rerun (gh-ocannl-1841), after its arguments
# and lists are checked and before any member starts; a caller's own enclosing
# slot is reused, not doubled. A slot that never ran it (refused, or the
# registry unreachable) exits 75 with `harnesses: FAILED (exit 75): SLOT
# REFUSED` and no member run; otherwise the copy records its status only on
# explicit exits via `exit_with` (leaving the record empty on an untrapped
# signal or implicit abort so `fleet_slot_rerun` propagates the slot status).
# --list takes no slot, and no probe answer (CI, a machine outside the
# fleet, OCANNL_TOOL_FLEET_WORKER=none) runs the members directly, silently.
set -u
# Name a locale before any subshell forks (gh-ocannl-2002, gh-ocannl-2462, gh-ocannl-2484).
[ -n "${LANG:-}" ] || [ -n "${LC_ALL:-}" ] || export LANG=C
# Set only by this script, for the copy of itself it runs inside the fleet
# slot (take_fleet_slot): the file that says it started and, written only by
# `exit_with` on explicit exits, its status. Cleared, so no member's own run
# of this script inherits it.
slot_record=${OCANNL_TOOL_HARNESSES_IN_SLOT-}
unset OCANNL_TOOL_HARNESSES_IN_SLOT
[ -z "$slot_record" ] || : >"$slot_record" || exit 2
# Record the slot status only on explicit exits, never from the EXIT trap: on an
# untrapped TERM/HUP Bash runs the EXIT trap with the previous command's $?
# before re-raising the signal (gh-ocannl-2450).
exit_with() {
  [ -z "$slot_record" ] || echo "$1" >"$slot_record"
  exit "$1"
}
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd) || exit_with 2
group=all list=0 expected= ci_shell=0 ci_real_dune=0 affected=0 since= files=() explicit_members=() jobs=1 timings=0 timings_printed=0 serial_running_idx= serial_running_started=0
original_args=("$@")
while [ $# -gt 0 ]; do
  arg=$1
  case $arg in
    --shell|--toolchain|--promotion)
      [ "$group" = all ] || { echo "choose one harness group" >&2; exit_with 2; }
      group=${arg#--} ;;
    --list) list=1 ;;
    --timings) timings=1 ;;
    --ci-shell) ci_shell=1 ;;
    --ci-real-dune) ci_real_dune=1 ;;
    --affected) affected=1 ;;
    --member)
      [ $# -ge 2 ] && [ -n "$2" ] || { echo "test-harnesses.sh: --member needs a harness path" >&2; exit_with 2; }
      explicit_members+=("$2")
      shift ;;
    --jobs)
      [ $# -ge 2 ] && [[ $2 =~ ^[1-9][0-9]*$ ]] ||
        { echo "test-harnesses.sh: --jobs needs a positive integer" >&2; exit_with 2; }
      jobs=$2
      shift ;;
    --since)
      [ $# -ge 2 ] && [ -n "$2" ] || { echo "test-harnesses.sh: --since needs a base revision" >&2; exit_with 2; }
      since=$2
      shift ;;
    --files)
      shift
      [ $# -ge 1 ] || { echo "test-harnesses.sh: --files needs at least one path" >&2; exit_with 2; }
      files=("$@")
      break ;;
    --expected-skips)
      [ $# -ge 2 ] && [ -n "$2" ] || { echo "test-harnesses.sh: --expected-skips needs a file" >&2; exit_with 2; }
      expected=$2
      shift ;;
    -h|--help) sed -n '2,${/^#/!q;p;}' "$0" | sed 's/^# \{0,1\}//'; exit_with 0 ;;
    *) echo "test-harnesses.sh: unknown argument '$arg'" >&2; exit_with 2 ;;
  esac
  shift
done
if [ "$ci_shell" = 1 ] && [ "$ci_real_dune" = 1 ]; then
  echo "test-harnesses.sh: choose one of --ci-shell and --ci-real-dune" >&2
  exit_with 2
fi
if [ "$ci_shell" = 1 ] || [ "$ci_real_dune" = 1 ]; then
  [ "$group" = all ] && [ -z "$expected" ] || {
    echo "test-harnesses.sh: --ci-shell and --ci-real-dune take no group and no --expected-skips" >&2
    exit_with 2
  }
  group=shell
fi
if [ "$affected" = 1 ]; then
  if [ -n "$since" ] && [ ${#files[@]} -gt 0 ]; then
    echo "test-harnesses.sh: --affected: choose one of --since and --files" >&2
    exit_with 2
  fi
  [ -n "$since" ] || [ ${#files[@]} -gt 0 ] ||
    { echo "test-harnesses.sh: --affected needs --since BASE or --files PATH..." >&2; exit_with 2; }
  [ -z "$expected" ] && [ "$ci_shell" = 0 ] && [ "$ci_real_dune" = 0 ] || {
    echo "test-harnesses.sh: --affected selects what the change reaches, so it takes no --expected-skips and no CI form" >&2
    exit_with 2
  }
  for path in ${files[@]+"${files[@]}"}; do
    case $path in
      ''|-*) echo "test-harnesses.sh: --affected: '$path' is no repository path --files can ask about" >&2; exit_with 2 ;;
    esac
  done
elif [ -n "$since" ] || [ ${#files[@]} -gt 0 ]; then
  echo "test-harnesses.sh: --since and --files are --affected's" >&2
  exit_with 2
fi
if [ ${#explicit_members[@]} -gt 0 ]; then
  [ "$group" = all ] && [ -z "$expected" ] && [ "$ci_shell" = 0 ] && [ "$ci_real_dune" = 0 ] && [ "$affected" = 0 ] || {
    echo "test-harnesses.sh: --member cannot be combined with groups, --expected-skips, --affected or CI forms" >&2
    exit_with 2
  }
fi
# This is the sole membership list: CI selects groups from this entrypoint.
manifest() {
  cat <<'HARNESS_LIST'
shell scripts/test-harness-support.sh
shell scripts/test-process-group.sh
shell tools/test-test-harnesses.sh
shell tools/test-real-dune-registrations.sh
shell tools/test-harness-legs-alone.sh
shell tools/test-pin-revisions.sh
shell tools/test-test-run-options.sh
shell tools/test-test-run-digest.sh
shell tools/test-test-run-devices.sh
shell tools/test-test-run-lifecycle.sh
shell tools/test-test-run-last.sh
shell tools/test-test-run-bisect.sh
shell tools/test-mutation-run.sh
shell scripts/test-setup-ocaml-env.sh
shell tools/test-windows-opam-cache.sh
shell tools/test-isolated-apt.sh
shell tools/test-ci-times.sh
shell tools/test-ci-durations.sh
shell tools/test-ci-shard.sh
shell tools/test-action-durations.sh
shell test/operations/apt_fallback.sh
shell test/operations/ci_matrix.sh
shell tools/test-machine-verify.sh
shell tools/test-merge-build-check.sh
shell tools/test-owed-run-tty.sh
shell tools/test-merge-build-replay.sh
shell tools/test-row-trace-compact.sh
shell tools/test-compare-bounded-outcomes.sh
shell tools/test-device-memory-load.sh
shell tools/test-refusal-manifest-sync.sh
shell benchmarks/test-gh1133-cells.sh
shell benchmarks/test-gh1253-cells.sh
shell benchmarks/test-gh834-cells.sh
shell benchmarks/test-check-box.sh
shell benchmarks/test-box-gate.sh
toolchain tools/test-fmt-check.sh
toolchain tools/test-typecheck-scan-cases.sh
toolchain tools/test-ppx-output-diff.sh
promotion tools/test-promote.sh
promotion tools/test-promotion-record.sh
promotion tools/test-affected-tests.sh
promotion tools/test-ci-failed-rules.sh
HARNESS_LIST
}
# --ci-real-dune's legs: `HARNESS<TAB>ERE`, the --only selector of the legs
# holding HARNESS's real-dune checks (each is checked to PASS, not trusted).
real_dune_legs() {
  printf '%s\t%s\n' tools/test-test-run-options.sh '^(89|107|120):'
  printf '%s\t%s\n' tools/test-test-run-digest.sh '^(36d|80b|80c|84|92b):'
  printf '%s\t%s\n' tools/test-test-run-bisect.sh '^(79|121):'
  printf '%s\t%s\n' tools/test-merge-build-check.sh '#compile-output|#compile-only-real'
}
members() {
  local tier script
  while read -r tier script; do
    [ "$group" = all ] || [ "$tier" = "$group" ] || continue
    printf '%s\n' "$script"
  done < <(manifest)
}
unset OCANNL_TOOL_HARNESS_EXPECTED_SKIPS OCANNL_TOOL_HARNESS_SKIP_RECORD OCANNL_TOOL_HARNESS_NO_DUNE \
  OCANNL_TOOL_HARNESS_DUNE_SKIP_RECORD OCANNL_TOOL_HARNESS_SUBJECT
scratch=()
# Bound how long serial mode waits for a live relay to finish streaming after a
# member exits, and how long relay teardown waits after TERM before escalating
# to KILL so a stopped or TERM-ignoring relay cannot hang cleanup.
serial_drain_ticks=20
serial_kill_grace_ticks=10
serial_relays=()
serial_jobs_live() { # sets live to newline-bounded running/stopped child PIDs
  live=$'\n'$(trap '' HUP INT TERM; jobs -pr; jobs -ps)$'\n'
}
in_live() { # PID
  case $live in
    *$'\n'"$1"$'\n'*) return 0 ;;
    *) return 1 ;;
  esac
}
kill_serial_relays() { # PID... : TERM, bounded grace, then KILL survivors
  local ticks=0 live still pid
  [ $# -gt 0 ] || return 0
  for pid in "$@"; do
    kill -TERM "$pid" 2>/dev/null || :
  done
  while [ "$ticks" -lt "$serial_kill_grace_ticks" ]; do
    serial_jobs_live
    still=0
    for pid in "$@"; do
      if in_live "$pid"; then still=1; break; fi
    done
    [ "$still" = 0 ] && return 0
    ticks=$((ticks + 1))
    sleep 0.05
  done
  serial_jobs_live
  for pid in "$@"; do
    if in_live "$pid"; then kill -KILL "$pid" 2>/dev/null || :; fi
  done
}
stop_serial_relays() {
  local live pid to_kill=() all_pids=() seen=$'\n'
  [ "$jobs" = 1 ] || return 0
  # In serial mode, live background jobs are exclusively our own relays; reading
  # the job table also covers a relay forked just before serial_relays recorded it.
  serial_jobs_live
  for pid in $live; do
    to_kill+=("$pid")
    all_pids+=("$pid")
    seen=$seen$pid$'\n'
  done
  for pid in ${serial_relays[@]+"${serial_relays[@]}"}; do
    case $seen in
      *$'\n'"$pid"$'\n'*) ;;
      *) all_pids+=("$pid"); seen=$seen$pid$'\n' ;;
    esac
  done
  [ ${#to_kill[@]} -eq 0 ] || kill_serial_relays "${to_kill[@]}"
  for pid in ${all_pids[@]+"${all_pids[@]}"}; do
    wait "$pid" 2>/dev/null || :
  done
  serial_relays=()
}
# The signals the parallel run relays are pinned at their default before the
# EXIT trap (gh-ocannl-2332): `trap ''` records the default as each one's
# original, and the reset leaves it untrapped, so a member forked as one
# arrives dies on delivery rather than outliving the run, while an untrapped
# signal still runs the EXIT trap at once. The parallel run ends its relay
# with exit traps, not a reset, which would restore that default and skip the
# EXIT trap. Do not write `$slot_record` here -- only `exit_with` records it
# (gh-ocannl-2450), because an untrapped TERM/HUP runs this trap with the
# previous command's `$?` before Bash re-raises the signal.
trap '' HUP INT TERM
trap - HUP INT TERM
print_timings() {
  [ "${timings:-0}" = 1 ] && [ "${timings_printed:-0}" = 0 ] || return 0
  local i any=0 n=${total:-0}
  for ((i = 0; i < n; i++)); do
    if [ -n "${ended[i]-}" ] || [ "${serial_running_idx:-}" = "$i" ]; then
      any=1; break
    fi
  done
  [ "$any" = 1 ] || return 0
  timings_printed=1
  printf '\nharnesses: timings:\n'
  for ((i = 0; i < n; i++)); do
    if [ -n "${ended[i]-}" ]; then
      printf '  HARNESS %s finished in %ss (exit %s)\n' "${run_commands[i]}" "${member_times[i]}" "${ended[i]}"
    elif [ "${serial_running_idx:-}" = "$i" ]; then
      printf '  HARNESS %s interrupted after %ss (in flight)\n' "${run_commands[i]}" "$((SECONDS - serial_running_started))"
    fi
  done
}
cleanup_harnesses() {
  stop_serial_relays
  print_timings
  rm -rf ${scratch[@]+"${scratch[@]}"}
}
trap cleanup_harnesses EXIT
# Refuse a list (exit 2, before any member runs) holding any line but the
# three forms the header names, or an entry naming a harness this run does not
# select, or one listed twice.
validate_list() { # FILE OPTION
  local selected n=0 seen=$'\n' line why
  selected=$(members)
  while IFS= read -r line || [ -n "$line" ]; do
    n=$((n + 1))
    case $line in
      ''|'#'*) continue ;;
      *$'\r'*) why='a carriage return' ;;
      *$'\t'*$'\t'*) why='more than one tab' ;;
      ?*$'\t'?*)
        why=
        case $'\n'$selected$'\n' in
          *$'\n'"${line%%$'\t'*}"$'\n'*) ;;
          *) why="a harness this run does not select (${line%%$'\t'*})" ;;
        esac
        case $seen in *$'\n'"$line"$'\n'*) why='an entry listed twice' ;; esac
        seen=$seen$line$'\n' ;;
      *) why='neither a comment nor HARNESS<TAB>LABEL' ;;
    esac
    if [ -n "$why" ]; then
      echo "test-harnesses.sh: $2 $1:$n: $why" >&2
      exit_with 2
    fi
  done <"$1"
}
readable_lists() { # OPTION FILE...
  local option=$1 list_file
  shift
  for list_file in "$@"; do
    [ -f "$list_file" ] && [ -r "$list_file" ] ||
      { echo "test-harnesses.sh: $option: cannot read $list_file" >&2; exit_with 2; }
  done
}
dune_list=$root/tools/harness-skips-ci-shell-dune.txt
if [ "$ci_shell" = 1 ]; then
  # The job's list, the hidden dune's, and on macOS that host's own
  # skips, as ONE list validated and checked like any other: its line numbers
  # run on through the later files.
  lists=("$root/tools/harness-skips-ci-shell.txt" "$dune_list")
  [ "$(uname -s 2>/dev/null)" != Darwin ] || lists+=("$root/tools/harness-skips-ci-shell-darwin.txt")
  readable_lists --ci-shell "${lists[@]}"
  joined=$(mktemp -d "${TMPDIR:-/tmp}/test-harnesses-ci-shell.XXXXXX") ||
    { echo "test-harnesses.sh: cannot create the --ci-shell list" >&2; exit_with 2; }
  scratch+=("$joined")
  for list_file in "${lists[@]}"; do cat "$list_file"; echo; done >"$joined/harness-skips-ci-shell+.txt"
  expected=$joined/harness-skips-ci-shell+.txt
fi
if [ "$ci_real_dune" = 1 ]; then
  readable_lists --ci-real-dune "$dune_list"
  validate_list "$dune_list" --ci-real-dune
  # Every member the dune list names has its selector, and every selector's
  # member is one this run selects: a member named without one would leave
  # its checks unrun.
  with_legs=$'\n'$(real_dune_legs | cut -f1)$'\n' shell_members=$'\n'$(members)$'\n'
  while IFS= read -r line || [ -n "$line" ]; do
    case $line in ''|'#'*) continue ;; esac
    case $with_legs in *$'\n'"${line%%$'\t'*}"$'\n'*) continue ;; esac
    echo "test-harnesses.sh: --ci-real-dune: no leg selector for ${line%%$'\t'*}, which $dune_list names" >&2
    exit_with 2
  done <"$dune_list"
  while IFS=$'\t' read -r script _; do
    case $shell_members in *$'\n'"$script"$'\n'*) continue ;; esac
    echo "test-harnesses.sh: --ci-real-dune: $script has a leg selector and is no shell-tier member" >&2
    exit_with 2
  done < <(real_dune_legs)
fi
if [ -n "$expected" ]; then
  case $expected in /*) ;; *) expected=$PWD/$expected ;; esac
  [ -f "$expected" ] && [ -r "$expected" ] ||
    { echo "test-harnesses.sh: --expected-skips: cannot read $expected" >&2; exit_with 2; }
  validate_list "$expected" --expected-skips
fi
# --affected's commands (see the header) as `HARNESS<TAB>ERE` lines in
# affected_table, ERE empty for a member reached whole. Asked of this checkout
# whatever the caller's directory: test-run.sh goes to its own root. The copy
# in the slot asks again, for a slot wait is minutes in which the change can
# move; what reached what was already said, so only a failure speaks there.
affected_table=
read_affected() {
  local dir rc=0 line n=0 seen=$'\n' every selected reached ask
  # The two forms `affected --harnesses` prints (tools/affected-tests.sh).
  local form=$'^([^ \t\r\']+)( --only \'([^\t\r\']+)\')?$'
  dir=$(mktemp -d "${TMPDIR:-/tmp}/test-harnesses-affected.XXXXXX") ||
    { echo "test-harnesses.sh: cannot create the --affected scratch directory" >&2; exit_with 2; }
  scratch+=("$dir")
  ask=(--since "$since")
  [ ${#files[@]} = 0 ] || ask=(--files "${files[@]}")
  "${BASH:-bash}" "$root/tools/test-run.sh" affected --harnesses "${ask[@]}" >"$dir/commands" 2>"$dir/err" || rc=$?
  if [ "$rc" != 0 ]; then
    cat "$dir/err" >&2
    echo "test-harnesses.sh: --affected: affected --harnesses ${ask[*]} exited $rc (above), so what this change reaches is not known; nothing run" >&2
    exit_with 2
  fi
  [ -n "$slot_record" ] || cat "$dir/err" >&2
  # A line is judged against every member: a group only keeps some of the good ones.
  every=$'\n'$(group=all members)$'\n'
  selected=$'\n'$(members)$'\n'
  affected_table=$dir/table
  while IFS= read -r line || [ -n "$line" ]; do
    n=$((n + 1))
    reached=
    if [[ $line =~ $form ]]; then
      reached=${BASH_REMATCH[1]}
      case $selected in
        *$'\n'"$reached"$'\n'*) printf '%s\t%s\n' "$reached" "${BASH_REMATCH[3]}" ;;
      esac
      case $every in *$'\n'"$reached"$'\n'*) ;; *) reached= ;; esac
      case $seen in *$'\n'"$reached"$'\n'*) reached= ;; esac
      seen=$seen$reached$'\n'
    fi
    if [ -z "$reached" ]; then
      echo "test-harnesses.sh: --affected: line $n of what affected --harnesses printed is no command of a member, once: $line" >&2
      exit_with 2
    fi
  done <"$dir/commands" >"$affected_table"
}
[ "$affected" = 0 ] || read_affected
explicit_set=$'\n'
if [ ${#explicit_members[@]} -gt 0 ]; then
  every_members=$'\n'$(group=all members)$'\n'
  for m in "${explicit_members[@]}"; do
    case $m in
      '' | *$'\n'* | *$'\t'* | *$'\r'*)
        echo "test-harnesses.sh: --member: invalid harness name '$m'" >&2
        exit_with 2 ;;
    esac
    case $every_members in
      *$'\n'"$m"$'\n'*) ;;
      *) echo "test-harnesses.sh: --member: '$m' is not a member of the manifest" >&2; exit_with 2 ;;
    esac
    case $explicit_set in
      *$'\n'"$m"$'\n'*) echo "test-harnesses.sh: --member: '$m' listed more than once" >&2; exit_with 2 ;;
    esac
    explicit_set=$explicit_set$m$'\n'
  done
fi
if [ "$affected" = 1 ] && [ ! -s "$affected_table" ]; then
  if [ "$list" = 0 ]; then
    if [ "$group" = all ]; then what='no harness'; else what="no harness of the $group group"; fi
    printf '\nharnesses: 0 run, 0 failed (this change reaches %s)\n' "$what"
  fi
  exit_with 0
fi
# On macOS, pin `awk` to `mawk` (gh-ocannl-2414) so harnesses exercise Linux
# mawk semantics instead of macOS's BWK `/usr/bin/awk`. Checked before taking a
# fleet slot so a missing `mawk` refuses immediately without waiting on a slot.
darwin_mawk_fallbacks=(/opt/homebrew/bin/mawk /usr/local/bin/mawk)
member_awk=
resolve_darwin_awk() {
  local cand dir
  [ "$(uname -s 2>/dev/null)" = Darwin ] || return 0
  cand=$(type -P mawk 2>/dev/null || :)
  if [ -n "$cand" ] && [ -f "$cand" ] && [ -x "$cand" ]; then
    case $cand in
      /*) member_awk=$cand ;;
      *)
        dir=$(CDPATH= cd -- "$(dirname -- "$cand")" && pwd) || exit_with 2
        member_awk=$dir/${cand##*/} ;;
    esac
    return 0
  fi
  for cand in ${darwin_mawk_fallbacks[@]+"${darwin_mawk_fallbacks[@]}"}; do
    if [ -f "$cand" ] && [ -x "$cand" ]; then
      member_awk=$cand
      return 0
    fi
  done
  echo "test-harnesses.sh: macOS requires mawk to pin awk for shell harnesses (macOS /usr/bin/awk is BWK awk and hides Linux mawk defects; no fallback). Install it with: brew install mawk" >&2
  exit_with 2
}
[ "$list" = 1 ] || resolve_darwin_awk
# The fleet's run-time correctness slot (see the header): the whole run, this
# script again, inside one `execution slot --cpu`. Taken here, after every
# refusal above and before anything below is exported -- the fleet-worker.sh
# runs with the caller's environment, FLEET_BOXES included.
take_fleet_slot() {
  local candidates slot_dir
  candidates=$root/tools/fleet-worker-candidates.sh
  [ -r "$candidates" ] || { echo "test-harnesses.sh: cannot read $candidates" >&2; exit_with 2; }
  # shellcheck source=fleet-worker-candidates.sh
  . "$candidates"
  slot_dir=$(mktemp -d "${TMPDIR:-/tmp}/test-harnesses-slot.XXXXXX") ||
    { echo "test-harnesses.sh: cannot create the slot's scratch directory" >&2; exit_with 2; }
  scratch+=("$slot_dir")
  # The script by its absolute path: the copy runs from wherever the caller is.
  fleet_slot_rerun OCANNL_TOOL_HARNESSES_IN_SLOT "$slot_dir" harnesses "the harnesses" -- \
    "${BASH:-bash}" "$root/tools/$(basename -- "$0")" "${original_args[@]+"${original_args[@]}"}"
}
[ "$list" = 1 ] || [ -n "$slot_record" ] || take_fleet_slot
timing_file=${OCANNL_TOOL_HARNESS_TIMINGS_FILE:-}
unset OCANNL_TOOL_HARNESS_TIMINGS_FILE
if [ -n "$expected" ]; then
  record=$(mktemp "${TMPDIR:-/tmp}/test-harnesses-skips.XXXXXX") ||
    { echo "test-harnesses.sh: cannot create the skip record" >&2; exit_with 2; }
  scratch+=("$record")
  export OCANNL_TOOL_HARNESS_STRICT=1 OCANNL_TOOL_HARNESS_EXPECTED_SKIPS=$expected \
    OCANNL_TOOL_HARNESS_SKIP_RECORD=$record
fi
if [ "$ci_shell" = 1 ]; then
  unset FLEET_BOXES
  export OCANNL_TOOL_HARNESS_NO_DUNE=1 OCANNL_TOOL_SWEEP_WAKE_LAB=$record.no-wake-lab \
    OCANNL_TOOL_HARNESS_DUNE_SKIP_RECORD=$record.dune
  : >"$record.dune"
  scratch+=("$record.dune")
fi
if [ "$ci_real_dune" = 1 ]; then
  export OCANNL_TOOL_HARNESS_STRICT=1
fi
failures=0 total=0
cd "$root" || exit_with 2
# The members this run selects, in the manifest's order: its script, the
# command --list prints for it, and its arguments as one --only selector
# (empty: every leg).
run_scripts=() run_commands=() run_selectors=() run_tiers=()
while read -r tier script; do
  [ "$group" = all ] || [ "$tier" = "$group" ] || continue
  selector=
  if [ "$ci_real_dune" = 1 ]; then
    selector=$(real_dune_legs | awk -F '\t' -v s="$script" '$1 == s { print $2 }')
    [ -n "$selector" ] || continue
  fi
  command=$script
  if [ ${#explicit_members[@]} -gt 0 ]; then
    case $explicit_set in
      *$'\n'"$script"$'\n'*) ;;
      *) continue ;;
    esac
  fi
  if [ "$affected" = 1 ]; then
    # `+` tells a member reached whole from one no line names.
    reached=$(awk -F '\t' -v s="$script" '$1 == s { print "+" $2 }' "$affected_table")
    [ -n "$reached" ] || continue
    selector=${reached#+}
    [ -z "$selector" ] || command="$script --only '$selector'"
  fi
  total=$((total + 1))
  if [ "$list" = 1 ]; then
    printf '%s\n' "$command"
    continue
  fi
  run_scripts+=("$script") run_commands+=("$command") run_selectors+=("$selector") run_tiers+=("$tier")
done < <(manifest)
[ "$total" -gt 0 ] || { echo "no harnesses selected" >&2; exit_with 2; }
[ "$list" = 1 ] && exit_with 0
# Pin both member interpreters and nested subjects, including env shebangs.
member_bash=${BASH:-bash}
if /bin/bash -c '[ "${BASH_VERSINFO[0]}.${BASH_VERSINFO[1]}" = 3.2 ]' 2>/dev/null; then
  member_bash=/bin/bash
fi
member_version=$("$member_bash" -c 'printf "%s.%s" "${BASH_VERSINFO[0]}" "${BASH_VERSINFO[1]}"') || exit_with 2
member_banner="$member_version $member_bash${member_awk:+; awk $member_awk}"
interpreter_dir=$(mktemp -d "${TMPDIR:-/tmp}/test-harnesses-bash.XXXXXX") || exit_with 2
scratch+=("$interpreter_dir")
ln -s "$member_bash" "$interpreter_dir/bash" || exit_with 2
if [ -n "$member_awk" ]; then
  ln -s "$member_awk" "$interpreter_dir/awk" || exit_with 2
fi
export PATH="$interpreter_dir:$PATH"
outputs=$(mktemp -d "${TMPDIR:-/tmp}/test-harnesses-output.XXXXXX") ||
  { echo "test-harnesses.sh: cannot create the member output directory" >&2; exit_with 2; }
scratch+=("$outputs")
member_output() { # SCRIPT: sets output
  output=$outputs/${1//\//_}.out
}
command_not_found='^(.+: line [0-9]+: .+|[^:]+: [^:]+): command not found$'
capture_status() { # SCRIPT COMMAND_RC RELAY_RC
  if [ "$3" != 0 ]; then
    printf 'harnesses: %s: could not relay member output (relay exit %s)\n' "$1" "$3" >&2
    [ "$2" != 0 ] || return 1
  fi
  return "$2"
}
# Stream a growing capture file to stdout until $done appears, then drain up to
# the file's size when $done was observed and exit. Using a regular file keeps
# verdict capture complete even if downstream blocks or a descendant holds the
# descriptor open after the member exits.
serial_relay_perl='
  my ($path, $done) = @ARGV;
  open my $fh, "<", $path or exit 1;
  binmode $fh;
  binmode STDOUT;
  my ($buf, $pos, $limit) = ("", 0, undef);
  while (1) {
    if (!defined $limit && -e $done) {
      $limit = -s $path;
      $limit = 0 if !defined $limit;
    }
    my $want = 8192;
    if (defined $limit) {
      $want = $limit - $pos;
      last if $want <= 0;
      $want = 8192 if $want > 8192;
    }
    my $n = sysread($fh, $buf, $want);
    exit 1 if !defined $n;
    if ($n > 0) {
      $pos += $n;
      my $off = 0;
      while ($off < $n) {
        my $w = syswrite(STDOUT, $buf, $n - $off, $off);
        exit 1 if !defined $w;
        $off += $w;
      }
      next;
    }
    last if defined $limit;
    select undef, undef, undef, 0.02;
  }
'
drain_serial_relays() { # SCRIPT: sets serial_relay_rc to first unexpected non-zero relay status
  local script=$1 ticks=0 live= still=0 cut=0 pid to_kill=() stopped=$'\n' relay_rc
  serial_relay_rc=0
  [ ${#serial_relays[@]} -gt 0 ] || return 0
  while :; do
    serial_jobs_live
    still=0
    for pid in ${serial_relays[@]+"${serial_relays[@]}"}; do
      if in_live "$pid"; then still=1; break; fi
    done
    [ "$still" = 0 ] && break
    [ "$ticks" -ge "$serial_drain_ticks" ] && break
    ticks=$((ticks + 1))
    sleep 0.05
  done
  if [ "$still" = 1 ]; then
    for pid in ${serial_relays[@]+"${serial_relays[@]}"}; do
      if in_live "$pid"; then
        to_kill+=("$pid")
        stopped=$stopped$pid$'\n'
      fi
    done
    [ ${#to_kill[@]} -eq 0 ] || kill_serial_relays "${to_kill[@]}"
  fi
  for pid in ${serial_relays[@]+"${serial_relays[@]}"}; do
    relay_rc=0
    wait "$pid" 2>/dev/null || relay_rc=$?
    case $stopped in
      *$'\n'"$pid"$'\n'*)
        case $relay_rc in 137|143) relay_rc=0; cut=1 ;; esac ;;
    esac
    [ "$relay_rc" = 0 ] || [ "$serial_relay_rc" != 0 ] || serial_relay_rc=$relay_rc
  done
  if [ "$cut" = 1 ]; then
    printf '\nharnesses: %s: live output cut after the drain bound; the verdict read the full capture\n' "$script" >&2
  fi
  serial_relays=()
}
serial_member() { # SCRIPT STDOUT STDERR [ARGS...] (STDOUT empty unless --ci-real-dune)
  local script=$1 out=$2 error=$3 rc=0 done_flag=$3.done
  shift 3
  rm -f "$done_flag"
  : >"$error" || {
    echo 'test-harnesses.sh: cannot create the live stderr capture file' >&2
    exit_with 2
  }
  if [ -n "$out" ]; then
    : >"$out" || {
      echo 'test-harnesses.sh: cannot create the live stdout capture file' >&2
      exit_with 2
    }
  fi
  perl -e "$serial_relay_perl" "$error" "$done_flag" >&2 &
  serial_relays=("$!")
  if [ -n "$out" ]; then
    perl -e "$serial_relay_perl" "$out" "$done_flag" &
    serial_relays+=("$!")
  fi
  local started_at=$SECONDS
  serial_running_started=$started_at
  if [ -n "$out" ]; then
    OCANNL_TOOL_HARNESS_NAME=$script "$member_bash" "$root/$script" "$@" \
      >"$out" 2>"$error" || rc=$?
  else
    OCANNL_TOOL_HARNESS_NAME=$script "$member_bash" "$root/$script" "$@" \
      2>"$error" || rc=$?
  fi
  member_elapsed=$((SECONDS - started_at))
  # Record observed exit status and elapsed time before draining relays so an
  # interruption during relay drain still reports this member in --timings;
  # member_ended updates ended[i] if verdict checks change the final status.
  [ -z "${serial_running_idx:-}" ] || ended[$serial_running_idx]=$rc member_times[$serial_running_idx]=$member_elapsed
  serial_running_idx=
  case $rc in
    # A descendant may outlive the interrupted member with stderr or stdout open.
    129|130|143)
      stop_serial_relays
      rm -f "$done_flag"
      return "$rc" ;;
  esac
  : >"$done_flag" || :
  drain_serial_relays "$script"
  rm -f "$done_flag"
  capture_status "$script" "$rc" "$serial_relay_rc"
}
member_timing() { # INDEX RC ELAPSED
  [ -n "$timing_file" ] || return 0
  # The optional timing sink cannot add a diagnostic to a shared-red log.
  { printf 'HARNESS %s finished in %ss (exit %s)\n' "${run_commands[$1]}" "$3" "$2" >>"$timing_file"; } 2>/dev/null || :
}
member_ended() { # INDEX RC OUTPUT ELAPSED: ordinary failures count once per member
  local script=${run_scripts[$1]} rc=$2 diagnostic scan_rc=0
  case $rc in
    129|130|143)
      printf 'HARNESS %s exit: %s\n' "$script" "$rc"
      ended[$1]=$rc member_times[$1]=$4
      member_timing "$1" "$rc" "$4"
      stop_members
      print_timings
      printf 'harnesses: interrupted (exit %s)\n' "$rc" >&2
      exit_with "$rc" ;;
  esac
  diagnostic=$(env LC_ALL=C grep -a -m 1 -E -- "$command_not_found" "$3") || scan_rc=$?
  case $scan_rc in
    0) printf 'harnesses: %s: missing command: %s\n' "$script" "$diagnostic" >&2 ;;
    1) ;;
    *) printf 'harnesses: %s: could not check member output (grep exit %s)\n' "$script" "$scan_rc" >&2 ;;
  esac
  [ "$scan_rc" = 1 ] || [ "$rc" != 0 ] || rc=1
  printf 'HARNESS %s exit: %s\n' "$script" "$rc"
  if [ "$jobs" = 1 ]; then
    ended[$1]=$rc member_times[$1]=$4
    member_timing "$1" "$rc" "$4"
  else
    ended[$1]=$rc
  fi
  [ "$rc" = 0 ] || failures=$((failures + 1))
}
pids=() ended=() member_started=() member_times=()
stop_members() { # TERM every member still running beside the others, and wait for each
  local i rc
  for ((i = 0; i < ${#pids[@]}; i++)); do
    [ -z "${ended[i]-}" ] || continue
    kill -TERM "${pids[i]}" 2>/dev/null
    rc=0
    wait "${pids[i]}" 2>/dev/null || rc=$?
    ended[i]=$rc member_times[i]=$((SECONDS - member_started[i]))
  done
}
record_parallel_timings() {
  local i
  [ -n "$timing_file" ] || return 0
  for ((i = 0; i < ${#pids[@]}; i++)); do
    [ -n "${ended[i]-}" ] || continue
    member_timing "$i" "${ended[i]}" "${member_times[i]}"
  done
}
member_argv() { # INDEX: sets args
  args=()
  [ -z "${run_selectors[$1]}" ] || args=(--only "${run_selectors[$1]}")
}
if [ "$jobs" = 1 ]; then
  for ((i = 0; i < total; i++)); do
    script=${run_scripts[i]}
    member_argv "$i"
    printf '\nHARNESS %s (%s; bash %s)\n' "${run_commands[i]}" "${run_tiers[i]}" "$member_banner"
    member_output "$script"
    rc=0
    serial_running_idx=$i serial_running_started=$SECONDS
    if [ "$ci_real_dune" = 1 ]; then
      serial_member "$script" "$output" "$output.err" ${args[@]+"${args[@]}"} || rc=$?
    else
      serial_member "$script" '' "$output.err" ${args[@]+"${args[@]}"} || rc=$?
    fi
    serial_running_idx=
    member_ended "$i" "$rc" "$output.err" "$member_elapsed"
  done
else
  # Up to $jobs members at once (see the header), each this shell's own
  # child, so its status is this shell's to wait for. No `wait -n`: this runs
  # under the Bash 3.2 a Mac ships, so a member's end is seen by polling its
  # pid, which the shell has reaped by then.
  # A completed member can still be waiting for its turn to print.
  trap 'record_parallel_timings; cleanup_harnesses' EXIT
  trap 'stop_members; exit_with 129' HUP
  trap 'stop_members; exit_with 130' INT
  trap 'stop_members; exit_with 143' TERM
  started=0 printed=0 running=0 interrupted=
  while [ "$printed" -lt "$total" ] && [ -z "$interrupted" ]; do
    progressed=0
    for ((i = printed; i < started; i++)); do
      [ -z "${ended[i]-}" ] && ! kill -0 "${pids[i]}" 2>/dev/null || continue
      rc=0
      wait "${pids[i]}" || rc=$?
      member_times[i]=$((SECONDS - member_started[i]))
      ended[i]=$rc running=$((running - 1)) progressed=1
      # Seen as it ends, not when its output comes up: no member starts after it.
      case $rc in 129|130|143) interrupted=$rc; break ;; esac
    done
    [ -z "$interrupted" ] || break
    while [ "$started" -lt "$total" ] && [ "$running" -lt "$jobs" ]; do
      script=${run_scripts[started]}
      member_output "$script"
      member_argv "$started"
      member_started[started]=$SECONDS
      OCANNL_TOOL_HARNESS_NAME=$script "$member_bash" "$root/$script" ${args[@]+"${args[@]}"} \
        >"$output" 2>"$output.err" </dev/null &
      pids[started]=$!
      started=$((started + 1)) running=$((running + 1)) progressed=1
    done
    while [ "$printed" -lt "$started" ] && [ -n "${ended[printed]-}" ]; do
      script=${run_scripts[printed]}
      member_output "$script"
      printf '\nHARNESS %s (%s; bash %s)\n' "${run_commands[printed]}" "${run_tiers[printed]}" "$member_banner"
      cat "$output" "$output.err"
      member_ended "$printed" "${ended[printed]}" "$output.err" "${member_times[printed]}"
      printed=$((printed + 1)) progressed=1
    done
    [ "$progressed" = 1 ] || sleep 0.2
  done
  trap 'exit_with 129' HUP
  trap 'exit_with 130' INT
  trap 'exit_with 143' TERM
  if [ -n "$interrupted" ]; then
    # The members beside it are stopped, then every member that ran is
    # printed, in order, before the run ends with the interrupting status.
    stop_members
    for ((i = printed; i < started; i++)); do
      script=${run_scripts[i]}
      member_output "$script"
      printf '\nHARNESS %s (%s; bash %s)\n' "${run_commands[i]}" "${run_tiers[i]}" "$member_banner"
      cat "$output" "$output.err"
      printf 'HARNESS %s exit: %s\n' "$script" "${ended[i]}"
    done
    print_timings
    printf 'harnesses: interrupted (exit %s)\n' "$interrupted" >&2
    exit_with "$interrupted"
  fi
fi
stale=0
if [ -n "$expected" ]; then
  # A listed skip that did not occur is coverage the list no longer
  # describes: red, so the list is pruned when a platform gains a leg.
  while IFS= read -r line || [ -n "$line" ]; do
    case $line in ''|'#'*) continue ;; esac
    grep -qxF -- "$line" "$record" && continue
    printf 'harnesses: expected skip did not occur: %s: %s\n' "${line%%$'\t'*}" "${line#*$'\t'}"
    stale=$((stale + 1))
  done <"$expected"
fi
misfiled=0
if [ "$ci_shell" = 1 ]; then
  # The dune list is exactly the skips the hidden dune caused: one of those
  # listed elsewhere would escape --ci-real-dune, and one listed there that
  # skipped for another reason would be run per PR for the wrong cause.
  while IFS= read -r line || [ -n "$line" ]; do
    grep -qxF -- "$line" "$dune_list" && continue
    printf 'harnesses: a skip of the hidden dune, listed outside %s: %s: %s\n' \
      "${dune_list##*/}" "${line%%$'\t'*}" "${line#*$'\t'}"
    misfiled=$((misfiled + 1))
  done < <(sort -u "$record.dune")
  while IFS= read -r line || [ -n "$line" ]; do
    case $line in ''|'#'*) continue ;; esac
    grep -qxF -- "$line" "$record" || continue # did not occur: counted above
    grep -qxF -- "$line" "$record.dune" && continue
    printf 'harnesses: listed in %s, and skipped for another reason than the hidden dune: %s: %s\n' \
      "${dune_list##*/}" "${line%%$'\t'*}" "${line#*$'\t'}"
    misfiled=$((misfiled + 1))
  done <"$dune_list"
fi
unreached=0
if [ "$ci_real_dune" = 1 ]; then
  # Every check the dune list names ran against the real dune and passed.
  while IFS= read -r line || [ -n "$line" ]; do
    case $line in ''|'#'*) continue ;; esac
    script=${line%%$'\t'*}
    member_output "$script"
    grep -qxF -- "PASS  ${line#*$'\t'}" "$output" 2>/dev/null && continue
    printf 'harnesses: a real-dune check did not PASS (unreached by its selector, skipped or red): %s: %s\n' \
      "$script" "${line#*$'\t'}"
    unreached=$((unreached + 1))
  done <"$dune_list"
fi
print_timings
printf '\nharnesses: %s run, %s failed' "$total" "$failures"
[ -z "$expected" ] || printf '; %s expected skip(s) did not occur' "$stale"
[ "$ci_shell" = 0 ] || printf '; %s hidden-dune skip(s) misfiled' "$misfiled"
[ "$ci_real_dune" = 0 ] || printf '; %s real-dune check(s) did not pass' "$unreached"
printf '\n'
exit_with $((failures + stale + misfiled + unreached > 0 ? 1 : 0))
