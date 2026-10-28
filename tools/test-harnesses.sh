#!/usr/bin/env bash
# Run the shell-harness tier CI runs, from any working directory.
# Usage: tools/test-harnesses.sh [--shell|--toolchain|--promotion] [--list]
#                               [--expected-skips FILE]
#        tools/test-harnesses.sh --ci-shell [--list]
#        tools/test-harnesses.sh --ci-real-dune [--list]
# With no group, run all three. --shell needs no OCaml toolchain; --toolchain
# runs the formatter controls and the scan-case type check (which links
# ppxlib) and --promotion the controls of tools that read
# Dune's own output (promotion, affected tests, the failed-rule listing), whose
# missing-toolchain legs report counted skips. CI runs both after installing
# its toolchain, and --promotion again at the declared Dune floor.
# Every selected harness runs after ordinary failures; any failure exits 1.
# HUP/INT/TERM stop immediately and preserve the interruption status.
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
# relaxes a strict run.
#
# --ci-shell runs the shell tier as the toolchain-free `harnesses` CI job
# does, which is the job's own command, so any Linux or macOS box reproduces
# its skip check before a push (gh-ocannl-1593): --shell --expected-skips
# tools/harness-skips-ci-shell.txt joined with
# tools/harness-skips-ci-shell-dune.txt, with that job's absences put on: dune
# hidden from every real-dune probe (OCANNL_TOOL_HARNESS_NO_DUNE=1,
# scripts/harness-support.sh's harness_real_dune), FLEET_BOXES unset and the
# fleet anchor's wake-lab.sh pointed away (OCANNL_TOOL_SWEEP_WAKE_LAB). On
# macOS the further skips of Linux-only legs that
# tools/harness-skips-ci-shell-darwin.txt lists are expected too, under the
# same rules. The dune list is held to exactly the skips the hidden dune
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
set -u
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd) || exit 2
group=all list=0 expected= ci_shell=0 ci_real_dune=0
while [ $# -gt 0 ]; do
  arg=$1
  case $arg in
    --shell|--toolchain|--promotion)
      [ "$group" = all ] || { echo "choose one harness group" >&2; exit 2; }
      group=${arg#--} ;;
    --list) list=1 ;;
    --ci-shell) ci_shell=1 ;;
    --ci-real-dune) ci_real_dune=1 ;;
    --expected-skips)
      [ $# -ge 2 ] && [ -n "$2" ] || { echo "test-harnesses.sh: --expected-skips needs a file" >&2; exit 2; }
      expected=$2
      shift ;;
    -h|--help) sed -n '2,${/^#/!q;p;}' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "test-harnesses.sh: unknown argument '$arg'" >&2; exit 2 ;;
  esac
  shift
done
if [ "$ci_shell" = 1 ] && [ "$ci_real_dune" = 1 ]; then
  echo "test-harnesses.sh: choose one of --ci-shell and --ci-real-dune" >&2
  exit 2
fi
if [ "$ci_shell" = 1 ] || [ "$ci_real_dune" = 1 ]; then
  [ "$group" = all ] && [ -z "$expected" ] || {
    echo "test-harnesses.sh: --ci-shell and --ci-real-dune take no group and no --expected-skips" >&2
    exit 2
  }
  group=shell
fi
# This is the sole membership list: CI selects groups from this entrypoint.
manifest() {
  cat <<'HARNESS_LIST'
shell scripts/test-harness-support.sh
shell tools/test-test-harnesses.sh
shell tools/test-harness-legs-alone.sh
shell tools/test-pin-revisions.sh
shell tools/test-test-run.sh
shell tools/test-mutation-run.sh
shell scripts/test-setup-ocaml-env.sh
shell tools/test-windows-opam-cache.sh
shell tools/test-ci-times.sh
shell tools/test-ci-durations.sh
shell tools/test-ci-shard.sh
shell tools/test-action-durations.sh
shell test/operations/ci_matrix.sh
shell tools/test-machine-verify.sh
shell tools/test-merge-build-check.sh
shell tools/test-merge-build-replay.sh
shell benchmarks/test-gh1133-cells.sh
shell benchmarks/test-gh834-cells.sh
shell benchmarks/test-check-box.sh
shell benchmarks/test-box-gate.sh
toolchain tools/test-fmt-check.sh
toolchain tools/test-typecheck-scan-cases.sh
promotion tools/test-promote.sh
promotion tools/test-promotion-record.sh
promotion tools/test-affected-tests.sh
promotion tools/test-ci-failed-rules.sh
HARNESS_LIST
}
# --ci-real-dune's legs: `HARNESS<TAB>ERE`, the --only selector of the legs
# holding HARNESS's real-dune checks (each is checked to PASS, not trusted).
real_dune_legs() {
  printf '%s\t%s\n' tools/test-test-run.sh '^(79|80b|84|89|92b):'
}
members() {
  local tier script
  while read -r tier script; do
    [ "$group" = all ] || [ "$tier" = "$group" ] || continue
    printf '%s\n' "$script"
  done < <(manifest)
}
unset OCANNL_TOOL_HARNESS_EXPECTED_SKIPS OCANNL_TOOL_HARNESS_SKIP_RECORD OCANNL_TOOL_HARNESS_NO_DUNE \
  OCANNL_TOOL_HARNESS_DUNE_SKIP_RECORD
scratch=()
trap 'rm -rf ${scratch[@]+"${scratch[@]}"}' EXIT
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
      exit 2
    fi
  done <"$1"
}
readable_lists() { # OPTION FILE...
  local option=$1 list_file
  shift
  for list_file in "$@"; do
    [ -f "$list_file" ] && [ -r "$list_file" ] ||
      { echo "test-harnesses.sh: $option: cannot read $list_file" >&2; exit 2; }
  done
}
dune_list=$root/tools/harness-skips-ci-shell-dune.txt
if [ "$ci_shell" = 1 ]; then
  # The job's list, the hidden dune's, and on macOS the Linux-only legs'
  # skips, as ONE list validated and checked like any other: its line numbers
  # run on through the later files.
  lists=("$root/tools/harness-skips-ci-shell.txt" "$dune_list")
  [ "$(uname -s 2>/dev/null)" != Darwin ] || lists+=("$root/tools/harness-skips-ci-shell-darwin.txt")
  readable_lists --ci-shell "${lists[@]}"
  joined=$(mktemp -d "${TMPDIR:-/tmp}/test-harnesses-ci-shell.XXXXXX") ||
    { echo "test-harnesses.sh: cannot create the --ci-shell list" >&2; exit 2; }
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
    exit 2
  done <"$dune_list"
  while IFS=$'\t' read -r script _; do
    case $shell_members in *$'\n'"$script"$'\n'*) continue ;; esac
    echo "test-harnesses.sh: --ci-real-dune: $script has a leg selector and is no shell-tier member" >&2
    exit 2
  done < <(real_dune_legs)
fi
if [ -n "$expected" ]; then
  case $expected in /*) ;; *) expected=$PWD/$expected ;; esac
  [ -f "$expected" ] && [ -r "$expected" ] ||
    { echo "test-harnesses.sh: --expected-skips: cannot read $expected" >&2; exit 2; }
  validate_list "$expected" --expected-skips
  record=$(mktemp "${TMPDIR:-/tmp}/test-harnesses-skips.XXXXXX") ||
    { echo "test-harnesses.sh: cannot create the skip record" >&2; exit 2; }
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
  outputs=$(mktemp -d "${TMPDIR:-/tmp}/test-harnesses-real-dune.XXXXXX") ||
    { echo "test-harnesses.sh: cannot create the --ci-real-dune output directory" >&2; exit 2; }
  scratch+=("$outputs")
  export OCANNL_TOOL_HARNESS_STRICT=1
fi
failures=0 total=0
cd "$root" || exit 2
while read -r tier script; do
  [ "$group" = all ] || [ "$tier" = "$group" ] || continue
  args=()
  if [ "$ci_real_dune" = 1 ]; then
    selector=$(real_dune_legs | awk -F '\t' -v s="$script" '$1 == s { print $2 }')
    [ -n "$selector" ] || continue
    args=(--only "$selector")
  fi
  total=$((total + 1))
  if [ "$list" = 1 ]; then
    printf '%s\n' "$script"
    continue
  fi
  printf '\nHARNESS %s (%s)\n' "$script" "$tier"
  rc=0
  if [ "$ci_real_dune" = 1 ]; then
    # Kept for the PASS check below, and shown as it runs.
    OCANNL_TOOL_HARNESS_NAME=$script "$root/$script" "${args[@]}" | tee "$outputs/${script//\//_}.out"
    rc=${PIPESTATUS[0]}
  else
    OCANNL_TOOL_HARNESS_NAME=$script "$root/$script" || rc=$?
  fi
  printf 'HARNESS %s exit: %s\n' "$script" "$rc"
  case $rc in
    129|130|143)
      printf 'harnesses: interrupted (exit %s)\n' "$rc" >&2
      exit "$rc" ;;
  esac
  [ "$rc" = 0 ] || failures=$((failures + 1))
done < <(manifest)
[ "$total" -gt 0 ] || { echo "no harnesses selected" >&2; exit 2; }
[ "$list" = 1 ] && exit 0
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
    grep -qxF -- "PASS  ${line#*$'\t'}" "$outputs/${script//\//_}.out" 2>/dev/null && continue
    printf 'harnesses: a real-dune check did not PASS (unreached by its selector, skipped or red): %s: %s\n' \
      "$script" "${line#*$'\t'}"
    unreached=$((unreached + 1))
  done <"$dune_list"
fi
printf '\nharnesses: %s run, %s failed' "$total" "$failures"
[ -z "$expected" ] || printf '; %s expected skip(s) did not occur' "$stale"
[ "$ci_shell" = 0 ] || printf '; %s hidden-dune skip(s) misfiled' "$misfiled"
[ "$ci_real_dune" = 0 ] || printf '; %s real-dune check(s) did not pass' "$unreached"
printf '\n'
exit $((failures + stale + misfiled + unreached > 0 ? 1 : 0))
