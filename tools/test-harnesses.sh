#!/usr/bin/env bash
# Run the shell-harness tier CI runs, from any working directory.
# Usage: tools/test-harnesses.sh [--shell|--toolchain|--promotion] [--list]
#                               [--expected-skips FILE]
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
set -u
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd) || exit 2
group=all list=0 expected=
while [ $# -gt 0 ]; do
  arg=$1
  case $arg in
    --shell|--toolchain|--promotion)
      [ "$group" = all ] || { echo "choose one harness group" >&2; exit 2; }
      group=${arg#--} ;;
    --list) list=1 ;;
    --expected-skips)
      [ $# -ge 2 ] && [ -n "$2" ] || { echo "test-harnesses.sh: --expected-skips needs a file" >&2; exit 2; }
      expected=$2
      shift ;;
    -h|--help) sed -n '2,${/^#/!q;p;}' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "test-harnesses.sh: unknown argument '$arg'" >&2; exit 2 ;;
  esac
  shift
done
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
members() {
  local tier script
  while read -r tier script; do
    [ "$group" = all ] || [ "$tier" = "$group" ] || continue
    printf '%s\n' "$script"
  done < <(manifest)
}
unset OCANNL_TOOL_HARNESS_EXPECTED_SKIPS OCANNL_TOOL_HARNESS_SKIP_RECORD
if [ -n "$expected" ]; then
  case $expected in /*) ;; *) expected=$PWD/$expected ;; esac
  [ -f "$expected" ] && [ -r "$expected" ] ||
    { echo "test-harnesses.sh: --expected-skips: cannot read $expected" >&2; exit 2; }
  selected=$(members)
  n=0 seen=$'\n'
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
      echo "test-harnesses.sh: --expected-skips $expected:$n: $why" >&2
      exit 2
    fi
  done <"$expected"
  record=$(mktemp "${TMPDIR:-/tmp}/test-harnesses-skips.XXXXXX") ||
    { echo "test-harnesses.sh: cannot create the skip record" >&2; exit 2; }
  trap 'rm -f "$record"' EXIT
  export OCANNL_TOOL_HARNESS_STRICT=1 OCANNL_TOOL_HARNESS_EXPECTED_SKIPS=$expected \
    OCANNL_TOOL_HARNESS_SKIP_RECORD=$record
fi
failures=0 total=0
cd "$root" || exit 2
while read -r tier script; do
  [ "$group" = all ] || [ "$tier" = "$group" ] || continue
  total=$((total + 1))
  if [ "$list" = 1 ]; then
    printf '%s\n' "$script"
    continue
  fi
  printf '\nHARNESS %s (%s)\n' "$script" "$tier"
  rc=0
  OCANNL_TOOL_HARNESS_NAME=$script "$root/$script" || rc=$?
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
printf '\nharnesses: %s run, %s failed' "$total" "$failures"
[ -z "$expected" ] || printf '; %s expected skip(s) did not occur' "$stale"
printf '\n'
exit $((failures + stale > 0 ? 1 : 0))
