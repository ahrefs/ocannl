#!/usr/bin/env bash
# Run the shell-harness tier CI runs, from any working directory.
# Usage: tools/test-harnesses.sh [--shell|--toolchain|--promotion] [--list]
# With no group, run all three. --shell needs no OCaml toolchain; --toolchain
# runs the formatter controls and --promotion the promotion controls, whose
# missing-toolchain legs report counted skips. CI runs both after installing
# its toolchain, and --promotion again at the declared Dune floor.
# Every selected harness runs after ordinary failures; any failure exits 1.
# HUP/INT/TERM stop immediately and preserve the interruption status.
set -u
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd) || exit 2
group=all list=0
for arg in "$@"; do
  case $arg in
    --shell|--toolchain|--promotion)
      [ "$group" = all ] || { echo "choose one harness group" >&2; exit 2; }
      group=${arg#--} ;;
    --list) list=1 ;;
    -h|--help) sed -n '2,${/^#/!q;p;}' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "test-harnesses.sh: unknown argument '$arg'" >&2; exit 2 ;;
  esac
done
# This is the sole membership list: CI selects groups from this entrypoint.
manifest() {
  cat <<'HARNESS_LIST'
shell scripts/test-harness-support.sh
shell tools/test-test-harnesses.sh
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
shell benchmarks/test-gh1133-cells.sh
toolchain tools/test-fmt-check.sh
promotion tools/test-promote.sh
promotion tools/test-promotion-record.sh
HARNESS_LIST
}
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
  "$root/$script" || rc=$?
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
printf '\nharnesses: %s run, %s failed\n' "$total" "$failures"
exit $((failures > 0 ? 1 : 0))
