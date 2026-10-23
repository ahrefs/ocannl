#!/usr/bin/env bash
# Type-check the OCaml snippets a scan's cases module feeds its scanner, before its arms are trusted
# (docs/agent-notes/build-and-test.md, Repository scans): the parser accepts programs the type
# checker rejects, and an arm over one pins the scanner's reading of code no compiler would build.
# Usage: tools/typecheck-scan-cases.sh [--stub STUB.mli | --no-stub] [-I DIR]... [--verbose] CASES.ml...
#        tools/typecheck-scan-cases.sh --stubbed [--verbose]
#
# Each CASES module is read by tools/scan_case_snippets.exe (built here with dune; its header is the
# contract): every {ocaml|...|ocaml} snippet in it becomes its own unit, type-checked with
# `ocamlc -stop-after typing` and reported by its case name and line -- REJECTED with the compiler's
# message, REFUSED by name when it carries project syntax (`let%op`, `[%cd ...]`, `[@@deriving]`)
# that bare ocamlc cannot expand. Snippets spelled as ordinary "..." literals are not read.
#
# The stand-in names a table's snippets call come from a stub module compiled first and OPENED at
# each snippet's top. A table's stub is its companion `scan_case_stubs/<table>.mli` beside it,
# used unless --stub names another or --no-stub drops it; the stub also lists, as
# `[@@@known_rejection ("case name", "error text")]`, the arms no compiler is meant to accept.
# --stubbed checks every table under test/operations that has a companion stub. -I adds a
# directory of compiled interfaces a snippet opens (`-I "$(opam var lib)/base"`).
#
# Exit status: the worst over the tables -- 0 clean, 1 a rejected snippet or a stale known
# rejection, 2 a usage error, a build or stub that fails, or a table in which nothing was checked.
set -u
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd) || exit 2
usage() { sed -n '5,6p' "$0" | sed 's/^# \{0,1\}//' >&2; exit 2; }
stub_mode=companion stub= stubbed=0 verbose=() includes=() tables=()
while [ $# -gt 0 ]; do
  case $1 in
    --stub)
      [ $# -ge 2 ] && [ -n "$2" ] || { echo "typecheck-scan-cases: --stub needs a file" >&2; usage; }
      stub_mode=given stub=$2
      shift ;;
    --no-stub) stub_mode=none ;;
    --stubbed) stubbed=1 ;;
    -I)
      [ $# -ge 2 ] && [ -n "$2" ] || { echo "typecheck-scan-cases: -I needs a directory" >&2; usage; }
      includes+=(-I "$2")
      shift ;;
    --verbose) verbose=(--verbose) ;;
    -h|--help) sed -n '2,${/^#/!q;p;}' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    -*) echo "typecheck-scan-cases: unknown option '$1'" >&2; usage ;;
    *) tables+=("$1") ;;
  esac
  shift
done
if [ "$stubbed" = 1 ]; then
  [ "${#tables[@]}" = 0 ] && [ "$stub_mode" = companion ] && [ "${#includes[@]}" = 0 ] || {
    echo "typecheck-scan-cases: --stubbed takes no table, stub or -I" >&2
    usage
  }
  cd "$root" || exit 2
  for companion in test/operations/scan_case_stubs/*.ml test/operations/scan_case_stubs/*.mli; do
    [ -e "$companion" ] || continue
    name=${companion##*/}
    tables+=("test/operations/${name%.*}.ml")
  done
  [ "${#tables[@]}" -gt 0 ] || { echo "typecheck-scan-cases: no companion stub found" >&2; exit 2; }
fi
[ "${#tables[@]}" -gt 0 ] || { echo "typecheck-scan-cases: no cases module given" >&2; usage; }

command -v dune >/dev/null 2>&1 || . "$root/tools/opam-env.sh" || exit 2
# Credentials never reach dune, which records every spawned process's environment in
# `_build/trace.csexp` (gh-ocannl-1280): the deny-list tools/test-run.sh applies.
# shellcheck source=credential-env.sh
. "$root/tools/credential-env.sh"
eval "$(credential_env_scrub_text)" || {
  echo "typecheck-scan-cases: cannot remove credential variables:$credential_env_left" >&2
  exit 2
}
build_log=$(mktemp "${TMPDIR:-/tmp}/typecheck-scan-cases-build.XXXXXX") || exit 2
trap 'rm -f "$build_log"' EXIT
if ! dune build --root "$root" tools/scan_case_snippets.exe >"$build_log" 2>&1; then
  cat "$build_log" >&2
  echo "typecheck-scan-cases: the engine did not build" >&2
  exit 2
fi
# Where dune just put it: DUNE_BUILD_DIR moves the build tree (as tools/batch-backends.sh reads it),
# a relative one from the workspace root.
build_dir=${DUNE_BUILD_DIR:-_build}
case $build_dir in /*) ;; *) build_dir=$root/$build_dir ;; esac
engine=$build_dir/default/tools/scan_case_snippets.exe

worst=0
for table in "${tables[@]}"; do
  stub_args=()
  case $stub_mode in
    given) stub_args=(--stub "$stub") ;;
    companion)
      dir=$(dirname -- "$table") base=$(basename -- "$table" .ml)
      for candidate in "$dir/scan_case_stubs/$base.mli" "$dir/scan_case_stubs/$base.ml"; do
        if [ -e "$candidate" ]; then stub_args=(--stub "$candidate"); break; fi
      done ;;
  esac
  printf '== %s%s\n' "$table" "${stub_args[1]:+ (stub ${stub_args[1]})}"
  rc=0
  "$engine" ${verbose[@]+"${verbose[@]}"} ${stub_args[@]+"${stub_args[@]}"} \
    ${includes[@]+"${includes[@]}"} "$table" || rc=$?
  [ "$rc" -le "$worst" ] || worst=$rc
done
exit "$worst"
