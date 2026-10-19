#!/usr/bin/env bash

# Run the formatter gate while also rejecting the invalid-odoc warnings that
# ocamlformat reports without changing its successful exit status.
#
# The default command passes `--force`: ocamlformat's warning is printed only
# when its action RUNS, and after a `dune fmt` every already-formatted file's
# action is up to date, so a plain `dune build @fmt` replays nothing and passes
# a tree that CI's fresh checkout rejects (gh-ocannl-1155). The last line of
# output is always a `fmt-check:` verdict naming the exit status, so a reader
# who piped the output through `tail` still sees whether it passed.
#
# The default command then builds the repository scans, `@test/operations/scans`
# (gh-ocannl-1321): this script is the documented last step before every push,
# so the scans run on every push without an edit first having to match a
# description of their reach -- a description that kept trailing it. They take
# seconds on a warm build, and the time is printed. Run from the repository
# root. OCANNL_TOOL_FMT_CHECK_SCANS=0 skips them, and says so: CI's `fmt` job,
# whose switch holds no project dependency (the build matrix runs the scans), or
# a formatting-only check. A command given as arguments runs alone, the scans
# not built: that is the harness's form, and a caller's own formatter.

set -uo pipefail

# Credentials never reach dune, which records every spawned process's environment in the
# worktree's `_build/trace.csexp` (gh-ocannl-1280): the deny-list tools/test-run.sh applies, to
# this script's environment and -- for the default command -- again inside the one `opam exec`
# builds, whose switch updates could set one.
credential_env=$(dirname "$0")/credential-env.sh
[ -r "$credential_env" ] || { echo "fmt-check: FAILED (exit 2): cannot read $credential_env"; exit 2; }
# shellcheck source=credential-env.sh
. "$credential_env"
eval "$(credential_env_scrub_text)" || {
  echo "fmt-check: FAILED (exit 2): cannot remove credential variables:$credential_env_left"
  exit 2
}

verdict() { # STATUS [REASON]
  if [ "$1" -eq 0 ]; then
    echo "fmt-check: PASSED${2:+ ($2)}"
  else
    echo "fmt-check: FAILED (exit $1)${2:+: $2}"
  fi
  exit "$1"
}

# Builds a command line running "$@" inside the switch, with the credentials
# removed again after `opam exec` applied the switch's environment.
in_switch() {
  scrubbed_command=(opam exec -- sh -c "{ $(credential_env_scrub_text); } || exit 2; exec \"\$@\"" \
    fmt-check "$@")
}

scans=
if [ "$#" -eq 0 ]; then
  case ${OCANNL_TOOL_FMT_CHECK_SCANS-1} in
    1) scans=yes ;;
    0) scans=skipped ;;
    *) verdict 2 "OCANNL_TOOL_FMT_CHECK_SCANS is 0 (skip the scans) or 1 (the default), not '${OCANNL_TOOL_FMT_CHECK_SCANS}'" ;;
  esac
  in_switch dune build @fmt --force
  set -- "${scrubbed_command[@]}"
fi

fmt_log=$(mktemp "${TMPDIR:-/tmp}/ocannl-fmt-check.XXXXXX") || exit 2
trap 'rm -f "$fmt_log"' EXIT

"$@" >"$fmt_log" 2>&1
fmt_status=$?
if ! cat "$fmt_log"; then
  verdict 2 "could not replay formatter output"
fi

# A real formatter failure owns the verdict, including statuses other than 1.
if [ "$fmt_status" -ne 0 ]; then
  verdict "$fmt_status" "the formatter command failed"
fi

if grep -Fq "Warning: Invalid documentation comment:" "$fmt_log"; then
  verdict 1 "invalid documentation comments are forbidden"
fi

case $scans in
  skipped) verdict 0 "repository scans skipped: OCANNL_TOOL_FMT_CHECK_SCANS=0" ;;
  yes) ;;
  *) verdict 0 ;;
esac

# Unpiped, so dune's own diff and promotion hints reach the reader as they are.
echo "fmt-check: building the repository scans (@test/operations/scans)"
scans_started=$SECONDS
in_switch dune build @test/operations/scans
"${scrubbed_command[@]}"
scans_status=$?
echo "fmt-check: the repository scans took $((SECONDS - scans_started)) s"
if [ "$scans_status" -ne 0 ]; then
  verdict "$scans_status" "the repository scans failed (dune build @test/operations/scans)"
fi

verdict 0
