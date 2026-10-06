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

set -uo pipefail

# Credentials never reach dune, which records every spawned process's environment in the
# worktree's `_build/trace.csexp` (gh-ocannl-1280): the deny-list tools/test-run.sh applies.
credential_env=$(dirname "$0")/credential-env.sh
[ -r "$credential_env" ] || { echo "fmt-check: FAILED (exit 2): cannot read $credential_env"; exit 2; }
# shellcheck source=credential-env.sh
. "$credential_env"
eval "$(credential_env_scrub_text)" || {
  echo "fmt-check: FAILED (exit 2): cannot remove credential variables:$credential_env_left"
  exit 2
}

if [ "$#" -eq 0 ]; then
  set -- opam exec -- dune build @fmt --force
fi

verdict() { # STATUS [REASON]
  if [ "$1" -eq 0 ]; then
    echo "fmt-check: PASSED"
  else
    echo "fmt-check: FAILED (exit $1)${2:+: $2}"
  fi
  exit "$1"
}

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

verdict 0
