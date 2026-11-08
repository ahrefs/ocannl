#!/usr/bin/env bash
# One-shot negative control: tools/mutant.sh <file> <perl-expression> <@alias>
# The expression runs as perl -0777 -pe on a private copy; a byte-identical result
# refuses before any test runs. The alias names one test with an .expected golden,
# as in mutation-run.sh. No baseline runs: establish a green baseline first.
# Reports the run and its false claims, preserving test-run.sh's status even for
# an incomplete run or a build failure; those are not proof a mutant was caught.
# Shares mutation-run.sh's cancellation/restoration traps: INT/TERM/HUP cancel
# and reap the runner before byte-identical restoration. A surviving lock or a
# failed restore exits 3 with a retained recovery copy; SIGKILL cannot be trapped.
# Use an idle isolated worktree; do not edit the file during the run.
set -eu
root=$(cd "$(dirname "$0")/.." && pwd)
[ "$#" = 3 ] || { echo 'usage: tools/mutant.sh <file> <perl-expression> <@alias>' >&2; exit 2; }
exec bash "$root/tools/mutation-run.sh" --perl "$@"
