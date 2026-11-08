#!/usr/bin/env bash
# One-shot negative control: tools/mutant.sh <file> <perl-expression> <@alias>
# The expression runs as perl -0777 -pe on a private copy; a byte-identical result
# refuses before any test runs. The alias names one test with an .expected golden,
# as in mutation-run.sh. A green unmutated baseline is required before mutation.
# Reports the run and its false claims: 1 means caught, 0 survived; incomplete
# or never-executed runs exit 4 (no evidence), and a red baseline refuses with 2.
# Other runner statuses pass through unchanged, including 69 (DEVICE UNHEALTHY)
# and 75 (SLOT REFUSED).
# Shares mutation-run.sh's cancellation/restoration traps: INT/TERM/HUP cancel
# and reap the runner before byte-identical restoration. A surviving lock or a
# failed restore exits 3 with a retained recovery copy; SIGKILL cannot be trapped.
# Use an idle isolated worktree; do not edit the file during the run.
set -eu
[ "$#" = 3 ] || { echo 'usage: tools/mutant.sh <file> <perl-expression> <@alias>' >&2; exit 2; }
exec bash "$(dirname -- "$0")/mutation-run.sh" --perl "$@"
