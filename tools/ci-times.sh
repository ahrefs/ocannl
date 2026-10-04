#!/usr/bin/env bash
# Where do the minutes of a CI run go?  Prints one line per job of a GitHub
# Actions run with the job's total wall-clock duration, and indented under it
# the steps that took more than ~30 seconds -- the question every CI-speed
# investigation starts with, previously answered by rewriting the same
# throwaway python each time (PRs #518/#519).
#
# Jobs and steps appear in the order GitHub lists them.  Pure `gh api` +
# python3; no OCANNL build involved.  A job's conclusion is appended when it
# is anything other than success (failure, cancelled, skipped...), and a job
# with no wall-clock duration to report says `(no time)` rather than a number
# (see `interval` in tools/ci-timing.py).
#
# tools/test-ci-times.sh is the hermetic harness for the reporting below; it
# runs this exact file against a recorded `gh` fixture.
#
# Usage: tools/ci-times.sh [RUN_ID]
#   tools/ci-times.sh              # latest completed ci.yml run on the current repo
#   tools/ci-times.sh 17274043031  # that specific run
#
# The repo is whatever `gh` resolves for the current directory's remotes; run
# it from the checkout whose CI you are asking about.

set -euo pipefail

STEP_THRESHOLD=30 # seconds; steps at or under this stay unlisted

run_id="${1-}"
if [ -z "$run_id" ]; then
  run_id=$(gh run list --workflow ci.yml --status completed --limit 1 \
    --json databaseId --jq '.[0].databaseId')
  if [ -z "$run_id" ] || [ "$run_id" = "null" ]; then
    echo "ci-times.sh: no completed ci.yml run found on this repo" >&2
    exit 1
  fi
  echo "run $run_id (latest completed ci.yml)"
fi

# Fetch first, pipe second: on an HTTP error gh prints the response body to
# stdout, which would otherwise reach python as unparseable "JSON".
jobs=$(gh api --paginate "repos/{owner}/{repo}/actions/runs/${run_id}/jobs?per_page=100" \
  --jq '.jobs[]')

printf '%s\n' "$jobs" | python3 "$(dirname "$0")/ci-timing.py" times "$STEP_THRESHOLD"
