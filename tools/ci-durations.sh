#!/usr/bin/env bash
# How long does each CI job actually take, across many runs?  Prints one row per
# (job name, conclusion) with the count, min, median and max wall-clock minutes
# over the last N completed runs of a workflow -- the distribution the
# `timeout-minutes` ceilings in `.github/workflows/ci.yml` have to clear.  Those
# ceilings were first set (PR #605) by hand-rolling this same
# `runs/<id>/jobs` loop once per question; this is that loop, kept.
#
# Durations come from each job's `started_at`/`completed_at`.  The
# `runs/<id>/timing` endpoint is NOT usable for this: it reports zero billable
# milliseconds on every job of this repository.  A job still running (null
# `completed_at`) is skipped and counted on stderr.
#
# Usage: tools/ci-durations.sh [options]
#   --repo owner/name    repository to query (default: lukstafi/ocannl-staging)
#   --workflow FILE      workflow file name (default: ci.yml)
#   --branch NAME        only runs of this branch (default: any)
#   --event NAME         only runs of this event, e.g. push, pull_request,
#                        schedule, workflow_dispatch (default: any)
#   -n N                 how many completed runs to read (default: 30)
#   --job REGEX         select job names (Python regular expression search)
#   --step REGEX        sum matching steps per job; print duration, paired job,
#                        rest-of-job and share distributions (minutes / percent)
#                        Renames can be selected with 'Old name|New name'.
#                        Missing/unusable steps are counted, never zero-filled.
#   -h, --help           this header
#
# Examples:
#   tools/ci-durations.sh                        # last 30 completed ci.yml runs
#   tools/ci-durations.sh --branch master -n 50
#   tools/ci-durations.sh --event schedule       # the extended (Windows) matrix
#   tools/ci-durations.sh --branch master --job Ubuntu --step 'Compile|Build and test'
#
# A step sample requires all matching intervals to be usable. Job/share/rest
# samples additionally require a positive job interval and a sum within it.
# Failed/cancelled jobs with usable times stay grouped by job conclusion;
# skipped steps never contribute, even when they have placeholder timestamps.
# Step names matched in each group are printed so renames remain visible.
# The share is calculated per job, then summarized (not a ratio of medians).
#
# Pure `gh api` + python3; no OCANNL build involved.  Requires an authenticated
# `gh`; any API failure aborts, so an empty table is never printed as a result.

set -euo pipefail

repo=lukstafi/ocannl-staging
workflow=ci.yml
branch=
event=
runs=30
selectors=()

die() {
  echo "ci-durations.sh: $*" >&2
  exit 1
}

usage() {
  # The header above is the documentation; print it rather than restating it.
  sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'
}

# Percent-encode one PATH segment.  Nothing the caller supplies reaches a URL
# raw: query parameters go through `gh api -f` (which encodes them), and the
# owner, repository and workflow file name are segments, where `-f` cannot help
# and `gh` passes the path through verbatim -- so a workflow legitimately named
# `ci#nightly.yml` would otherwise query `/actions/workflows/ci`, the `#`
# starting a fragment.  LC_ALL=C so the loop walks bytes, which is what a
# percent-encoding is defined over.
urlenc() {
  local LC_ALL=C s=$1 out= c i n
  for ((i = 0; i < ${#s}; i++)); do
    c=${s:i:1}
    case "$c" in
    [A-Za-z0-9._~-]) out=$out$c ;;
    *)
      # `"'$c"` is the byte's value, sign-extended for anything above 0x7F, so
      # mask it back to a byte before formatting: an unmasked non-ASCII
      # character renders as %FFFFFFFFFFFFFFC3 rather than %C3.
      printf -v n '%d' "'$c"
      out=$out$(printf '%%%02X' "$((n & 255))")
      ;;
    esac
  done
  printf '%s' "$out"
}

while [ $# -gt 0 ]; do
  case "$1" in
  --repo)
    repo=${2-}
    shift 2 || die "--repo needs a value"
    ;;
  --workflow)
    workflow=${2-}
    shift 2 || die "--workflow needs a value"
    ;;
  --branch)
    branch=${2-}
    shift 2 || die "--branch needs a value"
    ;;
  --event)
    event=${2-}
    shift 2 || die "--event needs a value"
    ;;
  --job | --step)
    [ $# -ge 2 ] || die "$1 needs a value"
    selectors+=("$1" "$2")
    shift 2
    ;;
  -n)
    runs=${2-}
    shift 2 || die "-n needs a value"
    ;;
  -h | --help)
    usage
    exit 0
    ;;
  *) die "unknown argument: $1 (try --help)" ;;
  esac
done

case "$repo" in
*/*/* | /* | */) die "--repo wants owner/name, got: $repo" ;;
*/*) ;;
*) die "--repo wants owner/name, got: $repo" ;;
esac
[ -n "$workflow" ] || die "--workflow wants a workflow file name"
repo_path=$(urlenc "${repo%%/*}")/$(urlenc "${repo#*/}")
workflow_path=$(urlenc "$workflow")
case "$runs" in '' | *[!0-9]*) die "-n wants a positive integer, got: $runs" ;; esac
[ "$runs" -gt 0 ] || die "-n wants a positive integer, got: $runs"

python3 "$(dirname "$0")/ci-timing.py" validate "${selectors[@]}"

# Every filter goes through `gh api --method GET -f`, which URL-encodes each
# value, rather than being concatenated into the path: a branch name may legally
# contain `#` or `&` (`release#1`, `feature&hotfix`), and interpolated those
# truncate or split the query, so the script would silently aggregate the wrong
# runs or none at all.
run_params=(-f status=completed)
# `[ ... ] && arr+=(…)` would abort the whole script under `set -e` whenever the
# filter is unset, which is the default; spell the conditionals out.
if [ -n "$branch" ]; then run_params+=(-f "branch=$branch"); fi
if [ -n "$event" ]; then run_params+=(-f "event=$event"); fi

# Page explicitly rather than with `--paginate`: that flag requests pages until
# the endpoint runs out, and a downstream `head`/`awk` only stops PRINTING --
# the walk continues, so a `-n 101` over a long workflow history would issue
# thousands of requests (and can exhaust the rate limit) after the first two
# pages already held the answer.  Here the loop stops as soon as it has -n ids
# or the API serves a short page.
per_page=100
if [ "$runs" -lt "$per_page" ]; then per_page=$runs; fi
run_ids=
run_count=0
page=1
while [ "$run_count" -lt "$runs" ]; do
  page_ids=$(gh api --method GET "repos/$repo_path/actions/workflows/$workflow_path/runs" \
    "${run_params[@]}" -f "per_page=$per_page" -f "page=$page" \
    --jq '.workflow_runs[].id') || die "listing runs of $workflow on $repo failed"
  page_count=$(printf '%s\n' "$page_ids" | awk 'NF { n++ } END { print n + 0 }')
  if [ "$page_count" -eq 0 ]; then break; fi
  run_ids=$(printf '%s\n%s' "$run_ids" "$page_ids" | awk 'NF')
  run_count=$((run_count + page_count))
  if [ "$page_count" -lt "$per_page" ]; then break; fi
  page=$((page + 1))
done

[ -n "$run_ids" ] || die "no completed runs of $workflow on $repo matched (branch=${branch:-any}, event=${event:-any})"

# The last page can overshoot -n; keep the newest -n of what arrived.
run_ids=$(printf '%s\n' "$run_ids" | awk -v n="$runs" 'NR <= n')
run_count=$(printf '%s\n' "$run_ids" | awk 'NF { n++ } END { print n + 0 }')

# Jobs are JSON lines, preserving step names and null timestamps without TSV loss.
jobs_json=$(
  for id in $run_ids; do
    gh api --paginate --method GET "repos/$repo_path/actions/runs/$id/jobs" -f per_page=100 \
      --jq '.jobs[]' || die "fetching jobs of run $id failed"
  done
)
[ -n "$jobs_json" ] || die "the $run_count matched run(s) reported no jobs"

printf '%s\n' "$jobs_json" | python3 "$(dirname "$0")/ci-timing.py" durations \
  "$repo  $workflow  branch=${branch:-any}  event=${event:-any}  runs=$run_count" "${selectors[@]}"
