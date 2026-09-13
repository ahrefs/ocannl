#!/usr/bin/env bash
# Hermetic regression and mutation tests for tools/ci-times.sh.
#
# The subject is pure `gh` + python3, so a fake `gh` on PATH is the whole
# isolation story: no network, no repository resolution, no GitHub account.
# The fixture is one recorded `runs/<id>/jobs` payload holding the three
# timestamp shapes a real run mixes -- an ordinary job with a measurable
# duration, a job whose timestamps are absent or half-absent, and a SKIPPED
# job, whose placeholder timestamps run BACKWARDS (completed_at one second
# before started_at, which the pre-guard subtraction rendered as `-1m59s`;
# gh-ocannl-901 measurement on staging#611).
#
#   tools/test-ci-times.sh          # run every leg
#   tools/test-ci-times.sh --keep   # keep the scratch directory
#
# Each shipping assertion has a fault-injected twin: a mutated copy of the
# subject with exactly one guard removed, which the same oracle must reject.
# That is what separates a test of the fix from a restatement of today's
# output -- both mutants below print a duration where there is none, and were
# checked to be rejected for that reason and not by failing to run.
#
# A failing leg reports the run directory holding the subject's stdout, stderr
# and recorded fake-`gh` calls; cleanup deletes it on the way out, so `--keep`
# is what makes that report actionable. It is the same flag, spelled the same
# way, as the sibling harnesses tools/test-pin-revisions.sh,
# tools/test-promote.sh and tools/test-test-run.sh.
#
# It joins tools/test-test-run.sh and scripts/test-setup-ocaml-env.sh in the
# Ubuntu CI step, and carries the contract those two share: a leg this host
# cannot decide prints `SKIP LABEL REASON` and the footer always states the
# skip count, so a green over legs that never ran cannot read as coverage.
# Here that is one host fact, `python3` under that exact name -- which is also
# why it is on no dune alias, since not every platform `dune runtest` covers
# calls it that, while the reporting it pins is platform-independent.

set -u

. "$(cd "$(dirname "$0")/../scripts" && pwd)/harness-support.sh"
harness_args "$@"

HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="$HERE/ci-times.sh"
[ -f "$SRC" ] || { echo "no $SRC" >&2; exit 2; }


# The one host fact this harness cannot work around: the subject invokes
# `python3` by that name, so a host without one decides nothing about it.  That
# is a skip under the contract the step's harnesses share, not a failure and
# not a silent pass -- the footer's count is what says so.
HAVE_PYTHON3=1
command -v python3 >/dev/null 2>&1 || HAVE_PYTHON3=0

# The skip count is printed on every run, not only when it is nonzero: "all
# legs passed" over a run that decided none of them is the reading to prevent.

if [ "$HAVE_PYTHON3" = 0 ]; then
  skip "every leg" "no python3 on PATH; the subject invokes it under that name"
  finish
fi

harness_scratch "test-ci-times"


mkdir -p "$TMP/bin" "$TMP/work"

# The recorded run.  Timestamps, not durations, because that is what the API
# serves and the arithmetic over them is the thing under test:
#   Set up matrix          equal timestamps -- a real 0s, which must stay a
#                          measurement rather than join the unavailable ones
#   Build (ubuntu-...)     an ordinary job, with a long step, a short step that
#                          stays unlisted, and a reversed post-step
#   Build (macos-...)      skipped, reversed by ONE second: the exact shape
#                          that printed -1m59s
#   Build (windows-...)    skipped, reversed by 119 seconds: so the guard is
#                          pinned on the sign and not on one magic value
#   Extended matrix        queued, no timestamps at all
#   Aggregate              running, a start but no finish
#   Report                 completed and failed, a finish but no start
cat >"$TMP/jobs.json" <<'FIXTURE'
{
  "total_count": 7,
  "jobs": [
    {
      "name": "Set up matrix",
      "status": "completed",
      "conclusion": "success",
      "started_at": "2026-09-01T10:00:00Z",
      "completed_at": "2026-09-01T10:00:00Z",
      "steps": []
    },
    {
      "name": "Build (ubuntu-latest, 5.5.x, main)",
      "status": "completed",
      "conclusion": "success",
      "started_at": "2026-09-01T10:00:00Z",
      "completed_at": "2026-09-01T10:12:34Z",
      "steps": [
        {
          "name": "Set up job",
          "started_at": "2026-09-01T10:00:00Z",
          "completed_at": "2026-09-01T10:00:07Z"
        },
        {
          "name": "Build and test",
          "started_at": "2026-09-01T10:00:07Z",
          "completed_at": "2026-09-01T10:10:07Z"
        },
        {
          "name": "Post job cleanup",
          "started_at": "2026-09-01T10:12:34Z",
          "completed_at": "2026-09-01T10:12:04Z"
        }
      ]
    },
    {
      "name": "Build (macos-latest, 5.5.x, main)",
      "status": "completed",
      "conclusion": "skipped",
      "started_at": "2026-09-01T10:14:00Z",
      "completed_at": "2026-09-01T10:13:59Z",
      "steps": []
    },
    {
      "name": "Build (windows-latest, 5.5.x, main)",
      "status": "completed",
      "conclusion": "skipped",
      "started_at": "2026-09-01T10:15:00Z",
      "completed_at": "2026-09-01T10:13:01Z",
      "steps": []
    },
    {
      "name": "Extended matrix",
      "status": "queued",
      "conclusion": null,
      "started_at": null,
      "completed_at": null,
      "steps": []
    },
    {
      "name": "Aggregate",
      "status": "in_progress",
      "conclusion": null,
      "started_at": "2026-09-01T10:15:00Z",
      "completed_at": null,
      "steps": []
    },
    {
      "name": "Report",
      "status": "completed",
      "conclusion": "failure",
      "started_at": null,
      "completed_at": "2026-09-01T10:20:00Z",
      "steps": []
    }
  ]
}
FIXTURE

# Fake `gh`.  It records every call and refuses anything the subject is not
# supposed to make, so a rewritten query is a loud failure rather than a
# silently different answer.  `--jq` is gh's own filter, which a fake cannot
# apply for free; the one filter the subject uses is `.jobs[]`, emulated here,
# and the contract check above it is what keeps that emulation honest.
cat >"$TMP/bin/gh" <<'FAKE_GH'
#!/usr/bin/env bash
set -u
printf '%s\n' "$*" >>"$FAKE_GH_CALLS"

# Argument-wise, never by matching the joined command line: adjacent ` --a b `
# ` --c d ` patterns share the space between them and can never both match.
has_arg() { # has_arg WANTED ARG...
  local wanted=$1 arg
  shift
  for arg in "$@"; do [ "$arg" = "$wanted" ] && return 0; done
  return 1
}

case "${1:-}" in
  run)
    [ "${2:-}" = list ] || { echo "unsupported fake gh call: $*" >&2; exit 64; }
    has_arg --workflow "$@" && has_arg ci.yml "$@" \
      && has_arg --status "$@" && has_arg completed "$@" \
      && has_arg --json "$@" && has_arg databaseId "$@" \
      || { echo "unexpected gh run list contract: $*" >&2; exit 64; }
    printf '%s\n' "${FAKE_LATEST_RUN:-424242}"
    ;;
  api)
    has_arg --jq "$@" && has_arg '.jobs[]' "$@" \
      || { echo "unexpected gh api filter: $*" >&2; exit 64; }
    # The path is asserted whole: the literal {owner}/{repo} is what makes the
    # subject ask about the CHECKOUT's repository rather than a hard-coded one.
    has_arg "repos/{owner}/{repo}/actions/runs/${FAKE_EXPECTED_RUN}/jobs?per_page=100" "$@" \
      || { echo "unexpected gh api path: $*" >&2; exit 64; }
    python3 -c '
import json, sys
with open(sys.argv[1]) as f:
    run = json.load(f)
for job in run["jobs"]:
    print(json.dumps(job))
' "$FAKE_JOBS_JSON"
    ;;
  *)
    echo "unsupported fake gh call: $*" >&2
    exit 64
    ;;
esac
FAKE_GH
chmod +x "$TMP/bin/gh"

# What the subject must print for that fixture.  Written out rather than
# recorded from a run: every line of it is a claim.  The two `(no time)` rows
# are where the reversed timestamps land; `0s` is the boundary that must NOT.
cat >"$TMP/expected-table" <<'TABLE'
      0s  Set up matrix
  12m34s  Build (ubuntu-latest, 5.5.x, main)
      10m00s    Build and test
(no time)  Build (macos-latest, 5.5.x, main)  [skipped]
(no time)  Build (windows-latest, 5.5.x, main)  [skipped]
(queued)  Extended matrix
(in_progress)  Aggregate
(no time)  Report  [failure]
TABLE

printf '%s\n' \
  'api --paginate repos/{owner}/{repo}/actions/runs/424242/jobs?per_page=100 --jq .jobs[]' \
  >"$TMP/expected-calls-explicit"
printf '%s\n' \
  'run list --workflow ci.yml --status completed --limit 1 --json databaseId --jq .[0].databaseId' \
  'api --paginate repos/{owner}/{repo}/actions/runs/424242/jobs?per_page=100 --jq .jobs[]' \
  >"$TMP/expected-calls-latest"

run_subject() { # run_subject SUBJECT LABEL [ARG...]
  local subject=$1 label=$2 dir="$TMP/runs/$2" status
  shift 2
  rm -rf "$dir"
  mkdir -p "$dir"
  : >"$dir/gh.calls"
  (
    cd "$TMP/work" || exit 125
    PATH="$TMP/bin:$PATH" \
      FAKE_GH_CALLS="$dir/gh.calls" \
      FAKE_JOBS_JSON="$TMP/jobs.json" \
      FAKE_EXPECTED_RUN="${FAKE_EXPECTED_RUN:-424242}" \
      FAKE_LATEST_RUN="${FAKE_LATEST_RUN:-424242}" \
      bash "$subject" "$@" >"$dir/stdout" 2>"$dir/stderr"
  )
  status=$?
  printf '%s\n' "$status" >"$dir/status"
  return 0
}

# No duration anywhere may be negative, whatever the fixture grows to hold:
# `-1m59s` and `-2m01s` are what the two reversed jobs printed, and a minus
# sign in this report has no other reading.
no_negative_durations() { # no_negative_durations DIR
  ! grep -qE '(^|[[:space:]])-[0-9]+(m[0-9]{2})?s([[:space:]]|$)' "$1/stdout"
}

oracle_explicit_run() { # oracle_explicit_run SUBJECT LABEL
  local subject=$1 label=$2 dir="$TMP/runs/$2"
  run_subject "$subject" "$label" 424242
  [ "$(cat "$dir/status")" -eq 0 ] \
    && cmp -s "$TMP/expected-table" "$dir/stdout" \
    && cmp -s "$TMP/expected-calls-explicit" "$dir/gh.calls" \
    && [ ! -s "$dir/stderr" ] \
    && no_negative_durations "$dir"
}

oracle_latest_run() { # oracle_latest_run SUBJECT LABEL
  local subject=$1 label=$2 dir="$TMP/runs/$2"
  run_subject "$subject" "$label"
  [ "$(cat "$dir/status")" -eq 0 ] \
    && [ "$(head -n 1 "$dir/stdout")" = 'run 424242 (latest completed ci.yml)' ] \
    && cmp -s "$TMP/expected-table" <(tail -n +2 "$dir/stdout") \
    && cmp -s "$TMP/expected-calls-latest" "$dir/gh.calls" \
    && [ ! -s "$dir/stderr" ] \
    && no_negative_durations "$dir"
}

oracle_no_run_loud() { # oracle_no_run_loud SUBJECT LABEL
  local subject=$1 label=$2 dir="$TMP/runs/$2"
  FAKE_LATEST_RUN=null run_subject "$subject" "$label"
  [ "$(cat "$dir/status")" -ne 0 ] \
    && grep -q '^ci-times.sh: no completed ci.yml run found on this repo$' "$dir/stderr" \
    && [ ! -s "$dir/stdout" ]
}

if oracle_explicit_run "$SRC" shipping-explicit; then
  report 0 "measured jobs and steps: exact durations, long steps only"
  report 0 "reversed skipped timestamps: reported as no duration, not negative"
  report 0 "absent timestamps: queued, running and completed each labelled"
  report 0 "equal timestamps: 0s stays a measurement"
  report 0 "run query: the checkout's repository, one paginated jobs call"
else
  report 1 "shipping run of an explicit run id" "see $TMP/runs/shipping-explicit"
fi
if oracle_latest_run "$SRC" shipping-latest; then
  report 0 "no run id: resolves the latest completed ci.yml run and says so"
else
  report 1 "shipping run with no run id" "see $TMP/runs/shipping-latest"
fi
if oracle_no_run_loud "$SRC" shipping-no-run; then
  report 0 "no completed run: fails loudly without printing a table"
else
  report 1 "no completed run available" "see $TMP/runs/shipping-no-run"
fi


# Drop the sign guard: the interval arithmetic goes back to reporting whatever
# the subtraction says, so the two skipped jobs print negative durations.
sign_mutant=$(mutant negative-durations '
  index($0, "return d if d >= 0 else None") { dropped++; next }
  index($0, "d = (datetime.strptime(end") { sub(/d = /, "return "); changed++; print; next }
  { print }
  END { if (changed != 1 || dropped != 1) exit 9 }')
if [ -n "$sign_mutant" ]; then
  expect_rejected "negative interval printed as a duration" "$sign_mutant" \
    oracle_explicit_run '^ *-1m59s  Build \(macos'
else
  report 1 "negative control: negative interval printed as a duration" \
    "could not build the mutant"
fi

# Keep the sign guard but restore the old label, which answered a job with no
# duration by naming its status -- `(completed)` for a skipped job, which reads
# as a report rather than as its absence.
label_mutant=$(mutant status-label '
  index($0, "label = label_for(job, total)") { print repl; changed++; next }
  { print }
  END { if (changed != 1) exit 9 }' \
  -v "repl=    label = human(total) if total is not None else '(' + str(job.get('status')) + ')'")
if [ -n "$label_mutant" ]; then
  expect_rejected "no duration reported as (completed)" "$label_mutant" \
    oracle_explicit_run '^\(completed\)  Build \(macos'
else
  report 1 "negative control: no duration reported as (completed)" \
    "could not build the mutant"
fi

finish
