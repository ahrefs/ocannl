#!/usr/bin/env bash
# Hermetic cross-run timing fixtures, including per-job shares and missing steps.
# Fake gh serves JSON lines: no network, credentials or performance campaign.
# Failure details read only this run's captured status, streams and fake API calls;
# they compare calls to the fixture list, and stdout only to an explicit golden.
# They do not diagnose host resource limits or inspect real GitHub calls.
# Run with --keep to retain the artifacts too; failures print evidence without it.
set -u
. "$(cd "$(dirname "$0")/../scripts" && pwd)/harness-support.sh"
harness_args "$@"
harness_require python3
harness_scratch test-ci-durations
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$TMP/bin" "$TMP/runs"
cat >"$TMP/bin/gh" <<'FAKE'
#!/usr/bin/env bash
if [ -n "${FAKE_LOCALE_OUT:-}" ] && [ ! -e "$FAKE_LOCALE_OUT" ]; then
  printf 'LANG=%s LC_ALL=%s LC_CTYPE=%s\n' \
    "${LANG-<unset>}" "${LC_ALL-<unset>}" "${LC_CTYPE-<unset>}" >"$FAKE_LOCALE_OUT"
fi
printf '%s\n' "$*" >>"$FAKE_CALLS"
case "$*" in
  'api --method GET repos/lukstafi/ocannl-staging/actions/workflows/ci.yml/runs -f status=completed -f branch=master -f per_page=8 -f page=1 --jq .workflow_runs[].id') printf '1\n2\n3\n4\n5\n6\n7\n8\n' ;;
  'api --paginate --method GET repos/lukstafi/ocannl-staging/actions/runs/'[1-8]'/jobs -f per_page=100 --jq .jobs[]')
    for arg in "$@"; do
      case "$arg" in */runs/*/jobs) id=${arg%/jobs}; id=${id##*/} ;; esac
    done
    cat "$FIXTURES/$id.json" ;;
  *) echo "unexpected call: $*" >&2; exit 64 ;;
esac
FAKE
chmod +x "$TMP/bin/gh"
# Two distinct paired shares: median(50%,80%) = 65%, NOT 9/15 = 60%.
# Renamed steps are matched by one explicit regex; unrelated jobs stay excluded.
python3 - "$TMP" <<'PY'
import json
from pathlib import Path
import sys
from datetime import datetime, timedelta
root = Path(sys.argv[1])
t0 = datetime(2026, 9, 1)
def stamp(seconds):
    return (t0 + timedelta(seconds=seconds)).strftime('%Y-%m-%dT%H:%M:%SZ')
for i, (total, seconds, name, conclusion) in enumerate([
    (1200,600,'Build and test','success'),
    (600,480,'Dune checks','success'),
    (600,None,None,'success'),
    (600,None,'Build and test','success'),
    (600,-1,'Build and test','success'),
    (600,0,'Build and test','skipped'),
    (300,120,'Build and test','cancelled'),
    (0,0,'Build and test','success'),
], 1):
    steps = [] if name is None else [dict(name=name, started_at=stamp(0),
        completed_at=None if seconds is None else stamp(seconds),
        conclusion='skipped' if i == 6 else conclusion)]
    if i == 1:
        # Two executed matches sum to the same 600s; this pins the combination.
        steps[0]['completed_at'] = stamp(300)
        steps.append(dict(name='Dune checks', started_at=stamp(300),
            completed_at=stamp(600), conclusion='success'))
    if i == 2:
        # GitHub lists the other conditional build action as skipped. Even
        # measurable-looking placeholders must not spoil the executed match.
        steps.append(dict(name='Build and test', started_at=stamp(0),
            completed_at=stamp(600), conclusion='skipped'))
    job = dict(name='Ubuntu', status='completed', conclusion=conclusion,
        started_at=stamp(0), completed_at=stamp(total), steps=steps)
    other = dict(job, name='macOS')
    (root / f'{i}.json').write_text(json.dumps(job)+'\n'+json.dumps(other)+'\n')
PY
cat >"$TMP/expected" <<'EXPECTED'
lukstafi/ocannl-staging  ci.yml  branch=master  event=any  runs=8

job selector: ^Ubuntu$; step selector: Build and test|Dune checks
distributions: n min median max (minutes; share in percent)

Ubuntu [cancelled] jobs=1 missing=0 unusable=0 unpaired=0
  matched steps: Build and test
  step: 1 2.0 2.0 2.0
  job: 1 5.0 5.0 5.0
  rest: 1 3.0 3.0 3.0
  share: 1 40.0 40.0 40.0

Ubuntu [skipped] jobs=1 missing=0 unusable=1 unpaired=0
  matched steps: Build and test
  step: 0 unavailable
  job: 0 unavailable
  rest: 0 unavailable
  share: 0 unavailable

Ubuntu [success] jobs=6 missing=1 unusable=2 unpaired=1
  matched steps: Build and test, Dune checks
  step: 3 0.0 8.0 10.0
  job: 2 10.0 15.0 20.0
  rest: 2 2.0 6.0 10.0
  share: 2 50.0 65.0 80.0
EXPECTED
cat >"$TMP/expected-default" <<'EXPECTED'
lukstafi/ocannl-staging  ci.yml  branch=master  event=any  runs=8

job [conclusion]                                         n      min   median      max
Ubuntu [cancelled]                                       1      5.0      5.0      5.0
Ubuntu [skipped]                                         1     10.0     10.0     10.0
Ubuntu [success]                                         6      0.0     10.0     20.0
macOS [cancelled]                                        1      5.0      5.0      5.0
macOS [skipped]                                          1     10.0     10.0     10.0
macOS [success]                                          6      0.0     10.0     20.0
EXPECTED
printf '%s\n' 'api --method GET repos/lukstafi/ocannl-staging/actions/workflows/ci.yml/runs -f status=completed -f branch=master -f per_page=8 -f page=1 --jq .workflow_runs[].id' >"$TMP/calls"
for id in 1 2 3 4 5 6 7 8; do
  printf '%s\n' "api --paginate --method GET repos/lukstafi/ocannl-staging/actions/runs/$id/jobs -f per_page=100 --jq .jobs[]" >>"$TMP/calls"
done
run_subject() {
  local subject=$1 label=$2 rc=0
  shift 2
  mkdir -p "$TMP/runs/$label"
  : >"$TMP/runs/$label/calls"
  PATH="$TMP/bin:$PATH" FAKE_CALLS="$TMP/runs/$label/calls" FIXTURES="$TMP" \
    "${BASH:-bash}" "$subject" --branch master -n 8 "$@" >"$TMP/runs/$label/stdout" 2>"$TMP/runs/$label/stderr" || rc=$?
  printf '%s\n' "$rc" >"$TMP/runs/$label/rc"
}
# Print evidence before cleanup, including a signal exit with empty stderr.
# A diff's nonzero status is evidence here, not the helper's outcome.
subject_detail() { # LABEL [EXPECTED_STDOUT]
  local run="$TMP/runs/$1" expected_calls="$TMP/calls"
  case $1 in invalid|base) expected_calls=/dev/null ;; esac
  printf 'subject exit: %s\n' "$(cat "$run/rc")"
  printf '%s\n' 'stderr (empty if nothing follows):'
  cat "$run/stderr"
  printf '%s\n' 'API calls diff (expected -> actual; empty means equal):'
  diff -u "$expected_calls" "$run/calls" || :
  if [ $# -gt 1 ]; then
    printf '%s\n' 'stdout diff (expected -> actual; empty means equal):'
    diff -u "$2" "$run/stdout" || :
  else
    printf '%s\n' 'stdout:'
    cat "$run/stdout"
  fi
}
oracle() {
  run_subject "$1" "$2" --job '^Ubuntu$' --step 'Build and test|Dune checks'
  [ "$(cat "$TMP/runs/$2/rc")" = 0 ] && cmp -s "$TMP/expected" "$TMP/runs/$2/stdout" \
    && cmp -s "$TMP/calls" "$TMP/runs/$2/calls" && [ ! -s "$TMP/runs/$2/stderr" ]
}
if oracle "$HERE/ci-durations.sh" shipping; then
  report 0 "cross-run selectors: exact distributions, paired share/rest and API calls"
else report 1 "cross-run selectors" "$(subject_detail shipping "$TMP/expected")"; fi
run_subject "$HERE/ci-durations.sh" jobs --job '^Ubuntu$'
if [ "$(cat "$TMP/runs/jobs/rc")" = 0 ] \
  && grep -qE '^Ubuntu \[success\] +6 +0.0 +10.0 +20.0$' "$TMP/runs/jobs/stdout"; then
  report 0 "whole-job aggregation retains zero and conclusion grouping"
else report 1 "whole-job aggregation" "$(subject_detail jobs)"; fi
run_subject "$HERE/ci-durations.sh" default
if [ "$(cat "$TMP/runs/default/rc")" = 0 ] \
  && cmp -s "$TMP/expected-default" "$TMP/runs/default/stdout" \
  && cmp -s "$TMP/calls" "$TMP/runs/default/calls" && [ ! -s "$TMP/runs/default/stderr" ]; then
  report 0 "selector-free default invocation aggregates all jobs without --job or --step"
else report 1 "selector-free default invocation" "$(subject_detail default "$TMP/expected-default")"; fi
run_subject "$HERE/ci-durations.sh" invalid --step '['
if [ "$(cat "$TMP/runs/invalid/rc")" != 0 ] && [ ! -s "$TMP/runs/invalid/calls" ] \
  && grep -q 'unterminated character set' "$TMP/runs/invalid/stderr"; then
  report 0 "invalid selectors fail before API calls"
else report 1 "invalid selector guard" "$(subject_detail invalid)"; fi
# Restore the pre-selector interface without depending on Git history (CI
# checkouts can be shallow). The original base was also rejected locally.
python3 - "$HERE/ci-durations.sh" "$TMP/base.sh" <<'PYTHON'
from pathlib import Path
import sys
source = Path(sys.argv[1]).read_text()
start = source.index("  --job | --step)")
end = source.index("  -n)", start)
Path(sys.argv[2]).write_text(source[:start] + source[end:])
PYTHON
run_subject "$TMP/base.sh" base --job '^Ubuntu$' --step 'Build and test|Dune checks'
if [ "$(cat "$TMP/runs/base/rc")" != 0 ] && grep -q 'unknown argument: --job' "$TMP/runs/base/stderr"; then
  report 0 "negative control: pre-selector interface refuses selection request"
else report 1 "selector-interface negative control" "$(subject_detail base)"; fi
# Dropping the shared sign guard must pollute step statistics, not fail to launch.
mkdir -p "$TMP/sign-mutant"
cp "$HERE/ci-durations.sh" "$TMP/sign-mutant/ci-durations.sh"
sed 's/return d if d >= 0 else None/return d/' "$HERE/ci-timing.py" >"$TMP/sign-mutant/ci-timing.py"
if oracle "$TMP/sign-mutant/ci-durations.sh" negative; then
  report 1 "negative control: shared interval sign guard" "$(subject_detail negative "$TMP/expected")"
elif [ "$(cat "$TMP/runs/negative/rc")" = 0 ] && grep -q 'step: 4 -0.0' "$TMP/runs/negative/stdout"; then
  report 0 "negative control: shared interval sign guard changes measured population"
else report 1 "sign mutation rejected for unrelated reason" "$(subject_detail negative "$TMP/expected")"; fi
# Restoring inclusion of skipped alternatives must change the sample population.
mkdir -p "$TMP/skip-mutant"
cp "$HERE/ci-durations.sh" "$TMP/skip-mutant/ci-durations.sh"
sed 's/selected = \[s for s in selected if s.get("conclusion") != "skipped"\]/selected = selected/' \
  "$HERE/ci-timing.py" >"$TMP/skip-mutant/ci-timing.py"
if oracle "$TMP/skip-mutant/ci-durations.sh" skipped-alternative; then
  report 1 "negative control: skipped alternatives excluded" "$(subject_detail skipped-alternative "$TMP/expected")"
elif [ "$(cat "$TMP/runs/skipped-alternative/rc")" = 0 ] \
  && grep -q 'step: 3 0.0 10.0 18.0' "$TMP/runs/skipped-alternative/stdout"; then
  report 0 "negative control: skipped alternatives change the measured population"
else report 1 "skip mutation rejected for unrelated reason" "$(subject_detail skipped-alternative "$TMP/expected")"; fi

# Unset-locale startup guard (gh-ocannl-2484): urlenc runs inside $(...) with
# local LC_ALL=C, so standalone invocations with unset or empty LANG/LC_ALL must
# export LANG=C while preserving LC_CTYPE, caller LANG/LC_ALL, and bytewise
# UTF-8 percent-encoding.
run_locale() { # LABEL ENV_ARGS... -- EXTRA_SUBJECT_ARGS...
  local label=$1; shift
  local env_args=() rc=0
  while [ $# -gt 0 ] && [ "$1" != -- ]; do env_args+=("$1"); shift; done
  shift
  mkdir -p "$TMP/runs/$label"
  : >"$TMP/runs/$label/calls"
  rm -f "$TMP/runs/$label/locale"
  env ${env_args[@]+"${env_args[@]}"} PATH="$TMP/bin:$PATH" \
    FAKE_CALLS="$TMP/runs/$label/calls" FIXTURES="$TMP" FAKE_LOCALE_OUT="$TMP/runs/$label/locale" \
    bash "$HERE/ci-durations.sh" --branch master -n 8 "$@" \
    >"$TMP/runs/$label/stdout" 2>"$TMP/runs/$label/stderr" || rc=$?
  printf '%s\n' "$rc" >"$TMP/runs/$label/rc"
}
run_locale locale-unset -u LANG -u LC_ALL LC_CTYPE=UTF-8 -- --job '^Ubuntu$' --step 'Build and test|Dune checks'
if [ "$(cat "$TMP/runs/locale-unset/rc")" = 0 ] &&
   cmp -s "$TMP/expected" "$TMP/runs/locale-unset/stdout" &&
   [ "$(cat "$TMP/runs/locale-unset/locale" 2>/dev/null)" = 'LANG=C LC_ALL=<unset> LC_CTYPE=UTF-8' ]; then
  report 0 'unset LANG/LC_ALL exports LANG=C while preserving LC_CTYPE and exact output'
else report 1 'unset LANG/LC_ALL exports LANG=C while preserving LC_CTYPE and exact output' \
  "locale=$(cat "$TMP/runs/locale-unset/locale" 2>/dev/null); $(subject_detail locale-unset "$TMP/expected")"; fi
run_locale locale-empty LANG= LC_ALL= LC_CTYPE=UTF-8 -- --job '^Ubuntu$' --step 'Build and test|Dune checks'
if [ "$(cat "$TMP/runs/locale-empty/rc")" = 0 ] &&
   [ "$(cat "$TMP/runs/locale-empty/locale" 2>/dev/null)" = 'LANG=C LC_ALL= LC_CTYPE=UTF-8' ]; then
  report 0 'empty LANG/LC_ALL exports LANG=C while preserving LC_CTYPE'
else report 1 'empty LANG/LC_ALL exports LANG=C while preserving LC_CTYPE' \
  "locale=$(cat "$TMP/runs/locale-empty/locale" 2>/dev/null); $(subject_detail locale-empty)"; fi
run_locale locale-lang -u LC_ALL LANG=en_US.UTF-8 LC_CTYPE=UTF-8 -- --job '^Ubuntu$' --step 'Build and test|Dune checks'
if [ "$(cat "$TMP/runs/locale-lang/rc")" = 0 ] &&
   [ "$(cat "$TMP/runs/locale-lang/locale" 2>/dev/null)" = 'LANG=en_US.UTF-8 LC_ALL=<unset> LC_CTYPE=UTF-8' ]; then
  report 0 'caller-supplied LANG is preserved untouched'
else report 1 'caller-supplied LANG is preserved untouched' \
  "locale=$(cat "$TMP/runs/locale-lang/locale" 2>/dev/null); $(subject_detail locale-lang)"; fi
run_locale locale-lcall -u LANG LC_ALL=POSIX LC_CTYPE=UTF-8 -- --job '^Ubuntu$' --step 'Build and test|Dune checks'
if [ "$(cat "$TMP/runs/locale-lcall/rc")" = 0 ] &&
   [ "$(cat "$TMP/runs/locale-lcall/locale" 2>/dev/null)" = 'LANG=<unset> LC_ALL=POSIX LC_CTYPE=UTF-8' ]; then
  report 0 'caller-supplied LC_ALL is preserved untouched'
else report 1 'caller-supplied LC_ALL is preserved untouched' \
  "locale=$(cat "$TMP/runs/locale-lcall/locale" 2>/dev/null); $(subject_detail locale-lcall)"; fi
non_ascii_wf=$(printf 'ci#\303\251.yml')
run_locale locale-urlenc -u LANG -u LC_ALL LC_CTYPE=UTF-8 -- --workflow "$non_ascii_wf" --job '^Ubuntu$'
if grep -q 'actions/workflows/ci%23%C3%A9\.yml/runs' "$TMP/runs/locale-urlenc/calls"; then
  report 0 'urlenc percent-encodes non-ASCII bytes under unset LANG/LC_ALL and LC_CTYPE=UTF-8'
else report 1 'urlenc percent-encodes non-ASCII bytes under unset LANG/LC_ALL and LC_CTYPE=UTF-8' \
  "calls=$(cat "$TMP/runs/locale-urlenc/calls"); rc=$(cat "$TMP/runs/locale-urlenc/rc")"; fi
finish
