#!/usr/bin/env bash
# Exercise shared harness lifecycle, counted skips (and a strict run's list of
# expected ones), the --only leg filter with its coverage account, and
# reason-specific rejection.
# Usage: scripts/test-harness-support.sh [--keep] [--help]
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
. "$HERE/harness-support.sh"
harness_args "$@"
harness_scratch harness-support
cat >"$TMP/probe.sh" <<'PROBE'
#!/usr/bin/env bash
# Shared support fixture help.
# Last help line.
. "$SUPPORT"
harness_args "$@"
harness_scratch contract
printf '%s\n' "$TMP" >"$RECORD"
case $MODE in
  skip) harness_require definitely_missing_harness_tool ;;
  skips) skip 'first platform check' 'no first platform'; skip 'second platform check' 'no second platform' ;;
  fail) report 1 'injected failure' ;;
  signal) kill -TERM "$$" ;;
  abort) exit 50 ;;
  errexit) set -e; (exit 7) ;;
  unfinished) exit 0 ;;
  legs)
    # Each body leaves a trace, so a filtered leg is seen not to have run.
    if harness_leg 'alpha leg #odd'; then echo alpha >>"$RECORD.ran"; report 0 'alpha body'; fi
    if harness_leg 'beta leg #even'; then
      echo beta >>"$RECORD.ran"; report 0 'beta body'
      [ -z "${BETA_SKIP:-}" ] || skip 'beta check' 'no beta host'
    fi
    if harness_leg 'gamma leg #odd'; then echo gamma >>"$RECORD.ran"; report "${GAMMA_RC:-0}" 'gamma body'; fi ;;
esac
finish
PROBE
export SUPPORT=$HERE/harness-support.sh RECORD=$TMP/record MODE=pass
# The probes' skip verdict is the subject; an ambient strict mode, or an
# ambient list of expected skips, is not.
unset OCANNL_TOOL_HARNESS_STRICT OCANNL_TOOL_HARNESS_EXPECTED_SKIPS \
  OCANNL_TOOL_HARNESS_NAME OCANNL_TOOL_HARNESS_SKIP_RECORD
probe() {
  local rc=0
  bash "$TMP/probe.sh" "$@" >"$TMP/out" 2>&1 || rc=$?
  return "$rc"
}
probe; rc=$?
path=$(cat "$RECORD")
if [ "$rc" = 0 ] && [ ! -d "$path" ] && grep -q 'all legs passed (0 skipped)' "$TMP/out" && ! grep -q 'aborted before finish' "$TMP/out"; then
  report 0 'success cleans scratch and counts zero skips'
else report 1 'success cleans scratch and counts zero skips'; fi
MODE=skip probe; rc=$?
if [ "$rc" = 0 ] && grep -q '^SKIP ' "$TMP/out" && grep -q 'all legs passed (1 skipped)' "$TMP/out"; then
  report 0 'missing host capability counts a skip'
else report 1 'missing host capability counts a skip'; fi
OCANNL_TOOL_HARNESS_STRICT=1 MODE=skip probe; rc=$?
if [ "$rc" = 1 ] && grep -q '^FAIL  every leg requiring definitely_missing_harness_tool$' "$TMP/out" \
   && grep -q '1 leg(s) failed (0 skipped)' "$TMP/out"; then
  report 0 'strict mode turns an undecided leg into a failure'
else report 1 'strict mode turns an undecided leg into a failure'; fi
# A strict run's list of expected platform skips (gh-ocannl-1236): the one
# exact HARNESS<TAB>LABEL line it names is a counted, recorded skip; the
# other skip of the same member stays red, and so does the listed one when
# the harness name, or the record the aggregator reads back, is missing.
printf '# comment\nprobe\tfirst platform check\nprobe first platform check\n' >"$TMP/expected"
expected_probe() { # NAME RECORD
  : >"$TMP/skip-record"
  OCANNL_TOOL_HARNESS_STRICT=1 OCANNL_TOOL_HARNESS_EXPECTED_SKIPS=$TMP/expected \
    OCANNL_TOOL_HARNESS_NAME=$1 OCANNL_TOOL_HARNESS_SKIP_RECORD=$2 MODE=skips probe
}
expected_probe probe "$TMP/skip-record"; rc=$?
if [ "$rc" = 1 ] && grep -q '^SKIP  first platform check$' "$TMP/out" \
   && grep -q '^      no first platform (expected: listed in expected)$' "$TMP/out" \
   && grep -q '^FAIL  second platform check$' "$TMP/out" \
   && grep -q '1 leg(s) failed (1 skipped)' "$TMP/out" \
   && [ "$(cat "$TMP/skip-record")" = "probe	first platform check" ]; then
  report 0 'strict mode counts and records only the skip its list names'
else report 1 'strict mode counts and records only the skip its list names' "exit $rc; $(cat "$TMP/out")"; fi
for control in name record; do
  case $control in
    name) expected_probe other "$TMP/skip-record"; rc=$? ;;
    record) expected_probe probe ''; rc=$? ;;
  esac
  if [ "$rc" = 1 ] && grep -q '^FAIL  first platform check$' "$TMP/out" \
     && grep -q '2 leg(s) failed (0 skipped)' "$TMP/out" && [ ! -s "$TMP/skip-record" ]; then
    report 0 "strict mode lists nothing without the matching $control"
  else report 1 "strict mode lists nothing without the matching $control" "exit $rc; $(cat "$TMP/out")"; fi
done
MODE=fail probe; rc=$?
if [ "$rc" = 1 ] && grep -q '1 leg(s) failed (0 skipped)' "$TMP/out" && ! grep -q 'aborted before finish' "$TMP/out"; then
  report 0 'failure controls exit and footer'
else report 1 'failure controls exit and footer'; fi
probe --keep; rc=$?
path=$(cat "$RECORD")
if [ "$rc" = 0 ] && [ -d "$path" ] && grep -q '^kept ' "$TMP/out"; then
  report 0 '--keep retains and identifies scratch'
else report 1 '--keep retains and identifies scratch'; fi
rm -rf "$path"
probe --help; rc=$?
if [ "$rc" = 0 ] && grep -q '^Last help line.$' "$TMP/out"; then
  report 0 '--help reads the full header'
else report 1 '--help reads the full header'; fi
probe --unknown; rc=$?
report "$([ "$rc" = 2 ]; echo $?)" 'unknown arguments exit 2'
MODE=signal probe; rc=$?
path=$(cat "$RECORD")
if [ "$rc" = 143 ] && [ ! -d "$path" ] && grep -q '^aborted before finish (rc=143)$' "$TMP/out"; then report 0 'TERM exits 143 and cleans scratch'
else report 1 'TERM exits 143 and cleans scratch'; fi
for mode in abort errexit unfinished; do
  case $mode in abort) want=50 ;; errexit) want=7 ;; unfinished) want=0 ;; esac
  MODE=$mode probe; rc=$?
  path=$(cat "$RECORD")
  if [ "$rc" = "$want" ] && [ ! -d "$path" ] \
     && grep -q "^aborted before finish (rc=$want)$" "$TMP/out" \
     && ! grep -q 'legs passed\|leg(s) failed' "$TMP/out"; then
    report 0 "$mode reports unfinished harness and preserves exit $want"
  else report 1 "$mode reports unfinished harness and preserves exit $want"; fi
done
MODE=abort probe --keep; rc=$?
path=$(cat "$RECORD")
if [ "$rc" = 50 ] && [ -d "$path" ] && grep -q '^aborted before finish (rc=50)$' "$TMP/out"; then
  report 0 'aborted --keep retains scratch and reports the original status'
else report 1 'aborted --keep retains scratch and reports the original status'; fi
rm -rf "$path"
# The leg filter: unfiltered, every leg runs and the full-pass line is earned;
# filtered, only matching bodies run, the rest say FILTERED, and the footer is
# never the full-pass line; a filter that selects nothing exits 2.
legs_probe() { # expected bodies run (space-separated), then probe arguments
  local want=$1 rc=0
  shift
  rm -f "$RECORD.ran"
  MODE=legs probe "$@" || rc=$?
  ran=
  [ ! -f "$RECORD.ran" ] || ran=$(tr '\n' ' ' <"$RECORD.ran" | sed 's/ $//')
  [ "$ran" = "$want" ] || { printf 'bodies run: "%s", wanted "%s"\n' "$ran" "$want" >>"$TMP/out"; rc=99; }
  return "$rc"
}
legs_probe 'alpha beta gamma'; rc=$?
if [ "$rc" = 0 ] && grep -q 'all legs passed (0 skipped)' "$TMP/out" && ! grep -q '^FILTERED' "$TMP/out" \
   && [ "$(grep -c '^PASS  .* body$' "$TMP/out")" = 3 ]; then
  report 0 'unfiltered: every leg runs and earns the full-pass line'
else report 1 'unfiltered: every leg runs and earns the full-pass line' "exit $rc; $(cat "$TMP/out")"; fi
legs_probe 'alpha' --only alpha; rc=$?
if [ "$rc" = 0 ] && grep -q '^PASS  alpha body$' "$TMP/out" \
   && grep -q '^FILTERED  beta leg #even$' "$TMP/out" && grep -q '^FILTERED  gamma leg #odd$' "$TMP/out" \
   && ! grep -q 'all legs passed' "$TMP/out" \
   && grep -q "^FILTERED RUN, not a full pass: 1 leg(s) selected, 2 filtered out by --only 'alpha'; no failure (0 skipped)$" "$TMP/out"; then
  report 0 '--only runs the matching leg alone and never prints the full-pass line'
else report 1 '--only runs the matching leg alone and never prints the full-pass line' "exit $rc; $(cat "$TMP/out")"; fi
legs_probe 'alpha gamma' '--only=^(alpha|gamma) leg'; rc=$?
if [ "$rc" = 0 ] && grep -q '^FILTERED  beta leg #even$' "$TMP/out" && ! grep -q 'all legs passed' "$TMP/out" \
   && grep -q 'FILTERED RUN, not a full pass: 2 leg(s) selected, 1 filtered out' "$TMP/out"; then
  report 0 '--only=REGEX is an extended regular expression over leg names'
else report 1 '--only=REGEX is an extended regular expression over leg names' "exit $rc; $(cat "$TMP/out")"; fi
GAMMA_RC=1 legs_probe 'gamma' --only gamma; rc=$?
if [ "$rc" = 1 ] && grep -q '^FAIL  gamma body$' "$TMP/out" \
   && grep -q "^1 leg(s) failed (0 skipped); FILTERED RUN: 1 leg(s) selected, 2 filtered out by --only 'gamma'$" "$TMP/out"; then
  report 0 'a failing selected leg exits 1 and its footer says the run was filtered'
else report 1 'a failing selected leg exits 1 and its footer says the run was filtered' "exit $rc; $(cat "$TMP/out")"; fi
# A `#group` tag closing a leg's name selects the group through the same
# filter, and a filtered run accounts for its coverage before the footer:
# each leg that ran, then each skipped check with its reason.
BETA_SKIP=1 legs_probe 'alpha beta gamma' --only '#odd|#even'; rc=$?
account=$(sed -n '/^ran 3 leg(s):$/,$p' "$TMP/out")
want_account="ran 3 leg(s):
  RAN   alpha leg #odd
  RAN   beta leg #even
  RAN   gamma leg #odd
skipped 1 check(s):
  SKIP  beta check -- no beta host
FILTERED RUN, not a full pass: 3 leg(s) selected, 0 filtered out by --only '#odd|#even'; no failure (1 skipped)"
if [ "$rc" = 0 ] && [ "$account" = "$want_account" ]; then
  report 0 'a filtered run accounts for each leg it ran and each check it skipped'
else report 1 'a filtered run accounts for each leg it ran and each check it skipped' "exit $rc; $(cat "$TMP/out")"; fi
legs_probe 'alpha gamma' --only '#odd'; rc=$?
if [ "$rc" = 0 ] && grep -q '^FILTERED  beta leg #even$' "$TMP/out" \
   && grep -q '^  RAN   alpha leg #odd$' "$TMP/out" && grep -q '^  RAN   gamma leg #odd$' "$TMP/out" \
   && ! grep -q '^skipped ' "$TMP/out"; then
  report 0 'a #group tag selects its legs whole'
else report 1 'a #group tag selects its legs whole' "exit $rc; $(cat "$TMP/out")"; fi
BETA_SKIP=1 legs_probe 'alpha beta gamma'; rc=$?
if [ "$rc" = 0 ] && grep -q 'all legs passed (1 skipped)' "$TMP/out" \
   && ! grep -q '^ran \|^  RAN \|^skipped ' "$TMP/out"; then
  report 0 'an unfiltered run keeps its output: no coverage account'
else report 1 'an unfiltered run keeps its output: no coverage account' "exit $rc; $(cat "$TMP/out")"; fi
OCANNL_TOOL_HARNESS_STRICT=1 legs_probe 'beta' --only beta; rc=$?
if [ "$rc" = 0 ] && grep -q 'FILTERED RUN, not a full pass: 1 leg(s) selected, 2 filtered out' "$TMP/out" \
   && ! grep -q '^FAIL' "$TMP/out"; then
  report 0 'a filtered leg is not a capability skip, so strict mode leaves it alone'
else report 1 'a filtered leg is not a capability skip, so strict mode leaves it alone' "exit $rc; $(cat "$TMP/out")"; fi
legs_probe '' --only 'no such leg'; rc=$?
if [ "$rc" = 2 ] && [ "$(grep -c '^FILTERED  ' "$TMP/out")" = 3 ] \
   && grep -q "^--only 'no such leg' selected no leg (3 filtered out): nothing was tested$" "$TMP/out" \
   && ! grep -q 'legs passed\|leg(s) failed\|FILTERED RUN\|aborted before finish' "$TMP/out"; then
  report 0 'a filter matching no leg fails loudly with exit 2'
else report 1 'a filter matching no leg fails loudly with exit 2' "exit $rc; $(cat "$TMP/out")"; fi
probe --only pass; rc=$?
if [ "$rc" = 2 ] && grep -q "^--only 'pass': this run reached no leg gate" "$TMP/out" \
   && ! grep -q 'legs passed\|FILTERED RUN' "$TMP/out"; then
  report 0 'a filter over a harness that declares no legs fails loudly'
else report 1 'a filter over a harness that declares no legs fails loudly' "exit $rc; $(cat "$TMP/out")"; fi
for bad in "--only" "--only=" "--only=("; do
  legs_probe '' $bad; rc=$?
  if [ "$rc" = 2 ] && grep -q -- '--only' "$TMP/out" && ! grep -q '^PASS\|^FILTERED' "$TMP/out"; then
    report 0 "'$bad' is refused before any leg runs"
  else report 1 "'$bad' is refused before any leg runs" "exit $rc; $(cat "$TMP/out")"; fi
done
reason() { echo 'claimed defect'; return 1; }
wrong_reason() { echo 'unrelated defect'; return 1; }
wrong_status() { echo 'claimed defect'; return 127; }
accepted() { echo 'claimed defect'; return 0; }
if harness_rejected 1 '^claimed defect$' "$TMP/rejection" reason; then
  report 0 'in-process twin rejects for its claimed reason'
else report 1 'in-process twin rejects for its claimed reason'; fi
for control in wrong_reason wrong_status accepted; do
  if harness_rejected 1 '^claimed defect$' "$TMP/rejection" "$control"; then
    report 1 "rejection refuses $control"
  else report 0 "rejection refuses $control"; fi
done
# The positive lifetime test owns an unreaped child. The independent timeout
# must kill/reap it and record exactly one failure, rather than hang on wait.
( sleep 30 ) & child=$!
never_dead() { return 1; }
before=$failures
harness_wait_child "$child" never_dead 'injected deadline' 1 >"$TMP/deadline"
if [ "$failures" = "$((before + 1))" ] && [ "$harness_child_rc" = 137 ]; then
  failures=$before
  report 0 'owned-child deadline kills and reaps the timed-out child'
else report 1 'owned-child deadline kills and reaps the timed-out child'; fi
finish
