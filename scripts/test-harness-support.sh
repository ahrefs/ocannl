#!/usr/bin/env bash
# Exercise shared harness lifecycle, counted skips and reason-specific rejection.
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
  fail) report 1 'injected failure' ;;
  signal) kill -TERM "$$" ;;
esac
finish
PROBE
export SUPPORT=$HERE/harness-support.sh RECORD=$TMP/record MODE=pass
probe() {
  local rc=0
  bash "$TMP/probe.sh" "$@" >"$TMP/out" 2>&1 || rc=$?
  return "$rc"
}
probe; rc=$?
path=$(cat "$RECORD")
if [ "$rc" = 0 ] && [ ! -d "$path" ] && grep -q 'all legs passed (0 skipped)' "$TMP/out"; then
  report 0 'success cleans scratch and counts zero skips'
else report 1 'success cleans scratch and counts zero skips'; fi
MODE=skip probe; rc=$?
if [ "$rc" = 0 ] && grep -q '^SKIP ' "$TMP/out" && grep -q 'all legs passed (1 skipped)' "$TMP/out"; then
  report 0 'missing host capability counts a skip'
else report 1 'missing host capability counts a skip'; fi
MODE=fail probe; rc=$?
if [ "$rc" = 1 ] && grep -q '1 leg(s) failed (0 skipped)' "$TMP/out"; then
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
if [ "$rc" = 143 ] && [ ! -d "$path" ]; then report 0 'TERM exits 143 and cleans scratch'
else report 1 'TERM exits 143 and cleans scratch'; fi
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
