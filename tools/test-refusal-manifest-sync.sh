#!/usr/bin/env bash
# Hermetic tests for tools/refusal-manifest-sync.sh's run-and-promote loop and its offers
# (gh-ocannl-1571).
#
#   tools/test-refusal-manifest-sync.sh          # run every leg
#   tools/test-refusal-manifest-sync.sh --keep   # keep the scratch directory
#   tools/test-refusal-manifest-sync.sh --only '^2:'
#
# The subject is copied into a scratch repository of each leg's own -- it resolves its repository
# as `dirname $0/..` -- beside stand-ins for everything it calls: a `dune` on PATH whose engine
# prints a canned plan and records the `--offers` call, a tools/test-run.sh that "runs" goldens
# by computing each one's output from the scratch tree and records dune's promotion list as a real
# run directory does, and a tools/promote.sh that copies a correction over its golden. So the loop
# is pinned on the shapes it must handle -- a chain whose second correction appears only after the
# first is promoted, a diff that never settles, a failure no promotion settles, a correction
# outside its goldens -- without a dune build, and with nothing of this repository touched. The
# engine's own planning and the offer lines it reads are the OCaml tests'
# (refusal_manifest_sync_cases, refusal_control_scan_cases).
#
# Legs:
#   1. a two-step correction chain converges in ONE invocation: both goldens promoted, a round
#      each, and the third run passes (exit 0).
#   2. the negative control: the same chain under `--rounds 2` -- the loop before gh-ocannl-1571,
#      run, promote, run -- is STALE (exit 3), naming the golden still corrected.
#   3. a diff that never settles stops at the bound, five runs by default, STALE.
#   4. a failure with no correction to promote is explained, exits with the run's status, and
#      promotes nothing.
#   5. a correction dune lists outside the sync's goldens is named and never promoted.
#   6. the first run is forced and the later ones are not, and every run's log reaches the offers,
#      through a `dune exec --no-build`: a build there would replace dune's promotion list.
#   7. --rounds refuses anything but a number of at least 2, before anything runs.
#   8. a plan naming no golden runs nothing and hands nothing to the offers.
#   1b. leg 1 under the host's /bin/bash, which on macOS is Bash 3.2: an empty array expanded under
#      `set -u` is an unbound variable there, which ended the second round.

set -u

. "$(cd "$(dirname "$0")/../scripts" && pwd)/harness-support.sh"
harness_args "$@"

HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="$HERE/refusal-manifest-sync.sh"
[ -f "$SRC" ] || { echo "no $SRC" >&2; exit 2; }
harness_require git
harness_scratch "test-refusal-manifest-sync"

echo "testing $SRC"

# A scratch repository REPO with the subject, what it sources, and the stand-ins. SCENARIO is the
# body of `outputs`, sourced by the stand-in runner on each run with ROUND set: it writes each
# golden's output to out/<golden path>.
fixture() { # REPO SCENARIO PLAN
  local repo=$1
  mkdir -p "$repo/tools" "$repo/bin" "$repo/test/support" "$repo/test/operations" "$repo/runs" \
    "$repo/out/test/operations"
  cp "$SRC" "$repo/tools/refusal-manifest-sync.sh"
  cp "$HERE/credential-env.sh" "$repo/tools/credential-env.sh"
  echo '(* manifest *)' >"$repo/test/support/refusal_control_manifest.ml"
  printf '%s\n' "$2" >"$repo/scenario"
  printf '%s' "$3" >"$repo/plan"
  echo 0 >"$repo/round"
  cat >"$repo/bin/dune" <<'STUB'
#!/usr/bin/env bash
repo=$(cd "$(dirname "$0")/.." && pwd)
echo "dune $*" >>"$repo/calls"
case $1 in
  build) exit 0 ;;
  exec)
    shift
    while [ "$1" != -- ]; do shift; done
    shift
    if [ "${1-}" = --offers ]; then shift; echo "offers $*" >>"$repo/calls"; exit 0; fi
    cat "$repo/plan" ;;
esac
STUB
  cat >"$repo/tools/test-run.sh" <<'STUB'
#!/usr/bin/env bash
repo=$(cd "$(dirname "$0")/.." && pwd)
cd "$repo" || exit 2
if [ "$1" = paths ]; then echo "$repo/runs/$(cat round)"; exit 0; fi
echo "test-run $*" >>calls
ROUND=$(($(cat round) + 1))
echo "$ROUND" >round
run=runs/$ROUND
mkdir -p "$run"
echo "run $ROUND" >"$run/log"
. ./scenario
: >"$run/promotions"
rc=0
for out in $(cd out && find . -type f | sort); do
  path=${out#./}
  cmp -s "out/$path" "$path" 2>/dev/null && continue
  echo "$path" >>"$run/promotions"
  rc=1
done
[ -f "$run/fail" ] && rc=1
exit $rc
STUB
  cat >"$repo/tools/promote.sh" <<'STUB'
#!/usr/bin/env bash
repo=$(cd "$(dirname "$0")/.." && pwd)
cd "$repo" || exit 2
echo "promote $*" >>calls
for path; do cp "out/$path" "$path"; done
STUB
  chmod +x "$repo/bin/dune" "$repo/tools/test-run.sh" "$repo/tools/promote.sh"
}

sync() { # REPO [ARGS...] -- runs the subject, its output in REPO/stdout, status in REPO/rc
  local repo=$1
  shift
  (cd "$repo" && PATH="$repo/bin:$PATH" ${SUBJECT_BASH:-bash} tools/refusal-manifest-sync.sh "$@" >stdout 2>&1)
  echo $? >"$repo/rc"
}

two_goldens='rerun: @test/operations/runtest-scanner
rerun: @test/operations/runtest-audit
'
# The scanner's golden gains a marker; the audit's output is what it reads of the scanner's golden,
# so its correction appears only once the scanner's is promoted.
chain='echo "marker: new" >out/test/operations/scanner.expected
{ echo "audit saw:"; cat test/operations/scanner.expected; } >out/test/operations/audit.expected'
seed_chain() { # REPO
  echo "marker: old" >"$1/test/operations/scanner.expected"
  printf 'audit saw:\nmarker: old\n' >"$1/test/operations/audit.expected"
}

leg_1() {
  local r=$TMP/1${SUBJECT_BASH:+b} label="1${SUBJECT_BASH:+b}: a two-step correction chain converges in one invocation${SUBJECT_BASH:+ under $SUBJECT_BASH}"
  fixture "$r" "$chain" "$two_goldens"
  seed_chain "$r"
  sync "$r"
  if [ "$(cat "$r/rc")" = 0 ] && [ "$(cat "$r/round")" = 3 ] &&
    grep -qx 'promote test/operations/scanner.expected' "$r/calls" &&
    grep -qx 'promote test/operations/audit.expected' "$r/calls" &&
    grep -q 'round 1: promoting test/operations/scanner.expected' "$r/stdout" &&
    grep -q 'round 2: promoting test/operations/audit.expected' "$r/stdout" &&
    printf 'audit saw:\nmarker: new\n' | cmp -s - "$r/test/operations/audit.expected"; then
    report 0 "$label"
  else
    report 1 "$label" "see $r (rc $(cat "$r/rc"))"
  fi
}
harness_leg_run '1: a two-step correction chain converges in one invocation' leg_1

leg_1b() { SUBJECT_BASH=/bin/bash leg_1; }
harness_leg_run "1b: the chain converges under the host's /bin/bash" leg_1b

leg_2() {
  local r=$TMP/2
  fixture "$r" "$chain" "$two_goldens"
  seed_chain "$r"
  sync "$r" --rounds 2
  if [ "$(cat "$r/rc")" = 3 ] && [ "$(cat "$r/round")" = 2 ] &&
    grep -q 'STALE: after 2 runs' "$r/stdout" && grep -qx '  test/operations/audit.expected' "$r/stdout" &&
    ! grep -qx 'promote test/operations/audit.expected' "$r/calls"; then
    report 0 "2: the same chain run, promoted and run once more is left stale (negative control)"
  else
    report 1 "2: the same chain run, promoted and run once more is left stale (negative control)" \
      "see $r (rc $(cat "$r/rc"))"
  fi
}
harness_leg_run '2: the same chain run, promoted and run once more is left stale' leg_2

leg_3() {
  local r=$TMP/3
  fixture "$r" 'echo "run $ROUND" >out/test/operations/scanner.expected' \
    'rerun: @test/operations/runtest-scanner
'
  echo seed >"$r/test/operations/scanner.expected"
  sync "$r"
  if [ "$(cat "$r/rc")" = 3 ] && [ "$(cat "$r/round")" = 5 ] &&
    [ "$(grep -c '^promote ' "$r/calls")" = 4 ] && grep -q 'STALE: after 5 runs' "$r/stdout"; then
    report 0 "3: a diff that never settles stops at five runs, stale"
  else
    report 1 "3: a diff that never settles stops at five runs, stale" "see $r (rc $(cat "$r/rc"))"
  fi
}
harness_leg_run '3: a diff that never settles stops at the bound' leg_3

leg_4() {
  local r=$TMP/4
  fixture "$r" 'touch "$run/fail"' "$two_goldens"
  sync "$r"
  if [ "$(cat "$r/rc")" = 1 ] && [ "$(cat "$r/round")" = 1 ] && ! grep -q '^promote' "$r/calls" &&
    grep -q 'round 1 failed with no correction of its goldens to promote' "$r/stdout"; then
    report 0 "4: a failure no promotion settles is explained, with the run's status"
  else
    report 1 "4: a failure no promotion settles is explained, with the run's status" \
      "see $r (rc $(cat "$r/rc"))"
  fi
}
harness_leg_run '4: a failure no promotion settles is explained' leg_4

leg_5() {
  local r=$TMP/5
  fixture "$r" 'echo corrected >out/test/operations/other.expected' "$two_goldens"
  echo stale >"$r/test/operations/other.expected"
  sync "$r"
  if [ "$(cat "$r/rc")" = 1 ] && ! grep -q '^promote' "$r/calls" &&
    grep -q 'correction of test/operations/other.expected, outside the goldens' "$r/stdout" &&
    [ "$(cat "$r/test/operations/other.expected")" = stale ]; then
    report 0 "5: a correction outside the sync's goldens is named, never promoted"
  else
    report 1 "5: a correction outside the sync's goldens is named, never promoted" \
      "see $r (rc $(cat "$r/rc"))"
  fi
}
harness_leg_run '5: a correction outside the goldens is named, never promoted' leg_5

leg_6() {
  local r=$TMP/6
  fixture "$r" "$chain" "$two_goldens"
  seed_chain "$r"
  sync "$r"
  local runs logs
  # The subject names its runs by the stand-in's `pwd`, which normalizes the scratch path.
  logs=$(cd "$r/runs" && pwd)
  runs=$(grep '^test-run run build' "$r/calls")
  if [ "$(printf '%s\n' "$runs" | grep -c -- ' --force ')" = 1 ] &&
    printf '%s\n' "$runs" | head -1 | grep -q -- 'build --force @test/operations/runtest-scanner' &&
    grep -qx "offers $logs/1/log $logs/2/log $logs/3/log" "$r/calls" &&
    grep -q '^dune exec --no-build .* -- --offers ' "$r/calls"; then
    report 0 "6: the first run alone is forced, and every run's log reaches the offers"
  else
    report 1 "6: the first run alone is forced, and every run's log reaches the offers" "see $r"
  fi
}
harness_leg_run '6: the first run alone is forced, every log reaches the offers' leg_6

leg_7() {
  local r=$TMP/7 bad ok=1
  fixture "$r" "$chain" "$two_goldens"
  for bad in 1 0 x ''; do
    rm -f "$r/calls"
    sync "$r" --rounds "$bad"
    [ "$(cat "$r/rc")" = 2 ] && [ ! -e "$r/calls" ] || ok=0
  done
  rm -f "$r/calls"
  sync "$r" --rounds
  [ "$(cat "$r/rc")" = 2 ] && [ ! -e "$r/calls" ] || ok=0
  if [ "$ok" = 1 ]; then
    report 0 "7: --rounds refuses anything but a number of at least 2, before anything runs"
  else
    report 1 "7: --rounds refuses anything but a number of at least 2, before anything runs" "see $r"
  fi
}
harness_leg_run '7: --rounds refuses a bad bound before anything runs' leg_7

leg_8() {
  local r=$TMP/8
  fixture "$r" "$chain" 'nothing to sync
'
  sync "$r"
  if [ "$(cat "$r/rc")" = 0 ] && ! grep -q '^test-run\|^offers' "$r/calls"; then
    report 0 "8: a plan naming no golden runs nothing"
  else
    report 1 "8: a plan naming no golden runs nothing" "see $r (rc $(cat "$r/rc"))"
  fi
}
harness_leg_run '8: a plan naming no golden runs nothing' leg_8

finish
