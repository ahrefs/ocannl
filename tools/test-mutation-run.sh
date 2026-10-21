#!/usr/bin/env bash
# Exercise the shipping mutation runner and real test-run supervisor in an
# isolated repository fixture, with a deterministic fake dune on PATH.
# Usage: tools/test-mutation-run.sh [--keep] [--only REGEX] [--list-legs]
# Each leg is a function run by `harness_leg_run 'N: ...' leg_N`
# (scripts/harness-support.sh) that starts from its own module, patch and
# probe mode, so any one passes run alone (tools/harness-legs-alone.sh); the
# fixture repository, the fake dune and the goldens are shared. A failed check
# still ends the harness (`set -e`), naming the leg it was in.
set -eu
root=$(cd "$(dirname "$0")/.." && pwd)
. "$root/scripts/harness-support.sh"
harness_args "$@"
harness_require perl git
harness_scratch mutation-tests
fixture=$TMP
mkdir -p "$fixture/repo/tools" "$fixture/repo/scripts" "$fixture/bin" "$fixture/runs"
cp "$root/tools/mutation-run.sh" "$root/tools/test-run.sh" "$fixture/repo/tools/"
# Whatever the shipping test-run.sh SOURCES, derived from the script itself
# rather than listed here: it dies at startup on a file the fixture lacks, and
# the failure surfaces as this harness's own exit 2 where a 1 was expected --
# which is how gh-ocannl-983's `. tools/box-jobs.sh` reddened CI. The floor
# makes a regex that stopped matching fail loudly instead of staging nothing.
staged=0
while IFS= read -r rel; do
  mkdir -p "$fixture/repo/$(dirname "$rel")"
  cp "$root/$rel" "$fixture/repo/$rel"
  staged=$((staged + 1))
done < <(sed -n 's/^\. \([A-Za-z0-9_./-]*\)$/\1/p' "$root/tools/test-run.sh")
[ "$staged" -ge 2 ] ||
  { echo "only $staged sourced file(s) found in tools/test-run.sh; the scan is broken" >&2; exit 2; }
cat > "$fixture/bin/dune" <<'DUNE'
#!/usr/bin/env bash
set -eu
# test-run.sh hands the batch a trace file of the run's own, as the second
# word; that one word, and only it, is not part of the command this fixture
# recognises (a second copy, or one anywhere else, still fails the match).
if [ -n "${OCANNL_TOOL_TESTRUN_OWN:-}" ] && [ $# -ge 2 ] &&
   [ "$2" = "--trace-file=$OCANNL_TOOL_TESTRUN_OWN/trace.csexp" ]; then
  sub=$1
  shift 2
  set -- "$sub" "$@"
fi
case "$*" in
  'build -j 4 @runtest-probe') name=probe ;;
  'build -j 4 @runtest-manifest') name=manifest ;;
  'build -j 4 @runtest-tail') name=tail ;;
  'build -j 4 @runtest-sections') name=sections ;;
  'build -j 4 @runtest-trailer') name=trailer ;;
  'build -j 4 @runtest-replay') name=replay ;;
  'build -j 4 @runtest-capture') name=capture ;;
  *) exit 98 ;;
esac
printf '%s\n' invoked >> "$PROBE_HOME/invocations"
out=${DUNE_BUILD_DIR:-_build}/default
# The unmutated module is the runner's baseline: green, writing nothing, unless the case says
# otherwise -- red, as a golden not promoted after a change to the test leaves it, or green
# having rewritten the test's stdout, as a real baseline does.
case "$(cat module.ml)" in
  *MUTATED*) ;;
  *)
    printf '%s\n' baseline >> "$PROBE_HOME/baselines"
    case "${PROBE_BASELINE:-green}" in
      green) exit 0 ;;
      rewrote) mkdir -p "$out"; rm -f "$out/$name.exe.output"; cp "$name.expected" "$out/$name.exe.output"; exit 0 ;;
      red) printf 'FAIL: first: false\nFAILED: 1 check did not hold: "first".\n'; exit 1 ;;
      sleep) touch "$PROBE_HOME/ready"; exec perl -e 'sleep 60' ;;
      *) exit 99 ;;
    esac ;;
esac
cp module.ml "$PROBE_HOME/observed"
# The test's stdout, where dune leaves it even when the test exits nonzero; the
# golden, probe.expected, has three rows, manifest.expected six, two of them
# refusal-manifest marker rows, and trailer.expected eight: manifest.expected's
# with a blank row and a success trailer (`OK: `) before its section.
mkdir -p "$out"
rows() { rm -f "$out/$name.exe.output"; printf "$1" > "$out/$name.exe.output"; }
m1='  [scanner-refusal:00000000000000000000000000000001] first refusal\n'
m2='  [scanner-refusal:00000000000000000000000000000002] last refusal\n'
section='\nSynthetic controls: scanner refusal diagnostics exercised by this control golden:\n'
closing='End of synthetic controls.\n'
case "$PROBE_MODE" in
  fail)
    # Two Verdict.fail messages ending like claims, on both streams as Verdict.fail prints them.
    rows 'first: false\nFAIL: a message ending like a claim: false\nnested: label: false\nlast: true\nFAIL: one ending like pass_fail: FAIL\n'
    printf 'FAIL: first: false\nFAIL: a message ending like a claim: false\nFAIL: nested: label: false\nFAIL: first: false\n'
    printf 'FAIL: one ending like pass_fail: FAIL\n'
    printf 'prefix FAIL: decoy: false\nFAIL: true claim: true\nFAIL: suffix: false extra\n'
    exit 1 ;;
  pass) rows 'first: true\nnested: label: true\nlast: true' ; exit 0 ;;
  # A raise after a failed claim, as Verdict reports it (gh-ocannl-1067): one
  # row printed of three, the other two never evaluated.
  raise)
    rows 'first: false\n'
    printf 'FAIL: first: false\nFAILED: 1 check did not hold: "first".\n'
    printf 'STOPPED EARLY: an uncaught exception ended the run, so no check after it ran:\n'
    printf 'Fatal error: exception Failure("injected")\n'
    exit 1 ;;
  # Each signal alone: rows short with no Verdict line, Verdict's line with every row.
  short) rows 'first: false\nnested: label: true\n'; printf 'FAIL: first: false\n'; exit 1 ;;
  stopped) rows 'first: false\nnested: label: true\nlast: true\n'
    printf 'FAIL: first: false\nSTOPPED EARLY: an uncaught exception ended the run\n'; exit 1 ;;
  more) rows 'first: false\nFAIL: a message\nextra row\nnested: label: true\nlast: true\n'
    printf 'FAIL: first: false\n'; exit 1 ;;
  # A golden of captured diagnostics, one reworded by the mutant: its row still printed.
  capture) rows 'FAIL: the first diagnostic\nFAIL: the second, reworded\nFAIL: a third, new\n'
    printf 'the golden differs\n'; exit 1 ;;
  # A run cut short after a multi-line failure message: its `FAIL: ` rows are no rows of the
  # golden, so they cannot make up the rows it never printed (gh-ocannl-1364).
  failpad) rows 'first: false\nFAIL: the message:\nFAIL:   one\nFAIL:   two\n'
    printf 'FAIL: first: false\nFAIL: the message:\nFAIL:   one\nFAIL:   two\n'; exit 1 ;;
  # A raising Verdict.case (gh-ocannl-1084): one failed claim in place of the
  # rows the raise kept from printing, and the cases after it still run.
  casemid) rows 'nested: the case ran to completion (raised Failure("injected")): false\nlast: true\n'
    printf 'FAIL: nested: the case ran to completion (raised Failure("injected")): false\n'
    printf 'FAILED: 1 check did not hold: "nested: the case ran to completion".\n'; exit 1 ;;
  caselast) rows 'first: true\nlast: the case ran to completion (raised Failure("injected")): false\n'
    printf 'FAIL: last: the case ran to completion (raised Failure("injected")): false\n'
    printf 'FAILED: 1 check did not hold: "last: the case ran to completion".\n'; exit 1 ;;
  # The same raise, then a crash: no teardown line, so nothing says a later case ran.
  casecrash) rows 'nested: the case ran to completion (raised Failure("injected")): false\n'
    printf 'FAIL: nested: the case ran to completion (raised Failure("injected")): false\n'
    printf 'Command got signal SEGV.\n'; exit 1 ;;
  # The same raise, then an exit inside a later case: Verdict says so, since the
  # stdout alone ends on a raise exactly as a run whose last case raised does.
  caseexit) rows 'nested: the case ran to completion (raised Failure("injected")): false\n'
    printf 'FAIL: nested: the case ran to completion (raised Failure("injected")): false\n'
    printf 'FAILED: 1 check did not hold: "nested: the case ran to completion".\n'
    printf 'STOPPED EARLY: an exit inside case "last" ended the run, so no case after it ran\n'
    exit 1 ;;
  # A teardown, but the stdout ends on neither the golden's last row nor a raise.
  casecut) rows 'first: the case ran to completion (raised Failure("injected")): false\nnested: label: true\n'
    printf 'FAIL: first: the case ran to completion (raised Failure("injected")): false\n'
    printf 'FAILED: 1 check did not hold: "first: the case ran to completion".\n'; exit 1 ;;
  # A raise, then a failure message and nothing more: the stdout ends on the message, not on the
  # raise, however few rows the message counts for.
  casefailmsg) rows 'first: the case ran to completion (raised Failure("injected")): false\nFAIL: a message\n'
    printf 'FAIL: first: the case ran to completion (raised Failure("injected")): false\nFAIL: a message\n'
    printf 'FAILED: 2 checks did not hold: "first: the case ran to completion", "a message".\n'; exit 1 ;;
  # The test never ran this time: the output a previous run left must not count.
  stale) printf 'FAIL: first: false\n'; exit 1 ;;
  reused) exit 0 ;;
  both) rows 'first: false\nnested: label: true\nlast: true\n'
    cp "$out/$name.exe.output" "$out/$name.actual"; printf 'FAIL: first: false\n'; exit 1 ;;
  # A failed claim's refusal marker is not printed (gh-ocannl-1216): a row short of
  # the golden, every other row printed -- the failed claim the first or the last.
  omitted) rows "first: false\n$section$m2"'last: true\n'
    printf 'FAIL: first: false\nFAILED: 1 check did not hold: "first".\n'; exit 1 ;;
  omittedlast) rows "first: true\n$section$m1"'last: false\n'
    printf 'FAIL: last: false\nFAILED: 1 check did not hold: "last".\n'; exit 1 ;;
  # The same shortfall from a run cut short: the golden's last row never printed,
  # or every row did but no teardown says the process ended through Verdict.
  omittedcut) rows "first: false\n$section$m2"
    printf 'FAIL: first: false\nFAILED: 1 check did not hold: "first".\n'; exit 1 ;;
  omittedkilled) rows "first: false\n$section$m2"'last: true\n'
    printf 'FAIL: first: false\nCommand got signal SEGV.\n'; exit 1 ;;
  # A row swapped for another while a marker is missing: the counts and the last row agree,
  # the rows do not.
  omittedswap) rows "first: false\nswapped in$section$m2"'last: true\n'
    printf 'FAIL: first: false\nFAILED: 1 check did not hold: "first".\n'; exit 1 ;;
  # tail.expected ends on its section's closing row, as every real manifest golden does
  # (gh-ocannl-1251, gh-ocannl-1318): a run cut inside the markers lacks that row, teardown or
  # not; a mutant silencing one refusal while every claim holds prints it after one marker
  # fewer, with no teardown, as does a failing run with one; and a run printing the row but
  # dropping another, or ending on a row of its own after it, is not the golden's ending.
  tailcut) rows "first: false\n$section$m1"
    printf 'FAIL: first: false\nFAILED: 1 check did not hold: "first".\n'; exit 1 ;;
  tailsilenced) rows "first: true\n$section$m1$closing"; printf 'the golden differs\n'; exit 1 ;;
  tailfailed) rows "first: false\n$section$m2$closing"
    printf 'FAIL: first: false\nFAILED: 1 check did not hold: "first".\n'; exit 1 ;;
  tailswap) rows "swapped in\n$section$m1$closing"; printf 'the golden differs\n'; exit 1 ;;
  # sections.expected holds two sections, each ending on the same closing row: a run cut after
  # the first one ends on a closing row too, but not on the golden's last.
  sectionscut) rows "first: true\n$section$m1$closing"; exit 1 ;;
  tailafter) rows "first: true\n$section$m1${closing}FAIL: a message after the section\n"
    printf 'FAIL: a message after the section\n'; exit 1 ;;
  # A failed claim leaves no success trailer (`OK: ` rows a scan prints only when no claim
  # failed), nor the blank row before it; nor, here, its own marker. The blank row kept, and
  # every marker printed (a failed claim with none). Then the same shortfall cut short, with
  # no teardown, or with a row swapped.
  trailer) rows "first: false\n$section$m2"'last: true\n'
    printf 'FAIL: first: false\nFAILED: 1 check did not hold: "first".\n'; exit 1 ;;
  trailerblank) rows "first: false\n\n$section$m1$m2"'last: true\n'
    printf 'FAIL: first: false\nFAILED: 1 check did not hold: "first".\n'; exit 1 ;;
  trailercut) rows "first: false\n$section$m2"
    printf 'FAIL: first: false\nFAILED: 1 check did not hold: "first".\n'; exit 1 ;;
  trailerkilled) rows "first: false\n$section$m2"'last: true\n'
    printf 'FAIL: first: false\nCommand got signal SEGV.\n'; exit 1 ;;
  trailerswap) rows "first: false\nswapped in$section$m2"'last: true\n'
    printf 'FAIL: first: false\nFAILED: 1 check did not hold: "first".\n'; exit 1 ;;
  # A failed claim elaborated by a multi-line Verdict.fail, on a run short by its success
  # trailer: every row of the message carries `FAIL: `, as Verdict.fail prints it; then the
  # shape it printed before, a continuation row with no prefix -- a row the golden lacks.
  trailermulti) rows "first: false\nFAIL: first: the emit child failed:\nFAIL:   detail one\n$section$m2"'last: true\n'
    printf 'FAIL: first: false\nFAILED: 2 checks did not hold: "first", "first: the emit child failed:".\n'; exit 1 ;;
  trailermultibare) rows "first: false\nFAIL: first: the emit child failed:\n  detail one\n$section$m2"'last: true\n'
    printf 'FAIL: first: false\nFAILED: 2 checks did not hold: "first", "first: the emit child failed:".\n'; exit 1 ;;
  # A real Verdict run's streams, staged by the harness from verdict_teardown's golden.
  replay) rm -f "$out/$name.exe.output"; cp "$PROBE_HOME/replay.stdout" "$out/$name.exe.output"
    cat "$PROBE_HOME/replay.stderr"; exit 1 ;;
  compile) echo 'Error: injected compile failure'; exit 1 ;;
  # A batch's mutants (gh-ocannl-1429), told apart by what each wrote into the module.
  batch)
    case "$(cat module.ml)" in
      *CAUGHT*) rows 'first: false\nnested: label: true\nlast: true\n'; printf 'FAIL: first: false\n'; exit 1 ;;
      *SURVIVED*) rows 'first: true\nnested: label: true\nlast: true\n'; exit 0 ;;
      *SLEEPING*) touch "$PROBE_HOME/ready"; exec perl -e 'sleep 60' ;;
      *) echo 'Error: injected compile failure'; exit 1 ;;
    esac ;;
  refused) printf 'dune: unknown option\nUsage: dune build [OPTION]…\n'; exit 1 ;;
  restore_error) rm module.ml; mkdir module.ml; exit 1 ;;
  sleep) touch "$PROBE_HOME/ready"; exec perl -e 'sleep 60' ;;
esac
DUNE
chmod +x "$fixture/bin/dune"
export TMPDIR="$fixture"
export PATH="$fixture/bin:$PATH" PROBE_HOME="$fixture" OCANNL_TOOL_TEST_RUNS="$fixture/runs"
# Hermetic against the box it runs on, as tools/test-test-run.sh is: on a fleet
# box the runner would take a real run-time slot through the deployed
# fleet-worker.sh (gh-ocannl-1004), and on a GPU box it would resolve the
# batch's backends -- a dune build this fixture's dune refuses -- for a width
# the fixture already names (gh-ocannl-1066). No slot, and no device to meet.
export OCANNL_TOOL_FLEET_WORKER=none OCANNL_TOOL_DXG_DEVICE="$fixture/no-such-dxg" \
  OCANNL_TOOL_KFD_TOPOLOGY="$fixture/no-such-kfd" OCANNL_TOOL_NVIDIA_DEVICE="$fixture/no-such-nvidia"
cd "$fixture/repo"
printf 'first: true\nnested: label: true\nlast: true\n' > probe.expected
printf 'first: true\n\nSynthetic controls: scanner refusal diagnostics exercised by this control golden:\n%s\n%s\nlast: true\n' \
  '  [scanner-refusal:00000000000000000000000000000001] first refusal' \
  '  [scanner-refusal:00000000000000000000000000000002] last refusal' > manifest.expected
printf 'first: true\n\nSynthetic controls: scanner refusal diagnostics exercised by this control golden:\n%s\n%s\nEnd of synthetic controls.\n' \
  '  [scanner-refusal:00000000000000000000000000000001] first refusal' \
  '  [scanner-refusal:00000000000000000000000000000002] last refusal' > tail.expected
printf 'FAIL: the first diagnostic\nFAIL: the second diagnostic\n' > capture.expected
printf 'first: true\n\nSynthetic controls: scanner refusal diagnostics exercised by this control golden:\n%s\nEnd of synthetic controls.\n\nSynthetic controls: scanner refusal diagnostics exercised by this control golden:\n%s\nEnd of synthetic controls.\n' \
  '  [scanner-refusal:00000000000000000000000000000001] first refusal' \
  '  [scanner-refusal:00000000000000000000000000000002] last refusal' > sections.expected
printf 'first: true\n\nOK: every claim held.\n\nSynthetic controls: scanner refusal diagnostics exercised by this control golden:\n%s\n%s\nlast: true\n' \
  '  [scanner-refusal:00000000000000000000000000000001] first refusal' \
  '  [scanner-refusal:00000000000000000000000000000002] last refusal' > trailer.expected
# Each leg starts from its own module and patch (reset_module), so it passes
# run alone; the goldens above and the helpers below are shared.
reset_module() { # MODULE-TEXT PATCH-TEXT, each a printf format
  printf "$1" > module.ml
  chmod 640 module.ml
  cp module.ml "$fixture/pristine"
  printf "$2" > patch
}
: > "$fixture/invocations"
run_case() {
  expected=$1
  shift
  actual=0
  tools/mutation-run.sh module.ml patch "${PROBE_ALIAS:-@runtest-probe}" > "$fixture/result" 2>&1 ||
    actual=$?
  if [ "$actual" != "$expected" ]; then cat "$fixture/result"; echo "wrong exit: $actual != $expected"; exit 1; fi
  cmp module.ml "$fixture/pristine"
  recovery=$(sed -n 's/^recovery: //p' "$fixture/result")
  if [ -n "$recovery" ] && [ -d "$(dirname "$recovery")" ]; then
    echo 'scratch directory survived a restored run'; exit 1
  fi
  perl -e 'exit((stat($ARGV[0]))[2] & 07777 ^ 0640 ? 1 : 0)' module.ml
}
# The listing between `false claims:` and the next section: the claims alone (gh-ocannl-1364).
claims() { awk '/^false claims:$/ { on = 1; next } /^(failure messages|rows)/ { on = 0 } on' "$fixture/result"; }
messages() { awk '/^failure messages/ { on = 1; next } /^rows/ { on = 0 } on' "$fixture/result"; }
reached() {
  if grep -Eq '^(STOPPED EARLY|NEVER RAN|NOT COUNTED): ' "$fixture/result"; then
    cat "$fixture/result"; echo "a run that reached its last row was flagged"; exit 1
  fi
}
# A run that did not reach its last row is no evidence: exit 4, never 1's "caught"
# or 0's "survived" (gh-ocannl-1083). Each line names why; `stale` follows runs
# that left a full three-row output behind, which it must not count.
flagged() {
  grep -q "^$1$" "$fixture/result" || { cat "$fixture/result"; echo "missing: $1"; exit 1; }
  grep -q '^(not evidence: the mutant was neither caught nor survived)$' "$fixture/result"
}
leg_1() {
reset_module 'prefix\r\nANCHOR\r\nsuffix without newline' 'ANCHOR@@@MUTATED'
export PROBE_MODE=fail
run_case 1
claims > "$fixture/claims"
printf 'FAIL: first: false\nFAIL: nested: label: false\nFAIL: first: false\n' > "$fixture/expected"
cmp "$fixture/claims" "$fixture/expected" || { cat "$fixture/result"; echo 'fail: the false claims are not the claims'; exit 1; }
# Verdict.fail messages ending like claims are listed as messages, apart from the claims.
messages > "$fixture/messages"
printf 'FAIL: a message ending like a claim: false\nFAIL: one ending like pass_fail: FAIL\n' > "$fixture/expected"
cmp "$fixture/messages" "$fixture/expected" || { cat "$fixture/result"; echo 'fail: the failure messages are not listed apart'; exit 1; }
grep -q '^run: ' "$fixture/result"
grep -q '^rows: 3 printed of 3 in probe.expected$' "$fixture/result"
grep -q "^rows not counted: 2 \`FAIL: \` rows of the run's own " "$fixture/result"
reached
grep -q '^restored: byte-identical (cmp)$' "$fixture/result"
printf 'PASS exact claims and red verdict, CRLF/no-final-newline restoration\n'
}
harness_leg_run '1: exact claims and red verdict, CRLF and no-final-newline restoration' leg_1
leg_2() {
reset_module 'prefix\r\nANCHOR\r\nsuffix without newline' 'ANCHOR@@@MUTATED'
for mode in raise short failpad stopped casecrash caseexit casecut casefailmsg stale reused both; do
  export PROBE_MODE=$mode
  run_case 4
  case $mode in
    raise)
      grep -q '^rows: 1 printed of 3 in probe.expected$' "$fixture/result"
      flagged 'STOPPED EARLY: Verdict reported an uncaught exception; no row after it ran' ;;
    short)
      grep -q '^FAIL: first: false$' "$fixture/result"
      flagged "STOPPED EARLY: the mutated run printed 2 of the golden's 3 rows" ;;
    failpad)
      grep -q '^rows: 1 printed of 3 in probe.expected$' "$fixture/result"
      grep -q "^rows not counted: 3 " "$fixture/result"
      flagged "STOPPED EARLY: the mutated run printed 1 of the golden's 3 rows" ;;
    stopped)
      grep -q '^rows: 3 printed of 3 in probe.expected$' "$fixture/result"
      flagged 'STOPPED EARLY: Verdict reported an uncaught exception; no row after it ran' ;;
    casecrash)
      flagged "STOPPED EARLY: the mutated run printed 1 of the golden's 3 rows" ;;
    caseexit)
      flagged 'STOPPED EARLY: Verdict reported an exit inside a case; no case after it ran' ;;
    casecut)
      flagged "STOPPED EARLY: the mutated run printed 2 of the golden's 3 rows" ;;
    casefailmsg)
      flagged "STOPPED EARLY: the mutated run printed 1 of the golden's 3 rows" ;;
    stale|reused)
      grep -q '^rows: not counted$' "$fixture/result"
      flagged 'NEVER RAN: this run wrote no stdout of probe (a build failure, or an executable the mutation left unchanged, so dune reused its result)' ;;
    both)
      rm "$fixture/repo/_build/default/probe.actual"
      flagged "NOT COUNTED: this run rewrote more than one candidate stdout of probe, so none is known to be the mutant's" ;;
  esac
done
export PROBE_MODE=more
run_case 1
grep -q '^rows: 4 printed of 3 in probe.expected$' "$fixture/result"
grep -q "^rows not counted: 1 " "$fixture/result"
reached
# A golden that captures diagnostics: the run's `FAIL: ` rows count toward its own, by text first
# and then by number, so a reworded diagnostic is still a row the run printed; one beyond the
# golden's is the run's own.
export PROBE_MODE=capture
PROBE_ALIAS=@runtest-capture run_case 1
grep -q '^rows: 2 printed of 2 in capture.expected$' "$fixture/result" ||
  { cat "$fixture/result"; echo 'capture: the reworded diagnostic was not counted'; exit 1; }
grep -q "^rows not counted: 1 " "$fixture/result"
reached
# A raising Verdict.case prints fewer rows than the golden and still reached its
# last case: caught, not stopped early (gh-ocannl-1084).
for mode in casemid caselast; do
  export PROBE_MODE=$mode
  run_case 1
  grep -q '^rows: 2 printed of 3 in probe.expected$' "$fixture/result"
  grep -q '^cases raised: 1 (the rows short of the golden are theirs; the run reached its last case)$' \
    "$fixture/result" || { cat "$fixture/result"; echo "$mode: no cases-raised line"; exit 1; }
  reached
done
# DUNE_BUILD_DIR moves the tree the output is read from; _build keeps a full stale output.
export PROBE_MODE=raise DUNE_BUILD_DIR="$fixture/build-elsewhere"
run_case 4
grep -q '^rows: 1 printed of 3 in probe.expected$' "$fixture/result"
unset DUNE_BUILD_DIR
printf 'PASS stopped early, never ran and unattributable runs exit 4 and say why; a raising Verdict.case that reached its last case is caught\n'
}
harness_leg_run '2: stopped early, never ran and unattributable runs exit 4' leg_2
leg_3() {
reset_module 'prefix\r\nANCHOR\r\nsuffix without newline' 'ANCHOR@@@MUTATED'
# Rows short only by refusal-manifest markers, the run having printed every other
# row and ended through Verdict's teardown, reached its last row (gh-ocannl-1216);
# the same shortfall with the last row missing, or with no teardown, did not.
for mode in omitted omittedlast; do
  export PROBE_MODE=$mode
  PROBE_ALIAS=@runtest-manifest run_case 1
  grep -q '^rows: 5 printed of 6 in manifest.expected$' "$fixture/result"
  grep -q "^refusal markers omitted: 1 (no marker prints for a failed claim or for a refusal that did not run; every other row did, so the run reached its last row)$" \
    "$fixture/result" || { cat "$fixture/result"; echo "$mode: no omitted-markers line"; exit 1; }
  reached
done
for mode in omittedcut omittedkilled omittedswap; do
  export PROBE_MODE=$mode
  PROBE_ALIAS=@runtest-manifest run_case 4
  case $mode in
    omittedcut) flagged "STOPPED EARLY: the mutated run printed 4 of the golden's 6 rows" ;;
    omittedkilled|omittedswap) flagged "STOPPED EARLY: the mutated run printed 5 of the golden's 6 rows" ;;
  esac
done
# A golden ending on its section's closing row (gh-ocannl-1251, gh-ocannl-1318): a run cut inside
# the markers stopped early, though a teardown says a claim failed; a run ending on the closing
# row reached its last row, with a teardown or -- a mutant silencing one refusal while every claim
# holds -- without one, and so did one whose other rows differ; a row of the run's own after the
# closing row, or a run ending on an earlier section's closing row, did not.
export PROBE_MODE=tailcut
PROBE_ALIAS=@runtest-tail run_case 4
flagged "STOPPED EARLY: the mutated run printed 4 of the golden's 6 rows"
for mode in tailsilenced tailfailed; do
  export PROBE_MODE=$mode
  PROBE_ALIAS=@runtest-tail run_case 1
  grep -q '^rows: 5 printed of 6 in tail.expected$' "$fixture/result"
  grep -q '^refusal markers omitted: 1 ' "$fixture/result" ||
    { cat "$fixture/result"; echo "$mode: no omitted-markers line"; exit 1; }
  reached
done
export PROBE_MODE=tailswap
PROBE_ALIAS=@runtest-tail run_case 1
grep -q "^rows short of the golden: 1 (the run ended on the golden's last section-closing row, so none was cut; the run reached its last row)$" \
  "$fixture/result" || { cat "$fixture/result"; echo 'tailswap: no closed-run line'; exit 1; }
reached
export PROBE_MODE=tailafter
PROBE_ALIAS=@runtest-tail run_case 4
flagged "STOPPED EARLY: the mutated run printed 5 of the golden's 6 rows"
export PROBE_MODE=sectionscut
PROBE_ALIAS=@runtest-sections run_case 4
flagged "STOPPED EARLY: the mutated run printed 5 of the golden's 9 rows"
printf 'PASS rows short only by a refusal marker are caught; cut, killed or row-swapped runs with the same shortfall exit 4; a golden ending on its closing row tells a silenced refusal from a cut\n'
}
harness_leg_run '3: rows short only by a failed claim'\''s refusal marker' leg_3
leg_4() {
reset_module 'prefix\r\nANCHOR\r\nsuffix without newline' 'ANCHOR@@@MUTATED'
# A failing run of a scan lacks its success trailer, the `OK: ` rows printed only when no claim
# failed, and the blank row before it when that is the trailer's own: still its last row reached,
# alone or beside an omitted marker; cut, killed or row-swapped, still no evidence.
export PROBE_MODE=trailer
PROBE_ALIAS=@runtest-trailer run_case 1
grep -q '^rows: 5 printed of 8 in trailer.expected$' "$fixture/result"
grep -q "^refusal markers omitted: 1 " "$fixture/result"
grep -q '^success trailer omitted: 2 rows (`OK: ` rows print only when no claim failed; every other row did, so the run reached its last row)$' \
  "$fixture/result" || { cat "$fixture/result"; echo "trailer: no omitted-trailer line"; exit 1; }
reached
export PROBE_MODE=trailerblank
PROBE_ALIAS=@runtest-trailer run_case 1
grep -q '^rows: 7 printed of 8 in trailer.expected$' "$fixture/result"
grep -q '^success trailer omitted: 1 row ' "$fixture/result"
if grep -q '^refusal markers omitted:' "$fixture/result"; then
  cat "$fixture/result"; echo "trailerblank: every marker printed, yet one reads omitted"; exit 1
fi
reached
for mode in trailercut trailerkilled trailerswap; do
  export PROBE_MODE=$mode
  PROBE_ALIAS=@runtest-trailer run_case 4
  case $mode in
    trailercut) flagged "STOPPED EARLY: the mutated run printed 4 of the golden's 8 rows" ;;
    trailerkilled|trailerswap) flagged "STOPPED EARLY: the mutated run printed 5 of the golden's 8 rows" ;;
  esac
done
printf 'PASS a failing run short by its success trailer is caught; cut, killed or row-swapped runs with the same shortfall exit 4\n'
# A failure message spanning lines: Verdict.fail prefixes every line, so the runner skips each row
# of it as it skips any `FAIL: ` row, and a run short by its trailer is still caught. Each
# unprefixed continuation row, the shape Verdict.fail printed before, is a row the golden lacks.
export PROBE_MODE=trailermulti
PROBE_ALIAS=@runtest-trailer run_case 1
grep -q '^rows: 5 printed of 8 in trailer.expected$' "$fixture/result"
grep -q "^rows not counted: 2 " "$fixture/result"
grep -q '^success trailer omitted: 2 rows ' "$fixture/result"
reached
export PROBE_MODE=trailermultibare
PROBE_ALIAS=@runtest-trailer run_case 4
flagged "STOPPED EARLY: the mutated run printed 6 of the golden's 8 rows"
printf 'PASS a multi-line failure, every row prefixed, is caught on a trailer-short run; an unprefixed continuation row exits 4\n'
}
harness_leg_run '4: a failing run short by its success trailer' leg_4
leg_5() {
reset_module 'prefix\r\nANCHOR\r\nsuffix without newline' 'ANCHOR@@@MUTATED'
# The text above is this fixture's own; what the runner reads is Verdict's. So the runner also reads
# real Verdict runs: the streams of verdict_teardown's children, which its golden pins byte for byte
# (each line prefixed with its child and stream). A refactor of verdict.ml that keeps its text
# changes nothing here; a rewording the runner no longer reads fails here once that golden is
# promoted. Each golden below is what the child's run would print had every claim held: written
# out here, or (`@<child>`) the stdout of a sibling child in which every claim did hold.
teardown_golden="$root/test/operations/verdict_teardown.expected"
# A child's stream, as the golden prints it: `  <child> <stream> | <line>`, or the bare bar for an
# empty line.
transcript() {
  sed -n -e "s/^  $1 $2 | //p" -e "s/^  $1 $2 |\$//p" "$teardown_golden" > "$3"
  [ -s "$3" ] || { echo "verdict_teardown.expected carries no $1 $2 transcript"; exit 1; }
}
replay() {
  for stream in stdout stderr; do transcript "$1" "$stream" "$fixture/replay.$stream"; done
  case $2 in
    @*) transcript "${2#@}" stdout replay.expected ;;
    *) printf "$2" > replay.expected ;;
  esac
  export PROBE_MODE=replay
  PROBE_ALIAS=@runtest-replay run_case "$3"
  if grep -q '^(none)$' "$fixture/result"; then
    cat "$fixture/result"; echo "$1: the runner listed none of Verdict's false claims"; exit 1
  fi
}
# A raising Verdict.case: its claim line, and the teardown line saying the process ended through exit.
replay case_raise 'first: setup: true\nfirst: result: true\nsecond: reached: true\n' 1
grep -q '^rows: 2 printed of 3 in replay.expected$' "$fixture/result"
grep -q '^cases raised: 1 ' "$fixture/result" ||
  { cat "$fixture/result"; echo 'case_raise: Verdict.case raising read as no case raise'; exit 1; }
reached
# An exit inside a case once a check failed: Verdict's own STOPPED EARLY line names that ending.
replay case_exit 'the claim: true\nexits: done: true\nafter the exit: true\n' 4
flagged 'STOPPED EARLY: Verdict reported an exit inside a case; no case after it ran'
# A scan-shaped run short by its success trailer: every line of a multi-line Verdict.fail carries
# `FAIL: `, and the teardown names several checks. Its `FAIL: ` rows are no rows of the golden, so
# they do not count among the rows printed (gh-ocannl-1364).
replay trailer_omitted 'first claim: true\nthird claim: true\nlast claim: true\n\nOK: every claim held.\nOK: four claims checked.\n' 1
grep -q '^rows: 3 printed of 6 in replay.expected$' "$fixture/result"
grep -q '^success trailer omitted: 3 rows ' "$fixture/result" ||
  { cat "$fixture/result"; echo 'trailer_omitted: no omitted-trailer line'; exit 1; }
reached
# Failed pass_fail claims, bare and detailed, beside Verdict.fail messages, one of them ending like a
# claim: a claim's stderr line is its stdout row behind `FAIL: `, so the runner lists every such
# line, and a message, a `FAIL: ` row on both streams, is none of them but one of the failure
# messages listed apart (gh-ocannl-1364).
replay pass_fail_forms 'held: PASS\nfailed: PASS\ndetailed: PASS\n' 1
claims > "$fixture/claims"
sed 's/^/FAIL: /' "$fixture/replay.stdout" > "$fixture/echoes"
grep -Fx -f "$fixture/echoes" "$fixture/replay.stderr" > "$fixture/expected" || true
[ "$(wc -l < "$fixture/expected")" -ge 2 ] ||
  { echo 'pass_fail_forms: fewer than two failed claims echo their row on stderr'; exit 1; }
cmp "$fixture/claims" "$fixture/expected" ||
  { cat "$fixture/result"; echo 'pass_fail_forms: the false claims are not the echoed rows'; exit 1; }
grep '^FAIL: ' "$fixture/replay.stdout" > "$fixture/expected"
[ "$(($(wc -l < "$fixture/expected")))" -ge 2 ] ||
  { echo 'pass_fail_forms: fewer than two Verdict.fail messages on stdout'; exit 1; }
messages > "$fixture/messages"
cmp "$fixture/messages" "$fixture/expected" ||
  { cat "$fixture/result"; echo 'pass_fail_forms: the failure messages are not the stdout FAIL rows'; exit 1; }
grep -q '^rows: 3 printed of 3 in replay.expected$' "$fixture/result"
reached
# A scan's ending through the printers the scans share (gh-ocannl-1364): a failed claim leaves out its
# refusal-control marker and the success trailer, which Refusal_control_manifest.print_section and
# Verdict.success_trailer print, and the golden is the same ending with every claim held. What the
# runner skips is thereby theirs: a marker or trailer printed differently fails here once
# verdict_teardown's golden is promoted, where a refactor that prints the same rows changes nothing.
replay manifest_failed @manifest_held 1
grep -q '^refusal markers omitted: 1 ' "$fixture/result" ||
  { cat "$fixture/result"; echo 'manifest_failed: the omitted marker was not read as one'; exit 1; }
grep -q '^success trailer omitted: 2 rows ' "$fixture/result" ||
  { cat "$fixture/result"; echo 'manifest_failed: the omitted success trailer was not read as one'; exit 1; }
reached
# A mutant silencing one refusal while every claim holds (gh-ocannl-1251, gh-ocannl-1318): no
# teardown line, its refusal's marker missing, the section's closing row still last -- caught.
# Its stderr carries no teardown, so only the closing row can say the run was not cut inside the
# markers; the runner must not have read one.
replay_unclaimed() {
  for stream in stdout stderr; do transcript "$1" "$stream" "$fixture/replay.$stream"; done
  transcript "${2#@}" stdout replay.expected
  export PROBE_MODE=replay
  PROBE_ALIAS=@runtest-replay run_case "$3"
}
replay_unclaimed manifest_silenced @manifest_held 1
if grep -q '^FAILED: ' "$fixture/replay.stderr"; then
  echo 'manifest_silenced: the silenced child printed a teardown line'; exit 1
fi
grep -q '^refusal markers omitted: 1 ' "$fixture/result" ||
  { cat "$fixture/result"; echo 'manifest_silenced: the silenced refusal was not read as caught'; exit 1; }
reached
printf 'PASS real Verdict runs, from verdict_teardown'"'"'s golden: a raising case is caught, an exit inside a case exits 4, a trailer-short run with a multi-line failure is caught, failed pass_fail claims are listed, a scan ending short by a failed claim'"'"'s marker and its trailer is caught, and so is one short only by a silenced refusal'"'"'s marker\n'
}
harness_leg_run '5: real Verdict runs, from verdict_teardown'\''s golden' leg_5
leg_6() {
reset_module 'prefix\r\nANCHOR\r\nsuffix without newline' 'ANCHOR@@@MUTATED'
for mode in pass compile refused; do
  export PROBE_MODE=$mode
  case $mode in pass) rc=0 ;; compile) rc=4 ;; refused) rc=2 ;; esac
  run_case "$rc"
  grep -q '^(none)$' "$fixture/result"
done
printf 'PASS surviving mutation, compile failure (never ran), invocation refusal\n'
}
harness_leg_run '6: surviving mutation, compile failure, invocation refusal' leg_6
leg_6b() {
reset_module 'prefix\r\nANCHOR\r\nsuffix without newline' 'ANCHOR@@@MUTATED'
# The baseline (gh-ocannl-1364): the unmutated test runs first, and a red one refuses before
# anything is mutated -- one invocation, the baseline's, and the module never seen mutated. A green
# baseline is named in the report; one that rewrote the test's stdout leaves an output the mutated
# run must still rewrite to be counted.
export PROBE_MODE=fail
run_case 1
grep -q '^baseline: [0-9TZ-]* (the unmutated test passed)$' "$fixture/result" ||
  { cat "$fixture/result"; echo 'no baseline line'; exit 1; }
count=$(wc -l < "$fixture/invocations")
rm -f "$fixture/observed"
PROBE_BASELINE=red run_case 2
grep -q '^mutation-run: BASELINE RED: the unmutated test exited 1 (run [0-9TZ-]*), so probe.expected is not what it prints' "$fixture/result" ||
  { cat "$fixture/result"; echo 'red baseline: no refusal'; exit 1; }
grep -q '^  FAIL: first: false$' "$fixture/result" || { cat "$fixture/result"; echo 'red baseline: its failure is not shown'; exit 1; }
[ $(($(wc -l < "$fixture/invocations"))) = $((count + 1)) ] && [ ! -e "$fixture/observed" ] ||
  { echo 'red baseline: the mutated module was run'; exit 1; }
if grep -q '^recovery: ' "$fixture/result"; then echo 'red baseline: a recovery copy was announced'; exit 1; fi
export PROBE_MODE=reused
PROBE_BASELINE=rewrote run_case 4
flagged 'NEVER RAN: this run wrote no stdout of probe (a build failure, or an executable the mutation left unchanged, so dune reused its result)'
printf 'PASS a red baseline refuses before mutating; the baseline'"'"'s own stdout never counts as the mutant'"'"'s\n'
}
harness_leg_run '6b: a red baseline refuses before mutating' leg_6b
leg_7() {
# A launch this leg must not make would pass under a refusing mode.
export PROBE_MODE=pass
reset_module 'prefix\r\nANCHOR\r\nsuffix without newline' 'ANCHOR@@@MUTATED'
count=$(wc -l < "$fixture/invocations")
for patch_text in 'MISSING@@@x' 'ANCHOR' '@@@x' 'ANCHOR@@@x@@@y' 'ANCHOR@@@ANCHOR'; do
  printf '%s' "$patch_text" > patch
  run_case 2
done
printf 'ANCHOR@@@MUTATED' > patch
# An alias naming no single test, or one with no golden, has no rows to count against.
for alias in @probe @runtest @runtest-missing @../runtest-probe @./runtest-probe; do
  PROBE_ALIAS=$alias run_case 2
  case $alias in
    @runtest-missing) reason='no golden missing.expected to count' ;;
    @../*|@./*) reason='the alias directory must not contain . or ..' ;;
    *) reason='the alias must name one test' ;;
  esac
  grep -q "^mutation-run: $reason" "$fixture/result" || { cat "$fixture/result"; exit 1; }
done
printf 'prefix@@@x' > patch
printf 'prefix prefix' > module.ml
cp module.ml "$fixture/pristine"
run_case 2
printf 'aaa' > module.ml
cp module.ml "$fixture/pristine"
printf 'aa@@@x' > patch
run_case 2
[ "$(wc -l < "$fixture/invocations")" = "$count" ]
printf 'PASS missing, ambiguous, overlapping, malformed and unchanged anchors, and aliases without a golden, refuse before launch\n'
}
harness_leg_run '7: malformed anchors and aliases without a golden refuse before launch' leg_7
leg_8() {
# A launch this leg must not make would pass under a refusing mode.
export PROBE_MODE=pass
reset_module 'ANCHOR' 'ANCHOR@@@MUTATED'
# Hard links share the mutated inode; refuse without touching either path.
ln module.ml peer.ml
run_case 2
cmp peer.ml "$fixture/pristine"
rm peer.ml
printf 'PASS multiply linked module refuses without mutating either path\n'
}
harness_leg_run '8: a multiply linked module refuses without mutating either path' leg_8
leg_9() {
# A launch this leg must not make would pass under a refusing mode.
export PROBE_MODE=pass
reset_module 'ANCHOR' 'ANCHOR@@@MUTATED'
# Fail the write-open in a copied runner, even when tests run as root.
perl - tools/mutation-run.sh > tools/mutation-open-failure.sh <<'PERL'
use strict;
use warnings;
local $/;
open my $f, '<', $ARGV[0] or die $!;
my $s = <$f>;
my $anchor = q{open my $f, '>:raw', $module};
my $at = index($s, $anchor);
die "missing write-open anchor" if $at < 0;
substr($s, $at, length($anchor), q{open my $f, '>:raw', "$module/not-a-directory"});
print $s;
PERL
inode_before=$(perl -e 'print((stat($ARGV[0]))[1])' module.ml)
actual=0
bash tools/mutation-open-failure.sh module.ml patch @runtest-probe > "$fixture/result" 2>&1 || actual=$?
[ "$actual" = 2 ]
[ "$(perl -e 'print((stat($ARGV[0]))[1])' module.ml)" = "$inode_before" ]
cmp module.ml "$fixture/pristine"
if grep -q '^restored:' "$fixture/result"; then echo 'failed open replaced source'; exit 1; fi
printf 'PASS failed write-open leaves the original inode and bytes untouched\n'
}
harness_leg_run '9: a failed write-open leaves the original inode and bytes untouched' leg_9
leg_10() {
reset_module 'ANCHOR' 'ANCHOR@@@MUTATED'
# Hold the fork child before resetting inherited handlers, deterministically.
perl - tools/mutation-run.sh > tools/mutation-fork-window.sh <<'PERL'
use strict;
use warnings;
local $/;
open my $f, '<', $ARGV[0] or die $!;
my $s = <$f>;
my $anchor = 'if (!$child) {';
my $delay = <<'DELAY';
if (!$child) {
    open my $ready, '>', "$ENV{PROBE_HOME}/ready" or exit 126;
    close $ready;
    sleep 60;
DELAY
my $at = index($s, $anchor);
die "missing child anchor" if $at < 0;
substr($s, $at, length($anchor), $delay);
print $s;
PERL
chmod +x tools/mutation-fork-window.sh
# Drive signals from Perl so an asynchronous shell does not inherit ignored INT.
# A signal during the baseline ends the run as one, with nothing mutated and so nothing restored.
for signal_case in normal:INT normal:TERM normal:HUP fork:INT fork:TERM fork:HUP \
  baseline:INT baseline:TERM baseline:HUP; do
  signal=${signal_case#*:}
  unset PROBE_BASELINE
  case $signal_case in
    normal:*) export PROBE_RUNNER=tools/mutation-run.sh ;;
    fork:*) export PROBE_RUNNER=tools/mutation-fork-window.sh ;;
    baseline:*) export PROBE_RUNNER=tools/mutation-run.sh PROBE_BASELINE=sleep ;;
  esac
  rm -f "$fixture/ready"
  export PROBE_MODE=sleep
  perl - "$signal" "$fixture" <<'PERL'
use strict;
use warnings;
my ($signal, $dir) = @ARGV;
my %rc = (INT => 130, TERM => 143, HUP => 129);
my $pid = fork();
die $! unless defined $pid;
if (!$pid) {
    open STDOUT, '>', "$dir/result" or die $!;
    open STDERR, '>&', \*STDOUT or die $!;
    exec $ENV{PROBE_RUNNER}, 'module.ml', 'patch', '@runtest-probe';
    die $!;
}
my $ready = 0;
for (1..200) { if (-e "$dir/ready") { $ready = 1; last } select undef, undef, undef, .05; }
kill($signal, $pid);
local $SIG{ALRM} = sub { kill 'KILL', $pid; die "signal case timed out\n" };
alarm 15;
waitpid($pid, 0);
my $got = $? >> 8;
alarm 0;
die "not ready or wrong signal status: $got\n" unless $ready && $got == $rc{$signal};
PERL
  cmp module.ml "$fixture/pristine"
  case $signal_case in
    baseline:*)
      if grep -Eq '^(recovery|restored): ' "$fixture/result"; then
        cat "$fixture/result"; echo "$signal_case: a run cancelled in its baseline announced a mutation"; exit 1
      fi ;;
    *) grep -q '^restored: byte-identical (cmp)$' "$fixture/result" ;;
  esac
done
unset PROBE_BASELINE
printf 'PASS INT TERM HUP cancel in the baseline, the fork window and during test-run, restoring what was mutated\n'
}
harness_leg_run '10: INT TERM HUP cancel in the baseline, the fork window and during test-run, then restore' leg_10
leg_10b() {
reset_module 'ANCHOR' 'ANCHOR@@@MUTATED'
# A signal landing after the baseline was judged, before the mutation's own handlers: injected into a
# copy of the runner right after the baseline, and again right after the run snapshots the
# baseline's record (the gap a review found once the mutant became a sub run per patch), it still
# cancels before anything is mutated.
for anchor in post-baseline post-snapshot; do
perl - tools/mutation-run.sh "$anchor" > tools/mutation-late-signal.sh <<'PERL'
use strict;
use warnings;
my $which = $ARGV[1];
local $/;
open my $f, '<', $ARGV[0] or die $!;
my $s = <$f>;
my $anchor = $which eq 'post-baseline' ? qq{\nunlink "\$scratch/baseline";\n}
    : qq{\nmy \$cancel = \$baseline_stop;\n};
my $at = index($s, $anchor);
die "missing $which anchor" if $at < 0;
substr($s, $at + length($anchor), 0, qq{kill 'TERM', \$\$;\n});
print $s;
PERL
export PROBE_MODE=pass
rm -f "$fixture/observed"
actual=0
bash tools/mutation-late-signal.sh module.ml patch @runtest-probe > "$fixture/result" 2>&1 || actual=$?
[ "$actual" = 143 ] && [ ! -e "$fixture/observed" ] ||
  { cat "$fixture/result"; echo "late baseline signal ($anchor): exit $actual, or the mutant ran"; exit 1; }
cmp module.ml "$fixture/pristine"
done
printf 'PASS a signal between the baseline and the mutation cancels before mutating\n'
}
harness_leg_run '10b: a signal between the baseline and the mutation cancels before mutating' leg_10b
leg_11() {
reset_module 'ANCHOR' 'ANCHOR@@@MUTATED'
export PROBE_MODE=sleep OCANNL_TOOL_TEST_CAP=1
run_case 142
unset OCANNL_TOOL_TEST_CAP
printf 'PASS test-run cap expiry preserves status and restores\n'
}
harness_leg_run '11: test-run cap expiry preserves status and restores' leg_11
leg_12() {
export PROBE_MODE=sleep
reset_module 'ANCHOR' 'ANCHOR@@@MUTATED'
export OCANNL_TOOL_TEST_CAP=1
perl -e '
    open my $pipe, "-|", "tools/mutation-run.sh", "module.ml", "patch", q{@runtest-probe} or die $!;
    my $first = <$pipe>;
    die "missing recovery announcement" unless $first =~ /^recovery: /;
    close $pipe;
'
unset OCANNL_TOOL_TEST_CAP
cmp module.ml "$fixture/pristine"
printf 'PASS closed output pipe cannot interrupt restoration\n'
}
harness_leg_run '12: a closed output pipe cannot interrupt restoration' leg_12
leg_13() {
export PROBE_MODE=pass
reset_module 'MUTATED\r\nANCHOR\r\nend' 'ANCHOR\r\n@@@'
run_case 0
printf 'MUTATED\r\nend' > "$fixture/expected"
cmp "$fixture/observed" "$fixture/expected"
printf 'PASS multiline anchor and empty replacement use literal bytes\n'
}
harness_leg_run '13: a multiline anchor and an empty replacement use literal bytes' leg_13
leg_14() {
export PROBE_MODE=pass
reset_module 'ANCHOR' 'ANCHOR@@@MUTATED'
# Fault-inject exec failure into a scratch copy: only the parent restores.
sed "s/exec('bash',/exec('\/no-such-mutation-run-shell',/" tools/mutation-run.sh > tools/mutation-no-shell.sh
actual=0
bash tools/mutation-no-shell.sh module.ml patch @runtest-probe > "$fixture/result" 2>&1 || actual=$?
[ "$actual" = 127 ]
cmp module.ml "$fixture/pristine"
grep -q '^restored: byte-identical (cmp)$' "$fixture/result"
printf 'PASS failed child exec cannot unwind into parent restoration\n'
}
harness_leg_run '14: a failed child exec cannot unwind into parent restoration' leg_14
leg_15() {
reset_module 'ANCHOR' 'ANCHOR@@@MUTATED'
# A failing independent comparison must never print a restoration confirmation.
cat > "$fixture/bin/cmp" <<'CMP'
#!/usr/bin/env bash
exit 1
CMP
chmod +x "$fixture/bin/cmp"
export PROBE_MODE=pass
actual=0
tools/mutation-run.sh module.ml patch @runtest-probe > "$fixture/result" 2>&1 || actual=$?
[ "$actual" = 3 ]
if grep -q '^restored:' "$fixture/result"; then echo 'unexpected restoration confirmation'; exit 1; fi
backup=$(sed -n 's/^recovery: //p' "$fixture/result")
[ -f "$backup" ]
rm "$fixture/bin/cmp"
cmp "$backup" "$fixture/pristine"
cmp module.ml "$fixture/pristine"
printf 'PASS failed comparison retains byte-identical recovery copy and exits 3\n'
}
harness_leg_run '15: a failed comparison retains a recovery copy and exits 3' leg_15
leg_16() {
reset_module 'ANCHOR' 'ANCHOR@@@MUTATED'
# Failure to publish the restoration also leaves the recovery bytes available.
export PROBE_MODE=restore_error
actual=0
tools/mutation-run.sh module.ml patch @runtest-probe > "$fixture/result" 2>&1 || actual=$?
[ "$actual" = 3 ]
if grep -q '^restored:' "$fixture/result"; then echo 'unexpected restoration confirmation'; exit 1; fi
backup=$(sed -n 's/^recovery: //p' "$fixture/result")
cmp "$backup" "$fixture/pristine"
rmdir module.ml
cp "$fixture/pristine" module.ml
printf 'PASS failed restoration rename retains recovery copy and exits 3\n'
}
harness_leg_run '16: a failed restoration rename retains a recovery copy and exits 3' leg_16
leg_17() {
reset_module 'ANCHOR' 'ANCHOR@@@MUTATED'
# A killed launcher/supervisor can leave a live Dune holder of the same flock.
# Check the actual ownership signal, retain the mutant, then recover explicitly.
for victim in launcher supervisor; do
  rm -f "$fixture/ready"
  export PROBE_MODE=sleep
  perl - "$victim" "$fixture" <<'PERL'
use strict;
use warnings;
my ($victim, $dir) = @ARGV;
my $pid = fork();
die $! unless defined $pid;
if (!$pid) {
    open STDOUT, '>', "$dir/result" or die $!;
    open STDERR, '>&', \*STDOUT or die $!;
    exec 'tools/mutation-run.sh', 'module.ml', 'patch', '@runtest-probe';
    die $!;
}
my $ready = 0;
for (1..200) { if (-e "$dir/ready") { $ready = 1; last } select undef, undef, undef, .05; }
die "survivor fixture never became ready\n" unless $ready;
my @owners = glob "$dir/runs/owner-*";
@owners == 1 or die "ambiguous fixture owner\n";
sub read_one { open my $f, '<', $_[0] or die $!; my $v = <$f>; chomp $v; $v }
my $run = read_one($owners[0]);
my $supervisor = read_one("$run/pid");
my $target = $supervisor;
if ($victim eq 'launcher') {
    $target = qx{ps -o ppid= -p $supervisor};
    $target =~ s/\s+//g;
}
$target =~ /^\d+$/ && $target > 1 or die "invalid victim\n";
kill('KILL', $target) == 1 or die "kill victim: $!";
local $SIG{ALRM} = sub { kill 'KILL', $pid; die "survivor fixture timed out\n" };
alarm 15;
waitpid($pid, 0);
my $got = $? >> 8;
alarm 0;
open my $id, '>', "$dir/survivor-run" or die $!;
print {$id} "$run\n";
close $id;
die "expected deferred restoration, got $got\n" unless $got == 3;
PERL
  actual=0
  tools/test-run.sh idle > "$fixture/idle" 2>&1 || actual=$?
  [ "$actual" = 3 ]
  grep -q MUTATED module.ml
  backup=$(sed -n 's/^recovery: //p' "$fixture/result")
  cmp "$backup" "$fixture/pristine"
  if grep -q '^restored:' "$fixture/result"; then echo 'restored under surviving Dune'; exit 1; fi
  # A competing mutation is refused without changing the live mutant.
  cp module.ml "$fixture/mutant"
  printf 'MUTATED@@@SECOND' > second.patch
  actual=0
  tools/mutation-run.sh module.ml second.patch @runtest-probe > "$fixture/busy" 2>&1 || actual=$?
  [ "$actual" = 2 ]
  cmp module.ml "$fixture/mutant"
  run=$(cat "$fixture/survivor-run")
  tools/test-run.sh stop "$run" > "$fixture/stop" 2>&1
  # The supervisor stop is asynchronous; wait for its completion when present.
  tools/test-run.sh wait "$run" --timeout 10 > "$fixture/wait" 2>&1 || :
  tools/test-run.sh idle
  cp "$backup" module.ml
  cmp module.ml "$fixture/pristine"
done
printf 'PASS independently killed launcher/supervisor retain source until survivor is stopped\n'
}
harness_leg_run '17: a killed launcher or supervisor retains the source until the survivor is stopped' leg_17
leg_18() {
reset_module 'ANCHOR' 'ANCHOR@@@MUTATED'
# A finished run leaves the worktree lock behind; this leg reads it.
export PROBE_MODE=pass
run_case 0
# Pin the remaining public idle status and ensure it never replaces a bad lock.
set -- "$fixture"/runs/lock-*
[ "$#" = 1 ]
lock=$1
mv "$lock" "$lock.saved"
mkdir "$lock"
actual=0
tools/test-run.sh idle > "$fixture/idle" 2>&1 || actual=$?
[ "$actual" = 2 ]
[ -d "$lock" ]
rmdir "$lock"
mv "$lock.saved" "$lock"
tools/test-run.sh idle
printf 'PASS idle returns 0/3/2 without creating or replacing lock state\n'
}
harness_leg_run '18: idle returns 0/3/2 without creating or replacing lock state' leg_18
leg_19() {
perl -e '
  use Fcntl ":flock";
  open my $lock, ">", ".test-run.lock" or die $!;
  flock($lock, LOCK_EX | LOCK_NB) or die $!;
  system "bash", "tools/test-run.sh", "idle";
  exit(($? >> 8) == 3 ? 0 : 1);
' > "$fixture/legacy" 2>&1
[ -f .test-run.lock ]
tools/test-run.sh idle
[ -f .test-run.lock ]
rm .test-run.lock
printf 'PASS idle observes the legacy lock without removing it\n'
}
harness_leg_run '19: idle observes the legacy lock without removing it' leg_19
leg_20() {
# A patch written before `dune fmt` applies after it, and the reverse (gh-ocannl-1429): OLD matches
# modulo whitespace -- each run of it, an OCaml string's `\`-newline continuation included, and the
# padding just inside a bracket -- where it occurs nowhere byte for byte, still exactly once; NEW
# replaces the bytes matched. The formatted text is ocamlformat 0.29's own rewrap of the one-line
# form at the repository's margin.
export PROBE_MODE=fail
formatted='let raw =\n  [\n    ( "synthetic_scan.ml:00000000000000000000000000000001",\n      "a control claim whose label ocamlformat broke across two lines when it grew past the \\\n       margin: true" );\n  ]\n'
one_line='let raw = [ ("synthetic_scan.ml:00000000000000000000000000000001", "a control claim whose label ocamlformat broke across two lines when it grew past the margin: true") ]\n'
reset_module "$formatted" '("synthetic_scan.ml:00000000000000000000000000000001", "a control claim whose label ocamlformat broke across two lines when it grew past the margin: true")@@@("MUTATED", "")'
run_case 1
grep -q '^anchor: matched modulo whitespace ' "$fixture/result" ||
  { cat "$fixture/result"; echo 'pre-format patch: no modulo-whitespace line'; exit 1; }
printf 'let raw =\n  [\n    ("MUTATED", "");\n  ]\n' > "$fixture/expected"
cmp "$fixture/observed" "$fixture/expected" || { cat "$fixture/observed"; echo 'pre-format patch: wrong mutant'; exit 1; }
reset_module "$one_line" '( "synthetic_scan.ml:00000000000000000000000000000001",\n      "a control claim whose label ocamlformat broke across two lines when it grew past the \\\n       margin: true" )@@@("MUTATED", "")'
run_case 1
grep -q '^anchor: matched modulo whitespace ' "$fixture/result"
printf 'let raw = [ ("MUTATED", "") ]\n' > "$fixture/expected"
cmp "$fixture/observed" "$fixture/expected" || { cat "$fixture/observed"; echo 'post-format patch: wrong mutant'; exit 1; }
# A byte-for-byte occurrence wins, and says nothing of whitespace, even beside one that differs.
reset_module 'a b KEEP\na  b\n' 'a b@@@MUTATED'
run_case 1
if grep -q '^anchor: ' "$fixture/result"; then cat "$fixture/result"; echo 'exact anchor read as modulo whitespace'; exit 1; fi
printf 'MUTATED KEEP\na  b\n' > "$fixture/expected"
cmp "$fixture/observed" "$fixture/expected"
# Two matches modulo whitespace, overlapping ones included, refuse as one byte-for-byte pair does;
# so does a difference beyond whitespace -- a separator changed -- and a tolerated match in which
# NEW changes nothing. None launches.
export PROBE_MODE=pass
count=$(wc -l < "$fixture/invocations")
for case_text in 'a  b\na\tb\n|a b@@@x|ambiguous anchor (modulo whitespace)' \
  'x  x  x\n|x x@@@y|ambiguous anchor (modulo whitespace)' \
  '[ a; b ]\n|[ a, b ]@@@x|missing anchor' \
  'f  (x)\n|f (x)@@@f  (x)|mutation does not change the module'; do
  module_text=${case_text%%|*}
  rest=${case_text#*|}
  reset_module "$module_text" "${rest%%|*}"
  run_case 2
  grep -q "^mutation-run: ${rest#*|}\$" "$fixture/result" || { cat "$fixture/result"; echo "missing refusal: ${rest#*|}"; exit 1; }
done
[ "$(wc -l < "$fixture/invocations")" = "$count" ]
printf 'PASS a patch written before or after an ocamlformat rewrap applies modulo whitespace, exactly once; byte-for-byte wins; ambiguous, non-whitespace and no-op matches refuse before launch\n'
}
harness_leg_run '20: anchors modulo whitespace survive an ocamlformat rewrap' leg_20
leg_21() {
# --batch (gh-ocannl-1429): one baseline, then each patch of the directory in name order, each
# restored before the next, and a table; every patch checked before anything runs.
export PROBE_MODE=batch
reset_module 'prefix\nANCHOR\nsuffix\n' 'unused'
rm -rf "$fixture/patches"
mkdir "$fixture/patches"
printf 'ANCHOR@@@MUTATED_CAUGHT' > "$fixture/patches/1-caught.patch"
printf 'ANCHOR@@@MUTATED_SURVIVED' > "$fixture/patches/2-survived.patch"
printf 'ANCHOR@@@MUTATED_BROKEN' > "$fixture/patches/3-broken.patch"
printf 'not a patch' > "$fixture/patches/notes.txt"
: > "$fixture/baselines"
count=$(wc -l < "$fixture/invocations")
batch_case() { # EXPECTED-EXIT
  actual=0
  tools/mutation-run.sh --batch module.ml "$fixture/patches" @runtest-probe > "$fixture/result" 2>&1 || actual=$?
  [ "$actual" = "$1" ] || { cat "$fixture/result"; echo "batch: wrong exit $actual != $1"; exit 1; }
  cmp module.ml "$fixture/pristine"
}
batch_case 1
[ "$(($(wc -l < "$fixture/baselines")))" = 1 ] || { cat "$fixture/result"; echo "batch: $(wc -l < "$fixture/baselines") baselines, not one"; exit 1; }
[ "$(($(wc -l < "$fixture/invocations")))" = $((count + 4)) ] || { echo 'batch: not one baseline plus three runs'; exit 1; }
[ "$(grep -c '^restored: byte-identical (cmp)$' "$fixture/result")" = 3 ]
[ "$(grep -c '^== [123]-[a-z]*\.patch ==$' "$fixture/result")" = 3 ]
for row in 'caught       1-caught.patch' 'SURVIVED     2-survived.patch' 'NO EVIDENCE  3-broken.patch'; do
  grep -q "^  $row  (run [0-9TZ-]*)\$" "$fixture/result" || { cat "$fixture/result"; echo "batch: no row $row"; exit 1; }
done
grep -q '^batch: 3 of 3 mutant(s) run against one baseline ([0-9TZ-]*)$' "$fixture/result"
if grep -q 'notes.txt' "$fixture/result"; then echo 'batch: a non-patch file was run'; exit 1; fi
# Every mutant caught: the batch exits 0; one no evidence and none surviving: 4.
rm "$fixture/patches/2-survived.patch"
batch_case 4
rm "$fixture/patches/3-broken.patch"
batch_case 0
# A bad patch refuses the whole batch before its baseline, naming it; so does an empty directory.
printf 'MISSING@@@MUTATED' > "$fixture/patches/0-missing.patch"
count=$(wc -l < "$fixture/invocations")
batch_case 2
grep -q '^mutation-run: 0-missing.patch: missing anchor$' "$fixture/result" || { cat "$fixture/result"; exit 1; }
rm "$fixture/patches"/*.patch
batch_case 2
grep -q '^mutation-run: no \*.patch file in ' "$fixture/result"
[ "$(wc -l < "$fixture/invocations")" = "$count" ]
# A signal during the second mutant restores it and stops the batch: the third never runs.
printf 'ANCHOR@@@MUTATED_CAUGHT' > "$fixture/patches/1-caught.patch"
printf 'ANCHOR@@@MUTATED_SLEEPING' > "$fixture/patches/2-sleeping.patch"
printf 'ANCHOR@@@MUTATED_SURVIVED' > "$fixture/patches/3-survived.patch"
rm -f "$fixture/ready"
perl - "$fixture" <<'PERL'
use strict;
use warnings;
my ($dir) = @ARGV;
my $pid = fork();
die $! unless defined $pid;
if (!$pid) {
    open STDOUT, '>', "$dir/result" or die $!;
    open STDERR, '>&', \*STDOUT or die $!;
    exec 'tools/mutation-run.sh', '--batch', 'module.ml', "$dir/patches", '@runtest-probe';
    die $!;
}
my $ready = 0;
# The second mutant's readiness follows a baseline and a whole first mutant: a minute's headroom,
# not the single-run legs' ten seconds, so a loaded box does not read as a failure.
for (1..1200) { if (-e "$dir/ready") { $ready = 1; last } select undef, undef, undef, .05; }
kill('TERM', $pid);
local $SIG{ALRM} = sub { kill 'KILL', $pid; die "batch signal case timed out\n" };
alarm 15;
waitpid($pid, 0);
my $got = $? >> 8;
alarm 0;
die "not ready or wrong signal status: $got\n" unless $ready && $got == 143;
PERL
cmp module.ml "$fixture/pristine"
grep -q '^  NOT RUN      3-survived.patch$' "$fixture/result" || { cat "$fixture/result"; echo 'batch: signal did not stop the batch'; exit 1; }
grep -q '^  caught       1-caught.patch ' "$fixture/result"
# A signal arriving while a mutant is being restored is recorded, not dropped: injected into a copy
# of the runner just as restoration begins, it stops the batch once the first mutant is restored.
perl - tools/mutation-run.sh > tools/mutation-restore-signal.sh <<'PERL'
use strict;
use warnings;
local $/;
open my $f, '<', $ARGV[0] or die $!;
my $s = <$f>;
my $anchor = "    if (\$changed) {\n";
my $at = index($s, $anchor);
die "missing restoration anchor" if $at < 0;
substr($s, $at, 0, qq{    kill 'TERM', \$\$;\n});
print $s;
PERL
rm "$fixture/patches/2-sleeping.patch"
actual=0
bash tools/mutation-restore-signal.sh --batch module.ml "$fixture/patches" @runtest-probe > "$fixture/result" 2>&1 || actual=$?
[ "$actual" = 143 ] || { cat "$fixture/result"; echo "batch restoration signal: exit $actual, not 143"; exit 1; }
cmp module.ml "$fixture/pristine"
grep -q '^  NOT RUN      3-survived.patch$' "$fixture/result" ||
  { cat "$fixture/result"; echo 'batch: a signal during restoration did not stop the batch'; exit 1; }
# A signal pending before the first mutant starts runs none: every row NOT RUN, nothing mutated.
perl - tools/mutation-run.sh > tools/mutation-early-signal.sh <<'PERL'
use strict;
use warnings;
local $/;
open my $f, '<', $ARGV[0] or die $!;
my $s = <$f>;
my $anchor = "\nmy \$cancel = \$baseline_stop;\n";
my $at = index($s, $anchor);
die "missing post-snapshot anchor" if $at < 0;
substr($s, $at + length($anchor), 0, qq{kill 'TERM', \$\$;\n});
print $s;
PERL
rm -f "$fixture/observed"
actual=0
bash tools/mutation-early-signal.sh --batch module.ml "$fixture/patches" @runtest-probe > "$fixture/result" 2>&1 || actual=$?
[ "$actual" = 143 ] && [ ! -e "$fixture/observed" ] ||
  { cat "$fixture/result"; echo "batch early signal: exit $actual, or a mutant ran"; exit 1; }
grep -q '^batch: 0 of 2 mutant(s) run ' "$fixture/result" &&
  [ "$(grep -c '^  NOT RUN ' "$fixture/result")" = 2 ] ||
  { cat "$fixture/result"; echo 'batch: a mutant that never started is not NOT RUN'; exit 1; }
cmp module.ml "$fixture/pristine"
printf 'PASS --batch runs one baseline, then each patch restored in turn, and tabulates them; a bad patch or empty directory refuses before launch; a signal stops it restored\n'
}
harness_leg_run '21: --batch runs one baseline and tabulates every patch' leg_21

finish
