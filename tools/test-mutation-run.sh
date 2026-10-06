#!/usr/bin/env bash
# Exercise the shipping mutation runner and real test-run supervisor in an
# isolated repository fixture, with a deterministic fake dune on PATH.
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
case "$*" in
  'build -j 4 @runtest-probe') name=probe ;;
  'build -j 4 @runtest-manifest') name=manifest ;;
  'build -j 4 @runtest-tail') name=tail ;;
  *) exit 98 ;;
esac
printf '%s\n' invoked >> "$PROBE_HOME/invocations"
case "$(cat module.ml)" in *MUTATED*) ;; *) exit 99 ;; esac
cp module.ml "$PROBE_HOME/observed"
# The test's stdout, where dune leaves it even when the test exits nonzero; the
# golden, probe.expected, has three rows, and manifest.expected six, two of them
# refusal-manifest marker rows.
out=${DUNE_BUILD_DIR:-_build}/default
mkdir -p "$out"
rows() { rm -f "$out/$name.exe.output"; printf "$1" > "$out/$name.exe.output"; }
m1='  [scanner-refusal:00000000000000000000000000000001] first refusal\n'
m2='  [scanner-refusal:00000000000000000000000000000002] last refusal\n'
section='\nSynthetic controls: scanner refusal diagnostics exercised by this control golden:\n'
case "$PROBE_MODE" in
  fail)
    rows 'first: false\nnested: label: false\nlast: true\n'
    printf 'FAIL: first: false\nFAIL: nested: label: false\nFAIL: first: false\n'
    printf 'prefix FAIL: decoy: false\nFAIL: true claim: true\nFAIL: suffix: false extra\n'
    exit 1 ;;
  pass) rows 'first: true\nnested: label: true\nlast: true' ; exit 0 ;;
  # A raise after a failed claim, as Verdict reports it (gh-ocannl-1067): one
  # row printed of three, the other two never evaluated.
  raise)
    rows 'first: false\n'
    printf 'FAIL: first: false\nFAILED: 1 check did not hold.\n'
    printf 'STOPPED EARLY: an uncaught exception ended the run, so no check after it ran:\n'
    printf 'Fatal error: exception Failure("injected")\n'
    exit 1 ;;
  # Each signal alone: rows short with no Verdict line, Verdict's line with every row.
  short) rows 'first: false\nnested: label: true\n'; printf 'FAIL: first: false\n'; exit 1 ;;
  stopped) rows 'first: false\nnested: label: true\nlast: true\n'
    printf 'FAIL: first: false\nSTOPPED EARLY: an uncaught exception ended the run\n'; exit 1 ;;
  more) rows 'first: false\nFAIL: extra\nnested: label: true\nlast: true\n'
    printf 'FAIL: first: false\n'; exit 1 ;;
  # A raising Verdict.case (gh-ocannl-1084): one failed claim in place of the
  # rows the raise kept from printing, and the cases after it still run.
  casemid) rows 'nested: the case ran to completion (raised Failure("injected")): false\nlast: true\n'
    printf 'FAIL: nested: the case ran to completion (raised Failure("injected")): false\n'
    printf 'FAILED: 1 check did not hold.\n'; exit 1 ;;
  caselast) rows 'first: true\nlast: the case ran to completion (raised Failure("injected")): false\n'
    printf 'FAIL: last: the case ran to completion (raised Failure("injected")): false\n'
    printf 'FAILED: 1 check did not hold.\n'; exit 1 ;;
  # The same raise, then a crash: no teardown line, so nothing says a later case ran.
  casecrash) rows 'nested: the case ran to completion (raised Failure("injected")): false\n'
    printf 'FAIL: nested: the case ran to completion (raised Failure("injected")): false\n'
    printf 'Command got signal SEGV.\n'; exit 1 ;;
  # The same raise, then an exit inside a later case: Verdict says so, since the
  # stdout alone ends on a raise exactly as a run whose last case raised does.
  caseexit) rows 'nested: the case ran to completion (raised Failure("injected")): false\n'
    printf 'FAIL: nested: the case ran to completion (raised Failure("injected")): false\n'
    printf 'FAILED: 1 check did not hold.\n'
    printf 'STOPPED EARLY: an exit inside case "last" ended the run, so no case after it ran\n'
    exit 1 ;;
  # A teardown, but the stdout ends on neither the golden's last row nor a raise.
  casecut) rows 'first: the case ran to completion (raised Failure("injected")): false\nnested: label: true\n'
    printf 'FAIL: first: the case ran to completion (raised Failure("injected")): false\n'
    printf 'FAILED: 1 check did not hold.\n'; exit 1 ;;
  # The test never ran this time: the output a previous run left must not count.
  stale) printf 'FAIL: first: false\n'; exit 1 ;;
  reused) exit 0 ;;
  both) rows 'first: false\nnested: label: true\nlast: true\n'
    cp "$out/$name.exe.output" "$out/$name.actual"; printf 'FAIL: first: false\n'; exit 1 ;;
  # A failed claim's refusal marker is not printed (gh-ocannl-1216): a row short of
  # the golden, every other row printed -- the failed claim the first or the last.
  omitted) rows "first: false\n$section$m2"'last: true\n'
    printf 'FAIL: first: false\nFAILED: 1 check did not hold.\n'; exit 1 ;;
  omittedlast) rows "first: true\n$section$m1"'last: false\n'
    printf 'FAIL: last: false\nFAILED: 1 check did not hold.\n'; exit 1 ;;
  # The same shortfall from a run cut short: the golden's last row never printed,
  # or every row did but no teardown says the process ended through Verdict.
  omittedcut) rows "first: false\n$section$m2"
    printf 'FAIL: first: false\nFAILED: 1 check did not hold.\n'; exit 1 ;;
  omittedkilled) rows "first: false\n$section$m2"'last: true\n'
    printf 'FAIL: first: false\nCommand got signal SEGV.\n'; exit 1 ;;
  # A row swapped for another while a marker is missing: the counts and the last row agree,
  # the rows do not.
  omittedswap) rows "first: false\nswapped in$section$m2"'last: true\n'
    printf 'FAIL: first: false\nFAILED: 1 check did not hold.\n'; exit 1 ;;
  # tail.expected ends on its markers, as every real manifest golden does: a run cut inside
  # them with a teardown reads exactly as one whose claim failed. The runner's header names
  # this residual; only a trailer row after the section would separate the two.
  tailcut) rows "first: false\n$section$m1"
    printf 'FAIL: first: false\nFAILED: 1 check did not hold.\n'; exit 1 ;;
  compile) echo 'Error: injected compile failure'; exit 1 ;;
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
printf 'first: true\n\nSynthetic controls: scanner refusal diagnostics exercised by this control golden:\n%s\n%s\n' \
  '  [scanner-refusal:00000000000000000000000000000001] first refusal' \
  '  [scanner-refusal:00000000000000000000000000000002] last refusal' > tail.expected
printf 'prefix\r\nANCHOR\r\nsuffix without newline' > module.ml
chmod 640 module.ml
cp module.ml "$fixture/pristine"
printf 'ANCHOR@@@MUTATED' > patch
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
export PROBE_MODE=fail
run_case 1
sed -n '/^false claims:$/,/^rows:/{ /^false claims:$/d; /^rows:/d; p; }' "$fixture/result" > "$fixture/claims"
printf 'FAIL: first: false\nFAIL: nested: label: false\nFAIL: first: false\n' > "$fixture/expected"
cmp "$fixture/claims" "$fixture/expected"
grep -q '^run: ' "$fixture/result"
grep -q '^rows: 3 printed of 3 in probe.expected$' "$fixture/result"
reached() {
  if grep -Eq '^(STOPPED EARLY|NEVER RAN|NOT COUNTED): ' "$fixture/result"; then
    cat "$fixture/result"; echo "a run that reached its last row was flagged"; exit 1
  fi
}
reached
grep -q '^restored: byte-identical (cmp)$' "$fixture/result"
printf 'PASS exact claims and red verdict, CRLF/no-final-newline restoration\n'
# A run that did not reach its last row is no evidence: exit 4, never 1's "caught"
# or 0's "survived" (gh-ocannl-1083). Each line names why; `stale` follows runs
# that left a full three-row output behind, which it must not count.
flagged() {
  grep -q "^$1$" "$fixture/result" || { cat "$fixture/result"; echo "missing: $1"; exit 1; }
  grep -q '^(not evidence: the mutant was neither caught nor survived)$' "$fixture/result"
}
for mode in raise short stopped casecrash caseexit casecut stale reused both; do
  export PROBE_MODE=$mode
  run_case 4
  case $mode in
    raise)
      grep -q '^rows: 1 printed of 3 in probe.expected$' "$fixture/result"
      flagged 'STOPPED EARLY: Verdict reported an uncaught exception; no row after it ran' ;;
    short)
      grep -q '^FAIL: first: false$' "$fixture/result"
      flagged "STOPPED EARLY: the mutated run printed 2 of the golden's 3 rows" ;;
    stopped)
      grep -q '^rows: 3 printed of 3 in probe.expected$' "$fixture/result"
      flagged 'STOPPED EARLY: Verdict reported an uncaught exception; no row after it ran' ;;
    casecrash)
      flagged "STOPPED EARLY: the mutated run printed 1 of the golden's 3 rows" ;;
    caseexit)
      flagged 'STOPPED EARLY: Verdict reported an exit inside a case; no case after it ran' ;;
    casecut)
      flagged "STOPPED EARLY: the mutated run printed 2 of the golden's 3 rows" ;;
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
# The two lines the runner reads are Verdict's own text; a rewording there must fail here.
grep -qF '(label ^ ": the case ran to completion") ("raised " ^ text)' "$root/test/support/verdict.ml" ||
  { echo 'Verdict.case no longer prints "<label>: the case ran to completion (raised …)"'; exit 1; }
grep -qF 'Printf.sprintf "FAILED: %d check%s did not hold."' "$root/test/support/verdict.ml" ||
  { echo 'Verdict.teardown_line no longer prints "FAILED: <n> check(s) did not hold."'; exit 1; }
grep -qF '"STOPPED EARLY: an exit inside case %S ended the run' "$root/test/support/verdict.ml" ||
  { echo 'Verdict no longer prints "STOPPED EARLY: an exit inside case …"'; exit 1; }
# DUNE_BUILD_DIR moves the tree the output is read from; _build keeps a full stale output.
export PROBE_MODE=raise DUNE_BUILD_DIR="$fixture/build-elsewhere"
run_case 4
grep -q '^rows: 1 printed of 3 in probe.expected$' "$fixture/result"
unset DUNE_BUILD_DIR
printf 'PASS stopped early, never ran and unattributable runs exit 4 and say why; a raising Verdict.case that reached its last case is caught\n'
# Rows short only by refusal-manifest markers, the run having printed every other
# row and ended through Verdict's teardown, reached its last row (gh-ocannl-1216);
# the same shortfall with the last row missing, or with no teardown, did not.
for mode in omitted omittedlast; do
  export PROBE_MODE=$mode
  PROBE_ALIAS=@runtest-manifest run_case 1
  grep -q '^rows: 5 printed of 6 in manifest.expected$' "$fixture/result"
  grep -q "^refusal markers omitted: 1 (a failed claim's marker is not printed; every other row was, so the run reached its last row)$" \
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
# The residual, pinned so that closing it shows here: a cut inside a golden's closing markers.
export PROBE_MODE=tailcut
PROBE_ALIAS=@runtest-tail run_case 1
grep -q '^refusal markers omitted: 1 ' "$fixture/result"
# The marker row the runner skips is the manifest's own: a rewording there must fail here.
grep -qF 'Printf.sprintf "[scanner-refusal:%s] %s"' "$root/test/support/refusal_control_scan.ml" ||
  { echo 'Refusal_control_scan.marker no longer writes "[scanner-refusal:<digest>] <fragment>"'; exit 1; }
grep -qF 'printf "  %s\n" marker' "$root/test/support/refusal_control_manifest.ml" ||
  { echo 'Refusal_control_manifest.print no longer indents a marker row by two spaces'; exit 1; }
printf 'PASS rows short only by a failed claim'"'"'s refusal marker are caught; cut, killed or row-swapped runs with the same shortfall exit 4\n'
for mode in pass compile refused; do
  export PROBE_MODE=$mode
  case $mode in pass) rc=0 ;; compile) rc=4 ;; refused) rc=2 ;; esac
  run_case "$rc"
  grep -q '^(none)$' "$fixture/result"
done
printf 'PASS surviving mutation, compile failure (never ran), invocation refusal\n'
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
printf 'ANCHOR' > module.ml
cp module.ml "$fixture/pristine"
printf 'ANCHOR@@@MUTATED' > patch
# Hard links share the mutated inode; refuse without touching either path.
ln module.ml peer.ml
run_case 2
cmp peer.ml "$fixture/pristine"
rm peer.ml
printf 'PASS multiply linked module refuses without mutating either path\n'
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
set +e
bash tools/mutation-open-failure.sh module.ml patch @runtest-probe > "$fixture/result" 2>&1
actual=$?
set -e
[ "$actual" = 2 ]
[ "$(perl -e 'print((stat($ARGV[0]))[1])' module.ml)" = "$inode_before" ]
cmp module.ml "$fixture/pristine"
if grep -q '^restored:' "$fixture/result"; then echo 'failed open replaced source'; exit 1; fi
printf 'PASS failed write-open leaves the original inode and bytes untouched\n'
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
for signal_case in normal:INT normal:TERM normal:HUP fork:INT fork:TERM fork:HUP; do
  signal=${signal_case#*:}
  case $signal_case in
    normal:*) export PROBE_RUNNER=tools/mutation-run.sh ;;
    fork:*) export PROBE_RUNNER=tools/mutation-fork-window.sh ;;
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
  grep -q '^restored: byte-identical (cmp)$' "$fixture/result"
done
printf 'PASS INT TERM HUP cancel in the fork window and during test-run, then restore\n'
export PROBE_MODE=sleep OCANNL_TOOL_TEST_CAP=1
run_case 142
unset OCANNL_TOOL_TEST_CAP
printf 'PASS test-run cap expiry preserves status and restores\n'
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

export PROBE_MODE=pass
printf 'MUTATED\r\nANCHOR\r\nend' > module.ml
cp module.ml "$fixture/pristine"
printf 'ANCHOR\r\n@@@' > patch
run_case 0
printf 'MUTATED\r\nend' > "$fixture/expected"
cmp "$fixture/observed" "$fixture/expected"
printf 'PASS multiline anchor and empty replacement use literal bytes\n'
printf 'ANCHOR' > module.ml
cp module.ml "$fixture/pristine"
printf 'ANCHOR@@@MUTATED' > patch
# Fault-inject exec failure into a scratch copy: only the parent restores.
sed "s/exec('bash',/exec('\/no-such-mutation-run-shell',/" tools/mutation-run.sh > tools/mutation-no-shell.sh
set +e
bash tools/mutation-no-shell.sh module.ml patch @runtest-probe > "$fixture/result" 2>&1
actual=$?
set -e
[ "$actual" = 127 ]
cmp module.ml "$fixture/pristine"
grep -q '^restored: byte-identical (cmp)$' "$fixture/result"
printf 'PASS failed child exec cannot unwind into parent restoration\n'
# A failing independent comparison must never print a restoration confirmation.
cat > "$fixture/bin/cmp" <<'CMP'
#!/usr/bin/env bash
exit 1
CMP
chmod +x "$fixture/bin/cmp"
export PROBE_MODE=pass
set +e
tools/mutation-run.sh module.ml patch @runtest-probe > "$fixture/result" 2>&1
actual=$?
set -e
[ "$actual" = 3 ]
if grep -q '^restored:' "$fixture/result"; then echo 'unexpected restoration confirmation'; exit 1; fi
backup=$(sed -n 's/^recovery: //p' "$fixture/result")
[ -f "$backup" ]
rm "$fixture/bin/cmp"
cmp "$backup" "$fixture/pristine"
cmp module.ml "$fixture/pristine"
printf 'PASS failed comparison retains byte-identical recovery copy and exits 3\n'
# Failure to publish the restoration also leaves the recovery bytes available.
export PROBE_MODE=restore_error
set +e
tools/mutation-run.sh module.ml patch @runtest-probe > "$fixture/result" 2>&1
actual=$?
set -e
[ "$actual" = 3 ]
if grep -q '^restored:' "$fixture/result"; then echo 'unexpected restoration confirmation'; exit 1; fi
backup=$(sed -n 's/^recovery: //p' "$fixture/result")
cmp "$backup" "$fixture/pristine"
rmdir module.ml
cp "$fixture/pristine" module.ml
printf 'PASS failed restoration rename retains recovery copy and exits 3\n'
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
  set +e
  tools/test-run.sh idle > "$fixture/idle" 2>&1
  actual=$?
  set -e
  [ "$actual" = 3 ]
  grep -q MUTATED module.ml
  backup=$(sed -n 's/^recovery: //p' "$fixture/result")
  cmp "$backup" "$fixture/pristine"
  if grep -q '^restored:' "$fixture/result"; then echo 'restored under surviving Dune'; exit 1; fi
  # A competing mutation is refused without changing the live mutant.
  cp module.ml "$fixture/mutant"
  printf 'MUTATED@@@SECOND' > second.patch
  set +e
  tools/mutation-run.sh module.ml second.patch @runtest-probe > "$fixture/busy" 2>&1
  actual=$?
  set -e
  [ "$actual" = 2 ]
  cmp module.ml "$fixture/mutant"
  run=$(cat "$fixture/survivor-run")
  tools/test-run.sh stop "$run" > "$fixture/stop" 2>&1
  # The supervisor stop is asynchronous; wait for its completion when present.
  set +e
  tools/test-run.sh wait "$run" --timeout 10 > "$fixture/wait" 2>&1
  set -e
  tools/test-run.sh idle
  cp "$backup" module.ml
  cmp module.ml "$fixture/pristine"
done
printf 'PASS independently killed launcher/supervisor retain source until survivor is stopped\n'
# Pin the remaining public idle status and ensure it never replaces a bad lock.
set -- "$fixture"/runs/lock-*
[ "$#" = 1 ]
lock=$1
mv "$lock" "$lock.saved"
mkdir "$lock"
set +e
tools/test-run.sh idle > "$fixture/idle" 2>&1
actual=$?
set -e
[ "$actual" = 2 ]
[ -d "$lock" ]
rmdir "$lock"
mv "$lock.saved" "$lock"
tools/test-run.sh idle
printf 'PASS idle returns 0/3/2 without creating or replacing lock state\n'
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

finish
