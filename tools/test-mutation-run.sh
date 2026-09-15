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
[ "$*" = 'build -j 4 @probe' ]
printf '%s\n' invoked >> "$PROBE_HOME/invocations"
case "$(cat module.ml)" in *MUTATED*) ;; *) exit 99 ;; esac
cp module.ml "$PROBE_HOME/observed"
case "$PROBE_MODE" in
  fail)
    printf 'FAIL: first: false\nFAIL: nested: label: false\nFAIL: first: false\n'
    printf 'prefix FAIL: decoy: false\nFAIL: true claim: true\nFAIL: suffix: false extra\n'
    exit 1 ;;
  pass) exit 0 ;;
  compile) echo 'Error: injected compile failure'; exit 1 ;;
  refused) printf 'dune: unknown option\nUsage: dune build [OPTION]…\n'; exit 1 ;;
  restore_error) rm module.ml; mkdir module.ml; exit 1 ;;
  sleep) touch "$PROBE_HOME/ready"; exec perl -e 'sleep 60' ;;
esac
DUNE
chmod +x "$fixture/bin/dune"
export TMPDIR="$fixture"
export PATH="$fixture/bin:$PATH" PROBE_HOME="$fixture" OCANNL_TOOL_TEST_RUNS="$fixture/runs"
cd "$fixture/repo"
printf 'prefix\r\nANCHOR\r\nsuffix without newline' > module.ml
chmod 640 module.ml
cp module.ml "$fixture/pristine"
printf 'ANCHOR@@@MUTATED' > patch
run_case() {
  expected=$1
  shift
  set +e
  tools/mutation-run.sh module.ml patch @probe > "$fixture/result" 2>&1
  actual=$?
  set -e
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
sed -n '/^false claims:$/,/^restored:/{ /^false claims:$/d; /^restored:/d; p; }' "$fixture/result" > "$fixture/claims"
printf 'FAIL: first: false\nFAIL: nested: label: false\nFAIL: first: false\n' > "$fixture/expected"
cmp "$fixture/claims" "$fixture/expected"
grep -q '^run: ' "$fixture/result"
grep -q '^restored: byte-identical (cmp)$' "$fixture/result"
printf 'PASS exact claims and red verdict, CRLF/no-final-newline restoration\n'
for mode in pass compile refused; do
  export PROBE_MODE=$mode
  case $mode in pass) rc=0 ;; compile) rc=1 ;; refused) rc=2 ;; esac
  run_case "$rc"
  grep -q '^(none)$' "$fixture/result"
done
printf 'PASS surviving mutation, compile failure, invocation refusal preserve verdict\n'
count=$(wc -l < "$fixture/invocations")
for patch_text in 'MISSING@@@x' 'ANCHOR' '@@@x' 'ANCHOR@@@x@@@y' 'ANCHOR@@@ANCHOR'; do
  printf '%s' "$patch_text" > patch
  run_case 2
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
printf 'PASS missing, ambiguous, overlapping, malformed and unchanged anchors refuse before launch\n'
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
bash tools/mutation-open-failure.sh module.ml patch @probe > "$fixture/result" 2>&1
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
    exec $ENV{PROBE_RUNNER}, 'module.ml', 'patch', '@probe';
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
    open my $pipe, "-|", "tools/mutation-run.sh", "module.ml", "patch", q{@probe} or die $!;
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
bash tools/mutation-no-shell.sh module.ml patch @probe > "$fixture/result" 2>&1
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
tools/mutation-run.sh module.ml patch @probe > "$fixture/result" 2>&1
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
tools/mutation-run.sh module.ml patch @probe > "$fixture/result" 2>&1
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
    exec 'tools/mutation-run.sh', 'module.ml', 'patch', '@probe';
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
  tools/mutation-run.sh module.ml second.patch @probe > "$fixture/busy" 2>&1
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
