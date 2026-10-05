#!/usr/bin/env bash
# One literal mutation, one focused test-run, byte-identical restoration.
# Usage: tools/mutation-run.sh <module> <patch-file> <@alias>
# Patch bytes are OLD@@@NEW (exactly one delimiter; no newline is stripped).
# OLD must occur exactly once, including overlapping occurrences. NEW may be empty.
# Use an otherwise idle, isolated worktree; do not edit the module during the run.
# The alias names ONE test -- @<dir>/runtest-<name>, slow-<name> or train-<name>
# -- whose golden is <dir>/<name>.expected: a mutated run counts as evidence only
# once it has REACHED ITS LAST ROW (gh-ocannl-1083). The stdout this run wrote
# (<name>.exe.output, .actual or .output under the build tree) must print as
# many rows as the golden, and Verdict must not have printed STOPPED EARLY (for
# an escaping exception, or an exit inside a Verdict.case). Rows short of the
# golden still count as reached when a Verdict.case raised, Verdict's teardown
# ended the process, and its stdout ended on the golden's last row or on a case's
# raise (gh-ocannl-1084); so do rows short only by refusal-manifest marker rows, the
# marker of a claim that failed, when every other row is the golden's, row for row (a
# claim of it may read false), and Verdict's teardown ended the process (gh-ocannl-1216).
# A golden ending on its marker section cannot tell a complete run from one cut
# inside the markers; the teardown is all that separates them. A run
# that stopped early, or never ran (a build failure; an executable the mutation
# left unchanged, so dune reused its result), prints STOPPED EARLY or NEVER RAN
# (NOT COUNTED when two candidates changed) and exits 4 in place of test-run's
# 1 ("caught") or 0 ("survived").
# Exit: test-run's status, 2 for refusal, 3 for deferred/failed restoration,
# 4 for a run that is no evidence (above); signals 128+N.
# A surviving test-run lock prevents automatic restoration; stop the surviving
# workers before manually recovering from the retained copy.
# INT/TERM/HUP cancel and reap test-run before restoration. SIGKILL cannot be
# trapped: its printed recovery copy may be used manually. This temporary copy
# is not durable storage and offers no power-loss/reboot recovery guarantee.
exec perl - "$0" "$@" <<'PERL'
use strict;
use warnings;
use Cwd qw(abs_path);
use File::Basename qw(dirname basename);
use File::Temp qw(tempdir tempfile);
use File::Copy qw(copy);
use Errno qw(EINTR);
use Time::HiRes ();

my $script = shift;
$SIG{__DIE__} = sub { if (!$^S) { print STDERR @_; exit 2; } };
sub refuse { die "mutation-run: $_[0]\n" }
@ARGV == 3 or refuse('usage: tools/mutation-run.sh <module> <patch-file> <@alias>');
my ($module, $patch, $alias) = @ARGV;
my $root = abs_path(dirname($script) . '/..');
$alias =~ /^\@[^\s]+$/ or refuse('expected one @alias');
-f $module && !-l $module or refuse('module must be a regular, non-symlink file');
(stat($module))[3] == 1 or refuse('module must have exactly one hard link');
$module = abs_path($module);
index($module, "$root/") == 0 or refuse('module must be inside this worktree');
sub bytes {
    open my $f, '<:raw', $_[0] or refuse("read $_[0]: $!");
    local $/;
    my $b = <$f>;
    close $f or refuse("close $_[0]: $!");
    return $b;
}
my $original = bytes($module);
my @parts = split /\@\@\@/, bytes($patch), -1;
@parts == 2 && length($parts[0]) or refuse('patch needs one @@@ delimiter and a nonempty OLD');
my ($old, $new) = @parts;
my $at = index($original, $old);
$at >= 0 or refuse('missing anchor');
index($original, $old, $at + 1) < 0 or refuse('ambiguous anchor');
$old ne $new or refuse('mutation does not change the module');
# The one test the alias runs, and its golden: the row count a complete run reaches.
my ($dir, $name) = $alias =~ m{^\@\@?((?:[A-Za-z0-9_.-]+/)*)(?:runtest|slow|train)-([A-Za-z0-9_]+)$}
    or refuse('the alias must name one test (@<dir>/runtest-<name>, slow-<name> or train-<name>): '
        . "a negative control counts the mutated run's rows against <name>.expected");
grep({ $_ eq '.' || $_ eq '..' } split m{/}, $dir) and refuse('the alias directory must not contain . or ..');
my $golden = "$dir$name.expected";
-f "$root/$golden" && !-l "$root/$golden"
    or refuse("no golden $golden to count the mutated run's rows against");
my $build = $ENV{DUNE_BUILD_DIR} // '_build';
$build = "$root/$build" unless $build =~ m{^/};
my @outputs = map { "$build/default/$dir$name$_" } qw(.exe.output .actual .output);
# What each candidate stdout file is BEFORE the run: an output this run did not
# rewrite is a previous run's, and counting it would read "never ran" as caught.
sub signature {
    my @s = Time::HiRes::lstat($_[0]);
    return @s ? join(':', @s[0, 1, 7, 9, 10]) : 'absent';
}
my %before = map { $_ => signature($_) } @outputs;
sub rows {
    open my $f, '<:raw', $_[0] or refuse("read $_[0]: $!");
    my ($n, $last, $chunk) = (0, "\n");
    while (read($f, $chunk, 65536)) {
        $n += ($chunk =~ tr/\n//);
        $last = substr($chunk, -1);
    }
    close $f or refuse("close $_[0]: $!");
    return $n + ($last ne "\n");
}
my $golden_rows = rows("$root/$golden");
# What a Verdict.case prints when its case raises (test/support/verdict.ml): the
# case's own failed claim, standing in for rows the raise kept from printing.
my $case_raise = qr/: the case ran to completion \(raised .*\): false$/;
# The number of case raises in a stdout, and its last row.
sub case_raises {
    open my $f, '<:raw', $_[0] or refuse("read $_[0]: $!");
    my ($n, $last) = (0, '');
    while (my $line = <$f>) {
        $line =~ s/\r?\n$//;
        $n++ if $line =~ $case_raise;
        $last = $line;
    }
    close $f or refuse("close $_[0]: $!");
    return ($n, $last);
}
my (undef, $golden_last) = case_raises("$root/$golden");
# A refusal-manifest marker row, as Test_utils.Refusal_control_manifest.print writes one: a
# claim's marker prints only once Verdict recorded that claim passing, so a run in which the
# claim failed is a marker row short of the golden while it still reached its last row
# (gh-ocannl-1216). The other rows, in order, and how many markers there were. A run's own
# `FAIL: ` rows are Verdict.fail's, never a golden's, so they are no row of either side.
my $marker_row = qr/^  \[scanner-refusal:[0-9a-f]{32}\] /;
sub plain_rows {
    open my $f, '<:raw', $_[0] or refuse("read $_[0]: $!");
    my ($markers, @rows) = (0);
    while (my $line = <$f>) {
        $line =~ s/\r?\n$//;
        if ($line =~ $marker_row) { $markers++ } elsif ($line !~ /^FAIL: /) { push @rows, $line }
    }
    close $f or refuse("close $_[0]: $!");
    return ($markers, @rows);
}
my ($golden_markers, @golden_plain) = plain_rows("$root/$golden");
# Whether a run's other rows are the golden's, row for row: each the same, or -- the golden row
# being a claim -- that claim failed, which is a claim whose marker the run may then lack. A count
# alone would take an omitted row plus an extra one for a complete run.
sub same_plain_rows {
    my @rows = @_;
    return 0 unless @rows == @golden_plain;
    for my $i (0 .. $#rows) {
        next if $rows[$i] eq $golden_plain[$i];
        my ($label) = $golden_plain[$i] =~ /^(.*): true$/ or return 0;
        return 0 unless $rows[$i] =~ /^\Q$label\E(?: \(.*\))?: false$/;
    }
    return 1;
}
my $mutated = $original;
substr($mutated, $at, length($old), $new);
system('bash', "$root/tools/test-run.sh", 'idle') == 0
    or refuse('worktree is busy or its test-run lock is unreadable; nothing mutated');
my $scratch = tempdir('ocannl-mutation-XXXXXXXX', TMPDIR => 1, CLEANUP => 0);
my $backup = "$scratch/original";
copy($module, $backup) or refuse("backup: $!");
my $mode = (stat($module))[2] & 07777;
chmod($mode, $backup) or refuse("backup permissions: $!");
$| = 1;
print "recovery: $backup\n";
my ($child, $cancel, $changed) = (0, 0, 0);
my $owner = $$;
for my $pair ([INT => 2], [TERM => 15], [HUP => 1]) {
    my ($name, $number) = @$pair;
    $SIG{$name} = sub {
        # A child can receive the forwarded signal before installing defaults.
        exit(128 + $number) if $$ != $owner;
        $cancel ||= 128 + $number;
        kill 'TERM', $child if $child;
    };
}
my $code = 2;
my $error;
eval {
    chdir $root or refuse("chdir: $!");
    refuse('cancelled before mutation') if $cancel;
    open my $f, '>:raw', $module or refuse("write module: $!");
    # A successful truncating open needs restoration, even if the write fails.
    $changed = 1;
    print {$f} $mutated or refuse("write module: $!");
    close $f or refuse("close module: $!");
    refuse('cancelled before launch') if $cancel;
    $child = fork();
    defined $child or refuse("fork: $!");
    if (!$child) {
        $SIG{$_} = 'DEFAULT' for qw(INT TERM HUP);
        # A fork child must never unwind into the parent's restoration block.
        open STDIN, '<', '/dev/null' or exit 126;
        open STDOUT, '>', "$scratch/transcript" or exit 126;
        open STDERR, '>&', \*STDOUT or exit 126;
        exec('bash', 'tools/test-run.sh', 'run', 'build', '-j', '4', $alias) or do {
            print STDERR "exec test-run: $!\n";
            exit 127;
        };
    }
    kill 'TERM', $child if $cancel;
    while (1) {
        my $got = waitpid($child, 0);
        next if $got < 0 && $! == EINTR;
        $got == $child or refuse("waitpid: $!");
        $code = ($? & 127) ? 128 + ($? & 127) : $? >> 8;
        $child = 0;
        last;
    }
    1;
} or $error = $@;
# No interrupt may cut the restoration in half. Publish a complete replacement
# in the same directory, then independently compare against the recovery copy.
$SIG{$_} = 'IGNORE' for qw(INT TERM HUP);
if ($changed) {
    # A dead launcher is not proof that its supervisor/Dune descendants ended.
    # Their inherited worktree flock is the existing harness ownership signal.
    if (system('bash', "$root/tools/test-run.sh", 'idle') != 0) {
        print STDERR "mutation-run: RESTORATION DEFERRED; worktree lock held or unreadable.\n",
            "Source may still be mutated; inspect/stop worktree runs before manual recovery from $backup\n";
        exit 3;
    }
    my $restored = eval {
    my ($restore_fh, $restore) = tempfile('.mutation-restore-XXXXXXXX', DIR => dirname($module), UNLINK => 0);
    close $restore_fh;
    copy($backup, $restore) && chmod($mode, $restore) && rename($restore, $module)
            && system('cmp', '-s', $backup, $module) == 0;
    };
    unless ($restored) {
        print STDERR "mutation-run: RESTORATION FAILED; recover from $backup ($!)\n";
        exit 3;
    }
}
unlink $backup;
# All potentially large reporting is after restoration. Stream both files and
# print each claim immediately rather than accumulating the log or claim list.
$SIG{$_} = 'DEFAULT' for qw(INT TERM HUP);
my $reported = eval {
    my $log;
    if (-f "$scratch/transcript") {
        open my $transcript, '<:raw', "$scratch/transcript" or refuse("read transcript: $!");
        while (my $line = <$transcript>) {
            print $line;
            $log = $1 if !defined($log) && $line =~ /^log: +(.+\/log)\r?\n?$/;
        }
        close $transcript or refuse("close transcript: $!");
    }
    # Read the digest from THIS invocation, never 'last' or its truncated tail.
    if (defined $log) {
        print 'run: ', basename(dirname($log)), "\nfalse claims:\n";
        open my $claims, '<:raw', $log or refuse("read $log: $!");
        my ($found, $stopped, $teardown) = (0, '', 0);
        while (my $line = <$claims>) {
            if ($line =~ /^(FAIL: .*: false)\r?\n?$/) {
                print "$1\n";
                $found = 1;
            }
            # Verdict's uncaught-exception handler (gh-ocannl-1067).
            # Which ending: an escaping exception, or an exit inside a Verdict.case
            # once a check had failed (gh-ocannl-1084), which a row count cannot see.
            $stopped ||= $line =~ /^STOPPED EARLY: an exit inside case / ? 'exit'
                : $line =~ /^STOPPED EARLY: / ? 'raise' : '';
            # Verdict's teardown (Verdict.teardown_line): the process ended through
            # exit, not a signal, so nothing after a caught raise was cut off.
            $teardown = 1 if $line =~ /^FAILED: \d+ checks? did not hold\.\r?$/;
        }
        close $claims or refuse("close $log: $!");
        print "(none)\n" unless $found;
        # Whether the run reached its last row. Only then is 1 "caught" and 0 "survived".
        my @fresh = grep { -f $_ && !-l $_ && signature($_) ne $before{$_} } @outputs;
        my $verdict;
        if (@fresh == 1) {
            my $printed = rows($fresh[0]);
            print "rows: $printed printed of $golden_rows in $golden\n";
            if ($stopped eq 'exit') {
                $verdict = 'STOPPED EARLY: Verdict reported an exit inside a case; no case after it ran';
            } elsif ($stopped) {
                $verdict = 'STOPPED EARLY: Verdict reported an uncaught exception; no row after it ran';
            } elsif ($printed < $golden_rows) {
                # A raising Verdict.case costs the rows it never printed, yet the cases
                # after it run (gh-ocannl-1084): the shortfall is that case's failure,
                # not a stopped run, once the process ended through Verdict's teardown
                # and its stdout where a complete one does -- on the golden's last row,
                # or on the last case's own raise. A raise line is the LAST case's only
                # because nothing after it ran and nothing stopped the run: a signal
                # leaves no teardown line, and an exit inside a later case its own
                # STOPPED EARLY, taken above.
                my ($raised, $last) = case_raises($fresh[0]);
                # Omitted marker rows: every other row printed, the golden's row for row, and the
                # process ended through Verdict's teardown, so a failed claim did end it.
                my ($markers, @plain) = plain_rows($fresh[0]);
                if ($raised && $teardown && ($last eq $golden_last || $last =~ $case_raise)) {
                    print "cases raised: $raised (the rows short of the golden are theirs; "
                        . "the run reached its last case)\n";
                } elsif ($teardown && $markers < $golden_markers && same_plain_rows(@plain)) {
                    my $omitted = $golden_markers - $markers;
                    print "refusal markers omitted: $omitted (a failed claim's marker is not printed; "
                        . "every other row was, so the run reached its last row)\n";
                } else {
                    $verdict = "STOPPED EARLY: the mutated run printed $printed of the golden's $golden_rows rows";
                }
            }
        } elsif (@fresh) {
            print "rows: not counted\n";
            $verdict = 'NOT COUNTED: this run rewrote more than one candidate stdout of '
                . "$name, so none is known to be the mutant's";
        } else {
            print "rows: not counted\n";
            $verdict = "NEVER RAN: this run wrote no stdout of $name (a build failure, or an "
                . 'executable the mutation left unchanged, so dune reused its result)';
        }
        if (defined $verdict) {
            print "$verdict\n",
                "(not evidence: the mutant was neither caught nor survived)\n";
            $code = 4 if $code == 0 || $code == 1;
        }
    } else {
        print "run: unavailable (test-run produced no digest)\n";
        $code = 2 unless $code;
    }
    1;
};
$error ||= $@ unless $reported;
unlink "$scratch/transcript" if -f "$scratch/transcript";
rmdir $scratch or warn "mutation-run: cannot remove scratch directory $scratch: $!\n";
print "restored: byte-identical (cmp)\n" if $changed;
print STDERR $error if $error;
exit($cancel || ($error ? 2 : $code));
PERL
