#!/usr/bin/env bash
# One literal mutation, one focused test-run, byte-identical restoration.
# Usage: tools/mutation-run.sh <module> <patch-file> <@alias>
#        tools/mutation-run.sh --batch <module> <patch-dir> <@alias>
# Patch bytes are OLD@@@NEW (exactly one delimiter; no newline is stripped).
# OLD must occur exactly once, including overlapping occurrences. NEW may be empty.
# Where OLD occurs nowhere byte for byte, it may match MODULO WHITESPACE (gh-ocannl-1429), so a
# patch written before `dune fmt` still applies after it, and the reverse: each run of whitespace
# in OLD -- an OCaml string's `\`-newline continuation included -- matches any such run, and
# whitespace just inside a bracket is optional on either side, which is what an ocamlformat
# rewrap changes. That match too must be the only one, overlapping ones included; the report says
# `anchor: matched modulo whitespace`, and NEW replaces the matched bytes verbatim. Only
# whitespace is tolerated: a rewrap that also moves a token (a `|>` to the next line's head, a
# parenthesis added) still needs the patch rewritten.
# --batch runs every `*.patch` in <patch-dir>, sorted by name, against the one module and alias:
# every patch is checked before anything runs (a bad one refuses the batch, exit 2, naming it),
# the BASELINE runs once, then each mutant runs and is restored in turn, its report under a
# `== <patch> ==` header, and a table of outcomes ends the batch: caught (the run's exit 1),
# SURVIVED (0), NO EVIDENCE (4), or the run's other status. The batch exits 0 when every mutant
# was caught, 1 when one survived, 4 when one was no evidence and none survived, else the first
# other status; a signal stops it after restoring the current mutant (128+N), and a deferred or
# failed restoration stops it at once (3).
# Use an otherwise idle, isolated worktree; do not edit the module during the run.
# First the unmutated test runs once, the BASELINE, and must pass: a golden not promoted
# after a change to the test, or a test red on its own, would read every mutant as caught or
# stopped early, so a red baseline refuses (exit 2) before anything is mutated (gh-ocannl-1364).
# It costs one more test run per mutation; the report names it (`baseline: <run>`).
# The alias names ONE test -- @<dir>/runtest-<name>, slow-<name> or train-<name>
# -- whose golden is <dir>/<name>.expected: a mutated run counts as evidence only
# once it has REACHED ITS LAST ROW (gh-ocannl-1083). The stdout this run wrote
# (<name>.exe.output, .actual or .output under the build tree) must print as
# many rows as the golden -- its own `FAIL: ` rows, Verdict.fail's, beyond those the
# golden has, are no rows of it (gh-ocannl-1364) -- and Verdict must not have printed
# STOPPED EARLY (for an escaping exception, or an exit inside a Verdict.case). Rows short of the
# golden still count as reached when a Verdict.case raised, Verdict's teardown
# ended the process, and its stdout ended on the golden's last row or on a case's
# raise (gh-ocannl-1084); so do rows short only by refusal-manifest marker rows, the
# marker of a claim that failed, when every other row is the golden's, row for row (a
# claim of it may read false), and Verdict's teardown ended the process (gh-ocannl-1216).
# The same holds for a golden's success trailer, its `OK: ` rows (and the blank row
# before them), which a scan prints only when no claim failed.
# A golden ending on a marker section's closing row (`End of synthetic controls.`, printed
# after every marker whatever came before it, gh-ocannl-1251/1318) needs neither: a run whose
# stdout ends on the golden's LAST closing row (as many of them as the golden has) printed
# everything it was going to, so its rows short of the golden are omissions, never a cut -- a
# mutant that silences one refusal while every claim passes is caught -- and a run cut inside
# the markers, missing that row, stopped early. Elsewhere the teardown is all that separates
# a complete failing run from one cut inside its markers or before its trailer. A run
# that stopped early, or never ran (a build failure; an executable the mutation
# left unchanged, so dune reused its result), prints STOPPED EARLY or NEVER RAN
# (NOT COUNTED when two candidates changed) and exits 4 in place of test-run's
# 1 ("caught") or 0 ("survived").
# The false claims are the log's `FAIL: ` claim lines; a Verdict.fail message, which
# prints the same line on stdout too, is listed apart under `failure messages`.
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
my $batch = @ARGV && $ARGV[0] eq '--batch' ? shift @ARGV : '';
@ARGV == 3 or refuse('usage: tools/mutation-run.sh <module> <patch-file> <@alias>, '
    . 'or tools/mutation-run.sh --batch <module> <patch-dir> <@alias>');
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
# Whitespace as an ocamlformat rewrap moves it: blanks and line breaks, and an OCaml string
# literal's `\`-newline continuation, which a rewrap inserts at a space inside a long string.
my $ws = qr/(?:[ \t\r\n]|\\\r?\n)/;
# OLD as a pattern matching it modulo whitespace: each run of whitespace in it matches any
# nonempty run, and just inside a bracket a run is optional (ocamlformat pads a broken tuple's
# parenthesis, `( "a",`, and not a one-line one's, `("a", [])`).
sub tolerant {
    my ($old) = @_;
    # OLD's characters, and an empty string for each run of whitespace in it.
    my @pieces = split /((?:[ \t\r\n]|\\\r?\n)+)/, $old, -1;
    my @items = map { $_ % 2 ? ('') : split(//, $pieces[$_]) } 0 .. $#pieces;
    my $opens = qr/\A[(\[{]\z/;
    my $closes = qr/\A[)\]}]\z/;
    my $re = '';
    for my $i (0 .. $#items) {
        my $c = $items[$i];
        my $prev = $i > 0 ? $items[$i - 1] : undef;
        my $next = $i < $#items ? $items[$i + 1] : undef;
        if ($c eq '') {
            my $loose = (defined $prev && $prev =~ $opens) || (defined $next && $next =~ $closes);
            $re .= $loose ? "$ws*" : "$ws+";
        } else {
            # No whitespace between a bracket and its neighbour in OLD: the module may have some.
            $re .= "$ws*" if defined $prev && $prev ne '' && ($prev =~ $opens || $c =~ $closes);
            $re .= quotemeta($c);
        }
    }
    return qr/$re/;
}
# One patch file, read and anchored against the module: OLD byte for byte, else modulo
# whitespace; either way the only occurrence, overlapping ones included. Answers the mutated
# module and whether the match was modulo whitespace, or refuses naming [$label].
sub anchored {
    my ($patch, $label) = @_;
    my @parts = split /\@\@\@/, bytes($patch), -1;
    @parts == 2 && length($parts[0]) or refuse("${label}patch needs one @@@ delimiter and a nonempty OLD");
    my ($old, $new) = @parts;
    my ($at, $length, $loose) = (index($original, $old), length($old), 0);
    if ($at >= 0) {
        index($original, $old, $at + 1) < 0 or refuse("${label}ambiguous anchor");
    } else {
        my $re = tolerant($old);
        my @found;
        while ($original =~ /$re/g) {
            push @found, [$-[0], $+[0] - $-[0]];
            last if @found > 1;
            pos($original) = $-[0] + 1;
        }
        @found or refuse("${label}missing anchor");
        @found == 1 or refuse("${label}ambiguous anchor (modulo whitespace)");
        ($at, $length, $loose) = (@{$found[0]}, 1);
    }
    my $mutated = $original;
    substr($mutated, $at, $length, $new);
    $mutated ne $original or refuse("${label}mutation does not change the module");
    return [$mutated, $loose];
}
# The mutants: one patch, or every `*.patch` of a batch directory, each checked before anything
# runs.
my @mutants;
if ($batch) {
    -d $patch or refuse("--batch needs a directory of *.patch files: $patch");
    opendir my $d, $patch or refuse("read $patch: $!");
    my @names = sort grep { /\.patch\z/ && -f "$patch/$_" } readdir $d;
    closedir $d;
    @names or refuse("no *.patch file in $patch");
    @mutants = map { [$_, @{anchored("$patch/$_", "$_: ")}] } @names;
} else {
    @mutants = ([basename($patch), @{anchored($patch, '')}]);
}
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
# A candidate stdout file's identity, taken of each before the mutated run (below).
sub signature {
    my @s = Time::HiRes::lstat($_[0]);
    return @s ? join(':', @s[0, 1, 7, 9, 10]) : 'absent';
}
my %before;
# What a Verdict.case prints when its case raises (test/support/verdict.ml): the
# case's own failed claim, standing in for rows the raise kept from printing.
my $case_raise = qr/: the case ran to completion \(raised .*\): false$/;
# A refusal-manifest marker row, as Test_utils.Refusal_control_manifest.print_section writes one: a
# claim's marker prints only once Verdict recorded that claim passing, so a run in which the
# claim failed is a marker row short of the golden while it still reached its last row
# (gh-ocannl-1216). tools/test-mutation-run.sh replays a run that printer ended, so this shape is
# pinned on its output, not its source.
my $marker_row = qr/^  \[scanner-refusal:[0-9a-f]{32}\] /;
# The row that section ends on, printed unconditionally after its markers (gh-ocannl-1251,
# gh-ocannl-1318); pinned, like the marker shape, on the printer's output that
# tools/test-mutation-run.sh replays.
my $closing_row = qr/^End of synthetic controls\.$/;
# A `FAIL: ` row is Verdict.fail's (every line of a multi-line message carries the prefix): the
# run's own report of what went wrong, no row the golden has -- unless the golden has some, as a
# golden capturing a refusing child's diagnostics does (config_usage_scan_script_hint's is nothing
# else). So a run's `FAIL: ` rows count toward the golden's only as far as the golden has `FAIL: `
# rows: first those with a golden row's exact text, then, while the golden has unmatched ones
# left, any other (a mutant rewording a captured diagnostic still printed its row). The rest are
# the run's own and count toward no row of the golden: a run cut short after a long enough failure
# message must not read as having reached the golden's last row (gh-ocannl-1364).
my %golden_fail;
my $golden_fails = 0;
# One pass over a stdout: its rows (the run's own `FAIL: ` rows apart, counted on their own), its
# last row, its case raises, its marker rows, and its other rows in order. [$own] says
# whether the `FAIL: ` rows beyond the golden's are to be set apart (a run's) or the file is the
# golden itself, whose rows all count.
sub read_rows {
    my ($path, $own) = @_;
    open my $f, '<:raw', $path or refuse("read $path: $!");
    my @lines;
    while (my $line = <$f>) {
        $line =~ s/\r?\n$//;
        push @lines, $line;
    }
    my %r = (rows => 0, own_fail => 0, last => '', raises => 0, markers => 0, closings => 0, plain => [],
        fail_rows => {});
    # The golden `FAIL: ` rows the run matches by text; the remainder is the budget for others.
    my %exact = %golden_fail;
    my $budget = $golden_fails;
    if ($own) {
        for my $line (grep { /^FAIL: / } @lines) {
            if ($exact{$line}) { $exact{$line}--; $budget-- }
        }
    }
    %exact = %golden_fail;
    for my $line (@lines) {
        # The last row is the physical one: a run ending on a failure message after an apparent
        # last case raise did not end where a complete run does.
        $r{last} = $line;
        if ($line =~ /^FAIL: /) {
            $r{fail_rows}{$line}++;
            if (!$own) { $golden_fail{$line}++; $golden_fails++ }
            elsif ($exact{$line}) { $exact{$line}-- }
            elsif ($budget > 0) { $budget-- }
            else { $r{own_fail}++; next }
        }
        $r{rows}++;
        $r{raises}++ if $line =~ $case_raise;
        $r{closings}++ if $line =~ $closing_row;
        if ($line =~ $marker_row) { $r{markers}++ } else { push @{$r{plain}}, $line }
    }
    close $f or refuse("close $path: $!");
    return \%r;
}
# One pass over a run's log, handing [$on_line] each line and whether it is one of the stdout's
# `FAIL: ` rows [$stdout_fails] (row => count), matched once each, in order.
sub log_pass {
    my ($log, $stdout_fails, $on_line) = @_;
    my %left = $stdout_fails ? %$stdout_fails : ();
    open my $f, '<:raw', $log or refuse("read $log: $!");
    while (my $line = <$f>) {
        my $message = 0;
        if ($line =~ /^(FAIL: .*?)\r?\n?$/ && $left{$1}) {
            $left{$1}--;
            $message = 1;
        }
        $on_line->($line, $message);
    }
    close $f or refuse("close $log: $!");
}
my $golden_read = read_rows("$root/$golden", 0);
my ($golden_rows, $golden_last, $golden_markers, $golden_closings, @golden_plain) =
    ($golden_read->{rows}, $golden_read->{last}, $golden_read->{markers}, $golden_read->{closings},
     @{$golden_read->{plain}});
# A scan's success trailer: its `OK: ` rows, which it prints only once no claim failed
# (Verdict.success_trailer, whose output tools/test-mutation-run.sh replays),
# so a complete failing run lacks them -- and the blank row a trailer's leading `\n` puts before
# its first row (gh-ocannl-1083's row count read every failing run of such a scan as stopped
# early). The forms a complete failing run of the golden can take: the golden's rows as they are,
# without the trailer, and without the trailer and the blank row before it; each with the number
# of rows it omits.
my $ok_row = qr/^OK: /;
sub without_trailer {
    my ($drop_blank, @rows) = @_;
    return map { $rows[$_] } grep {
        $rows[$_] !~ $ok_row
            && !($drop_blank && $rows[$_] eq '' && $_ < $#rows && $rows[$_ + 1] =~ $ok_row)
    } 0 .. $#rows;
}
my @golden_forms = ([0, @golden_plain]);
if (my $trailer_rows = grep { $_ =~ $ok_row } @golden_plain) {
    my @kept_blank = without_trailer(0, @golden_plain);
    my @no_blank = without_trailer(1, @golden_plain);
    push @golden_forms, [$trailer_rows, @kept_blank];
    push @golden_forms, [@golden_plain - @no_blank, @no_blank] if @no_blank < @kept_blank;
}
# Whether a run's other rows are a form of the golden's, row for row: each the same, or -- the
# golden row being a claim -- that claim failed, which is a claim whose marker the run may then
# lack. A count alone would take an omitted row plus an extra one for a complete run. Answers the
# number of trailer rows the matching form omits, or undef when none matches.
sub same_plain_rows {
    my @rows = @_;
    FORM: for my $form (@golden_forms) {
        my ($omitted, @form_rows) = @$form;
        next FORM unless @rows == @form_rows;
        for my $i (0 .. $#rows) {
            next if $rows[$i] eq $form_rows[$i];
            my ($label) = $form_rows[$i] =~ /^(.*): true$/ or next FORM;
            next FORM unless $rows[$i] =~ /^\Q$label\E(?: \(.*\))?: false$/;
        }
        return $omitted;
    }
    return undef;
}
system('bash', "$root/tools/test-run.sh", 'idle') == 0
    or refuse('worktree is busy or its test-run lock is unreadable; nothing mutated');
my $scratch = tempdir('ocannl-mutation-XXXXXXXX', TMPDIR => 1, CLEANUP => 0);
my $owner = $$;
# The digest line test-run prints of its run, in a transcript of it.
sub run_log {
    open my $f, '<:raw', $_[0] or return undef;
    while (my $line = <$f>) { return $1 if $line =~ /^log: +(.+\/log)\r?\n?$/ }
    return undef;
}
# The BASELINE: the unmutated test, run before anything is mutated. A mutant's rows are counted
# against the golden, so the golden must be what the unmutated source prints and its claims must
# hold. A golden not yet promoted after a change to the test read a mutant as stopped early
# (gh-ocannl-1361's worker, twice), and one the test fails regardless of the mutation would read
# every mutant as caught (gh-ocannl-1364). A red baseline refuses with nothing mutated. Nothing is
# printed until the module is restored, so that the first line stays the recovery copy's.
my ($baseline_status, $baseline_stop) = (0, 0);
{
    my $pid = 0;
    for my $pair ([INT => 2], [TERM => 15], [HUP => 1]) {
        my ($name, $number) = @$pair;
        $SIG{$name} = sub {
            exit(128 + $number) if $$ != $owner;
            $baseline_stop ||= 128 + $number;
            kill 'TERM', $pid if $pid;
        };
    }
    $pid = fork();
    defined $pid or do { rmdir $scratch; refuse("fork: $!") };
    if ($pid == 0) {
        $SIG{$_} = 'DEFAULT' for qw(INT TERM HUP);
        chdir $root or exit 126;
        open STDIN, '<', '/dev/null' or exit 126;
        open STDOUT, '>', "$scratch/baseline" or exit 126;
        open STDERR, '>&', \*STDOUT or exit 126;
        exec 'bash', 'tools/test-run.sh', 'run', 'build', '-j', '4', $alias;
        exit 127;
    }
    kill 'TERM', $pid if $baseline_stop;
    while (1) {
        my $got = waitpid($pid, 0);
        next if $got < 0 && $! == EINTR;
        $baseline_status = ($? & 127) ? 128 + ($? & 127) : $? >> 8;
        # Reaped: a late signal must not reach whatever process reuses the pid.
        $pid = 0;
        last;
    }
}
my $baseline_log = run_log("$scratch/baseline");
if ($baseline_stop || $baseline_status) {
    $| = 1;
    if (!$baseline_stop && open my $f, '<:raw', "$scratch/baseline") {
        print while <$f>;
        close $f;
    }
    unlink "$scratch/baseline";
    rmdir $scratch;
    exit $baseline_stop if $baseline_stop;
    refuse("BASELINE RED: the unmutated test exited $baseline_status"
        . (defined $baseline_log ? ' (run ' . basename(dirname($baseline_log)) . ')' : '')
        . ", so $golden is not what it prints, or its claims fail without a mutation (promote the "
        . 'golden, or fix the test): no mutant can be counted against it; nothing was mutated');
}
unlink "$scratch/baseline";
# One mutant: written over the module, its test run, the module restored byte for byte, then
# its report. Answers its status (test-run's, 4 for no evidence, 2 for a refusal, 128+N for a
# signal) and its run's name; a deferred or failed restoration exits 3 on the spot.
my ($child, $changed) = (0, 0);
# A signal that reached the baseline's handlers after the baseline was judged cancels the run too.
my $cancel = $baseline_stop;
sub mutate {
    my ($mutated, $loose) = @_;
    # This mutant's handlers first, then the baseline's record: a signal in between lands in one of
    # the two, never in neither (the baseline's handlers only record what they saw).
    ($child, $changed) = (0, 0);
    for my $pair ([INT => 2], [TERM => 15], [HUP => 1]) {
        my ($name, $number) = @$pair;
        $SIG{$name} = sub {
            # A child can receive the forwarded signal before installing defaults.
            exit(128 + $number) if $$ != $owner;
            $cancel ||= 128 + $number;
            kill 'TERM', $child if $child;
        };
    }
    $cancel ||= $baseline_stop;
    # A batch mutant a signal reached before it started is not run at all: it answers nothing, and
    # its row reads NOT RUN.
    return () if $batch && $cancel;
    # What each candidate stdout file is BEFORE the mutated run, the baseline's rewrite included: an
    # output the mutated run did not rewrite is an earlier run's, and counting it would read "never
    # ran" as caught or survived.
    %before = map { $_ => signature($_) } @outputs;
    my $backup = "$scratch/original";
    copy($module, $backup) or refuse("backup: $!");
    my $mode = (stat($module))[2] & 07777;
    chmod($mode, $backup) or refuse("backup permissions: $!");
    $| = 1;
    print "recovery: $backup\n";
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
    # A batch records a signal here without acting on it, so the batch stops after this mutant.
    for my $pair ([INT => 2], [TERM => 15], [HUP => 1]) {
        my ($name, $number) = @$pair;
        $SIG{$name} = $batch ? sub { $cancel ||= 128 + $number } : 'IGNORE';
    }
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
    # All potentially large reporting is after restoration. Stream the transcript and
    # the log, printing each claim immediately rather than accumulating the log or claim list.
    # Between a batch's mutants a signal stops the batch, nothing being mutated.
    for my $pair ([INT => 2], [TERM => 15], [HUP => 1]) {
        my ($name, $number) = @$pair;
        $SIG{$name} = $batch ? sub { $cancel ||= 128 + $number } : 'DEFAULT';
    }
    my $run_name;
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
        print 'baseline: ', basename(dirname($baseline_log)), " (the unmutated test passed)\n"
            if defined $baseline_log;
        print "anchor: matched modulo whitespace (OLD occurs nowhere byte for byte; NEW replaced the "
            . "bytes it matched)\n" if $loose;
        # Read the digest from THIS invocation, never 'last' or its truncated tail.
        if (defined $log) {
            # The one stdout this run wrote, when it wrote exactly one: the rows it counts against the
            # golden's, and the `FAIL: ` rows that tell a failure message from a claim below.
            my @fresh = grep { -f $_ && !-l $_ && signature($_) ne $before{$_} } @outputs;
            my $run = @fresh == 1 ? read_rows($fresh[0], 1) : undef;
            # A claim's stderr line is its stdout row behind `FAIL: `: `<label>: false`, or a failed
            # pass_fail's `<label>: FAIL` or `<label>: FAIL (<detail>)` (verdict_teardown pins both
            # streams' bytes). Verdict.fail prints its message on BOTH streams, every line behind
            # `FAIL: `, so a message ending like a claim reads as one in the log alone; the stdout
            # tells them apart (gh-ocannl-1364). A log line `FAIL: X` is a message's while the stdout
            # has a row `FAIL: X` it has not yet been matched to, and a claim's otherwise (`claim` and
            # `claimf` print no stdout row at all). Each pass over the log matches afresh, so no line
            # is listed both ways.
            my $claim_line = qr/^(FAIL: .*: (?:false|FAIL(?: \(.*\))?))\r?\n?$/;
            $run_name = basename(dirname($log));
            print "run: $run_name\nfalse claims:\n";
            my ($found, $stopped, $teardown) = (0, '', 0);
            log_pass($log, $run && $run->{fail_rows}, sub {
                my ($line, $message) = @_;
                if (!$message && $line =~ $claim_line) {
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
                $teardown = 1 if $line =~ /^FAILED: \d+ checks? did not hold(?:: .*)?\.\r?$/;
            });
            print "(none)\n" unless $found;
            if ($run) {
                my $header = 0;
                log_pass($log, $run->{fail_rows}, sub {
                    my ($line, $message) = @_;
                    return unless $message;
                    print "failure messages (Verdict.fail's, no claims):\n" unless $header++;
                    $line =~ s/\r?\n?$/\n/;
                    print $line;
                });
            } elsif ($found) {
                print "(no stdout of this run to tell a Verdict.fail message ending like a claim from one: "
                    . "every such line is listed)\n";
            }
            # Whether the run reached its last row. Only then is 1 "caught" and 0 "survived".
            my $verdict;
            if ($run) {
                my $printed = $run->{rows};
                print "rows: $printed printed of $golden_rows in $golden\n";
                print "rows not counted: $run->{own_fail} `FAIL: ` row" . ($run->{own_fail} == 1 ? '' : 's')
                    . " of the run's own (Verdict.fail's; the golden lacks them)\n" if $run->{own_fail};
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
                    my ($raised, $last) = ($run->{raises}, $run->{last});
                    # Omitted marker rows and an omitted success trailer: every other row printed, the
                    # golden's row for row, and the process ended through Verdict's teardown, so a
                    # failed claim did end it.
                    my ($markers, @plain) = ($run->{markers}, @{$run->{plain}});
                    # Or the golden ends on a marker section's closing row and this run's stdout ends
                    # on its LAST one -- as many closing rows as the golden, since every section ends
                    # on the same text: stdout is a prefix of what the program printed, so nothing
                    # before that row was cut, whether or not a check failed, and no teardown is
                    # needed to say so.
                    my $closed = $golden_last =~ $closing_row && $last eq $golden_last
                        && $run->{closings} == $golden_closings;
                    my $trailer = ($teardown || $closed) && $markers <= $golden_markers
                        ? same_plain_rows(@plain) : undef;
                    if ($raised && $teardown && ($last eq $golden_last || $last =~ $case_raise)) {
                        print "cases raised: $raised (the rows short of the golden are theirs; "
                            . "the run reached its last case)\n";
                    } elsif (defined $trailer && ($trailer || $markers < $golden_markers)) {
                        my $omitted = $golden_markers - $markers;
                        print "refusal markers omitted: $omitted (no marker prints for a failed claim or for a "
                            . "refusal that did not run; every other row did, so the run reached its last row)\n"
                            if $omitted;
                        print "success trailer omitted: $trailer row" . ($trailer == 1 ? '' : 's')
                            . " (`OK: ` rows print only when no claim failed; every other row did, so the "
                            . "run reached its last row)\n" if $trailer;
                    } elsif ($closed) {
                        print "rows short of the golden: " . ($golden_rows - $printed) . " (the run ended on "
                            . "the golden's last section-closing row, so none was cut; the run reached its "
                            . "last row)\n";
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
    print "restored: byte-identical (cmp)\n" if $changed;
    print STDERR $error if $error;
    return ($cancel || ($error ? 2 : $code), $run_name);
}
my @results;
for my $mutant (@mutants) {
    # A pending cancellation -- the baseline's handlers record one until a mutant installs its own
    # -- stops a batch before its next mutant (that row and the rest NOT RUN); a single run still
    # enters, refusing before it mutates, as it always did.
    $cancel ||= $baseline_stop;
    last if $cancel && ($batch || @results);
    my ($name, $mutated, $loose) = @$mutant;
    print "== $name ==\n" if $batch;
    my @outcome = mutate($mutated, $loose);
    last unless @outcome;
    push @results, [$name, @outcome];
    last if $cancel;
}
rmdir $scratch or warn "mutation-run: cannot remove scratch directory $scratch: $!\n";
exit $results[0][1] unless $batch;
# The batch's table: one row per patch, in the order run, and those a signal kept from running.
my %outcome = (1 => 'caught', 0 => 'SURVIVED', 4 => 'NO EVIDENCE');
print "\nbatch: ", scalar(@results), " of ", scalar(@mutants), " mutant(s) run against one baseline ("
    . (defined $baseline_log ? basename(dirname($baseline_log)) : 'no digest') . ")\n";
for my $i (0 .. $#mutants) {
    my $name = $mutants[$i][0];
    if ($i > $#results) { printf "  %-12s %s\n", 'NOT RUN', $name; next }
    my ($status, $run) = @{$results[$i]}[1, 2];
    printf "  %-12s %s%s\n", $outcome{$status} // "EXIT $status", $name, defined $run ? "  (run $run)" : '';
}
exit $cancel if $cancel;
my @statuses = map { $_->[1] } @results;
exit 1 if grep { $_ == 0 } @statuses;
exit 4 if grep { $_ == 4 } @statuses;
my ($other) = grep { $_ != 1 } @statuses;
exit($other // 0);
PERL
