#!/usr/bin/env bash
# The per-test cap (gh-ocannl-1211), shared by tools/test-run.sh (`run`/`start`)
# and tools/sweep.sh (each GPU unit's test leg): its default, which targets it
# leaves alone, and the watcher that enforces it. One copy, for the reason
# tools/box-jobs.sh is one copy: the sweep ran dune without the guard, and on
# 2026-10-14 (sweep 20261014T051751Z) sixteen tuf hip tests hung at once and
# held the unit to its full 5400 s cap with no test named. Sourced, never
# executed.

# The per-test cap's default (gh-ocannl-1211; test_cap_watch enforces it), for
# a GPU batch whose targets do not reach the `slow` aliases -- in
# tools/test-run.sh (plan_test_cap) and in each GPU unit of tools/sweep.sh,
# whose `@slow` leg it leaves to the unit's cap. 1500s
# clears every runtest/train action in the fleet's dune traces (the longest,
# 1021s, under correctness-slot load), but `slow-cifar_conv` runs ~300s solo
# and loads of 5-15x were measured, plus its dataset download -- so `@slow`
# and CPU batches keep only the run's cap unless asked. Re-derive it from a
# box's traces with tools/action-durations.sh, or from the `slowest` record
# every test-run.sh run directory keeps (see its explicit_trace_file).
TEST_CAP_DEFAULT=1500

# Whether dune arguments reach the slow tests, which the default leaves to the
# run's own cap; prints the first argument that does. Arguments after dune's
# own `--` are the program's, not targets.
test_cap_reaches_slow() { # dune-argv...; 0 (printing the target) iff one reaches @slow
  local arg
  for arg; do
    case $arg in
      --) return 1 ;;
      @slow | @@slow | @*/slow | @@*/slow | @slow-* | @@slow-* | @*/slow-* | @@*/slow-*)
        printf '%s' "$arg"
        return 0 ;;
    esac
  done
  return 1
}

# The per-test cap (gh-ocannl-1211): the run's cap, applied to each process
# dune starts. A GPU fault on tuf (gfxhub/CPC page fault, then the test's
# HIP_ERROR_ILLEGAL_ADDRESS) left the device wedged, and every test dune
# started after it spun a core in the runtime's device init with no output --
# eight at once, for 32 minutes, until a person cancelled the run; the run's
# own cap had another half hour to go. A device that wedges mid-batch hangs
# every LATER test the same way, so a test that outlives this cap ends the
# whole run rather than just itself: killing one hung test would only let
# dune start the next.
#
# Arguments: the cap, the RESOLVER (the shell dune runs under, whose child the
# watcher is), the SUPERVISOR to signal, a directory for its records, the batch
# kind (gpu or cpu), and optionally the log prefix and the remedy its message
# names (test-run's own by default). In tools/test-run.sh the resolver is
# `_resolve`; in tools/sweep.sh it is the unit's `sh -c`, whose parent is the
# unit's capped supervisor.
#
# Run by the resolver beside dune, in the run's process group with the worktree
# lock closed, it lists the processes every few seconds (`ps`, the portable
# reader: /proc on Linux, the BSD ps on macOS) and takes as a test every
# child of a `dune` descending from the resolver (or of a process named in
# `test-parents` there: a direct repeat's replay) -- dune starts each action as
# its own process-group leader, so the run's group kill reaches dune but not
# them, while dune's own TERM handling kills every action it has running
# (measured on dune 3.24, a TERM-ignoring action included). Children of dune
# rather than "every group leader in the run", because the fleet slot holds
# its sleep inhibitor in a group of its own for the whole run. Past the cap it
# records the test in `test-cap`, says so in the log, and sends the
# supervisor the SIGALRM the run's own cap sends: the same reap, the same 142,
# with the digest naming the test from the record. It never signals once its
# parent is no longer the resolver (dune exited and the resolver moved on, or the
# run is being reaped), so it cannot outlive the supervisor and signal a
# recycled pid. Where `ps` cannot list processes (Git Bash) it records that in
# `test-cap-off`, for the digest, and exits, leaving the run's cap as the only
# bound -- never in the log, whose first line after the prelude must stay
# dune's own (dune_refusal).
test_cap_watch='
  my ($cap, $resolver, $sup, $rd, $kind, $who, $remedy) = @ARGV;
  $who = "test-run" unless defined $who && $who ne "";
  $remedy = "Rerun with --test-cap 0, or a larger cap" unless defined $remedy && $remedy ne "";
  $| = 1;
  $ENV{LC_ALL} = "C";
  my $poll = $cap < 60 ? 1 : 5;
  # ps elapsed time: [[dd-]hh:]mm:ss
  my $seconds = sub {
    my $t = shift;
    my $d = ($t =~ s/^(\d+)-//) ? $1 : 0;
    my $s = 0;
    $s = $s * 60 + $_ for split /:/, $t;
    $d * 86400 + $s;
  };
  # A test is timed on the watcher\x27s own clock (gh-ocannl-1319), never by the
  # age ps reports: procps derives etime from the boot time, which a suspend and
  # resume can skew -- minix once read 38109073018720s for a test a minute old
  # and ended the run. The clock is one that stops while the host sleeps
  # (CLOCK_UPTIME_RAW on macOS, CLOCK_MONOTONIC on Linux), so a suspend never
  # counts against a test either; wall time only where Time::HiRes has neither.
  my $now = sub { time };
  if (eval { require Time::HiRes; 1 }) {
    for my $name ("CLOCK_UPTIME_RAW", "CLOCK_MONOTONIC") {
      my $id = eval { &{"Time::HiRes::$name"}() };
      next unless defined $id;
      my $t0 = eval { Time::HiRes::clock_gettime($id) };
      next unless defined $t0 && $t0 > 0;
      $now = sub { Time::HiRes::clock_gettime($id) };
      last;
    }
  }
  # Each test is born where the watcher first sees it, less its ps etime but
  # never more than the time since the previous listing, which did not show it:
  # a skewed etime moves a birth by at most one poll. A pid is the same test
  # while its parent and its start token stay and it is in every listing -- not
  # its name, which an exec (a wrapper handing over to the test) changes, and
  # which must not restart the test\x27s age; an exec keeps the start token.
  #
  # The start token (gh-ocannl-1448) tells a test from another process the
  # system gave its pid within one poll, under the same dune, which would
  # otherwise inherit the first one\x27s age and be capped early: on Linux the
  # starttime of /proc/<pid>/stat (field 22, clock ticks since boot, as the
  # kernel keeps it -- not a wall time derived from the boot time, as procps
  # derives etime and lstart), elsewhere ps\x27s lstart, which the kernel keeps
  # there as a start time. A token that cannot be read (the process is gone, or
  # neither reader answers) decides nothing: the parent alone does, as before.
  # OCANNL_TOOL_TEST_CAP_PROC names a directory read ahead of /proc, a seam for
  # tools/test-test-run.sh: a pid it holds a stat for is read there.
  my @procs = ("/proc");
  unshift @procs, $ENV{OCANNL_TOOL_TEST_CAP_PROC}
    if defined $ENV{OCANNL_TOOL_TEST_CAP_PROC} && $ENV{OCANNL_TOOL_TEST_CAP_PROC} ne "";
  my $procfs = -d "/proc/self";
  my $tokens = sub { # pids -> { pid => token }
    my (%tok, @rest);
    for my $p (@_) {
      for my $root (@procs) {
        my $s;
        if (open my $sf, "<", "$root/$p/stat") { $s = <$sf>; close $sf }
        # The LAST ") " ends the name, which may hold both (Codex review round 7).
        next unless defined $s && $s =~ /^(?s:.*)\)\s+(.*)$/;
        my @f = split /\s+/, $1;
        $tok{$p} = "t$f[19]" if defined $f[19] && $f[19] =~ /^\d+$/;
        last;
      }
      push @rest, $p unless exists $tok{$p} || $procfs;
    }
    if (@rest) {
      local $ENV{TZ} = "UTC";
      my $list = join ",", @rest;
      for (qx{ps -o pid= -o lstart= -p $list 2>/dev/null}) {
        next unless /^\s*(\d+)\s+(\S.*?)\s*$/;
        (my $l = $2) =~ s/\s+/ /g;
        $tok{$1} = "l$l";
      }
    }
    \%tok;
  };
  my %born;
  my $last = $now->();
  while (getppid() == $resolver) {
    my $t = $now->();
    my @ps = qx{ps -A -o pid= -o ppid= -o etime= -o comm= 2>/dev/null};
    my (%ppid, %etime, %comm, %kids);
    for (@ps) {
      next unless /^\s*(\d+)\s+(\d+)\s+(\S+)\s+(.*?)\s*$/;
      ($ppid{$1}, $etime{$1}, $comm{$1}) = ($2, $3, $4);
      push @{ $kids{$2} }, $1;
    }
    unless (exists $ppid{$$}) {
      if (open my $fh, ">", "$rd/test-cap-off") { print $fh "ps cannot list this host\x27s processes\n"; close $fh }
      exit 0;
    }
    # The processes that start tests as dune does: also those that wrote their
    # pid into `test-parents` in the record directory -- the replay of a direct
    # repeat (tools/replay-rules.sh), which runs the rules of a test without dune.
    # Only one descending from the resolver counts, so a stale line cannot
    # name an unrelated process.
    my (%dune, %parents, @queue);
    if (open my $pf, "<", "$rd/test-parents") {
      while (<$pf>) { $parents{$1} = 1 if /^(\d+)$/ }
      close $pf;
    }
    @queue = ($resolver);
    while (@queue) {
      for my $k (@{ $kids{ shift @queue } || [] }) {
        push @queue, $k;
        # By name: a `dune` on PATH that is a shebang WRAPPER would be named
        # dune too (Linux names a script by its basename), and the real dune
        # below it would then be timed as a test.
        $dune{$k} = 1 if $comm{$k} =~ m{(?:^|/)dune(?:\.exe)?$} || $parents{$k};
      }
    }
    my %seen;
    my @tests = sort { $a <=> $b } grep { $dune{ $ppid{$_} } } keys %ppid;
    my $tok = @tests ? $tokens->(@tests) : {};
    for my $p (@tests) {
      my $id = $ppid{$p};
      my $token = defined $tok->{$p} ? $tok->{$p} : "";
      if ($born{$p} && $born{$p}[0] eq $id
          && ($token eq "" || $born{$p}[2] eq "" || $born{$p}[2] eq $token)) {
        $seen{$p} = [$id, $born{$p}[1], $born{$p}[2] ne "" ? $born{$p}[2] : $token];
      } else {
        my $hint = $etime{$p} =~ /^(?:\d+-)?(?:\d+:){1,2}\d+$/ ? $seconds->($etime{$p}) : 0;
        my $gap = $t - $last;
        $hint = $gap if $hint > $gap;
        $hint = 0 if $hint < 0;
        $seen{$p} = [$id, $t - $hint, $token];
      }
      my $e = int($t - $seen{$p}[1]);
      next if $e < $cap;
      my $args = qx{ps -o args= -p $p 2>/dev/null};
      chomp $args;
      $args = $comm{$p} if $args eq "";
      my $dir = readlink("/proc/$p/cwd");
      $dir = "" unless defined $dir;
      if (open my $fh, ">", "$rd/test-cap.tmp") {
        print $fh "cap $cap\nelapsed $e\npid $p\nkind $kind\ncommand $args\n", ($dir ne "" ? "dir $dir\n" : "");
        close $fh;
        rename "$rd/test-cap.tmp", "$rd/test-cap";
      }
      print "$who: per-test cap: pid $p has run ${e}s, past the ${cap}s cap: $args",
        ($dir ne "" ? " (in $dir)" : ""), "\n",
        "$who: ending the run, not judged: a test that outlives the per-test cap is taken as hung",
        ($kind eq "gpu" ? ", and on a GPU batch a wedged device hangs every test after it" : ""),
        " (gh-ocannl-1211). Legitimately long? $remedy.\n";
      kill "ALRM", $sup if getppid() == $resolver;
      exit 0;
    }
    %born = %seen;
    $last = $t;
    sleep $poll;
  }
'
