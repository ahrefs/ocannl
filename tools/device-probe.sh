#!/usr/bin/env bash
# The device-health probe (gh-ocannl-1211, gh-ocannl-1252), shared by
# tools/test-run.sh (a `run`/`start` GPU batch's `_probe`, and each `repeat`
# iteration of one), tools/sweep.sh (each GPU unit, and its serial rerun) and
# tools/machine-verify-far.sh (a trip's GPU operations, the text passed in like
# the width table). One copy, for the reason tools/per-test-cap.sh is one copy:
# the sweep and machine-verify ran a wedged tuf device's tests until their
# whole-unit caps. Sourced or evaluated, never executed, and POSIX sh
# throughout: machine-verify's far side evaluates it under dash.

# The bound on one query when nothing names another, in seconds: a healthy one
# answers in 0.2-0.3s, a wedged one never does.
DEVICE_PROBE_CAP_DEFAULT=60

# One fresh device query, bounded. A wedged device does not fail a query, it
# never answers one -- the recovery trip after the tuf fault found
# `device_props` spinning at 100% CPU before printing anything -- so the answer
# read here is whether the query FINISHED within the bound, and a query that
# fails fast (no such device on this box) is recorded and passed over: its
# tests fail fast too, rather than spin. Prints `exit <code> <seconds>`, or
# `hung <seconds>` for a query KILLed at the bound; the query's own output goes
# to the file given. Polled rather than waited for, so a query stuck where even
# KILL cannot reach it (in the driver, uninterruptibly) still gets its verdict
# on time.
#
# The query leads a process group of its own, and the bound KILLs that whole
# group: a stand-in that forks the real query rather than exec'ing it must not
# leave it running on the faulted device behind a DEVICE UNHEALTHY verdict,
# whose callers do not all reap their groups on a normal exit (Codex review
# round 2 on PR #1172). Out of the caller's group, the query no longer gets
# the caller's signals, so a TERM, INT or HUP to this process KILLs the query's
# group before it exits with the signal's status.
probe_perl='
  use POSIX ();
  use Time::HiRes ();
  my ($bound, $out, @cmd) = @ARGV;
  my $t0 = Time::HiRes::time();
  my $pid;
  my $kill_query = sub { kill("KILL", -$pid) or kill("KILL", $pid) if $pid };
  for my $sig ([TERM => 143], [INT => 130], [HUP => 129]) {
    my $code = $sig->[1];
    $SIG{ $sig->[0] } = sub { $kill_query->(); exit $code };
  }
  $pid = fork();
  defined $pid or do { print "error fork: $!\n"; exit 0 };
  if (!$pid) {
    $SIG{$_} = "DEFAULT" for qw(TERM INT HUP);
    setpgrp(0, 0);
    open STDIN, "<", "/dev/null";
    open STDOUT, ">", $out or exit 126;
    open STDERR, ">&", \*STDOUT;
    exec @cmd or exit 127;
  }
  my $st;
  while (1) {
    if (waitpid($pid, POSIX::WNOHANG()) == $pid) { $st = $?; last }
    last if Time::HiRes::time() - $t0 >= $bound;
    Time::HiRes::sleep(0.05);
  }
  my $took = sprintf "%.1f", Time::HiRes::time() - $t0;
  if (defined $st) {
    print "exit ", (($st & 127) ? 128 + ($st & 127) : $st >> 8), " $took\n";
  } else {
    $kill_query->();
    for (1 .. 50) { last if waitpid($pid, POSIX::WNOHANG()) != 0; Time::HiRes::sleep(0.1) }
    print "hung $took\n";
  }
'

# The probe itself, as a POSIX sh program run with `sh -c "$device_probe_sh"
# device-probe WHO PERL BOUND RECORDS PROGRAM BACKEND...` from the root of the
# tree whose tests it guards. Each backend is asked once, through PROGRAM
# --ocannl_backend=<backend> from test/config -- so the query reads the
# configuration the tests read, with the backend pinned on its command line --
# under probe_perl (PERL) bounded at BOUND seconds. Each answer is a line of
# RECORDS/device-probe (the query's own output in RECORDS/device-probe-<b>.out);
# a backend whose query did not answer refuses: the explanation, WHO-prefixed,
# and the record go to stdout and the program exits 69, the code every caller
# reads as DEVICE UNHEALTHY. Otherwise it prints nothing and exits 0: the
# callers keep their logs' first line dune's (tools/test-run.sh's dune_refusal
# reads it).
device_probe_sh='
  who=$1 perl_src=$2 bound=$3 rd=$4 prog=$5
  shift 5
  wedged=
  for b do
    answer=$(cd test/config && perl -e "$perl_src" "$bound" "$rd/device-probe-$b.out" \
      "$prog" "--ocannl_backend=$b" 9>&-)
    read -r how code took <<EOF
$answer
EOF
    case $how in
      exit)
        if [ "$code" = 0 ]; then line="$b: answered in ${took}s"
        else line="$b: failed fast (exit $code in ${took}s; no such device here?), not counted against the device"
        fi ;;
      hung)
        line="$b: NO ANSWER within ${bound}s; the query was killed"
        wedged="$wedged $b" ;;
      *) line="$b: the query could not be run ($answer)" ;;
    esac
    printf "%s\n" "$line" >>"$rd/device-probe"
  done
  [ -n "$wedged" ] || exit 0
  printf "%s: device probe: a fresh device query for%s did not answer within %ss: the\n" \
    "$who" "$wedged" "$bound"
  echo "  device is taken as wedged, and dune was not started -- a faulted GPU makes every test spin"
  echo "  rather than fail (gh-ocannl-1211). Stop GPU work on this box and read its kernel journal"
  echo "  (journalctl -k: gfxhub/CPC page faults, ring timeouts, NVRM Xid); recovery may need a reboot."
  sed "s/^/  /" "$rd/device-probe"
  exit 69
'
