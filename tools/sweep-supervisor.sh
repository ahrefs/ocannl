#!/usr/bin/env bash
# The sweep's process supervisor and the quoting it is shipped with (gh-ocannl-1380), sourced by
# tools/sweep.sh and by its fixtures and live witnesses, so they all drive this text rather than a
# copy of it. Sourcing defines `sq`, `capped_perl`, `capped`, `capped_bg` and `remote_capped` and
# runs nothing; it needs no caller-owned variable. `capped_bg` replaces the calling shell, so it is
# valid only backgrounded (see below). `$capped_perl` is a perl program, run as
# `perl -e "$capped_perl" -- [--hold <why>] <cap> <command...>`.

# POSIX single-quoting, so a generated command can be handed to `sh -c` on the
# far side without the remote shell re-splitting or re-expanding any of it.
sq() { printf "'%s'" "$(printf %s "$1" | sed "s/'/'\\\\''/g")"; }

# Cap a local command, killing its whole process group when the cap expires.
# `perl -e 'alarm N; exec ...'` is not enough on its own: alarm survives exec, so
# SIGALRM reaches only the immediate child while dune and every compiler it
# spawned keep running -- holding _build locks that the NEXT unit on this machine
# (the local backends share one worktree) would then contend with, turning one timeout
# into a cascade. Exits 142 on expiry, matching the outcome mapping of
# tools/sweep.sh (run_unit).
#
# INT, TERM and HUP are forwarded to that group and REAPED before this exits.
# Putting the child in its own group is what makes forwarding necessary: a signal
# sent to the sweep's group (a terminal Ctrl-C, the scheduler cancelling the
# task) reaches bash and this supervisor but not the child, so without this dune
# would survive while the caller's trap released the lock -- letting the next
# sweep reset the worktree under a still-running build, which is the corruption
# the lock exists to prevent.
#
# The TERM->KILL grace is measured on the GROUP, not on the leader. The leader is
# a `sh -c` that dies on the first TERM, so waiting for IT to exit would end the
# grace in ~0.1s -- and the descendants are the entire reason to escalate: a dune
# worker or a compiler with a TERM handler wants a moment to unlink its
# temporaries and drop its _build lock cleanly. `kill 0, -$pid` counts what is
# still in the group, and the leader is reaped on each pass so its zombie does
# not read as a survivor and hold the grace open for the full interval.
#
# THE SLEEP GUARD (gh-ocannl-1035). With a leading `--hold <why>` the supervisor
# first takes a logind BLOCK inhibitor on sleep:idle, and the child runs only
# once it is held. On a native-Ubuntu box nothing else stops a suspend from
# taking the box mid-unit -- another session's `wake-lab.sh sleep`, or the
# laptop's own idle policy -- because the lab lock is advisory and binds only
# tools that go through wake-lab.sh; with a block inhibitor held, systemd's
# `--check-inhibitors=yes suspend` (wake-lab's path) and logind itself refuse.
# The WSL lanes do not take it: a guest's inhibitor cannot stop its Windows host
# from sleeping, and their Windows-side holder is what keeps them alive.
#
# It is the design of fleet-worker.sh's `execution hold` (ludics-lite#317),
# restated here rather than called, and both halves of that are deliberate:
#   - the inhibitor is held by a HELPER BESIDE the unit, never a wrapper around
#     it. A wrapping `systemd-inhibit -- <unit>` closes every descriptor above 2
#     in its child (the far-side worktree lock is fd 9), turns a signal death into
#     exit 1 (the outcome mapping reads 142 as `timeout`), and is itself the
#     process an ssh drop hangs up on -- so a dropped connection that kills it
#     either takes the unit with it or, if the unit outlives it, leaves the unit
#     running unguarded. The helper is `systemd-inhibit ... sh -c 'echo HELD; exec
#     cat'` reading a LIFETIME PIPE whose write end this supervisor and the unit
#     inherit, in its own session so no signal aimed at the unit's session reaches
#     it: the inhibitor lives exactly as long as anything in the unit's process
#     tree does, and the kernel ends it however the tree ends.
#   - it is not `fleet-worker.sh execution hold`, because the sweep must not
#     depend on a box's skills checkout being current: a box outside the fleet's
#     worker roster is never refreshed by the fleet's preflight, and on
#     2026-09-23 tuf's (the box this guard matters most on -- a laptop) was at a
#     revision with no `execution hold` at all, where the call would have exited 2
#     and filed every unit as a red suite. perl, which this supervisor already
#     is, carries no such dependency.
# Fail-open and loud, as `hold` is: no systemd-inhibit, or a refusal (the polkit
# grant self-improve's Linux bootstrap installs is missing), prints a
# `sweep-hold: WARNING` line into the unit's log -- the sweep repeats it on the
# unit's summary line -- and runs the unit bare. A run refused over a setup step
# would cost the coverage the guard exists to protect. No `--no-ask-password`:
# systemd-inhibit gained it in v257 and Ubuntu 24.04's 255 rejects it; with pipes
# for stdio there is no terminal to start a polkit agent on, so a denial is
# immediate anyway.
capped_perl='
  use POSIX ();
  use Fcntl ();
  my $why;
  if (@ARGV && $ARGV[0] eq "--hold") { shift; $why = shift; }
  my $cap = shift;
  my $life = defined $why ? hold_sleep($why) : undef;
  my $pid = fork();
  die "fork: $!" unless defined $pid;
  if (!$pid) { setpgrp(0, 0); exec @ARGV; exit 127 }
  my $reap = sub {
    my $code = shift;
    kill "TERM", -$pid;
    for (1 .. 100) {
      waitpid($pid, POSIX::WNOHANG());
      last unless kill 0, -$pid;
      select undef, undef, undef, 0.1;
    }
    kill "KILL", -$pid;
    waitpid($pid, 0);
    exit $code;
  };
  $SIG{ALRM} = sub { $reap->(142) };
  $SIG{INT} = sub { $reap->(130) };
  $SIG{TERM} = sub { $reap->(143) };
  $SIG{HUP} = sub { $reap->(129) };
  alarm $cap;
  waitpid($pid, 0);
  my $st = $?;
  alarm 0;
  exit($st & 127 ? 128 + ($st & 127) : $st >> 8);
  sub unheld {
    my ($reason) = @_;
    print STDERR "sweep-hold: WARNING: running WITHOUT a sleep guard, so nothing at the OS level"
      . " stops a suspend under this run -- $reason\n";
    return undef;
  }
  sub hold_sleep {
    my ($why) = @_;
    my ($dir) = grep { length && -x "$_/systemd-inhibit" } split /:/, ($ENV{PATH} // "");
    return unheld("no systemd-inhibit on PATH") unless defined $dir;
    my $inhibitor = "$dir/systemd-inhibit";
    pipe(my $life_r, my $life_w) or return unheld("pipe: $!");
    pipe(my $ready_r, my $ready_w) or return unheld("pipe: $!");
    my $middle = fork();
    return unheld("fork: $!") unless defined $middle;
    if (!$middle) {
      POSIX::setsid();
      my $helper = fork();
      if (defined $helper && !$helper) {
        POSIX::dup2(fileno($life_r), 0);
        POSIX::dup2(fileno($ready_w), 1);
        POSIX::dup2(fileno($ready_w), 2);
        POSIX::close($_) for 3 .. 255;
        exec {$inhibitor} $inhibitor, "--what=sleep:idle", "--mode=block", "--who=ocannl-sweep",
          "--why=$why", "--", "sh", "-c", "echo HELD; exec cat >/dev/null";
        POSIX::_exit(127);
      }
      # The helper is no child of this supervisor (the double fork), so its pid is reported up the
      # readiness pipe: a helper that never answers is killed by pid, not left behind.
      syswrite($ready_w, "PID $helper\n") if defined $helper;
      POSIX::_exit(0);
    }
    waitpid($middle, 0);
    close $life_r;
    close $ready_w;
    my $wait = ($ENV{OCANNL_TOOL_SWEEP_HOLD_WAIT} // "") =~ /^[1-9][0-9]*$/
      ? $ENV{OCANNL_TOOL_SWEEP_HOLD_WAIT} : 30;
    my ($text, $held, $deadline) = ("", 0, time + $wait);
    while (!$held) {
      my $left = $deadline - time;
      last if $left <= 0;
      my $rin = "";
      vec($rin, fileno($ready_r), 1) = 1;
      last unless select($rin, undef, undef, $left) > 0;
      my $n = sysread($ready_r, my $chunk, 4096);
      last unless $n;
      $text .= $chunk;
      $held = 1 if $text =~ /^HELD$/m;
    }
    close $ready_r;
    if (!$held) {
      close $life_w;
      my ($helper) = $text =~ /^PID ([0-9]+)$/m;
      kill "TERM", $helper if $helper;
      $text =~ s/^PID [0-9]+\n?//mg;
      $text =~ s/\s+/ /g;
      $text =~ s/^ | $//g;
      $text = "no answer after ${wait}s" if $text eq "";
      return unheld("$inhibitor refused: " . substr($text, 0, 200));
    }
    my $flags = fcntl($life_w, Fcntl::F_GETFD(), 0);
    fcntl($life_w, Fcntl::F_SETFD(), $flags & ~Fcntl::FD_CLOEXEC());
    print STDERR "sweep-hold: sleep:idle block inhibitor held for: $why\n";
    return $life_w;
  }
'
# First argument is the budget in seconds, so the remote path can allow for
# far-side cleanup and ssh teardown on top of its own cap.
capped() { perl -e "$capped_perl" -- "$@"; }

# The same supervisor, for the backgrounded unit calls. `exec` is the point:
# backgrounding a shell FUNCTION runs it in a subshell, so `capped ... &` would
# put the SUBSHELL's pid in $! -- and a relayed signal would kill that wrapper
# while the supervisor, and dune under it, carried on. exec replaces the subshell
# so $! names the supervisor itself. Valid only backgrounded: called
# synchronously it would replace this script.
capped_bg() { exec perl -e "$capped_perl" -- "$@"; }

# The FAR-SIDE cap, emitted as shell text for the remote shell to run. It is the
# same perl supervisor, and for the same reason: `timeout -k` is not a
# group-killing bound everywhere it exists. uutils coreutils -- Ubuntu's default
# since 25.10, and what rog-nv's WSL side runs -- delivers the TERM phase to the
# group but escalates the -k KILL to the DIRECT CHILD only, so a descendant that
# ignores or outlives TERM (a wedged pool worker, a CUDA call that never returns)
# is reparented and keeps running while `timeout` reports 137 (gh-ocannl-727).
# The unit would then be filed as `timeout` -- coverage lost -- while still
# holding the GPU and the remote worktree lock that the NEXT sweep's preparation
# must take. perl(1) is a firmer assumption than GNU-vs-uutils semantics: the
# sweep hosts are Ubuntu (native or WSL), where it is part of the base system, and
# tools/test-run.sh already requires it there.
#
# The supervisor exits 142 on expiry -- capped()'s code, which the outcome
# mapping in tools/sweep.sh already reads as `timeout` -- so both sides now report a hang
# identically. It also reaps on HUP, which is what the remote end gets when the
# local ssh is killed by the outer run_capped, so a lost connection tears the
# remote unit down instead of orphaning it.
#
# A third argument is the sleep guard's `why` (see capped_perl): sleep_guard_why
# gives one for a native-Ubuntu destination and nothing otherwise, so a call site
# passes it unconditionally and the transport decides.
remote_capped() { # cap command [sleep-guard-why]
  if [ -n "${3:-}" ]; then
    printf 'perl -e %s -- --hold %s %s sh -c %s' "$(sq "$capped_perl")" "$(sq "$3")" "$1" "$(sq "$2")"
  else
    printf 'perl -e %s -- %s sh -c %s' "$(sq "$capped_perl")" "$1" "$(sq "$2")"
  fi
}
