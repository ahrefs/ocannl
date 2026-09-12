#!/usr/bin/env bash
# Run dune test targets (`dune runtest ...`, `dune build @slow`, directory
# aliases) with a correct, compact, machine-readable verdict -- so that nobody,
# and especially no coding agent, hand-rolls shell around dune again.
#
# Each ingredient answers a failure mode that has actually cost a session:
#   - Unpiped status: `dune runtest 2>&1 | tail` reports TAIL's status (no
#     pipefail), so a promotion diff reads as a green run.
#   - No `status` variable anywhere: in zsh it is a read-only alias for `$?`,
#     so ad-hoc wrappers that use it die before printing their sentinel and a
#     green suite looks failed.
#   - A wall-clock cap: a hung run (macOS XProtect stalling a fresh exe, a
#     wedged backend) otherwise strands whatever is waiting on it.
#   - A verdict FILE, not a process probe: waiter loops on `pgrep -x dune`
#     match the editor's immortal `dune ocaml-merlin` daemons and spin forever
#     (one PR review accumulated ten such stranded shells); `kill -0 $pid` can
#     latch onto a recycled pid, and answers yes for a ZOMBIE -- which is why
#     every liveness question here reads process state too (proc_alive,
#     group_alive). `wait` here polls for the verdict file with a hard timeout,
#     so it cannot strand.
#
# Usage:
#   tools/test-run.sh run   [--cap N] [DUNE ARGS...]   # foreground; digest; dune's status (2: refused)
#   tools/test-run.sh start [--cap N] [DUNE ARGS...]   # detached; survives the session
#   tools/test-run.sh repeat [--cap N] [--alone] N [DUNE ARGS...]
#                                                      # compare N isolated runs
#   tools/test-run.sh status [RUN|last]                # one-shot, never blocks
#   tools/test-run.sh wait   [RUN|last] [--timeout N]  # bounded; exits with the run's status
#   tools/test-run.sh stop   [RUN|last]                # TERM the run's process group
#   tools/test-run.sh list                             # recent runs and their states
#   tools/test-run.sh paths FIELD [RUN|last]          # one absolute path (see below)
#   tools/test-run.sh lock-status [RUN|last]          # idle/held; exits 0/3 (2 error)
#   tools/test-run.sh idle                             # 0 idle, 3 locked, 2 unreadable
#                                                     # snapshot, not a reservation
#
# Read-only queries: `paths run [RUN|last]` resolves a recorded run (default last).
# `paths worktree|runs|lock|owner|last [RUN|last]` without a run names THIS
# script's worktree and state root, even before its first launch; with a run it
# names that run's recorded worktree/state (including legacy in-tree locks).
# `paths last` names the pointer FILE, not the run it points to. Output is one
# raw absolute path and a newline, never shell code; exit 2 means invalid input
# or unavailable metadata. `lock-status` without a run probes THIS worktree's
# current and legacy locks; with a run, only its recorded lock. It prints idle
# (0) or held (3); an unreadable lock is an error (2), never idle. These are
# snapshots, not reservations or evidence that a particular process is alive.
# Neither query creates state, publishes pointers, or launches a toolchain.
#
# `repeat` runs each iteration through dune in a freshly cleaned, cache-disabled
# build context, keeps its separate stdout/stderr and exit status, and compares
# every pair. `--alone` adds `-j 1`,
# so no sibling dune action overlaps the selected target. Its cap is per
# iteration; N must be at least 2. An stdout/status difference is red (exit 1
# when dune itself stayed green); stderr-only drift is reported distinctly but
# is not red. Any red dune iteration keeps a nonzero dune status.
#
# Exit codes: `run` and `wait` exit with dune's status, with ONE substitution:
# an invocation dune's own CLI refused -- an unknown option or subcommand, a
# missing or malformed operand -- exits 2, this script's usage code, under the
# verdict `INVOCATION REFUSED`. dune prints `dune: <complaint>` then `Usage:
# dune ...`, exits 1 and runs nothing, and that 1 is not a test result: read as
# one it sends a session debugging code that was never compiled (staging#652).
# The recorded status (the `exit:` sentinel, `status`, `list`) stays dune's
# own 1; the substitution is the caller-facing one, so a caller branching on
# the status needs no log parsing to tell "fix the command line" from "read
# the failures". `repeat` preserves the
# first nonzero dune status, or exits 1 when otherwise-green stdout/statuses
# differ, and exits 2 with the same verdict when its first iteration is refused
# (there is nothing to repeat) (142 = the cap expired,
# 143/130 = cancelled, 137 = SIGKILLed, 124 = `wait` itself timed out; dune
# never reaches those on its own). `status` exits 0 finished, 3 still running
# (or verdict publication in flight), 1 died without a verdict. Usage and lock
# refusals exit 2 -- every one of them, including EITHER misplacement of this
# script's own options: before the subcommand (`--cap 5400 run ...`) and after
# the dune arguments (`run build @alias --cap 900`). Options go in between:
# `run --cap 5400 build @alias`. The second misplacement is also refused BEFORE
# dune is spawned, by the scan below, because for those two words the correct
# order is known and the message can name it; every other refusal is dune's
# to make and is recognised afterwards in the digest, quoting dune's complaint.
# But a launcher that swallows the status -- e.g. an agent harness's background
# mode reporting its own wrapper's 0 -- turns that refusal into a false green,
# so read the message, not only the code: a usage error runs no tests at all.
#
# Everything after the options is dune's argv, verbatim (default: `runtest`):
#   tools/test-run.sh run runtest test/operations
#   tools/test-run.sh run build @slow
#   OCANNL_BACKEND=cuda tools/test-run.sh run runtest
#
# Prefer `run`. In an agent harness, launch `run` through the harness's own
# background mode and let the harness notify on exit -- that already removes
# every reason to write a waiter. `start`/`wait` exist only for a run that must
# outlive the launching session. The cap defaults to $OCANNL_TOOL_TEST_CAP or 3600s;
# `--cap 0` disables it (then supply your own bound).
#
# One run at a time per worktree, enforced with an flock: a second `run`/`start`/`repeat`
# refuses loudly, pointing at the active run, instead of queueing behind dune's
# own lock -- "I lost track of a run so I started another" is exactly the spiral
# this script exists to prevent. `stop` the active run if it is truly stale.
# The lock, its owner pointer and the `last` pointer live under the runs
# directory ($OCANNL_TOOL_TEST_RUNS, default ~/.ocannl-test-runs), keyed by the
# worktree's path -- never in the worktree, which a run leaves exactly as it
# found it. A run is two processes: this launching shell, which takes the lock
# and publishes the run, and a perl supervisor that inherits the lock, caps
# and signals dune, and records the verdict (see supervisor_perl).
#
# Windows: run it from Git Bash, whose MSYS perl carries the flock and the cap.
# Best-effort even there -- process-group kills may only reach dune itself, not
# its compiler children. A Cygwin bash ships no perl by default, so the
# preflight below refuses it outright rather than letting the lock misreport
# what happened (gh-ocannl-662).

set -u

die() { echo "test-run: $*" >&2; exit 2; }

normalize_cap() {
  # A mistyped cap would reach perl's numeric compare as 0 and silently
  # disable the alarm -- the one property this script must never lose.
  case $cap in '' | *[!0-9]*) die "--cap must be a nonnegative integer of seconds (0 disables)" ;; esac
  # Bounded BEFORE arithmetic: an oversized value would wrap in bash's signed
  # arithmetic, and perl's alarm range is signed too. Nine digits is ~31 years.
  [ ${#cap} -le 9 ] || die "--cap too large (max 9 digits)"
  # Leading zeroes are accepted as decimal rather than reaching bash as octal.
  cap=$(( 10#$cap ))
}

reject_misplaced_options() {
  # dune has no `--cap` and no `--alone`, so either word among the arguments
  # FORWARDED to dune is this script's options written on the wrong side of the
  # target -- never something a caller could mean. Left to reach dune it exits 1
  # with `dune: unknown option`, having built nothing; the digest would now
  # report that as INVOCATION REFUSED, but only this guard knows the correct
  # order, so it refuses first and names it, without spawning dune at all.
  # The scan stops at dune's own `--`, past which the words belong to an
  # executable (`run exec foo.exe -- --cap 900` passes its cap to foo.exe).
  for arg do
    case $arg in
      --) return 0 ;;
      --cap | --cap=* | --alone | --alone=*)
        case $sub in
          repeat) form='repeat [--cap N] [--alone] N [DUNE ARGS...]' ;;
          *) form="$sub [--cap N] [DUNE ARGS...]" ;;
        esac
        die "$arg belongs before the dune arguments, not after them:
  tools/test-run.sh $form
  dune has no such option, so this would have exited 1 having run nothing." ;;
    esac
  done
}

select_dune() {
  # On Windows the environment rewrite is required even if dune is already on
  # PATH: opam's native output otherwise leaves an MSYS shell half-configured.
  case ${OSTYPE:-} in
    msys* | cygwin*)
      . tools/opam-env.sh ||
        die "tools/opam-env.sh failed; refusing an unrewritten Windows toolchain"
      ;;
  esac
  command -v dune >/dev/null 2>&1 || . tools/opam-env.sh
  command -v dune >/dev/null 2>&1 || die "dune not found (opam environment not set up?)"
  # Preserve dune's status while filtering only the known Windows linker noise.
  case ${OSTYPE:-} in msys* | cygwin*) DUNE=tools/dune-quiet.sh ;; *) DUNE=dune ;; esac
}

# Pin to the repo containing THIS script (promote.sh convention): dune then runs
# at this worktree's root no matter where the caller's cwd wandered, and the
# per-worktree lock below keys on the tree actually being tested.
# -P: the physical path, so the same worktree entered through a symlink and
# through its real path derive the same key, lock file, and recorded wt.
cd -P "$(dirname "$0")/.." || die "cannot cd to repo root"

[ -r scripts/process-group.sh ] || die "cannot read scripts/process-group.sh"
# shellcheck source=../scripts/process-group.sh
. scripts/process-group.sh

# perl is load-bearing rather than a convenience: the per-worktree flock, the
# cap supervisor and the atomic rename behind the `last` pointer are all
# `perl -e` one-liners, and every subcommand reaches at least one of them.
#
# Checked ONCE, here, because the alternative is what this check was written
# from: with perl absent, `take_lock`'s perl exited 127, which is not one of the
# statuses that block assigns a meaning, so it fell through to the catch-all
# "another test-run is active in this worktree" -- a phantom lock holder,
# reported with recovery advice (`stop` it) that cannot work because there is
# nothing to stop. That is how the Windows CI smoke step refused with exit 2
# under the Cygwin bash `shell: bash` resolves to once setup-ocaml has prepended
# opam's cygwin to PATH (gh-ocannl-662).
#
# The module is loaded rather than the binary merely found: `use Fcntl ":flock"`
# is what the lock needs, and a perl that cannot provide it would fail in the
# same misread place.
perl -e 'use Fcntl ":flock"; exit 0' 2>/dev/null || die "perl with Fcntl not found; it carries the lock, the cap and the \`last\` pointer.
  On Windows run this from Git Bash, whose MSYS perl has both; a Cygwin bash
  ships neither unless its perl package is installed."

# OCANNL_TOOL_ is the namespace the library reserves for names that address OCANNL
# without being configuration: an OCANNL executable warns at startup about any other
# `OCANNL_...` variable it finds (gh-ocannl-629), and these are exported into the
# environment of every test this script runs.
RUNS=${OCANNL_TOOL_TEST_RUNS:-$HOME/.ocannl-test-runs}
case ${1:-} in
  paths | lock-status)
    # Resolve existing symlink prefixes physically, but allow an absent state
    # root without creating it. Walking components also handles missing/../x.
    RUNS=$(perl -MCwd=abs_path -MFile::Spec -e '
      my $path = File::Spec->rel2abs($ARGV[0]);
      my $out = "/";
      for my $part (split m{/+}, $path) {
        next if $part eq "" || $part eq ".";
        if ($part eq "..") { $out =~ s{/[^/]+$}{}; $out ||= "/"; next; }
        $out =~ s{/$}{};
        $out .= "/$part";
        if (-e $out || -l $out) {
          $out = abs_path($out) // die "cannot resolve $out\n";
          -d $out or die "not a directory: $out\n";
        }
      }
      print "$out\n";
    ' "$RUNS") || die "cannot resolve runs directory"
    ;;
  *) mkdir -p "$RUNS" || die "cannot create $RUNS"
     RUNS=$(cd "$RUNS" && pwd -P) || die "cannot resolve $RUNS" ;;
esac
# Canonicalized: run identities (owner pointer, `last`, wt cross-references)
# are compared as strings, so a relative override must not record a
# different spelling than a later absolute reference resolves to.
# The worktree key makes every per-worktree fact under $RUNS -- the lock, its
# owner pointer and the `last` pointer -- per-checkout, so concurrent sessions
# in different worktrees never read each other's verdicts or contend on each
# other's lock. The readable basename is for humans listing $RUNS; the crc of
# the full path is what keeps two paths differing only in punctuation from
# sharing a key. One definition, applied by the launcher to its own root and
# by `stop` and retention to the root a run RECORDED, so a sibling worktree
# derives the same key from the same path.
wt_key_of() { # <worktree root, physical path> -> its key
  printf '%s' "$(basename "$1" | tr -c 'A-Za-z0-9' '_')$(printf %s "$1" | cksum | awk '{print $1}')"
}
wt_key=$(wt_key_of "$PWD")
# Every file this script keeps per worktree lives under $RUNS, keyed as above,
# and NOT in the worktree: a run must leave nothing behind in the tree it
# tested. A gitignored dotfile that survives the run reads as "ignored local
# data" to anything that judges a worktree finished by its cleanliness, and
# the teardown after every worktree-based session refused over three of them
# (gh-ocannl-606). The cost, stated: OCANNL_TOOL_TEST_RUNS relocates the lock
# together with the diagnostics, so two sessions testing the SAME worktree
# see each other's lock only under the same override. A split there is caught
# one layer down, by dune's own `_build/.lock` -- two dune instances cannot
# both build one tree -- but without this script's pointer at the run to stop.
LOCK=$RUNS/lock-$wt_key   # the flock target: fd 9 of every process of a run
OWNER=$RUNS/owner-$wt_key # the run directory holding the lock (see take_lock)
LAST=$RUNS/last-$wt_key   # the run this worktree published most recently

# The supervisor: ONE perl process that owns a run from the instant it
# inherits lock fd 9 until the verdict is on disk -- the second of a run's two
# parties, the launching shell being the first. It caps the run, relays
# INT/TERM to dune's process group and reaps it, records its own identity (pid
# and start token) for status/stop, and on every exit path writes the `exit:`
# sentinel and the verdict file, then exits -- which is the lock's release
# (see take_lock). The sibling of sweep.sh's supervisor (see the rationale
# there). `perl -e 'alarm N; exec ...'` alone is not enough: alarm survives
# exec, so SIGALRM would reach only dune while every compiler it spawned kept
# running and holding _build locks. Exits 142 on expiry.
#
# This used to be three parties. A wrapper subshell between launcher and
# supervisor published the verdict and held the lock through a publication
# handshake with the launcher -- an arbiter file, an acknowledgement, an
# abandonment marker and a metadata lock, each closing the interleaving the
# previous one had opened (gh-ocannl-606). It existed because bash 3.2 has no
# BASHPID for a subshell to record itself by. Perl knows its pid, so the
# wrapper's duties moved here, and the handshake went with its reason: the
# launcher publishes the run's pointers BEFORE this process exists, holding
# the lock itself, so no fast run can finish and release under a publication
# still in flight. What the collapse gives up: a SIGKILL aimed at this process
# alone leaves no party to record a verdict for it -- `status` then reports
# the run as died, and `stop` reaps the dune group it recorded, exactly as
# after a group-wide SIGKILL before.
#
# Signals: a foreground run gets HUP relayed as TERM by its launcher, and a
# detached run must survive its session -- so HUP is ignored here whenever
# OCANNL_TOOL_TESTRUN_BG is set, and defaulted back in dune's child. Once a
# signal or dune's own exit has started the finish, every later signal is
# ignored: the verdict write is the one critical section, and KILL remains
# available. setpgrp is eval-guarded and the group kill falls back to a plain
# kill, for MSYS perl where process groups are shaky. The child deliberately
# KEEPS lock fd 9: whatever dune leaves behind -- in its own group or setsid'd
# out of it -- goes on holding the worktree lock while it can still mutate
# _build, and the lock clears exactly when the last such holder exits. The
# child records its pgid (only once confirmed to LEAD its own group, so a
# group-kill can never hit the caller where setpgrp failed) and its start
# token, so `stop` can reap a group that outlived this process.
#
# OCANNL_TOOL_TESTRUN_RD: where pgid/gtoken go (the run directory, or a repeat
# iteration's). OCANNL_TOOL_TESTRUN_OWN: the run directory whose identity and
# verdict this process owns -- unset for a repeat iteration, whose coordinator
# owns both and reads this process's exit status instead.
supervisor_perl='
  use POSIX ();
  my $cap = shift;
  my $rd = $ENV{OCANNL_TOOL_TESTRUN_RD};
  my $own = $ENV{OCANNL_TOOL_TESTRUN_OWN};
  my ($pid, $finishing);
  my $code_of = sub { my $st = shift; ($st & 127) ? 128 + ($st & 127) : $st >> 8 };
  # The start token of THIS process, in the rendering ps_token recomputes for
  # any pid: Linux reads /proc/self/stat (clock ticks); elsewhere ps renders
  # lstart under the same pinned locale/TZ and squeeze. Sampled by the process
  # itself, never by a parent that could catch a recycled pid after a fast
  # exit.
  my $self_token = sub {
    my $tok = "";
    if (open my $sf, "<", "/proc/self/stat") {
      my $s = <$sf>;
      if ($s =~ /\)\s+(.*)$/) {
        my @f = split /\s+/, $1;
        $tok = defined $f[19] ? $f[19] : "";
      }
    } else {
      local $ENV{LC_ALL} = "C";
      local $ENV{TZ} = "UTC";
      $tok = qx{ps -o lstart= -p $$ 2>/dev/null};
      $tok =~ s/\s+$//;
      $tok =~ s/ +/ /g;
    }
    $tok;
  };
  my $write = sub { # path, content -> 1, or 0 with $! set
    open(my $fh, ">", $_[0]) or return 0;
    print $fh $_[1] or return 0;
    close $fh or return 0;
    1;
  };
  # Every exit path ends here. The verdict file is written aside and renamed:
  # its EXISTENCE is the completion signal status/wait key on, so it must
  # never be observable empty. Retried with backoff -- a filesystem that
  # filled up during the run may clear, and giving up silently would
  # downgrade a real verdict into a generic "died without a verdict".
  my $finish = sub {
    my $code = shift;
    $finishing = 1;
    $SIG{$_} = "IGNORE" for qw(ALRM INT TERM HUP);
    alarm 0;
    if ($own) {
      print STDOUT "exit: $code\n";
      my $ok = 0;
      for my $try (1 .. 3) {
        if ($write->("$own/exit.tmp", "$code\n") && rename("$own/exit.tmp", "$own/exit")) {
          $ok = 1;
          last;
        }
        sleep 2 if $try < 3;
      }
      print STDOUT "test-run: FAILED to record verdict $code (filesystem?)\n" unless $ok;
    }
    exit $code;
  };
  my $blast = sub { my $sig = shift; kill($sig, -$pid) or kill($sig, $pid) };
  my $reap = sub {
    my $code = shift;
    # A signal landing once the finish has begun -- after dune was reaped, or
    # during the verdict write -- must neither blast the stale pid/group (a
    # recycled pid could make that an innocent process) nor replace the real
    # verdict with the signal code.
    return if $finishing;
    $finishing = 1;
    if ($pid) {
      # The WNOHANG probe covers the statements between the reaping waitpid
      # and the bookkeeping -- there, $? still holds the status the main flow
      # has not yet read (captured before our own waitpid resets it).
      my $saved = $?;
      my $r = waitpid($pid, POSIX::WNOHANG());
      $finish->($code_of->($saved)) if $r == -1;
      $finish->($code_of->($?)) if $r == $pid;
      $blast->("TERM");
      # Grace for the WHOLE group, not just the leader: a leader that exits
      # fast must not collapse the grace of its descendants to one polling
      # interval. (Where the child has no group of its own, the group probe
      # fails and this degrades to the plain leader wait.)
      my $gone = 0;
      for (1 .. 50) {
        $gone = 1 if !$gone && waitpid($pid, POSIX::WNOHANG()) != 0;
        last if $gone && !kill(0, -$pid);
        select undef, undef, undef, 0.1;
      }
      if (!$gone || kill(0, -$pid)) {
        $blast->("KILL");
        # Bounded reap: a child stuck in uninterruptible kernel I/O keeps
        # even SIGKILL pending, and a blocking waitpid here would hang the
        # supervisor -- the cap must report its 142/143 regardless; the
        # stuck process then keeps the worktree lock through its fd 9.
        unless ($gone) {
          for (1 .. 50) {
            last if waitpid($pid, POSIX::WNOHANG()) != 0;
            select undef, undef, undef, 0.1;
          }
        }
        # Then let the group VANISH before finishing, bounded: the KILLed
        # members are orphans now, and until init reaps them a Linux group
        # probe still counts their corpses -- a repeat coordinator asking
        # right after this exit would find a reachable group with no leader
        # to verify and refuse to reuse its build tree. Under an init that
        # never reaps, the corpses are permanent and this wait ends on its
        # bound; that refusal is then the fail-closed answer it was built
        # to give.
        for (1 .. 10) {
          last unless kill(0, -$pid);
          select undef, undef, undef, 0.1;
        }
      }
    }
    $finish->($code);
  };
  # Armed BEFORE the identity record and the fork: a stop landing in the
  # launch window is honoured -- a pre-fork signal finishes with its code
  # before dune ever starts, and the verdict says CANCELLED.
  $SIG{ALRM} = sub { $reap->(142) };
  $SIG{INT} = sub { $reap->(130) };
  $SIG{TERM} = sub { $reap->(143) };
  $SIG{HUP} = $ENV{OCANNL_TOOL_TESTRUN_BG} ? "IGNORE" : sub { $reap->(129) };
  if ($own) {
    # STDOUT is the run log, shared with dune: unbuffered, so the sentinel
    # lands before the verdict file appears.
    $| = 1;
    # Identity first, token before pid: a reader that finds the pid finds the
    # token too. A run whose supervisor cannot record itself would be
    # invisible to status and uncancellable by stop while dune ran on, so
    # dune is not started: nothing ran, and the verdict (126) says so.
    unless ($write->("$own/ptoken", $self_token->() . "\n") && $write->("$own/pid", "$$\n")) {
      print STDOUT "test-run: cannot record the supervisor identity in $own: $!\n";
      $finish->(126);
    }
  }
  $pid = fork();
  unless (defined $pid) {
    print STDOUT "test-run: fork: $!\n" if $own;
    $finish->(126);
  }
  if (!$pid) {
    # SIG_IGN survives fork AND exec (HUP is ignored above in detached mode,
    # and the launching shell may have inherited an ignored INT): without this
    # reset dune would start deaf to the very signals the cap and `stop` rely
    # on, degrading every cancellation to the KILL escalation.
    $SIG{TERM} = "DEFAULT"; $SIG{INT} = "DEFAULT"; $SIG{HUP} = "DEFAULT";
    # Perl otherwise reserves the right to close inherited descriptors above
    # $^F at exec. fd 9 is the worktree lock and repeat additionally supplies
    # fd 6/8 as its descendant witness; keep that containment state attached
    # to Dune and to anything Dune launches.
    $^F = 9;
    eval { setpgrp(0, 0) };
    eval {
      if ($rd && getpgrp(0) == $$) {
        open my $fh, ">", "$rd/pgid" or die;
        print $fh $$;
        close $fh;
        # The leader publishes its OWN start token before exec: a token
        # sampled later by a parent could capture a recycled pid if the
        # leader exited first.
        my $tok = $self_token->();
        if ($tok ne "") {
          open my $gf, ">", "$rd/gtoken" or die;
          print $gf "$tok\n";
          close $gf;
        }
      }
    };
    exec @ARGV;
    exit 127;
  }
  alarm $cap if $cap > 0;
  waitpid($pid, 0);
  my $st = $?;
  # The finish flagged BEFORE anything else -- a handler firing in between
  # reads the status through its WNOHANG probe and finishes with the real
  # code -- and the group reaped BEFORE the verdict: a run is not finished
  # while a descendant dune left in its group still runs, holding lock fd 9
  # (every later launch refused) or, having closed it, mutating _build
  # under no lock at all. Only a group the child confirmed LEADING (its
  # pgid record) is reaped: where setpgrp failed there is no group of ours
  # to signal, only whatever process the number lands on next. The group
  # id is the pid of the reaped leader, which POSIX forbids reusing while
  # the group exists, so a reachable group is what dune left behind -- the
  # residual, stated: a group that emptied and whose id a fresh leader took
  # within this two-second grace. (An escapee that called setsid is beyond
  # this and every census; the lock it inherited is what stops the next
  # launch, and `stop` reaps it.)
  $finishing = 1;
  my $led = 0;
  if ($rd && open(my $pf, "<", "$rd/pgid")) {
    my $recorded = <$pf>;
    $led = 1 if defined $recorded && $recorded =~ /^\s*(\d+)\s*$/ && $1 == $pid;
  }
  if ($led && kill(0, -$pid)) {
    kill("TERM", -$pid);
    for (1 .. 20) {
      last unless kill(0, -$pid);
      select undef, undef, undef, 0.1;
    }
    kill("KILL", -$pid) if kill(0, -$pid);
  }
  $pid = 0;
  $finish->($code_of->($st));
'

# Take the per-worktree lock on fd 9, non-blocking. perl takes it and exits;
# the lock lives on the open file DESCRIPTION, which every process of the
# run inherits through fd 9 -- released by the kernel when the last holder
# closes: the supervisor's exit in the common case, otherwise the exit of the
# last descendant dune left behind, with nothing to reclaim after a crash
# (see sweep.sh). The launcher closes its own copy as soon as the supervisor
# has inherited it.
#
# Acquisition and the owner pointer's rewrite happen inside ONE process, so
# the pointer names THIS run from the instant the lock is held: there is no
# moment at which the lock is held without an owner. A stop of the previous
# run cannot blame its leftovers for our presence on the lock file and TERM
# this very launcher, and a launcher stalled anywhere after acquisition is
# reachable through `stop <run>` on the owner it published. That is why
# new_run creates the run directory BEFORE the lock is taken: the directory
# is the pointer's target. Exit codes: 1 lock busy, 2 pointer unwritable.
take_lock() {
  # Transitional (delete, with the `.gitignore` rule for these files, once
  # no run launched before gh-ocannl-606 can be in flight): a `start`/
  # `repeat` of the previous version holds the lock that version kept
  # beside the worktree, which this acquisition would not see -- and two
  # managed runs in one worktree is what the lock exists to refuse. So
  # where that file exists it is TAKEN, on fd 5, not merely probed: held
  # through this acquisition and inherited by the run like fd 9, it keeps
  # a launcher of the previous version refused for as long as this run
  # holds the new lock. Its owner pointer sits beside it, so a refusal can
  # still name the run. Once held, the residue that version left -- lock,
  # owner and launcher record -- is removed: unlocked, it belongs to no run,
  # and it is what made every worktree that ran a suite ignored-dirty.
  if [ -e "$PWD/.test-run.lock" ]; then
    exec 5>>"$PWD/.test-run.lock" || die "cannot open the previous version's lock file"
    if ! perl -e 'use Fcntl ":flock"; exit(flock(STDIN, LOCK_EX | LOCK_NB) ? 0 : 1)' <&5; then
      rm -rf "$run_dir"
      owner=$(cat "$PWD/.test-run.lock.owner" 2>/dev/null)
      owner=$(printf %q "${owner:-last}")
      echo "test-run: a test-run of the previous version is still active in this worktree" >&2
      echo "  (it holds $PWD/.test-run.lock); check it with: tools/test-run.sh status $owner" >&2
      echo "(a stale one can be stopped with: tools/test-run.sh stop $owner)" >&2
      exit 2
    fi
    rm -f "$PWD/.test-run.lock" "$PWD/.test-run.lock.owner" "$PWD/.test-run.lock.launcher"
  fi
  exec 9>>"$LOCK" || die "cannot open lock file $LOCK"
  # The pointer is written aside and renamed into place: a reader refused by
  # the lock in the same instant sees the previous owner or this run, never
  # a truncated or half-written path that would send it to `last` instead
  # of the actual holder.
  perl -e '
    use Fcntl ":flock";
    exit 1 unless flock(STDIN, LOCK_EX | LOCK_NB);
    my $tmp = "$ARGV[0].tmp.$$";
    open(my $fh, ">", $tmp) or exit 2;
    print $fh "$ARGV[1]\n" or exit 2;
    close $fh or exit 2;
    rename($tmp, $ARGV[0]) or do { unlink $tmp; exit 2 };
    exit 0;
  ' "$OWNER" "$run_dir" <&9
  case $? in
    0) ;;
    2)
      rm -rf "$run_dir"
      die "cannot record the lock owner pointer at $OWNER (state directory not writable?)"
      ;;
    *)
      # A run directory made for a launch that is refused was never
      # published; it goes, or `list` would show a dead run that never was.
      rm -rf "$run_dir"
      # The owner pointer, not `last`: `last` is the run published most
      # recently, and what the reader needs to inspect or stop is the lock's
      # HOLDER. %q, so a runs directory containing spaces or metacharacters
      # survives copy-pasting the recovery commands.
      owner=$(cat "$OWNER" 2>/dev/null)
      owner=$(printf %q "${owner:-last}")
      echo "test-run: another test-run is active in this worktree; check it with:" >&2
      echo "  tools/test-run.sh status $owner" >&2
      echo "(a stale one can be stopped with: tools/test-run.sh stop $owner)" >&2
      exit 2
      ;;
  esac
}

new_run() {
  run_dir=$RUNS/$(date -u +%Y%m%dT%H%M%SZ)-$$
  mkdir "$run_dir" || die "cannot create $run_dir"
  # Every pre-launch metadata write is checked: a quota hit or squatter that
  # slipped through here would let the launch report success while status,
  # wait and the supervisor all operate on a run that cannot be tracked.
  { { printf '%q ' "$@"; echo; } >"$run_dir/cmd" &&
    printf '%s\n' "$cap" >"$run_dir/cap" &&
    printf '%s\n' "$PWD" >"$run_dir/wt" &&
    printf '%s\n' "$RUNS" >"$run_dir/runs" &&
    : >"$run_dir/log"; } || die "cannot write run metadata in $run_dir"
  # `runs` is the state root this run's lock and pointers live under: `stop`
  # and retention read them from there, whichever OCANNL_TOOL_TEST_RUNS the
  # caller has -- and its absence marks a run of the version that kept them
  # beside the worktree (see lock_paths_of).
  # Runs are throwaway diagnostics; reap old ones so the directory cannot grow
  # without bound. Deletion demands the full run schema -- the timestamped
  # name AND this script's metadata files -- never mere position under $RUNS,
  # because the override can point $RUNS at a directory with other tenants
  # and "has a file named exit" must not mark those for deletion. Age is the
  # VERDICT file's, since a verdict-less directory may be a live run
  # (`--cap 0` is supported and unbounded); those are reaped only on a much
  # longer leash, as crash leftovers.
  # -mindepth 1 plus the explicit guard: find emits the starting directory
  # itself at depth 0, and an override pointing $RUNS at a directory that
  # happens to match the schema must not get the whole state root deleted.
  find "$RUNS" -mindepth 1 -maxdepth 1 -type d -name '2*Z-*' 2>/dev/null |
    while IFS= read -r d; do
      [ "$d" = "$RUNS" ] && continue
      # The find glob is only a pre-filter; deletion requires the EXACT
      # generated shape -- YYYYMMDDTHHMMSSZ-<pid> -- so a stray directory
      # like `2-oldZ-backup` cannot qualify even if it contains files with
      # the metadata names.
      b=$(basename "$d")
      case $b in
        [0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]T[0-9][0-9][0-9][0-9][0-9][0-9]Z-*) ;;
        *) continue ;;
      esac
      # NON-greedy strip: the prefix pattern pins the first `Z-` to position
      # 15, so this is the entire remainder of the basename -- a name like
      # `...Z-archiveZ-456` leaves `archiveZ-456` here and is rejected,
      # where a greedy strip would leave `456` and pass it.
      case ${b#*Z-} in '' | *[!0-9]*) continue ;; esac
      [ -f "$d/cmd" ] && [ -f "$d/cap" ] || continue
      if [ -f "$d/exit" ]; then
        if [ -n "$(find "$d/exit" -mtime +7 2>/dev/null)" ]; then
          # A completed run whose leftovers still hold ITS worktree's lock
          # keeps its metadata -- deleting it would strand that worktree
          # with the advertised status/stop recovery pointing at nothing.
          lock_still_owned "$d" && continue
          rm -rf "$d"
        fi
      else
        # A verdict-less directory may be a LIVE run (--cap 0 is supported and
        # unbounded, and $RUNS is shared across worktrees): never reap while
        # its owner still runs, and age it by the newest file INSIDE --
        # appending to `log` does not touch the directory's mtime.
        sup_alive "$d" && continue
        # Same held-lock guard as the completed branch: a crashed run's
        # descendant can hold its worktree lock without a verdict on record.
        lock_still_owned "$d" && continue
        [ -z "$(find "$d" -mtime -30 2>/dev/null | head -1)" ] && rm -rf "$d"
      fi
    done
}

# "Is the recorded supervisor still running?" -- answered with the pid AND a
# start-time token, because a bare `kill -0` latches onto whatever process
# recycled the pid after a reboot or a supervisor crash: `status`/`list` would
# report a stale run as active forever and `stop` would TERM an innocent
# process. An empty token (MSYS ps without lstart) degrades to the plain
# pid check rather than failing.
lock_held() { # <lock-file>; exits 0 iff some process holds its flock
  perl -e 'use Fcntl ":flock";
           open(my $fh, ">>", $ARGV[0]) or exit 1;
           exit(flock($fh, LOCK_EX | LOCK_NB) ? 1 : 0)' "$1" 2>/dev/null
}

# Non-writing lock snapshot shared by the public queries and idle. Missing
# locks are idle; inspection errors must never be mistaken for absence.
probe_locks() {
  perl -e '
      use Fcntl ":flock";
      use Errno qw(EWOULDBLOCK EAGAIN ENOENT);
      my @handles;
      for my $path (@ARGV) {
        unless (lstat $path) {
          next if $! == Errno::ENOENT;
          exit 2;
        }
        -f $path or exit 2;
        # Read/write access matches the existing lock on MSYS too, but this
        # probe never writes, creates, truncates, or unlinks a lock file.
        open my $fh, "+<", $path or exit 2;
        flock($fh, LOCK_EX | LOCK_NB)
          or exit(($! == EWOULDBLOCK || $! == EAGAIN) ? 3 : 2);
        push @handles, $fh;
      }
      exit 0;
  ' "$@"
}

# Is the RECORDED worktree of run-dir $1 still locked with $1 as the named
# owner? Then $1's leftovers are what is holding it -- grounds both for
# reaping them (stop) and for keeping $1's metadata alive (retention).
lock_still_owned() {
  lock_paths_of "$1" || return 1
  [ "$(cat "$run_owner" 2>/dev/null)" = "$1" ] && lock_held "$run_lock"
}
# Where run-dir $1's lock and owner pointer live: under the state root it
# recorded in `runs`, keyed by its worktree. A run with no `runs` on record
# was launched by the version that kept both BESIDE the worktree
# (`.test-run.lock`, `.test-run.lock.owner`); its leftovers, if any, hold
# THAT lock, and only those paths can attribute and reap them. Sets
# run_wt, run_lock, run_owner; fails when the run recorded no worktree.
lock_paths_of() {
  local r k
  run_wt=$(cat "$1/wt" 2>/dev/null) || return 1
  [ -n "$run_wt" ] || return 1
  r=$(cat "$1/runs" 2>/dev/null)
  if [ -n "$r" ]; then
    k=$(wt_key_of "$run_wt")
    run_lock=$r/lock-$k
    run_owner=$r/owner-$k
  else
    run_lock=$run_wt/.test-run.lock
    run_owner=$run_wt/.test-run.lock.owner
  fi
}

# One fixed rendering for start-time tokens: lstart is locale- AND
# timezone-formatted, so an unpinned rendering makes the same process print
# differently from a shell in another TZ -- classifying a live run as dead
# and letting stop refuse to reach it.
ps_token() {
  # Linux: starttime in clock ticks from /proc/<pid>/stat -- lstart's
  # one-second resolution cannot tell a pid recycled within the same second
  # from the original. Elsewhere: lstart under a pinned locale/TZ.
  if [ -r "/proc/$1/stat" ]; then
    perl -e '
      open my $fh, "<", "/proc/$ARGV[0]/stat" or exit 0;
      my $s = <$fh>;
      $s =~ /\)\s+(.*)$/ or exit 0;   # comm may contain spaces/parens
      my @f = split /\s+/, $1;        # $f[0] is field 3 (state)
      print defined $f[19] ? $f[19] : "";
    ' "$1" 2>/dev/null
  else
    # Squeezed AND trimmed: ps pads lstart to a fixed width, and tokens are
    # compared as strings against records written by other renderers (the
    # group leader trims its own) -- one canonical form for all writers.
    LC_ALL=C TZ=UTC ps -o lstart= -p "$1" 2>/dev/null |
      tr -s ' ' | sed 's/^ *//; s/ *$//'
  fi
}

proc_identity_matches() { # pid-file token-file -- zombies retain identity
  local pid tok now
  pid=$(cat "$1" 2>/dev/null) || return 1
  # A corrupted or forged pid file must never reach kill: 0 and negative
  # values are POSIX kill specials (caller's group / broadcast), so only a
  # positive decimal integer is a pid at all.
  case $pid in '' | *[!0-9]* | 0) return 1 ;; esac
  kill -0 "$pid" 2>/dev/null || return 1
  tok=$(tr -s ' ' <"$2" 2>/dev/null | sed 's/^ *//; s/ *$//') || tok=
  # No RECORDED identity (MSYS ps without lstart) degrades to the plain pid
  # check -- but where one was recorded, a pid that can no longer produce a
  # token (exited between kill -0 and ps_token, possibly recycled) must
  # fail, not pass.
  [ -z "$tok" ] && return 0
  now=$(ps_token "$pid")
  [ -n "$now" ] || return 1
  [ "$tok" = "$now" ]
}
proc_alive() { # pid-file token-file
  local pid
  proc_identity_matches "$1" "$2" || return 1
  pid=$(cat "$1" 2>/dev/null) || return 1
  # A zombie retains identity but will never publish anything; status/wait
  # must not treat it as live. Empty state (ps without the column) falls
  # through as the portable over-reporting fallback.
  case $(ps -o state= -p "$pid" 2>/dev/null | tr -d ' ') in Z*) return 1 ;; esac
  return 0
}
# The run's OWNER: the supervisor of a `run`/`start`, the coordinator of a
# `repeat` -- the process that publishes the verdict and that `stop` TERMs.
# Transitional: a run recorded by the previous version (no `runs`, see
# lock_paths_of) kept its owner -- the wrapper subshell, or the repeat
# coordinator, in `wpid`/`wtoken` -- apart from the supervisor in `pid`,
# and that owner outlives the supervisor while it publishes, or between
# repeat iterations; delete with the legacy paths.
sup_alive() { proc_alive "$1/pid" "$1/ptoken" || legacy_owner_alive "$1"; }
legacy_owner_alive() { [ ! -f "$1/runs" ] && proc_alive "$1/wpid" "$1/wtoken"; }
# Group signaling demands a RECORDED leader token: proc_alive's empty-token
# fallback exists for supervisor pids on platforms without lstart,
# and is too weak to aim a signal at a whole, possibly recycled, process
# group.
group_verified() { [ -s "$1/gtoken" ] && proc_alive "$1/pgid" "$1/gtoken"; }
group_identity_matches() {
  [ -s "$1/gtoken" ] && proc_identity_matches "$1/pgid" "$1/gtoken"
}

# "Is anything in this process group still RUNNING?" -- the group-scale twin of
# the zombie filter proc_alive applies per pid, and for the same reason: a
# zombie is a process-table entry, so `kill -0 -- -PGID` succeeds on a group
# whose every member has already exited, and group_verified does not rescue the
# check either -- a zombie leader still prints its recorded lstart. Reading such
# a group as alive makes `stop` announce "orphaned process group N ignored TERM"
# for a group holding nothing but corpses, which is exactly the report someone
# consults when working out why a worktree lock will not clear. Under an init
# that reaps, the corpse is transient and the bare probe merely lost a race with
# it; under one that does not -- the ordinary container case -- it is PERMANENT,
# so waiting it out was never the fix (gh-ocannl-742; the same misreading cost
# scripts/setup-ocaml-env.sh 30s on every failed ssh fetch; both callers now
# source the same `scripts/process-group.sh` ladder).
#
# States are not signal-visible, so they are read where the system publishes
# them: /proc on Linux (with the shell's own `read`, no fork per process), `ps`
# on the BSDs and macOS. Where neither answers -- a Cygwin `ps` takes no `-o`,
# and that shell is supported here -- it degrades to the signal alone, which
# over-reports; Cygwin reaps its own children, so the motivating case does not
# arise on that path.
#
# The signal probe stays as a NECESSARY condition rather than only a fallback:
# every caller follows this with a kill to the same group, and keeping it first
# makes this predicate a strict narrowing of the bare `kill -0` it replaces --
# it can turn a phantom alive into dead and never the reverse, whatever the
# state reader sees (a group of another uid, say, which `ps -A` lists and no
# kill of ours could reach).
#
# It remains a CENSUS, and a census is a SNAPSHOT: the glob (or the `ps`) is
# taken at one instant, so a child forked while it is being read is not in it,
# while a leader that exited into a zombie during that instant is. Both callers
# are therefore written so that this answer can shorten a reap or reword a
# report but never SKIP one -- the signals go out on reachability alone. A wrong
# answer then costs a less graceful shutdown, never a survivor left mutating
# _build behind a released worktree lock (Codex review round 1, P1).
# Make the launched run discoverable as `last`. The launcher calls this while
# it holds the lock and BEFORE the supervisor exists: publication and the
# run's completion are then ordered by construction -- no run can finish and
# release the lock under a pointer write still in flight, which is what the
# former three-party publication handshake existed to prevent (gh-ocannl-606).
# The price is a window of milliseconds in which `last` names a run whose
# supervisor has not yet recorded itself: `status` reports that state by
# name, and the launcher reports the launch only once the record exists.
publish_run() { # 0 published; 1 error
  # A PLAIN FILE holding the run directory's path, not a symlink: under MSYS
  # (Git Bash) with the default `winsymlinks` mode, `ln -s` does not create a
  # link at all -- it silently COPIES, so a directory target left a full copy
  # of the run directory at `last-<key>` and every `last` lookup afterwards
  # resolved to nothing. Native symlinks there need a privilege the shell may
  # not hold, so the portable pointer is the file. Rename-based, hence still
  # atomic: a reader sees the old path or the new one, never a partial write.
  #
  # Anything at the pointer path that is neither a regular file nor a
  # symlink left by an earlier version is a squatter -- notably the
  # DIRECTORY the copying `ln -s` used to leave behind. (`-f` follows, so a
  # symlink to a run directory needs the explicit `-L` arm.)
  { [ ! -e "$LAST" ] || [ -f "$LAST" ] || [ -L "$LAST" ]; } || {
    echo "test-run: $LAST exists and is not a regular file; remove it" >&2
    return 1
  }
  # Written through a temporary and put in place with rename(2), which
  # REPLACES whatever the path names -- a previous pointer, or a symlink one
  # written before this script stopped using symlinks -- in a single atomic
  # step, then re-read to prove the pointer names this run before the launch
  # reports success. Neither of the obvious spellings works here: `mv` stats
  # the destination THROUGH the link, so a symlink to a run directory makes
  # it move the pointer INSIDE that directory; and unlinking first would
  # leave `last` absent for an interval, which is the very gap the
  # old-or-new guarantee exists to rule out (a launcher killed inside it
  # would strand a recorded, possibly live, run with no `last` at all).
  # rename(2) also refuses a directory destination outright, so the squatter
  # above cannot be absorbed even if the guard is somehow raced.
  { printf '%s\n' "$run_dir" >"$LAST.tmp.$$" &&
    perl -e 'rename($ARGV[0], $ARGV[1]) or exit 1' "$LAST.tmp.$$" "$LAST" &&
    [ "$(cat "$LAST" 2>/dev/null)" = "$run_dir" ]; } || {
    rm -f "$LAST.tmp.$$"
    echo "test-run: cannot update $LAST" >&2
    return 1
  }
}

resolve_run() {
  local ref=${1:-last}
  if [ "$ref" = last ]; then
    # The pointer is a plain file (see publish_run). A symlink there was
    # written by a version predating that change and may still name a run
    # that is LIVE -- `stop last` has to be able to reach one, and only a
    # later publication replaces the link -- so both spellings are read.
    #
    # Deliberately NOT a `-L` test selecting the matching read: publish_run's
    # rename can replace the symlink with the pointer file between the test
    # and the read, and a reader must not fail on a path that named a valid
    # run throughout. Trying the reads themselves has no such gap, in THIS
    # order: a symlink can only ever become a file (publish_run writes
    # nothing else), so a failed `readlink` means `cat` now applies. The
    # reverse order would still race -- `cat` fails on a symlink to a
    # directory, and the rename could land before the `readlink` retry.
    run_dir=$(readlink "$LAST" 2>/dev/null) || run_dir=
    [ -n "$run_dir" ] ||
      { run_dir=$(cat "$LAST" 2>/dev/null) || run_dir=; }
    [ -n "$run_dir" ] || die "no runs recorded for this worktree"
    [ -d "$run_dir" ] || die "no such run: $run_dir"
  elif [ -d "$ref" ]; then
    # Canonicalized (physically -- symlink spellings differ per referrer)
    # for the same reason as $RUNS: identity is compared as a string
    # against recorded pointers.
    run_dir=$(cd "$ref" && pwd -P) || die "cannot resolve $ref"
  elif [ -d "$RUNS/$ref" ]; then
    # The bare identifiers `list` prints resolve here.
    run_dir=$RUNS/$ref
  else
    die "no such run: $ref (neither a directory nor an entry under $RUNS)"
  fi
  # Only directories this runner created may be trusted: without this, an
  # arbitrary directory's stray `pid` file (with the missing token read as
  # the documented fallback) could get an unrelated process TERMed by stop.
  [ -f "$run_dir/cmd" ] && [ -f "$run_dir/cap" ] ||
    die "not a test-run directory (no cmd/cap metadata): $run_dir"
}

# A dune invocation dune's own command-line parser refused -- an unknown option
# or subcommand, a missing or malformed operand -- is a stable two-part shape
# on stderr: `dune: <complaint>` (continued onto indented lines when the parser
# wraps it) followed by `Usage: dune ...`, then exit 1 with nothing built or
# run. Neither an `Error:` nor a `File "..."` line ever accompanies it, so the
# fingerprint has nothing to quote and the digest used to fall through to
# `FAIL (exit 1)` plus a raw log tail -- the verdict of a red suite, read at
# the moment the reader decides between "read the failures" and "fix the
# command line" (gh-ocannl-944). The shape is required WHOLE, FIRST and ALONE:
# a `dune:` line alone could be a test's own output; dune's usage errors are
# emitted before anything else is; and a refusal that ran nothing leaves
# nothing after itself -- past `Usage:` only dune's own `Try '... --help'`
# line, blank lines and this script's `exit: N` sentinel may follow. Anything
# else after it is evidence that something DID run (a `dune exec` program
# printing a nested dune usage error and then failing, say), and the log is
# then an ordinary red run whose fingerprint the digest must show. Prints the
# complaint (the lines before `Usage:`), 0 iff FILE is such a refusal.
dune_refusal() { # FILE
  head -c 20000 "$1" 2>/dev/null | awk '
    NR == 1 && $0 !~ /^dune: / { exit 1 }
    !found && /^Usage: dune/ { found = 1; next }
    !found && NR > 20 { exit 1 }
    !found { print; next }
    /^Try \047.*--help/ || /^exit: [0-9]+$/ || /^$/ { next }
    { found = 0; exit 1 }
    END { exit found ? 0 : 1 }'
}

# The compact report `run`, `wait` and `status` all end with. Fingerprint in
# the sweep.sh sense: the `File "..."` and `Error ...` lines, deduplicated, so
# a new failure is distinguishable from a standing one without opening the log.
# Both of dune's location spellings, since a stanza-level diagnostic -- what a
# failing explicit-rule test produces -- says `lines N-M` and would otherwise
# fall through to the raw log tail. The line numbers are kept as printed: this
# report is read against the tree that produced it, unlike sweep.sh's, which is
# diffed across commits and so normalizes a dune location to its stanza.
# Sets `digest_rc`, the status `run` and `wait` exit with: the recorded status,
# except 2 for a refused invocation (see the header's exit-code contract).
digest_rc=
digest() {
  local dir=$1 rc verdict fp complaint= refusal_src
  rc=$(cat "$dir/exit" 2>/dev/null) || die "no verdict recorded in $dir"
  digest_rc=$rc
  # Where dune's stderr opens: the run log for `run`/`start`; for a repeat the
  # log opens with the iteration banner, and the refusal (if any) is the first
  # iteration's own stderr, which is also the only iteration a refusal leaves.
  refusal_src=$dir/log
  [ "$(cat "$dir/mode" 2>/dev/null)" = repeat ] && refusal_src=$dir/iteration-1/stderr
  # 142 is the ONLY code the cap produces (the supervisor's SIGALRM exit), so
  # only it may say "timeout". 137 is a SIGKILL -- an OOM kill or a forced
  # external kill -- and labeling it a timeout would send triage hunting a
  # hang that never happened. And 1 is the only status dune's parser exits
  # with, so only it is examined for a refusal: a refused-looking log under
  # any other code still reports that code's verdict.
  case $rc in
    0) verdict=pass ;;
    1)
      if complaint=$(dune_refusal "$refusal_src"); then
        verdict="INVOCATION REFUSED (dune rejected the arguments; nothing ran)"
        digest_rc=2
      else
        verdict=FAIL
      fi
      ;;
    142) verdict="TIMEOUT (cap expired; run was killed, not judged)" ;;
    137) verdict="KILLED (SIGKILL: OOM or forced kill; not judged)" ;;
    129 | 130 | 143) verdict="CANCELLED (run was killed, not judged)" ;;
    126 | 127) verdict="ERROR (toolchain/setup: nothing ran)" ;;
    *) verdict=FAIL ;;
  esac
  echo "command: dune $(cat "$dir/cmd")"
  echo "verdict: $verdict (exit $digest_rc)"
  echo "log:     $dir/log"
  if [ "$digest_rc" = 2 ]; then
    # dune's own words name the fix; nothing below (promotion diffs, the
    # fingerprint, a log tail) could apply to a run in which no rule ran.
    echo "dune said:"
    printf '%s\n' "$complaint" | sed 's/^/  /'
    echo "fix the command line and run again -- no test was built or judged" \
         "(dune's own status was $rc; this script exits 2, its usage code)"
    return 0
  fi
  # Digest sits on `wait`'s deadline path, so it examines at most the last
  # 10MB of the log rather than scaling with an arbitrarily noisy run.
  scan_log() { tail -c 10000000 "$dir/log" 2>/dev/null; }
  if scan_log | grep -qE '^File "[^"]*\.expected"|\.corrected'; then
    echo "promotion diffs present -- inspect the log, accept with \`dune promote\`" \
         "(tools/promote.sh on Windows)"
  fi
  if [ "$rc" != 0 ]; then
    fp=$({ scan_log | grep -oE '^File "[^"]+", lines? [0-9]+(-[0-9]+)?'
           scan_log | grep -oE '^(Error|Fatal error|Exception)[^,]*'
         } 2>/dev/null | sort -u | head -40)
    if [ -n "$fp" ]; then
      echo "fingerprint:"
      printf '%s\n' "$fp" | sed 's/^/  /'
    else
      echo "no Error/File lines matched; log tail:"
      # Byte-capped BEFORE the line cap: a giant newline-free record would
      # otherwise ride through `tail -n` whole.
      tail -c 4000 "$dir/log" | tail -n 25 | sed 's/^/  /'
    fi
  fi
}

finish_run() { # rc -> append sentinel, record verdict
  printf 'exit: %s\n' "$1" >>"$run_dir/log"
  # Written aside and renamed: the verdict file's EXISTENCE is the completion
  # signal `status`/`wait` key on, so it must never be observable empty. The
  # write is retried with backoff -- a filesystem that filled up during the
  # run may clear, and giving up silently would downgrade a real verdict
  # into a generic "died without recording a verdict".
  local n=0
  until printf '%s\n' "$1" >"$run_dir/exit.tmp" &&
        mv -f "$run_dir/exit.tmp" "$run_dir/exit"; do
    n=$(( n + 1 ))
    if [ "$n" -ge 3 ]; then
      printf 'test-run: FAILED to record verdict %s (filesystem?)\n' "$1" \
        >>"$run_dir/log" 2>/dev/null
      return 1
    fi
    sleep 2
  done
}

sub=${1:-}
[ -n "$sub" ] || die "usage: tools/test-run.sh run|start|repeat|status|wait|stop|list|idle|paths|lock-status ... (see header)"
shift

case $sub in
  paths)
    [ $# -ge 1 ] && [ $# -le 2 ] || die "usage: paths FIELD [RUN|last]"
    field=$1
    case $field in run | worktree | runs | lock | owner | last) ;;
      *) die "unknown path field: $field (run, worktree, runs, lock, owner, last)" ;;
    esac
    if [ "$field" = run ]; then
      resolve_run "${2:-last}"
      # Legacy last symlinks may spell a run through a symlinked directory.
      (cd "$run_dir" && pwd -P) || die "cannot resolve $run_dir"
      exit 0
    fi
    query_wt=$PWD query_runs=$RUNS query_lock=$LOCK query_owner=$OWNER query_last=$LAST
    if [ $# -eq 2 ]; then
      resolve_run "$2"
      lock_paths_of "$run_dir" || die "run has no recorded worktree: $run_dir"
      query_wt=$run_wt query_lock=$run_lock query_owner=$run_owner
      query_runs=$(cat "$run_dir/runs" 2>/dev/null) || query_runs=
      # Legacy runs have no stored state root; the run directory identifies it.
      [ -n "$query_runs" ] || query_runs=$(cd "$run_dir/.." && pwd -P)
      query_last=$query_runs/last-$(wt_key_of "$query_wt")
    fi
    case $field in
      worktree) printf '%s\n' "$query_wt" ;;
      runs) printf '%s\n' "$query_runs" ;;
      lock) printf '%s\n' "$query_lock" ;;
      owner) printf '%s\n' "$query_owner" ;;
      last) printf '%s\n' "$query_last" ;;
    esac
    ;;
  lock-status)
    [ $# -le 1 ] || die "usage: lock-status [RUN|last]"
    if [ $# -eq 1 ]; then
      resolve_run "$1"
      lock_paths_of "$run_dir" || die "run has no recorded worktree: $run_dir"
      probe_locks "$run_lock"
    else
      probe_locks "$LOCK" "$PWD/.test-run.lock"
    fi
    query_rc=$?
    case $query_rc in
      0) echo idle ;;
      3) echo held ;;
      *) die "cannot inspect worktree lock" ;;
    esac
    exit "$query_rc"
    ;;
  repeat)
    cap=${OCANNL_TOOL_TEST_CAP:-3600}
    alone=0
    while [ $# -gt 0 ]; do
      case $1 in
        --cap) [ $# -ge 2 ] || die "--cap requires a value"; cap=$2; shift 2 ;;
        --alone) alone=1; shift ;;
        --) shift; break ;;
        *) break ;;
      esac
    done
    normalize_cap
    [ $# -gt 0 ] || die "repeat requires a count of at least 2"
    repeats=$1
    shift
    case $repeats in '' | *[!0-9]*) die "repeat count must be an integer of at least 2" ;; esac
    [ ${#repeats} -le 6 ] || die "repeat count too large (max 6 digits)"
    repeats=$(( 10#$repeats ))
    [ "$repeats" -ge 2 ] || die "repeat count must be at least 2"
    reject_misplaced_options "$@"
    [ $# -gt 0 ] || set -- runtest
    select_dune
    new_run "$@"
    take_lock
    # The coordinator is the run's OWNER: its identity is what status reads
    # as running and what stop TERMs -- the set-wide cancellation bit lives
    # here, not in any one iteration's supervisor, which the coordinator
    # records under the iteration instead.
    { printf '%s\n' repeat >"$run_dir/mode" &&
      printf '%s\n' "$repeats" >"$run_dir/repeats" &&
      printf '%s\n' "$alone" >"$run_dir/alone" &&
      ps_token "$$" >"$run_dir/ptoken" &&
      printf '%s\n' "$$" >"$run_dir/pid"; } ||
      die "cannot record repeat metadata in $run_dir"
    repeat_sup=
    repeat_cancelled=
    completed=0
    first_nonzero=0
    repeat_refused=
    repeat_signal() {
      repeat_cancelled=$1
      [ -n "$repeat_sup" ] && kill "-$1" "$repeat_sup" 2>/dev/null
    }
    trap 'repeat_signal INT' INT
    trap 'repeat_signal TERM' TERM
    trap 'repeat_signal TERM' HUP
    # Once published, every coordinator exit must publish a verdict too. This
    # is also the group-cancellation backstop: a foreground finalizer child
    # (diff/cmp/remove_tree) shares the terminal's process group and can die
    # from the same signal even though the coordinator traps it. If that makes
    # a checked finalizer command call die, EXIT still records cancellation.
    repeat_exit() {
      local rc=$?
      trap - EXIT
      # Verdict publication itself is the final, tiny critical section. Ignore
      # another terminal/group signal here so finish_run's children inherit
      # that disposition and the atomic exit file cannot be interrupted.
      trap '' INT TERM HUP
      if [ -n "$repeat_cancelled" ]; then
        if [ "$first_nonzero" != 0 ]; then
          rc=$first_nonzero
        else
          case $repeat_cancelled in INT) rc=130 ;; *) rc=143 ;; esac
        fi
        if [ -z "${repeat_reported_cancelled:-}" ]; then
          printf 'repeat result: CANCELLED -- completed %s of %s iterations\n' \
            "$completed" "$repeats"
          printf 'repeat result: CANCELLED -- completed %s of %s iterations\n' \
            "$completed" "$repeats" >>"$run_dir/log"
        fi
        printf 'repeat cancellation observed: %s\n' "$repeat_cancelled" >>"$run_dir/log"
      fi
      if [ ! -f "$run_dir/exit" ]; then
        finish_run "$rc" ||
          printf 'test-run: repeat exited %s but its verdict could not be recorded\n' "$rc" >&2
      fi
      exec 9>&-
      # Same contract as `run`: the RECORDED status is dune's own, the process
      # status of a refused invocation is the usage code (header).
      [ -n "$repeat_cancelled" ] || [ -z "$repeat_refused" ] || rc=2
      exit "$rc"
    }
    trap repeat_exit EXIT

    reap_repeat_group() { # iteration dir -- no verified survivor may share repeat_build
      local iter_dir=$1 pg n
      pg=$(cat "$iter_dir/pgid" 2>/dev/null) || return 0
      # Reachability gates cleanup. group_alive is a fallible census and may
      # miss a member that forks/exits while /proc is enumerated; it must never
      # turn a surviving group into permission to reuse repeat_build.
      kill -0 -- "-$pg" 2>/dev/null || return 0
      group_identity_matches "$iter_dir" ||
        die "iteration group $pg survived without a verifiable identity; refusing to reuse $repeat_build"
      printf 'repeat: iteration group %s survived its supervisor; reaping before reuse\n' "$pg" \
        | tee -a "$run_dir/log"
      kill -TERM -- "-$pg" 2>/dev/null
      for n in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
        kill -0 -- "-$pg" 2>/dev/null || return 0
        sleep 0.1
      done
      # Revalidate immediately before the destructive escalation. A zombie
      # leader still has the original token and safely identifies its group;
      # a missing leader while the group remains reachable can also mean a
      # live descendant survived TERM, so it is not permission to reuse.
      if ! group_identity_matches "$iter_dir"; then
        # Do not ask the fallible process census to authorize reuse here. A
        # descendant can fork and exit while /proc is being enumerated, making
        # a still-live chain momentarily look zombie-only. Without the recorded
        # leader we can neither safely aim KILL nor prove the build tree inert.
        die "leaderless iteration group $pg survived TERM; refusing to reuse $repeat_build"
      fi
      kill -KILL -- "-$pg" 2>/dev/null
      for n in 1 2 3 4 5 6 7 8 9 10; do
        kill -0 -- "-$pg" 2>/dev/null || return 0
        if ! group_identity_matches "$iter_dir"; then
          # KILL already reached the verified original group, so a live
          # leaderless member is now an unkillable survivor; only a zombie-only
          # residue is inert enough to release the shared build tree.
          group_alive "$pg" &&
            die "leaderless iteration group $pg survived KILL; refusing to reuse $repeat_build"
          return 0
        fi
        sleep 0.1
      done
      # KILL has already reached the identity-verified original group, so no
      # live member can subsequently fork. A remaining verified group whose
      # census is now zombie-only is inert residue under a non-reaping init,
      # not a reason to discard the iteration's real verdict.
      group_alive "$pg" || return 0
      die "iteration group $pg survived TERM/KILL; refusing to reuse $repeat_build"
    }

    # A repeat can be long enough to manage from another shell. Its complete
    # cancellation state and exit finalizer are armed BEFORE it becomes `last`:
    # every externally discoverable coordinator can therefore publish a
    # verdict even if stop or a group signal lands in the publication gap.
    publish_run || die "cannot publish repeat run $run_dir"

    repeat_build=$run_dir/build
    i=1
    while [ "$i" -le "$repeats" ] && [ -z "$repeat_cancelled" ]; do
      iter=$run_dir/iteration-$i
      mkdir "$iter" || die "cannot create $iter"
      # Dune's first `--` ends Dune option parsing (`dune exec PROG -- ARGS`).
      # Isolation options belong immediately before it; appending them would
      # silently hand them to PROG and run Dune in its ambient build context.
      repeat_cmd=("$DUNE")
      repeat_separator=0
      for repeat_arg in "$@"; do
        if [ "$repeat_separator" = 0 ] && [ "$repeat_arg" = -- ]; then
          repeat_cmd+=(--force --cache=disabled --build-dir="$repeat_build")
          [ "$alone" = 0 ] || repeat_cmd+=(-j 1)
          repeat_separator=1
        fi
        repeat_cmd+=("$repeat_arg")
      done
      if [ "$repeat_separator" = 0 ]; then
        repeat_cmd+=(--force --cache=disabled --build-dir="$repeat_build")
        [ "$alone" = 0 ] || repeat_cmd+=(-j 1)
      fi
      {
        printf '%q clean --build-dir=%q && ' "$DUNE" "$repeat_build"
        printf '%q ' "${repeat_cmd[@]}"
        echo
      } >"$iter/cmd" ||
        die "cannot record iteration $i command"
      printf 'repeat: iteration %s/%s -- dune%s\n' "$i" "$repeats" \
        "$([ "$alone" = 1 ] && printf ' (alone, -j 1)' || :)"
      # A stop can land after the loop condition. Refuse a launch already
      # known to be cancelled, then recheck immediately after starting the
      # supervisor to close the unavoidable signal-between-commands window.
      [ -z "$repeat_cancelled" ] || break
      # A process-group reap cannot see a Dune action that calls setsid. Give
      # every descendant an inherited FIFO writer as a second containment
      # witness: after the supervisor exits, EOF proves no session-escaped
      # descendant remains able to mutate this iteration's shared build tree.
      # The read/write anchor makes the two one-way opens nonblocking; it is
      # closed before launch and never reaches the child.
      descendant_fifo=$iter/descendants.fifo
      mkfifo "$descendant_fifo" || die "cannot create descendant witness for iteration $i"
      exec 7<>"$descendant_fifo" || die "cannot anchor descendant witness for iteration $i"
      exec 8<"$descendant_fifo" || die "cannot read descendant witness for iteration $i"
      exec 6>"$descendant_fifo" || die "cannot write descendant witness for iteration $i"
      exec 7>&-
      OCANNL_TOOL_TESTRUN_BG=0 OCANNL_TOOL_TESTRUN_RD=$iter \
        perl -e "$supervisor_perl" -- "$cap" /bin/bash -c \
        'dune=$1; build=$2; shift 2; "$dune" clean --build-dir="$build" || exit 126; exec "$dune" "$@"' \
        -- "$DUNE" "$repeat_build" "${repeat_cmd[@]:1}" \
        >"$iter/stdout" 2>"$iter/stderr" &
      repeat_sup=$!
      exec 6>&-
      [ -z "$repeat_cancelled" ] || kill "-$repeat_cancelled" "$repeat_sup" 2>/dev/null
      { printf '%s\n' "$repeat_sup" >"$iter/pid" &&
        ps_token "$repeat_sup" >"$iter/ptoken"; } ||
        kill -TERM "$repeat_sup" 2>/dev/null
      while :; do
        wait "$repeat_sup"
        iter_rc=$?
        proc_alive "$iter/pid" "$iter/ptoken" || break
      done
      # The numeric pid is no longer ours once wait/proc_alive prove the
      # supervisor gone. Clear it before the slower orphan-group reap so a
      # cancellation cannot signal a recycled, unrelated process.
      repeat_sup=
      # SIGKILL can remove the supervisor without reaching the Dune process
      # group it owned. Reap that identity-verified group before recording the
      # iteration, starting another one, or deleting their shared build tree.
      reap_repeat_group "$iter"
      # A zero-byte read is EOF. Bash 3.2 returns status 1 for BOTH EOF and a
      # `read -t` timeout, so use the Perl already required by this harness to
      # distinguish them: 0 is EOF, 2 is timeout, 1 is data/error. Anything
      # but EOF retains the build context and refuses concurrent reuse.
      perl -e '
        $SIG{ALRM} = sub { exit 2 };
        alarm 2;
        my $n = sysread(STDIN, my $byte, 1);
        exit(defined($n) && $n == 0 ? 0 : 1);
      ' <&8
      descendant_rc=$?
      exec 8>&-
      [ "$descendant_rc" = 0 ] ||
        die "iteration $i left a session-escaped descendant; refusing to reuse $repeat_build"
      rm -f "$descendant_fifo"
      printf '%s\n' "$iter_rc" >"$iter/exit" || die "cannot record iteration $i verdict"
      [ "$first_nonzero" != 0 ] || [ "$iter_rc" = 0 ] || first_nonzero=$iter_rc
      {
        printf '=== repeat iteration %s/%s stdout ===\n' "$i" "$repeats"
        cat "$iter/stdout"
        printf '=== repeat iteration %s/%s stderr ===\n' "$i" "$repeats"
        cat "$iter/stderr"
        printf '=== repeat iteration %s/%s exit %s ===\n' "$i" "$repeats" "$iter_rc"
      } >>"$run_dir/log" || die "cannot append iteration $i to $run_dir/log"
      printf 'repeat: iteration %s/%s exit %s; stdout=%s stderr=%s\n' \
        "$i" "$repeats" "$iter_rc" "$iter/stdout" "$iter/stderr"
      completed=$i
      # An invocation dune's parser refused ran nothing, so there is nothing
      # whose stability the remaining iterations could measure: stop after the
      # first, and let the verdict below say so rather than reporting N
      # identical refusals as IDENTICAL. Only the first iteration can decide
      # this -- every iteration runs the same argv.
      if [ "$i" = 1 ] && [ "$iter_rc" = 1 ] && dune_refusal "$iter/stderr" >/dev/null; then
        repeat_refused=1
        break
      fi
      [ -z "$repeat_cancelled" ] || break
      i=$(( i + 1 ))
    done
    # The isolated build tree can be very large (a training target pulls most
    # of the library graph). Only the diagnostic streams and pairwise diffs are
    # promised artifacts, so remove this exact generated child before keeping
    # the run directory for seven days.
    [ "$repeat_build" = "$run_dir/build" ] || die "refusing an unexpected repeat build path"
    perl -MFile::Path=remove_tree -e 'remove_tree($ARGV[0])' "$repeat_build" ||
      die "cannot remove repeat build context $repeat_build"
    [ ! -e "$repeat_build" ] || die "repeat build context survived cleanup: $repeat_build"

    mkdir "$run_dir/diffs" || die "cannot create $run_dir/diffs"
    differing=0
    stderr_only=0
    identical=0
    i=1
    while [ "$i" -lt "$completed" ]; do
      j=$(( i + 1 ))
      while [ "$j" -le "$completed" ]; do
        left=$run_dir/iteration-$i
        right=$run_dir/iteration-$j
        pair=$i-$j
        stdout_same=0; stderr_same=0; exit_same=0
        cmp -s "$left/stdout" "$right/stdout" && stdout_same=1
        cmp -s "$left/stderr" "$right/stderr" && stderr_same=1
        cmp -s "$left/exit" "$right/exit" && exit_same=1
        if [ "$stdout_same" = 1 ] && [ "$stderr_same" = 1 ] && [ "$exit_same" = 1 ]; then
          pair_result=identical
          identical=$(( identical + 1 ))
        elif [ "$stdout_same" = 1 ] && [ "$exit_same" = 1 ]; then
          pair_result=stderr-only
          stderr_only=$(( stderr_only + 1 ))
          diff -u "$left/stderr" "$right/stderr" >"$run_dir/diffs/$pair.stderr" ||
            [ $? -eq 1 ] || die "cannot diff stderr for iterations $i and $j"
        else
          pair_result=differing
          differing=$(( differing + 1 ))
          if [ "$stdout_same" = 0 ]; then
            diff -u "$left/stdout" "$right/stdout" >"$run_dir/diffs/$pair.stdout" ||
              [ $? -eq 1 ] || die "cannot diff stdout for iterations $i and $j"
          fi
          if [ "$stderr_same" = 0 ]; then
            diff -u "$left/stderr" "$right/stderr" >"$run_dir/diffs/$pair.stderr" ||
              [ $? -eq 1 ] || die "cannot diff stderr for iterations $i and $j"
          fi
          if [ "$exit_same" = 0 ]; then
            diff -u "$left/exit" "$right/exit" >"$run_dir/diffs/$pair.exit" ||
              [ $? -eq 1 ] || die "cannot diff exits for iterations $i and $j"
          fi
        fi
        printf 'repeat: pair %s/%s: %s\n' "$i" "$j" "$pair_result" |
          tee -a "$run_dir/log"
        j=$(( j + 1 ))
      done
      i=$(( i + 1 ))
    done

    if [ -n "$repeat_cancelled" ]; then
      repeat_result="CANCELLED -- completed $completed of $repeats iterations"
      repeat_reported_cancelled=1
      final_rc=$first_nonzero
      [ "$final_rc" != 0 ] || final_rc=143
    elif [ -n "$repeat_refused" ]; then
      repeat_result="INVOCATION REFUSED -- dune rejected the arguments; nothing ran, so nothing was repeated"
      final_rc=$first_nonzero
    elif [ "$differing" -gt 0 ]; then
      repeat_result="DIFFERING -- stdout or exit status moved across $differing pair(s)"
      final_rc=$first_nonzero
      [ "$final_rc" != 0 ] || final_rc=1
    elif [ "$stderr_only" -gt 0 ]; then
      repeat_result="STDERR-ONLY -- stdout and exit status were stable; stderr moved across $stderr_only pair(s)"
      final_rc=$first_nonzero
    else
      repeat_result="IDENTICAL -- stdout, stderr and exit status matched across all $identical pair(s)"
      final_rc=$first_nonzero
    fi
    printf 'repeat result: %s\n' "$repeat_result" | tee -a "$run_dir/log"
    if [ -n "$repeat_refused" ]; then
      { echo "dune said:"
        dune_refusal "$run_dir/iteration-1/stderr" | sed 's/^/  /'
        echo "fix the command line and run again -- no test was built or judged" \
             "(dune's own status was $final_rc; this script exits 2, its usage code)"
      } | tee -a "$run_dir/log"
    fi
    printf 'repeat artifacts: %s (pairwise diffs under %s/diffs)\n' "$run_dir" "$run_dir" |
      tee -a "$run_dir/log"
    # repeat_exit keeps cancellation deferred through cleanup/comparison and
    # publishes the atomic verdict while signals are ignored.
    exit "$final_rc"
    ;;
  run | start)
    cap=${OCANNL_TOOL_TEST_CAP:-3600}
    while [ $# -gt 0 ]; do
      case $1 in
        --cap) [ $# -ge 2 ] || die "--cap requires a value"; cap=$2; shift 2 ;;
        --) shift; break ;;
        *) break ;;
      esac
    done
    normalize_cap
    reject_misplaced_options "$@"
    [ $# -gt 0 ] || set -- runtest
    # Toolchain checks gate only launches: status/wait/stop/list remain usable
    # from a shell whose opam environment is no longer active.
    select_dune
    # Cancellation is armed BEFORE the lock is taken, for BOTH modes: from
    # here on the launcher holds state a signal must not abandon halfway (the
    # lock, then a published run). For `run` the signal is forwarded to the
    # supervisor -- at once when there is one, and right after the launch for
    # one that arrived before; for `start` it is merely deferred past the
    # launch: the launcher is about to exit anyway, and the run is MEANT to
    # survive it.
    cancelled= sup=
    forward_cancel() {
      [ "$sub" = run ] && [ -n "$sup" ] || return 0
      # The supervisor is signalled only while it can be identified: as this
      # shell's own unreaped child before it has recorded its identity, and
      # under its recorded start token afterwards -- never by a bare pid that
      # a process may have recycled after the reaping wait.
      { [ ! -f "$run_dir/pid" ] || sup_alive "$run_dir"; } || return 0
      # Per-signal, so an interrupt records 130 and not a generic 143: INT is
      # relayed as INT (the supervisor maps it to 130); HUP relays as TERM
      # because the detached-mode supervisor deliberately ignores HUP.
      case $cancelled in
        INT) kill -INT "$sup" 2>/dev/null ;;
        *) kill -TERM "$sup" 2>/dev/null ;;
      esac
    }
    fwd_sig() { cancelled=$1; forward_cancel; }
    trap 'fwd_sig INT' INT
    trap 'fwd_sig TERM' TERM
    trap 'fwd_sig HUP' HUP
    new_run "$@"
    take_lock
    publish_run || { rm -rf "$run_dir"; die "cannot publish $run_dir"; }
    # The supervisor inherits lock fd 9 and owns the run from here: it records
    # its identity, runs dune under the cap, and publishes the verdict (see
    # supervisor_perl). Nothing about the fate of THIS shell -- HUP from a
    # closed terminal, harness cancellation, a plain kill -- can lose the
    # verdict; `run` differs from `start` only in staying attached to wait
    # and digest.
    OCANNL_TOOL_TESTRUN_BG=1 OCANNL_TOOL_TESTRUN_RD=$run_dir OCANNL_TOOL_TESTRUN_OWN=$run_dir \
      perl -e "$supervisor_perl" -- "$cap" "$DUNE" "$@" </dev/null >>"$run_dir/log" 2>&1 &
    sup=$!
    # The launcher's own fd 9 copy served its purpose the moment the
    # supervisor inherited the lock's description: close it, so an attached
    # launcher never appears in a leftover census -- a concurrent stop must
    # not reap the process that is about to deliver the digest.
    exec 9>&-
    # A signal flagged before the supervisor existed is converted now.
    [ -z "$cancelled" ] || forward_cancel
    # The launch is reported only once the supervisor has recorded itself:
    # `wait last` from the very next command must find a live owner, not a
    # published run with no supervisor. Bounded. A supervisor that died first,
    # or gave up (it records a verdict for that, and dune never ran), ends
    # the wait at once; one wedged before its first write is reported after
    # a minute, with the run left to it.
    i=0
    while [ ! -s "$run_dir/pid" ] && [ ! -f "$run_dir/exit" ]; do
      kill -0 "$sup" 2>/dev/null || break
      case $(ps -o state= -p "$sup" 2>/dev/null | tr -d ' ') in Z*) break ;; esac
      i=$(( i + 1 ))
      if [ "$i" -gt 600 ]; then
        echo "test-run: the supervisor (pid $sup) has not recorded itself after 60s;" >&2
        echo "  the run is left to it: $run_dir (log: $run_dir/log)" >&2
        exit 2
      fi
      sleep 0.1
    done
    if [ ! -s "$run_dir/pid" ]; then
      # Gone, or given up, before recording itself: its verdict, if it left
      # one, says why (nothing ran); otherwise only its log can.
      wait "$sup" 2>/dev/null
      sup=
      trap - INT TERM HUP
      if [ -f "$run_dir/exit" ]; then
        digest "$run_dir"
        exit "$digest_rc"
      fi
      echo "run died before its supervisor recorded itself: $run_dir (log: $run_dir/log)"
      exit 1
    fi
    if [ "$sub" = run ]; then
      # Attached: wait for the supervisor -- its exit means the verdict file
      # is on disk. A trapped signal returns from `wait` early, hence the
      # retry loop; sup_alive rather than a bare kill -0, since after the
      # supervisor is reaped its pid can be recycled and probing the number
      # alone would spin this loop on an unrelated process.
      while sup_alive "$run_dir"; do wait "$sup" 2>/dev/null; done
      sup=
      trap - INT TERM HUP
      if [ -f "$run_dir/exit" ]; then
        digest "$run_dir"
        exit "$digest_rc"
      fi
      echo "run died without recording a verdict (supervisor killed?): $run_dir"
      exit 1
    fi
    trap - INT TERM HUP
    disown
    echo "started: $run_dir"
    echo "  command: dune $*"
    echo "  log:     $run_dir/log"
    echo "  check:   tools/test-run.sh status last    # from this worktree; never blocks"
    echo "  gate:    tools/test-run.sh wait last      # bounded; exits with dune's status"
    ;;
  idle)
    [ $# -eq 0 ] || die "idle takes no arguments"
    # Use the same flock as run/start/repeat, including inherited holders after
    # a launcher or supervisor dies. This snapshot does not reserve the tree.
    probe_locks "$LOCK" "$PWD/.test-run.lock"
    idle_rc=$?
    case $idle_rc in
      0) ;;
      3) echo "test-run: worktree lock is held: $LOCK" >&2 ;;
      *) echo "test-run: cannot inspect worktree lock: $LOCK" >&2; idle_rc=2 ;;
    esac
    exit "$idle_rc"
    ;;
  status)
    resolve_run "${1:-last}"
    # One-shot and honest -- no sleeping in `status`. Ordered by LIVENESS,
    # never by pid-file presence: a supervisor killed after recording itself
    # must read as dead, not as running forever. The explicit `exit 0` on the
    # finished paths is the header contract: `status` reports PUBLICATION,
    # not the run's verdict, so it must not inherit whatever `digest`'s last
    # command happened to return.
    if [ -f "$run_dir/exit" ]; then
      digest "$run_dir"
      exit 0
    elif proc_alive "$run_dir/pid" "$run_dir/ptoken"; then
      echo "running: dune $(cat "$run_dir/cmd")  (log: $run_dir/log)"
      exit 3
    elif legacy_owner_alive "$run_dir"; then
      # A run of the previous version whose wrapper outlives its supervisor
      # to publish the verdict (or, for a repeat, its coordinator between
      # iterations): in flight, not dead.
      echo "managed, verdict pending (the previous version's wrapper is publishing): $run_dir"
      exit 3
    elif [ -f "$run_dir/exit" ]; then
      digest "$run_dir" # published between the checks above
      exit 0
    elif [ ! -f "$run_dir/pid" ] && lock_still_owned "$run_dir"; then
      # Published, but no supervisor on record, and the run holds its
      # worktree lock: the milliseconds between a launcher's publication and
      # its supervisor's first write. In flight -- the launcher is alive and
      # holding the lock -- so the running code, not the dead one.
      echo "launch in progress (supervisor not yet recorded): $run_dir"
      exit 3
    elif [ ! -f "$run_dir/pid" ]; then
      echo "launch interrupted before its supervisor started: $run_dir"
      exit 1
    else
      echo "run died without recording a verdict (killed externally?): $run_dir"
      exit 1
    fi
    ;;
  wait)
    ref=last budget=
    while [ $# -gt 0 ]; do
      case $1 in
        --timeout)
          [ $# -ge 2 ] || die "--timeout requires a value"
          budget=$2 budget_given=1
          shift 2
          ;;
        *) ref=$1; shift ;;
      esac
    done
    resolve_run "$ref"
    if [ -n "${budget_given:-}" ]; then
      # Same traps --cap had: leading zeros reach bash arithmetic as octal,
      # and oversized values wrap. Gated on the option being GIVEN, so an
      # explicitly empty value ('--timeout ""' from an unset harness
      # variable) is rejected rather than silently using the default budget.
      case $budget in '' | *[!0-9]*) die "--timeout must be a nonnegative integer of seconds" ;; esac
      [ ${#budget} -le 9 ] || die "--timeout too large (max 9 digits)"
      budget=$(( 10#$budget ))
    fi
    # Bounded by construction: default budget is every capped iteration plus
    # one cleanup allowance, so a managed repeat does not time out merely
    # because `cap` is per iteration. Ordinary runs have an implicit count 1.
    cap=$(cat "$run_dir/cap" 2>/dev/null) || cap=3600
    [ -n "$cap" ] || cap=3600
    cap=$(( 10#$cap )) # decimal, whatever an older run recorded
    wait_repeats=1
    if [ "$(cat "$run_dir/mode" 2>/dev/null)" = repeat ]; then
      wait_repeats=$(cat "$run_dir/repeats" 2>/dev/null) || wait_repeats=1
      case $wait_repeats in '' | *[!0-9]*) wait_repeats=1 ;; esac
      wait_repeats=$(( 10#$wait_repeats ))
      [ "$wait_repeats" -gt 0 ] || wait_repeats=1
    fi
    [ -n "$budget" ] || budget=$(( cap > 0 ? cap * wait_repeats + 120 : 7200 ))
    waited=0
    while [ ! -f "$run_dir/exit" ]; do
      # Every sleep here -- the poll AND the dead-supervisor grace -- is
      # bounded by the remaining budget, so a short --timeout is honored to
      # the second rather than rounded up to an interval.
      remaining=$(( budget - waited ))
      if ! sup_alive "$run_dir"; then
        # The owner is gone: no process is left to publish. One short settle,
        # bounded like every sleep here, covers a verdict renamed into place
        # between the checks above (the supervisor writes it last, and only
        # then exits); an already-expired budget reports the documented
        # timeout.
        step=$(( remaining < 1 ? remaining : 1 ))
        [ "$step" -gt 0 ] && sleep "$step"
        waited=$(( waited + step ))
        [ -f "$run_dir/exit" ] && break
        # Re-asked after the settle: `wait last` can arrive in the
        # milliseconds between a launcher's publication and its
        # supervisor's first write, and an owner that has appeared since is
        # a running run, not a dead one.
        sup_alive "$run_dir" && continue
        # Recomputed AFTER the settle: consuming the final second must report
        # the documented timeout, not slip through on the pre-sleep value.
        [ $(( budget - waited )) -le 0 ] && { echo "wait timed out after ${budget}s: $run_dir"; exit 124; }
        echo "run died without recording a verdict (killed externally?): $run_dir"
        exit 1
      fi
      [ "$remaining" -le 0 ] && { echo "wait timed out after ${budget}s: $run_dir"; exit 124; }
      step=$(( remaining < 5 ? remaining : 5 ))
      sleep "$step"
      waited=$(( waited + step ))
    done
    digest "$run_dir"
    exit "$digest_rc"
    ;;
  stop)
    resolve_run "${1:-last}"
    # Leftover recovery, shared by the finished and dead-without-verdict
    # branches. Descendants of dune may hold the run's worktree lock through
    # inherited fd 9, possibly having setsid'd out of the recorded group.
    # Everything uses the run's RECORDED worktree and state root, not the
    # caller's -- an explicit run directory may belong to another checkout,
    # or to another OCANNL_TOOL_TEST_RUNS. The gate:
    # the lock is HELD with the owner pointer naming THIS run, so whatever
    # holds it is this run's leftovers. The recorded group is signaled only
    # under a matching leader token (a recycled numeric pgid is never
    # trusted); the lsof census on the lock file covers detached holders
    # the group cannot see. reap_cycle reports honestly when the lock is
    # STILL held afterwards (no lsof on this system, or unkillable holders).
    lock_paths_of "$run_dir" || {
      # No worktree on record: nothing can attribute leftovers to this run,
      # so the leftover branches below stay closed and only the recorded
      # owner and group can be signalled.
      run_wt=$PWD run_lock= run_owner=
    }
    # The census prefers /proc/locks (only pids actually HOLDING the flock,
    # matched by device AND inode -- inode numbers repeat across
    # filesystems); lsof is the fallback and lists any process with the
    # file open. A NON-EMPTY proc answer is authoritative, but an empty one
    # proves nothing: the flock outlives its acquiring pid on the shared
    # description (our take_lock perl exits immediately), and /proc/locks
    # can then name no holder while the lock is demonstrably held -- so an
    # empty scan falls through to lsof rather than concluding "nobody".
    lock_holder_pids() {
      local out
      out=$(perl -e '
        my @st = stat($ARGV[0]) or exit 1;
        my $dev = $st[0];
        my ($maj, $min) = (($dev >> 8) & 0xfff, ($dev & 0xff) | (($dev >> 12) & ~0xff));
        open my $fh, "<", "/proc/locks" or exit 1;
        while (<$fh>) {
          my @f = split;
          next unless defined $f[1] && $f[1] eq "FLOCK";
          my ($lmaj, $lmin, $ino) = split /:/, $f[5];
          print "$f[4]\n"
            if defined $ino && $ino == $st[1] &&
               hex($lmaj) == $maj && hex($lmin) == $min;
        }
      ' "$run_lock" 2>/dev/null)
      if [ -n "$out" ]; then
        printf '%s\n' "$out"
        return 0
      fi
      if [ -d /proc ]; then
        # /proc/locks named nobody, yet the lock is held (inherited-only
        # descriptions can be invisible there) -- sweep every process's
        # fdinfo instead of depending on lsof, which need not be installed.
        for pd in /proc/[0-9]*; do
          p=${pd#/proc/}
          fd_holds_lock "$p" && printf '%s\n' "$p"
        done
        return 0
      fi
      # macOS: lsof lists OPENERS -- the accepted approximation, there being
      # no portable flock-holder API.
      lsof -t -- "$run_lock" 2>/dev/null
    }
    fd_holds_lock() { # Linux: does <pid> hold a FLOCK on the lock file NOW?
      perl -e '
        my ($pid, $target) = @ARGV;
        my @st = stat($target) or exit 1;
        opendir(my $dh, "/proc/$pid/fd") or exit 1;
        for my $fd (readdir $dh) {
          next if $fd =~ /^\./;
          my @fs = stat("/proc/$pid/fd/$fd") or next;
          next unless $fs[0] == $st[0] && $fs[1] == $st[1];
          open(my $fi, "<", "/proc/$pid/fdinfo/$fd") or next;
          while (<$fi>) { exit 0 if /^lock:.*FLOCK/ }
        }
        exit 1
      ' "$1" "$run_lock" 2>/dev/null
    }
    holds_lock_now() { # revalidated at SIGNAL time, not census time
      if [ -d /proc ]; then
        fd_holds_lock "$1"
      else
        lsof -t -- "$run_lock" 2>/dev/null | grep -qx "$1"
      fi
    }
    reap_leftovers() { # <signal>
      # Ownership rechecked at entry (and per census kill below): if the
      # lock changed hands since the caller's gate, nothing here may fire.
      lock_still_owned "$run_dir" || return 0
      if group_verified "$run_dir"; then
        kill "-$1" -- "-$(cat "$run_dir/pgid")" 2>/dev/null
      fi
      for p in $(lock_holder_pids); do
        case $p in '' | *[!0-9]* | 0) continue ;; esac
        # Re-verified per pid immediately before the signal: the census ran
        # as one batch, and a holder that exited meanwhile could have had
        # its pid recycled by an unrelated process. The ownership recheck
        # closes the handoff race too -- if a NEW launch acquired the lock
        # mid-reap, take_lock pointed the owner at the new run atomically
        # with acquisition, so the pointer can no longer name this run and
        # the new run's holders are never signaled.
        holds_lock_now "$p" || continue
        lock_still_owned "$run_dir" || return 0
        kill "-$1" "$p" 2>/dev/null
      done
    }
    reap_cycle() { # exits 0 iff this run's hold on the lock is gone
      lock_still_owned "$run_dir" || return 0
      reap_leftovers TERM
      sleep 2
      if lock_still_owned "$run_dir"; then
        reap_leftovers KILL
        sleep 1
      fi
      ! lock_still_owned "$run_dir"
    }
    report_reap() {
      if reap_cycle; then
        echo "$1, but its leftover processes held the worktree lock; reaped them"
      else
        echo "$1, but leftover processes STILL hold $run_wt's lock" \
             "(no lsof on this system, or unkillable holders); inspect with:" \
             "lsof $(printf %q "$run_lock")"
      fi
    }
    if [ -f "$run_dir/exit" ]; then
      lock_still_owned "$run_dir" && report_reap "finished"
      echo "already finished:"
      digest "$run_dir"
      exit 0
    fi
    # A stop landing in the launch window -- the run published, its
    # supervisor not yet on record -- must not read the live launcher as
    # leftovers and reap it: give the record the moment it needs, and let
    # the owner branch take the TERM. (A bounded wait, so a launch that
    # really died there still reaches the recovery below.)
    if [ ! -f "$run_dir/pid" ] && [ ! -f "$run_dir/exit" ] && lock_still_owned "$run_dir"; then
      for _ in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30; do
        [ -f "$run_dir/pid" ] && break
        sleep 0.1
      done
    fi
    if sup_alive "$run_dir"; then
      # The run's owner takes the TERM: the supervisor reaps dune and records
      # the cancellation; a repeat's coordinator owns the set-wide
      # cancellation bit -- killing only its current iteration's supervisor
      # would produce exit 143 and then let the outer loop launch every
      # remaining iteration while stop claimed success. For a run of the
      # previous version that coordinator is the recorded wrapper (see
      # sup_alive), and its `pid` is only the current iteration.
      # Name the run explicitly (%q-quoted): `last` may resolve to a
      # DIFFERENT run when this stop targeted an identifier from another
      # worktree's history.
      if [ "$(cat "$run_dir/mode" 2>/dev/null)" = repeat ] && legacy_owner_alive "$run_dir"; then
        kill -TERM "$(cat "$run_dir/wpid")" 2>/dev/null
        echo "sent TERM to the repeat coordinator; confirm with: tools/test-run.sh wait $(printf %q "$run_dir")"
      elif proc_alive "$run_dir/pid" "$run_dir/ptoken"; then
        kill -TERM "$(cat "$run_dir/pid")" 2>/dev/null
        if [ "$(cat "$run_dir/mode" 2>/dev/null)" = repeat ]; then
          echo "sent TERM to the repeat coordinator; confirm with: tools/test-run.sh wait $(printf %q "$run_dir")"
        else
          echo "sent TERM; confirm with: tools/test-run.sh wait $(printf %q "$run_dir")"
        fi
      else
        # The previous version's wrapper, publishing its verdict after the
        # supervisor exited: it ignores TERM by design, and there is nothing
        # left to cancel.
        echo "run is finishing (verdict publication in flight); confirm with:" \
             "tools/test-run.sh wait $(printf %q "$run_dir")"
      fi
    elif group_verified "$run_dir" &&
         pg=$(cat "$run_dir/pgid") && kill -0 -- "-$pg" 2>/dev/null; then
      # SIGKILL can remove the supervisor around a dune that survives in its
      # own recorded group -- still holding the worktree lock, beyond
      # its cap. Identity is the group LEADER's recorded start token (the
      # same mechanism as every other liveness check here); a recycled pgid
      # -- even one leading a process named dune -- fails the token and is
      # never signaled. A leaderless surviving group is refused too, the
      # conservative side: clean that up by hand. TERM gets a bounded grace,
      # then a revalidated KILL -- an orphan that ignores TERM has lost its
      # cap and would otherwise hold the worktree lock indefinitely.
      #
      # Reachability, not liveness, opens this branch, and every signal below
      # is sent on it alone: a census is a snapshot, so a child forked while it
      # was taken is missing from it, and letting that veto -- or downgrade --
      # the reap would leave the child holding the lock, or take away the TERM
      # it needed to shut down cleanly. group_alive decides only the GRACE and
      # the WORDING, which is where the phantom lived (gh-ocannl-742): a group
      # of unreaped corpses announced as a runaway dune ignoring TERM.
      kill -TERM -- "-$pg" 2>/dev/null
      # Asked AFTER the TERM, so a member the earlier census could have missed
      # is included: a grace has a point only where something can still act on
      # the signal, and corpses under a non-reaping init would otherwise cost
      # every stop two seconds.
      group_alive "$pg" && sleep 2
      if group_verified "$run_dir" && kill -0 -- "-$pg" 2>/dev/null; then
        # Something is still there. It may be running work or only unreaped
        # corpses; the KILL goes out either way, and one sentence covers both
        # because a second census can distinguish them only through a race.
        kill -KILL -- "-$pg" 2>/dev/null
        echo "orphaned process group $pg survived TERM (possibly only as" \
             "unreaped exited processes); escalated to KILL"
      else
        echo "sent TERM to the orphaned process group $pg; re-run stop to confirm"
      fi
    elif lock_still_owned "$run_dir"; then
      # Dead without a verdict, yet its leftovers still hold the worktree
      # lock (a setsid escapee outliving a killed supervisor) --
      # the same recovery as the finished branch, or later runs stay
      # refused forever.
      report_reap "run is dead without a verdict"
    else
      echo "nothing left to signal (supervisor gone; no identity-verified surviving group)"
    fi
    ;;
  list)
    found=0
    for d in "$RUNS"/2*/; do
      [ -d "$d" ] || continue
      found=1
      d=${d%/}
      if [ -f "$d/exit" ]; then state="exit $(cat "$d/exit")"
      elif sup_alive "$d"; then state=running
      else state=dead
      fi
      printf '%s  %-8s  dune %s\n' "$(basename "$d")" "$state" "$(cat "$d/cmd" 2>/dev/null)"
    done
    [ "$found" = 1 ] || echo "no recorded runs in $RUNS"
    ;;
  *) die "unknown subcommand: $sub (run|start|repeat|status|wait|stop|list|idle)" ;;
esac
