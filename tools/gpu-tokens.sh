#!/usr/bin/env bash
# The fleet's GPU tokens as a sweep GPU unit meets them (gh-ocannl-1177), sourced by tools/sweep.sh
# and driven directly by test/operations/sweep_components.sh. The caller provides `sq` (POSIX
# single-quoting). Sourcing defines functions only; nothing here opens an ssh or writes state the
# caller did not name.
#
# A fleet box bounds how many batches hold its GPU with GPU TOKENS (lukstafi/ludics-lite#391): the
# first T of the box's S run-time correctness slot files, each a real flock that an `fleet-worker.sh
# execution slot` wrapper holds for as long as its batch runs. rog-nv-linux has S=4 and T=2, because
# three concurrent cuda batches were measured to run out of device memory (lukstafi/ludics-lite#316,
# #344); every other box has as many tokens as slots, where they do not bind. The sweep took none,
# so a sweep unit could be the third batch. Two answers live here:
#
#   RECORD, every remote GPU unit: which slot files are held, by which process, read from the box's
#     own flock table at the unit's start (`opened`) and after its suite (`closed`), so a launch OOM
#     can be attributed to contention rather than diagnosed by hand. One JSON line per reading in
#     the unit's `<log stem>.gpu-tokens` sidecar, in the vocabulary of the fleet's co-tenant records
#     (lukstafi/ludics-lite#566: `box`, `event`, `at`, and `pid`/`user`/`command` per process), so a
#     reader can line the two up. A `reservations` line beside them carries the fleet registry's
#     active reservations on the box (`execution list --active`, read before the unit): those name
#     the workers, but a reservation holds no token, so the counts come from the flocks alone.
#   TAKE, where the tokens bind (T < S; today rog-nv-linux, so rog-nv's cuda unit): the unit's GPU
#     legs -- its suite, its serial rerun, the completion pass -- each run inside one GPU token,
#     taken through the box's own `fleet-worker.sh execution slot --gpu` with a bounded wait. A token
#     that does not come free in time is a `skip` naming the holders, never a third batch. Workers'
#     batches and the fleet's GPU integration loop (lukstafi/ludics-lite#615) wait for the same two
#     tokens, so the sweep competes for them like any other batch, neither ahead nor behind.
#
# THE READER'S BOUNDARY (a fail-closed allowlist of what it accepts):
#   - the box's fleet name, slots and tokens come only from a line of exactly the form
#     `EXECUTION SLOT PROBE <box> <slots> <tokens>` (anything after a further space ignored), from the
#     first fleet-worker.sh candidate that answers `execution slot --probe`; any other answer, or no
#     candidate, is `status: unread` and nothing is counted;
#   - a holder is a line of the kernel's flock table (`/proc/locks`, or OCANNL_TOOL_PROC_LOCKS on the
#     box, the harness's stand-in) of exactly the form `<n>: FLOCK <ADVISORY|MANDATORY> WRITE <pid>
#     <maj hex>:<min hex>:<inode> ...` whose device and inode are a slot file's
#     (`$FLEET_SLOT_STATE/<box>/slot.<i>`, its default under $HOME); a blocked waiter (`->`), a
#     read lock, a POSIX lock and every other line are not holders;
#   - a holder's kind is `cpu` when its slot is past the tokens (only a `--cpu` batch can hold one
#     there) or when its argv says `--cpu` before the first `--`, and `gpu` otherwise -- the fleet's
#     own fail-closed rule, under which an undeclared batch holds the GPU.
# A box without /proc/locks (macOS) cannot be read, so the local metal unit records nothing, and a
# token taken and released between two readings is not seen.

# How long a unit waits for a token before it is a `skip` (OCANNL_TOOL_SWEEP_TOKEN_WAIT overrides).
# A worker's cuda batch on rog runs 15-40 minutes, and the daily sweep starts at dawn, when the
# fleet is usually idle: half an hour clears a batch already running without losing the morning.
GPU_TOKEN_WAIT_DEFAULT=1800

# The reader, run on the box with perl (Ubuntu's, which ships JSON::PP): one JSON line on stdout.
# Arguments: event, the fleet-worker.sh that answered (empty for none), its probe line, the flock
# table's path and the slot state directory.
gpu_tokens_reader='
  use strict;
  use warnings;
  use JSON::PP ();
  use POSIX ();
  my ($event, $fw, $probe, $locks, $state) = @ARGV;
  my %r = (event => $event, at => POSIX::strftime("%Y-%m-%dT%H:%M:%S+00:00", gmtime));
  my $out = sub { print JSON::PP->new->canonical->encode(\%r), "\n"; exit 0 };
  my $unread = sub { $r{status} = "unread"; $r{detail} = shift; $out->() };
  $unread->("no fleet-worker.sh answered \x27execution slot --probe\x27") if $fw eq "";
  my ($box, $slots, $tokens) = $probe =~ /^EXECUTION SLOT PROBE (\S+) ([1-9][0-9]*) ([1-9][0-9]*)(?: |$)/;
  $unread->("the probe answered: " . substr($probe, 0, 200)) unless defined $tokens;
  @r{qw(box slots gpu_tokens fleet_worker)} = ($box, 0 + $slots, 0 + $tokens, $fw);
  open(my $l, "<", $locks) or $unread->("cannot read $locks: $!");
  my %held;
  while (my $line = <$l>) {
    next unless $line =~ /^\d+:\s+FLOCK\s+(?:ADVISORY|MANDATORY)\s+WRITE\s+(\d+)\s+([0-9a-f]+):([0-9a-f]+):(\d+)\s/i;
    push @{ $held{ hex($2) . ":" . hex($3) . ":" . $4 } }, 0 + $1;
  }
  close $l;
  my @holders;
  for my $i (1 .. $slots) {
    my @st = stat "$state/$box/slot.$i" or next;
    my $d = $st[0];
    # st_dev as glibc encodes it, back into the major:minor the kernel prints.
    my $key = ((($d >> 8) & 0xfff) | (($d >> 32) & ~0xfff)) . ":" . (($d & 0xff) | (($d >> 12) & ~0xff))
      . ":" . $st[1];
    for my $pid (@{ $held{$key} || [] }) {
      my @argv;
      if (open(my $c, "<", "/proc/$pid/cmdline")) { local $/; @argv = split /\0/, (<$c> // "") }
      my $args = qx{ps -o args= -p $pid 2>/dev/null};
      chomp $args;
      my $user = qx{ps -o user= -p $pid 2>/dev/null};
      $user =~ s/\s+//g;
      @argv = split " ", $args unless @argv;
      my $cpu = $tokens < $slots && $i > $tokens;
      for (@argv) { last if $_ eq "--"; $cpu = 1 if $_ eq "--cpu" }
      my %h = (slot => $i, kind => $cpu ? "cpu" : "gpu", pid => $pid, user => $user,
        command => substr($args, 0, 300));
      my $cwd = readlink "/proc/$pid/cwd";
      $h{cwd} = $cwd if defined $cwd;
      push @holders, \%h;
    }
  }
  $r{status} = "read";
  $r{token_holders} = \@holders;
  $r{gpu_holders} = scalar grep { $_->{kind} eq "gpu" } @holders;
  $out->();
'

gpu_tokens_sidecar() { # log -> its readings' file
  printf '%s.gpu-tokens' "${1%.log}"
}

# The reading as shell text for the box: the fleet-worker.sh candidates (absolute paths on the box)
# are asked in turn, as tools/fleet-worker-candidates.sh asks them here.
gpu_tokens_cmd() { # event candidate...
  local event=$1 c cands=
  shift
  for c; do cands="$cands $(sq "$c")"; done
  printf 'fw=; probe=; for c in%s; do [ -x "$c" ] || continue; ' "$cands"
  printf 'probe=$("$c" execution slot --probe 2>/dev/null </dev/null) || continue; fw=$c; break; done; '
  printf 'exec perl -e %s -- %s "$fw" "$probe" "${OCANNL_TOOL_PROC_LOCKS:-/proc/locks}" ' \
    "$(sq "$gpu_tokens_reader")" "$(sq "$event")"
  printf '"${FLEET_SLOT_STATE:-$HOME/.local/state/fleet-execution-slots}"'
}

# Append one reading to a sidecar: the box's answer when it is one JSON object on one line, else an
# `unread` line saying why -- a reading that did not happen is not a box with no holders.
gpu_tokens_append() { # sidecar event answer-file rc
  perl -MJSON::PP -MPOSIX -e '
    my ($sidecar, $event, $answer, $rc) = @ARGV;
    my $json = JSON::PP->new->canonical;
    my $text = "";
    if (open(my $a, "<", $answer)) { local $/; $text = <$a> // "" }
    my $r = eval { $json->decode($text) };
    unless ($rc == 0 && ref $r eq "HASH" && ($r->{event} // "") eq $event && $text =~ /\A[^\n]*\n?\z/) {
      $r = { event => $event, at => POSIX::strftime("%Y-%m-%dT%H:%M:%S+00:00", gmtime),
        status => "unread", detail => "the box gave no reading (exit $rc)" };
    }
    open(my $s, ">>", $sidecar) or exit 1;
    print $s $json->encode($r), "\n" or exit 1;
    close $s or exit 1;' "$1" "$2" "$3" "$4"
}

# The registry's active reservations on the box, as the lane read them before the unit.
gpu_tokens_append_reservations() { # sidecar box reservations-json-array
  perl -MJSON::PP -MPOSIX -e '
    my ($sidecar, $box, $list) = @ARGV;
    my $json = JSON::PP->new->canonical;
    my $r = eval { $json->decode($list) };
    $r = [] unless ref $r eq "ARRAY";
    open(my $s, ">>", $sidecar) or exit 1;
    print $s $json->encode({ event => "reservations", box => $box, reservations => $r,
      at => POSIX::strftime("%Y-%m-%dT%H:%M:%S+00:00", gmtime) }), "\n" or exit 1;
    close $s or exit 1;' "$1" "$2" "$3"
}

# The sidecar summarised by one perl reader, in one of four forms:
#   field        the run record's `gpu_tokens` field: `<event>=<gpu holders>/<tokens>` per reading,
#                comma-separated (`?` for an unread one), or `-` with no readings at all;
#   fingerprint  its stable half: whether any reading saw another batch holding the GPU -- never the
#                counts or the pids, which differ between two equally contended runs;
#   report       a summary line per reading that saw a GPU holder, naming each;
#   take         `<fleet-worker.sh> <box> <slots> <tokens>` from the `opened` reading, when it
#                answered the probe -- what a unit needs to take a token.
gpu_tokens_summary() { # form sidecar
  [ -s "$2" ] || { [ "$1" != field ] || printf '%s' -; return 0; }
  perl -MJSON::PP -e '
    my ($form, $sidecar) = @ARGV;
    open(my $s, "<", $sidecar) or exit 1;
    my @r = grep { ref $_ eq "HASH" && ($_->{event} // "") ne "reservations" }
      map { my $l = $_; eval { JSON::PP->new->decode($l) } } <$s>;
    my $known = sub { ($_[0]{status} // "") eq "read" };
    if ($form eq "field") {
      print join(",", map { $_->{event} . "=" . ($known->($_) ? "$_->{gpu_holders}/$_->{gpu_tokens}" : "?") } @r);
    } elsif ($form eq "fingerprint") {
      my @read = grep { $known->($_) } @r;
      print !@read ? "gpu tokens: not read\n"
        : (grep { $_->{gpu_holders} > 0 } @read) ? "gpu tokens: other batches held the GPU\n"
        : "gpu tokens: no other batch held the GPU\n";
    } elsif ($form eq "report") {
      for my $r (grep { $known->($_) && $_->{gpu_holders} > 0 } @r) {
        my @h = map { sprintf "slot %d pid %d%s%s: %s", $_->{slot}, $_->{pid},
            $_->{user} ne "" ? " ($_->{user})" : "", defined $_->{cwd} ? " in $_->{cwd}" : "",
            substr($_->{command}, 0, 120) } grep { $_->{kind} eq "gpu" } @{ $r->{token_holders} };
        printf "gpu tokens: %d of %d held by other batches at %s on %s -- %s\n", $r->{gpu_holders},
          $r->{gpu_tokens}, $r->{event}, $r->{box}, join("; ", @h);
      }
    } elsif ($form eq "take") {
      my ($o) = grep { $_->{event} eq "opened" && defined $_->{gpu_tokens} } @r;
      print "$o->{fleet_worker} $o->{box} $o->{slots} $o->{gpu_tokens}\n" if $o;
    }' "$1" "$2"
}

# One leg run inside a GPU token, as shell text for the box. The leg is `capped` -- the far-side
# supervisor's text (remote_capped), spliced here as words, so the supervisor stays the direct
# parent of the leg's `sh -c` (the per-test cap's watcher signals that parent) -- and the token is
# held around it by `<fleet-worker.sh> execution slot --gpu --wait <wait>`, which leads the batch's
# process group and ends it when the slot wrapper is signalled, as an ssh drop does.
#
# The slot's own lines go to `<dir>/slot.out` and the leg's to the log, through fds 3 and 4 that
# the wrapper passes on (it closes none). Only the leg itself creates `<dir>/entered`, so after the
# wrapper returns the text knows whether the leg ran:
#   - it ran: its status is the leg's (the slot returns its command's), and the slot's lines --
#     which token it held -- follow in the log as `sweep: gpu token:` lines. Only its own
#     `EXECUTION ...` lines, each cut before ` held for: `: the slot names its command there, and the
#     leg's text holds the very `=== serial rerun <alias>: exit $rc ===` echoes the rerun's verdict
#     greps the log for, which a copy in the log would answer instead of the leg;
#   - the slot REFUSED for want of a token (`EXECUTION SLOT REFUSED <box>: all <n> GPU tokens ...`)
#     or for a measurement holding the box (`EXECUTION SLOT REFUSED <box>: a measurement ...` or
#     `... <box>: measurement window ...`): exit 75 after
#     a `sweep: gpu token: REFUSED -- <why>` line, which the sweep reads as a `skip`; those two
#     forms are the whole list;
#   - anything else (an anchor whose registry did not answer, a fleet-worker.sh that would not
#     start): the leg runs WITHOUT a token after a `sweep: gpu token: WARNING` line, as the sweep
#     fails open whenever the fleet's machinery cannot answer.
gpu_token_take_cmd() { # fleet-worker wait dir capped-leg-text
  local fw=$1 wait=$2 dir=$3 capped=$4 said
  said="awk $(sq '/^EXECUTION / { sub(/ held for: .*/, " held"); print "sweep: gpu token: " substr($0, 1, 300) }') $(sq "$dir/slot.out")"
  printf 'rm -rf %s && mkdir -p %s || exit 126; ' "$(sq "$dir")" "$(sq "$dir")"
  printf '%s execution slot --gpu --wait %s -- sh -c %s sweep-token %s %s 3>&1 4>&2 >%s 2>&1; rc=$?; ' \
    "$(sq "$fw")" "$wait" \
    "$(sq ': >"$1/entered" || exit 126; exec >&3 2>&4 3>&- 4>&-; shift; exec "$@"')" \
    "$(sq "$dir")" "$capped" "$(sq "$dir/slot.out")"
  printf '%s; ' "$said"
  printf '[ ! -e %s ] || exit $rc; ' "$(sq "$dir/entered")"
  printf 'if grep -Eq "^EXECUTION SLOT REFUSED [^ ]+: all [0-9]+ GPU tokens " %s; then ' "$(sq "$dir/slot.out")"
  printf 'echo "sweep: gpu token: REFUSED -- no GPU token came free within %ss"; exit 75; fi; ' "$wait"
  printf 'if grep -Eq "^EXECUTION SLOT REFUSED [^ ]+: (a )?measurement " %s; then ' "$(sq "$dir/slot.out")"
  printf 'echo "sweep: gpu token: REFUSED -- a measurement holds the box"; exit 75; fi; '
  printf 'echo "sweep: gpu token: WARNING: not taken (execution slot exited $rc), so this leg runs WITHOUT one"; '
  printf '%s' "$capped"
}

# The refusal a leg's log ends with, or nothing: the reason after `sweep: gpu token: REFUSED -- `.
gpu_token_refusal() { # log [first-line]
  tail -n "+${2:-1}" "$1" 2>/dev/null | sed -n 's/^sweep: gpu token: REFUSED -- //p' | tail -1
}
