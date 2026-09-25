#!/usr/bin/env bash
# Verify a pushed branch on a named machine -- another box over SSH, or this
# one -- without borrowing that machine's checkout, build tree, or ambient
# shell setup.
#
# Usage:
#   tools/machine-verify.sh BOX BRANCH [OPTIONS]
#
# Options:
#   --backend NAME           Pin and prove the resolved backend configuration.
#   --expect-lib LIB         Prove cudajit, hipjit or metal was compiled and
#                            selected. This implies backend cuda, hip or metal
#                            respectively.
#   --test ALIAS             Build one named test alias (repeatable).
#   --run 'COMMAND'          Run an OCANNL probe under opam and the pinned
#                            backend (repeatable).
#   --record-golden ALIAS    Run one golden alias, print corrected contents and
#                            an apply-ready patch (repeatable).
#   --local                  BOX must be this machine: run here, never over
#                            SSH, and refuse if BOX's endpoint is elsewhere.
#   --ssh                    Always go through SSH, even when BOX is this
#                            machine.
#   --repo PATH              Staging checkout on BOX (default:
#                            $HOME/ocannl-staging there).
#   --remote NAME            Git remote pointing to lukstafi/ocannl-staging
#                            (default: derive it by URL on BOX).
#   --worktree-root PATH     Temporary-worktree parent on BOX (default:
#                            $HOME/ocannl-staging-worktrees there).
#   --cap SECONDS            Per-build/test/probe wall-clock cap; 0 disables
#                            it (default: 5400).
#   --trip-cap SECONDS       Whole-trip cap, including setup and cleanup;
#                            0 disables it (default: 21600). --ssh-cap is the
#                            same option under its former name.
#   -j, --jobs N             Dune concurrency, 1..4 (default: 4).
#
# Where it runs: without --local or --ssh, BOX runs here exactly when its SSH
# endpoint is this machine -- `ssh -G BOX` names the host and port (no
# connection is made), and every address that host resolves to is one this
# machine can bind, at port 22, with no ProxyJump/ProxyCommand. An endpoint
# with no address here goes over SSH; a split answer (some addresses here,
# some not, or a local address on another port) is refused as ambiguous
# rather than guessed. `localhost` is always this machine.
#
# Examples (BOX is the booted system's ssh alias: rog-nv-linux / minix-amd-linux
# on native Ubuntu, rog-nv-wsl / minix-amd-wsl on WSL -- the site's kind_of says
# which -- or mac-studio for Metal):
#   tools/machine-verify.sh rog-nv-linux codex/my-branch \
#     --expect-lib cudajit --test @arrayjit/runtest-test_cuda_arch_flags \
#     --run 'dune build tools/fp8_soak.exe -j 4 && \
#       _build/default/tools/fp8_soak.exe --arm=cuda --sweep=f32'
#   tools/machine-verify.sh minix-amd-linux codex/my-branch \
#     --expect-lib hipjit \
#     --record-golden @test/training/train-transformer_names
#   tools/machine-verify.sh mac-studio codex/my-branch --local \
#     --expect-lib metal --test @test/operations/runtest-hello_world_op
#
# The output is deliberately unpiped. A failed dune diff must be dune's status,
# not tail's or tee's. The verification procedure prints an exit sentinel only
# after removing the temporary worktree; this side then prints the transport's
# status (`ssh exit:` or `local exit:`).
#
# The procedure itself is tools/machine-verify-far.sh, the single definition
# both transports run. Locally it runs under `env -i` keeping only the session
# basics (HOME, USER, LOGNAME, PATH, SHELL, TMPDIR, LANG, LC_ALL, LC_CTYPE,
# SSH_AUTH_SOCK) and from $HOME, so a caller's opam, Dune or OCANNL settings
# reach it no more than they would reach an SSH session.
#
# `--run` is intentionally a shell command: device probes often need several
# build/run arguments. It is executed by `opam exec -- sh -c` from the pinned
# worktree with OCANNL_BACKEND exported. The command must itself be an OCANNL
# probe whose output demonstrates the device/backend property being checked;
# this harness proves its source, configuration and exit status, but cannot turn
# an arbitrary command into backend evidence.

set -u

die() { echo "machine-verify: $*" >&2; exit 2; }

usage() {
  sed -n '2,/^# The output/s/^# \{0,1\}//p' "$0" >&2
  exit 2
}

sq() { printf "'%s'" "$(printf %s "$1" | sed "s/'/'\\\\''/g")"; }

here=$(cd "$(dirname "$0")" && pwd -P) || die "cannot locate the tools directory"
far=$here/machine-verify-far.sh
[ -r "$far" ] || die "cannot read the verification procedure $far"

[ $# -ge 2 ] || usage
box=$1
branch=$2
shift 2

backend=
expect_lib=
remote_repo=
staging_remote=
worktree_root=
cap=5400
trip_cap=21600
jobs=4
placement=auto
operations=()
operation_count=0

while [ $# -gt 0 ]; do
  case $1 in
    --backend)
      [ $# -ge 2 ] || die "--backend needs a value"
      backend=$2
      shift 2
      ;;
    --expect-lib)
      [ $# -ge 2 ] || die "--expect-lib needs cudajit, hipjit or metal"
      expect_lib=$2
      shift 2
      ;;
    --test | --run | --record-golden)
      [ $# -ge 2 ] || die "$1 needs a value"
      operations+=("${1#--}" "$2")
      operation_count=$((operation_count + 2))
      shift 2
      ;;
    --local | --ssh)
      [ "$placement" = auto ] || [ "$placement" = "${1#--}" ] ||
        die "--local and --ssh are mutually exclusive"
      placement=${1#--}
      shift
      ;;
    --repo)
      [ $# -ge 2 ] || die "--repo needs an absolute path on BOX"
      remote_repo=$2
      shift 2
      ;;
    --remote)
      [ $# -ge 2 ] || die "--remote needs a name"
      staging_remote=$2
      shift 2
      ;;
    --worktree-root)
      [ $# -ge 2 ] || die "--worktree-root needs an absolute path on BOX"
      worktree_root=$2
      shift 2
      ;;
    --cap)
      [ $# -ge 2 ] || die "--cap needs a value"
      cap=$2
      shift 2
      ;;
    --trip-cap | --ssh-cap)
      [ $# -ge 2 ] || die "$1 needs a value"
      trip_cap=$2
      shift 2
      ;;
    -j | --jobs)
      [ $# -ge 2 ] || die "$1 needs a value"
      jobs=$2
      shift 2
      ;;
    -h | --help) usage ;;
    *) die "unknown argument: $1" ;;
  esac
done

case $box in '' | -*) die "BOX must not be empty or begin with '-'" ;; esac
git check-ref-format --branch "$branch" >/dev/null 2>&1 || die "invalid branch name: $branch"

case $jobs in 1 | 2 | 3 | 4) ;; *) die "jobs must be between 1 and 4" ;; esac
case $cap in '' | *[!0-9]*) die "cap must be a non-negative integer" ;; esac
case $trip_cap in '' | *[!0-9]*) die "trip cap must be a non-negative integer" ;; esac
case $staging_remote in -*) die "--remote must not begin with '-'" ;; esac
case $backend in
  '' | cc | multidev_cc | cuda | hip | metal) ;;
  *) die "unknown backend '$backend'; expected cc, multidev_cc, cuda, hip, or metal" ;;
esac
case $expect_lib in
  '') ;;
  cudajit | hipjit | metal)
    case $expect_lib in
      cudajit) lib_backend=cuda ;;
      hipjit) lib_backend=hip ;;
      metal) lib_backend=metal ;;
    esac
    [ -z "$backend" ] || [ "$backend" = "$lib_backend" ] ||
      die "--expect-lib $expect_lib conflicts with --backend $backend"
    backend=$lib_backend
    ;;
  *) die "unknown optional library '$expect_lib'; expected cudajit, hipjit or metal" ;;
esac

for ((i = 0; i < operation_count; i += 2)); do
  kind=${operations[i]}
  value=${operations[i + 1]}
  case $kind in
    test | record-golden)
      case $value in @?*) ;; *) die "--$kind expects a named alias beginning with @" ;; esac
      ;;
  esac
  [ -n "$backend" ] ||
    die "--$kind requires --backend (or --expect-lib) for explicit configuration provenance"
done

case $remote_repo in '' | /*) ;; *) die "--repo must be an absolute path on BOX" ;; esac
case $worktree_root in '' | /*) ;; *) die "--worktree-root must be an absolute path on BOX" ;; esac

# One process-group supervisor is used on both sides of the transport. This
# instance bounds the whole trip, setup plus cleanup; the same source is passed
# as a positional argument to the verification procedure and bounds each
# build/test/probe there. Exit 142 means the wall-clock cap expired. It is perl
# rather than timeout(1) because macOS ships no timeout(1).
capped_perl='
  use POSIX ();
  my $cap = shift;
  my ($pid, $done);
  my $blast = sub { my $sig = shift; kill($sig, -$pid) or kill($sig, $pid) };
  my $reap = sub {
    my $code = shift;
    exit $done if defined $done;
    if ($pid) {
      my $saved = $?;
      my $r = waitpid($pid, POSIX::WNOHANG());
      if ($r == -1) {
        exit(($saved & 127) ? 128 + ($saved & 127) : $saved >> 8);
      }
      if ($r == $pid) {
        my $st = $?;
        exit(($st & 127) ? 128 + ($st & 127) : $st >> 8);
      }
      $blast->("TERM");
      my $gone = 0;
      for (1 .. 50) {
        $gone = 1 if !$gone && waitpid($pid, POSIX::WNOHANG()) != 0;
        last if $gone && !kill(0, -$pid);
        select undef, undef, undef, 0.1;
      }
      if (!$gone || kill(0, -$pid)) {
        $blast->("KILL");
        unless ($gone) {
          for (1 .. 50) {
            last if waitpid($pid, POSIX::WNOHANG()) != 0;
            select undef, undef, undef, 0.1;
          }
        }
      }
    }
    exit $code;
  };
  $SIG{ALRM} = sub { $reap->(142) };
  $SIG{INT} = sub { $reap->(130) };
  $SIG{TERM} = sub { $reap->(143) };
  $SIG{HUP} = sub { $reap->(129) };
  $pid = fork();
  die "fork: $!" unless defined $pid;
  if (!$pid) {
    $SIG{TERM} = "DEFAULT"; $SIG{INT} = "DEFAULT"; $SIG{HUP} = "DEFAULT";
    eval { setpgrp(0, 0) };
    exec @ARGV;
    exit 127;
  }
  alarm $cap if $cap > 0;
  waitpid($pid, 0);
  my $st = $?;
  $done = ($st & 127) ? 128 + ($st & 127) : $st >> 8;
  $pid = 0;
  alarm 0;
  exit $done;
'
local_capped() {
  local budget=$1
  shift
  perl -e "$capped_perl" -- "$budget" "$@"
}

# Is BOX this machine? The question is what `ssh BOX` would reach, so it is
# asked of ssh's own configuration rather than of a hostname: Apple names the
# mac-studio host `LukaszsacStudio`, and a map of the site's box names kept here
# would be a second copy of that configuration. `ssh -G` applies the config
# offline, and binding each resolved address is the kernel's own answer to "is
# this address assigned here" -- the same on Linux and macOS, with no
# interface-listing tool (ip vs ifconfig) to parse. An address family this host
# cannot open a socket for is one ssh could not use either, so it is skipped.
resolve_here_perl='
  use Socket qw(:addrinfo SOCK_STREAM);
  my $host = shift;
  my ($err, @res) = getaddrinfo($host, "", { socktype => SOCK_STREAM });
  if ($err) { print STDERR "machine-verify: cannot resolve $host: $err\n"; exit 3 }
  my %seen;
  for my $r (@res) {
    my ($nerr, $ip) = getnameinfo($r->{addr}, NI_NUMERICHOST);
    $ip = "?" if $nerr;
    next if $seen{$ip}++;
    socket(my $s, $r->{family}, SOCK_STREAM, 0) or next;
    print "$ip ", (bind($s, $r->{addr}) ? "here" : "elsewhere"), "\n";
    close $s;
  }
'
locate_box() {
  local cfg resolved here_n=0 else_n=0 addr where
  locality=none
  endpoint=$box
  endpoint_port=22
  endpoint_proxy=
  endpoint_here=
  endpoint_else=
  if cfg=$(local_capped 20 ssh -G "$box" </dev/null 2>/dev/null); then
    endpoint=$(printf '%s\n' "$cfg" | sed -n 's/^hostname //p' | sed -n 1p)
    endpoint_port=$(printf '%s\n' "$cfg" | sed -n 's/^port //p' | sed -n 1p)
    endpoint_proxy=$(printf '%s\n' "$cfg" | sed -n 's/^proxyjump //p; s/^proxycommand //p' |
      grep -vx none | sed -n 1p)
  else
    echo "machine-verify: ssh -G $box failed; reading BOX as a host name at port 22"
  fi
  [ -n "$endpoint" ] || endpoint=$box
  [ -n "$endpoint_port" ] || endpoint_port=22
  # A proxied endpoint's HostName is resolved from the proxy, not from here.
  [ -z "$endpoint_proxy" ] || return 0
  resolved=$(local_capped 20 perl -e "$resolve_here_perl" -- "$endpoint") || {
    locality=unresolved
    return 0
  }
  while read -r addr where; do
    case $where in
      here) here_n=$((here_n + 1)) endpoint_here="$endpoint_here $addr" ;;
      elsewhere) else_n=$((else_n + 1)) endpoint_else="$endpoint_else $addr" ;;
    esac
  done <<EOF
$resolved
EOF
  if [ "$here_n" -gt 0 ] && [ "$else_n" -eq 0 ]; then
    locality=all
  elif [ "$here_n" -gt 0 ]; then
    locality=some
  fi
}

endpoint_story() {
  printf '%s -> %s:%s' "$box" "$endpoint" "$endpoint_port"
  [ -z "$endpoint_proxy" ] || printf ' via proxy %s' "$endpoint_proxy"
  [ -z "$endpoint_here" ] || printf '; on this machine:%s' "$endpoint_here"
  [ -z "$endpoint_else" ] || printf '; elsewhere:%s' "$endpoint_else"
  [ "$locality" != unresolved ] || printf '; unresolved'
}

if [ "$placement" = ssh ]; then
  transport=ssh
  transport_story="ssh (--ssh given; endpoint not inspected)"
else
  locate_box
  story=$(endpoint_story)
  case $placement/$locality in
    auto/all)
      [ "$endpoint_port" = 22 ] ||
        die "ambiguous placement: $story; a local address on port $endpoint_port may be a forward to another system -- pass --local or --ssh"
      transport=local
      ;;
    local/all | local/some) transport=local ;;
    auto/some)
      die "ambiguous placement: $story -- pass --local to run here or --ssh to connect"
      ;;
    auto/none | auto/unresolved) transport=ssh ;;
    local/*)
      die "--local refused: BOX $box is not this machine ($(hostname 2>/dev/null || uname -n)): $story"
      ;;
  esac
  if [ "$transport" = local ]; then
    transport_story="local, no ssh ($story)"
  else
    transport_story="ssh ($story)"
  fi
fi
echo "machine-verify: transport: $transport_story"

# ssh concatenates its remote argv into shell text. Quote every value once here,
# then let /bin/sh recover the exact positional arguments. The outer shell moves
# the verification procedure to fd 3 and replaces stdin with /dev/null before the
# inner shell reads that program; no child can consume the caller's remaining
# control flow. In particular, --run commands are never interpolated into this
# string. The local transport hands the SAME string to /bin/sh -c, as sshd does.
remote_command="/bin/sh -c 'exec 3<&0; exec </dev/null; exec /bin/sh /dev/fd/3 \"\$@\"' machine-verify"
for arg in "$box" "$branch" "$backend" "$expect_lib" "$remote_repo" "$staging_remote" \
  "$worktree_root" "$cap" "$trip_cap" "$jobs" "$transport_story" "$capped_perl"; do
  remote_command="$remote_command $(sq "$arg")"
done
if [ "$operation_count" -gt 0 ]; then
  for arg in "${operations[@]}"; do
    remote_command="$remote_command $(sq "$arg")"
  done
fi

if [ "$transport" = local ]; then
  # What an SSH session would start from: a fresh environment holding only the
  # session basics, in $HOME. Names are printed, never values.
  [ -n "${HOME:-}" ] || die "HOME is not set; the local transport starts from it"
  local_env=(env -i)
  kept=
  for name in HOME USER LOGNAME PATH SHELL TMPDIR LANG LC_ALL LC_CTYPE SSH_AUTH_SOCK; do
    if value=$(printenv "$name"); then
      local_env+=("$name=$value")
      kept="$kept $name"
    fi
  done
  echo "machine-verify: local environment: cleared except$kept"
  cd "$HOME" || die "cannot enter $HOME"
  local_capped "$trip_cap" "${local_env[@]}" /bin/sh -c "$remote_command" <"$far"
  trip_rc=$?
  echo "machine-verify: local exit: $trip_rc"
else
  local_capped "$trip_cap" ssh -o BatchMode=yes -o ConnectTimeout=8 \
    -o ServerAliveInterval=30 -o ServerAliveCountMax=10 \
    "$box" "$remote_command" <"$far"
  trip_rc=$?
  echo "machine-verify: ssh exit: $trip_rc"
fi
exit "$trip_rc"
