#!/usr/bin/env bash
# Verify a pushed branch on another machine without borrowing that machine's
# checkout, build tree, or ambient shell setup.
#
# Usage:
#   tools/remote-verify.sh BOX BRANCH [OPTIONS]
#
# Options:
#   --backend NAME           Pin and prove the resolved backend configuration.
#   --expect-lib LIB         Prove cudajit or hipjit was compiled and selected.
#                            This implies backend cuda or hip respectively.
#   --test ALIAS             Build one named test alias (repeatable).
#   --run 'COMMAND'          Run an OCANNL probe under opam and the pinned
#                            backend (repeatable).
#   --record-golden ALIAS    Run one golden alias, print corrected contents and
#                            an apply-ready patch (repeatable).
#   --repo PATH              Remote staging checkout (default:
#                            $HOME/ocannl-staging on the remote).
#   --remote NAME            Remote pointing to lukstafi/ocannl-staging
#                            (default: derive it by URL on the remote box).
#   --worktree-root PATH     Remote temporary-worktree parent (default:
#                            $HOME/ocannl-staging-worktrees on the remote).
#   --cap SECONDS            Per-build/test/probe wall-clock cap; 0 disables
#                            it (default: 5400).
#   --ssh-cap SECONDS        Whole SSH-trip cap, including setup and cleanup;
#                            0 disables it (default: 21600).
#   -j, --jobs N             Dune concurrency, 1..4 (default: 4).
#
# Examples (BOX is the booted system's ssh alias: rog-nv-linux / minix-amd-linux
# on native Ubuntu, rog-nv-wsl / minix-amd-wsl on WSL -- the site's kind_of says
# which):
#   tools/remote-verify.sh rog-nv-linux codex/my-branch \
#     --expect-lib cudajit --test @arrayjit/runtest-test_cuda_arch_flags \
#     --run 'dune build tools/fp8_soak.exe -j 4 && \
#       _build/default/tools/fp8_soak.exe --arm=cuda --sweep=f32'
#   tools/remote-verify.sh minix-amd-linux codex/my-branch \
#     --expect-lib hipjit \
#     --record-golden @test/training/train-transformer_names
#
# The output is deliberately unpiped. A failed dune diff must be dune's status,
# not tail's or tee's. The far side prints an exit sentinel only after removing
# the temporary worktree; the local side then prints ssh's status.
#
# `--run` is intentionally a shell command: device probes often need several
# build/run arguments. It is executed by `opam exec -- sh -c` from the pinned
# worktree with OCANNL_BACKEND exported. The command must itself be an OCANNL
# probe whose output demonstrates the device/backend property being checked;
# this harness proves its source, configuration and exit status, but cannot turn
# an arbitrary command into backend evidence.

set -u

die() { echo "remote-verify: $*" >&2; exit 2; }

usage() {
  sed -n '2,/^# The output/s/^# \{0,1\}//p' "$0" >&2
  exit 2
}

sq() { printf "'%s'" "$(printf %s "$1" | sed "s/'/'\\\\''/g")"; }

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
ssh_cap=21600
jobs=4
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
      [ $# -ge 2 ] || die "--expect-lib needs cudajit or hipjit"
      expect_lib=$2
      shift 2
      ;;
    --test | --run | --record-golden)
      [ $# -ge 2 ] || die "$1 needs a value"
      operations+=("${1#--}" "$2")
      operation_count=$((operation_count + 2))
      shift 2
      ;;
    --repo)
      [ $# -ge 2 ] || die "--repo needs an absolute remote path"
      remote_repo=$2
      shift 2
      ;;
    --remote)
      [ $# -ge 2 ] || die "--remote needs a name"
      staging_remote=$2
      shift 2
      ;;
    --worktree-root)
      [ $# -ge 2 ] || die "--worktree-root needs an absolute remote path"
      worktree_root=$2
      shift 2
      ;;
    --cap)
      [ $# -ge 2 ] || die "--cap needs a value"
      cap=$2
      shift 2
      ;;
    --ssh-cap)
      [ $# -ge 2 ] || die "--ssh-cap needs a value"
      ssh_cap=$2
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
case $ssh_cap in '' | *[!0-9]*) die "ssh cap must be a non-negative integer" ;; esac
case $staging_remote in -*) die "--remote must not begin with '-'" ;; esac
case $backend in
  '' | cc | multidev_cc | cuda | hip | metal) ;;
  *) die "unknown backend '$backend'; expected cc, multidev_cc, cuda, hip, or metal" ;;
esac
case $expect_lib in
  '') ;;
  cudajit)
    [ -z "$backend" ] || [ "$backend" = cuda ] ||
      die "--expect-lib cudajit conflicts with --backend $backend"
    backend=cuda
    ;;
  hipjit)
    [ -z "$backend" ] || [ "$backend" = hip ] ||
      die "--expect-lib hipjit conflicts with --backend $backend"
    backend=hip
    ;;
  *) die "unknown optional library '$expect_lib'; expected cudajit or hipjit" ;;
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

case $remote_repo in '' | /*) ;; *) die "--repo must be an absolute path on the remote" ;; esac
case $worktree_root in '' | /*) ;; *) die "--worktree-root must be an absolute path on the remote" ;; esac

# One process-group supervisor is used on both sides of SSH. The local instance
# bounds setup plus cleanup; the same source is passed as a positional argument
# to the remote shell and bounds each build/test/probe there. Exit 142 means the
# wall-clock cap expired.
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

# ssh concatenates its remote argv into shell text. Quote every value once here,
# then let /bin/sh recover the exact positional arguments. The outer shell moves
# the verifier program to fd 3 and replaces stdin with /dev/null before the inner
# shell reads that program; no child can consume the caller's remaining control
# flow. In particular, --run commands are never interpolated into this string.
remote_command="/bin/sh -c 'exec 3<&0; exec </dev/null; exec /bin/sh /dev/fd/3 \"\$@\"' remote-verify"
for arg in "$box" "$branch" "$backend" "$expect_lib" "$remote_repo" "$staging_remote" \
  "$worktree_root" "$cap" "$ssh_cap" "$jobs" "$capped_perl"; do
  remote_command="$remote_command $(sq "$arg")"
done
if [ "$operation_count" -gt 0 ]; then
  for arg in "${operations[@]}"; do
    remote_command="$remote_command $(sq "$arg")"
  done
fi

local_capped "$ssh_cap" ssh -o BatchMode=yes -o ConnectTimeout=8 \
  -o ServerAliveInterval=30 -o ServerAliveCountMax=10 \
  "$box" "$remote_command" <<'REMOTE_VERIFY'
set -u

requested_box=$1
branch=$2
backend=$3
expect_lib=$4
repo_arg=$5
staging_remote_arg=$6
worktree_root_arg=$7
cap=$8
ssh_cap=$9
jobs=${10}
capped_perl=${11}
shift 11

# Non-login SSH shells on rog need both locations; harmless when the
# directories do not exist (tools/sweep.sh uses the same prefix).
PATH=/usr/local/cuda/bin:/usr/lib/wsl/lib:$PATH
export PATH

repo=${repo_arg:-$HOME/ocannl-staging}
worktree_root=${worktree_root_arg:-$HOME/ocannl-staging-worktrees}
wt=
wt_registered=0
finished=0

fail() {
  echo "remote-verify: $*" >&2
  exit 2
}

finish() {
  main_rc=$1
  [ "$finished" -eq 0 ] || exit "$main_rc"
  finished=1
  trap - EXIT HUP INT TERM
  cleanup_rc=0

  if [ -n "$wt" ]; then
    if [ "$wt_registered" -eq 1 ]; then
      capped git -C "$repo" worktree remove --force "$wt" || cleanup_rc=1
    elif [ -d "$wt" ]; then
      rmdir "$wt" 2>/dev/null || cleanup_rc=1
    fi
    [ ! -e "$wt" ] || cleanup_rc=1
  fi

  if [ "$cleanup_rc" -eq 0 ]; then
    echo "remote-verify: cleanup: PASS${wt:+ ($wt removed)}"
  else
    echo "remote-verify: cleanup: FAIL ($wt may need manual removal)" >&2
    [ "$main_rc" -ne 0 ] || main_rc=125
  fi
  echo "remote-verify: exit: $main_rc"
  exit "$main_rc"
}

trap 'finish $?' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM HUP

# This is the exact supervisor source the local side passed in; keeping one
# copy prevents the outer SSH and inner command caps from drifting apart.
capped() { perl -e "$capped_perl" -- "$cap" "$@"; }

# Configuration from the SSH service or the selected switch must not outrank
# the pushed tree's ocannl_config. Print names for provenance, never values;
# later commands inject only the backend named on this invocation.
ambient_ocannl_names=$(env | sed -n 's/^\(OCANNL_[A-Za-z0-9_]*\)=.*/\1/p') ||
  fail "cannot inspect ambient OCANNL configuration"
if [ -n "$ambient_ocannl_names" ]; then
  echo "remote-verify: clearing ambient OCANNL variables:"
  old_ifs=$IFS
  IFS='
'
  for name in $ambient_ocannl_names; do
    echo "  $name"
    unset "$name" || fail "cannot clear ambient variable $name"
  done
  IFS=$old_ifs
fi
remaining_ocannl_names=$(env | sed -n 's/^\(OCANNL_[A-Za-z0-9_]*\)=.*/\1/p') ||
  fail "cannot verify ambient OCANNL configuration cleanup"
[ -z "$remaining_ocannl_names" ] ||
  fail "ambient OCANNL variables remain after cleanup: $remaining_ocannl_names"
echo "remote-verify: ambient OCANNL configuration: cleared"

staging_url_matches() {
  case $1 in
    https://github.com/lukstafi/ocannl-staging | \
      https://github.com/lukstafi/ocannl-staging.git | \
      git@github.com:lukstafi/ocannl-staging | \
      git@github.com:lukstafi/ocannl-staging.git | \
      ssh://git@github.com/lukstafi/ocannl-staging | \
      ssh://git@github.com/lukstafi/ocannl-staging.git | \
      git://github.com/lukstafi/ocannl-staging | \
      git://github.com/lukstafi/ocannl-staging.git) return 0 ;;
    *) return 1 ;;
  esac
}

git -C "$repo" rev-parse --git-dir >/dev/null 2>&1 || fail "no git repository at $repo"
git check-ref-format --branch "$branch" >/dev/null 2>&1 || fail "invalid branch name: $branch"
mkdir -p "$worktree_root" || fail "cannot create worktree root $worktree_root"
repo=$(cd "$repo" && pwd -P) || fail "cannot resolve repository path $repo"
worktree_root=$(cd "$worktree_root" && pwd -P) ||
  fail "cannot resolve worktree root $worktree_root"

# Dune roots at the outermost ancestor holding dune-workspace/dune-project. A
# nested worktree would therefore report this commit while building its parent.
# Start at the worktree's future parent; the detached tree's own dune-project is
# expected and is below every directory inspected here.
ancestor=$worktree_root
while :; do
  if [ -e "$ancestor/dune-workspace" ] || [ -e "$ancestor/dune-project" ]; then
    fail "worktree root $worktree_root is nested under Dune root $ancestor"
  fi
  [ "$ancestor" = / ] && break
  ancestor=$(dirname "$ancestor")
done

if [ -n "$staging_remote_arg" ]; then
  staging_remote=$staging_remote_arg
  staging_url=$(git -C "$repo" remote get-url "$staging_remote" 2>/dev/null) ||
    fail "remote $staging_remote does not exist in $repo"
  staging_url_matches "$staging_url" ||
    fail "remote $staging_remote does not point to lukstafi/ocannl-staging: $staging_url"
else
  staging_remote=
  staging_url=
  for candidate in $(git -C "$repo" remote); do
    candidate_url=$(git -C "$repo" remote get-url "$candidate" 2>/dev/null) || continue
    if staging_url_matches "$candidate_url"; then
      [ -z "$staging_remote" ] ||
        fail "multiple remotes point to lukstafi/ocannl-staging; choose one with --remote"
      staging_remote=$candidate
      staging_url=$candidate_url
    fi
  done
  [ -n "$staging_remote" ] ||
    fail "no remote in $repo points to lukstafi/ocannl-staging (use --remote after adding one)"
fi

opam_switch=$(cd "$repo" && capped opam switch show --safe) ||
  fail "cannot resolve the opam switch selected by $repo"
[ -n "$opam_switch" ] || fail "the remote checkout has no selected opam switch"
switch_environment=$(capped opam exec --switch="$opam_switch" -- env) ||
  fail "cannot inspect the selected opam switch environment"
switch_ocannl_names=$(printf '%s\n' "$switch_environment" |
  sed -n 's/^\(OCANNL_[A-Za-z0-9_]*\)=.*/\1/p') ||
  fail "cannot identify OCANNL configuration from the selected opam switch"
if [ -n "$switch_ocannl_names" ]; then
  echo "remote-verify: stripping OCANNL variables injected by opam switch $opam_switch:"
  old_ifs=$IFS
  IFS='
'
  for name in $switch_ocannl_names; do echo "  $name"; done
  IFS=$old_ifs
fi

opam_exec() {
  old_ifs=$IFS
  IFS='
'
  for name in $switch_ocannl_names; do
    set -- -u "$name" "$@"
  done
  IFS=$old_ifs
  capped opam exec --switch="$opam_switch" -- env "$@"
}

sanitized_switch_environment=$(opam_exec env) ||
  fail "cannot verify the sanitized opam switch environment"
remaining_switch_ocannl_names=$(printf '%s\n' "$sanitized_switch_environment" |
  sed -n 's/^\(OCANNL_[A-Za-z0-9_]*\)=.*/\1/p') ||
  fail "cannot inspect the sanitized opam switch environment"
[ -z "$remaining_switch_ocannl_names" ] ||
  fail "opam switch OCANNL variables remain after sanitization: $remaining_switch_ocannl_names"
echo "remote-verify: opam switch OCANNL configuration: stripped"

actual_box=$(hostname 2>/dev/null || uname -n)
echo "=== remote-verify provenance ==="
echo "requested box: $requested_box"
echo "actual box:    $actual_box"
echo "repository:    $repo"
echo "staging remote: $staging_remote ($staging_url)"
echo "opam switch:    $opam_switch (resolved from the checkout)"
echo "pushed branch: $branch"
echo "requested backend: ${backend:-none (@check compiles only)}"
echo "expected optional library: ${expect_lib:-none}"
echo "dune jobs:     $jobs"
echo "per-command cap: ${cap}s"
echo "whole SSH cap: ${ssh_cap}s"
echo "remote PATH prefix: /usr/local/cuda/bin:/usr/lib/wsl/lib"

# Fetch the named pushed branch explicitly. Resolving an already-present remote
# tracking ref after a failed fetch would certify stale source.
capped git -C "$repo" fetch -q --no-write-fetch-head "$staging_remote" \
  "+refs/heads/$branch:refs/remotes/$staging_remote/$branch" ||
  fail "cannot fetch pushed branch $staging_remote/$branch"
full_sha=$(git -C "$repo" rev-parse --verify "refs/remotes/$staging_remote/$branch^{commit}") ||
  fail "cannot resolve $staging_remote/$branch to a commit"
echo "resolved commit: $full_sha"

wt=$(mktemp -d "$worktree_root/remote-verify.XXXXXX") ||
  fail "cannot allocate a temporary worktree path"
rmdir "$wt" || fail "cannot prepare temporary worktree path $wt"
capped git -C "$repo" worktree add -q --detach "$wt" "$full_sha" ||
  fail "cannot create detached worktree at $wt"
wt_registered=1

actual_sha=$(git -C "$wt" rev-parse HEAD) || fail "cannot read worktree HEAD"
[ "$actual_sha" = "$full_sha" ] ||
  fail "worktree commit $actual_sha differs from resolved commit $full_sha"
worktree_status=$(git -C "$wt" status --porcelain --untracked-files=all) ||
  fail "cannot read fresh worktree status"
[ -z "$worktree_status" ] || fail "fresh worktree is not clean"
if [ -L "$wt/ocannl_config" ]; then
  fail "pushed root ocannl_config must not be a symlink"
elif [ -e "$wt/ocannl_config" ]; then
  [ -f "$wt/ocannl_config" ] || fail "pushed root ocannl_config is not a regular file"
  config_boundary_kind=pushed
  config_boundary="$wt/ocannl_config (from pushed commit)"
else
  touch "$wt/ocannl_config" || fail "cannot create an empty root ocannl_config boundary"
  config_boundary_kind=empty
  config_boundary="$wt/ocannl_config (empty boundary created by remote-verify)"
fi
echo "worktree:      $wt"
echo "worktree HEAD: $actual_sha"
echo "source state:  clean, detached, exact commit"
echo "config boundary: $config_boundary"
echo "=== end provenance ==="

cd "$wt" || fail "cannot enter $wt"

assert_source_state() {
  state_context=$1
  observed_sha=$(git rev-parse HEAD) || fail "cannot read HEAD $state_context"
  [ "$observed_sha" = "$full_sha" ] ||
    fail "HEAD changed to $observed_sha $state_context (expected $full_sha)"
  observed_status=$(git status --porcelain --untracked-files=all) ||
    fail "cannot read worktree status $state_context"
  [ -z "$observed_status" ] || fail "source changed $state_context: $observed_status"
  [ -f ocannl_config ] && [ ! -L ocannl_config ] ||
    fail "root ocannl_config boundary changed type $state_context"
  if [ "$config_boundary_kind" = empty ]; then
    [ ! -s ocannl_config ] || fail "empty root ocannl_config boundary was rewritten $state_context"
  fi
  echo "remote-verify: source assertion: PASS ($state_context; commit=$full_sha; clean; boundary=$config_boundary_kind)"
}

assert_only_promoted_goldens() {
  golden_context=$1
  status_file=_build/.remote-verify-status.$$
  git status --porcelain=v1 -z --untracked-files=all >"$status_file" ||
    fail "cannot inspect source changes $golden_context"
  set --
  old_ifs=$IFS
  IFS='
'
  for promoted in $promotions; do
    [ -n "$promoted" ] && set -- "$@" "$promoted"
  done
  IFS=$old_ifs
  perl -0 -e '
    my %allowed = map { $_ => 1 } @ARGV;
    my (%seen, $ok);
    $ok = 1;
    while (defined(my $entry = <STDIN>)) {
      chomp $entry;
      next if $entry eq "";
      if (length($entry) < 4) {
        print STDERR "remote-verify: malformed git status entry during golden recording\n";
        $ok = 0;
        next;
      }
      my $code = substr($entry, 0, 2);
      my $path = substr($entry, 3);
      if ($code =~ /[RC]/ || !$allowed{$path}) {
        print STDERR "remote-verify: non-golden source change during golden recording: $code $path\n";
        $ok = 0;
      } else {
        $seen{$path} = 1;
      }
    }
    for my $path (keys %allowed) {
      if (!$seen{$path}) {
        print STDERR "remote-verify: listed golden has no source change after apply: $path\n";
        $ok = 0;
      }
    }
    exit($ok ? 0 : 1);
  ' -- "$@" <"$status_file"
  status_rc=$?
  rm -f "$status_file" || fail "cannot remove temporary golden status"
  [ "$status_rc" -eq 0 ] || fail "source changes are not limited to listed goldens $golden_context"
  echo "remote-verify: golden source scope: PASS ($golden_context)"
}

dune_build() {
  if [ -n "$backend" ]; then
    opam_exec env "OCANNL_BACKEND=$backend" dune build -j "$jobs" "$@"
  else
    opam_exec dune build -j "$jobs" "$@"
  fi
}

assert_backend() {
  [ -n "$backend" ] || return 0
  dune_build test/config/ocannl_backend.txt ||
    fail "cannot resolve the configured backend"
  resolved_backend=$(cat _build/default/test/config/ocannl_backend.txt) ||
    fail "cannot read the resolved backend artifact"
  [ "$resolved_backend" = "$backend" ] ||
    fail "requested backend $backend resolved as $resolved_backend"
  echo "remote-verify: backend evidence: requested=$backend resolved=$resolved_backend"
}

assert_optional_library() {
  [ -n "$expect_lib" ] || return 0
  case $expect_lib in
    cudajit)
      impl=cuda
      other=hip
      arm=cudajit
      ;;
    hipjit)
      impl=hip
      other=cuda
      arm=hipjit
      ;;
  esac
  cmi="_build/default/arrayjit/lib/.$impl"_backend.objs/byte/"$impl"_backend.cmi
  other_cmi="_build/default/arrayjit/lib/.$other"_backend.objs/byte/"$other"_backend.cmi
  selected="_build/default/arrayjit/lib/$impl"_backend_impl.ml
  [ -f "$cmi" ] || fail "$expect_lib evidence missing: $cmi"
  [ ! -e "$other_cmi" ] ||
    fail "negative control failed: opposite backend artifact exists at $other_cmi"
  [ -f "$selected" ] || fail "select-arm evidence missing: $selected"
  first_line=$(sed -n '1p' "$selected") || fail "cannot read $selected"
  case $first_line in
    *"${impl}_backend_impl.${arm}.ml"*) ;;
    *) fail "$selected selected the wrong arm: $first_line" ;;
  esac
  echo "remote-verify: optional-library evidence: PASS $cmi"
  echo "remote-verify: select-arm evidence: PASS $first_line"
  echo "remote-verify: opposite-backend negative control: PASS $other_cmi absent"
}

echo "remote-verify: build: opam exec --switch=$opam_switch -- dune build -j $jobs @check"
dune_build @check || exit $?
echo "remote-verify: @check: PASS (compilation only; no backend execution claimed)"
assert_backend
assert_optional_library
assert_source_state "after @check and provenance checks"

while [ $# -gt 0 ]; do
  [ $# -ge 2 ] || fail "internal operation argument is incomplete"
  kind=$1
  value=$2
  shift 2
  case $kind in
    test)
      echo "remote-verify: test alias ($backend): $value"
      dune_build "$value" || exit $?
      assert_backend
      echo "remote-verify: test alias: PASS $value with resolved backend configuration $backend"
      echo "remote-verify: test alias backend execution: not claimed (the alias may be backend-independent)"
      ;;
    run)
      echo "remote-verify: probe ($backend): $value"
      opam_exec env "OCANNL_BACKEND=$backend" sh -c "$value" </dev/null || exit $?
      assert_backend
      echo "remote-verify: probe: PASS with resolved backend configuration $backend"
      echo "remote-verify: probe backend execution: see the probe's own output above"
      ;;
    record-golden)
      echo "remote-verify: record golden ($backend): $value"
      # Establish the configuration BEFORE the alias. Any later Dune invocation
      # clears the pending-promotion registry that this mode must consume.
      assert_backend
      dune_build "$value"
      build_rc=$?
      promotions=$(opam_exec dune promotion list --root .) ||
        fail "cannot list golden corrections after $value"
      if [ -z "$promotions" ]; then
        [ "$build_rc" -eq 0 ] || exit "$build_rc"
        assert_backend
        echo "remote-verify: golden: PASS, already current ($value; backend=$backend)"
      else
        echo "=== remote-verify corrected golden contents (backend=$backend) ==="
        old_ifs=$IFS
        IFS='
'
        for promoted in $promotions; do
          [ -n "$promoted" ] || continue
          case $promoted in
            *.expected | test/ppx/*_expected.ml) ;;
            *) fail "--record-golden produced a non-golden correction: $promoted" ;;
          esac
          echo "--- $promoted (.actual) ---"
          opam_exec dune promotion show --root . "$promoted" ||
            fail "cannot show corrected contents for $promoted"
        done
        IFS=$old_ifs
        echo "=== end corrected golden contents ==="
        opam_exec dune promotion apply --root . ||
          fail "cannot apply remote golden corrections"
        assert_only_promoted_goldens "after applying corrections for $value"
        git diff --quiet
        diff_rc=$?
        case $diff_rc in
          0) fail "dune listed corrections but applying them changed no source file" ;;
          1) ;;
          *) fail "cannot inspect the recorded golden changes" ;;
        esac
        git diff --check || fail "recorded golden patch has whitespace errors"
        echo "=== remote-verify apply-ready golden patch (commit $full_sha; backend=$backend) ==="
        git diff --no-ext-diff --binary -- '*.expected' 'test/ppx/*_expected.ml'
        echo "=== end apply-ready golden patch ==="
        echo "remote-verify: golden: correction recorded; re-running $value to retain all failures"
        dune_build "$value" || exit $?
        remaining=$(opam_exec dune promotion list --root .) ||
          fail "cannot check for corrections after re-running $value"
        [ -z "$remaining" ] || fail "$value still has promotable corrections after its re-run"
        assert_backend
        assert_only_promoted_goldens "after re-running $value"
        echo "remote-verify: golden: RECORDED and re-run PASS ($value; original dune exit=$build_rc)"
        git reset -q --hard "$full_sha" || fail "cannot restore source after recording $value"
        git clean -q -fd || fail "cannot remove untracked source after recording $value"
        assert_source_state "after restoring recorded golden $value"
      fi
      ;;
    *) fail "internal unknown operation: $kind" ;;
  esac
  assert_source_state "after $kind $value"
done

assert_source_state "before final certification"
echo "remote-verify: verified box=$actual_box commit=$full_sha backend=${backend:-not-run}"
exit 0
REMOTE_VERIFY
ssh_rc=$?
echo "remote-verify: ssh exit: $ssh_rc"
exit "$ssh_rc"
