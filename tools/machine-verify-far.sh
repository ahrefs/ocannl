#!/bin/sh
# The verification procedure of tools/machine-verify.sh: the ONE definition of
# what runs on the verified machine (gh-ocannl-1047). Never run it directly.
# machine-verify.sh reads this file and feeds it, unchanged and through the same
# quoted argument vector, either to `ssh BOX` (another machine) or to a local
# `/bin/sh` in a cleared environment (BOX is this machine, where self-ssh fails
# on the native fleet boxes). There is no second copy for the local path to
# drift from: a check added here runs on both transports.
#
# POSIX sh, and no GNU-only tool: it runs under dash on Ubuntu and under the
# macOS /bin/sh on mac-studio. Wall-clock caps come from the perl supervisor the
# caller passes in, never timeout(1), which macOS does not ship.
set -u

[ $# -ge 12 ] || {
  echo "machine-verify-far.sh: run tools/machine-verify.sh, which supplies this procedure's arguments" >&2
  exit 2
}
requested_box=$1
branch=$2
backend=$3
expect_lib=$4
repo_arg=$5
staging_remote_arg=$6
worktree_root_arg=$7
cap=$8
trip_cap=$9
jobs=${10}
transport=${11}
capped_perl=${12}
shift 12

# Non-login SSH shells on rog need both locations; harmless when the
# directories do not exist (tools/sweep.sh uses the same prefix), as on macOS.
PATH=/usr/local/cuda/bin:/usr/lib/wsl/lib:$PATH
export PATH

repo=${repo_arg:-$HOME/ocannl-staging}
worktree_root=${worktree_root_arg:-$HOME/ocannl-staging-worktrees}
wt=
wt_registered=0
finished=0

fail() {
  echo "machine-verify: $*" >&2
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
    echo "machine-verify: cleanup: PASS${wt:+ ($wt removed)}"
  else
    echo "machine-verify: cleanup: FAIL ($wt may need manual removal)" >&2
    [ "$main_rc" -ne 0 ] || main_rc=125
  fi
  echo "machine-verify: exit: $main_rc"
  exit "$main_rc"
}

trap 'finish $?' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM HUP

# This is the exact supervisor source the local side passed in; keeping one
# copy prevents the outer whole-trip and inner command caps from drifting apart.
capped() { perl -e "$capped_perl" -- "$cap" "$@"; }

# Configuration from the SSH service, the caller's session or the selected switch must not outrank
# the pushed tree's ocannl_config. Print names for provenance, never values;
# later commands inject only the backend named on this invocation.
ambient_ocannl_names=$(env | sed -n 's/^\(OCANNL_[A-Za-z0-9_]*\)=.*/\1/p') ||
  fail "cannot inspect ambient OCANNL configuration"
if [ -n "$ambient_ocannl_names" ]; then
  echo "machine-verify: clearing ambient OCANNL variables:"
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
echo "machine-verify: ambient OCANNL configuration: cleared"

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
[ -n "$opam_switch" ] || fail "the checkout $repo has no selected opam switch"
switch_environment=$(capped opam exec --switch="$opam_switch" -- env) ||
  fail "cannot inspect the selected opam switch environment"
switch_ocannl_names=$(printf '%s\n' "$switch_environment" |
  sed -n 's/^\(OCANNL_[A-Za-z0-9_]*\)=.*/\1/p') ||
  fail "cannot identify OCANNL configuration from the selected opam switch"
if [ -n "$switch_ocannl_names" ]; then
  echo "machine-verify: stripping OCANNL variables injected by opam switch $opam_switch:"
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
echo "machine-verify: opam switch OCANNL configuration: stripped"

actual_box=$(hostname 2>/dev/null || uname -n)
echo "=== machine-verify provenance ==="
echo "requested box: $requested_box"
echo "actual box:    $actual_box"
echo "transport:     $transport"
echo "repository:    $repo"
echo "staging remote: $staging_remote ($staging_url)"
echo "opam switch:    $opam_switch (resolved from the checkout)"
echo "pushed branch: $branch"
echo "requested backend: ${backend:-none (@check compiles only)}"
echo "expected optional library: ${expect_lib:-none}"
echo "dune jobs:     $jobs"
echo "per-command cap: ${cap}s"
echo "whole-trip cap: ${trip_cap}s"
echo "PATH prefix:   /usr/local/cuda/bin:/usr/lib/wsl/lib"

# Fetch the named pushed branch explicitly. Resolving an already-present remote
# tracking ref after a failed fetch would certify stale source.
capped git -C "$repo" fetch -q --no-write-fetch-head "$staging_remote" \
  "+refs/heads/$branch:refs/remotes/$staging_remote/$branch" ||
  fail "cannot fetch pushed branch $staging_remote/$branch"
full_sha=$(git -C "$repo" rev-parse --verify "refs/remotes/$staging_remote/$branch^{commit}") ||
  fail "cannot resolve $staging_remote/$branch to a commit"
echo "resolved commit: $full_sha"

wt=$(mktemp -d "$worktree_root/machine-verify.XXXXXX") ||
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
  config_boundary="$wt/ocannl_config (empty boundary created by machine-verify)"
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
  echo "machine-verify: source assertion: PASS ($state_context; commit=$full_sha; clean; boundary=$config_boundary_kind)"
}

assert_only_promoted_goldens() {
  golden_context=$1
  status_file=_build/.machine-verify-status.$$
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
        print STDERR "machine-verify: malformed git status entry during golden recording\n";
        $ok = 0;
        next;
      }
      my $code = substr($entry, 0, 2);
      my $path = substr($entry, 3);
      if ($code =~ /[RC]/ || !$allowed{$path}) {
        print STDERR "machine-verify: non-golden source change during golden recording: $code $path\n";
        $ok = 0;
      } else {
        $seen{$path} = 1;
      }
    }
    for my $path (keys %allowed) {
      if (!$seen{$path}) {
        print STDERR "machine-verify: listed golden has no source change after apply: $path\n";
        $ok = 0;
      }
    }
    exit($ok ? 0 : 1);
  ' -- "$@" <"$status_file"
  status_rc=$?
  rm -f "$status_file" || fail "cannot remove temporary golden status"
  [ "$status_rc" -eq 0 ] || fail "source changes are not limited to listed goldens $golden_context"
  echo "machine-verify: golden source scope: PASS ($golden_context)"
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
  echo "machine-verify: backend evidence: requested=$backend resolved=$resolved_backend"
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
  echo "machine-verify: optional-library evidence: PASS $cmi"
  echo "machine-verify: select-arm evidence: PASS $first_line"
  echo "machine-verify: opposite-backend negative control: PASS $other_cmi absent"
}

echo "machine-verify: build: opam exec --switch=$opam_switch -- dune build -j $jobs @check"
dune_build @check || exit $?
echo "machine-verify: @check: PASS (compilation only; no backend execution claimed)"
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
      echo "machine-verify: test alias ($backend): $value"
      dune_build "$value" || exit $?
      assert_backend
      echo "machine-verify: test alias: PASS $value with resolved backend configuration $backend"
      echo "machine-verify: test alias backend execution: not claimed (the alias may be backend-independent)"
      ;;
    run)
      echo "machine-verify: probe ($backend): $value"
      opam_exec env "OCANNL_BACKEND=$backend" sh -c "$value" </dev/null || exit $?
      assert_backend
      echo "machine-verify: probe: PASS with resolved backend configuration $backend"
      echo "machine-verify: probe backend execution: see the probe's own output above"
      ;;
    record-golden)
      echo "machine-verify: record golden ($backend): $value"
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
        echo "machine-verify: golden: PASS, already current ($value; backend=$backend)"
      else
        echo "=== machine-verify corrected golden contents (backend=$backend) ==="
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
          fail "cannot apply golden corrections"
        assert_only_promoted_goldens "after applying corrections for $value"
        git diff --quiet
        diff_rc=$?
        case $diff_rc in
          0) fail "dune listed corrections but applying them changed no source file" ;;
          1) ;;
          *) fail "cannot inspect the recorded golden changes" ;;
        esac
        git diff --check || fail "recorded golden patch has whitespace errors"
        echo "=== machine-verify apply-ready golden patch (commit $full_sha; backend=$backend) ==="
        git diff --no-ext-diff --binary -- '*.expected' 'test/ppx/*_expected.ml'
        echo "=== end apply-ready golden patch ==="
        echo "machine-verify: golden: correction recorded; re-running $value to retain all failures"
        dune_build "$value" || exit $?
        remaining=$(opam_exec dune promotion list --root .) ||
          fail "cannot check for corrections after re-running $value"
        [ -z "$remaining" ] || fail "$value still has promotable corrections after its re-run"
        assert_backend
        assert_only_promoted_goldens "after re-running $value"
        echo "machine-verify: golden: RECORDED and re-run PASS ($value; original dune exit=$build_rc)"
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
echo "machine-verify: verified box=$actual_box commit=$full_sha backend=${backend:-not-run}"
exit 0
