#!/usr/bin/env bash
# Run one named test through the compiler version that ubuntu-latest exposed in
# the gh-ocannl-752 CI failure, without installing that compiler system-wide.
#
# Usage:
#   tools/ci-compiler-test.sh [OPTIONS] @DIR/runtest-NAME
#   tools/ci-compiler-test.sh [OPTIONS] --pieces census
#
# Options:
#   --aarch64-clang   Also fetch clang 21 and the arm64 cross headers, then set
#                     AARCH64_CROSS_GCC to clang --target=aarch64-linux-gnu
#                     with Apple's NEON assembly syntax. The named test must
#                     invoke this cross compiler as well as the host compiler.
#   --dry-run         Validate arguments, create the scratch layout, and print
#                     every fetch/extract/test step without fetching or running.
#   --toolchain DIR   Keep the GCC 13 in DIR (absolute) rather than in scratch:
#                     staged there on first use, reused unfetched afterwards
#                     (gh-ocannl-1696; minix keeps one at
#                     /home/lukstafi/.local/opt/x86_64-linux-gnu-gcc-13).
#   --keep            Keep the scratch directory even on success (a failing
#                     run always keeps it, so the build tree's diagnostics --
#                     e.g. cc_march_census_kernels/<piece>/rows.txt -- survive it;
#                     otherwise cleanup precedes the exit sentinel).
#   --pieces census   Instead of one alias, run every cc_march_census piece
#                     (gh-ocannl-1852), one after another against the one staged
#                     toolchain and build directory, and print one verdict per
#                     piece; the run fails if any piece does. The pieces are the
#                     ones test/operations/dune runs, read from its rules (the
#                     census's emission piece holds that list to its own).
#   --cap SECONDS     Named-test wall-clock cap; 0 disables it (default: 3600).
#                     Under --pieces it caps each piece.
#   -j, --jobs N      Dune concurrency, 1..4 (default: 4).
#
# Examples:
#   tools/ci-compiler-test.sh @test/operations/runtest-cpu_simd_reduction
#   tools/ci-compiler-test.sh --aarch64-clang \
#     @test/operations/runtest-cc_march_census-w16_native_foreign
#   tools/ci-compiler-test.sh --dry-run --aarch64-clang \
#     @test/operations/runtest-cc_march_census-w16_native_foreign
#   tools/ci-compiler-test.sh --aarch64-clang --toolchain DIR --pieces census
#
# The real fetch is Linux/Debian-family only: it deliberately uses the
# no-root recipe established by gh-ocannl-752, `apt-get download` followed by
# `dpkg-deb -x` into a temporary prefix. The GCC 13 packages come from an
# isolated scratch index over Ubuntu 24.04 (noble), the series ubuntu-latest
# runs, so a newer host's own gcc-13 (resolute's 13.4) cannot stand in for
# CI's 13.3 (gh-ocannl-1696). Each isolated `apt-get update` is capped the way
# CI's are (gh-ocannl-1809): `timeout` (240s, apt's own retries and
# per-connection timeouts inside it), one more fresh try, then a failure naming
# apt -- unbounded, a stalled mirror hung a local leg (gh-ocannl-1852). The
# downloads carry the same retries and per-connection timeouts, which end a
# stalled transfer without capping a slow but moving one. --dry-run is portable and is the
# supported way to inspect and validate the staging plan elsewhere. The real
# host must be x86_64 because the fetched compiler produces executable x86_64
# cc kernels, not merely cross-compiled assembly.
#
# Output is deliberately unpiped. Dune's exit status remains the verdict. The
# compiler wrappers log every invocation, so a cached build or an alias that
# never reaches the requested compiler fails rather than being certified.

set -u
# shellcheck source=../scripts/process-group.sh
. "$(dirname -- "${BASH_SOURCE[0]}")/../scripts/process-group.sh"

die() { echo "ci-compiler-test: $*" >&2; exit 2; }

usage() {
  sed -n '2,/^# Output/s/^# \{0,1\}//p' "$0" >&2
  exit 2
}

shell_quote() {
  printf "'%s'" "$(printf %s "$1" | sed "s/'/'\\\\''/g")"
}

print_command() {
  printf '  +'
  for word in "$@"; do
    printf ' %s' "$(shell_quote "$word")"
  done
  printf '\n'
}

dry_run=0
keep=0
aarch64_clang=0
jobs=4
cap=3600
alias_name=
toolchain=
pieces=

while [ $# -gt 0 ]; do
  case $1 in
    --aarch64-clang)
      aarch64_clang=1
      shift
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    --toolchain)
      [ $# -ge 2 ] || die "--toolchain needs a directory"
      toolchain=$2
      shift 2
      ;;
    --keep)
      keep=1
      shift
      ;;
    --pieces)
      [ $# -ge 2 ] || die "--pieces needs a piece set (census)"
      pieces=$2
      shift 2
      ;;
    --cap)
      [ $# -ge 2 ] || die "--cap needs a value"
      cap=$2
      shift 2
      ;;
    -j | --jobs)
      [ $# -ge 2 ] || die "$1 needs a value"
      jobs=$2
      shift 2
      ;;
    -h | --help) usage ;;
    -*) die "unknown argument: $1" ;;
    *)
      [ -z "$alias_name" ] || die "expected exactly one named test alias"
      alias_name=$1
      shift
      ;;
  esac
done

case $jobs in 1 | 2 | 3 | 4) ;; *) die "jobs must be between 1 and 4" ;; esac
case $cap in '' | *[!0-9]*) die "cap must be a non-negative integer" ;; esac
case $toolchain in
  '' | /*) ;;
  *) die "--toolchain must be an absolute directory: $toolchain" ;;
esac
case $pieces in
  '') ;;
  census) [ -z "$alias_name" ] || die "--pieces census runs its own aliases; name no test beside it" ;;
  *) die "unknown piece set: $pieces (the one set is census)" ;;
esac
if [ -z "$pieces" ]; then
  case $alias_name in
    @*/runtest-?*) ;;
    '') usage ;;
    *) die "test must be a named Dune alias of the form @DIR/runtest-NAME" ;;
  esac
fi

script_dir=$(cd "$(dirname "$0")" && pwd -P) || die "cannot resolve tools directory"
repo=$(git -C "$script_dir/.." rev-parse --show-toplevel 2>/dev/null) ||
  die "script is not inside a git checkout"
repo=$(cd "$repo" && pwd -P) || die "cannot resolve repository root"
script_path="$repo/tools/ci-compiler-test.sh"
[ -f "$script_path" ] || die "cannot find the script at $script_path"

# The census pieces, read from the dune rules that run them: the word after each
# `%{dep:cc_march_census.exe}`, the reading cc_march_census.ml's dune_pieces
# makes, so the emission piece holds this list to the census's own. Each must
# be a one-line rule argument with an alias of its own, or nothing is run.
aliases=()
if [ "$pieces" = census ]; then
  census_dune="$repo/test/operations/dune"
  [ -f "$census_dune" ] || die "cannot find the census's dune file at $census_dune"
  census_rules=$(grep -c '%{dep:cc_march_census\.exe}' "$census_dune")
  census_pieces=$(sed -n 's/.*%{dep:cc_march_census\.exe}[[:space:]]\{1,\}\([a-z0-9_-]\{1,\}\).*/\1/p' \
    "$census_dune")
  set -- $census_pieces
  [ "$#" -gt 0 ] || die "found no cc_march_census piece in $census_dune"
  [ "$#" -eq "$census_rules" ] ||
    die "read $# census pieces from $census_rules lines running cc_march_census.exe in $census_dune"
  for piece in "$@"; do
    grep -q "^ *(alias runtest-cc_march_census-$piece)\$" "$census_dune" ||
      die "census piece $piece has no runtest-cc_march_census-$piece alias in $census_dune"
    aliases+=("@test/operations/runtest-cc_march_census-$piece")
  done
else
  aliases=("$alias_name")
fi

scratch=$(mktemp -d "${TMPDIR:-/tmp}/ocannl-ci-compiler.XXXXXX" 2>/dev/null) || scratch=
[ -n "$scratch" ] && [ -d "$scratch" ] || die "cannot create a scratch directory"

finished=0
active_child=
finish() {
  main_rc=$1
  [ "$finished" -eq 0 ] || exit "$main_rc"
  finished=1
  trap - EXIT HUP INT TERM
  cleanup_rc=0
  if [ "$keep" -eq 1 ] || [ "$main_rc" -ne 0 ]; then
    echo "ci-compiler-test: scratch kept: $scratch"
  elif [ -n "$scratch" ] && [ -d "$scratch" ] && [ "$scratch" != / ]; then
    rm -rf -- "$scratch" || cleanup_rc=1
    if [ "$cleanup_rc" -eq 0 ]; then
      echo "ci-compiler-test: cleanup: PASS ($scratch removed)"
    else
      echo "ci-compiler-test: cleanup: FAIL ($scratch may need manual removal)" >&2
      [ "$main_rc" -ne 0 ] || main_rc=125
    fi
  fi
  echo "ci-compiler-test: exit: $main_rc"
  exit "$main_rc"
}
trap 'finish $?' EXIT

forward_signal() {
  signal_rc=$1
  trap - HUP INT TERM
  if [ -n "$active_child" ]; then
    kill -s TERM "$active_child" 2>/dev/null || true
    wait "$active_child" 2>/dev/null || true
    active_child=
  fi
  exit "$signal_rc"
}
trap 'forward_signal 130' INT
trap 'forward_signal 143' TERM HUP

# Every potentially long child is attached and reaped. This makes the signal
# traps above an ownership boundary: scratch cannot disappear while a fetch,
# extraction, or the capped test harness still has it open.
run_tracked() {
  "$@" &
  active_child=$!
  wait_child "$active_child"
  tracked_rc=$?
  active_child=
  return "$tracked_rc"
}

run_tracked_in() {
  tracked_dir=$1
  shift
  (cd "$tracked_dir" && exec "$@") &
  active_child=$!
  wait_child "$active_child"
  tracked_rc=$?
  active_child=
  return "$tracked_rc"
}

deb_dir="$scratch/debs"
prefix="$scratch/prefix"
build_dir="$scratch/dune-build"
wrapper_dir="$scratch/bin"
mkdir -p "$deb_dir" "$prefix" "$build_dir" "$wrapper_dir" ||
  die "cannot stage scratch directories"
# GCC 13's packages go to their own prefix: scratch's by default, the
# persistent --toolchain directory when one is named, which then holds GCC 13
# alone (the clang proxy's packages always stay in scratch).
gcc_prefix=${toolchain:-$prefix}
gcc_deb_dir="$scratch/gcc-debs"
gcc_suite=noble
gcc_keyring=/usr/share/keyrings/ubuntu-archive-keyring.gpg

gcc_packages=(gcc-13-x86-64-linux-gnu cpp-13-x86-64-linux-gnu libgcc-13-dev)
clang_packages=(clang-21 libclang-cpp21 libllvm21 libclang-common-21-dev)
arm64_packages=(libc6-dev-arm64-cross linux-libc-dev-arm64-cross)
# CI's apt bounds (gh-ocannl-1809): apt's retries and per-connection timeouts,
# on every update and download; an update also runs under `timeout`.
apt_bounds=(-o Acquire::Retries=3 -o Acquire::http::Timeout=30 -o Acquire::https::Timeout=30)

echo "=== ci-compiler-test plan ==="
echo "repository:       $repo"
echo "source commit:    $(git -C "$repo" rev-parse HEAD)"
if [ -z "$(git -C "$repo" status --porcelain --untracked-files=all)" ]; then
  echo "source state:     clean"
else
  echo "source state:     DIRTY (the working tree, not only the commit, will be tested)"
fi
if [ "$pieces" = census ]; then
  echo "test aliases:     ${#aliases[@]} cc_march_census pieces, one verdict each:"
  for alias in "${aliases[@]}"; do echo "                  $alias"; done
else
  echo "test alias:       $alias_name"
fi
echo "dune jobs:        $jobs"
echo "test cap:         ${cap}s$([ "$pieces" = census ] && echo ' per piece')"
echo "scratch:          $scratch"
echo "host toolchain:   GCC 13 x86_64-linux-gnu from Ubuntu $gcc_suite (Ubuntu CI proxy)"
echo "toolchain prefix: ${toolchain:-scratch (pass --toolchain DIR to keep it)}"
echo "real host:        x86_64 Linux with Debian-family apt indexes"
echo "test runner:      tools/test-run.sh from the resolved repository"
echo "environment:      clear OCANNL config, compiler selectors, and LD_* overrides"
echo "                  preserve OCANNL_TOOL_* harness controls"
if [ "$aarch64_clang" -eq 1 ]; then
  echo "cross toolchain:  clang 21 --target=aarch64-linux-gnu"
  echo "assembly dialect: Apple NEON (-mllvm -aarch64-neon-syntax=apple)"
else
  echo "cross toolchain:  not requested (use --aarch64-clang)"
fi
echo "mode:             $([ "$dry_run" -eq 1 ] && echo DRY RUN || echo execute)"
echo "=== end plan ==="

if [ "$dry_run" -eq 1 ]; then
  if [ -n "$toolchain" ] && [ -x "$toolchain/usr/bin/x86_64-linux-gnu-gcc-13" ]; then
    echo "ci-compiler-test: planned reuse of the staged GCC 13 in $toolchain (no download)"
  else
    echo "ci-compiler-test: planned package download (host compiler):"
    echo "  source: isolated Ubuntu $gcc_suite index signed by $gcc_keyring"
    print_command timeout --kill-after=15 240 apt-get -q '<isolated noble options>' \
      "${apt_bounds[@]}" update
    print_command apt-get '<isolated noble options>' "${apt_bounds[@]}" download "${gcc_packages[@]}"
    echo "ci-compiler-test: planned extraction:"
    print_command dpkg-deb -x '<each downloaded GCC .deb>' "$gcc_prefix"
  fi
  if [ "$aarch64_clang" -eq 1 ]; then
    echo "ci-compiler-test: planned package download (aarch64 clang proxy):"
    echo "  source: isolated https://apt.llvm.org/<VERSION_CODENAME>/ index with pinned signing key"
    print_command timeout --kill-after=15 240 apt-get '<isolated apt.llvm.org options>' \
      "${apt_bounds[@]}" update
    print_command apt-get '<isolated apt.llvm.org options>' "${apt_bounds[@]}" download \
      "${clang_packages[@]}"
    print_command apt-get "${apt_bounds[@]}" download "${arm64_packages[@]}"
    echo "ci-compiler-test: planned cross invocation:"
    echo "  loader path: derived from the extracted libLLVM.so.21 multiarch directory"
    print_command "$prefix/usr/bin/clang-21" --target=aarch64-linux-gnu \
      "--sysroot=$prefix" -isystem "$prefix/usr/aarch64-linux-gnu/include" \
      -mllvm -aarch64-neon-syntax=apple
  fi
  echo "ci-compiler-test: planned isolated test invocation$([ "${#aliases[@]}" -gt 1 ] && echo 's, one per piece'):"
  for alias in "${aliases[@]}"; do
    if [ "$aarch64_clang" -eq 1 ]; then
      print_command env OCANNL_BACKEND=cc \
        "OCANNL_CC_BACKEND_COMPILER_COMMAND=$(shell_quote "$wrapper_dir/gcc-13")" \
        "AARCH64_CROSS_GCC=$(shell_quote "$wrapper_dir/clang-aarch64")" \
        "DUNE_BUILD_DIR=$build_dir" "$repo/tools/test-run.sh" run --cap "$cap" \
        build -j "$jobs" "$alias"
    else
      print_command env OCANNL_BACKEND=cc \
        "OCANNL_CC_BACKEND_COMPILER_COMMAND=$(shell_quote "$wrapper_dir/gcc-13")" \
        "DUNE_BUILD_DIR=$build_dir" "$repo/tools/test-run.sh" run --cap "$cap" \
        build -j "$jobs" "$alias"
    fi
  done
  echo "ci-compiler-test: dry-run: PASS (arguments and scratch staging only)"
  exit 0
fi

[ "$(uname -s)" = Linux ] ||
  die "real toolchain fetch is Linux/Debian-family only (this host is $(uname -s)); use --dry-run here"
[ "$(uname -m)" = x86_64 ] ||
  die "real GCC 13 execution requires an x86_64 host (this host is $(uname -m)); use --dry-run here"
for command_name in apt-get dpkg-deb git opam sed timeout; do
  command -v "$command_name" >/dev/null 2>&1 || die "required command not found: $command_name"
done
if [ "$aarch64_clang" -eq 1 ]; then
  for command_name in curl sha256sum; do
    command -v "$command_name" >/dev/null 2>&1 || die "required command not found: $command_name"
  done
fi

download_packages() {
  label=$1
  shift
  echo "ci-compiler-test: apt download ($label):"
  print_command apt-get "${apt_bounds[@]}" download "$@"
  run_tracked_in "$deb_dir" apt-get "${apt_bounds[@]}" download "$@" ||
    die "apt-get could not download the $label packages"
}

# One isolated index refresh, capped and retried as CI's are (gh-ocannl-1809):
# each attempt is a fresh apt-get session under `timeout`, attached like every
# other child so a cancellation reaches it; the second failure ends the run,
# saying whether the last attempt stalled. LABEL names the index; the rest are
# apt-get's isolating options.
update_index() {
  label=$1
  shift
  last=
  for attempt in 1 2; do
    update_rc=0
    run_tracked timeout --kill-after=15 240 apt-get "$@" "${apt_bounds[@]}" update || update_rc=$?
    [ "$update_rc" -eq 0 ] && return 0
    case $update_rc in
      124 | 137) last="stalled past 240s (apt mirror stall)" ;;
      *) last="failed with exit $update_rc" ;;
    esac
    echo "ci-compiler-test: apt-get update ($label): attempt $attempt of 2 $last" >&2
  done
  die "cannot update the isolated $label index in 2 capped attempts; the last $last"
}

gcc_binary="$gcc_prefix/usr/bin/x86_64-linux-gnu-gcc-13"
if [ -n "$toolchain" ] && [ -x "$gcc_binary" ]; then
  echo "ci-compiler-test: reusing the staged GCC 13 in $toolchain (no download)"
else
  [ -r "$gcc_keyring" ] || die "the Ubuntu archive keyring is absent: $gcc_keyring"
  gcc_apt="$scratch/gcc-apt"
  mkdir -p "$gcc_apt/lists/partial" "$gcc_apt/archives/partial" "$gcc_apt/sourceparts" \
    "$gcc_apt/trustedparts" "$gcc_apt/cache" "$gcc_deb_dir" "$gcc_prefix" ||
    die "cannot stage the isolated $gcc_suite index"
  : >"$gcc_apt/sources.list" || die "cannot write the isolated $gcc_suite source"
  # An empty dpkg status too: the host's would offer its own installed gcc-13
  # (resolute's 13.4) as the candidate, which the noble index cannot serve.
  : >"$gcc_apt/status" || die "cannot write the isolated $gcc_suite status"
  for s in "$gcc_suite" "$gcc_suite-updates" "$gcc_suite-security"; do
    printf 'deb [arch=amd64 signed-by=%s] http://archive.ubuntu.com/ubuntu %s main universe\n' \
      "$gcc_keyring" "$s" >>"$gcc_apt/sources.list" ||
      die "cannot write the isolated $gcc_suite source"
  done
  # Disable global trust stores; each source names its signing key.
  gcc_apt_options=(
    -o "Dir::Etc::sourcelist=$gcc_apt/sources.list"
    -o "Dir::Etc::sourceparts=$gcc_apt/sourceparts"
    -o "Dir::Etc::trusted="
    -o "Dir::Etc::trustedparts=$gcc_apt/trustedparts"
    -o "Dir::State::lists=$gcc_apt/lists"
    -o "Dir::State::status=$gcc_apt/status"
    -o "Dir::Cache=$gcc_apt/cache"
    -o "Dir::Cache::archives=$gcc_apt/archives"
    -o APT::Architecture=amd64
    -o APT::Architectures=amd64
    -o Acquire::Languages=none
    -o APT::Get::List-Cleanup=0
  )
  echo "ci-compiler-test: GCC 13 source: Ubuntu $gcc_suite (isolated scratch index)"
  update_index "$gcc_suite" -q "${gcc_apt_options[@]}"
  echo "ci-compiler-test: apt download (GCC 13 host):"
  print_command apt-get '<isolated noble options>' "${apt_bounds[@]}" download "${gcc_packages[@]}"
  run_tracked_in "$gcc_deb_dir" apt-get "${gcc_apt_options[@]}" "${apt_bounds[@]}" download \
    "${gcc_packages[@]}" ||
    die "apt-get could not download the GCC 13 host packages from $gcc_suite"
  set -- "$gcc_deb_dir"/*.deb
  [ -e "$1" ] || die "apt-get reported success but downloaded no GCC 13 .deb files"
  for deb in "$@"; do
    echo "ci-compiler-test: extracting $(basename "$deb") into $gcc_prefix"
    run_tracked dpkg-deb -x "$deb" "$gcc_prefix" || die "cannot extract $deb"
  done
fi
if [ "$aarch64_clang" -eq 1 ]; then
  os_release=/etc/os-release
  [ -f "$os_release" ] || die "cannot identify the Debian-family release: $os_release is absent"
  llvm_codename=$(sed -n 's/^VERSION_CODENAME=//p' "$os_release" | sed -n '1p')
  llvm_codename=${llvm_codename#\"}
  llvm_codename=${llvm_codename%\"}
  case $llvm_codename in
    '' | *[!a-z0-9]*) die "unsupported VERSION_CODENAME for apt.llvm.org: $llvm_codename" ;;
  esac
  llvm_apt="$scratch/llvm-apt"
  llvm_lists="$llvm_apt/lists"
  llvm_archives="$llvm_apt/archives"
  llvm_sourceparts="$llvm_apt/sourceparts"
  llvm_trustedparts="$llvm_apt/trusted.gpg.d"
  llvm_source="$llvm_apt/llvm.list"
  llvm_key="$llvm_apt/apt.llvm.org.asc"
  mkdir -p "$llvm_lists/partial" "$llvm_archives/partial" "$llvm_sourceparts" \
    "$llvm_trustedparts" "$llvm_apt/cache" || die "cannot stage the isolated apt.llvm.org index"
  llvm_url="https://apt.llvm.org/$llvm_codename/"
  llvm_suite="llvm-toolchain-$llvm_codename-21"
  printf 'deb [signed-by=%s] %s %s main\n' "$llvm_key" "$llvm_url" "$llvm_suite" >"$llvm_source" ||
    die "cannot write the isolated apt.llvm.org source"
  # An empty dpkg status, as the GCC index has: a host with its distribution's
  # own clang-21 installed (resolute's 1:21.1.8-6ubuntu1 on minix) would offer
  # that version as the candidate, which apt.llvm.org's index cannot serve
  # ("Can't find a source to download version ...", gh-ocannl-1852).
  : >"$llvm_apt/status" || die "cannot write the isolated apt.llvm.org status"
  echo "ci-compiler-test: fetching pinned apt.llvm.org signing key"
  run_tracked curl -fsSL https://apt.llvm.org/llvm-snapshot.gpg.key -o "$llvm_key" ||
    die "cannot fetch the apt.llvm.org signing key"
  llvm_key_sha=$(sha256sum "$llvm_key" | sed -n 's/ .*//p') ||
    die "cannot hash the apt.llvm.org signing key"
  expected_llvm_key_sha=8b2a587ffd672c4687e7581dad4b2f6c1bb2ad6b480cd9771ba2ff48e0b8c75d
  [ "$llvm_key_sha" = "$expected_llvm_key_sha" ] ||
    die "apt.llvm.org signing key digest changed: $llvm_key_sha (expected $expected_llvm_key_sha)"
  # trustedparts stays empty: the pinned key belongs only to this source.
  llvm_apt_options=(
    -o "Dir::Etc::sourcelist=$llvm_source"
    -o "Dir::Etc::sourceparts=$llvm_sourceparts"
    -o "Dir::Etc::trusted="
    -o "Dir::Etc::trustedparts=$llvm_trustedparts"
    -o "Dir::State::lists=$llvm_lists"
    -o "Dir::State::status=$llvm_apt/status"
    -o "Dir::Cache=$llvm_apt/cache"
    -o "Dir::Cache::archives=$llvm_archives"
    -o APT::Get::List-Cleanup=0
  )
  echo "ci-compiler-test: apt.llvm.org source: $llvm_url $llvm_suite (isolated scratch index)"
  echo "ci-compiler-test: apt.llvm.org key sha256: $llvm_key_sha"
  update_index "apt.llvm.org ($llvm_codename)" "${llvm_apt_options[@]}"
  echo "ci-compiler-test: apt download (clang 21):"
  print_command apt-get "${llvm_apt_options[@]}" "${apt_bounds[@]}" download "${clang_packages[@]}"
  run_tracked_in "$deb_dir" apt-get "${llvm_apt_options[@]}" "${apt_bounds[@]}" download \
    "${clang_packages[@]}" ||
    die "apt-get could not download clang 21 from apt.llvm.org"
  download_packages "arm64 sysroot" "${arm64_packages[@]}"
fi

if [ "$aarch64_clang" -eq 1 ]; then
  set -- "$deb_dir"/*.deb
  [ -e "$1" ] || die "apt-get reported success but downloaded no clang .deb files"
  for deb in "$@"; do
    echo "ci-compiler-test: extracting $(basename "$deb")"
    run_tracked dpkg-deb -x "$deb" "$prefix" || die "cannot extract $deb"
  done
fi

[ -x "$gcc_binary" ] || die "extracted GCC 13 binary was not found at $gcc_binary"

gcc_log="$scratch/gcc-invocations.log"
gcc_wrapper="$wrapper_dir/gcc-13"
gcc_command=$(shell_quote "$gcc_wrapper")
quoted_gcc=$(shell_quote "$gcc_binary")
quoted_gcc_log=$(shell_quote "$gcc_log")
printf '%s\n' '#!/bin/sh' 'set -u' \
  "printf '%s\\n' \"\$*\" >>$quoted_gcc_log" \
  "exec $quoted_gcc \"\$@\"" >"$gcc_wrapper" || die "cannot write GCC wrapper"
chmod +x "$gcc_wrapper" || die "cannot make GCC wrapper executable"

gcc_version=$($gcc_binary -dumpfullversion 2>/dev/null) || die "extracted GCC cannot report its version"
case $gcc_version in 13 | 13.*) ;; *) die "expected GCC major 13, extracted version is $gcc_version" ;; esac
gcc_target=$($gcc_binary -dumpmachine 2>/dev/null) || die "extracted GCC cannot report its target"
case $gcc_target in
  x86_64-linux-gnu) ;;
  *) die "expected the Ubuntu x86_64 target, extracted GCC targets $gcc_target" ;;
esac
gcc_banner=$($gcc_binary --version 2>/dev/null | sed -n '1p') ||
  die "extracted GCC cannot print its version banner"

clang_wrapper=
clang_log=
clang_target=
if [ "$aarch64_clang" -eq 1 ]; then
  clang_binary="$prefix/usr/bin/clang-21"
  [ -x "$clang_binary" ] || die "extracted clang 21 binary was not found at $clang_binary"
  llvm_library=$(find "$prefix/usr/lib" -type f -name 'libLLVM.so.21*' -print -quit 2>/dev/null)
  [ -n "$llvm_library" ] || die "extracted libLLVM.so.21 was not found under $prefix/usr/lib"
  clang_library_path=$(dirname "$llvm_library")
  clang_log="$scratch/clang-aarch64-invocations.log"
  clang_wrapper="$wrapper_dir/clang-aarch64"
  clang_command=$(shell_quote "$clang_wrapper")
  cross_include="$prefix/usr/aarch64-linux-gnu/include"
  [ -d "$cross_include" ] ||
    die "arm64 cross headers were not extracted at $cross_include"
  quoted_clang=$(shell_quote "$clang_binary")
  quoted_clang_log=$(shell_quote "$clang_log")
  quoted_prefix=$(shell_quote "$prefix")
  quoted_cross_include=$(shell_quote "$cross_include")
  quoted_clang_library_path=$(shell_quote "$clang_library_path")
  printf '%s\n' '#!/bin/sh' 'set -u' \
    "export LD_LIBRARY_PATH=$quoted_clang_library_path" \
    "printf '%s\\n' \"\$*\" >>$quoted_clang_log" \
    "exec $quoted_clang --target=aarch64-linux-gnu --sysroot=$quoted_prefix -isystem $quoted_cross_include -mllvm -aarch64-neon-syntax=apple \"\$@\"" \
    >"$clang_wrapper" || die "cannot write clang wrapper"
  chmod +x "$clang_wrapper" || die "cannot make clang wrapper executable"
  clang_version=$(LD_LIBRARY_PATH="$clang_library_path" "$clang_binary" --version 2>/dev/null | sed -n '1p') ||
    die "extracted clang cannot print its version banner"
  case $clang_version in *"clang version 21."*) ;; *) die "expected clang major 21, got: $clang_version" ;; esac
  clang_target=$(LD_LIBRARY_PATH="$clang_library_path" "$clang_binary" \
    --target=aarch64-linux-gnu -print-target-triple 2>/dev/null) ||
    die "extracted clang cannot report its aarch64 target"
fi

echo "=== ci-compiler-test provenance ==="
echo "host compiler binary:  $gcc_binary"
echo "host compiler version: $gcc_banner"
echo "host compiler target:  $gcc_target"
echo "host compiler wrapper: $gcc_wrapper"
if [ "$aarch64_clang" -eq 1 ]; then
  echo "cross compiler binary:  $clang_binary"
  echo "cross compiler version: $clang_version"
  echo "cross compiler target:  $clang_target"
  echo "cross compiler loader:  LD_LIBRARY_PATH=$clang_library_path"
  echo "cross compiler wrapper: $clang_wrapper"
  echo "cross compiler flags:   --target=aarch64-linux-gnu --sysroot=$prefix -isystem $cross_include -mllvm -aarch64-neon-syntax=apple"
fi
echo "source tree:            $repo"
echo "source commit:          $(git -C "$repo" rev-parse HEAD)"
echo "isolated Dune build:    $build_dir"
echo "=== end provenance ==="

opam_environment=$(cd "$repo" && opam env --shell=sh) || die "cannot resolve the checkout's opam environment"
eval "$opam_environment"
opam_switch=$(cd "$repo" && opam switch show --safe) || die "cannot identify the selected opam switch"
echo "ci-compiler-test: opam switch: $opam_switch"

# Explicit compiler and backend settings must be the only OCANNL environment
# configuration that reaches the test. Print names, never values: a profile or
# compiler flag from the caller (or injected by the selected opam switch) would
# otherwise outrank the checked-in test configuration and falsify provenance.
ambient_ocannl_names=()
ambient_tool_names=()
while IFS='=' read -r name _; do
  case $name in
    OCANNL_TOOL_*) ambient_tool_names+=("$name") ;;
    OCANNL_*) ambient_ocannl_names+=("$name") ;;
  esac
done < <(env)
if [ "${#ambient_ocannl_names[@]}" -gt 0 ]; then
  echo "ci-compiler-test: clearing ambient OCANNL variables:"
  for name in "${ambient_ocannl_names[@]}"; do
    echo "  $name"
    unset "$name" || die "cannot clear ambient variable $name"
  done
fi
if [ "${#ambient_tool_names[@]}" -gt 0 ]; then
  echo "ci-compiler-test: preserving harness-control variables:"
  for name in "${ambient_tool_names[@]}"; do echo "  $name"; done
fi
remaining_ocannl_names=()
while IFS='=' read -r name _; do
  case $name in
    OCANNL_TOOL_*) ;;
    OCANNL_*) remaining_ocannl_names+=("$name") ;;
  esac
done < <(env)
[ "${#remaining_ocannl_names[@]}" -eq 0 ] ||
  die "ambient OCANNL variables remain after cleanup: ${remaining_ocannl_names[*]}"
echo "ci-compiler-test: ambient OCANNL configuration: cleared"

# These generic compiler selectors are not OCANNL settings, but can redirect
# the fetched drivers to a caller-owned toolchain or headers just as surely.
unset GCC_EXEC_PREFIX COMPILER_PATH CPATH C_INCLUDE_PATH LIBRARY_PATH SDKROOT MACOSX_DEPLOYMENT_TARGET
unset AARCH64_CROSS_GCC

# Treat the dynamic loader as part of the toolchain boundary too. Clear the
# family, rather than today's two common members: LD_PRELOAD/LD_LIBRARY_PATH
# can replace compiler dependencies, while LD_AUDIT and friends can perturb
# both the compiler and the generated shared object. The clang wrapper exports
# its one derived scratch path internally after this cleanup.
ambient_loader_names=()
while IFS='=' read -r name _; do
  case $name in LD_*) ambient_loader_names+=("$name") ;; esac
done < <(env)
if [ "${#ambient_loader_names[@]}" -gt 0 ]; then
  echo "ci-compiler-test: clearing ambient dynamic-loader variables:"
  for name in "${ambient_loader_names[@]}"; do
    echo "  $name"
    unset "$name" || die "cannot clear ambient loader variable $name"
  done
fi
remaining_loader_names=()
while IFS='=' read -r name _; do
  case $name in LD_*) remaining_loader_names+=("$name") ;; esac
done < <(env)
[ "${#remaining_loader_names[@]}" -eq 0 ] ||
  die "ambient loader variables remain after cleanup: ${remaining_loader_names[*]}"
echo "ci-compiler-test: ambient dynamic-loader overrides: cleared"

rm -f "$gcc_log"
[ -z "$clang_log" ] || rm -f "$clang_log"
# One alias through the capped harness, in the isolated build directory.
run_test() {
  if [ "$aarch64_clang" -eq 1 ]; then
    run_tracked env OCANNL_BACKEND=cc OCANNL_CC_BACKEND_COMPILER_COMMAND="$gcc_command" \
      AARCH64_CROSS_GCC="$clang_command" DUNE_BUILD_DIR="$build_dir" \
      "$repo/tools/test-run.sh" run --cap "$cap" build -j "$jobs" "$1"
  else
    run_tracked env OCANNL_BACKEND=cc OCANNL_CC_BACKEND_COMPILER_COMMAND="$gcc_command" \
      DUNE_BUILD_DIR="$build_dir" "$repo/tools/test-run.sh" run --cap "$cap" \
      build -j "$jobs" "$1"
  fi
}

log_lines() { if [ -s "$1" ]; then wc -l <"$1" | tr -d ' '; else echo 0; fi; }

if [ "$pieces" = census ]; then
  # Every piece runs, whatever the one before it did, and is judged as one
  # named alias is below: its own exit status, and its own calls to the
  # fetched GCC (the logs are emptied before each piece). The cross compiler
  # is asked of the census as a whole: only the foreign pieces compile with it.
  verdicts=()
  failures=0
  clang_total=0
  for alias in "${aliases[@]}"; do
    piece=${alias#@test/operations/runtest-cc_march_census-}
    rm -f "$gcc_log"
    [ -z "$clang_log" ] || rm -f "$clang_log"
    echo "ci-compiler-test: piece $piece: test (unpiped): $alias"
    piece_rc=0
    run_test "$alias" || piece_rc=$?
    gcc_calls=$(log_lines "$gcc_log")
    detail="host compiler: $gcc_calls calls"
    if [ "$aarch64_clang" -eq 1 ]; then
      clang_calls=$(log_lines "$clang_log")
      clang_total=$((clang_total + clang_calls))
      detail="$detail; cross compiler: $clang_calls calls"
    fi
    case $piece_rc in
      0)
        if [ "$gcc_calls" -gt 0 ]; then
          verdict=PASS
        else
          verdict="FAIL (passed without invoking the fetched GCC 13 compiler)"
        fi
        ;;
      75) verdict="NOT RUN (exit 75: SLOT REFUSED, the slot's verdict, not the piece's)" ;;
      *) verdict="FAIL (exit $piece_rc)" ;;
    esac
    [ "$verdict" = PASS ] || failures=$((failures + 1))
    verdicts+=("$piece: $verdict; $detail")
    echo "ci-compiler-test: piece ${verdicts[${#verdicts[@]} - 1]}"
  done
  echo "=== ci-compiler-test census verdicts ==="
  for line in "${verdicts[@]}"; do echo "ci-compiler-test: piece $line"; done
  echo "=== end census verdicts ==="
  if [ "$aarch64_clang" -eq 1 ] && [ "$clang_total" -eq 0 ]; then
    echo "ci-compiler-test: no census piece invoked the clang/aarch64 proxy; no cross result is certified" >&2
    failures=$((failures + 1))
  fi
  if [ "$failures" -gt 0 ]; then
    echo "ci-compiler-test: census: FAIL ($failures of ${#aliases[@]} pieces not passed or uncertified)" >&2
    exit 1
  fi
  echo "ci-compiler-test: census: PASS (${#aliases[@]} pieces)"
  echo "ci-compiler-test: verified host=$gcc_banner target=$gcc_target${clang_target:+ cross=$clang_target}"
  exit 0
fi

echo "ci-compiler-test: test (unpiped): $alias_name"
run_test "$alias_name" || exit $?

[ -s "$gcc_log" ] ||
  die "$alias_name passed without invoking the fetched GCC 13 compiler; no compiler result is certified"
gcc_invocations=$(wc -l <"$gcc_log" | tr -d ' ')
echo "ci-compiler-test: host compiler invocation evidence: PASS ($gcc_invocations calls)"

if [ "$aarch64_clang" -eq 1 ]; then
  [ -s "$clang_log" ] ||
    die "$alias_name passed without invoking the clang/aarch64 proxy; use it with a test that honors AARCH64_CROSS_GCC"
  clang_invocations=$(wc -l <"$clang_log" | tr -d ' ')
  echo "ci-compiler-test: cross compiler invocation evidence: PASS ($clang_invocations calls)"
fi

echo "ci-compiler-test: test alias: PASS $alias_name"
echo "ci-compiler-test: verified host=$gcc_banner target=$gcc_target${clang_target:+ cross=$clang_target}"
exit 0
