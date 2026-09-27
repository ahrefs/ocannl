#!/usr/bin/env bash
# Run one named test through the compiler version that ubuntu-latest exposed in
# the gh-ocannl-752 CI failure, without installing that compiler system-wide.
#
# Usage:
#   tools/ci-compiler-test.sh [OPTIONS] @DIR/runtest-NAME
#
# Options:
#   --aarch64-clang   Also fetch clang 21 and the arm64 cross headers, then set
#                     AARCH64_CROSS_GCC to clang --target=aarch64-linux-gnu
#                     with Apple's NEON assembly syntax. The named test must
#                     invoke this cross compiler as well as the host compiler.
#   --dry-run         Validate arguments, create the scratch layout, and print
#                     every fetch/extract/test step without fetching or running.
#   --keep            Keep the scratch directory even on success (a failing
#                     run always keeps it, so the build tree's diagnostics --
#                     e.g. cc_march_census_kernels/rows.txt -- survive it;
#                     otherwise cleanup precedes the exit sentinel).
#   --cap SECONDS     Named-test wall-clock cap; 0 disables it (default: 3600).
#   -j, --jobs N      Dune concurrency, 1..4 (default: 4).
#
# Examples:
#   tools/ci-compiler-test.sh @test/operations/runtest-cpu_simd_reduction
#   tools/ci-compiler-test.sh --aarch64-clang \
#     @test/operations/runtest-cc_march_census
#   tools/ci-compiler-test.sh --dry-run --aarch64-clang \
#     @test/operations/runtest-cc_march_census
#
# The real fetch is Linux/Debian-family only: it deliberately uses the
# no-root recipe established by gh-ocannl-752, `apt-get download` followed by
# `dpkg-deb -x` into a temporary prefix. --dry-run is portable and is the
# supported way to inspect and validate the staging plan elsewhere. The real
# host must be x86_64 because the fetched compiler produces executable x86_64
# cc kernels, not merely cross-compiled assembly.
#
# Output is deliberately unpiped. Dune's exit status remains the verdict. The
# compiler wrappers log every invocation, so a cached build or an alias that
# never reaches the requested compiler fails rather than being certified.

set -u

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
    --keep)
      keep=1
      shift
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
case $alias_name in
  @*/runtest-?*) ;;
  '') usage ;;
  *) die "test must be a named Dune alias of the form @DIR/runtest-NAME" ;;
esac

script_dir=$(cd "$(dirname "$0")" && pwd -P) || die "cannot resolve tools directory"
repo=$(git -C "$script_dir/.." rev-parse --show-toplevel 2>/dev/null) ||
  die "script is not inside a git checkout"
repo=$(cd "$repo" && pwd -P) || die "cannot resolve repository root"
script_path="$repo/tools/ci-compiler-test.sh"
[ -f "$script_path" ] || die "cannot find the script at $script_path"

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
  wait "$active_child"
  tracked_rc=$?
  active_child=
  return "$tracked_rc"
}

run_tracked_in() {
  tracked_dir=$1
  shift
  (cd "$tracked_dir" && exec "$@") &
  active_child=$!
  wait "$active_child"
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

gcc_packages=(gcc-13-x86-64-linux-gnu cpp-13-x86-64-linux-gnu libgcc-13-dev)
clang_packages=(clang-21 libclang-cpp21 libllvm21 libclang-common-21-dev)
arm64_packages=(libc6-dev-arm64-cross linux-libc-dev-arm64-cross)

echo "=== ci-compiler-test plan ==="
echo "repository:       $repo"
echo "source commit:    $(git -C "$repo" rev-parse HEAD)"
if [ -z "$(git -C "$repo" status --porcelain --untracked-files=all)" ]; then
  echo "source state:     clean"
else
  echo "source state:     DIRTY (the working tree, not only the commit, will be tested)"
fi
echo "test alias:       $alias_name"
echo "dune jobs:        $jobs"
echo "test cap:         ${cap}s"
echo "scratch:          $scratch"
echo "host toolchain:   GCC 13 x86_64-linux-gnu (Ubuntu CI proxy)"
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
  echo "ci-compiler-test: planned package download (host compiler):"
  print_command apt-get download "${gcc_packages[@]}"
  echo "ci-compiler-test: planned extraction:"
  print_command dpkg-deb -x '<each downloaded .deb>' "$prefix"
  if [ "$aarch64_clang" -eq 1 ]; then
    echo "ci-compiler-test: planned package download (aarch64 clang proxy):"
    echo "  source: isolated https://apt.llvm.org/<VERSION_CODENAME>/ index with pinned signing key"
    print_command apt-get '<isolated apt.llvm.org options>' update
    print_command apt-get '<isolated apt.llvm.org options>' download "${clang_packages[@]}"
    print_command apt-get download "${arm64_packages[@]}"
    echo "ci-compiler-test: planned cross invocation:"
    echo "  loader path: derived from the extracted libLLVM.so.21 multiarch directory"
    print_command "$prefix/usr/bin/clang-21" --target=aarch64-linux-gnu \
      "--sysroot=$prefix" -isystem "$prefix/usr/aarch64-linux-gnu/include" \
      -mllvm -aarch64-neon-syntax=apple
  fi
  echo "ci-compiler-test: planned isolated test invocation:"
  if [ "$aarch64_clang" -eq 1 ]; then
    print_command env OCANNL_BACKEND=cc \
      "OCANNL_CC_BACKEND_COMPILER_COMMAND=$(shell_quote "$wrapper_dir/gcc-13")" \
      "AARCH64_CROSS_GCC=$(shell_quote "$wrapper_dir/clang-aarch64")" \
      "DUNE_BUILD_DIR=$build_dir" "$repo/tools/test-run.sh" run --cap "$cap" \
      build -j "$jobs" "$alias_name"
  else
    print_command env OCANNL_BACKEND=cc \
      "OCANNL_CC_BACKEND_COMPILER_COMMAND=$(shell_quote "$wrapper_dir/gcc-13")" \
      "DUNE_BUILD_DIR=$build_dir" "$repo/tools/test-run.sh" run --cap "$cap" \
      build -j "$jobs" "$alias_name"
  fi
  echo "ci-compiler-test: dry-run: PASS (arguments and scratch staging only)"
  exit 0
fi

[ "$(uname -s)" = Linux ] ||
  die "real toolchain fetch is Linux/Debian-family only (this host is $(uname -s)); use --dry-run here"
[ "$(uname -m)" = x86_64 ] ||
  die "real GCC 13 execution requires an x86_64 host (this host is $(uname -m)); use --dry-run here"
for command_name in apt-get dpkg-deb git opam sed; do
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
  print_command apt-get download "$@"
  run_tracked_in "$deb_dir" apt-get download "$@" ||
    die "apt-get could not download the $label packages"
}

download_packages "GCC 13 host" "${gcc_packages[@]}"
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
  llvm_key="$llvm_trustedparts/apt.llvm.org.asc"
  mkdir -p "$llvm_lists/partial" "$llvm_archives/partial" "$llvm_sourceparts" \
    "$llvm_trustedparts" || die "cannot stage the isolated apt.llvm.org index"
  llvm_url="https://apt.llvm.org/$llvm_codename/"
  llvm_suite="llvm-toolchain-$llvm_codename-21"
  printf 'deb %s %s main\n' "$llvm_url" "$llvm_suite" >"$llvm_source" ||
    die "cannot write the isolated apt.llvm.org source"
  echo "ci-compiler-test: fetching pinned apt.llvm.org signing key"
  run_tracked curl -fsSL https://apt.llvm.org/llvm-snapshot.gpg.key -o "$llvm_key" ||
    die "cannot fetch the apt.llvm.org signing key"
  llvm_key_sha=$(sha256sum "$llvm_key" | sed -n 's/ .*//p') ||
    die "cannot hash the apt.llvm.org signing key"
  expected_llvm_key_sha=8b2a587ffd672c4687e7581dad4b2f6c1bb2ad6b480cd9771ba2ff48e0b8c75d
  [ "$llvm_key_sha" = "$expected_llvm_key_sha" ] ||
    die "apt.llvm.org signing key digest changed: $llvm_key_sha (expected $expected_llvm_key_sha)"
  llvm_apt_options=(
    -o "Dir::Etc::sourcelist=$llvm_source"
    -o "Dir::Etc::sourceparts=$llvm_sourceparts"
    -o "Dir::Etc::trusted=/dev/null"
    -o "Dir::Etc::trustedparts=$llvm_trustedparts"
    -o "Dir::State::lists=$llvm_lists"
    -o "Dir::Cache::archives=$llvm_archives"
    -o APT::Get::List-Cleanup=0
  )
  echo "ci-compiler-test: apt.llvm.org source: $llvm_url $llvm_suite (isolated scratch index)"
  echo "ci-compiler-test: apt.llvm.org key sha256: $llvm_key_sha"
  run_tracked apt-get "${llvm_apt_options[@]}" update ||
    die "cannot update the isolated apt.llvm.org index for $llvm_codename"
  echo "ci-compiler-test: apt download (clang 21):"
  print_command apt-get "${llvm_apt_options[@]}" download "${clang_packages[@]}"
  run_tracked_in "$deb_dir" apt-get "${llvm_apt_options[@]}" download "${clang_packages[@]}" ||
    die "apt-get could not download clang 21 from apt.llvm.org"
  download_packages "arm64 sysroot" "${arm64_packages[@]}"
fi

set -- "$deb_dir"/*.deb
[ -e "$1" ] || die "apt-get reported success but downloaded no .deb files"
for deb in "$@"; do
  echo "ci-compiler-test: extracting $(basename "$deb")"
  run_tracked dpkg-deb -x "$deb" "$prefix" || die "cannot extract $deb"
done

gcc_binary="$prefix/usr/bin/x86_64-linux-gnu-gcc-13"
if [ ! -x "$gcc_binary" ]; then
  gcc_binary="$prefix/usr/bin/gcc-13"
fi
[ -x "$gcc_binary" ] || die "extracted GCC 13 binary was not found under $prefix/usr/bin"

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
echo "ci-compiler-test: test (unpiped): $alias_name"
if [ "$aarch64_clang" -eq 1 ]; then
  run_tracked env OCANNL_BACKEND=cc OCANNL_CC_BACKEND_COMPILER_COMMAND="$gcc_command" \
    AARCH64_CROSS_GCC="$clang_command" DUNE_BUILD_DIR="$build_dir" \
    "$repo/tools/test-run.sh" run --cap "$cap" build -j "$jobs" "$alias_name" || exit $?
else
  run_tracked env OCANNL_BACKEND=cc OCANNL_CC_BACKEND_COMPILER_COMMAND="$gcc_command" \
    DUNE_BUILD_DIR="$build_dir" "$repo/tools/test-run.sh" run --cap "$cap" \
    build -j "$jobs" "$alias_name" || exit $?
fi

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
