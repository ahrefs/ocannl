#!/usr/bin/env bash
# Stage the aarch64 cross gcc that cc_march_census's two aarch64 columns
# compile with, without root, and print its path on stdout (gh-ocannl-1120).
#
# Usage:
#   tools/ci-aarch64-cross.sh PREFIX
#
# CI's ubuntu leg exports the printed path as AARCH64_CROSS_GCC. Everything
# else this prints goes to stderr, so `$(tools/ci-aarch64-cross.sh ...)` is the
# path and nothing but the path.
#
# Why gcc 15, and why from Ubuntu 26.04 (resolute) rather than the runner's
# own release: the census's claims were written against the gcc 15 cross the
# fleet boxes carry, and the aarch64 cross gccs ubuntu-24.04 can install fail
# four of them -- gcc 13 (`gcc-aarch64-linux-gnu`) and gcc 14 both spill the
# 4x6 register tile at w16 and lower the fp16 widening bridge lane by lane,
# where gcc 15 does neither. Those are facts about older compilers, not about
# this change; see the issue for the follow-up. The packages are resolved from
# an isolated scratch apt index over the Ubuntu archive (signed with the
# archive key every Ubuntu host already trusts), so a superseded version never
# 404s, and the host's own apt configuration is neither read nor touched.
# resolute's build needs glibc 2.38 and the isl/mpc/mpfr sonames 24.04 ships;
# should a rebuild ever need more, the smoke compile below fails the step
# loudly instead of the census skipping its columns quietly.
#
# Only `cc1` runs: the census compiles to assembly (`-S`), so no binutils are
# staged, and the arm64 libc headers come from the same index.

set -u

die() { echo "ci-aarch64-cross: $*" >&2; exit 2; }

[ "$#" -eq 1 ] || die "usage: tools/ci-aarch64-cross.sh PREFIX"
prefix=$1
case $prefix in
  /*) ;;
  *) die "PREFIX must be absolute: $prefix" ;;
esac

suite=resolute
major=15
keyring=/usr/share/keyrings/ubuntu-archive-keyring.gpg
packages=(
  "gcc-$major-aarch64-linux-gnu"
  "gcc-$major-aarch64-linux-gnu-base"
  "cpp-$major-aarch64-linux-gnu"
  "libgcc-$major-dev-arm64-cross"
  libc6-dev-arm64-cross
  linux-libc-dev-arm64-cross
)

for command_name in apt-get dpkg-deb; do
  command -v "$command_name" >/dev/null 2>&1 || die "$command_name is not on PATH"
done
[ -r "$keyring" ] || die "the Ubuntu archive keyring is absent: $keyring"

apt_dir="$prefix.apt"
debs="$prefix.debs"
mkdir -p "$apt_dir/lists/partial" "$apt_dir/archives/partial" "$apt_dir/sourceparts" \
  "$apt_dir/trustedparts" "$debs" "$prefix" || die "cannot stage $prefix"
: >"$apt_dir/sources.list"
for s in "$suite" "$suite-updates" "$suite-security"; do
  printf 'deb [arch=amd64 signed-by=%s] http://archive.ubuntu.com/ubuntu %s main universe\n' \
    "$keyring" "$s" >>"$apt_dir/sources.list"
done
apt_options=(
  -o "Dir::Etc::sourcelist=$apt_dir/sources.list"
  -o "Dir::Etc::sourceparts=$apt_dir/sourceparts"
  -o "Dir::Etc::trusted=/dev/null"
  -o "Dir::Etc::trustedparts=$apt_dir/trustedparts"
  -o "Dir::State::lists=$apt_dir/lists"
  -o "Dir::Cache::archives=$apt_dir/archives"
  -o APT::Architecture=amd64
  -o APT::Architectures=amd64
  -o Acquire::Languages=none
  -o APT::Get::List-Cleanup=0
)
echo "ci-aarch64-cross: resolving gcc $major aarch64 cross from Ubuntu $suite (isolated index)" >&2
apt-get -q "${apt_options[@]}" update >&2 || die "cannot update the isolated $suite index"
(cd "$debs" && apt-get -q "${apt_options[@]}" download "${packages[@]}") >&2 ||
  die "apt-get could not download: ${packages[*]}"
for deb in "$debs"/*.deb; do
  echo "ci-aarch64-cross: extracting $(basename "$deb")" >&2
  dpkg-deb -x "$deb" "$prefix" || die "cannot extract $deb"
done

gcc="$prefix/usr/bin/aarch64-linux-gnu-gcc-$major"
[ -x "$gcc" ] || die "no $gcc after extraction"
version=$("$gcc" -dumpfullversion) || die "$gcc does not run on this host"
case $version in
  "$major".*) ;;
  *) die "$gcc reports version $version, expected $major.x" ;;
esac
# A smoke compile through the headers the census's kernels include and the
# NEON fp16 target its second column names: a toolchain that cannot do this
# would make the census report its columns as not accepted.
smoke="$prefix.smoke"
mkdir -p "$smoke" || die "cannot create $smoke"
printf '#include <stdint.h>\n#include <string.h>\n#include <arm_neon.h>\nfloat32x4_t f(float32x4_t a, float32x4_t b, float32x4_t c) { return vfmaq_f32(c, a, b); }\n' \
  >"$smoke/smoke.c"
"$gcc" -march=armv8.2-a+fp16 -O2 -S -o "$smoke/smoke.s" "$smoke/smoke.c" ||
  die "$gcc cannot compile the smoke kernel"
grep -q fmla "$smoke/smoke.s" || die "$gcc emitted no fmla for vfmaq_f32"
echo "ci-aarch64-cross: $gcc is gcc $version and compiles NEON under armv8.2-a+fp16" >&2
printf '%s\n' "$gcc"
