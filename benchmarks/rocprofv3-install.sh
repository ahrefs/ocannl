#!/usr/bin/env bash
# A rootless rocprofv3 for gh1133_cells.sh's HIP kprof step (gh-ocannl-1392), on a box whose ROCm
# comes from the distribution.
#
# Usage: benchmarks/rocprofv3-install.sh [PREFIX]
#   PREFIX  where the packages are unpacked (default ~/.local/opt/rocprofv3-$ROCM_VERSION); the
#           last line printed is the rocprofv3 to pass as ROCPROFV3=.
# Environment: ROCM_VERSION (default 7.2.4) and ROCM_DIST (default noble) pick the release in AMD's
# apt repository (https://repo.radeon.com/rocm/apt/<version>, dists/<dist>).
# Exit: 0 installed (or already there); 1 a download, checksum or unpack failed; 2 usage.
#
# Why not the distribution's packages: Ubuntu 26.04's ROCm (minix, tuf) ships no rocprofiler-sdk,
# and its HSA runtime is built without rocprofiler-register, through which rocprofiler-sdk
# intercepts kernel dispatches -- so no profiler that build could load would see a kernel. AMD's own
# HSA runtime registers. rocprofv3 puts its own lib/ on LD_LIBRARY_PATH, so whatever this prefix
# holds replaces the system's copy for the traced process and nothing else does. It therefore holds
# the profiler and the HSA runtime it needs, and NOT the HIP runtime, hiprtc or comgr: a traced
# step compiles its kernels with the system's compiler and launches them through the system's HIP,
# as an untraced step does, and differs from one only in the HSA runtime under it. Measured on
# gfx1151 under Ubuntu's 7.1 HIP: AMD's 7.1.0 HSA runtime faults at the first stream (with or
# without AMD's 7.1 HIP above it); 7.2.4's runs the traced step.
#
# Nothing is installed system-wide and nothing needs root: each package's SHA256 is checked against
# the repository's Packages index, then unpacked with dpkg-deb -x into a staging directory beside
# PREFIX, which becomes PREFIX (with a completion marker) only once every package is in and
# rocprofv3 --version answers -- so a failed run leaves nothing a retry would take for an install.
set -u
case ${1:-} in -h | --help) sed -n '/^# Usage/,/^# Exit/p' "$0" >&2; exit 0 ;; esac
[ $# -le 1 ] || { sed -n '/^# Usage/,/^# Exit/p' "$0" >&2; exit 2; }
version=${ROCM_VERSION:-7.2.4}
dist=${ROCM_DIST:-noble}
prefix=${1:-$HOME/.local/opt/rocprofv3-$version}
repo=https://repo.radeon.com/rocm/apt/$version
# The profiler, its tool libraries' dependencies, and the HSA runtime they register with.
packages="rocprofiler-sdk rocprofiler-sdk-roctx rocprofiler-sdk-rocpd rocprofiler-register rocm-core hsa-rocr hsa-amd-aqlprofile"

for tool in curl gunzip dpkg-deb sha256sum awk; do
  command -v "$tool" >/dev/null || { echo "rocprofv3-install: $tool is not on PATH" >&2; exit 1; }
done
marker=.rocprofv3-install-complete
if [ -f "$prefix/$marker" ]; then
  echo "rocprofv3-install: already installed under $prefix"
  cat "$prefix/$marker"
  exit 0
fi
if [ -e "$prefix" ]; then
  echo "rocprofv3-install: $prefix exists but holds no completed install; remove it or name another PREFIX" >&2
  exit 1
fi
stage=$prefix.partial-$$
trap 'rm -rf "$stage"' EXIT
mkdir -p "$stage/debs" || exit 1
index=$stage/debs/Packages
curl -fsS --max-time 120 "$repo/dists/$dist/main/binary-amd64/Packages.gz" | gunzip >"$index" || {
  echo "rocprofv3-install: cannot read $repo/dists/$dist/main/binary-amd64/Packages.gz" >&2
  exit 1
}
for pkg in $packages; do
  # The first stanza of the package: Filename and SHA256.
  read -r file sum < <(awk -v P="$pkg" '
    /^Package: / { here = ($2 == P) }
    here && /^Filename: / { f = $2 }
    here && /^SHA256: / { s = $2 }
    /^$/ { if (here && f != "") { print f, s; exit } here = 0 }' "$index")
  if [ -z "${file:-}" ] || [ -z "${sum:-}" ]; then
    echo "rocprofv3-install: $pkg is not in $repo ($dist)" >&2
    exit 1
  fi
  deb=$stage/debs/${file##*/}
  curl -fsS --max-time 600 -o "$deb" "$repo/$file" || { echo "rocprofv3-install: cannot fetch $repo/$file" >&2; exit 1; }
  echo "$sum  $deb" | sha256sum -c --quiet - || { echo "rocprofv3-install: $deb fails its SHA256" >&2; exit 1; }
  dpkg-deb -x "$deb" "$stage" || { echo "rocprofv3-install: cannot unpack $deb" >&2; exit 1; }
  echo "rocprofv3-install: $pkg ${file##*/}"
done
found=$(ls "$stage"/opt/rocm-*/bin/rocprofv3 2>/dev/null | head -1)
[ -n "$found" ] || { echo "rocprofv3-install: no bin/rocprofv3 after unpacking" >&2; exit 1; }
"$found" --version >/dev/null 2>&1 || { echo "rocprofv3-install: $found --version fails" >&2; exit 1; }
mv "$stage" "$prefix" || exit 1
trap - EXIT
found=$prefix${found#"$stage"}
echo "$found" >"$prefix/$marker" || exit 1
echo "$found"
