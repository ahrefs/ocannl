#!/bin/bash
# gh-ocannl-170 measurement driver: host<->device transfer throughput from pageable memory (what
# Cuda_backend / Hip_backend from_host and to_host copy from today) against page-locked, staged,
# chunk-staged and registered host memory. benchmarks/gh170_transfer_probe.c says what each arm is.
#
# Usage:
#   benchmarks/gh170_transfer.sh cuda|hip OUT [REPS [MAX_MIB]]
#
#   OUT      results directory, created, refused unless empty; gets bin/ and probe.out.
#   REPS     repeats per row (default 15); MAX_MIB the largest size (default 256).
#
# Builds the probe with gcc against the toolkit (CUDA_HOME, default /usr/local/cuda; ROCM_HOME,
# default /opt/rocm -- read here only, never exported), then runs it pinned to PROBE_CPUS (default
# 2) when taskset exists. A measurement: run it under the box's hold or as a leg with a deadline.
# Exit: 0 the probe finished (its last line is `done`); 1 a build or probe failure; 2 usage.
set -u
usage() { sed -n '/^# Usage/,/^# Exit/p' "$0" | sed '$d' >&2; exit 2; }
[ $# -ge 2 ] && [ $# -le 4 ] || usage
api=$1 out=$2 reps=${3:-15} max_mib=${4:-256}
case $api in cuda | hip) ;; *) usage ;; esac
if [ -e "$out" ] && [ -n "$(ls -A "$out" 2>/dev/null)" ]; then
  echo "gh170_transfer: $out is not empty" >&2
  exit 2
fi
src=$(cd "$(dirname "$0")" && pwd -P)/gh170_transfer_probe.c
mkdir -p "$out/bin" || exit 1
exe=$out/bin/gh170_transfer_probe
if [ "$api" = cuda ]; then
  cuda_home=${CUDA_HOME:-/usr/local/cuda}
  gcc -O2 -o "$exe" "$src" -I"$cuda_home/include" -L"$cuda_home/lib64" -lcuda || exit 1
else
  rocm_home=${ROCM_HOME:-/opt/rocm}
  gcc -O2 -DUSE_HIP -D__HIP_PLATFORM_AMD__ -o "$exe" "$src" -I"$rocm_home/include" \
    -L"$rocm_home/lib" -Wl,-rpath,"$rocm_home/lib" -lamdhip64 || exit 1
fi
pin=()
command -v taskset >/dev/null && pin=(taskset -c "${PROBE_CPUS:-2}")
{
  echo "host $(hostname) rev $(git -C "$(dirname "$src")" rev-parse HEAD 2>/dev/null || echo -)"
  "${pin[@]}" "$exe" "$reps" "$max_mib"
} >"$out/probe.out" 2>&1
rc=$?
cat "$out/probe.out"
[ $rc -eq 0 ] && [ "$(tail -n 1 "$out/probe.out")" = done ] || exit 1
