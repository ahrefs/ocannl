#!/usr/bin/env bash
# Build and dry-run the precompiled CUDA A/B driver over two emitted kernels (gh-ocannl-1190).
#
#   mma_register_scope_probe_build.sh ARM0.cu ARM1.cu OUTPUT LABEL0 LABEL1
#
# ARM0/ARM1 are kernels exported by bench_mma_register_scope_emit (any revision, any layout);
# the driver reports ARM1 over ARM0 under the labels, which are required: they are what the
# output says was compared (gh-ocannl-1073 timed per_block vs resident, gh-ocannl-1190 plain vs
# swizzled), and no default can know which. This script pins the nvcc flags, prints the toolchain and the SHA-256 of
# both sources and the binary, the load-path census of each source, compiles, and runs the
# driver's --dry-run, which checks both arms against every exact host product cell. Run it under
# a correctness reservation; run OUTPUT without --dry-run only in an exclusive timing window.
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "usage: $0 ARM0.cu ARM1.cu OUTPUT LABEL0 LABEL1" >&2
  exit 2
fi
here=$(cd "$(dirname "$0")" && pwd)
abs() { (cd "$(dirname "$1")" && printf '%s/%s\n' "$(pwd)" "$(basename "$1")"); }
arm0=$(abs "$1")
arm1=$(abs "$2")
out=$(abs "$3")
label0=$4
label1=$5
for label in "$label0" "$label1"; do
  # The labels become key names in the driver's key=value output.
  if [[ ! $label =~ ^[a-z][a-z0-9_]*$ ]]; then
    echo "label must match [a-z][a-z0-9_]*: $label" >&2
    exit 2
  fi
done
[[ $label0 != "$label1" ]] || { echo "the two labels must differ" >&2; exit 2; }
for src in "$arm0" "$arm1"; do
  [[ -f $src ]] || { echo "missing source: $src" >&2; exit 2; }
done

sha256() {
  if command -v sha256sum >/dev/null; then sha256sum "$1" | cut -d' ' -f1
  else shasum -a 256 "$1" | cut -d' ' -f1; fi
}

# compute_89 PTX, forward-JIT to the device: the CUDA backend's fp8 architecture floor.
flags=(-O3 -gencode "arch=compute_89,code=compute_89")
echo "nvcc: $(nvcc --version | tail -n 1)"
echo "flags: ${flags[*]}"
for i in 0 1; do
  if [[ $i == 0 ]]; then src=$arm0 label=$label0; else src=$arm1 label=$label1; fi
  # The static load-path census: a plain twin gathers per lane and emits no ldmatrix.
  echo "arm$i=$label source=$src sha256=$(sha256 "$src") ldmatrix_lines=$(grep -c ldmatrix "$src" || true) mma_sync_lines=$(grep -c 'mma\.sync' "$src" || true)"
done
nvcc "${flags[@]}" \
  "-DMMA_GENERATED_ARM0=\"$arm0\"" "-DMMA_ARM0_LABEL=\"$label0\"" \
  "-DMMA_GENERATED_ARM1=\"$arm1\"" "-DMMA_ARM1_LABEL=\"$label1\"" \
  "$here/mma_register_scope_probe.cu" -o "$out"
echo "binary=$out sha256=$(sha256 "$out")"
"$out" --dry-run
