#!/usr/bin/env bash
# Build and dry-run the precompiled CUDA A/B driver over two emitted kernels (gh-ocannl-1190).
#
#   mma_register_scope_probe_build.sh [--negative-control] ARM0.cu ARM1.cu OUTPUT LABEL0 LABEL1
#
# ARM0/ARM1 are kernels exported by bench_mma_register_scope_emit (any revision, any layout);
# the driver reports ARM1 over ARM0 under the labels, which are required: they are what the
# output says was compared (gh-ocannl-1073 timed per_block vs resident, gh-ocannl-1190 plain vs
# swizzled), and no default can know which. This script pins the nvcc flags, prints the toolchain and the SHA-256 of
# both sources and the binary, the load-path census of each source, compiles, and runs the
# driver's --dry-run, which checks both arms against every exact host product cell. Run it under
# a correctness reservation; run OUTPUT without --dry-run only in an exclusive timing window.
#
# --negative-control proves that dry run can fail. ARM1 must be a swizzled kernel (ldmatrix A);
# the script writes OUTPUT.broken-a.cu, a copy whose ldmatrix A address drops its swizzle XOR
# (exactly one line changes), builds the driver over ARM0 and that copy, and succeeds only when
# the dry run fails the way the fault predicts: a cell where ARM0 equals the exact host product
# and the broken arm does not. A passing dry run, or any other failure, exits 1. OUTPUT is then
# the broken driver: never time it.
set -euo pipefail

negative=false
if [[ ${1-} == --negative-control ]]; then
  negative=true
  shift
fi
if [[ $# -ne 5 ]]; then
  echo "usage: $0 [--negative-control] ARM0.cu ARM1.cu OUTPUT LABEL0 LABEL1" >&2
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

if $negative; then
  # Swizzle_b128 emits the 16-byte chunk index as ((col >> s) ^ (row & m)) << s; dropping the
  # XOR term on the ldmatrix A address line feeds every lane an unswizzled row address.
  broken=$out.broken-a.cu
  perl -pe 's/\^ \(\(.*?\) & \d+\)\) (<< \d+\))/^ 0) $1/ if /__cvta_generic_to_shared\(__mma_ap /' \
    "$arm1" >"$broken"
  mutated=$(diff "$arm1" "$broken" | grep -c '^>' || true)
  echo "negative_control: source=$arm1 broken=$broken mutated_lines=$mutated"
  if [[ $mutated != 1 ]]; then
    echo "NEGATIVE CONTROL INVALID: expected exactly one mutated ldmatrix A address line" \
      "(is $arm1 a swizzled kernel?)" >&2
    exit 1
  fi
  arm1=$broken
fi

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
if ! $negative; then
  exec "$out" --dry-run
fi

errlog=$(mktemp)
trap 'rm -f "$errlog"' EXIT
rc=0
"$out" --dry-run 2>"$errlog" || rc=$?
cat "$errlog" >&2
# The driver's first mismatching cell: mismatch (i,j) want=W A=<arm0> B=<arm1>.
read -r want got0 got1 < <(sed -n 's/^mismatch ([0-9]*,[0-9]*) want=\([^ ]*\) A=\([^ ]*\) B=\([^ ]*\)$/\1 \2 \3/p' "$errlog") || true
if [[ $rc == 1 && -n ${want-} && $got0 == "$want" && $got1 != "$want" ]]; then
  echo "negative control: the broken A address fails the dry run, as designed ($label0 exact, $label1 off)"
  exit 0
fi
if [[ $rc == 0 ]]; then
  echo "NEGATIVE CONTROL FAILED: the broken A address passed the dry run" >&2
else
  echo "NEGATIVE CONTROL INCONCLUSIVE: dry run exited $rc without a $label1-only cell mismatch" >&2
fi
exit 1
