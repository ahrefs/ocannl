#!/usr/bin/env bash
# Build and dry-run the precompiled CUDA A/B driver over two emitted kernels (gh-ocannl-1190).
#
#   mma_register_scope_probe_build.sh [--negative-control[=a|b]] ARM0.cu ARM1.cu OUTPUT LABEL0 LABEL1
#
# ARM0/ARM1 are kernels exported by bench_mma_register_scope_emit (any revision, any layout);
# the driver reports ARM1 over ARM0 under the labels, which are required: they are what the
# output says was compared (gh-ocannl-1073 timed per_block vs resident, gh-ocannl-1190 plain vs
# swizzled), and no default can know which. This script pins the nvcc flags, prints the toolchain and the SHA-256 of
# both sources and the binary, the load-path census of each source, compiles, and runs the
# driver's --dry-run, which checks both arms against every exact host product cell. Run it under
# a correctness reservation; run OUTPUT without --dry-run only in an exclusive timing window.
# `dune build @benchmarks/runners/ocannl/mma-register-scope-probe` (CUDA only) exports both
# layouts and runs this script plainly and under both negative controls, as one target.
#
# --negative-control proves that dry run can fail. ARM1 must be a swizzled kernel; the script
# writes OUTPUT.broken-<operand>.cu, a copy with one operand's swizzle XOR dropped, builds the
# driver over ARM0 and that copy, and succeeds only when the dry run fails the way the fault
# predicts: a cell where ARM0 equals the exact host product and the broken arm does not. A passing
# dry run, or any other failure, exits 1. OUTPUT is then the broken driver: never time it.
#   =a (the default) drops the XOR from the ldmatrix A address (exactly one line changes). The
#      fault permutes K within an A row, so it is visible only because B varies along K.
#   =b drops it from every byte of the B gather (exactly the two lines building __mma_b0 and
#      __mma_b1 change). B is K-row-major, so the XOR is keyed by the K row and permutes N within
#      it: the fault reads another output column's bytes on odd K rows, not the same terms in
#      another order.
set -euo pipefail

usage="usage: $0 [--negative-control[=a|b]] ARM0.cu ARM1.cu OUTPUT LABEL0 LABEL1"
negative=
case ${1-} in
  --negative-control | --negative-control=a) negative=a; shift ;;
  --negative-control=b) negative=b; shift ;;
  --negative-control=*) echo "$usage" >&2; exit 2 ;;
esac
if [[ $# -ne 5 ]]; then
  echo "$usage" >&2
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

if [[ -n $negative ]]; then
  # Swizzle_b128 emits the 16-byte chunk index as ((col >> s) ^ (row & m)) << s; dropping the XOR
  # term feeds the faulted operand's loads unswizzled addresses into the swizzled tile.
  case $negative in
    a) line='__cvta_generic_to_shared\(__mma_ap ' what='ldmatrix A address' lines=1 ;;
    b) line='unsigned __mma_b[01] = \(unsigned\)__mma_bp\[' what='B gather' lines=2 ;;
  esac
  broken=$out.broken-$negative.cu
  LINE=$line perl -pe 's/\^ \(\(.*?\) & \d+\)\) (<< \d+\))/^ 0) $1/g if /$ENV{LINE}/' \
    "$arm1" >"$broken"
  mutated=$(diff "$arm1" "$broken" | grep -c '^>' || true)
  # Every XOR on the faulted lines is gone: a B gather line carries one per byte.
  remaining=$(perl -ne 'print if /$ENV{LINE}/ && /\^ \(\(/' "$broken" | wc -l | tr -d ' ')
  echo "negative_control=$negative source=$arm1 broken=$broken mutated_lines=$mutated"
  if [[ $mutated != "$lines" || $remaining != 0 ]]; then
    echo "NEGATIVE CONTROL INVALID: expected exactly $lines mutated $what line(s) with no XOR" \
      "left, got $mutated with $remaining still swizzled (is $arm1 a swizzled kernel?)" >&2
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
if [[ -z $negative ]]; then
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
  echo "negative control: the broken $what fails the dry run, as designed ($label0 exact, $label1 off)"
  exit 0
fi
if [[ $rc == 0 ]]; then
  echo "NEGATIVE CONTROL FAILED: the broken $what passed the dry run" >&2
else
  echo "NEGATIVE CONTROL INCONCLUSIVE: dry run exited $rc without a $label1-only cell mismatch" >&2
fi
exit 1
