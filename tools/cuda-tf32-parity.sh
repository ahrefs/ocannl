#!/usr/bin/env bash
# Run bench_conv_diag's seed parity (BENCH_PARITY=1, gh-ocannl-1490) on the cifar_resnet
# materialize-all training step, on CUDA under tf32_matmuls, and say in one line what it found
# (gh-ocannl-1705). It is the only check that runs every tensorized conv seed of a real training
# step against the default pipeline (rounded to tf32 as the seed rounds, gh-ocannl-1602):
# bench_seed_parity_smoke covers an in-memory convnet, and on CUDA only cifar_resnet reaches the
# dissolved-fragment case gh-ocannl-1677 fixed. tools/sweep.sh runs this after each CUDA unit whose
# suite ran (its diagnostic legs); about 40 s on rog-nv's RTX 5070 Ti.
#
# Usage: tools/cuda-tf32-parity.sh --fixture-dir DIR   (from any directory; DIR holds
#                                                       cifar_resnet.safetensors, as a box's main
#                                                       checkout's benchmarks/fixtures does)
# The environment passes through: BENCH_PARITY_PERTURB=1 runs the parity's negative control, under
# which every seed must come back a mismatch (bench_conv_diag then exits 0, and this says holds).
#
# The parity's own seed lines come first. The last line on stdout is the verdict, and the exit
# status says the same:
#   cuda-tf32-parity: verdict: holds (...)               0  every seed matched (its summary line)
#   cuda-tf32-parity: verdict: SEED PARITY FAILED (...)  1  a seed mismatched or was unjudged
#   cuda-tf32-parity: verdict: NO SEED EXECUTED          1  the step seeded no conv candidate
#   cuda-tf32-parity: verdict: STEP BROKE (...)          1  bench_conv_diag stopped before a verdict:
#                                                           the model did not build or run
#   cuda-tf32-parity: verdict: unavailable: ...          2  no fixture, or the build failed
# Its boundary: it reads bench_conv_diag's exit status, its `parity (` summary line and its first
# `Fatal error:` line, nothing else of its output.
set -u
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd) || exit 2
cd "$root" || exit 2

verdict() { # status text
  printf 'cuda-tf32-parity: verdict: %s\n' "$2"
  exit "$1"
}

fixtures=
while [ $# -gt 0 ]; do
  case $1 in
    --fixture-dir) fixtures=${2:-}; shift ;;
    *) echo "usage: tools/cuda-tf32-parity.sh --fixture-dir DIR" >&2; exit 2 ;;
  esac
  shift
done
[ -n "$fixtures" ] || { echo "usage: tools/cuda-tf32-parity.sh --fixture-dir DIR" >&2; exit 2; }
fixture=$fixtures/cifar_resnet.safetensors
[ -r "$fixture" ] || verdict 2 "unavailable: no $fixture"

# Dune records the environment of every process it spawns in _build/trace.csexp (gh-ocannl-1280).
# shellcheck source=credential-env.sh
. tools/credential-env.sh || verdict 2 "unavailable: cannot read tools/credential-env.sh"
eval "$(credential_env_scrub_text)" ||
  verdict 2 "unavailable: cannot remove credential variables:$credential_env_left"

work=$(mktemp -d "${TMPDIR:-/tmp}/cuda-tf32-parity.XXXXXX") ||
  verdict 2 "unavailable: no temporary directory"
trap 'rm -rf "$work"' EXIT
# The build first, on its own: a status from it is not the step's.
echo "=== cuda-tf32-parity: build ==="
dune build --no-print-directory benchmarks/runners/ocannl/bench_conv_diag.exe 2>&1 ||
  verdict 2 "unavailable: bench_conv_diag.exe did not build"
echo "=== cuda-tf32-parity: $fixture ==="
OCANNL_BACKEND=cuda OCANNL_TF32_MATMULS=true BENCH_FIXTURE=$fixture BENCH_MATERIALIZE=1 \
  BENCH_PARITY=1 dune exec --no-build --no-print-directory \
  benchmarks/runners/ocannl/bench_conv_diag.exe >"$work/out" 2>&1
rc=$?
cat "$work/out"
summary=$(grep '^parity (' "$work/out" | tail -n 1)
summary=${summary#parity }
fatal=$(grep -m 1 '^Fatal error: ' "$work/out")
case $rc in
  0) verdict 0 "holds (${summary:-no summary line})" ;;
  4) verdict 1 "SEED PARITY FAILED (${summary:-no summary line})" ;;
  5) verdict 1 "NO SEED EXECUTED (${summary:-no summary line})" ;;
  *) verdict 1 "STEP BROKE (bench_conv_diag exited $rc: ${fatal:-no Fatal error line})" ;;
esac
