# Evaluate CUDA `__constant__` memory

Issue: [#195](https://github.com/ahrefs/ocannl/issues/195)
ROADMAP: v0.8 stretch item not taken up; now backlog

## Current state

CUDA kernels are parallel and routinely use staged, packed, and tensor-core
schedules. Constants already live in per-device constant pools and hoisted
`Stage` can pack suitable operands once. CUDA `__constant__` memory is used
only for fixed ThreeFry tables.

The old proposal mapped every small `known_constant` tensor into module
globals. That is too broad:

- constant memory helps warp-uniform addresses but serializes divergent
  addresses;
- its 64 KiB budget is per module, while OCANNL batches and fissions kernels;
- module-global storage complicates context isolation and relinking;
- removing a pointer parameter is not itself a performance win worth this
  machinery.

## Goal

Establish whether any current OCANNL workload benefits from CUDA constant
memory, then implement only the profitable, statically safe subset.

## Direction

1. Build a microbenchmark comparing the current constant-pool pointer path
   with a module `__constant__` symbol for uniform and divergent warp access.
2. Use generated-kernel access information, not memory-mode intent alone, to
   identify warp-uniform reads. Embedded lookup tables or small hoisted
   constants are better first candidates than arbitrary model parameters.
3. If the benchmark wins, make placement a CUDA codegen decision with a
   per-module budget and ordinary-pointer fallback. Population must happen
   before any kernel in that module runs and remain isolated across linked
   contexts.
4. Record placement/decline diagnostics so autotuning and benchmark reports
   reveal which path actually rendered.

## Completion criteria

- Measurements cover uniform and divergent access on supported CUDA hardware.
- Any implemented path is guarded by access pattern, size, lifetime, and
  module budget—not merely `Tnode.known_constant`.
- Multi-context and batched/fissioned linking have correctness tests.
- A representative workload shows a repeatable gain; otherwise close the
  proposal with the negative result.
