# Axis-type annotations on Low_level loops

**Date**: 2026-06-12
**Status**: Stub — seeded by the tinygrad deep dive
([a-range-is-not-its-shape](../blog/a-range-is-not-its-shape.md), port area 2); first
site of the decision is [#412](https://github.com/ahrefs/ocannl/issues/412)'s
grid/block mapping.

## Goal

Make the loop-to-hardware mapping explicit in the IR instead of a backend convention:
an axis-type annotation on `For_loop` (or on the loop symbol) in the spirit of
tinygrad's `AxisType` (GLOBAL / LOCAL / THREAD / LOOP / REDUCE / GROUP_REDUCE / UPCAST
/ UNROLL — `tinygrad/uop/ops.py`), so that backends emit grid/block/thread indices for
annotated axes rather than loops, and so that schedule transforms
([schedule-ir-optops](schedule-ir-optops.md)) have a type to assign when splitting.

## Key points from the article's analysis

- Today every OCANNL backend emits single-threaded kernels (CUDA: `grid_dim_x:1,
  block_dim_x:1` plus the `threadIdx.x != 0 || blockIdx.x != 0` guard; Metal: one
  threadgroup). The loop-to-hardware question is unanswered anywhere — annotation vs.
  backend convention is a live choice, and tinygrad's experience argues for annotation.
- The achievable 80% of "thread synchronization": GROUP_REDUCE axes + workgroup
  Barrier + LOCAL-addrspace buffers = shared-memory reductions and tiled matmuls.
- Honest boundary: tinygrad's Barrier is workgroup-scoped and kernels split at
  reduction edges — grid-level persistence/megakernels (#318 write-up) are not solved
  by tinygrad's spec either. This proposal is the staircase, not the summit.
- Addrspace is the companion annotation: LOCAL buffers here, the `__constant__` end in
  [#195](https://github.com/ahrefs/ocannl/issues/195).

## Relations

[#412](https://github.com/ahrefs/ocannl/issues/412) (consumer and first site),
[#195](https://github.com/ahrefs/ocannl/issues/195) (addrspace cousin),
[#318](https://github.com/ahrefs/ocannl/issues/318) (megakernel exploration, landed),
[gh-ocannl-263](gh-ocannl-263.md) (Flash attention — eventual consumer),
[schedule-ir-optops](schedule-ir-optops.md) (depends on this).

## Acceptance criteria (for the elaborated proposal)

- [ ] Annotation carrier decided (`For_loop` field vs. symbol metadata) with migration
      sketch for `to_low_level`/backends.
- [ ] Barrier + LOCAL buffer representation in `Low_level.t` sketched.
- [ ] Backend mapping rules (CUDA grid/block, Metal threadgroups, CC threads) drafted.
