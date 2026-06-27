# Follow-up: CUDA codegen produces wrong results for specific einsum / projection / conv patterns

**Status**: Open. Filed from task-bfc7c7b5 (CUDA backend bring-up) after the backend was made to
build and run on the real device (RTX 3050 Ti, WSL CUDA). This is a **pre-existing** CUDA codegen
bug — the backend never built before, so these patterns were never exercised on-device. It is
**not** the pool-allocator addressing issue task-bfc7c7b5 fixed (commits `ebca9212`, `aed1e3a6`):
the inputs below are **deterministic** (no PRNG), general matmul/reduction/backprop pass on CUDA
(training tests meet their loss/accuracy thresholds), and `cuda_pool_offset` round-trips correctly.

## Symptom

A few specific einsum / projection / convolution patterns compute **wrong values on CUDA** with
deterministic inputs, while passing under `sync_cc`.

1. `test/operations/test_virtual_diagonal` — input `TDSL.range 5`; expected an identity matrix of
   1.0 with the diagonal carrying `[1;2;3;4;5]`; CUDA returns a shifted/wrong diagonal (e.g. `0.00`
   at `[0,0]`, `2.00` at `[2,2]` instead of `3.00`). Deterministic → a codegen bug in the virtual
   diagonal projection.
2. `test/operations/test_einsum3` — deterministic 3-way `einsum3 "ij;jk;km=>im"` on literal inputs
   (`[1;2;3;4]`, `[0;1;1;0]`, `[1;0;0;1]`); CUDA returns wrong values (e.g. a reduce branch giving
   `[20 40]` where `[12 43]` is expected).
3. `test/einsum/test_conv_padding` — input `TDSL.range_of_shape [5;5;1]` convolved; CUDA returns
   wrong values (e.g. first row `9.65 7.15 6.65 5.15` vs expected `5.30 1.05e1 4.57 4.77`).

## Reproduction

```sh
export PATH="/usr/lib/wsl/lib:$PATH"
eval $(opam env)
OCANNL_BACKEND=cuda dune runtest test/operations/test_virtual_diagonal.exe
OCANNL_BACKEND=cuda dune runtest test/operations/test_einsum3.exe
OCANNL_BACKEND=cuda dune runtest test/einsum/test_conv_padding.exe
# or the whole fast suite:
OCANNL_BACKEND=cuda dune runtest test/operations test/einsum
```

## Likely location

CUDA codegen for diagonal/projection and convolution einsum patterns: the projection/indexing
lowering in `arrayjit/lib/` (`indexing.ml` projections, `low_level.ml`, and the CUDA emission in
`cuda_backend.ml` / `c_syntax.ml`). Since `sync_cc` is correct and CUDA is wrong on identical
deterministic inputs, the divergence is CUDA-specific index/loop emission for these patterns, not
the high-level shape inference.

## Acceptance for the fix

`OCANNL_BACKEND=cuda dune runtest test/operations/test_virtual_diagonal.exe
test/operations/test_einsum3.exe test/einsum/test_conv_padding.exe` all match their (backend-neutral)
`.expected` output on the real device.
