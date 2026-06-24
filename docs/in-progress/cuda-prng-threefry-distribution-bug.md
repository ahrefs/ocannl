# Follow-up: CUDA threefry/PRNG produces a broken distribution on-device

**Status**: Open. Filed from task-bfc7c7b5 (CUDA backend bring-up) after the backend was made to
build and run on the real device (RTX 3050 Ti, WSL CUDA). This is a **pre-existing** CUDA codegen
bug — the CUDA backend never built before, so the PRNG path was never exercised on-device. It is
**not** related to the pool-allocator addressing work that task-bfc7c7b5 fixed (commits
`ebca9212`, `aed1e3a6`); general compute, reductions, matmuls and training all pass on CUDA.

## Symptom

The counter-based (threefry) PRNG produces a **statistically broken** distribution on the CUDA
device. `test/operations/test_random_histograms.ml` does statistical (not bit-exact) assertions and
fails hard:

- Uniform: `Chi-square statistic: 475.26 (df=19, critical value at 0.05: ~30.14)` — i.e. wildly
  non-uniform (a correct PRNG, even with a different sequence, gives chi-square ~10–30).
- Normal: `Within 2 std dev %% (expected: ~95.4, tolerance: 2.00): FAIL (got 100.0000)` — no tails;
  `Overall: SOME TESTS FAILED`.
- Kaiming: `Mean: 1.12 (expected: ~0.12)` — off by ~10×.

The clustering/no-tails pattern suggests the per-element threefry counter is not varying correctly
across output elements on CUDA (values come out correlated), rather than a wrong-but-uniform map.

## Affected tests (all under `OCANNL_BACKEND=cuda`)

1. `test/operations/test_random_histograms` — statistical assertions fail (primary evidence).
2. `test/operations/test_uniform1` — uniform values differ from the (CPU-generated) `.expected`.
3. `test/operations/threefry4x32_demo` — threefry output differs.
4. `test/operations/test_block_tensor` — random-init values differ (downstream).
5. `test/operations/test_param_shape_error` — random-init param render differs (downstream).
6. `test/einsum/moons_demo_variant` — random weight init differs (downstream).

## Reproduction

```sh
export PATH="/usr/lib/wsl/lib:$PATH"        # WSL: exposes nvidia-smi / libcuda
eval $(opam env)
OCANNL_BACKEND=cuda dune build test/operations/test_random_histograms.exe
OCANNL_BACKEND=cuda (cd test/operations && ../../_build/default/test/operations/test_random_histograms.exe)
# or the whole fast suite:
OCANNL_BACKEND=cuda dune runtest test/operations test/einsum
```

## Likely location

`arrayjit/lib/builtins_cuda.ml` — the "light threefry" bit-spreading and `uint32_to_*_uniform`
conversions (lines ~15–60, 226–355), and/or how the per-element counter is fed into
`arrayjit_threefry4x32_*` in the CUDA codegen. Compare against the CPU implementation in
`builtins.c` and the known-answer inline test `test/operations/test_threefry4x32.ml`.

## Acceptance for the fix

`OCANNL_BACKEND=cuda dune runtest test/operations/test_random_histograms.exe` passes its statistical
assertions (chi-square below critical, normal tails within tolerance), and the bit-exact PRNG tests
either match CPU or get justified `.cu.expected` fixtures once the distribution is correct.
