# CUDA runtest divergence notes

This note records backend divergences seen while bringing `OCANNL_BACKEND=cuda dune runtest`
back to green.

## Accepted divergences

- Heavy training checks (`bigram`, `fsm_transformer`, `transformer_names`) are compiled by normal
  builds but run only under `@slow`. CUDA currently runs single-threaded kernels, so these tests can
  take far longer than the rest of the default suite. They remain useful integration tests, but they
  are not good default-runtest smoke tests for the CUDA backend.
- CUDA routine debug stream logs are not stable CPU-style goldens. CUDA lowering inlines more
  expressions into logged statements than `sync_cc`, and the debug printer can read intermediate
  values while formatting expressions that are not real tensor reads in the computation contract.
  The CUDA variant records an explicit skip line instead of relaxing a huge generated log.
- Random-stream sample output is backend-specific at textual precision. The CUDA and `sync_cc` PRNG
  paths should satisfy the same range and distribution properties, but exact rounded sample values
  are not portable golden-test material.

## Reconciled failures

- CUDA generated-source goldens may differ only by formatting choices such as extra parentheses or
  host line endings. These should be normalized or refreshed when the generated code is equivalent.
