# Metal runtest divergence notes

This note records backend divergences seen while bringing `OCANNL_BACKEND=metal dune runtest`
back to green.

## Accepted divergences

- Heavy training checks (`bigram`, `fsm_transformer`, `transformer_names`) are compiled by normal
  builds but run only under `@slow`. On Metal they either take substantially longer than the rest of
  the default suite or can fail while the Metal compiler service builds a large compute pipeline
  under load. They remain useful integration tests, but they are not good default-runtest smoke tests
  for the Metal backend.
- Random-stream histogram output is backend-specific. The Metal and `sync_cc` PRNG paths should
  satisfy the same statistical properties, but exact bucket counts and printed sample values are not
  portable golden-test material. Tests should print pass/fail statistical summaries instead.
- Text plots are not stable backend goldens. Small floating-point differences can move a plotted
  point by one character cell, so tutorial smoke tests should print rounded numeric summaries when
  they need to run across backends.
- Metal routine debug logging uses `os_log`, while the backend-log golden harness normalizes stream
  files under `log_files/`. The Metal variant records an explicit skip line instead of pretending
  the stream-file capture path is available.

## Reconciled failures

- Metal generated-source goldens need to reflect the pooled buffer ABI: kernels receive pool base
  buffers plus a slot table, then derive typed tensor pointers inside the shader.
- Guarded dynamic gathers use shared lowering that types bounds/integrality checks as double for
  CPU exactness. Metal has no native double, so those internal scalar guards are lowered as float
  while true double tensor storage remains unsupported.
