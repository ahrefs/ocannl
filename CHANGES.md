## [0.1.1] -- 2023-05-06

### Added

- An _inline constants_ optimization that compile-time computes scalar constant subexpressions and inlines the values.

### Changed

- Improved debuggability.

### Fixed

- A last-minute breaking bug (would be nice to have a pre-release or a pre-publish hook to run tests!).
- The virtual nodes optimization is more robust, correct even with aggressive inlining settings (e.g. escaping variables check).

## [0.1.0] -- 2023-05-04

### Added

- The first changes-tracking release. Earlier development history is still somewhat documented via closed issues.
- Supports single and double precision floats, more precisions in the future.
- Generates a monolithic step update routine executed by `refresh_session ()`, but can generate arbitrary additional routines at arbitrary times to be executed at arbitrary other times within a session.
- An `Interpreter` backend that can for example log all individual tensor modifications.
- A `Gccjit` backend that can sometimes be 400x faster than the `Interpreter` backend (without any debug work/output).
- A _virtual nodes (tensors)_ optimization that inlines computation of a cell in lieu of tensor accesses, can sometimes reduce memory consumption by 1/3.
