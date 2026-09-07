# ocannl

OCANNL is sponsored by [Ahrefs](https://ocaml.org/success-stories/peta-byte-scale-web-crawler)! [Visit the Ahrefs website.](https://ahrefs.com/)

## OCANNL -- OCaml Compiles Algorithms for Neural Networks Learning

* A from-scratch, compiled Deep Learning framework.
* Implements backpropagation (i.e. first-order reverse mode autodiff) and shape inference.
* The long-term goal is to provide several "low-level" backends, aiming to seek inspiration from projects such as [tinygrad](https://github.com/tinygrad/tinygrad), [TVM](https://github.com/apache/tvm), [Luminal](https://github.com/jafioti/luminal).
  * OCANNL starts with a high-level representation, but can compile everything down to `for` loops.
* The library users can compile any amount of code into a routine (i.e. a compilation unit). The user decides explicitly what the scope of a compilation unit is, by putting together the corresponding code. Depending on the use case:
  * the whole training update step can be a single routine,
  * or the step can be composed of a gradient update routine (a forward pass and a backprop pass) and a params update routine (e.g. SGD with momentum, ADAM, etc.),
  * or the user can compile parts of a model separately, manually composing the corresponding forward pass code and the backprop code.
* Tensor axes are split into kinds: batch, input and output. Tensor dimensions have an optional basis.
  * The basis (aka dimension units) ensures a more precise semantics for dimension matching. It's **not** an axis selection mechanism.
* OCANNL has full support for a significantly extended `einsum` notation, integrated with shape inference. See [comparison with einops](docs/einops_comparison.md) for how this relates to the popular [einops](https://einops.rocks/) library. Supports static indexing, with a built-in operation to take a slice of the batch axes, integrated with shape inference. Extensible to more static indexing patterns as needs arise.
  * OCANNL does not have dynamic indexing (using the last axis of one tensor as indices into another tensor). If it's needed, it can be added (we had a prototype once, removed to reduce complexity). Then it would also be integrated with shape inference.
* OCANNL offers two main levels of abstraction.
  * Tensor expressions as differentiable computations, centered around the [`%op`](tensor/ppx_op.ml) syntax extension.
    * `%op` stands for "operation", it's meant to express tensors: `Tensor.t`, and tensor functions.
  * Plain computations, centered around the [`%cd`](tensor/ppx_cd.ml) syntax extension. It integrates the `arrayjit` backend library with shape inference.
    * `%cd` stands for "code", it's meant to express assignment computations: `Assignments.comp`.
* Fully supports mixed-precision computations, with bidirectional precision inference.
  * E.g. higher-precision network components, or gradients at a higher precision than values.
* Should be easily extensible.
* Model surgery should be straightforward (not sure if we are there yet).

## Usage

The CUDA backend requires at least CUDA version 12.8. The Metal backend requires at least MSL version 3.1. The HIP backend (AMD GPUs) requires ROCm / the AMD HIP SDK, via the [hipjit](https://github.com/lukstafi/ocaml-hipjit) bindings (`opam install hipjit`).

[API documentation entry point](https://ahrefs.github.io/ocannl/dev/).

A possible route to learning OCANNL:

1. Read [the introductory slides](https://ahrefs.github.io/ocannl/docs/basics_backprop_training_codegen.html).
2. Read: [shapes and the generalized einsum beginner-to-advanced slides](https://ahrefs.github.io/ocannl/docs/shapes_and_einsum.html).
3. Read [Tensors and Contexts](docs/tensors_and_contexts.md) and, for the runtime API, [`Context`](arrayjit/lib/context.mli).
4. Read [the migration guide](docs/migration_guide.md).
5. Read the syntax extensions documentation [docs/syntax_extensions.md](docs/syntax_extensions.md).
6. Read the NN building blocks file [lib/nn_blocks.ml](lib/nn_blocks.ml) and the training recipes [lib/train.ml](lib/train.ml).
  * Work through the [makemore tutorial](docs/makemore_tutorial.md) — a character-level language-model progression mirroring Andrej Karpathy's *Neural Networks: Zero to Hero* lectures.
7. Read the introductory part of the shape inference documentation [docs/shape_inference.md](docs/shape_inference.md).
8. For the paper-facing account, read the workshop article [docs/ocannl_workshop_article_human.md](docs/ocannl_workshop_article_human.md) (an archival artifact: it was written for the OCaml Workshop / FProPer submission and describes the project as of the 0.8 release; no conference submission is currently scheduled), the formal core technical report [ocannl-formal-core-technical-report.pdf](https://ahrefs.github.io/ocannl/docs/pdfs/ocannl-formal-core-technical-report.pdf) (LaTeX source in [docs/](docs/ocannl-formal-core-technical-report.latex)), and the constraint-generation notes [docs/shape-constraint-generation.md](docs/shape-constraint-generation.md).
9. Skim the configuration documentation [ocannl_config.reference](ocannl_config.reference).
10. Improve your understanding by reading or skimming the framework internals: [tensor/shape.mli](tensor/shape.mli), [tensor/tensor.mli](tensor/tensor.mli), [tensor/operation.ml](tensor/operation.ml), [arrayjit/lib/context.mli](arrayjit/lib/context.mli).
11. Read the implementation overview:
   1. The various tests.
   2. The end-to-end compilation walkthrough [docs/life_of_a_training_step.md](docs/life_of_a_training_step.md) -- one small program traced through every stage from `%op` code to device execution; the map to the deeper documents below.
   3. Shape inference details [docs/shape_inference.md](docs/shape_inference.md).
   4. Backend-independent optimizations [docs/lowering_and_inlining.md](docs/lowering_and_inlining.md) -- _lowering_ means translating (compiling) from the high-level representation (as assignments) to the low-level representation.
   5. Schedules and autotuning [docs/schedules_and_autotuning.md](docs/schedules_and_autotuning.md) -- the loop-nest transform layer (parallelization, tiling, staging, tensor cores) and the empirical search over it.

### Using the tracing debugger with CUDA and HIP computations

To use debugging as provided by configuring `Utils.settings.debug_log_from_routines <- true` with the `cuda` or `hip` backend, wrap the code that schedules work and synchronizes the GPU with `Utils.capture_stdout_logs`. Both GPU APIs expose device-side `printf`, but not `fprintf`; the runtime drains the device printing buffer to process `stdout` around synchronization. Synchronize the context inside the capture window so all device output is available before stdout is restored.

NOTE: debug logging from CUDA or HIP in complex settings is a bit tricky, as it involves another thread (domain) intercepting and filtering `stdout`. If facing issues, try the setting `never_capture_stdout=true` (see [ocannl_config.reference](ocannl_config.reference)).

## Milestones

See [ROADMAP.md](ROADMAP.md) for the detailed schedule, its history of rebalances and renumberings, and the venue history of the paper artifacts. GitHub issue assignments are the source of truth for release scope. **v1.0.1 was released on August 26, 2026**; the next target is **v1.0.2** (robustness pulled forward), September 16, 2026, with **v1.2** targeted for October 28, 2026. Release dates are now project-internal and aspirational — through v1.0 they were pinned to conference deadlines. The version sequence is `0.7 → 0.8 → 0.9 → 1.0 → 1.0.1 → 1.0.2 → 1.1 → 1.1.1 → 1.2`: version-number depth tracks release *scope* (feature releases take a second component, consolidation/robustness releases a third), not semver.

### Releases

For more details, see [CHANGES](CHANGES.md).

* **1.0.1: Consolidation after v1.0 — making a green result mean what it says.**
  * Soundness of inlining: a guarded or looped setter is rejected for virtualization rather than replayed without its `If` or its repetition loop; cross-routine splices declare their leaf reads and are reconciled into the routine interface; `Local_scope` has an enforced purity contract, and a scope over a materialized node is an error rather than a silent collapse.
  * Reduction accumulator width is policy, not schedule, on every backend — with precision-neutral localization worth −74% on the Metal `gpt2_mini` forward step, and batch-grid twins worth 1.72x on the HIP step.
  * Test seams that cannot report a false pass: `Verdict` with its ratchet and quantified claims, generated-kernel provenance, per-test aliases with the `@scans` family, one tracked environment spelling with reserved namespaces, complete and floor-checked repository scans.
  * Benchmark trust: result rows carry search provenance and fixture digests, diverged cells report divergence, and the measurement path has a fixture-free smoke test.
  * Training-loop mechanics: LR schedules, global-norm clipping, gradient accumulation, mmap-backed checkpoint loading (Windows arm measured, not asserted), and `trainable_params`.
  * CPU SIMD: whole-vector FMA (packed f32 GEBP 12.6 → 127.2 GFLOP/s at the default flags), the auto-resolved AVX-512 width (130.5 → 225.7 GFLOP/s on a Zen 5), `Max`/`Min` reductions off the per-lane libm calls, and fp8 codecs exhaustively verified — the host sweep and the CUDA/HIP soak arms in-tree, the Metal codec's bit-identity to `builtins.c` established off-tree.
* **1.0: Advanced compiler tiers and schedule-quality follow-through.**
  * Schedule inference as branch-and-bound: a refinement tree over *partial* schedules, legality verdicts carrying their witnesses, and admissible cost floors, so a subtree is refuted or priced before any of its members is built.
  * Inlining joined scheduling as a searchable decision surface, once the concrete-index tracer was retired in favor of the affine access relations.
  * CUDA/HIP graph capture of the fissioned step, software-pipelined double-buffered staging (`cp.async` on CUDA), budget-driven rematerialization, and the CUDA tensor-core profile's remaining shapes (fp8, `ldmatrix` over swizzled staging).
  * Reduced precision on CPU: 16-bit storage with f32 compute, and native fp16 arithmetic where the hardware has it (~2x on an M-series).
  * The `gpt2_mini` step roughly halved on both GPU backends by judging companion coverage at the site's arity, after a time-attribution profile located 70% of the step in five declined kernels.
  * Config profiles (`reproducible` / `performance`), `cc` worker-pool uniformity on hybrid CPUs, and use-site row resolution narrowed to the leaf-tensor rule it always was.
* **0.9: Program search and optimization.**
  * Native affine program analysis and a `Schedule.op_legality` oracle; an analytic roofline cost model that picks untuned defaults and pre-filters the autotune beam (advisory throughout).
  * Deterministic split reductions, pad-to-tile scheduling (PADTO), static partitioning, epilogue fusion, and launch-time symbolic extents.
  * Convolution schedule families: implicit GEMM via packing `Stage`, blocked tile flavors, epilogue twins, compacting strided-row staging, and clamped-window pooling.
  * Mixed precision: a precision-assignment policy, master weights with cast twins, dynamic loss scaling with a fused on-device gate, tf32 matmuls, and forward-only reduced precision by load-time conversion.
  * Liveness-based buffer aliasing; a packed `uniform` that is total over shapes and now backs default parameter initialization.
  * Search survivability: typed candidate-failure containment across Metal/CUDA/HIP, HIP scratch pre-validation, no unparallelized GPU dispatches, and a decline census that accounts for every refusal.
  * A cross-machine benchmark sweep on Metal, CUDA and HIP, with the reports checked in under `benchmarks/`.
* **0.8: Parallel schedules, autotuning, tensor cores, and the AMD HIP backend.**
  * Automatic GPU schedules: hardware axis types render to grid/block/thread loops with launch dimensions, barriers, and shared-memory tiles, validated against per-backend hardware limits.
  * Kernel fission with aligned cross-nest parallelism; CPU kernel-level parallelism through a thread pool; backends renamed to `cc` / `multidev_cc`.
  * Register-tiled `Tile_mma` microkernels, SIMD vector-extension codegen, and CUDA WMMA / Metal simdgroup-matrix / HIP rocWMMA tensor-core paths.
  * Measured schedule search (`Autotune.tune`) with a digest-guarded cache, per-segment schedules, sketch seeding, and placement A/B tuning.
  * The AMD HIP backend via the independent [hipjit](https://github.com/lukstafi/ocaml-hipjit) bindings, and a cross-framework benchmark suite gated on loss parity.
* **0.7: Frontend finalization, compiler optimizations, and paper-ready formal docs.**
  * Removed hosted tensors in favor of explicit context-mediated access.
  * Added axis concatenation/block tensors, RoPE, the decoder-only transformer toy, ternary einsum, sharding primitives, and zero-copy leading-axis slice views.
  * Added loop hoisting, CSE, broader virtual-node inlining, and the universal pool allocator across backends.
  * Added the workshop article, formal core technical report, and shape-constraint-generation notes.
* **0.6.3: Padding inference for convolutions.**
  * Padding inference during shape inference.
  * Toy CNN example: circle counting.
* **0.6.2: "you forgot to specify a hidden dimension".**
  * Menhir einsum parser.
  * Detection of user errors where there is missing information about a hidden dimension: disables guessing "no axes" or "dimension 1" for shapes of parameters.
* **0.6.1: Syntax extension improvements, transformer building blocks.**
  * Heterogeneous precision operations.
  * Counter-based randomness via threefry, second pass (pointwise and weak-but-efficient variants); normal distribution operation.
  * New syntax for inline parameter definitions; record-based syntax instead of string-based.
  * Add transformer and convnet building blocks.
  * Better shape error messages.
* **0.6: more precisions, initialization, counter-based randomness, strided iteration.**
  * BF16, FP8.
  * Extended expressivity of projections and the generalized einsum notation to cover strided iteration and convolution.
  * Parameter initialization on devices.
  * Counter-based randomness via threefry, first pass (vectorized and cryptographic strength).
  * Better precision inference, including top-down propagation.
* **0.5.3: Apple Metal backend.**
  * Also, CUDA backend works on native Windows.
* **0.5.2: More primitive operations.**
  * Supports a lot of primitive operations (including ternary ops), and ternary tensor operations.
  * `%cd` and `%op` support both curried and uncurried operator application syntax.
  * More flexible gradient construction via the `%cd` syntax (better projections inference).
  * Works on Native Windows with the C compiler backend (but CUDA backend blocked by cudajit still).
* **0.5.1: Automatic synchronization and transfers between host and devices.**
* **0.5.0: Stream-to-stream synchronization at the buffer level.**
  * Support for CUDA events, and `Condition`-based events for CPU backends.
  * Overhaul of the backend interfaces, both user-facing but especially internal: full code sharing.
  * Automatic stream-to-stream synchronization on a per-tensor-node basis.
* **0.4.1 Half precision, mixed precision, CUDA virtual devices** (virtual devices renamed to streams in 0.5.0)
  * Half precision. Maybe improvements for mixed-precision computations.
  * Resolve remaining issues with the new scheduler.
  * Initial version of [lib/nn_blocks.ml](lib/nn_blocks.ml).
* **v0.4 Merge buffers, C-syntax backend builder**: a significant refactoring of the API.
* **v0.3 Shape inference, jitted routines**: a major rewrite of the whole project.
  * **v0.3.3**: continuous integration and opam release.
  * **v0.3.2**: new shape inference feature: tracking leftmost axes -- complete inference for splicing, ellipsis-in-the-middle allowed in einsum notation.
  * **v0.3.1**: sanitizing code inclusion (rootness checks).
  * **v0.3.0**: declarative shape inference; replaced the session interface with a "jitted code routines" API. Cuda defunct.
* **v0.2 Inching toward GPU**:
  * **v0.2.1 naive-cuda**: a Cuda backend where blocks and threads are exposed via dedicated axis types.
  * **v0.2.0 stack-as-device**: treating the C function stack as the "device memory".
* **v0.1 GCCJIT backend**:
  * **v0.1.2**: multicore computations using a thread-local "task id" index.
  * **v0.1.1**: inlining scalar constants, improved inlining for virtual nodes.
  * **v0.1.0**: a `Gccjit` backend, single and double precision floats, code compiled as a monolithic update step function.
* **v0.0 Untagged**: basic design around shape inference, high-level and low-level code representation. Now-abandoned Meta-OCaml and OCaml backends.

## Why not just use [OWL](https://ocaml.xyz/)?

OCANNL follows different design choices than [OWL](https://ocaml.xyz/). For example:

* OCANNL is not functorized, except that it uses first-class modules for backends.
* OCANNL has fewer abstraction layers.
* OCANNL has a more powerful shape inference.
* OCANNL only supports backpropagation, while OWL supports full forward and backward auto-diff.
* Some aspects are more centralized in OCANNL than in OWL and form the "infrastructure":
  * Tensor indexing mechanisms are not extensible, other than changing OCANNL code.
  * Shape inference is fully handled by OCANNL and not extensible, other than changing OCANNL code.
  * [`Tensor`](tensor/tensor.ml) implements "putting pieces together".
  * [`Train`](lib/train.ml) has the optimization "frontend" and utilities.
  * [`arrayjit`](arrayjit/), which may one day become a standalone library: generates the code, performs backend-agnostic optimizations (_virtual nodes_ whose computation is inlined), implements the backends.
* Some aspects that are more core to OWL are less encapsulated in OCANNL, so it should be more natural to extend them.
  * Specifically, [`Operation`](tensor/operation.ml) and [`Train`](lib/train.ml) are just collections of functions.
* OCANNL provides lower-level compilation backends than OWL, it is more self-contained in this sense.

## Installation

Although the project is called `ocannl`, the main package is called `neural_nets_lib`, to avoid the (opam linter's) complaint that the name can be confused with other packages. This also clarifies that `ocannl` is composed of `arrayjit` and `neural_nets_lib`.

The dependency on `cudajit` is optional so you have to install it first to enable the CUDA backend. The dependency on `metal` is MacOS-specific but automatic.

### Code Organization

The codebase is organized to separate user-facing recipes from framework internals:

- **`lib/`**: User-facing recipes and utilities
  - `train.ml` - Training utilities and optimizers
  - `nn_blocks.ml` - Neural network building blocks (transformers, attention, convolution, etc.)
  - `ocannl.ml` - Re-exports for backward compatibility
  
- **`tensor/`**: Framework internals (separate library `ocannl_tensor`)
  - `tensor.ml/mli` - Core tensor type and operations
  - `shape.ml/mli` - Shape inference system  
  - `operation.ml` - Tensor operations and DSL modules
  - `ppx_*.ml` - Syntax extensions implementation
  
- **`arrayjit/`**: Low-level optimizing compiler with multiple backends

## Development

NOTE TO POTENTIAL CONTRIBUTORS: while I ~~am~~ might be slowly starting to work with PRs in separate branches rather than just a stream of commits on the main branch, design migrations will be broken into small PRs to avoid main (master) branch staleness; and many changes will still be commits on the main branch. We allow for failing tests on the main branch, although going forward this would hopefully be happening less. Tagged i.e. released versions of the code are guaranteed to work as well as the given stage of the project permitted, the policy is that all tests must pass for releases with the backend `cc` and must have the behavior expected of a backend with all other backends. We try to minimize discrepancy across backends but prefer more stringent tests even if some backends only pass them "in spirit" rather than with exact expectations of the `cc` backend.

**Developing on macOS**: the `cc` backend links each compiled kernel as a fresh shared
library and `dlopen`s it, and macOS scans every freshly created binary the first time it is
mapped for execution (Gatekeeper/XProtect). On a default setup this dominates test and
autotune wall time — one autotune test measured 12x slower than on comparable Linux hardware,
almost all of it the scanner. The fix is one setting: add your terminal (and any app that runs
builds for you, e.g. an AI-agent desktop app) to System Settings → Privacy & Security →
**Developer Tools** ("Allow the apps below to run software locally that does not meet the
system's security policy"). The exemption covers every process those apps spawn; nothing in
OCANNL needs configuring. The Metal backend is unaffected either way — its kernels compile
in-process, not through dylibs.

OCANNL uses [`ppx_minidebug`](https://github.com/lukstafi/ppx_minidebug) for debugging. Currently, we migrated to a per-file opt-in scheme for enabling ppx_minidebug at compile time (via environment variables, see the top of `.ml` files in question), and then a unified log level configuration (`ocannl_log_level`) for tuning logging at runtime. Due to the compile-time nature of the per-file settings, run `dune clean` after setting/exporting one of these environment variables.
