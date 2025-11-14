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
* Tensor axes are split into kinds: batch, input and output. Tensor dimensions have optional labels.
  * The labels ensure a more precise semantics for dimension matching, alternative name: dimension units. It's **not** an axis selection mechanism.
* OCANNL has full support for a significantly extended `einsum` notation, integrated with shape inference. Supports static indexing, with a built-in operation to take a slice of the batch axes, integrated with shape inference. Extensible to more static indexing patterns as needs arise.
  * OCANNL does not have dynamic indexing (using the last axis of one tensor as indices into another tensor). If it's needed, it can be added (we had a prototype once, removed to reduce complexity). Then it would also be integrated with shape inference.
* OCANNL offers two main levels of abstraction.
  * Tensor expressions as differentiable computations, centered around the [`%op`](tensor/ppx_op.ml) syntax extension.
    * `%op` stands for "operation", it's meant to express tensors: `Tensor.t`, and tensor functions.
  * Plain computations, centered around the [`%cd`](tensor/ppx_cd.ml) syntax extension. It integrates the `arrayjit` backend library with shape inference.
    * `%cd` stands for "code", it's meant to express assignment computations: `Assignments.comp`.
* Fully supports mixed-precision computations, with bidirectional precision inference.
  * E.g. higher-precision network components, or gradients at a higher precision than values.
* Should be easily extensible.
* Model surgery should be starightforward (not sure if we are there yet).

## Usage

The CUDA backend requires at least CUDA version 12.8. The Metal backend requires at least MSL version 3.1.

[API documentation entry point](https://ahrefs.github.io/ocannl/dev/).

A possible route to learning OCANNL:

1. Read [the introductory slides](https://ahrefs.github.io/ocannl/docs/basics_backprop_training_codegen.html).
2. Read: [shapes and the generalized einsum beginner-to-advanced slides](https://ahrefs.github.io/ocannl/docs/shapes_and_einsum.html).
3. Upcoming in v0.7: slides about [`Context`](arrayjit/lib/context.mli).
4. Read [the migration guide](docs/migration_guide.md).
5. Read the syntax extensions documentation [docs/syntax_extensions.md](docs/syntax_extensions.md).
6. Read the NN building blocks file [lib/nn_blocks.ml](lib/nn_blocks.ml) and the training recipes [lib/train.ml](lib/train.ml).
7. Read the introductory part of the shape inference documentation [docs/shape_inference.md](docs/shape_inference.md).
8. Skim the configuration documentation [ocannl_config.example](ocannl_config.example).
9. Improve your understanding by reading or skimming the framework internals: [tensor/shape.mli](tensor/shape.mli), [tensor/tensor.mli](tensor/tensor.mli), [tensor/operation.ml](tensor/operation.ml), [arrayjit/lib/context.mli](arrayjit/lib/context.mli).
10. Read the implementation overview:
   1. The various tests.
   2. Shape inference details [docs/shape_inference.md](docs/shape_inference.md).
   3. Backend-independent optimizations [docs/lowering_and_inlining.md](arrayjit/lib/lowering_and_inlining.md) -- _lowering_ means translating (compiling) from the high-level representation (as assignments) to the low-level representation.

### Using the tracing debugger with CUDA computations

To use debugging as provided by configuring `Utils.settings.debug_log_from_routines <- true` with the `cuda` backend, you need to wrap the code scheduling tasks and synchronizing `cuda` devices with `Utils.capture_stdout_logs`. The reason is that CUDA kernels are allowed to use `printf`, but not `fprintf` -- the driver dumps the printing buffer of a device to `stdout` at certain times (e.g. when synchronizing the device). For an example, see the implementation of `Train.example_train_loop`. Specifically, it wraps two sections: the call to `Train.parallel_update`, and the body of the returned `infer_callback`.

NOTE: debug logging from CUDA in complex settings is a bit tricky, it involves another thread (domain) intercepting and filtering `stdout`. If facing issues, try the setting `never_capture_stdout=true` (see [ocannl_config.example](ocannl_config.example)).

## Upcoming milestones

This is very tentative.

* **0.6.2: "you forgot to specify a hidden dimension".**
  * Detection of user errors where there is missing information about a hidden dimension: disables guessing "no axes" or "dimension 1" for shapes of parameters.
  * RoPE embeddings.
  * Transformer for the Names dataset.
* **0.6.3: Padding inference for convolutions; HIP backend.**
  * Padding inference during shape inference.
  * A standalone bindings package for HIP (AMD hardware), and a backend using the bindings.
  * Sokoban RL policy gradient example with a CNN.
* **0.6.4: Real world examples.**
  * Add convnet examples: MNIST and CIFAR.
  * Bindings to a tokenizer (e.g. _llama.cpp_).
  * Transformer inference for a small open-weights model (one of GPT2, LLaMA, Gemma).
* **0.7: CPU-style performance and memory efficiency.**
  * Cleanup of deprecated streams functionality.
  * Migrating from the "hosted tensor" idea to always requiring a context when accessing tensors and dealing with devices directly.
  * Optimizations: loop invariant lifting and common subexpression elimination.
  * Universal Pool Allocator.
  * Milestone phrasing: Enhancements for: inlining-related and simplification-related optimizations, memory management, session management.
* **0.8: GPU-style performance -- low hanging fruit.**
  * First harvested from [Fast Multidimensional Matrix Multiplication on CPU from Scratch](https://siboehm.com/articles/22/Fast-MMM-on-CPU).
  * Then harvested from [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog](https://siboehm.com/articles/22/CUDA-MMM).
  * Finally from [llm.c](https://github.com/karpathy/llm.c).
  * These will either require splitting a routine into multiple kernels, or implementing the megakernel approach.
  * Milestone phrasing: GPU tiling and related optimizations in the polyhedral style, with heuristic syntactic metrics for now.
* **0.8.1: shape understanding and manipulation enhancements.**
  * Verify or rethink usefulness of dimension labels aka. dimension units, and whether to introduce axis labels.
  * Add concatenation to the einsum syntax (an axis that isq a concatenation of two axes each from another tensor); it's a generalization of stacking tensors.
  * An academic-style paper e.g. for the OCaml Workshop.
* **0.9: Optimize performance: program search.**
  * Instead of dynamic scheduling as in tinygrad, we can schedule statically by program search.
  * We should also reproduce the search that tinygrad is doing. Inspiration: Halide.
  * Check which optimizations are missing against the implementation of [llm.c](https://github.com/karpathy/llm.c).
  * Milestone phrasing: Program search with execution-based per-backend or aggregate-of-backends cost functions. Starting with augmenting the tiling and layout mechanisms from v0.8 with cost functions, progressing to a broader range of code graph rewriting rules.
* **1.0: Few documentation gaps, some degree of feature completeness, ergonomics, safety.**
  * Feature completeness demonstrated by resolving / implementing a few of the $\color{green}{\text{explore}}$ issues.
  * Concise syntax for transfers into the merge buffer since we know which tensor node is transferred and where to.
  * Similarly to how contexts track initialization dependencies for compilation, we should also track them for execution.

### Releases

For more details, see [CHANGES](CHANGES.md).

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

NOTE TO POTENTIAL CONTRIBUTORS: while I ~~am~~ might be slowly starting to work with PRs in separate branches rather than just a stream of commits on the main branch, design migrations will be broken into small PRs to avoid main (master) branch staleness; and many changes will still be commits on the main branch. We allow for failing tests on the main branch, although going forward this would hopefully be happening less. Tagged i.e. released versions of the code are guaranteed to work as well as the given stage of the project permitted, the policy is that all tests must pass for releases with the backend `sync_cc` and must have the behavior excpected of a backend with all other backends. We try to minimize discrepancy across backends but prefer more stringent tests even if some backends only pass them "in spirit" rather than with exact expectations of the `sync_cc` backend.

OCANNL uses [`ppx_minidebug`](https://github.com/lukstafi/ppx_minidebug) for debugging. Currently, we migrated to a per-file opt-in scheme for enabling ppx_minidebug at compile time (via environment variables, see the top of `.ml` files in question), and then a unified log level configuration (`ocannl_log_level`) for tuning logging at runtime. Due to the compile-time nature of the per-file settings, run `dune clean` after setting/exporting one of these environment variables.
