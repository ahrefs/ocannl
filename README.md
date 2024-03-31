# ocannl

OCANNL is sponsored by [Ahrefs](https://ocaml.org/success-stories/peta-byte-scale-web-crawler)! [Visit the Ahrefs website.](https://ahrefs.com/)

## OCANNL -- OCaml Compiles Algorithms for Neural Networks Learning

* A from-scratch, compiled Deep Learning framework.
* Implements backpropagation (i.e. first-order reverse mode autodiff) and shape inference.
* The long-term goal is to provide several "low-level" backends, aiming to seek inspiration from projects such as [TinyGrad](https://github.com/tinygrad/tinygrad), [TVM](https://github.com/apache/tvm), [Luminal](https://github.com/jafioti/luminal).
  * OCANNL starts with a high-level representation, but can compile everything down to `for` loops.
* The library users can compile any amount of code into a monolithic routine. Depending on the use case:
  * the whole training update step can be a single routine,
  * or the step can be composed of a gradient update routine (a forward pass and a backprop pass) and a params update routine (e.g. SGD with momentum, ADAM, etc.),
  * or the user can compile parts of a model separately, manually composing the corresponding forward pass code and the backprop code.
* Tensor axes are split into kinds: batch, input and output. Tensor dimensions have optional labels.
  * The labels ensure a more precise semantics for dimension matching.
  * In the future we might introduce axis labels as an alternative to positional axis selection, it would be a separate naming mechanism.
* OCANNL has full support for the `einsum` notation, integrated with shape inference. Supports static indexing, with a built-in operation to take a slice of the batch axes, integrated with shape inference. Extensible to more static indexing patterns as needs arise.
  * OCANNL does not have dynamic indexing (using the last axis of one tensor as indices into another tensor). If it's needed, it can be added (we had a prototype once, removed to reduce complexity). Then it would also be integrated with shape inference.
* OCANNL has a suite of tutorials doubling as tests with inline expectations.
* OCANNL offers two main levels of abstraction.
  * Differentiable computations, centered around the [`%op`](lib/ppx_op.ml) syntax extension.
  * Plain computations, centered around the [`%cd`](lib/ppx_cd.ml) syntax extension. It integrates the `arrayjit` backend library with shape inference.
* The support for mixed-precision computations is upcoming.
  * E.g. higher-precision network components, or gradients at a higher precision than values.
  * Currently (v0.3), you can select the precision, and individual computation nodes track their precision, but mixing precisions might break things.
* Should be easily extensible.
* Model surgery should be starightforward (not sure if we are there yet).
* It's a feature, not a bug!
  * To scale a tensor by a number, always use pointwise-multiplication, e.g. `2*.m` or `m*.2`.
  * Matrix-multiplying a tensor `m` by a constant number, e.g. `m*2`, broadcasts the number to the shape of the input axes of the tensor. This results in an output-axes-only tensor (multi-axis-vector) that is the scaled sum over the input axes of the tensor `m`.
  * Matrix-multiplying a constant number by a tensor `m`, e.g. `2*m`, broadcasts the number to the shape of the output axes of the tensor. This results in a tensor whose inputs are of the same shape as the inputs of `m`, and the output shape is 1D (scalar), that is the scaled sum over the output axes of the tensor `m`.
  * The matrix-multiply operation behaves pointwise along the batch axes.
  
## Upcoming milestones

On the critical path for the next release:

* Mixed-precision computations: working and convenient.
* Restore signs of life for the Cuda backend.

### Releases

For more details, see [CHANGES](CHANGES.md).

* **v0.3 shape inference, jitted routines**: a major rewrite of the whole project; declarative shape inference; replaced the session interface with a "jitted code routines" API.
* **v0.2 inching toward GPU**. Abandoned design choices.
  * **v0.2.1 naive-cuda**: a Cuda backend where blocks and threads are exposed via dedicated axis types.
  * **v0.2.0 stack-as-device**: treating the C function stack as the "device memory".
* **v0.1 GCCJIT backend**:
  * **v0.1.2**: multicore computations using a thread-local "task id" index.
  * **v0.1.1**: inlining scalar constants, improved inlining for virtual nodes.
  * **v0.1.0**: a `Gccjit` backend, single and double precision floats, code compiled as a monolithic update step function.
* **v0.0 untagged**: basic design around shape inference, high-level and low-level code representation. Now-abandoned Meta-OCaml and OCaml backends.


## Why not just use [OWL](https://ocaml.xyz/)?

OCANNL follows different design choices than [OWL](https://ocaml.xyz/). For example:

* OCANNL is not functorized.
* OCANNL has fewer abstraction layers.
* OCANNL has a more powerful shape inference.
* OCANNL only supports backpropagation, while OWL supports full forward and backward auto-diff.
* Some aspects are more centralized in OCANNL than in OWL and form the "infrastructure":
  * Tensor indexing mechanisms are not extensible, other than changing OCANNL code.
  * Shape inference is fully handled by OCANNL and not extensible, other than changing OCANNL code.
  * [`Tensor`](lib/tensor.ml) implements "putting pieces together".
  * [`Train`](lib/train.ml) has the optimization "frontend" and utilities.
  * [`arrayjit`](arrayjit/), which may one day become a standalone library: generates the code, performs backend-agnostic optimizations (_virtual nodes_ whose computation is inlined), implements the backends.
* Some aspects that are more core to OWL are less encapsulated in OCANNL, so it should be more natural to extend them.
  * Specifically, [`Operation`](lib/operation.ml) and [`Train`](lib/train.ml) are just collections of functions.
* OCANNL provides lower-level compilation backends than OWL, it is more self-contained in this sense.

## Installation

The dependency on `ocaml-cudajit` is optional, so you have to install it first to enable the Cuda backend.
