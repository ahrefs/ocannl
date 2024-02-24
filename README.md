# ocannl

Warning disclaimer: this project is still "not announced". The features described might not be implemented yet.

## OCANNL -- OCaml Compiles Algorithms for Neural Networks Learning

* A from-scratch, compiled Deep Learning framework.
* Implements backpropagation (i.e. first-order reverse mode autodiff) and shape inference.
* Tensor axes are split into kinds: batch, input and output. Tensor dimensions have optional labels that act like .
* Has full support for the `einsum` notation, integrated with shape inference. Supports static indexing, with a built-in operation to take a slice of the batch axes, integrated with shape inference. Extensible to more static indexing patterns as needs arise.
  * Does not have dynamic indexing (using the last axis of one tensor as indices into another tensor). If there is demand, it can be added (had a prototype once, removed to reduce complexity). Then it would also be integrated with shape inference.
* Optionally, can deduce output axes from input axes (and vice-versa TODO), e.g. with scaling to make expansion or bottleneck layers auto-adapting to the dimensionality of the data.
* Has a suite of tutorials doubling as tests with inline expectations.
* Does not (need to) use any external computation libraries.
  * Starts with a high-level representation, but can compile everything down to `for` loops.
  * Has support multiple backends: compiled via pure C, compiled via CUDA.
  * Currently, compiles all computation of a single update step into a monolithic routine. But users can compile any additional routines at any time.
* Offers only two levels of abstraction.
  * Differentiable computations, centered around the [`%op`](lib/ppx_op.ml) syntax extension.
  * Plain computations, centered around the [`%cd`](lib/ppx_cd.ml) syntax extension. It integrates the `arrayjit` backend library with shape inference.
* Supports mixed-precision computations, e.g. higher-precision network components, or gradients at a higher precision than values.
* Should be easily extensible.
* Model surgery should be starightforward (not sure if we are there yet).
* It's a feature, not a bug!
  * To scale a tensor by a number, always use pointwise-multiplication, e.g. `2*.m` or `m*.2`.
  * Matrix-multiplying a tensor `m` by a constant number, e.g. `m*2`, broadcasts the number to the shape of the input axes of the tensor. This results in an output-axes-only tensor (multi-axis-vector) that is the scaled sum over the input axes of the tensor `m`.
  * Matrix-multiplying a constant number by a tensor `m`, e.g. `2*m`, broadcasts the number to the shape of the output axes of the tensor. This results in a tensor whose inputs are of the same shape as the inputs of `m`, and the output shape is 1D (scalar), that is the scaled sum over the output axes of the tensor `m`.
  * The matrix-multiply operation behaves pointwise along the batch axes.
  
## Future milestones

* **v0.4 usability**: convolutional networks with shape inference support; examples covering most of Andrej Karpathy's "Neural Networks Zero to Hero" series; data loading; checkpointing.
* **v0.5 CPU performance**: optimizations targetting the CPU.
* **v0.6 CUDA backend**: major milestone, "tensor fission".
* **v0.7 LLVM and Triton?**:
  * **v0.7.1 triton-C**: a Triton backend?
  * **v0.7.2 llvm**: an LLVM backend as an alternative to the GCCJIT backend?
  * **v0.7.3 triton-llvm**: an LLVM-based Triton backend?
* **v0.8 documentation**: more `.mli` files and maybe more documentation.
* **v0.9 scale**: model parallelism; data ingestion; maybe basic distributed computation; maybe autotuning optimization settings.
* **v1 completeness**: whatever not-yet-implemented features that still seem needed and impact the framework design. (E.g. at the time of v0.1.X, convolutions, reshaping, concatenation are not easily expressible.)

### Releases

For details, see [CHANGES](CHANGES.md).

* **v0.3 shape inference**: a major rewrite of the whole project; declarative shape inference; removed the session interface.
* **v0.2 inching toward GPU**:
  * **v0.2.0 stack-as-device**: for multicore CPU, improve cache locality and reduce cache contention by treating the C function stack as the "device memory".
  * **v0.2.1 naive-cuda**: a Cuda backend where "task id" becomes parallelization over blocks, and a new dedicated axis "sample num" becomes parallelization over threads in a block.
  * Abandoned, in an archival branch.
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
  * Shape inference is fully handled by [`Shape`](lib/shape.ml).
  * [`Tensor`](lib/tensor.ml) implements "putting pieces together".
  * [`Train`](lib/train.ml) has the optimization "frontend" and utilities.
  * [`arrayjit`](arrayjit/), which may one day become a standalone library: generates the code, performs backend-agnostic optimizations (_virtual nodes_ whose computation is inlined), implements the backends.
* Some aspects that are more core to OWL are "delegated to user-land" in OCANNL.
  * [`Operation`](lib/operation.ml) is just a bunch of functions, what users implementing new computational primitives would do.
  * Specific network architectures, e.g. MLP, CNN, Transformer, can hopefully be concisely expressed and belong to individual projects in OCANNL -- while it seems to me they are more part of the library in OWL. In this regard working on new architectures is not impeded by OCANNL.
  * But the enabling mechanisms, such as "generalized `einsum`", belong to the OCANNL library/infrastructure. In this regard OCANNL is less extensible.
* OCANNL provides lower-level compilation backends than OWL, it is more self-contained in this sense.

## Installation

There is no dependency on `ocaml-cudajit`, so you have to install it first to enable the Cuda backend.

## Interesting links to other projects

* Andrej Karpathy's [Micrograd](https://github.com/karpathy/micrograd)
* [Tinygrad](https://github.com/tinygrad)
* [JAX autodidax](https://jax.readthedocs.io/en/latest/autodidax.html)
* [Fast GPT: GPT-2 inference in Fortran](https://github.com/certik/fastGPT/), [picoGPT: code-golf GPT-2 inference in NumPy](https://github.com/jaymody/picoGPT)

## Memory model

A tensor consists of a value tensor node, a forward computation logic for the value (can be no-op if it's a data tensor), and, if the tensor is differentiable, a gradient tensor node and a backprop computation logic.

Memory materialization levels for tensor nodes: virtual, local, device-only, hosted.

* Virtual: the computations are inlined, and cached only on a per-scalar basis.
* Local: the computations happen during, and are guaranteed to be cached a for the duration of, a single function call.
* Device-only: the tensor node is stored on devices that computed it, persists across function calls to the device if the functions share a relevant ancestor context.
* Hosted: the tensor node is stored in a way visible to the CPU, can be visualized and stored to disk.
