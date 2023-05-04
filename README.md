ocannl
======

Warning disclaimer: this project is still "not announced". The features described might not be implemented yet.

## OCANNL -- OCaml Compiles Algorithms for Neural Networks Learning

* A from-scratch, compiled Deep Learning framework.
* Implements backpropagation (i.e. first-order reverse mode autodiff) and shape inference.
* Tensor axes have optional labels and are split into kinds: batch, input and output.
* Has full support for the `einsum` notation, integrated with shape inference.
* Optionally, can deduce output axes from input axes (and vice-versa TODO), e.g. with scaling to make expansion or bottleneck layers auto-adapting to the dimensionality of the data.
* Has a suite of tutorials doubling as tests with inline expectations. `dune runtest`, and `dune promote` if diffs look OK.
* Does not (need to) use any external computation libraries.
  * Starts with a high-level representation, but can compile everything down to `for` loops.
  * Has multiple "backends": interpreted, compiled via OCaml, compiled via pure C, compiled via CUDA.
  * Currently, compiles all computation of a single step of training into two programs: the forward pass and the backpropagation pass.
* Offers only two levels of abstraction.
  * Differentiable computations, centered around the [`%nn_op`](lib/ppx_nn_op.ml) syntax extension.
  * Plain computations, centered around the [`%nn_cd`](lib/ppx_nn_cd.ml) syntax extension.
  * Both abstraction levels share infrastructure. [`Formula.t`](lib/formula.ml) represent tensors, and are usually potentially differentiable (we call them _form_ formulas), but need not be (_non-form_ formulas). _non-form_ (non-differentiable) formulas cannot be subformulas of differentiable formulas. The [`%nn_cd`](lib/ppx_nn_cd.ml) syntax can be used to build up _non-form_ formulas, but also to express "primitive/glue" computations ([`Code.t`](lib/code.ml)) that do not introduce new tensors.
* Supports mixed-precision computations, e.g. higher-precision network components, or gradients at a higher precision than values.
* Should be easily extensible.
* Model surgery should be starightforward (not sure if we are there yet).
* It's a feature, not a bug!
  * To scale a tensor by a number, always use pointwise-multiplication, e.g. `2*.m` or `m*.2`.
  * Matrix-multiplying a tensor `m` by a constant number, e.g. `m*2`, broadcasts the number to the shape of the input axes of the tensor. This results in an output-axes-only tensor (multi-axis-vector) that is the scaled sum over the input axes of the tensor `m`.
  * Matrix-multiplying a constant number by a tensor `m`, e.g. `2*m`, broadcasts the number to the shape of the output axes of the tensor. This results in a tensor whose inputs are of the same shape as the inputs of `m`, and the output shape is 1D (scalar), that is the scaled sum over the output axes of the tensor `m`.
  * The matrix-multiply operation behaves pointwise along the batch axes.
  
## Why not just use [OWL](https://ocaml.xyz/)?

OCANNL follows different design choices than [OWL](https://ocaml.xyz/). For example:
* OCANNL is not functorized.
* OCANNL has fewer abstraction layers.
* OCANNL has arguably a more powerful shape inference.
* OCANNL only supports backpropagation, while OWL supports full forward and backward auto-diff.
* Some aspects are more centralized in OCANNL than in OWL and form the "infrastructure", with less of an intention to be extended or even read by end-users:
  * Shape inference is fully handled by [`Shape`](lib/shape.ml).
  * [`Formula`](lib/formula.ml) implements "putting pieces together".
  * [`Session`](lib/session.ml) implements the session logic.
* Some aspects that are more core to OWL are "delegated to user-land" in OCANNL.
  * [`Operation`](lib/operation.ml) is just a bunch of functions, what users implementing new computational primitives would do.
  * Specific network architectures, e.g. MLP, CNN, Transformer, can be concisely formulated and belong to individual projects in OCANNL -- while ti seems to me they are more part of the library in OWL. In this regard working on new architectures is not impeded by OCANNL.
  * But the enabling mechanisms, such as "generalized `einsum`", belong to the OCANNL library/infrastructure. In this regard OCANNL is less extensible.
* OCANNL provides lower-level compilation backends than OWL, it is more self-contained in this sense.

## Installation

Some ideas regarding installation (skip or substitute equivalent actions etc.):
* `gcc --version`, then install `libgccjit-`version`-dev`
* opam switch create 5.0-flambda ocaml-variants.5.0.0+options ocaml-option-flambda
* eval $(opam env --switch=5.0-flambda)
* opam install lsp ocaml-lsp-server ocamlformat
* gh repo clone savonet/ocaml-mem_usage; cd ocaml-mem_usage; dune build; dune install
* cd ~/ocannl
* opam install . --deps-only
* eval $(opam env)
* dune runtest

## Interesting links to other projects

* Andrej Karpathy's [Micrograd](https://github.com/karpathy/micrograd)
* [JAX autodidax](https://jax.readthedocs.io/en/latest/autodidax.html)
* [Fast GPT: GPT-2 inference in Fortran](https://github.com/certik/fastGPT/), [picoGPT: code-golf GPT-2 inference in NumPy](https://github.com/jaymody/picoGPT)