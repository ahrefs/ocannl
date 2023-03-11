ocannl
======

Warning disclaimer: this project is still "not announced". The features described might not be implemented yet.

## OCaNNL: OCaml Neural Networks Library

* A from-scratch, compiled Deep Learning framework.
* Implements backpropagation (i.e. first-order reverse mode autodiff) and shape inference.
* Tensor axes have optional labels and are split into kinds: batch, input and output.
* Has full support for the `einsum` notation, integrated with shape inference.
* Supports mixed-precision computations, e.g. higher-precision network components, or gradients at a higher precision than values.
* Optionally, can deduce output axes from input axes (and vice-versa TODO), e.g. with scaling to make expansion or bottleneck layers auto-adapting to the dimensionality of the data.
* Has a suite of tutorials doubling as tests with inline expectations. `dune runtest`, and `dune promote` if diffs look OK.
* Does not (need to) use any external computation libraries.
  * Starts with a high-level representation, but can compile everything down to `for` loops.
  * Has multiple "backends": interpreted, compiled via OCaml, compiled via pure C, compiled via CUDA.
  * Currently, compiles all computation of a single step of training into two programs: the forward pass and the backpropagation pass.
* Offers three levels of abstraction:
  * [`Network`](lib/network.ml) for trainable components.
  * [`Operation`](lib/operation.ml) for differentiable computations.
  * [`Code`](lib/code.ml) for computations, [`Node`](lib/node.ml) maintains a store of n-dimensional arrays that the code operates on.
* Does not hide anything. Should be easily extensible.
* Model surgery should be starightforward (not sure if we are there yet).
* It's a feature, not a bug!
  * To scale a tensor by a number, always use pointwise-multiplication, e.g. `2*.m` or `m*.2`.
  * Matrix-multiplying a tensor `m` by a constant number, e.g. `m*2`, broadcasts the number to the shape of the input axes of the tensor. This results in an output-axes-only tensor (multi-axis-vector) that is the scaled sum over the input axes of the tensor `m`.
  * Matrix-multiplying a constant number by a tensor `m`, e.g. `2*m`, broadcasts the number to the shape of the output axes of the tensor. This results in a tensor whose inputs are of the same shape as the inputs of `m`, and the output shape is 1D (scalar), that is the scaled sum over the output axes of the tensor `m`.
  * The matrix-multiply operation behaves pointwise along the batch axes.
  

## Installation

Some ideas regarding installation (skip or substitute equivalent actions etc.):
* opam switch create 5.0-flambda ocaml-variants.5.0.0+options ocaml-option-flambda
* eval $(opam env --switch=5.0-flambda)
* opam install lsp ocamlformat
* cd ~/ocannl
* opam install . --deps-only
* eval $(opam env)
* dune runtest

## Interesting links to other projects

* Andrej Karpathy's [Micrograd](https://github.com/karpathy/micrograd)
* [JAX autodidax](https://jax.readthedocs.io/en/latest/autodidax.html)