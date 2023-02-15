ocannl
======

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
* Does not hide anything. Model surgery should be starightforward (not sure if we are there yet).

## Installation

Some ideas regarding installation (skip or substitute equivalent actions etc.):
* opam switch create 5.0-flambda ocaml-variants.5.0.0+options ocaml-option-flambda
* eval $(opam env --switch=5.0-flambda)
* opam install dune
* opam install base stdio ppx_jane
* opam install printbox printbox-text printbox-html
* opam install ocaml-canvas
* opam install ocaml-lsp-server ocamlformat
* opam install utop
* opam install magic-trace
* opam install ppx_expect mdx
* eval $(opam env)
* cd ~/ocannl
* dune runtest

## Interesting links to other projects

* Andrej Karpathy's [Micrograd](https://github.com/karpathy/micrograd)
* [JAX autodidax](https://jax.readthedocs.io/en/latest/autodidax.html)