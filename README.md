ocannl
======

## OCaNNL: OCaml Neural Networks Library

* A from-scratch, compiled Deep Learning framework based on MetaOCaml.
* Implements backpropagation (i.e. first-order reverse mode autodiff) and shape inference.
* Tensor axes have optional labels and are split into kinds: batch, input and output.
* Has full support for the `einsum` notation, integrated with shape inference.
* Optionally, can deduce output axes from input axes, e.g. with scaling to make expansion or bottleneck layers auto-adapting to the dimensionality of the data.
* Has a suite of tutorials doubling as tests with inline expectations. `dune runtest`, and `dune promote` if diffs look OK.
* Does not use any external computation libraries. Compiles everything down to `for` loops.
  * The generated code is dynamically linked with the "user land" code.
  * Currently, compiles all computation of a single step of training into two programs: the forward pass and the backpropagation pass.
  * I plan to implement offshoring to CUDA at some point.
* Offers three levels of abstraction:
  * [`Network`](lib/network.ml) for trainable components.
  * [`Operation`](lib/operation.ml) for differentiable computations.
  * [`Node`](lib/node.ml) maintains a store of n-dimensional arrays that the compiled code operates on.
* Does not hide anything. Model surgery should be starightforward (not sure if we are there yet).
* Does not build an explicit computation graph.
  * Instead, directly composes code fragments.
  * Generation of the final code is suspended since it requires results of shape inference. Users can force code generation for intermediate components, for debugging.
  * The generated code should be somewhat readable and debuggable, but is very low-level.

## Installation

Some ideas regarding installation (skip or substitute equivalent actions etc.):
* sudo add-apt-repository ppa:avsm/ppa
* sudo apt update --allow-insecure-repositories
* sudo apt-get install opam
* opam init -a
* opam switch create 4.11.1+BER
* opam remote add metaocaml git+https://github.com/metaocaml/metaocaml-opam.git
* opam install dune
* eval $(opam env)
* opam install printbox printbox-text printbox-html
* opam install ocaml-canvas
* cd ~/ocannl
* dune runtest

## Interesting links to other projects

* Andrej Karpathy's [Micrograd](https://github.com/karpathy/micrograd)
* [JAX autodidax](https://jax.readthedocs.io/en/latest/autodidax.html)