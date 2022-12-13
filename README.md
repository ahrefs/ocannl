ocannl
======

## OCaNNL: OCaml Neural Networks Library

The most bare-bones from-scratch implementation based on MetaOCaml.
Inspired by Andrej Karpathy's [Micrograd](https://github.com/karpathy/micrograd) and by [JAX](https://jax.readthedocs.io/en/latest/autodidax.html).

MetaOCaml's generated code can be both run directly (encapsulating the steps: natively compile, dynlink and invoke), or it can be offshored, saved as text, etc. It offers debuggability and lots of flexibility going forward.

OCaNNL's codeflow is as follows:
1. Construct a model, represented by a loss formula composed of appropriate subformulas all the way down to parameters. Inputs are just parameters that the training loop treats specially. The primitives are in the module [`Formula`](lib/formula.ml). The subformulas store code pieces that are not compiled (not `Runcode.run`ed) yet.
2. MetaOCaml's `Runcode.run` combines compiling and executing the code. Executing code of a `Formula` specializes it to particular [`Ndarray`](lib/ndarray.ml) dimensions, and outputs the procedures for forward inference and backward gradient propagation.
3. A step of the training loop sets the inputs, invokes the "forward" and "backprop" procedures, and updates the parameters (adding to a parameter's `value`, its `grad` scaled by the learning rate). Gradient zeroing is performed automatically by the "backprop" procedure.
4. The low-level computations are encapsulated by [`Ndarray`](lib/ndarray.ml). I plan to implement them in CUDA.

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
* opam install ocaml-canvas
* cd ~/ocannl
* dune exec ocannl
