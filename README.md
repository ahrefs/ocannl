ocannl
======

OCaNNL: OCaml Neural Networks and NLP Library

The most bare-bones from-scratch implementation based on MetaOCaml.
Tries to be minimalistic. Progresses from matrix multiplication using "einsum" to transformers.

Some ideas regarding installation (skip or substitute equivalent actions where using other systems):
* sudo add-apt-repository ppa:avsm/ppa
* sudo apt update --allow-insecure-repositories
* sudo apt-get install opam
* opam init -a
* opam switch create 4.11.1+BER
* opam remote add metaocaml git+https://github.com/metaocaml/metaocaml-opam.git
* opam install dune
* eval $(opam env)
* pip install matplotlib
* opam install matplotlib
* sudo apt-get install qt5-assistant
* pip install PyQt5
* sudo apt-get install pyplot*
* opam install pyplot
* cd ~/ocannl
* dune exec ocannl
