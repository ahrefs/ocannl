(lang dune 3.6)

(using ctypes 0.2)

(name ocannl)

(version 0.2.2)

(generate_opam_files true)

(source
 (github lukstafi/ocannl))

(authors "Lukasz Stafiniak")

(maintainers "Lukasz Stafiniak")

(license LICENSE)

(documentation https://github.com/lukstafi/ocannl/blob/master/README.md)

(package
 (name ocannl)
 (synopsis
  "A from-scratch Deep Learning library with CUDA, operator fusion, staged compilation, backprop")
 (description "A longer description")
 (depends
  ocaml
  dune
  base
  core
  ctypes
  ctypes-foreign
  printbox
  printbox-text
  npy
  stdio
  ppxlib
  ppx_jane
  ppx_expect
  ppx_minidebug)
 (tags
  (deeplearning backprop jit gccjit CUDA)))

; See the complete stanza docs at https://dune.readthedocs.io/en/stable/dune-files.html#dune-project
