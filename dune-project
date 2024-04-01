(lang dune 3.11)

(using ctypes 0.3)

(name ocannl)

(version 0.3.0)

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
  (ocaml
   (>= 5.1.0))
  dune
  base
  core
  ctypes
  ctypes-foreign
  (gccjit
   (>= 0.3.2))
  cudajit
  printbox
  printbox-text
  npy
  stdio
  num
  ppxlib
  ppx_jane
  ppx_expect
  (ppx_minidebug
   (>= 1.5)))
 (tags
  (deeplearning backprop jit gccjit CUDA)))

; See the complete stanza docs at https://dune.readthedocs.io/en/stable/dune-files.html#dune-project
