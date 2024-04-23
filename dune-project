(lang dune 3.11)

(using ctypes 0.3)

(name ocannl)

(version 0.3.2)

(generate_opam_files true)

(source
 (github lukstafi/ocannl))

(authors "Lukasz Stafiniak")

(maintainers "Lukasz Stafiniak")

(license "BSD-2-Clause")

(documentation https://github.com/lukstafi/ocannl/blob/master/README.md)

; While we wait to upstream Npy.Npz.restore, we vendor Npy. Alternative:
; (pin
;  (url "git+https://github.com/lukstafi/npy-ocaml#master")
;  (package
;   (name npy)
;   (version 0.0.9)))

(package
 (name ocannl_npy)
 (synopsis "Numpy file format support for ocaml")
 (authors "Laurent Mazare")
 (maintainers "Laurent Mazare <lmazare@gmail.com>")
 (depends
  (ocaml
   (>= 4.06))
  camlzip
  (dune :build)
  (conf-python-3 :with-test)))

(package
 (name arrayjit)
 (synopsis
  "An array language compiler with multiple backends (CPU, Cuda), staged compilation")
 (description
  "The optimizing compiler sub-package of OCANNL. Use OCANNL instead to also get: nice syntax, shape inference, backpropagation, optimizers.")
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
  printbox
  printbox-text
  ocannl_npy
  stdio
  num
  ppxlib
  ppx_jane
  ppx_expect
  (ppx_minidebug
   (>= 1.5)))
 (depopts cudajit)
 (tags
  (deeplearning array jit gccjit CUDA)))

(package
 (name ocannl)
 (synopsis
  "A from-scratch Deep Learning framework with an optimizing compiler, shape inference, concise syntax")
 (description
  "OCaml Compiles Algorithms for Neural Networks Learning is a compiled Deep Learning framework that puts emphasis on low-level backends (like TinyGrad), shape inference, concise notation (ab)using PPX.")
 (depends
  (ocaml
   (>= 5.1.0))
  dune
  base
  core
  arrayjit
  printbox
  printbox-text
  ocannl_npy
  angstrom
  stdio
  num
  ppxlib
  ppx_jane
  ppx_expect
  (ppx_minidebug
   (>= 1.5))
  (odoc :with-doc)
  (md2mld :with-doc))
 (tags
  (deeplearning tensor backprop jit gccjit CUDA)))

; See the complete stanza docs at https://dune.readthedocs.io/en/stable/dune-files.html#dune-project
