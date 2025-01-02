(lang dune 3.11)

(using ctypes 0.3)

(name ocannl)

(version 0.5.1)

(generate_opam_files true)

(source
 (github lukstafi/ocannl))

(authors "Lukasz Stafiniak")

(maintainers "Lukasz Stafiniak <lukstafi@gmail.com>")

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
 (maintainers "Lukasz Stafiniak <lukstafi@gmail.com>")
 (depends
  ; The real dependency is >= 4.06, but because of multi-package project
  ; need to keep it in sync with the other packages.
  (ocaml
   (>= 5.2.0))
  camlzip
  dune
  (conf-python-3 :with-test)))

(package
 (name arrayjit)
 (synopsis
  "An array language compiler with multiple backends (CPU, CUDA), staged compilation")
 (description
  "The optimizing compiler sub-package of OCANNL. Use neural_nets_lib instead to also get: nice syntax, shape inference, backpropagation, optimizers.")
 (depends
  (ocaml
   (>= 5.2.0))
  dune
  (base
   (>= v0.17.0))
  (ctypes
   (>= 0.23))
  (ctypes-foreign
   (>= 0.23))
  ; opam 2.2.0 has with-dev-setup. Is it supported, what's the syntax?
  ; (ocamlformat
  ;  (>= 0.27.0)
  ;  :with-dev-setup)
  (printbox
   (>= 0.12))
  (printbox-text
   (>= 0.12))
  ocannl_npy
  stdio
  sexplib
  num
  (saturn_lockfree
   (>= 0.5.0))
  ppxlib
  ppx_compare
  ppx_hash
  ppx_here
  ppx_sexp_conv
  ppx_string
  ppx_variants_conv
  ppx_expect
  (ppx_minidebug
   (>= 2.0.3)))
 (depopts
  (cudajit
   (>= 0.6.1))
  (gccjit
   (>= 0.3.2)))
 (conflicts
  (cudajit
   (< 0.6.1))
  (gccjit
   (< 0.3.2)))
 (tags
  (deeplearning array jit CUDA)))

(package
 (name neural_nets_lib)
 (synopsis
  "A from-scratch Deep Learning framework with an optimizing compiler, shape inference, concise syntax")
 (description
  "OCaml Compiles Algorithms for Neural Networks Learning is a compiled Deep Learning framework that puts emphasis on low-level backends (like tinygrad), shape inference, concise notation (ab)using PPX.")
 (depends
  (ocaml
   (>= 5.2.0))
  dune
  (base
   (>= v0.17.0))
  ; opam 2.2.0 has with-dev-setup. Is it supported, what's the syntax?
  ; (ocamlformat
  ;  (>= 0.26.2)
  ;  :with-dev-setup)
  ; Here we mean (arrayjit (= 0.4.1)), but we must be compatible with arrayjit.dev during installation.
  (arrayjit
   (>= 0.5.1))
  (printbox
   (>= 0.12))
  (printbox-text
   (>= 0.12))
  (printbox-ext-plot
   (>= 0.12))
  ocannl_npy
  (angstrom
   (>= 0.15))
  stdio
  sexplib
  num
  time_now
  ppxlib
  ppx_compare
  ppx_fields_conv
  ppx_hash
  ppx_here
  ppx_sexp_conv
  ppx_string
  ppx_variants_conv
  ppx_expect
  (ppx_minidebug
   (>= 2.0.3))
  ; Cannot make it (patdiff :with-test), because `dune build -p neural_nets_lib @install`
  ; builds the tutorials target from dune/test, despite it being inline_tests!
  (patdiff
   (>= v0.15.0))
  (odoc :with-doc)
  (md2mld :with-doc))
 (tags
  (deeplearning tensor backprop jit gccjit CUDA)))

; See the complete stanza docs at https://dune.readthedocs.io/en/stable/dune-files.html#dune-project
