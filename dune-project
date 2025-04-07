(lang dune 3.16)

(using ctypes 0.3)

(name ocannl)

(version 0.5.2)

(generate_opam_files true)

(source
 (github lukstafi/ocannl))

(authors "Lukasz Stafiniak")

(maintainers "Lukasz Stafiniak <lukstafi@gmail.com>")

(license "BSD-2-Clause")

(documentation https://github.com/lukstafi/ocannl/blob/master/README.md)

; We give up on npy / ocannl_npy for now.

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
   (>= 2.2.0)))
 (depopts
  (cudajit
   (>= 0.7.0))
  (gccjit
   (>= 0.3.2)))
 (conflicts
  (cudajit
   (< 0.7.0))
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
  ; Here we mean (arrayjit (= 0.5.2)), but we must be compatible with arrayjit.dev during installation.
  (arrayjit
   (>= 0.5.2))
  (printbox
   (>= 0.12))
  (printbox-text
   (>= 0.12))
  (printbox-ext-plot
   (>= 0.12))
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
  ; ppx_minidebug 2.2.0 will be a breaking change.
  (ppx_minidebug
   (>= 2.2.0))
  (odoc :with-doc)
  (md2mld :with-doc))
 (tags
  (deeplearning tensor backprop jit gccjit CUDA)))

; See the complete stanza docs at https://dune.readthedocs.io/en/stable/dune-files.html#dune-project
