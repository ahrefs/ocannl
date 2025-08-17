(lang dune 3.18)

(using ctypes 0.3)
(using dune_site 0.1)

(using directory-targets 0.1)

(name ocannl)

(version 0.6.0)

(generate_opam_files true)

(source
 (github lukstafi/ocannl))

(authors "Lukasz Stafiniak")

(maintainers "Lukasz Stafiniak <lukstafi@gmail.com>")

(maintenance_intent "(latest)")

(license "BSD-2-Clause")

(documentation https://github.com/lukstafi/ocannl/blob/master/README.md)

; We give up on npy / ocannl_npy for now.

(package
 (name arrayjit)
 (synopsis
  "An array language compiler with multiple backends (CPU, CUDA, Metal), staged compilation")
 (description
  "The optimizing compiler sub-package of OCANNL. Use neural_nets_lib instead to also get: nice syntax, shape inference, backpropagation, optimizers.")
 (depends
  (ocaml
   (>= 5.3.0))
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
  ; FIXME: Depend on printbox-text > 0.12.0 when it is released.
  (printbox-text
   (>= 0.12))
  pprint
  stdio
  sexplib
  num
  (saturn_lockfree
   (>= 0.5.0))
  (ppxlib (>= 0.36.0))
  ppx_compare
  ppx_hash
  ppx_here
  ppx_sexp_conv
  ppx_string
  ppx_variants_conv
  ppx_expect
  (ppx_minidebug
   (>= 2.3.0))
  (metal
   (= :os macos)))
 (depopts
  (cudajit
   (>= 0.7.0)))
 (conflicts
  (cudajit
   (< 0.7.0)))
 (tags
  (deeplearning array jit CUDA Metal)))

(package
 (name neural_nets_lib)
 (synopsis
  "A from-scratch Deep Learning framework with an optimizing compiler, shape inference, concise syntax")
 (description
  "OCaml Compiles Algorithms for Neural Networks Learning is a compiled Deep Learning framework that puts emphasis on low-level backends (like tinygrad), shape inference, concise notation (ab)using PPX.")
 (sites (share data))
 (depends
  (ocaml
   (>= 5.3.0))
  dune
  dune-site
  (base
   (>= v0.17.0))
  ; opam 2.2.0 has with-dev-setup. Is it supported, what's the syntax?
  ; (ocamlformat
  ;  (>= 0.26.2)
  ;  :with-dev-setup)
  ; Here we mean (arrayjit (= 0.5.3)), but we must be compatible with arrayjit.dev during installation.
  (arrayjit
   (>= 0.5.3))
  (printbox
   (>= 0.12))
  ; FIXME: Depend on printbox-text > 0.12.0 when it is released.
  (printbox-text
   (>= 0.12))
  (printbox-ext-plot
   (>= 0.12))
  (angstrom
   (>= 0.15))
  stdio
  sexplib
  num
  csv
  curl
  time_now
  camlzip
  (ppxlib (>= 0.36.0))
  ppx_compare
  ppx_fields_conv
  ppx_hash
  ppx_here
  ppx_sexp_conv
  ppx_string
  ppx_variants_conv
  ppx_expect
  (ppx_minidebug
   (>= 2.3.0))
  (odoc :with-doc)
  (md2mld :with-doc))
 (tags
  (deeplearning tensor backprop jit CUDA Metal)))

; See the complete stanza docs at https://dune.readthedocs.io/en/stable/dune-files.html#dune-project
