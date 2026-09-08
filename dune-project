(lang dune 3.20)

; 3.20 is the floor the repository's own test workflow needs, not a wish: dune generates the
; per-test `runtest-<name>` aliases only from 3.20, and the targeted-test arrangement
; (gh-ocannl-726) and the focused aggregate families (gh-ocannl-783) are built on them. The `dune`
; constraint in the generated opam files follows this line, so declaring 3.18 while requiring 3.20
; promised a toolchain on which `@metal-codegen` reaches nothing.

(using ctypes 0.3)

(using dune_site 0.1)

(using directory-targets 0.1)

(using menhir 3.0)

(name ocannl)

(version 1.0.1)

(generate_opam_files true)

(source
 (github lukstafi/ocannl))

(authors "Lukasz Stafiniak")

(maintainers "Lukasz Stafiniak <lukstafi@gmail.com>")

(maintenance_intent "(latest)")

(license "BSD-2-Clause")

(documentation https://ahrefs.github.io/ocannl/docs/)

; We give up on npy / ocannl_npy for now.

(package
 (name arrayjit)
 (synopsis
  "An array language compiler with multiple backends (CPU, CUDA, HIP, Metal), staged compilation")
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
  ; Development tooling, behind opam's `with-dev-setup` filter: `opam install . --deps-only
  ; --with-dev-setup` installs it, a plain `--deps-only` (and so every CI solve) does not, which is
  ; why a fresh switch has no ocamlformat until asked. ocamlformat is pinned to the release that
  ; `.ocamlformat` names -- that file makes any other release refuse to run, this line makes opam
  ; install the one it accepts; move the two together (a mismatch fails loudly at the first `@fmt`).
  (ocamlformat
   (and
    (= 0.29.0)
    :with-dev-setup))
  (ocaml-lsp-server :with-dev-setup)
  (printbox
   (>= 0.12))
  ; FIXME: Depend on printbox-text > 0.12.0 when it is released.
  (printbox-text
   (>= 0.12))
  (pprint
   (>= 20230830))
  stdio
  sexplib
  num
  (saturn_lockfree
   (>= 0.5.0))
  ; Bounded ABOVE as well, because what this project depends on is not ppxlib's API but the AST
  ; that API selects -- `Astlib.Ast_502` throughout the permitted range. Every fixed shape the ppx
  ; matches is now written as a metaquot quotation, which ppxlib regenerates for whichever AST it
  ; selects; what still names constructors is mapped class by class, with the fix each class
  ; needs, at the head of `tensor/ppx_shared.ml`. The source scanners under `test/support` match
  ; a handful more.
  ;
  ; Three classes actually move, checked against Ast_503..Ast_506: literal values at `Ast_503`,
  ; where `constant` becomes a record `{ pconst_desc; pconst_loc }`; tuple payloads at `Ast_504`,
  ; which adds labelled tuples; and `Pexp_open`/`Pexp_letmodule` at `Ast_505`, which deletes both
  ; in favour of `Pexp_struct_item`. Only the last is more than a rename. `Pexp_function`'s
  ; params/constraint/body triple, named here as a mover before it was checked, is in fact
  ; byte-identical through Ast_506.
  ;
  ; A ppxlib that selects a later AST does not break these subtly: they stop compiling, which is
  ; the loud failure and not the dangerous one -- but it stops them compiling in a solve nothing
  ; warned about, which is what an upper bound is for. 0.39 is announced as the release that moves
  ; the selection.
  ;
  ; Deliberately NOT resolved by pinning `Astlib.Ast_502` directly and migrating into it: ppxlib
  ; generates `Ast_traverse` and metaquot for the SELECTED ast alone, so targeting a versioned one
  ; costs both the iterators and the quotation syntax, across every site. Raising this bound is a
  ; deliberate edit that recompiles the ppx; that is the right shape for the decision.
  (ppxlib
   (and
    (>= 0.36.0)
    (< 0.39.0)))
  ppx_compare
  ppx_enumerate
  ppx_hash
  ppx_here
  ppx_sexp_conv
  ppx_string
  ppx_variants_conv
  ppx_expect
  (notty-community
   (>= 2.4.0))
  (ppx_minidebug
   (>= 3.0.0))
  (mtime
   (>= 2.0.0))
  (metal
   (= :os macos)))
 (depopts
  (cudajit
   (>= 0.8.0))
  (hipjit
   (>= 0.2.0)))
 (conflicts
  (cudajit
   (< 0.8.0))
  (hipjit
   (< 0.2.0)))
 (tags
  (deeplearning array jit CUDA HIP Metal)))

(package
 (name neural_nets_lib)
 (synopsis
  "A from-scratch Deep Learning framework with an optimizing compiler, shape inference, concise syntax")
 (description
  "OCaml Compiles Algorithms for Neural Networks Learning is a compiled Deep Learning framework that puts emphasis on low-level backends (like tinygrad), shape inference, concise notation (ab)using PPX.")
 (depends
  (ocaml
   (>= 5.3.0))
  dune
  dune-site
  (base
   (>= v0.17.0))
  ; The same development tooling as arrayjit above, same pin, same filter.
  (ocamlformat
   (and
    (= 0.29.0)
    :with-dev-setup))
  (ocaml-lsp-server :with-dev-setup)
  (arrayjit
   (= :version))
  (dataprep
   (>= 0.1.0))
  (yojson
   (>= 2.0.0))
  (printbox
   (>= 0.12))
  ; FIXME: Depend on printbox-text > 0.12.0 when it is released.
  (printbox-text
   (>= 0.12))
  (printbox-ext-plot
   (>= 0.12))
  (menhir
   (>= 20240715))
  (pprint
   (>= 20230830))
  stdio
  sexplib
  num
  time_now
  ; Bounded above for the AST the ppx and the source scanners match; see the note in the arrayjit
  ; stanza above, and the residue map at the head of `tensor/ppx_shared.ml`.
  (ppxlib
   (and
    (>= 0.36.0)
    (< 0.39.0)))
  ppx_compare
  ppx_enumerate
  ppx_fields_conv
  ppx_hash
  ppx_here
  ppx_sexp_conv
  ppx_string
  ppx_variants_conv
  ppx_expect
  (notty-community
   (>= 2.4.0))
  (ppx_minidebug
   (>= 3.0.0))
  (odoc :with-doc)
  ; No slipshow: the slide decks are compiled by the prebuilt binary that
  ; gh-pages-docs.yml downloads, not by an opam-installed one, so the
  ; dependency bought nothing while dragging js_of_ocaml-compiler -- and hence
  ; a `ocaml < 5.5` bound -- into every --with-doc solve.
  (md2mld :with-doc))
 (tags
  (deeplearning tensor backprop jit CUDA Metal)))

; See the complete stanza docs at https://dune.readthedocs.io/en/stable/dune-files.html#dune-project
