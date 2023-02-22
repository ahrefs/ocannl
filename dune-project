(lang dune 3.6)

(name ocannl)

(version 0.0.1)

(generate_opam_files true)

(source
 (github lukstafi/ocannl))

(authors "Lukasz Stafiniak")

(maintainers "Lukasz Stafiniak")

(license LICENSE)

(documentation https://github.com/lukstafi/ocannl/blob/master/README.md)

(package
 (name ocannl)
 (synopsis "A from-scratch Deep Learning library with CUDA, operator fusion, staged compilation, backprop")
 (description "A longer description")
 (depends ocaml dune base core printbox printbox-text stdio ppx_jane ppx_expect)
 (tags
  (deeplearning backprop CUDA)))

; See the complete stanza docs at https://dune.readthedocs.io/en/stable/dune-files.html#dune-project
