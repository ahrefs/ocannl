(** Multidimensional arrays of float32. *)
(* The code is inspired by OWL - OCaml Scientific Computing:
 * Copyright (c) 2016-2022 Liang Wang <liang@ocaml.xyz> *)
module A = Bigarray
type elt = A.float32_elt
type t = (float, elt, c_layout) A.Genarray.t

 