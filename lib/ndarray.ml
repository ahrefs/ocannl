(** Multidimensional arrays of float32. *)
(* The code is inspired by OWL - OCaml Scientific Computing:
 * Copyright (c) 2016-2022 Liang Wang <liang@ocaml.xyz> *)
module A = Bigarray.Genarray
type elt = Bigarray.float32_elt
type t = (float, elt, Bigarray.c_layout) A.t

 let dims: t -> 'a = A.dims
 let create = A.create Bigarray.Float32 Bigarray.C_layout
 let empty = create [||]
 
let create_ones =
  let v = A.create Bigarray.Float32 Bigarray.C_layout in
  (* TODO: FIXME: NOT IMPLEMENTED *)
  v
