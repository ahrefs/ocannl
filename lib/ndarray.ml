open Base

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

  let assign_add lhs rhs1 rhs2 =
    let dims_l = dims lhs in
    let dims_r1 = dims rhs1 in
    let dims_r2 = dims rhs2 in
    (* TODO: checks not needed *)
    assert (Array.equal (=) dims_l dims_r1);
    assert (Array.equal (=) dims_l dims_r2);
    (* TODO: FIXME: NOT IMPLEMENTED *)
    ()