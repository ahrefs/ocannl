(** Multidimensional arrays. *)
open Base

(* Note also the OWL library.
 * OCaml Scientific Computing: Copyright (c) 2016-2022 Liang Wang <liang@ocaml.xyz> *)
module A = Bigarray.Genarray
type elt = Bigarray.float32_elt
type t = (float, elt, Bigarray.c_layout) A.t

 let shape: t -> 'a = A.shape
 let create = A.create Bigarray.Float32 Bigarray.C_layout
 let empty = create [||]
 
let reset_ones (arr: t) =
  A.fill arr 1.0

let reset_zeros (arr: t) =
    A.fill arr 0.0

let get_val v shape =
  let arr = create shape in
  A.fill arr v;
  arr

let get_uniform ~(low:float) ~(high:float) shape =
  let arr = create shape in
  (* TODO: FIXME: NOT IMPLEMENTED *)
  ignore(low, high);
  arr

let assign_add lhs rhs1 rhs2 =
  let shape_l = shape lhs in
  let shape_r1 = shape rhs1 in
  let shape_r2 = shape rhs2 in
  (* TODO: checks not needed *)
  assert (Array.equal (=) shape_l shape_r1);
  assert (Array.equal (=) shape_l shape_r2);
  (* TODO: FIXME: NOT IMPLEMENTED *)
  ()

let assign_mul lhs rhs1 rhs2 =
  let shape_l = shape lhs in
  let shape_r1 = shape rhs1 in
  let shape_r2 = shape rhs2 in
  (* TODO: checks not needed *)
  assert (Array.equal (=) shape_l shape_r1);
  assert (Array.equal (=) shape_l shape_r2);
  (* TODO: FIXME: NOT IMPLEMENTED *)
  ()

let mul rhs1 rhs2 =
  let shape_r1 = shape rhs1 in
  let shape_r2 = shape rhs2 in
  (* TODO: checks not needed *)
  assert (Array.equal (=) shape_r1 shape_r2);
  let arr = A.create Bigarray.Float32 Bigarray.C_layout shape_r1 in
  (* TODO: FIXME: NOT IMPLEMENTED *)
  A.fill arr 1.0;
  arr

let assign_relu lhs rhs =
  let shape_l = shape lhs in
  let shape_r = shape rhs in
  (* TODO: checks not needed *)
  assert (Array.equal (=) shape_l shape_r);
  (* TODO: FIXME: NOT IMPLEMENTED *)
  ()

(** Computes [if rhs1 > 0 then rhs2 else 0]. *)
let relu_gate rhs1 rhs2 =
  let shape_r1 = shape rhs1 in
  let shape_r2 = shape rhs2 in
  (* TODO: checks not needed *)
  assert (Array.equal (=) shape_r1 shape_r2);
  let arr = A.create Bigarray.Float32 Bigarray.C_layout shape_r1 in
  (* TODO: FIXME: NOT IMPLEMENTED *)
  A.fill arr 1.0;
  arr
