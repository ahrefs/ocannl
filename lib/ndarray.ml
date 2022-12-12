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
 
let reset_ones (arr: t) =
  A.fill arr 1.0

let reset_zeros (arr: t) =
    A.fill arr 0.0

let assign_add lhs rhs1 rhs2 =
  let dims_l = dims lhs in
  let dims_r1 = dims rhs1 in
  let dims_r2 = dims rhs2 in
  (* TODO: checks not needed *)
  assert (Array.equal (=) dims_l dims_r1);
  assert (Array.equal (=) dims_l dims_r2);
  (* TODO: FIXME: NOT IMPLEMENTED *)
  ()

let assign_mul lhs rhs1 rhs2 =
  let dims_l = dims lhs in
  let dims_r1 = dims rhs1 in
  let dims_r2 = dims rhs2 in
  (* TODO: checks not needed *)
  assert (Array.equal (=) dims_l dims_r1);
  assert (Array.equal (=) dims_l dims_r2);
  (* TODO: FIXME: NOT IMPLEMENTED *)
  ()

let mul rhs1 rhs2 =
  let dims_r1 = dims rhs1 in
  let dims_r2 = dims rhs2 in
  (* TODO: checks not needed *)
  assert (Array.equal (=) dims_r1 dims_r2);
  let arr = A.create Bigarray.Float32 Bigarray.C_layout dims_r1 in
  (* TODO: FIXME: NOT IMPLEMENTED *)
  A.fill arr 1.0;
  arr

let assign_relu lhs rhs =
  let dims_l = dims lhs in
  let dims_r = dims rhs in
  (* TODO: checks not needed *)
  assert (Array.equal (=) dims_l dims_r);
  (* TODO: FIXME: NOT IMPLEMENTED *)
  ()

(** Computes `if rhs1 > 0 then rhs2 else 0`. *)
let relu_gate rhs1 rhs2 =
  let dims_r1 = dims rhs1 in
  let dims_r2 = dims rhs2 in
  (* TODO: checks not needed *)
  assert (Array.equal (=) dims_r1 dims_r2);
  let arr = A.create Bigarray.Float32 Bigarray.C_layout dims_r1 in
  (* TODO: FIXME: NOT IMPLEMENTED *)
  A.fill arr 1.0;
  arr
