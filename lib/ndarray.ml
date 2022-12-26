(** Multidimensional arrays. *)
open Base

(* Note also the OWL library.
 * OCaml Scientific Computing: Copyright (c) 2016-2022 Liang Wang <liang@ocaml.xyz> *)
module A = Bigarray.Genarray
type elt = Bigarray.float32_elt
type t = (float, elt, Bigarray.c_layout) A.t

let pp_print fmt (arr: t) =
  ignore (fmt, arr);
  (* FIXME(13): *)
  Caml.Format.pp_print_string fmt "NOT IMPLEMENTED YET"

 let create = A.create Bigarray.Float32 Bigarray.C_layout
 let empty = create [||]
 
let reset_ones (arr: t) =
  A.fill arr 1.0

let reset_zeros (arr: t) =
    A.fill arr 0.0

let get_val v dims =
  let arr = create dims in
  A.fill arr v;
  arr

let get_uniform ~(low:float) ~(high:float) dims =
  let arr = create dims in
  (* TODO: FIXME: NOT IMPLEMENTED *)
  ignore(low, high);
  arr

let assign_add lhs rhs1 rhs2 =
  (* TODO: FIXME: NOT IMPLEMENTED *)
  ignore (lhs, rhs1, rhs2)

let assign_mul lhs rhs1 rhs2 =
  (* TODO: FIXME: NOT IMPLEMENTED *)
  ignore (lhs, rhs1, rhs2)

let mul dims rhs1 rhs2 =
  let arr = A.create Bigarray.Float32 Bigarray.C_layout dims in
  (* TODO: FIXME: NOT IMPLEMENTED *)
  A.fill arr 1.0;
  ignore(rhs1, rhs2);
  arr

let assign_relu lhs rhs =
  (* TODO: FIXME: NOT IMPLEMENTED *)
  ignore (lhs, rhs)

(** Computes [if rhs1 > 0 then rhs2 else 0]. *)
let relu_gate dims rhs1 rhs2 =
  let arr = A.create Bigarray.Float32 Bigarray.C_layout dims in
  (* TODO: FIXME: NOT IMPLEMENTED *)
  A.fill arr 1.0;
  ignore (rhs1, rhs2);
  arr
