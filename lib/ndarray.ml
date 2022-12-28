(** Multidimensional arrays. *)
open Base

(* Note also the OWL library.
 * OCaml Scientific Computing: Copyright (c) 2016-2022 Liang Wang <liang@ocaml.xyz> *)
module A = Bigarray.Genarray
type elt = Bigarray.float32_elt
type t = (float, elt, Bigarray.c_layout) A.t

(** Prints 0-based [indices] entries out of [arr], where [-1] in an axis means to print out the axis,
    and a non-negative index means to print out only the indexed dimension of the axis. Up to [4] axes
    can be [-1]. Prints up to [entries_per_axis] or [entries_per_axis+1] entries per axis. If
    [order_of_axes] specifies the layout of the printed matrices (default [1, 3, 2, 4]), and should
    be a list of up to 4 numbers in range [1] to the length of the list. [labels] specifies the labels
    (if any) of the printed out axes. *)
let pp_print fmt ?(entries_per_axis=5) ?(order_of_axes=[1; 2; 4; 3]) ?(labels=[]) ~indices (arr: t) =
  let dims = A.dims arr in
  let indices = Array.copy indices in
  let entries_per_axis = if entries_per_axis % 2 = 0 then entries_per_axis + 1 else entries_per_axis in
  let var_indices =
  match Array.filter_mapi indices ~f:(fun i d -> if d = -1 then Some i else None) with
    | [||] -> [-1; -1; -1; -1]
    | [|ind1|] -> [-1; -1; -1; ind1]
    | [|ind1; ind2|] -> [-1; -1; ind1; ind2]
    | [|ind1; ind2; ind3|] -> [-1; ind1; ind2; ind3]
    | [|ind1; ind2; ind3; ind4|] -> [ind1; ind2; ind3; ind4]
    | _ ->  invalid_arg "Ndarray.pp_print: more than 4 axes to print out not supported" in
  let labels =
    match labels with
    | [] | [None] | [None; None] | [None; None; None] | [None; None; None; None] ->
      [""; ""; ""; ""]
    | [Some l1] | [None; Some l1] | [None; None; Some l1] | [None; None; None; Some l1] -> 
      [""; ""; ""; l1^":"]
    | [Some l1; Some l2] | [None; Some l1; Some l2] | [None; None; Some l1; Some l2] -> 
      [""; ""; l1^":"; l2^":"]
    | [Some l1; None; Some l2] | [None; Some l1; None; Some l2] -> [""; l1^":"; ""; l2^":"]
    | [Some l1; Some l2; Some l3] | [None; Some l1; Some l2; Some l3] -> [""; l1^":"; l2^":"; l3^":"]
    | [Some l1; Some l2; Some l3; Some l4] -> [l1^":"; l2^":"; l3^":"; l4^":"]
  | _ -> invalid_arg "pp_print: ~labels should have at most 4 entries" in
  let rec ord axes = if List.length axes >= 4 then axes
    else ord (1::List.map axes ~f:(fun d->d + 1)) in
  let order_of_axes = ord order_of_axes in
  let var_indices = List.map ~f:snd @@ List.sort ~compare:(fun (a,_) (b,_) -> Int.compare a b) @@
    List.zip_exn order_of_axes var_indices in
  let ind1, ind2, ind3, ind4 =
    match var_indices with 
    | [ind1; ind2; ind3; ind4] -> ind1, ind2, ind3, ind4
    | _ -> assert false in
  let labels = List.map ~f:snd @@ List.sort ~compare:(fun (a,_) (b,_) -> Int.compare a b) @@
    List.zip_exn order_of_axes labels in
  let label1, label2, label3, label4 =
    match labels with 
    | [label1; label2; label3; label4] -> label1, label2, label3, label4
    | _ -> assert false in
  let to1 = if ind1 = -1 then 0 else min (dims.(ind1) - 1) entries_per_axis in
  let to2 = if ind2 = -1 then 0 else min (dims.(ind2) - 1) entries_per_axis in
  let to3 = if ind3 = -1 then 0 else min (dims.(ind3) - 1) entries_per_axis in
  let to4 = if ind4 = -1 then 0 else min (dims.(ind4) - 1) entries_per_axis in
  let open Caml.Format in
  pp_open_tbox fmt ();
  (* Headers. *)
  pp_set_tab fmt ();
  fprintf fmt "   ";
for k = 0 to to3 do
    for l = 0 to to4 do
      pp_set_tab fmt ();
      if ind3 <> -1 || ind4 <> -1 then fprintf fmt "<";
      if ind3 <> -1 then fprintf fmt "%s%d=%d" label3 ind3 k;
      if ind3 <> -1 && ind4 <> -1 then fprintf fmt ",";
      if ind4 <> -1 then fprintf fmt "%s%d=%d" label4 ind4 l;
      if ind3 <> -1 || ind4 <> -1 then fprintf fmt ">";
     done;
     if k <> to3 then (pp_set_tab fmt (); fprintf fmt "|")
    done;
  (* Tables. *)
  for i = 0 to to1 do
    if ind1 <> -1 then (
      let pos1 = if dims.(ind1) < entries_per_axis || i < entries_per_axis / 2 then i
        else dims.(ind1) + entries_per_axis / 2 - i in
      indices.(ind1) <- pos1;
    );
    for j = 0 to to2 do
      if ind2 <> -1 then (
        let pos2 = if dims.(ind2) < entries_per_axis || j < entries_per_axis / 2 then j
          else dims.(ind2) + entries_per_axis / 2 - j in
        indices.(ind2) <- pos2;
      );
        (* Header. *)
      pp_print_tab fmt ();
      if ind1 <> -1 || ind2 <> -1 then fprintf fmt "<";
      if ind1 <> -1 then fprintf fmt "%s%d=%d" label1 ind1 i;
      if ind1 <> -1 && ind2 <> -1 then fprintf fmt ",";
      if ind2 <> -1 then fprintf fmt "%s%d=%d" label2 ind2 j;
      if ind1 <> -1 || ind2 <> -1 then fprintf fmt ">";
      for k = 0 to to3 do
        if ind3 <> -1 then (
          let pos3 = if dims.(ind3) < entries_per_axis || k < entries_per_axis / 2 then k
            else dims.(ind3) + entries_per_axis / 2 - k in
          indices.(ind3) <- pos3;
        );
        for l = 0 to to4 do
          if ind4 <> -1 then (
            let pos4 = if dims.(ind4) < entries_per_axis || l < entries_per_axis / 2 then l
              else dims.(ind4) + entries_per_axis / 2 - l in
            indices.(ind4) <- pos4;
          );
          pp_print_tab fmt ();
          fprintf fmt "%f" @@ A.get arr indices
        done;
        if k <> to3 then (pp_print_tab fmt (); fprintf fmt "|")
      done
    done
  done;
  pp_close_tbox fmt (); pp_print_newline fmt ()

let dims (arr: t) = A.dims arr
  
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

let assign lhs rhs =
  (* TODO: FIXME: NOT IMPLEMENTED *)
  ignore (lhs, rhs)

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
