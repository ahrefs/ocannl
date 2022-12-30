(** Multidimensional arrays and the code for operating on them. *)
open Base

module A = Bigarray.Genarray
type elt = Bigarray.float32_elt
type t = (float, elt, Bigarray.c_layout) A.t

(** Index into an ndarray: encapsulates broadcasting, transpose, permuting axes for [einsum], etc.
    The input is the inner-loop parent/result indices, the output is the subformula/argument indices. *)
type index_t = int Codelib.code array -> int array Codelib.code

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

let precompile_unop unop_code:
  lhs:t -> rhs:t -> index_rhs:(int array -> int array) -> unit =
  Runnative.run .< fun ~lhs ~rhs ~index_rhs ->
    .~(unop_code ~lhs: .<lhs>. ~rhs: .<rhs>.
       ~index_rhs:(fun idx -> .<index_rhs .~(Lifts.lift_array idx)>.)
      ) >.

let precompile_binop binop_code:
  lhs:t -> rhs1:t -> index1:(int array -> int array) -> rhs2:t -> index2:(int array -> int array) -> unit =
  Runnative.run .< fun ~lhs ~rhs1 ~index1 ~rhs2 ~index2 ->
    .~(binop_code ~lhs: .<lhs>. ~rhs1: .<rhs1>.
       ~index1:(fun idx -> .<index1 .~(Lifts.lift_array idx)>.)
       ~rhs2: .<rhs2>.
       ~index2:(fun idx -> .<index2 .~(Lifts.lift_array idx)>.)
       ) >.

let precompile_unop_sum unop_code:
  lhs:t -> index_lhs:(int array -> int array) -> rhs:t -> unit =
  Runnative.run .< fun ~lhs ~index_lhs ~rhs ->
    .~(unop_code ~lhs: .<lhs>. ~index_lhs:(fun idx -> .<index_lhs .~(Lifts.lift_array idx)>.)
         ~rhs: .<rhs>.
      ) >.

let precompile_binop_sum1 binop_code:
  lhs:t -> index_lhs:(int array -> int array) ->
     rhs1:t -> rhs2:t -> index2:(int array -> int array) -> unit =
  Runnative.run .< fun ~lhs ~index_lhs ~rhs1 ~rhs2 ~index2 ->
    .~(binop_code ~lhs: .<lhs>. ~index_lhs:(fun idx -> .<index_lhs .~(Lifts.lift_array idx)>.)
       ~rhs1: .<rhs1>.
       ~rhs2: .<rhs2>.
       ~index2:(fun idx -> .<index2 .~(Lifts.lift_array idx)>.)
       ) >.

let precompile_binop_sum2 binop_code:
  lhs:t -> index_lhs:(int array -> int array) ->
     rhs1:t -> index1:(int array -> int array) -> rhs2:t -> unit =
  Runnative.run .< fun ~lhs ~index_lhs ~rhs1 ~index1 ~rhs2 ->
    .~(binop_code ~lhs: .<lhs>. ~index_lhs:(fun idx -> .<index_lhs .~(Lifts.lift_array idx)>.)
       ~rhs1: .<rhs1>.
       ~index1:(fun idx -> .<index1 .~(Lifts.lift_array idx)>.)
       ~rhs2: .<rhs2>.
       ) >.

(** Accumulates the results of the operation on [rhs] by adding them into the [lhs]. *)
let unop_sum unop ~lhs ~(index_lhs: index_t) ~rhs =
  (* TODO: FIXME: NOT IMPLEMENTED *)
  .< ignore (.~(unop .<1.0>.), .~lhs, .~rhs, .~(index_lhs [||])) >.

(** Accumulates the results of the operation on [rhs1] with side input [rhs2] by adding them into the [lhs]. *)
let binop_sum1 binop ~lhs ~(index_lhs: index_t) ~rhs1 ~rhs2 ~(index2: index_t) =
  (* TODO: FIXME: NOT IMPLEMENTED *)
  .< ignore (.~(binop .<1.0>. .<1.0>.), .~lhs, .~rhs1, .~rhs2, .~(index_lhs [||]), .~(index2 [||])) >.

(** Accumulates the results of the operation on [rhs2] with side input [rhs1] by adding them into the [lhs]. *)
let binop_sum2 binop ~lhs ~(index_lhs: index_t) ~rhs1 ~(index1: index_t) ~rhs2 =
  (* TODO: FIXME: NOT IMPLEMENTED *)
  .< ignore (.~(binop .<1.0>. .<1.0>.), .~lhs, .~rhs1, .~rhs2, .~(index_lhs [||]), .~(index1 [||])) >.

let assign lhs rhs =
  (* TODO: FIXME: NOT IMPLEMENTED *)
  ignore (lhs, rhs)
  
(** Iteraters through [lhs] and assigns [lhs.(lhs_pos) <- rhs1.(index1 lhs_pos) + rhs2.(index2 lhs_pos)]. *)
let assign_add_code ~lhs ~rhs1 ~(index1: index_t) ~rhs2 ~(index2: index_t) =
  (* TODO: FIXME: NOT IMPLEMENTED *)
  .< ignore (.~lhs, .~rhs1, .~(index1 [||]), .~rhs2, .~(index2 [||])) >.

(** Same as [assign_add_code] but precompiled. *)
let assign_add_call = precompile_binop assign_add_code

(** Iteraters through [rhs] and assigns [lhs.(index_lhs rhs_pos) <- lhs.(index_lhs lhs_pos) + rhs.(rhs_pos)]. *)
let accum_sum_code = unop_sum (fun rhs -> rhs)

(** Same as [accum_sum_code] but precompiled. *)
let accum_sum_call = precompile_unop_sum accum_sum_code

(** Iteraters through [lhs] and assigns [lhs.(lhs_pos) <- rhs1.(index1 lhs_pos) *. rhs2.(index2 lhs_pos)]. *)
let assign_pointmul_code ~lhs ~rhs1 ~(index1: index_t) ~rhs2 ~(index2: index_t) =
  (* TODO: FIXME: NOT IMPLEMENTED *)
  .< ignore (.~lhs, .~rhs1, .~(index1 [||]), .~rhs2, .~(index2 [||])) >.

(** Same as [assign_mul_code] but precompiled. *)
let assign_pointmul_call = precompile_binop assign_pointmul_code

(** Computes [lhs += rhs1 * rhs2]. *)
let accum_pointmul_code = binop_sum1 (fun rhs1 rhs2 -> .< Float.(.~rhs1 * .~rhs2) >.)

(** Same as [accum_pointmul_code] but precompiled. *)
let accum_pointmul_call = precompile_binop_sum1 accum_pointmul_code

let assign_relu_code ~lhs ~rhs ~(index_rhs: index_t) =
  (* TODO: FIXME: NOT IMPLEMENTED *)
  .< ignore (.~lhs, .~(index_rhs [||]), .~rhs) >.

(** Same as [assign_relu_code] but precompiled. *)
let assign_relu_call = precompile_unop assign_relu_code

(** Computes [lhs += if rhs1 > 0 then rhs2 else 0]. *)
let accum_relu_gate_code =
  binop_sum2 (fun rhs1 rhs2 -> .< Float.(if .~rhs1 > 0.0 then .~rhs2 else 0.0) >.)

(** Same as [relu_gate_code] but precompiled. *)
let accum_relu_gate_call = precompile_binop_sum2 accum_relu_gate_code


(** Prints 0-based [indices] entries out of [arr], where [-1] in an axis means to print out the axis,
    and a non-negative index means to print out only the indexed dimension of the axis. Up to [5] axes
    can be [-1]. Prints up to [entries_per_axis] or [entries_per_axis+1] entries per axis.
    [order_of_axes] specifies the layout (priorities) of the printed matrices, and should be a list of
    up to 5 integers. [labels] specifies the labels (if any) of the printed out axes. The printed out
    axes are arranged as, from highest priority: horizontal by vertical in inner rectangles,
    horizontal by vertical in outer rectangles, vertical list (of outer rectangles) repeated until
    a callback called in between each outer rectangle returns true. *)
let pp_print fmt ?(entries_per_axis=5) ?(order_of_axes=[]) ?(prefer_vertical=false)
    ?(labels=[]) ~screen_stop ~indices (arr: t) =
  let dims = A.dims arr in
  let indices = Array.copy indices in
  let entries_per_axis = if entries_per_axis % 2 = 0 then entries_per_axis + 1 else entries_per_axis in
  let var_indices =
    match Array.filter_mapi indices ~f:(fun i d -> if d = -1 then Some i else None) with
    | [||] -> [-1; -1; -1; -1; -1]
    | [|ind1|] -> [-1; -1; -1; -1; ind1]
    | [|ind1; ind2|] -> [-1; -1; ind1; -1; ind2]
    | [|ind1; ind2; ind3|] ->
      if prefer_vertical then [-1; ind1; ind2; -1; ind3] else [-1; -1; ind1; ind2; ind3]
    | [|ind1; ind2; ind3; ind4|] -> [-1; ind1; ind2; ind3; ind4]
    | [|ind0; ind1; ind2; ind3; ind4|] -> [ind0; ind1; ind2; ind3; ind4]
    | _ ->  invalid_arg "Ndarray.pp_print: more than 5 axes to print out not supported" in
  let labels =
    match labels with
    | [] | [None] | [None; None] | [None; None; None] | [None; None; None; None]
    | [None; None; None; None; None] ->
      [""; ""; ""; ""; ""]
    | [Some l1] | [None; Some l1] | [None; None; Some l1] | [None; None; None; Some l1] -> 
      [""; ""; ""; ""; l1^":"]
    | [Some l1; Some l2] | [None; Some l1; Some l2] | [None; None; Some l1; Some l2] -> 
      [""; ""; ""; l1^":"; l2^":"]
    | [Some l1; None; Some l2] | [None; Some l1; None; Some l2] -> [""; ""; l1^":"; ""; l2^":"]
    | [Some l1; Some l2; Some l3] | [None; Some l1; Some l2; Some l3] -> [""; ""; l1^":"; l2^":"; l3^":"]
    | [Some l1; Some l2; Some l3; Some l4] -> [""; l1^":"; l2^":"; l3^":"; l4^":"]
    | [Some l0; Some l1; Some l2; Some l3; Some l4] -> [l0^":"; l1^":"; l2^":"; l3^":"; l4^":"]
    | _ -> invalid_arg "pp_print: ~labels should have at most 5 entries" in
  let rec ord axes = if List.length axes >= 5 then axes
    else ord (1::List.map axes ~f:(fun d->d + 1)) in
  let order_of_axes = ord order_of_axes in
  let var_indices = List.map ~f:snd @@ List.sort ~compare:(fun (a,_) (b,_) -> Int.compare a b) @@
    List.zip_exn order_of_axes var_indices in
  (* Swap second-to-highest and third-to-highest priority axes: horizontal looping takes precedence
     over vertical. *)
  let ind0, ind1, ind2, ind3, ind4 =
    match var_indices with 
    | [ind0; ind1; ind2; ind3; ind4] -> ind0, ind1, ind3, ind2, ind4
    | _ -> assert false in
  let labels = List.map ~f:snd @@ List.sort ~compare:(fun (a,_) (b,_) -> Int.compare a b) @@
    List.zip_exn order_of_axes labels in
  let label0, label1, label2, label3, label4 =
    match labels with 
    | [label0; label1; label2; label3; label4] -> label0, label1, label3, label2, label4
    | _ -> assert false in
  let to0 = if ind0 = -1 then 0 else min (dims.(ind0) - 1) entries_per_axis in
  let to1 = if ind1 = -1 then 0 else min (dims.(ind1) - 1) entries_per_axis in
  let to2 = if ind2 = -1 then 0 else min (dims.(ind2) - 1) entries_per_axis in
  let to3 = if ind3 = -1 then 0 else min (dims.(ind3) - 1) entries_per_axis in
  let to4 = if ind4 = -1 then 0 else min (dims.(ind4) - 1) entries_per_axis in
  let open Caml.Format in
  let exception Stop_outermost_axis in
  (try for v = 0 to to0 do
       if ind0 <> -1 then fprintf fmt "%s%d=%d" label0 ind0 v;
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
       if v <> to0 && screen_stop() then raise Stop_outermost_axis
     done with Stop_outermost_axis -> ());
  pp_close_tbox fmt (); pp_print_newline fmt ()

(* Debug navi-parens.
let debug_navi_parens fmt dims ~indices =
  let ind1, ind2, ind3 =
    match indices with 
    | [ind1; ind2; ind3] -> ind1, ind2, ind3
    | _ -> assert false in
  let to1 = if ind1 = -1 then 0 else dims.(ind1) - 1 in
  let to2 = if ind2 = -1 then 0 else dims.(ind2) - 1 in
  let to3 = if ind3 = -1 then 0 else dims.(ind3) - 1 in
  let open Caml.Format in
  for i = 0 to to1 do
    for j = 0 to to2 do
      for k = 0 to to3 do
        if k <> to3 then (pp_print_tab fmt (); fprintf fmt "|")
        else (
          (* FIXME: sort out if we need [pp_print_tbreak fmt 0 0]. *)
            
        )
      done
    done
  done;
  pp_print_newline fmt ()*)