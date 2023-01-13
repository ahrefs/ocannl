(** Multidimensional arrays and the code for operating on them. *)
open Base

module A = Bigarray.Genarray
type elt = Bigarray.float32_elt
type t = (float, elt, Bigarray.c_layout) A.t

let dims (arr: t) = A.dims arr
  
 let create = A.create Bigarray.Float32 Bigarray.C_layout
 let empty = create [||]

(** Accumulates the results of the operation on [rhs] by adding them into the [lhs]. *)
let accum_binop_code ~accum ~op ~lhs ~rhs1 ~rhs2 projections =
  (* FIXME: NOT IMPLEMENTED *)
  .< ignore (accum, op, .~lhs, .~rhs1, .~rhs2, projections) >.

let accum_unop_code ~accum ~op ~lhs ~rhs projections =
  (* FIXME: NOT IMPLEMENTED *)
  .< ignore (accum, op, .~lhs, .~rhs, projections) >.

(** Accumulates the results of the operation on [rhs] by adding them into the [lhs]. *)
let accum_binop_call ~accum ~op ~lhs ~rhs1 ~rhs2 projections =
  (* FIXME: NOT IMPLEMENTED *)
  .< ignore (accum, op, .~lhs, .~rhs1, .~rhs2, projections) >.

let accum_unop_call ~accum ~op ~lhs ~rhs projections =
  (* FIXME: NOT IMPLEMENTED *)
  .< ignore (accum, op, .~lhs, .~rhs, projections) >.

let skip_arg_code (_n1: float Codelib.code) (n2: float Codelib.code) = n2

let skip_arg_call (_n1: float) (n2: float) = n2

let add_code n1 n2 = .< Float.(.~n1 + .~n2) >.

let add_call n1 n2 = Float.(n1 + n2)

let mul_code n1 n2 = .< Float.(.~n1 * .~n2) >.

let mul_call n1 n2 = Float.(n1 * n2)

let relu_code n = .< Float.(if .~n > 0.0 then .~n else 0.0) >.

let relu_call n = Float.(if n > 0.0 then n else 0.0)

let relu_gate_code n1 n2 = .< Float.(if .~n1 > 0.0 then .~n2 else 0.0) >.

let relu_gate_call n1 n2 = Float.(if n1 > 0.0 then n2 else 0.0)

let zero_code = .< 1.0 >.

let zero_call = 1.0

let one_code = .< 1.0 >.

let one_call = 1.0

let value_code (v: float) = Lifts.Lift_float.lift v

let value_call (v: float) = v

let uniform_call ~low ~high = Random.float_range low high

let uniform_code ~low ~high = .< Random.float_range low high >.


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