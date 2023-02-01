(** The code for operating on n-dimensional arrays. *)
open Base

type precision =
  | Half
  | Single
  | Double
  (* FIXME(28): implement precision setting and precision-specific code generation. *)
  
 let zero = .< 0.0 >.

 let one = .< 1.0 >.
 
(** Accumulates the results of the operation: [lhs = accum lhs (op rhs1 rhs2)]. *)
let accum_binop ?(zero_out=false) ~accum ~op ?lhs ?rhs1 ?rhs2 projections =
  let lhs_idx = Shape.(derive_index projections.product_iterators projections.project_lhs) in
  let rhs1_idx = Shape.(derive_index projections.product_iterators projections.project_rhs1) in
  let rhs2_idx = match projections.project_rhs2 with
    | None -> invalid_arg "accum_binop: projections missing project_rhs2"
    | Some rhs2 -> Shape.(derive_index projections.product_iterators rhs2) in
  let rhs1 iters =
    match rhs1 with
    | None -> zero
    | Some rhs1 ->
      let rhs1_idx = Lifts.lift_array @@ rhs1_idx iters in
      .<Bigarray.Genarray.get .~rhs1 .~rhs1_idx>. in
  let rhs2 iters =
    match rhs2 with
    | None -> zero
    | Some rhs2 -> 
      let rhs2_idx = Lifts.lift_array @@ rhs2_idx iters in
      .<Bigarray.Genarray.get .~rhs2 .~rhs2_idx>. in
  match lhs with
  | None -> .<()>.
  | Some lhs ->
    let basecase rev_iters =
      let iters = Array.of_list_rev rev_iters in
      let lhs_idx = Lifts.lift_array @@ lhs_idx iters in
      .< Bigarray.Genarray.set .~lhs .~lhs_idx
           .~(accum .<Bigarray.Genarray.get .~lhs .~lhs_idx>. @@ op (rhs1 iters) (rhs2 iters) ) >. in
    let rec loop rev_iters = function
      | [] -> basecase rev_iters
      | dim::product ->
        .< for i = 0 to .~(Lifts.Lift_int.lift dim) - 1 do
          .~(loop (.<i>. ::rev_iters) product)
        done >. in
    if zero_out then
      .< Bigarray.Genarray.fill .~lhs .~zero; .~(loop [] @@ Array.to_list projections.product_space) >.
    else
      loop [] @@ Array.to_list projections.product_space

(** Accumulates the results of the operation: [lhs = accum lhs (op rhs)]. *)
let accum_unop ?(zero_out=false) ~accum ~op ?lhs ?rhs projections =
  let lhs_idx = Shape.(derive_index projections.product_iterators projections.project_lhs) in
  let rhs1_idx = Shape.(derive_index projections.product_iterators projections.project_rhs1) in
  let rhs iters =
    match rhs with
    | None -> zero
    | Some rhs ->
      let rhs1_idx = Lifts.lift_array @@ rhs1_idx iters in
      .<Bigarray.Genarray.get .~rhs .~rhs1_idx>. in
  match lhs with
  | None -> .<()>.
  | Some lhs ->
    let basecase rev_iters =
      let iters = Array.of_list_rev rev_iters in
      let lhs_idx = Lifts.lift_array @@ lhs_idx iters in
      .< Bigarray.Genarray.set .~lhs .~lhs_idx
          .~(accum .<Bigarray.Genarray.get .~lhs .~lhs_idx>. @@ op (rhs iters) ) >. in
    let rec loop rev_iters = function
      | [] -> basecase rev_iters
      | dim::product ->
        .< for i = 0 to .~(Lifts.Lift_int.lift dim) - 1 do
          .~(loop (.<i>. ::rev_iters) product)
        done >. in
    if zero_out then
      .< Bigarray.Genarray.fill .~lhs .~zero; .~(loop [] @@ Array.to_list projections.product_space) >.
    else
      loop [] @@ Array.to_list projections.product_space

let skip_arg (_n1: float Codelib.code) (n2: float Codelib.code) = n2
let num_id (n: float Codelib.code) = n

let identity (n: float Codelib.code) = n

let add n1 n2 = .< Float.(.~n1 + .~n2) >.

let mul n1 n2 = .< Float.(.~n1 * .~n2) >.

let relu n = .< Float.(if .~n > 0.0 then .~n else 0.0) >.

let relu_gate n1 n2 = .< Float.(if .~n1 > 0.0 then .~n2 else 0.0) >.

let value (v: float) = Lifts.Lift_float.lift v

let uniform ~low ~high = .< Random.float_range low high >.


(* TODO: perhaps the rest of this file should be a separate [Ndarray] module. *)

(** Dimensions to string, ["x"]-separated, e.g. 1x2x3 for batch dims 1, input dims 3, output dims 2.
    Outputs ["-"] for empty dimensions. *)
let dims_to_string dims =
  if Array.is_empty dims then "-"
  else String.concat_array ~sep:"x" @@ Array.map dims ~f:Int.to_string_hum

(** Converts ID, label and the dimensions of a node to a string. *)
let node_header n =
  let open Ocannl_runtime.Node in
  let v_dims_s = dims_to_string @@ dims n.value in
  let g_dims_s = dims_to_string @@ dims n.grad in
  let dims_s =
    if String.equal v_dims_s g_dims_s then "dims "^v_dims_s else "dims val "^v_dims_s^" grad "^g_dims_s in
  (if String.is_empty n.label then " #" else n.label^" #")^Int.to_string n.id^" "^dims_s

(** Prints 0-based [indices] entries out of [arr], where a number between [-5] and [-1] in an axis means
    to print out the axis, and a non-negative number means to print out only the indexed dimension of the axis.
    Prints up to [entries_per_axis] or [entries_per_axis+1] entries per axis, possibly with ellipsis
    in the middle. [labels] specifies the labels of the printed out axes, use [""] for no label.
    The last label corresponds to axis [-1] etc. The printed out axes are arranged as:
    * -1: a horizontal segment in an inner rectangle (i.e. column numbers of the inner rectangle),
    * -2: a sequence of segments in a line of text (i.e. column numbers of an outer rectangle),
    * -3: a vertical segment in an inner rectangle (i.e. row numbers of the inner rectangle),
    * -4: a vertical sequence of segments (i.e. column numbers of an outer rectangle),
    * -5: a sequence of screens of text (i.e. stack numbers of outer rectangles).
    Printing out of axis [-5] is interrupted when a callback called in between each outer rectangle
    returns true. *)
let pp_print fmt ?(entries_per_axis=4) ?(labels=[||]) ~screen_stop ~indices (arr: Ocannl_runtime.Node.data) =
  let open Ocannl_runtime.Node in
  let dims = A.dims arr in
  Stdio.Out_channel.(print_string "dims: "; print_s @@ Array.sexp_of_t (Int.sexp_of_t) dims; flush stdout);
  let indices = Array.copy indices in
  let entries_per_axis = if entries_per_axis % 2 = 0 then entries_per_axis + 1 else entries_per_axis in
  let var_indices = Array.filter_mapi indices ~f:(fun i d -> if d <= -1 then Some (~-d, i) else None) in
  let var_indices = Array.append (Array.create ~len:(5 - Array.length var_indices) (0, -1)) var_indices in
  Array.sort ~compare:(fun (a,_) (b,_) -> Int.compare a b) var_indices;
  let var_indices = Array.map ~f:snd @@ var_indices in
  let ind0, ind1, ind2, ind3, ind4 =
    match var_indices with 
    | [|ind4; ind3; ind2; ind1; ind0|] -> ind0, ind1, ind3, ind2, ind4
    | _ -> invalid_arg "Ndcode.pp_print: indices should contain at most 5 negative numbers" in
  let labels = Array.append (Array.create ~len:(5 - Array.length labels) "") @@
    Array.map labels ~f:(fun l -> if String.is_empty l then l else l^":") in
  let label0, label1, label2, label3, label4 =
    match labels with
    | [|l0; l1; l2; l3; l4|] -> l0, l1, l2, l3, l4
    | _ -> invalid_arg "pp_print: ~labels should have at most 5 entries" in
  let to0 = if ind0 = -1 then 0 else min (dims.(ind0) - 1) entries_per_axis in
  let to1 = if ind1 = -1 then 0 else min (dims.(ind1) - 1) entries_per_axis in
  let to2 = if ind2 = -1 then 0 else min (dims.(ind2) - 1) entries_per_axis in
  let to3 = if ind3 = -1 then 0 else min (dims.(ind3) - 1) entries_per_axis in
  let to4 = if ind4 = -1 then 0 else min (dims.(ind4) - 1) entries_per_axis in
  let open Caml.Format in
  let exception Stop_outermost_axis in
  (* Each screen is a separately flushed box of formatting. *)
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
        if ind0 <> -1 then (
          let pos0 = if dims.(ind0) <= entries_per_axis || v < entries_per_axis / 2 then v
            else dims.(ind0) + entries_per_axis / 2 - v in
          indices.(ind0) <- pos0;
        );
        for i = 0 to to1 do
          if ind1 <> -1 then (
            let pos1 = if dims.(ind1) <= entries_per_axis || i < entries_per_axis / 2 then i
              else dims.(ind1) + entries_per_axis / 2 - i in
            indices.(ind1) <- pos1;
          );
          for j = 0 to to2 do
            if ind2 <> -1 then (
              let pos2 = if dims.(ind2) <= entries_per_axis || j < entries_per_axis / 2 then j
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
                let pos3 = if dims.(ind3) <= entries_per_axis || k < entries_per_axis / 2 then k
                  else dims.(ind3) + entries_per_axis / 2 - k in
                indices.(ind3) <- pos3;
              );
              for l = 0 to to4 do
                if ind4 <> -1 then (
                  let pos4 = if dims.(ind4) <= entries_per_axis || l < entries_per_axis / 2 then l
                    else dims.(ind4) + entries_per_axis / 2 - l in
                  indices.(ind4) <- pos4;
                );
                pp_print_tab fmt ();
                try fprintf fmt "%f" @@ A.get arr indices
                with Invalid_argument _ as error ->
                  Stdio.Out_channel.printf "Invalid indices: %s into array: %s\n%!"
                    (dims_to_string indices) (dims_to_string dims);
                  raise error
              done;
              if k <> to3 then (pp_print_tab fmt (); fprintf fmt "|")
            done
          done
        done;
        if v <> to0 && screen_stop() then raise Stop_outermost_axis
      done with Stop_outermost_axis -> ());
  pp_close_tbox fmt (); pp_print_newline fmt ()

let print_node ~with_grad ~indices n =
  let open Ocannl_runtime.Node in
  let screen_stop () =
    Stdio.print_endline "Press [Enter] for next screen, [q] [Enter] to quit.";
    String.(Stdio.In_channel.input_line_exn Stdio.stdin = "q")  in
  Stdio.print_endline @@ "["^node_header n^"] "^n.label;
  pp_print Caml.Format.std_formatter ~screen_stop ~indices n.value;
  if with_grad then (
    Stdio.print_endline "Gradient:";
    pp_print Caml.Format.std_formatter ~screen_stop ~indices n.grad)
