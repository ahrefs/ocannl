(** Utilities for working with [Node] that could be part of the runtime, but are not currently needed
    in the runtime. *)
open Base

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
    in the middle. [labels] provides the axis labels for all axes (use [""] for no label).
    The last label corresponds to axis [-1] etc. The printed out axes are arranged as:
    * -1: a horizontal segment in an inner rectangle (i.e. column numbers of the inner rectangle),
    * -2: a sequence of segments in a line of text (i.e. column numbers of an outer rectangle),
    * -3: a vertical segment in an inner rectangle (i.e. row numbers of the inner rectangle),
    * -4: a vertical sequence of segments (i.e. column numbers of an outer rectangle),
    * -5: a sequence of screens of text (i.e. stack numbers of outer rectangles).
    Printing out of axis [-5] is interrupted when a callback called in between each outer rectangle
    returns true. *)
let pp_print fmt ?(prefix="") ?(entries_per_axis=4) ?(labels=[||]) ~screen_stop ~indices
    (arr: Ocannl_runtime.Node.data) =
  let open Ocannl_runtime.Node in
  let dims = A.dims arr in
  Stdio.Out_channel.(print_endline @@ prefix ^ "dims: "^dims_to_string dims; flush stdout);
  let indices = Array.copy indices in
  let entries_per_axis = if entries_per_axis % 2 = 0 then entries_per_axis + 1 else entries_per_axis in
  let var_indices = Array.filter_mapi indices ~f:(fun i d -> if d <= -1 then Some (5 + d, i) else None) in
  let var_indices = Array.append (Array.create ~len:(5 - Array.length var_indices) (-1, -1)) var_indices in
  Array.sort ~compare:(fun (a,_) (b,_) -> Int.compare a b) var_indices;
  let var_indices = Array.map ~f:snd @@ var_indices in
  let ind0, ind1, ind2, ind3, ind4 =
    match var_indices with 
    | [|ind0; ind1; ind2; ind3; ind4|] -> ind0, ind1, ind3, ind2, ind4
    | _ -> invalid_arg "Ndcode.pp_print: indices should contain at most 5 negative numbers" in
  let labels = Array.map labels ~f:(fun l -> if String.is_empty l then l else l^":") in
  let to0 = if ind0 = -1 then 0 else min (dims.(ind0) - 1) entries_per_axis in
  let to1 = if ind1 = -1 then 0 else min (dims.(ind1) - 1) entries_per_axis in
  let to2 = if ind2 = -1 then 0 else min (dims.(ind2) - 1) entries_per_axis in
  let to3 = if ind3 = -1 then 0 else min (dims.(ind3) - 1) entries_per_axis in
  let to4 = if ind4 = -1 then 0 else min (dims.(ind4) - 1) entries_per_axis in
  let no_label ind = Array.length labels <= ind in
  let label0 = if ind0 = -1 || no_label ind0 then "" else labels.(ind0) in
  let label1 = if ind1 = -1 || no_label ind1 then "" else labels.(ind1) in
  let label2 = if ind2 = -1 || no_label ind2 then "" else labels.(ind2) in
  let label3 = if ind3 = -1 || no_label ind3 then "" else labels.(ind3) in
  let label4 = if ind4 = -1 || no_label ind4 then "" else labels.(ind4) in
  let open Caml.Format in
  let exception Stop_outermost_axis in
  (* Each screen is a separately flushed box of formatting. *)
  try
    for v = 0 to to0 do
      if ind0 <> -1 then fprintf fmt "%s%d=%d\n%!" label0 ind0 v;
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
    done;
    pp_close_tbox fmt (); pp_print_newline fmt ()
  with Stop_outermost_axis ->
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
