(** Utilities for working with [Node] that could be part of the runtime, but are not currently needed
    in the runtime. *)
open Base

open Ocannl_runtime

let create_of_same_precision_as (node: Node.t) =
  match node.value, node.grad with
  | Single_nd _, Single_nd _ -> Node.create ~value_prec:Single ~grad_prec:Single
  | Single_nd _, Double_nd _ -> Node.create ~value_prec:Single ~grad_prec:Double
  | Double_nd _, Double_nd _ -> Node.create ~value_prec:Double ~grad_prec:Double
  | _ -> invalid_arg @@
  "create_of_same_precision_as: unsupported combination of precisions value: "^
  Node.ndarray_precision_to_string node.value^", grad: "^
  Node.ndarray_precision_to_string node.grad

let create_of_promoted_precision (n1: Node.t) (n2: Node.t) =
  match n1.value, n2.value with
  | Single_nd _, Single_nd _ ->
    (match n1.grad, n2.grad with
     | Single_nd _, Single_nd _ ->
       Node.create ~value_prec:Single ~grad_prec:Single
     | Single_nd _, Double_nd _
     | Double_nd _, Single_nd _
     | Double_nd _, Double_nd _ -> Node.create ~value_prec:Single ~grad_prec:Double
     | _ -> invalid_arg @@
     "create_of_promoted_precision: unsupported combination of precisions n1 grad: "^
     Node.ndarray_precision_to_string n1.grad^", n2 grad: "^
     Node.ndarray_precision_to_string n2.grad)
  | Single_nd _, Double_nd _
  | Double_nd _, Single_nd _
  | Double_nd _, Double_nd _ -> Node.create ~value_prec:Double ~grad_prec:Double
  | _ -> invalid_arg @@
  "create_of_promoted_precision: unsupported combination of precisions n1 value: "^
  Node.ndarray_precision_to_string n1.value^", n2 value: "^
  Node.ndarray_precision_to_string n2.value

(** Dimensions to string, ["x"]-separated, e.g. 1x2x3 for batch dims 1, input dims 3, output dims 2.
    Outputs ["-"] for empty dimensions. *)
let dims_to_string ?(with_axis_numbers=false) dims =
  if Array.is_empty dims then "-"
  else if with_axis_numbers then
    String.concat_array ~sep:" x " @@ Array.mapi dims ~f:(fun d s ->  Int.to_string d^":"^Int.to_string s)
  else
    String.concat_array ~sep:"x" @@ Array.map dims ~f:Int.to_string

let ndarray_dims_to_string ?(with_axis_numbers=false) arr =
  Node.(ndarray_precision_to_string arr ^" prec "^dims_to_string ~with_axis_numbers (dims arr))

(** Converts ID, label and the dimensions of a node to a string. *)
let node_header n =
  let open Node in
  let v_dims_s = ndarray_dims_to_string n.value in
  let g_dims_s = ndarray_dims_to_string n.grad in
  let dims_s =
    if String.equal v_dims_s g_dims_s then "dims "^v_dims_s else "dims val "^v_dims_s^" grad "^g_dims_s in
  "#"^Int.to_string n.id^" "^dims_s 
  (*^" "^PrintBox_text.to_string (PrintBox.Simple.to_box n.label)*)

(** When rendering tensors, outputs this many decimal digits. *)
let print_decimals_precision = ref 2

(** Prints 0-based [indices] entries out of [arr], where a number between [-5] and [-1] in an axis means
    to print out the axis, and a non-negative number means to print out only the indexed dimension of the axis.
    Prints up to [entries_per_axis] or [entries_per_axis+1] entries per axis, possibly with ellipsis
    in the middle. [labels] provides the axis labels for all axes (use [""] or ["_"] for no label).
    The last label corresponds to axis [-1] etc. The printed out axes are arranged as:
    * -1: a horizontal segment in an inner rectangle (i.e. column numbers of the inner rectangle),
    * -2: a sequence of segments in a line of text (i.e. column numbers of an outer rectangle),
    * -3: a vertical segment in an inner rectangle (i.e. row numbers of the inner rectangle),
    * -4: a vertical sequence of segments (i.e. column numbers of an outer rectangle),
    * -5: a sequence of screens of text (i.e. stack numbers of outer rectangles).
    Printing out of axis [-5] is interrupted when a callback called in between each outer rectangle
    returns true. *)
let render_tensor ?(prefix="") ?(entries_per_axis=4) ?(labels=[||]) ~indices (arr: Node.ndarray) =
  let module B = PrintBox in
  let open Node in
  let dims = dims arr in
  if Array.is_empty dims then B.line "<empty or not initialized>"
  else
    let header = prefix in
    let indices = Array.copy indices in
    let entries_per_axis = if entries_per_axis % 2 = 0 then entries_per_axis + 1 else entries_per_axis in
    let var_indices = Array.filter_mapi indices ~f:(fun i d -> if d <= -1 then Some (5 + d, i) else None) in
    let extra_indices =
      [|0, -1; 1, -1; 2, -1; 3, -1; 4, -1|] |>
      Array.filter ~f:(Fn.non @@ Array.mem var_indices ~equal:(fun (a,_) (b,_) -> Int.equal a b)) in
    let var_indices = Array.append extra_indices var_indices in
    Array.sort ~compare:(fun (a,_) (b,_) -> Int.compare a b) var_indices;
    let var_indices = Array.map ~f:snd @@ var_indices in
    let ind0, ind1, ind2, ind3, ind4 =
      match var_indices with
      | [|ind0; ind1; ind2; ind3; ind4|] -> ind0, ind1, ind2, ind3, ind4
      | _ -> invalid_arg "NodeUI.render: indices should contain at most 5 negative numbers" in
    let labels = Array.map labels ~f:(fun l -> if String.is_empty l then "" else l^"=") in
    let entries_per_axis = (entries_per_axis / 2) * 2 + 1 in
    let size0 = if ind0 = -1 then 1 else min dims.(ind0) entries_per_axis in
    let size1 = if ind1 = -1 then 1 else min dims.(ind1) entries_per_axis in
    let size2 = if ind2 = -1 then 1 else min dims.(ind2) entries_per_axis in
    let size3 = if ind3 = -1 then 1 else min dims.(ind3) entries_per_axis in
    let size4 = if ind4 = -1 then 1 else min dims.(ind4) entries_per_axis in
    let no_label ind = Array.length labels <= ind in
    let label0 = if ind0 = -1 || no_label ind0 then "" else labels.(ind0) in
    let label1 = if ind1 = -1 || no_label ind1 then "" else labels.(ind1) in
    let label2 = if ind2 = -1 || no_label ind2 then "" else labels.(ind2) in
    let label3 = if ind3 = -1 || no_label ind3 then "" else labels.(ind3) in
    let label4 = if ind4 = -1 || no_label ind4 then "" else labels.(ind4) in
    (* FIXME: handle ellipsis. *)
    let halfpoint = entries_per_axis / 2 + 1 in
    let expand i ~ind =
      if dims.(ind) <= entries_per_axis then i
      else if i < halfpoint then i
      else dims.(ind) - entries_per_axis + i in
    let update_indices v i j k l =
      if ind0 <> -1 then indices.(ind0) <- expand v ~ind:ind0;
      if ind1 <> -1 then indices.(ind1) <- expand i ~ind:ind1;
      if ind2 <> -1 then indices.(ind2) <- expand j ~ind:ind2;
      if ind3 <> -1 then indices.(ind3) <- expand k ~ind:ind3;
      if ind4 <> -1 then indices.(ind4) <- expand l ~ind:ind4 in
    let elide_for i ~ind =
      ind >= 0 && dims.(ind) > entries_per_axis && i + 1 = halfpoint in
    let is_ellipsis() = Array.existsi indices ~f:(fun ind i -> elide_for i ~ind) in
    let inner_grid v i j =
      B.init_grid ~bars:false ~line:size3 ~col:size4 (fun ~line ~col ->
          update_indices v i j line col;
          try B.hpad 1 @@ B.line @@
            if is_ellipsis() then "..." else
              Printf.sprintf "%.*e" !print_decimals_precision (get_as_float arr indices) |>
              (* The C99 standard requires at least two digits for the exponent, but the leading zero
                 is a waste of space. *)
              String.substr_replace_first ~pattern:"e+0" ~with_:"e+" |>
              String.substr_replace_first ~pattern:"e-0" ~with_:"e-"
          with Invalid_argument _ as error ->
            Stdio.Out_channel.printf "Invalid indices: %s into array: %s\n%!"
              (dims_to_string indices) (dims_to_string dims);
            raise error) in
    let tag ?pos label ind =
      if ind = -1 then ""
      else match pos with
      | Some pos when elide_for pos ~ind -> "~~~~~"
      | Some pos when pos >= 0 -> Int.to_string (expand pos ~ind)^" @ "^label^Int.to_string ind
        | _ -> "axis "^label^Int.to_string ind in
    let outer_grid v =
      B.frame @@
      B.init_grid ~bars:true ~line:(size1+1) ~col:(size2+1) (fun ~line ~col ->
          if line = 0 && col = 0
          then B.lines @@ List.filter ~f:(Fn.non String.is_empty) @@ [tag ~pos:v label0 ind0]
          else if line = 0
          then
            B.lines @@ List.filter ~f:(Fn.non String.is_empty) @@
            [tag ~pos:(col-1) label2 ind2; tag label4 ind4]
          else if col = 0
          then
            B.lines @@ List.filter ~f:(Fn.non String.is_empty) @@
            [tag ~pos:(line-1) label1 ind1; tag label3 ind3]
          else if elide_for (col-1) ~ind:ind2 || elide_for (line-1) ~ind:ind1
          then B.hpad 1 @@ B.line "..."
          else inner_grid v (line-1) (col-1)) in
    let screens = B.init_grid ~bars:true ~line:size0 ~col:1 (fun ~line ~col:_ ->
       if elide_for line ~ind:ind0
       then B.hpad 1 @@ B.line "..." else outer_grid line) in
    B.frame @@ B.vlist ~bars:false [B.text header; screens]

let pp_tensor fmt ?prefix ?entries_per_axis ?labels ~indices arr =
  PrintBox_text.pp fmt @@ render_tensor ?prefix ?entries_per_axis ?labels ~indices arr

(** Prints the whole tensor in an inline syntax. *)
let pp_tensor_inline fmt ~num_batch_axes ~num_output_axes ~num_input_axes ?labels_spec arr =
  let dims = Node.dims arr in
  let num_all_axes = num_batch_axes + num_output_axes + num_input_axes in
  let open Caml.Format in
  let ind = Array.copy dims in
  (match labels_spec with None -> () | Some spec -> fprintf fmt "\"%s\" " spec);
  let rec loop axis =
    let sep =
      if axis < num_batch_axes then ";"
      else if axis < num_batch_axes + num_output_axes then ";"
      else "," in
    let open_delim =
      if axis < num_batch_axes then "[|"
      else if axis < num_batch_axes + num_output_axes then "["
      else if axis = num_batch_axes + num_output_axes then ""
      else "(" in
    let close_delim =
      if axis < num_batch_axes then "|]"
      else if axis < num_batch_axes + num_output_axes then "]"
      else if axis = num_batch_axes + num_output_axes then ""
      else ")" in
    if axis = num_all_axes then fprintf fmt "%.*f" !print_decimals_precision (Node.get_as_float arr ind)
    else (fprintf fmt "@[<hov 2>%s@," open_delim;
          for i = 0 to dims.(axis) - 1 do
            ind.(axis) <- i; loop (axis + 1);
            if i < dims.(axis) - 1 then fprintf fmt "%s@ " sep;
          done;
          fprintf fmt "@,%s@]" close_delim) in
  loop 0


let to_dag ?entries_per_axis ~with_value ~with_grad n_id =
  let rec to_dag {Node.sub_node_id; computed_externally}: PrintBox_utils.dag =
    let n = Node.get sub_node_id in
    let id = Int.to_string sub_node_id in
    let children = List.map ~f:to_dag n.children in
    let prefix = "["^id^"] "^n.op_label in
    let labels = n.default_display_labels in
    match computed_externally, with_value, with_grad, n.default_display_indices with
    | true, _, _, _ -> `Embed_subtree_ID (Int.to_string sub_node_id)
    | _, false, false, _
    | _, _, _, None -> `Subtree_with_ID (id, `Tree (`Text n.op_label, children))
    | _, true, false, Some indices ->
      let node = `Box (render_tensor ~prefix ?entries_per_axis ?labels ~indices n.value) in
      `Subtree_with_ID (id, `Tree (node, children))
    | _, false, true, Some indices ->
      let prefix = prefix^" Gradient" in
      let node = `Box (render_tensor ~prefix ?entries_per_axis ?labels ~indices n.grad) in
      `Subtree_with_ID (id, `Tree (node, children))
    | _, true, true, Some indices ->
      let node =
        let value = render_tensor ~prefix ?entries_per_axis ?labels ~indices n.value in
        let grad = render_tensor ~prefix:"Gradient" ?entries_per_axis ?labels ~indices n.grad in
        `Vlist [`Box value; `Box grad] in
      `Subtree_with_ID (id, `Tree (node, children)) in
  to_dag {Node.sub_node_id=n_id; computed_externally=false}

let to_printbox ?entries_per_axis ?(with_value=true) ~with_grad ~depth n_id =
  to_dag ?entries_per_axis ~with_value ~with_grad n_id |> PrintBox_utils.reformat_dag depth
