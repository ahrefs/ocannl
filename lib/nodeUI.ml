(** Utilities for working with [Node] that do not belong in the runtime. *)
open Base

open Ocannl_runtime

(** A DAG of decorated [Node]s, also storing the shape information. *)
type t = {
  id: int;
  node: Node.t;
  children: sub_node list;
  op_label: string;
  shape: Shape.t;
} [@@deriving sexp_of]
and sub_node = {
  sub_node_id: int;
  computed_externally: bool;
} [@@deriving sexp_of]

let global_node_store = Hashtbl.create (module Int)
let get uid = Hashtbl.find_exn global_node_store uid

(** Constructs a node with empty tensors of the specified precision and registers it in the global store.
    Note that the precision for gradients should not be lower than the precision for values. *)
let create (type grad_arr_t value_arr_t) ~(value_prec: ('a, value_arr_t) Node.precision)
    ?(grad_prec: ('a, grad_arr_t) Node.precision option) ~is_form ()
    ~op_label ?batch_dims ?input_dims ?output_dims ?axis_labels ?deduced ~children () =
  let node = Node.create ~value_prec ?grad_prec ~is_form () in
  let shape =
    Shape.make ?batch_dims ?input_dims ?output_dims ?axis_labels ?deduced ~id:node.id () in
  let data = { id=node.id; node; op_label; children; shape } in
  Hashtbl.add_exn global_node_store ~key:node.id ~data;
  data

let create_of_same_precision_as ~is_form (node: Node.t) =
  match node.value, node.form with
  | Single_nd _, Some {grad=Single_nd _; _} ->
    create ~value_prec:Single ~grad_prec:Single ~is_form ()
  | Single_nd _, Some {grad=Double_nd _; _} ->
    create ~value_prec:Single ~grad_prec:Double ~is_form ()
  | Double_nd _, Some {grad=Double_nd _; _} ->
    create ~value_prec:Double ~grad_prec:Double ~is_form ()
  | Single_nd _, None ->
    if is_form then
      invalid_arg @@
      "create_of_same_precision_as: ~is_form:true but a non-form subnode ["^
      Int.to_string node.id
    else create ~value_prec:Single ~is_form:false ()
  | Double_nd _, None ->
    if is_form then
      invalid_arg @@
      "create_of_same_precision_as: ~is_form:true but a non-form subnode ["^
      Int.to_string node.id
    else create ~value_prec:Double ~is_form:false ()
  | _, Some {grad; _} -> invalid_arg @@
  "create_of_same_precision_as: unsupported combination of precisions value: "^
  Node.ndarray_precision_to_string node.value^", grad: "^
  Node.ndarray_precision_to_string grad
  | _ -> invalid_arg @@
  "create_of_same_precision_as: unsupported combination of precisions value: "^
  Node.ndarray_precision_to_string node.value

let create_of_promoted_precision ~is_form (n1: Node.t) (n2: Node.t) =
  match n1.value, n2.value with
  | Single_nd _, Single_nd _ ->
    (match n1.form, n2.form with
     | Some {grad=Single_nd _; _}, Some {grad=Single_nd _; _} ->
       create ~value_prec:Single ~grad_prec:Single ~is_form ()
     | Some {grad=Single_nd _; _}, Some {grad=Double_nd _; _}
     | Some {grad=Double_nd _; _}, Some {grad=Single_nd _; _}
     | Some {grad=Double_nd _; _}, Some {grad=Double_nd _; _} ->
       create ~value_prec:Single ~grad_prec:Double ~is_form ()
     | None, _ | _, None ->
      if is_form then
        invalid_arg @@
        "create_of_promoted_precision: ~is_form:true but a non-form subnode ["^
        Int.to_string n1.id^"] or ["^Int.to_string n2.id^"]"
      else create ~value_prec:Single ~is_form:false ()
     | Some {grad=n1grad; _}, Some {grad=n2grad; _} ->
       invalid_arg @@
       "create_of_promoted_precision: unsupported combination of precisions n1 grad: "^
       Node.ndarray_precision_to_string n1grad^", n2 grad: "^
       Node.ndarray_precision_to_string n2grad)
  | (Single_nd _, Double_nd _ | Double_nd _, Single_nd _ | Double_nd _, Double_nd _)
    when Option.is_some n1.form || Option.is_some n2.form ->
    create ~value_prec:Double ~grad_prec:Double ~is_form:true ()
  | (Single_nd _, Double_nd _ | Double_nd _, Single_nd _ | Double_nd _, Double_nd _) ->
    create ~value_prec:Double ~is_form:false ()
  | _ -> invalid_arg @@
  "create_of_promoted_precision: unsupported combination of precisions n1 value: "^
  Node.ndarray_precision_to_string n1.value^", n2 value: "^
  Node.ndarray_precision_to_string n2.value
  
let param_nodes ?(from_id=0) () =
  Hashtbl.filter global_node_store ~f:(
    fun n -> n.node.id >= from_id && Option.is_some n.node.form && List.is_empty n.children)

let retrieve_2d_points ?from_axis ~xdim ~ydim arr =
  let dims = Node.dims arr in
  let n_axes = Array.length dims in
  let from_axis = Option.value from_axis ~default:(n_axes - 1) in
  let result = ref [] in
  let idx = Array.create ~len:n_axes 0 in
  let rec iter axis =
    if axis = n_axes then
      let x = idx.(from_axis) <- xdim; Node.get_as_float arr idx in
      let y = idx.(from_axis) <- ydim; Node.get_as_float arr idx in
      result := (x, y) :: !result
    else if axis = from_axis then iter (axis + 1)
    else for p = 0 to dims.(axis) - 1 do
      idx.(axis) <- p;
      iter (axis + 1)
    done in
  iter 0;
  Array.of_list_rev !result

let retrieve_1d_points ?from_axis ~xdim arr =
  let dims = Node.dims arr in
  let n_axes = Array.length dims in
  let from_axis = Option.value from_axis ~default:(n_axes - 1) in
  let result = ref [] in
  let idx = Array.create ~len:n_axes 0 in
  let rec iter axis =
    if axis = n_axes then
      let x = idx.(from_axis) <- xdim; Node.get_as_float arr idx in
      result := x :: !result
    else if axis = from_axis then iter (axis + 1)
    else for p = 0 to dims.(axis) - 1 do
      idx.(axis) <- p;
      iter (axis + 1)
    done in
  iter 0;
  Array.of_list_rev !result

    
(* *** Printing *** *)

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
  let v_dims_s = ndarray_dims_to_string n.node.value in
  let g_dims_s =
    match n.node.form with None -> "<no-form>" | Some form -> ndarray_dims_to_string form.grad in
  let dims_s =
    if String.equal v_dims_s g_dims_s then "dims "^v_dims_s
    else "dims val "^v_dims_s^" grad "^g_dims_s in
  "#"^Int.to_string n.id^" op "^n.op_label^" "^dims_s^" ["^
  String.concat ~sep:"," (List.map n.children ~f:(fun {sub_node_id=i; _} -> Int.to_string i))^
  "]"
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
let render_tensor ?(brief=false) ?(prefix="") ?(entries_per_axis=4) ?(labels=[||]) ~indices (arr: Node.ndarray) =
  let module B = PrintBox in
  let open Node in
  let dims = dims arr in
  if Array.is_empty dims then B.line "<void>"
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
              PrintBox_utils.concise_float ~prec:!print_decimals_precision
                (get_as_float arr indices)
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
    let nlines = if brief then size1 else size1 + 1 in
    let ncols = if brief then size2 else size2 + 1 in
    let outer_grid v =
      (if brief then Fn.id else B.frame) @@
      B.init_grid ~bars:true ~line:nlines ~col:ncols (fun ~line ~col ->
          if not brief && line = 0 && col = 0
          then B.lines @@ List.filter ~f:(Fn.non String.is_empty) @@ [tag ~pos:v label0 ind0]
          else if not brief && line = 0
          then
            B.lines @@ List.filter ~f:(Fn.non String.is_empty) @@
            [tag ~pos:(col-1) label2 ind2; tag label4 ind4]
          else if not brief && col = 0
          then
            B.lines @@ List.filter ~f:(Fn.non String.is_empty) @@
            [tag ~pos:(line-1) label1 ind1; tag label3 ind3]
          else
            let nline = if brief then line else line - 1 in
            let ncol = if brief then col else col - 1 in
             if elide_for ncol ~ind:ind2 || elide_for nline ~ind:ind1
             then B.hpad 1 @@ B.line "..."
             else inner_grid v nline ncol) in
    let screens = B.init_grid ~bars:true ~line:size0 ~col:1 (fun ~line ~col:_ ->
       if elide_for line ~ind:ind0
       then B.hpad 1 @@ B.line "..." else outer_grid line) in
    (if brief then Fn.id else B.frame) @@ B.vlist ~bars:false [B.text header; screens]

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


(** We print out up to 5 axes when printing an [Code], as a grid (outer rectangle) of (inner)
    rectangles, possibly repeated (screens). *)
type array_print_style =
[ `Default
(** The inner rectangles comprise both an input and an output axis, if available. Similarly,
    the outer rectangle comprises a second-from-end input axis and a second-from-end output axis,
    if available. At least one batch axis is output, when available.
    The axes that couldn't be output are printed at position/dimension [0]. *)
| `N5_layout of string
(** The string should provide exclusively non-negative integer pseudo-labels. The numbers [0]-[4] represent
    the priorities of the axes to be printed out, where the priorities correspond to, from highest:
    horizontal, vertical direction of the inner rectangle, horizontal, vertical direction of the outer
    rectangle, repetition (see also [NodeUI.pp_print]). The numbers [n >= 5] stand for the actual
    positions [n - 5] within the corresponding axes. *)
| `Label_layout of (string * int) list
(** The association from axis labels to integers. The negative numbers [-5] to [-1] represent
    the priorities of the axes to be printed out, where the priorities correspond to, from highest:
    horizontal, vertical direction of the inner rectangle, horizontal, vertical direction of the outer
    rectangle, repetition (as above). The numbers [n >= 0] stand for the actual positions
    within the corresponding axes. Unspecified axes are printed at position [0]. *)
| `Inline
(** The tensors are printed linearly, in a bracketed manner, optionally prefixed with the labels
    specification. Note that the syntax causes ambiguity for 1-dimensional input axes (underscores are
    used for axes without explicit labels); when there is a 1-dimensional input axis, we output
    the labels specification even if there are no axis labels as a way to display the number of axes.
    The axis nesting is right-to-left (rightmost is innermost).
    The input axes are innermost and the batch axes outermost. The input axes use [,] as a separator
    and [()] as axis delimiters, but the delimiter for the outermost (i.e. leftmost) axis is omitted.
    The output axes use [;] as a separator and [[]] as axis delimiters (obligatory).
    The batch axes use [;] as a separator and [[||]] as axis delimiters (obligatory). *)
]

let default_display_indices sh =
  let axes = Shape.axis_keys_to_idcs sh |> Map.map ~f:(fun _ -> 0) in
  let occupied = Array.create ~len:5 false in
  let set_occu prio = occupied.(prio + 5) <- true; prio in
  let occu prio = occupied.(prio + 5) in
  let num_input_axes = List.length Shape.(list_of_dims @@ dims_of_kind Input sh) in
  let remaining = Stack.of_list @@ List.filter ~f:(Map.mem axes) @@
    Shape.AxisKey.[
      {in_axes=Input; from_end=1}; {in_axes=Output; from_end=1};
      {in_axes=Input; from_end=2}; {in_axes=Output; from_end=2};
      (if num_input_axes > 1 then {in_axes=Batch; from_end=1} else {in_axes=Output; from_end=3});
      {in_axes=Batch; from_end=1}; {in_axes=Batch; from_end=2};
      {in_axes=Input; from_end=3}; {in_axes=Output; from_end=3};
      {in_axes=Input; from_end=4}; {in_axes=Output; from_end=4};
      {in_axes=Input; from_end=5}; {in_axes=Output; from_end=5} ] in
  let rec loop offset axes =
    if Stack.is_empty remaining || offset > 5 then axes
    else if Fn.non occu ~-offset
    then
      loop (offset + 1) @@ Map.change axes (Stack.pop_exn remaining)
        ~f:(Option.map ~f:(fun _ -> set_occu ~-offset))
    else loop (offset + 1) axes in
  let axes = loop 1 axes in
  Shape.axis_map_to_dims_index axes

let to_dag ?entries_per_axis ~with_value ~with_grad n_id =
  let rec to_dag {sub_node_id; computed_externally}: PrintBox_utils.dag =
    let n = get sub_node_id in
    let id = Int.to_string sub_node_id in
    let children = List.map ~f:to_dag n.children in
    let prefix = "["^id^"] "^n.op_label in
    let labels = Shape.axis_map_to_dims_index ~default:"" n.shape.axis_labels in
    let indices = default_display_indices n.shape in
    match computed_externally, with_value, with_grad, n.node.form with
    | true, _, _, _ -> `Embed_subtree_ID (Int.to_string sub_node_id)
    | _, false, false, _
    | _, false, true, None -> `Subtree_with_ID (id, `Tree (`Text n.op_label, children))
    | _, true, false, _
    | _, true, true, None ->
      let node =
        `Box (render_tensor ~brief:true ~prefix ?entries_per_axis ~labels ~indices n.node.value) in
      `Subtree_with_ID (id, `Tree (node, children))
    | _, false, true, Some form ->
      let prefix = prefix^" Gradient" in
      let node =
        `Box (render_tensor ~brief:true ~prefix ?entries_per_axis ~labels ~indices form.grad) in
      `Subtree_with_ID (id, `Tree (node, children))
    | _, true, true, Some form ->
      let node =
        let value =
          render_tensor ~brief:true ~prefix ?entries_per_axis ~labels ~indices n.node.value in
        let grad =
          render_tensor ~brief:true ~prefix:"Gradient" ?entries_per_axis ~labels ~indices form.grad in
        `Vlist (false, [`Box value; `Box grad]) in
      `Subtree_with_ID (id, `Tree (node, children)) in
  to_dag {sub_node_id=n_id; computed_externally=false}

let to_printbox ?entries_per_axis ?(with_value=true) ~with_grad ~depth n_id =
  to_dag ?entries_per_axis ~with_value ~with_grad n_id |> PrintBox_utils.reformat_dag depth
