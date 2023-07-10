open Base
(** Utilities for working with [Node] that do not belong in the runtime. *)

module Nd = Ndarray

type node = { mutable value : Nd.t; mutable grad : Nd.t option; id : int } [@@deriving sexp_of]

let host_size_in_bytes n =
  (* Cheating here because 1 number Bigarray is same size as empty Bigarray:
     it's more informative to report the cases differently. *)
  let f arr = if Array.is_empty @@ Nd.A.dims arr then 0 else Nd.A.size_in_bytes arr in
  let size = Nd.map { f } in
  size n.value + Option.value_map ~f:size n.grad ~default:0

let unique_id = ref 1

(** Constructs a node with empty tensors of the specified precision and registers it in the global store.
    Note that the precision for gradients should not be lower than the precision for values. *)
let create_node (type grad_elt_t value_elt_t) ~(value_prec : value_elt_t Nd.bigarray Nd.precision)
    ?(grad_prec : grad_elt_t Nd.bigarray Nd.precision option) ~needs_gradient () =
  let id =
    let uid = !unique_id in
    unique_id := !unique_id + 1;
    uid
  in
  let grad =
    match (grad_prec, needs_gradient) with
    | Some grad_prec, true -> Some (Nd.as_t grad_prec @@ Nd.empty grad_prec)
    | None, true -> invalid_arg "Node.create: ~needs_gradient:true requires providing ~grad_prec"
    | _, false -> None
  in
  { value = Nd.as_t value_prec @@ Nd.empty value_prec; grad; id }

type t = {
  id : int;
  node : node;
  children : sub_node list;
  op_label : string;
  desc_label : string option;
  shape : Shape.t;
  mutable value_never_virtual : bool;
  mutable value_never_device_only : bool;
  mutable grad_never_virtual : bool;
  mutable grad_never_device_only : bool;
  mutable value_distributes_over_sum : bool;
      (** [value_distributes_over_sum] is a heuristic marker for deciding synchronization strategies. *)
  literal : bool;
      (** To avoid confusion, try to maintain the following for a literal:
      - empty [children],
      - [op_label] stores the approximate human-readable numerical value or representation of the node,
      - [never_virtual] and [never_device_only] are never true,
      - [node.grad] is always [None]. *)
  mutable backend_info : string;
      (** Information about e.g. the memory strategy that the most recent backend chose for the tensor. *)
}
[@@deriving sexp_of]
(** A DAG of decorated [Node]s, also storing the shape information. *)

and sub_node = { sub_node_id : int; computed_externally : bool } [@@deriving sexp_of]

let global_node_store = Hashtbl.create (module Int)
let get uid = Hashtbl.find_exn global_node_store uid

let global_host_size_in_bytes () =
  Hashtbl.fold global_node_store ~init:0 ~f:(fun ~key:_ ~data sum -> sum + host_size_in_bytes data.node)

type data_kind = Value | Grad [@@deriving sexp, equal, hash, variants]
type tensor_ptr = { id : int; field : data_kind } [@@deriving sexp, equal, hash]

let tensor_ptr_name { id; field } =
  match field with Value -> "n" ^ Int.to_string id ^ "_value" | Grad -> "n" ^ Int.to_string id ^ "_grad"

let get_tensor tensor =
  let n = get tensor.id in
  match tensor.field with Value -> Some n.node.value | Grad -> n.node.grad

let host_size_in_bytes ptr =
  (* 1 number bigarray is reporting the same size as an empty bigarray, but we use size 0 to indicate
     that the bigarray is empty. *)
  let f arr = if Array.is_empty @@ Nd.A.dims arr then 0 else Nd.A.size_in_bytes arr in
  Option.value ~default:0 @@ Option.map ~f:(Nd.map { f }) @@ get_tensor ptr

let get_prec ptr = match get_tensor ptr with None -> Nd.Void_prec | Some arr -> Nd.get_prec arr

(** Constructs a node with empty tensors of the specified precision and registers it in the global store.
    Note that the precision for gradients should not be lower than the precision for values. *)
let create ~(value_prec : Nd.prec) ?(grad_prec : Nd.prec option) ?(literal = false) ~needs_gradient ()
    ~op_label ?desc_label ?batch_dims ?input_dims ?output_dims ?axis_labels ?deduced ~children () =
  let node =
    match value_prec with
    | Void_prec -> assert false
    | Half_prec value_prec -> (
        match grad_prec with
        | None -> create_node ~value_prec ~needs_gradient ()
        | Some Void_prec -> assert false
        | Some (Half_prec grad_prec) -> create_node ~value_prec ~grad_prec ~needs_gradient ()
        | Some (Single_prec grad_prec) -> create_node ~value_prec ~grad_prec ~needs_gradient ()
        | Some (Double_prec grad_prec) -> create_node ~value_prec ~grad_prec ~needs_gradient ())
    | Single_prec value_prec -> (
        match grad_prec with
        | None -> create_node ~value_prec ~needs_gradient ()
        | Some Void_prec -> assert false
        | Some (Half_prec grad_prec) -> create_node ~value_prec ~grad_prec ~needs_gradient ()
        | Some (Single_prec grad_prec) -> create_node ~value_prec ~grad_prec ~needs_gradient ()
        | Some (Double_prec grad_prec) -> create_node ~value_prec ~grad_prec ~needs_gradient ())
    | Double_prec value_prec -> (
        match grad_prec with
        | None -> create_node ~value_prec ~needs_gradient ()
        | Some Void_prec -> assert false
        | Some (Half_prec grad_prec) -> create_node ~value_prec ~grad_prec ~needs_gradient ()
        | Some (Single_prec grad_prec) -> create_node ~value_prec ~grad_prec ~needs_gradient ()
        | Some (Double_prec grad_prec) -> create_node ~value_prec ~grad_prec ~needs_gradient ())
  in
  let shape = Shape.make ?batch_dims ?input_dims ?output_dims ?axis_labels ?deduced ~id:node.id () in
  let data =
    {
      id = node.id;
      node;
      op_label;
      desc_label;
      children;
      shape;
      value_never_virtual = false;
      value_never_device_only = false;
      grad_never_virtual = false;
      grad_never_device_only = false;
      value_distributes_over_sum = false;
      literal;
      backend_info = "";
    }
  in
  Hashtbl.add_exn global_node_store ~key:node.id ~data;
  data

let create_of_same_precision_as ~needs_gradient node =
  match (node.value, node.grad) with
  | Single_nd _, (Some (Single_nd _) | None) ->
      create ~value_prec:Nd.single ~grad_prec:Nd.single ~needs_gradient ()
  | Single_nd _, Some (Double_nd _) -> create ~value_prec:Nd.single ~grad_prec:Nd.double ~needs_gradient ()
  | Double_nd _, (Some (Double_nd _) | None) ->
      create ~value_prec:Nd.double ~grad_prec:Nd.double ~needs_gradient ()
  | _, Some grad ->
      invalid_arg @@ "create_of_same_precision_as: unsupported combination of precisions value: "
      ^ Nd.precision_string node.value ^ ", grad: " ^ Nd.precision_string grad
  | _ ->
      invalid_arg @@ "create_of_same_precision_as: unsupported combination of precisions value: "
      ^ Nd.precision_string node.value

let create_of_promoted_precision ~needs_gradient n1 n2 =
  match (n1.value, n2.value) with
  | Single_nd _, Single_nd _ -> (
      match (n1.grad, n2.grad) with
      | _, Some (Double_nd _) | Some (Double_nd _), _ ->
          create ~value_prec:Nd.single ~grad_prec:Nd.double ~needs_gradient ()
      | _ -> create ~value_prec:Nd.single ~grad_prec:Nd.single ~needs_gradient ())
  | _, Double_nd _ | Double_nd _, _ -> create ~value_prec:Nd.double ~grad_prec:Nd.double ~needs_gradient ()
  | _ ->
      invalid_arg @@ "create_of_promoted_precision: unsupported combination of precisions n1 value: "
      ^ Nd.precision_string n1.value ^ ", n2 value: " ^ Nd.precision_string n2.value

let param_nodes ?(from_id = 0) () =
  Hashtbl.filter global_node_store ~f:(fun n ->
      n.node.id >= from_id && List.is_empty n.children && Option.is_some n.node.grad)

(* *** Printing *** *)

(** Dimensions to string, ["x"]-separated, e.g. 1x2x3 for batch dims 1, input dims 3, output dims 2.
    Outputs ["-"] for empty dimensions. *)
let dims_to_string ?(with_axis_numbers = false) dims =
  if Array.is_empty dims then "-"
  else if with_axis_numbers then
    String.concat_array ~sep:" x "
    @@ Array.mapi dims ~f:Shape.(fun d s -> Int.to_string d ^ ":" ^ dim_to_string s)
  else String.concat_array ~sep:"x" @@ Array.map dims ~f:Shape.dim_to_string

let int_dims_to_string ?with_axis_numbers dims =
  dims_to_string ?with_axis_numbers @@ Array.map ~f:(fun d -> Shape.dim d) dims

let ndarray_dims_to_string ?(with_axis_numbers = false) arr =
  Nd.precision_string arr ^ " prec " ^ int_dims_to_string ~with_axis_numbers @@ Nd.dims arr

(** Converts ID, label and the dimensions of a node to a string. *)
let node_header n =
  let v_dims_s = ndarray_dims_to_string n.node.value in
  let g_dims_s = match n.node.grad with None -> "<no-grad>" | Some grad -> ndarray_dims_to_string grad in
  let dims_s =
    if String.equal v_dims_s g_dims_s then "dims " ^ v_dims_s
    else "dims val " ^ v_dims_s ^ " grad " ^ g_dims_s
  in
  let desc_l = match n.desc_label with None -> "" | Some l -> " " ^ l in
  "#" ^ Int.to_string n.id ^ desc_l ^ " op " ^ n.op_label ^ " " ^ dims_s ^ " ["
  ^ String.concat ~sep:"," (List.map n.children ~f:(fun { sub_node_id = i; _ } -> Int.to_string i))
  ^ "]"
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
let render_tensor ?(brief = false) ?(prefix = "") ?(entries_per_axis = 4) ?(labels = [||]) ~indices
    (arr : Nd.t) =
  let module B = PrintBox in
  let dims = Nd.dims arr in
  let has_nan = Nd.fold ~init:false ~f:(fun has_nan _ v -> has_nan || Float.is_nan v) arr in
  let has_inf = Nd.fold ~init:false ~f:(fun has_inf _ v -> has_inf || Float.(v = infinity)) arr in
  let has_neg_inf =
    Nd.fold ~init:false ~f:(fun has_neg_inf _ v -> has_neg_inf || Float.(v = neg_infinity)) arr
  in
  let header =
    prefix
    ^ (if has_nan || has_inf || has_neg_inf then " includes" else "")
    ^ (if has_nan then " NaN" else "")
    ^ (if has_inf then " pos. inf." else "")
    ^ if has_neg_inf then " neg. inf." else ""
  in
  if Array.is_empty dims then B.vlist ~bars:false [ B.text header; B.line "<void>" ]
  else
    let indices = Array.copy indices in
    let entries_per_axis = if entries_per_axis % 2 = 0 then entries_per_axis + 1 else entries_per_axis in
    let var_indices = Array.filter_mapi indices ~f:(fun i d -> if d <= -1 then Some (5 + d, i) else None) in
    let extra_indices =
      [| (0, -1); (1, -1); (2, -1); (3, -1); (4, -1) |]
      |> Array.filter ~f:(Fn.non @@ Array.mem var_indices ~equal:(fun (a, _) (b, _) -> Int.equal a b))
    in
    let var_indices = Array.append extra_indices var_indices in
    Array.sort ~compare:(fun (a, _) (b, _) -> Int.compare a b) var_indices;
    let var_indices = Array.map ~f:snd @@ var_indices in
    let ind0, ind1, ind2, ind3, ind4 =
      match var_indices with
      | [| ind0; ind1; ind2; ind3; ind4 |] -> (ind0, ind1, ind2, ind3, ind4)
      | _ -> invalid_arg "Node.render: indices should contain at most 5 negative numbers"
    in
    let labels = Array.map labels ~f:(fun l -> if String.is_empty l then "" else l ^ "=") in
    let entries_per_axis = (entries_per_axis / 2 * 2) + 1 in
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
    let halfpoint = (entries_per_axis / 2) + 1 in
    let expand i ~ind =
      if dims.(ind) <= entries_per_axis then i
      else if i < halfpoint then i
      else dims.(ind) - entries_per_axis + i
    in
    let update_indices v i j k l =
      if ind0 <> -1 then indices.(ind0) <- expand v ~ind:ind0;
      if ind1 <> -1 then indices.(ind1) <- expand i ~ind:ind1;
      if ind2 <> -1 then indices.(ind2) <- expand j ~ind:ind2;
      if ind3 <> -1 then indices.(ind3) <- expand k ~ind:ind3;
      if ind4 <> -1 then indices.(ind4) <- expand l ~ind:ind4
    in
    let elide_for i ~ind = ind >= 0 && dims.(ind) > entries_per_axis && i + 1 = halfpoint in
    let is_ellipsis () = Array.existsi indices ~f:(fun ind i -> elide_for i ~ind) in
    let inner_grid v i j =
      B.init_grid ~bars:false ~line:size3 ~col:size4 (fun ~line ~col ->
          update_indices v i j line col;
          try
            B.hpad 1 @@ B.line
            @@
            if is_ellipsis () then "..."
            else PrintBox_utils.concise_float ~prec:!print_decimals_precision (Nd.get_as_float arr indices)
          with Invalid_argument _ as error ->
            Stdio.Out_channel.printf "Invalid indices: %s into array: %s\n%!" (int_dims_to_string indices)
              (int_dims_to_string dims);
            raise error)
    in
    let tag ?pos label ind =
      if ind = -1 then ""
      else
        match pos with
        | Some pos when elide_for pos ~ind -> "~~~~~"
        | Some pos when pos >= 0 -> Int.to_string (expand pos ~ind) ^ " @ " ^ label ^ Int.to_string ind
        | _ -> "axis " ^ label ^ Int.to_string ind
    in
    let nlines = if brief then size1 else size1 + 1 in
    let ncols = if brief then size2 else size2 + 1 in
    let outer_grid v =
      (if brief then Fn.id else B.frame)
      @@ B.init_grid ~bars:true ~line:nlines ~col:ncols (fun ~line ~col ->
             if (not brief) && line = 0 && col = 0 then
               B.lines @@ List.filter ~f:(Fn.non String.is_empty) @@ [ tag ~pos:v label0 ind0 ]
             else if (not brief) && line = 0 then
               B.lines
               @@ List.filter ~f:(Fn.non String.is_empty)
               @@ [ tag ~pos:(col - 1) label2 ind2; tag label4 ind4 ]
             else if (not brief) && col = 0 then
               B.lines
               @@ List.filter ~f:(Fn.non String.is_empty)
               @@ [ tag ~pos:(line - 1) label1 ind1; tag label3 ind3 ]
             else
               let nline = if brief then line else line - 1 in
               let ncol = if brief then col else col - 1 in
               if elide_for ncol ~ind:ind2 || elide_for nline ~ind:ind1 then B.hpad 1 @@ B.line "..."
               else inner_grid v nline ncol)
    in
    let screens =
      B.init_grid ~bars:true ~line:size0 ~col:1 (fun ~line ~col:_ ->
          if elide_for line ~ind:ind0 then B.hpad 1 @@ B.line "..." else outer_grid line)
    in
    (if brief then Fn.id else B.frame) @@ B.vlist ~bars:false [ B.text header; screens ]

let pp_tensor fmt ?prefix ?entries_per_axis ?labels ~indices arr =
  PrintBox_text.pp fmt @@ render_tensor ?prefix ?entries_per_axis ?labels ~indices arr

(** Prints the whole tensor in an inline syntax. *)
let pp_tensor_inline fmt ~num_batch_axes ~num_output_axes ~num_input_axes ?labels_spec arr =
  let dims = Nd.dims arr in
  let num_all_axes = num_batch_axes + num_output_axes + num_input_axes in
  let open Caml.Format in
  let ind = Array.copy dims in
  (match labels_spec with None -> () | Some spec -> fprintf fmt "\"%s\" " spec);
  let rec loop axis =
    let sep =
      if axis < num_batch_axes then ";" else if axis < num_batch_axes + num_output_axes then ";" else ","
    in
    let open_delim =
      if axis < num_batch_axes then "[|"
      else if axis < num_batch_axes + num_output_axes then "["
      else if axis = num_batch_axes + num_output_axes then ""
      else "("
    in
    let close_delim =
      if axis < num_batch_axes then "|]"
      else if axis < num_batch_axes + num_output_axes then "]"
      else if axis = num_batch_axes + num_output_axes then ""
      else ")"
    in
    if axis = num_all_axes then fprintf fmt "%.*f" !print_decimals_precision (Nd.get_as_float arr ind)
    else (
      fprintf fmt "@[<hov 2>%s@," open_delim;
      for i = 0 to dims.(axis) - 1 do
        ind.(axis) <- i;
        loop (axis + 1);
        if i < dims.(axis) - 1 then fprintf fmt "%s@ " sep
      done;
      fprintf fmt "@,%s@]" close_delim)
  in
  loop 0

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
      rectangle, repetition (see also [Node.pp_print]). The numbers [n >= 5] stand for the actual
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
(** We print out up to 5 axes when printing a tensor, as a grid (outer rectangle) of (inner)
    rectangles, possibly repeated (screens). *)

let default_display_indices sh =
  let axes = Shape.axis_keys_to_idcs sh |> Map.map ~f:(fun _ -> 0) in
  let occupied = Array.create ~len:5 false in
  let set_occu prio =
    occupied.(prio + 5) <- true;
    prio
  in
  let occu prio = occupied.(prio + 5) in
  let num_input_axes = List.length Shape.(list_of_dims @@ dims_of_kind Input sh) in
  let remaining =
    Stack.of_list
    @@ List.filter ~f:(Map.mem axes)
    @@ Shape.AxisKey.
         [
           { in_axes = Input; from_end = 1 };
           { in_axes = Output; from_end = 1 };
           { in_axes = Input; from_end = 2 };
           { in_axes = Output; from_end = 2 };
           (if num_input_axes > 1 then { in_axes = Batch; from_end = 1 }
            else { in_axes = Output; from_end = 3 });
           { in_axes = Batch; from_end = 1 };
           { in_axes = Batch; from_end = 2 };
           { in_axes = Input; from_end = 3 };
           { in_axes = Output; from_end = 3 };
           { in_axes = Input; from_end = 4 };
           { in_axes = Output; from_end = 4 };
           { in_axes = Input; from_end = 5 };
           { in_axes = Output; from_end = 5 };
         ]
  in
  let rec loop offset axes =
    if Stack.is_empty remaining || offset > 5 then axes
    else if Fn.non occu ~-offset then
      loop (offset + 1)
      @@ Map.change axes (Stack.pop_exn remaining) ~f:(Option.map ~f:(fun _ -> set_occu ~-offset))
    else loop (offset + 1) axes
  in
  let axes = loop 1 axes in
  Shape.axis_map_to_dims_index axes

let to_dag ?(single_node = false) ?entries_per_axis ?(with_backend_info = false) ~with_id ~with_value
    ~with_grad n_id =
  let rec to_dag { sub_node_id; computed_externally } : PrintBox_utils.dag =
    let n = get sub_node_id in
    let id = Int.to_string sub_node_id in
    let children = if single_node then [] else List.map ~f:to_dag n.children in
    let desc_l = match n.desc_label with None -> "" | Some l -> l ^ " " in
    let op_l = match n.op_label with "" -> "" | l -> "<" ^ l ^ ">" in
    let prefix = "[" ^ id ^ "] " ^ desc_l ^ op_l in
    let prefix =
      if String.is_empty n.backend_info || not with_backend_info then prefix
      else prefix ^ " " ^ n.backend_info
    in
    let labels = Shape.axis_map_to_dims_index ~default:"" n.shape.axis_labels in
    let indices = default_display_indices n.shape in
    match (computed_externally, with_value, with_grad, n.node.grad) with
    | true, _, _, _ -> `Embed_subtree_ID (Int.to_string sub_node_id)
    | _, false, false, _ | _, false, true, None ->
        let txt = if with_id then prefix else desc_l ^ n.op_label in
        `Subtree_with_ID (id, `Tree (`Text txt, children))
    | _, true, false, _ | _, true, true, None ->
        let node = `Box (render_tensor ~brief:true ~prefix ?entries_per_axis ~labels ~indices n.node.value) in
        `Subtree_with_ID (id, `Tree (node, children))
    | _, false, true, Some grad ->
        let prefix = prefix ^ " Gradient" in
        let node = `Box (render_tensor ~brief:true ~prefix ?entries_per_axis ~labels ~indices grad) in
        `Subtree_with_ID (id, `Tree (node, children))
    | _, true, true, Some grad ->
        let node =
          let value = render_tensor ~brief:true ~prefix ?entries_per_axis ~labels ~indices n.node.value in
          let grad = render_tensor ~brief:true ~prefix:"Gradient" ?entries_per_axis ~labels ~indices grad in
          `Vlist (false, [ `Box value; `Box grad ])
        in
        `Subtree_with_ID (id, `Tree (node, children))
  in
  to_dag { sub_node_id = n_id; computed_externally = false }

let to_printbox ?single_node ?entries_per_axis ?with_backend_info ?(with_id = false) ?(with_value = true)
    ~with_grad ~depth n_id =
  to_dag ?single_node ?entries_per_axis ?with_backend_info ~with_id ~with_value ~with_grad n_id
  |> PrintBox_utils.reformat_dag depth

let print_node_preamble ?(print_missing = true) ~full_shape id =
  try
    let n = get id in
    Caml.Format.printf "Node %s%s" (node_header n)
      (if String.is_empty n.backend_info then "" else " " ^ n.backend_info);
    if full_shape then Caml.Format.printf " -- full shape:@ %a\n%!" Sexp.pp_hum (Shape.sexp_of_t n.shape)
    else Caml.Format.printf "\n%!"
  with Not_found_s _ | Caml.Not_found ->
    if print_missing then Caml.Format.printf "Node #%d does not exist.\n%!" id

let print_preamble ?(from = 0) ?(full_shape = false) () =
  for id = from to !unique_id - 1 do
    print_node_preamble ~print_missing:false ~full_shape id
  done
