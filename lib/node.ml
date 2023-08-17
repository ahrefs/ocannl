open Base
(** Computation nodes; global state not directly related to code generation or session management. *)

open Arrayjit

let num_domains = Caml.Domain.recommended_domain_count ()
let task_pool = Domainslib.Task.setup_pool ~name:"session_task_pool" ~num_domains ()

module Nd = Ndarray

type node = { mutable value : Nd.t; mutable grad : Nd.t option; id : int } [@@deriving sexp_of]

let size_in_bytes n =
  (* Cheating here because 1 number Bigarray is same size as empty Bigarray:
     it's more informative to report the cases differently. *)
  let f arr = if Array.is_empty @@ Nd.A.dims arr then 0 else Nd.A.size_in_bytes arr in
  let size = Nd.map { f } in
  size n.value + Option.value_map ~f:size n.grad ~default:0

(** Constructs a node with empty tensors of the specified precision.
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

type 'a t = {
  id : int;
  node : node;
  children : 'a sub_node list;
  op_label : string;
  desc_label : string option;
  axis_labels : string array ref;
  default_display_indices : int array ref;
  annot : 'a;
      (** To avoid confusion, try to maintain the following for a literal:
      - empty [children],
      - [op_label] stores the approximate human-readable numerical value or representation of the node,
      - [node.grad] is always [None]. *)
}
[@@deriving sexp_of]

and 'a sub_node = { sub_node : 'a t; computed_externally : bool } [@@deriving sexp_of]

type data_kind = Value | Grad [@@deriving compare, sexp, equal, hash, variants]
type tensor_ptr = { id : int; field : data_kind } [@@deriving compare, sexp, equal, hash]

let tensor_ptr_name { id; field } =
  match field with Value -> "n" ^ Int.to_string id ^ "_value" | Grad -> "n" ^ Int.to_string id ^ "_grad"

module Compare_tensor_ptr = struct
  type t = tensor_ptr = { id : int; field : data_kind } [@@deriving compare, sexp, equal, hash]
end

module Tensor_ptr = struct
  include Compare_tensor_ptr
  include Comparator.Make (Compare_tensor_ptr)
end

type tensor_ptr_iset = Set.M(Tensor_ptr).t

let sexp_of_tensor_ptr_iset s = [%sexp_of: tensor_ptr Sequence.t] @@ Set.to_sequence s
let tensor_ptr_iset_of_sexp l = Set.of_list (module Tensor_ptr) @@ List.t_of_sexp tensor_ptr_of_sexp l
let equal_tensor_ptr_iset (s1 : tensor_ptr_iset) (s2 : tensor_ptr_iset) = Set.equal s1 s2

(** Constructs a node with empty tensors of the specified precision and registers it in the global store.
    Note that the precision for gradients should not be lower than the precision for values. *)
let create ~(value_prec : Nd.prec) ?(grad_prec : Nd.prec option) ?(literal = false) ~needs_gradient ()
    ~op_label ?desc_label ~axis_labels ~default_display_indices ~children annot =
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
  { id = node.id; node; op_label; desc_label; children; axis_labels; default_display_indices; annot; literal }

let create_of_same_precision_as ~needs_gradient ?literal node =
  match (node.value, node.grad) with
  | Single_nd _, (Some (Single_nd _) | None) ->
      create ~value_prec:Nd.single ~grad_prec:Nd.single ~needs_gradient ?literal ()
  | Single_nd _, Some (Double_nd _) ->
      create ~value_prec:Nd.single ~grad_prec:Nd.double ~needs_gradient ?literal ()
  | Double_nd _, (Some (Double_nd _) | None) ->
      create ~value_prec:Nd.double ~grad_prec:Nd.double ~needs_gradient ?literal ()
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
  | _, Double_nd _ | Double_nd _, _ ->
      create ~value_prec:Nd.double ~grad_prec:Nd.double ~needs_gradient ()
  | _ ->
      invalid_arg @@ "create_of_promoted_precision: unsupported combination of precisions n1 value: "
      ^ Nd.precision_string n1.value ^ ", n2 value: " ^ Nd.precision_string n2.value

(* *** Printing *** *)

let ndarray_dims_to_string ?(with_axis_numbers = false) arr =
  Nd.precision_string arr ^ " prec " ^ Nd.int_dims_to_string ~with_axis_numbers @@ Nd.dims arr

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
  ^ String.concat ~sep:"," (List.map n.children ~f:(fun { sub_node = { id; _ }; _ } -> Int.to_string id))
  ^ "]"
(*^" "^PrintBox_text.to_string (PrintBox.Simple.to_box n.label)*)

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

let to_dag ?(single_node = false) ?entries_per_axis ?extra_prefix ~with_id ~with_value ~with_grad n =
  let rec to_dag { sub_node = n; computed_externally } : PrintBox_utils.dag =
    let id = Int.to_string n.id in
    let children = if single_node then [] else List.map ~f:to_dag n.children in
    let desc_l = match n.desc_label with None -> "" | Some l -> l ^ " " in
    let op_l = match n.op_label with "" -> "" | l -> "<" ^ l ^ ">" in
    let prefix = "[" ^ id ^ "] " ^ desc_l ^ op_l in
    let prefix =
      match extra_prefix with
      | None -> prefix
      | Some f ->
          let extra = f n.annot in
          if String.is_empty extra then prefix else prefix ^ " " ^ extra
    in
    let labels = !(n.axis_labels) in
    let indices = !(n.default_display_indices) in
    match (computed_externally, with_value, with_grad, n.node.grad) with
    | true, _, _, _ -> `Embed_subtree_ID (Int.to_string n.id)
    | _, false, false, _ | _, false, true, None ->
        let txt = if with_id then prefix else desc_l ^ n.op_label in
        `Subtree_with_ID (id, `Tree (`Text txt, children))
    | _, true, false, _ | _, true, true, None ->
        let node =
          `Box (Nd.render_tensor ~brief:true ~prefix ?entries_per_axis ~labels ~indices n.node.value)
        in
        `Subtree_with_ID (id, `Tree (node, children))
    | _, false, true, Some grad ->
        let prefix = prefix ^ " Gradient" in
        let node = `Box (Nd.render_tensor ~brief:true ~prefix ?entries_per_axis ~labels ~indices grad) in
        `Subtree_with_ID (id, `Tree (node, children))
    | _, true, true, Some grad ->
        let node =
          let value = Nd.render_tensor ~brief:true ~prefix ?entries_per_axis ~labels ~indices n.node.value in
          let grad =
            Nd.render_tensor ~brief:true ~prefix:"Gradient" ?entries_per_axis ~labels ~indices grad
          in
          `Vlist (false, [ `Box value; `Box grad ])
        in
        `Subtree_with_ID (id, `Tree (node, children))
  in
  to_dag { sub_node = n; computed_externally = false }

let to_printbox ?single_node ?entries_per_axis ?extra_prefix ?(with_id = false) ?(with_value = true)
    ~with_grad ~depth n_id =
  to_dag ?single_node ?entries_per_axis ?extra_prefix ~with_id ~with_value ~with_grad n_id
  |> PrintBox_utils.reformat_dag depth

let print_node_preamble ?(print_missing = true) ?extra_prefix n =
  try
    let prefix = node_header n in
    let prefix =
      match extra_prefix with
      | None -> prefix
      | Some f ->
          let extra = f n.annot in
          if String.is_empty extra then prefix else prefix ^ " " ^ extra
    in
    Caml.Format.printf "Node %s" prefix;
    Caml.Format.printf "\n%!"
  with Not_found_s _ | Caml.Not_found ->
    if print_missing then Caml.Format.printf "Node #%d does not exist.\n%!" n.id

(* *** Global store *** *)
let global_node_store = Hashtbl.create (module Int)
let get uid = Hashtbl.find_exn global_node_store uid


let print_preamble ?(from = 0) ?extra_prefix () =
  for id = from to !Node.unique_id - 1 do
    Node.print_node_preamble ~print_missing:false ?extra_prefix (get id)
  done

let get_tensor tensor =
  let n = get tensor.Node.id in
  match tensor.Node.field with Value -> Some n.node.value | Grad -> n.node.grad

let get_prec ptr = match get_tensor ptr with None -> Nd.Void_prec | Some arr -> Nd.get_prec arr

let default_value_prec = ref Ndarray.single
let default_grad_prec = ref Ndarray.single

let create_node_helper data shape =
  let annot = annot shape in
  let axis_labels = ref @@ Shape.axis_map_to_dims_index @@ shape.axis_labels in
  let default_display_indices = ref @@ Shape.default_display_indices shape in
  let data : node = data ~axis_labels ~default_display_indices annot in
  Hashtbl.add_exn global_node_store ~key:data.id ~data;
  data

(** Constructs a node with empty tensors of the specified precision and registers it in the global store.
    Note that the precision for gradients should not be lower than the precision for values. *)
let create_node ~(value_prec : Nd.prec) ?(grad_prec : Nd.prec option) ?(literal = false) ~needs_gradient
    ~op_label ?desc_label ~children shape =
  let data = Node.create ~value_prec ?grad_prec ~literal ~needs_gradient () ~op_label ?desc_label ~children in
  create_node_helper data shape

let create_node_promoted_precision n1 n2 ~needs_gradient ~op_label ?desc_label ~children shape =
  let data = Node.create_of_promoted_precision ~needs_gradient n1 n2 ~op_label ?desc_label ~children in
  create_node_helper data shape

let create_node_same_precision_as ~needs_gradient ?literal n ~op_label ?desc_label ~children shape =
  let data = Node.create_of_same_precision_as ~needs_gradient ?literal n ~op_label ?desc_label ~children in
  create_node_helper data shape

  let global_host_size_in_bytes () =
    Hashtbl.fold global_node_store ~init:0 ~f:(fun ~key:_ ~data sum -> sum + Node.size_in_bytes data.node)
  
  let param_nodes ?(from_id = 0) () =
    Hashtbl.filter global_node_store ~f:(fun n ->
        n.node.id >= from_id && List.is_empty n.children && Option.is_some n.node.grad)
  