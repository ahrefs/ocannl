(** Managing a computation session. *)

open Base

let get_root id =
  let open Formula in
  match Map.find !global_roots id with
  | Some r -> r
  | None ->
    let msg = 
      if id >= !first_session_id && id < Ocannl_runtime.Node.global.unique_id then
        "get_root: Node "^Int.to_string id^" is a subformula"
      else if id >= Ocannl_runtime.Node.global.unique_id then
        "get_root: Node "^Int.to_string id^" has not been created yet"
      else if id < 1 then "get_root: Node IDs start from 1"
      else
        "get_root: Node "^Int.to_string id^" is outside the current session" in
    raise @@ Session_error (msg, None)

let get_node id =
  let open Ocannl_runtime.Node in
  match Hashtbl.find global.node_store id with
  | Some r -> r
  | None ->
    let msg = 
      if id >= global.unique_id then
        "get_node: Node "^Int.to_string id^" has not been created yet"
      else if id < 1 then "get_root: Node IDs start from 1"
      else
        "get_node: Node "^Int.to_string id^" has been removed or lives on a different machine" in
    raise @@ Formula.Session_error (msg, None)

(** *** Printing. *** *)

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

let reformat_dag (_style: array_print_style) box_depth b =
  let s: ('a, 'cmp) Comparator.Module.t = (module String) in
  let rec reused = function
  | `Pad (`Text id) -> Set.singleton s id
  | `Pad b -> reused b
  | `Text _ | `Empty -> Set.empty s
  | `Tree (n, bs) -> Set.union_list s (reused n::List.map ~f:reused bs)
  | `Hlist bs -> Set.union_list s @@ List.map ~f:reused bs
  | `Vlist bs -> Set.union_list s @@ List.map ~f:reused bs
  | `Table bss ->
    Set.union_list s @@ Array.to_list @@ Array.concat_map bss
      ~f:(fun bs -> Array.map ~f:reused bs) in
  let reused = reused b in
  let rec cleanup = function
  | `Pad (`Text id) -> `Text ("["^id^"]")
  | `Tree (n, bs) -> `Tree (cleanup n, List.map ~f:cleanup bs)
  | `Hlist [`Text id; `Text op] when Set.mem reused id -> `Text ("["^id^"] "^op)
  | `Hlist [`Text id; `Text op] when not @@ Set.mem reused id -> `Text op
  | `Hlist bs -> `Hlist (List.map ~f:cleanup bs)
  | `Vlist bs -> `Vlist (List.map ~f:cleanup bs)
  | b -> b in
  let rec boxify depth = function
  | `Tree (n, bs) when depth > 0 -> `Vlist [n; `Hlist (List.map ~f:(boxify @@ depth - 1) bs)]
  | `Hlist bs -> `Hlist (List.map ~f:(boxify @@ depth - 1) bs)
  | `Vlist bs -> `Vlist (List.map ~f:(boxify @@ depth - 1) bs)
  | b -> b in
  boxify box_depth @@ cleanup b

let to_printbox b =
  let open PrintBox in
  let rec to_box = function
  | `Empty -> empty
  | `Pad b -> pad (to_box b)
  | `Text t -> text t
  | `Vlist [h; b] -> vlist [align ~h:`Center ~v:`Bottom @@ to_box h; to_box b]
  | `Vlist l -> vlist (List.map ~f:to_box l)
  | `Hlist l -> hlist (List.map ~f:to_box l)
  | `Table a -> grid (map_matrix to_box a)
  | `Tree (`Text _ | `Hlist [`Text _; `Text _] as h, l) -> tree (frame @@ to_box h) (List.map ~f:to_box l)
  | `Tree (b, l) -> tree (to_box b) (List.map ~f:to_box l) in
  to_box b

let print_formula ?with_tree ~with_grad ~with_code (style: array_print_style) m =
  let open Formula in
  let sh = m.shape in
  let prefix =
    "["^Int.to_string m.node_id^"]: shape "^
    Shape.to_string_hum ~style:`Axis_number_and_size sh^" " in
  let indices =
    match style with
    | `Default ->
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
      
    | `N5_layout priorities ->
      let p_labels = Shape.(axis_labels_of_spec priorities).labels |>
                     Map.map ~f:(Fn.compose ((-) 5) Int.of_string) in
      Shape.axis_map_to_dims_index p_labels

    | `Label_layout label_idcs ->
      let inv_labels = Map.to_alist sh.axis_labels |> List.map ~f:(fun (a,b) -> b,a) |>
                       Map.of_alist (module String) in
      let inv_labels = match inv_labels with
        | `Duplicate_key l -> raise @@ Session_error ("`Label_layout found a repeating label: "^l, Some m)
        | `Ok inv_labels -> inv_labels in
      let idcs = List.map label_idcs ~f:(fun (l, i) ->
        match Map.find inv_labels l with Some axis -> axis, i | None ->
          raise @@ Session_error ("`Label_layout label not found in shape: "^l, Some m)) in
      Shape.axis_map_to_dims_index @@ Map.of_alist_exn (module Shape.AxisKey) idcs
    | `Inline -> [||] in
  let needs_spec = Fn.non Map.is_empty sh.axis_labels ||
                   Shape.(List.exists ~f:((=) 1) @@ list_of_dims @@ dims_of_kind Input sh) in
  let labels = Shape.axis_map_to_dims_index ~default:"" sh.axis_labels in
  let labels_spec = if needs_spec then Some (Shape.to_string_hum ~style:`Only_labels sh) else None in
  let num_axes kind = List.length Shape.(list_of_dims @@ dims_of_kind kind sh) in
  let num_batch_axes = num_axes Shape.AxisKey.Batch in
  let num_input_axes = num_axes Shape.AxisKey.Input in
  let num_output_axes = num_axes Shape.AxisKey.Output in
  let tree = Option.map with_tree
      ~f:(fun depth -> to_printbox @@ reformat_dag style depth m.comp_node.label) in
  (match style with
   | `Inline ->
     NodeUI.pp_tensor_inline Caml.Format.std_formatter ~num_batch_axes ~num_input_axes ~num_output_axes
       ?labels_spec m.comp_node.value
   | _ -> NodeUI.pp_tensor Caml.Format.std_formatter ~prefix ?tree ~labels ~indices m.comp_node.value);
  if with_grad then (
    match style with
    | `Inline ->
      NodeUI.pp_tensor_inline Caml.Format.std_formatter ~num_batch_axes ~num_input_axes ~num_output_axes
        ?labels_spec m.comp_node.grad
    | _ -> NodeUI.pp_tensor Caml.Format.std_formatter ~prefix:(prefix^" Gradient ") ?tree ~labels
             ~indices m.comp_node.grad);
  if with_code then (
    (match m.forward_body with
     | Noop -> ()
     | fwd_code ->
       Caml.Format.printf "Current forward body:@ %a@ " Code.fprint_code fwd_code);
    (match m.backprop_body with
     | Noop -> ()
     | bwd_code ->
      Caml.Format.printf "Current backprop body:@ %a@ " Code.fprint_code bwd_code)
  );
  Stdio.printf "\n%!"

let print_global_root ~with_tree ~with_grad ~with_code (style: array_print_style) root =
  let open Formula in
  print_formula ~with_tree ~with_grad ~with_code:false style root.formula;
  if with_code then (
    (match root.forward_code with
     | None -> ()
     | Some fwd ->
       Caml.Format.printf "Forward:@ %a@ " Code.fprint_program fwd);
    (match root.backprop_code with
     | None -> ()
     | Some bwd ->
      Caml.Format.printf "Backprop:@ %a@ " Code.fprint_program bwd)
  );
  Stdio.printf "\n%!"

let print_global_roots ~with_tree ~with_grad ~with_code (style: array_print_style) =
  let open Formula in
  List.iter (Map.to_alist ~key_order:`Increasing !global_roots) ~f:(fun (node_id, root) ->
      assert (node_id = root.formula.node_id);
      print_global_root ~with_tree ~with_grad ~with_code style root)

let print_preamble() =
  Stdio.printf "%s\n%!" (Formula.prefix_with_preamble "")

(** *** Session management. *** *)
let executor = ref Exec_as_OCaml.load_native
let executor_error_message = ref Exec_as_OCaml.error_message
let set_executor = function
  | `Interpreter ->
     executor := Code.interpret_program;
     executor_error_message := Code.interpreter_error_message

  | `OCaml ->
    executor := Exec_as_OCaml.load_native;
    executor_error_message := Exec_as_OCaml.error_message

let refresh_session ?(with_debug=true) ?(regenerate=false) ?(reinit=false) ?(run=true)
    ?(force_no_init=false) () =
  let open Formula in
  if force_no_init && (regenerate || reinit || run) then
    invalid_arg "refresh_session: set other triggers to false when using force_no_init";
  (* Initialization and the forward processing. *)
  List.iter (Map.to_alist ~key_order:`Increasing !global_roots) ~f:(fun (_node_id, root) ->
    let m = root.formula in
    if regenerate || Option.is_none root.forward_code || Option.is_none root.backprop_code then (
      Sequence.iter root.subtree_shape_updates ~f:Shape.propagate_shapes;
      let forward_prog, backprop_prog = get_toplevel m in
       root.forward_code <- Some forward_prog;
       root.formula.comp_node.forward <- None;
       root.backprop_code <- Some backprop_prog;
       root.formula.comp_node.backprop <- None
    );
    if not force_no_init && 
        (reinit || Option.is_none root.formula.comp_node.forward) then (
      try
        let contents = Exec_as_OCaml.load_native ~with_debug (Option.value_exn root.forward_code) in
        match contents, m.comp_node.forward with
        | Some contents, Some forward ->
          m.comp_node.forward <-
            Some (fun () -> try forward() with error ->
                Formula.handle_error ~formula:m @@ !executor_error_message "Forward error:" ~contents error)
        | Some contents, None ->
          let msg = "refresh_session: error loading `forward`: routine not set in code:\n"^contents in
          raise @@ Session_error (msg, Some m)
        | _, None ->
          failwith ("refresh_session: error loading `forward`: routine not set"^
                    (if with_debug then "" else " (use `~with_debug:true` for more information)"))
        | _ -> ()
      with Session_error (msg, None) ->
        let msg = "Forward init error: "^msg in
        raise @@ Session_error (msg, Some m);
    );
    if not force_no_init && 
        (reinit || Option.is_none root.formula.comp_node.backprop) then (
      try
        let contents = Exec_as_OCaml.load_native ~with_debug (Option.value_exn root.backprop_code) in
        match contents, m.comp_node.backprop with
        | Some contents, Some backprop ->
          m.comp_node.backprop <-
            Some (fun () ->
                try backprop() with error ->
                  Formula.handle_error ~formula:m @@ !executor_error_message "Backprop error:" 
                    ~contents error)
        | Some contents, None ->
          Formula.handle_error ~formula:m @@
          "refresh_session: error loading `backprop`: routine not set in code:\n"^contents
        | None, None ->
          failwith ("refresh_session: error loading `backprop`: routine not set"^
                    (if with_debug then "" else " (use `~with_debug:true` for more information)"))
        | _ -> ()
      with Session_error (msg, None) ->
        Caml.Format.printf "Forward code (context for backprop init error):@ %a@\n"
          Code.fprint_program @@ Option.value_exn root.forward_code;
        Formula.handle_error ~formula:m @@ "Backprop init error: "^msg
    );
    if run then match root.formula.comp_node.forward with
      | Some forward -> forward()
      | None -> assert false
  );
  (* The backpropagation. *)
  if run then
    List.iter (Map.to_alist ~key_order:`Decreasing !global_roots) ~f:(fun (_node_id, root) ->
      Option.value_exn root.formula.comp_node.backprop ())

(** Discards global roots, rolls back [Node.state.unique_id] to [Formula.first_session_id], discards
    the corresponding elements from [Node.state.node_store]. *)
let drop_session() =
  Formula.global_roots := Map.empty (module Int);
  for i = !Formula.first_session_id to Ocannl_runtime.Node.global.unique_id - 1 do
    Hashtbl.remove Ocannl_runtime.Node.global.node_store i
  done;
  Ocannl_runtime.Node.global.unique_id <- !Formula.first_session_id

(** Discards global roots, advances [Formula.first_session_id] to [Node.state.unique_id]. *)
let close_session() =
  Formula.first_session_id := Ocannl_runtime.Node.global.unique_id;
  Formula.global_roots := Map.empty (module Int)
      
module CLI = struct
  let set_executor = set_executor
  let refresh_session = refresh_session
  let drop_session = drop_session
  let close_session = close_session
  let print_global_root = print_global_root
  let print_node = NodeUI.print_node
  let max_sublabel_length = Formula.max_sublabel_length
  let print_formula = print_formula
  let print_global_roots = print_global_roots
  let print_preamble = print_preamble
  let print_decimals_precision = NodeUI.print_decimals_precision
  let get_root = get_root
  let get_node = get_node
end
