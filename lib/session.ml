(** Managing a computation session. *)

open Base

let get_root id =
  let open Tensor in
  match Map.find !global_roots id with
  | Some r -> r
  | None ->
      let msg =
        if id >= !first_session_id && id < session_state.next_session_id then
          "get_root: Node " ^ Int.to_string id ^ " is a subtensor"
        else if id >= session_state.next_session_id then "get_root: Node " ^ Int.to_string id ^ " has not been created yet"
        else if id < 1 then "get_root: Node IDs start from 1"
        else "get_root: Node " ^ Int.to_string id ^ " is outside the current session"
      in
      raise @@ Session_error (msg, None)

let get_node id =
  match Hashtbl.find Low_level.global_node_store id with
  | Some r -> r
  | None ->
      let msg =
        if id >= session_state.next_session_id then "get_node: Node " ^ Int.to_string id ^ " has not been created yet"
        else if id < 1 then "get_root: Node IDs start from 1"
        else "get_node: Node " ^ Int.to_string id ^ " has been removed or lives on a different machine"
      in
      raise @@ Tensor.Session_error (msg, None)

(** *** Printing. *** *)

let ndarray_dims_to_string ?(with_axis_numbers = false) arr =
  Nd.precision_string arr ^ " prec " ^ Nd.int_dims_to_string ~with_axis_numbers @@ Nd.dims arr

(** Converts ID, label and the dimensions of a node to a string. *)
let node_header v =
  let v_dims_s = ndarray_dims_to_string v.node.value in
  let g_dims_s = match v.node.grad with None -> "<no-grad>" | Some grad -> ndarray_dims_to_string grad in
  let dims_s =
    if String.equal v_dims_s g_dims_s then "dims " ^ v_dims_s
    else "dims val " ^ v_dims_s ^ " grad " ^ g_dims_s
  in
  let desc_l = match v.desc_label with None -> "" | Some l -> " " ^ l in
  "#" ^ Int.to_string v.id ^ desc_l ^ " op " ^ v.op_label ^ " " ^ dims_s ^ " ["
  ^ String.concat ~sep:"," (List.map v.children ~f:(fun { sub_node = { id; _ }; _ } -> Int.to_string id))
  ^ "]"
(*^" "^PrintBox_text.to_string (PrintBox.Simple.to_box v.label)*)

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

let to_dag ?(single_node = false) ?entries_per_axis ?extra_prefix ~with_id ~with_value ~with_grad v =
  let rec to_dag { sub_node = v; computed_externally } : PrintBox_utils.dag =
    let id = Int.to_string v.id in
    let children = if single_node then [] else List.map ~f:to_dag v.children in
    let desc_l = match v.desc_label with None -> "" | Some l -> l ^ " " in
    let op_l = match v.op_label with "" -> "" | l -> "<" ^ l ^ ">" in
    let prefix = "[" ^ id ^ "] " ^ desc_l ^ op_l in
    let prefix =
      match extra_prefix with
      | None -> prefix
      | Some f ->
          let extra = f v.annot in
          if String.is_empty extra then prefix else prefix ^ " " ^ extra
    in
    let labels = !(v.axis_labels) in
    let indices = !(v.default_display_indices) in
    match (computed_externally, with_value, with_grad, v.node.grad) with
    | true, _, _, _ -> `Embed_subtree_ID (Int.to_string v.id)
    | _, false, false, _ | _, false, true, None ->
        let txt = if with_id then prefix else desc_l ^ v.op_label in
        `Subtree_with_ID (id, `Tree (`Text txt, children))
    | _, true, false, _ | _, true, true, None ->
        let node =
          `Box (Nd.render_tensor ~brief:true ~prefix ?entries_per_axis ~labels ~indices v.node.value)
        in
        `Subtree_with_ID (id, `Tree (node, children))
    | _, false, true, Some grad ->
        let prefix = prefix ^ " Gradient" in
        let node = `Box (Nd.render_tensor ~brief:true ~prefix ?entries_per_axis ~labels ~indices grad) in
        `Subtree_with_ID (id, `Tree (node, children))
    | _, true, true, Some grad ->
        let node =
          let value = Nd.render_tensor ~brief:true ~prefix ?entries_per_axis ~labels ~indices v.node.value in
          let grad =
            Nd.render_tensor ~brief:true ~prefix:"Gradient" ?entries_per_axis ~labels ~indices grad
          in
          `Vlist (false, [ `Box value; `Box grad ])
        in
        `Subtree_with_ID (id, `Tree (node, children))
  in
  to_dag { sub_node = v; computed_externally = false }

let to_printbox ?single_node ?entries_per_axis ?extra_prefix ?(with_id = false) ?(with_value = true)
    ~with_grad ~depth n_id =
  to_dag ?single_node ?entries_per_axis ?extra_prefix ~with_id ~with_value ~with_grad n_id
  |> PrintBox_utils.reformat_dag depth

let print_node_preamble ?(print_missing = true) ?extra_prefix v =
  try
    let prefix = node_header v in
    let prefix =
      match extra_prefix with
      | None -> prefix
      | Some f ->
          let extra = f v.annot in
          if String.is_empty extra then prefix else prefix ^ " " ^ extra
    in
    Caml.Format.printf "Node %s" prefix;
    Caml.Format.printf "\n%!"
  with Not_found_s _ | Caml.Not_found ->
    if print_missing then Caml.Format.printf "Node #%d does not exist.\n%!" v.id

let print_tensor ~with_grad ~with_code ?(with_low_level = false) (style : array_print_style) t =
  let open Tensor in
  let sh = t.shape in
  let label =
    (match t.node.desc_label with None -> "" | Some l -> l ^ " ")
    ^ match t.node.op_label with "" -> "" | l -> "<" ^ l ^ "> "
  in
  let prefix =
    "[" ^ Int.to_string t.id ^ "]: " ^ label ^ "shape "
    ^ Shape.to_string_hum ~style:`Axis_number_and_size sh
    ^ " "
  in
  let indices =
    match style with
    | `Default -> Shape.default_display_indices sh
    | `N5_layout priorities ->
        let f = function
          | Either.Second i -> i
          | First _ -> invalid_arg "`N5_layout requires integer-only labels"
        in
        let p_labels = Shape.(axis_labels_of_spec priorities).labels |> Map.map ~f in
        Array.map (Shape.axis_map_to_dims_index p_labels) ~f:(fun d -> d.dim)
    | `Label_layout label_idcs ->
        let inv_labels =
          Map.to_alist sh.axis_labels |> List.map ~f:(fun (a, b) -> (b, a)) |> Map.of_alist (module String)
        in
        let inv_labels =
          match inv_labels with
          | `Duplicate_key l -> raise @@ Session_error ("`Label_layout found a repeating label: " ^ l, Some t)
          | `Ok inv_labels -> inv_labels
        in
        let idcs =
          List.map label_idcs ~f:(fun (l, i) ->
              match Map.find inv_labels l with
              | Some axis -> (axis, i)
              | None -> raise @@ Session_error ("`Label_layout label not found in shape: " ^ l, Some t))
        in
        Shape.axis_map_to_dims_index @@ Map.of_alist_exn (module Shape.AxisKey) idcs
    | `Inline -> [||]
  in
  let needs_spec =
    Fn.non Map.is_empty sh.axis_labels
    || Shape.(List.exists ~f:Shape.dim_1 @@ list_of_dims @@ dims_of_kind Input sh)
  in
  let labels = Shape.axis_map_to_dims_index ~default:"" sh.axis_labels in
  let axes_spec = if needs_spec then Some (Shape.to_string_hum ~style:`Only_labels sh) else None in
  let num_axes kind = List.length Shape.(list_of_dims @@ dims_of_kind kind sh) in
  let num_batch_axes = num_axes Shape.AxisKey.Batch in
  let num_input_axes = num_axes Shape.AxisKey.Input in
  let num_output_axes = num_axes Shape.AxisKey.Output in
  (match style with
  | `Inline ->
      Ndarray.pp_tensor_inline Caml.Format.std_formatter ~num_batch_axes ~num_input_axes ~num_output_axes
        ?axes_spec t.node.node.value
  | _ ->
      Ndarray.pp_tensor Caml.Format.std_formatter ~prefix ~labels ~indices t.node.node.value;
      Caml.Format.print_newline ());
  (if with_grad then
     match (style, t.node.node.grad) with
     | `Inline, Some grad ->
         Ndarray.pp_tensor_inline Caml.Format.std_formatter ~num_batch_axes ~num_input_axes ~num_output_axes
           ?axes_spec grad;
         Caml.Format.print_newline ()
     | _, Some grad ->
         Ndarray.pp_tensor Caml.Format.std_formatter ~prefix:(prefix ^ " Gradient ") ~labels ~indices grad;
         Caml.Format.print_newline ()
     | _ -> ());
  if with_code then (
    (match t.forward_body with
    | Noop -> ()
    | fwd_code -> Caml.Format.printf "Current forward body:@ %a@ " Low_level.fprint_code fwd_code);
    match t.diff with
    | Some { backprop_body = Noop; _ } -> ()
    | Some { backprop_body = bwd_code; _ } ->
        Caml.Format.printf "Current backprop body:@ %a@ " Low_level.fprint_code bwd_code
    | None -> ());
  if with_low_level then (
    (match t.forward_body with
    | Noop -> ()
    | fwd_code -> Caml.Format.printf "Current forward low-level body:@ %a@ " Low_level.fprint_low_level fwd_code);
    match t.diff with
    | Some { backprop_body = Noop; _ } -> ()
    | Some { backprop_body = bwd_code; _ } ->
        Caml.Format.printf "Current backprop low-level body:@ %a@ " Low_level.fprint_low_level bwd_code
    | None -> ());
  Stdio.printf "\n%!"

let print_global_roots ~with_grad ~with_code (style : array_print_style) =
  let open Tensor in
  List.iter (Map.to_alist ~key_order:`Increasing !global_roots) ~f:(fun (id, root) ->
      assert (id = root.id);
      print_tensor ~with_grad ~with_code style root)

let print_preamble () =
  (* Stdio.printf "%s\n%!" (Tensor.prefix_with_preamble "") *)
  Low_level.print_preamble ()

(** *** Session management. *** *)
type backend = Interpreter | Gccjit | Cuda [@@deriving sexp, equal]

let exec = ref Exec_as_gccjit.jit
let executor_error_message = ref Exec_as_gccjit.error_message
let cleanup_executor_session = ref Exec_as_gccjit.cleanup_session

let set_executor = function
  | Interpreter ->
      exec := Low_level.interpret;
      executor_error_message := Low_level.interpreter_error_message;
      Low_level.virtualize_settings.sequential_minibatch <- true;
      cleanup_executor_session := fun () -> ()
  | Gccjit ->
      exec := Exec_as_gccjit.jit;
      executor_error_message := Exec_as_gccjit.error_message;
      Low_level.virtualize_settings.sequential_minibatch <- true;
      cleanup_executor_session := Exec_as_gccjit.cleanup_session
  | Cuda ->
      exec := Exec_as_cuda.jit;
      executor_error_message := Exec_as_cuda.error_message;
      Low_level.virtualize_settings.sequential_minibatch <- false;
      cleanup_executor_session := Exec_as_cuda.cleanup_session

let initialize_host_tensors traced_store =
  List.iter ~f:(function
    | { Low_level.tensor = { id; field = Value } as ptr; dims; init_op } ->
        let dims = Array.map ~f:(fun d -> d.Shape.dim) @@ dims () in
        let tn = Low_level.get_node traced_store ptr in
        if tn.non_virtual && tn.non_device_only then
          (Low_level.get id).node.value <- Ndarray.create !Tensor.default_value_prec dims init_op
    | { tensor = { id; field = Grad } as ptr; dims; init_op } ->
        let dims = Array.map ~f:(fun d -> d.Shape.dim) @@ dims () in
        let tn = Low_level.get_node traced_store ptr in
        let v = (Low_level.get id).node in
        if tn.non_virtual && tn.non_device_only then
          g <- Some (Ndarray.create !Tensor.default_grad_prec dims init_op)
        else assert (Option.is_some g))

let compile_routine ~name code =
  let open Tensor in
  let num_inits = List.length !session_initializations in
  let to_init = num_inits - !session_initialized in
  session_initialized := num_inits;
  let traced_store, compiled = Low_level.compile_proc ~name ~for_step_update:false code in
  (* Only initialize after compilation, to know which nodes are virtual. *)
  initialize_host_tensors traced_store @@ List.take !session_initializations to_init;
  !exec ~name (traced_store, compiled)

let session_params () = Low_level.param_nodes ~from_id:!Tensor.first_session_id ()
let minus_learning_rate : Tensor.t option ref = ref None
let last_refresh_roots = ref !Tensor.global_roots
let last_with_backprop = ref false
let last_update_params = ref false
let last_updates_per_run = ref 1
let session_step_update = ref Low_level.Noop
let session_step_update_compiled = ref (Hashtbl.Poly.create (), Low_level.(Comment "Noop"))
let session_step_update_routine = ref (fun () -> ())

let generate_params_update ~(minus_lr : Tensor.t) ?params () =
  let params = match params with Some p -> p | None -> Hashtbl.data @@ session_params () in
  let module CDSL = Low_level.CDSL in
  let module NTDSL = Operation.NTDSL in
  List.map params ~f:(fun v -> [%nn_cd v =+ minus_lr * g ~logic:"."])

let print_session_code ?(compiled = false) () =
  (* FIXME: figure out if / why this isn't idempotent. *)
  Caml.Format.set_margin !Arrayjit.Low_level.code_sexp_margin;
  if compiled then
    Caml.Format.printf "Compiled session step update code:@ %a" Sexp.pp_hum
      (sexp_of_t @@ snd !session_step_update_compiled)
  else Caml.Format.printf "Session step update code:@ %a" fprint_code !session_step_update;
  Caml.Format.print_newline ()

let refresh_session ?(regenerate = false) ?(with_backprop = true) ?update_params ?(reinit = false)
    ?(updates_per_run = 1) ?(run = true) ?(force_no_init = false) ?(verbose = false) () =
  let open Tensor in
  let update_params = Option.value update_params ~default:with_backprop in
  if verbose then
    Stdio.printf "refresh_session: regenerate=%b, update_params=%b, reinit=%b, run=%b, force_no_init=%b\n%!"
      regenerate update_params reinit run force_no_init;
  if force_no_init && reinit then invalid_arg "refresh_session: ~force_no_init conflicts with ~reinit";
  if update_params && not with_backprop then
    invalid_arg "refresh_session: ~update_params:true requires ~with_backprop:true";
  (* Initialization and the forward processing. *)
  let roots_changed = not @@ phys_equal !last_refresh_roots !Tensor.global_roots in
  last_refresh_roots := !Tensor.global_roots;
  let backprop_changed = Bool.(!last_with_backprop <> with_backprop) in
  last_with_backprop := with_backprop;
  let update_params_changed = Bool.(!last_update_params <> update_params) in
  last_update_params := update_params;
  let updates_per_run_changed = !last_updates_per_run <> updates_per_run in
  last_updates_per_run := updates_per_run;
  if regenerate || roots_changed then List.iter !session_shape_updates ~f:Low_level.update_shape;
  if regenerate then session_initialized := 0;
  let generating =
    regenerate || roots_changed || backprop_changed || update_params_changed || updates_per_run_changed
  in
  let name = "session_step_update" in
  if generating then (
    let open Code in
    let forward =
      Block_comment
        ( "Forward pass",
          sequential
          @@ List.map (Map.to_alist ~key_order:`Increasing !global_roots) ~f:(fun (_node_id, root) ->
                 get_toplevel_forward root) )
    in
    let zero_grads =
      if not with_backprop then Noop
      else
        Block_comment
          ( "Zero grads",
            sequential
            @@ List.filter_map (Map.to_alist ~key_order:`Decreasing !global_roots) ~f:(fun (_node_id, root) ->
                   Option.some_if (Option.value_exn root.diff).needs_gradient @@ root) )
    in
    let backprop =
      if not with_backprop then Noop
      else
        Block_comment
          ( "Backprop pass",
            sequential
            @@ List.filter_map (Map.to_alist ~key_order:`Decreasing !global_roots) ~f:(fun (_node_id, root) ->
                   Option.some_if (Option.value_exn root.diff).needs_gradient
                     (Option.value_exn root.diff).backprop_body) )
    in
    let update_params_code =
      match (update_params, !minus_learning_rate) with
      | true, Some minus_lr -> generate_params_update ~minus_lr ()
      | _ -> []
    in
    let params_update =
      if List.is_empty update_params_code then Noop
      else Synchronize ("Params update", all_parallel update_params_code)
    in
    (* Roots at the time of compilation are hosted, so that they can be consumed downstream. *)
    Map.iter_keys !Tensor.global_roots ~f:(fun id ->
        let v = Low_level.get id in
        v.annot.value_never_virtual <- true;
        v.annot.value_never_device_only <- true);
    (* Params are hosted also, so they can be updated over multiple steps, stored, updated etc.
       Params would typically be automatically found non-virtual and non-device-only, but there
       are corner cases we prevent here. *)
    Hashtbl.iter ~f:(fun v ->
        v.Node.annot.value_never_virtual <- true;
        v.Node.annot.value_never_device_only <- true)
    @@ session_params ();
    session_step_update := sequential [ preparation; forward; backprop; params_update ];
    if verbose then Stdio.printf "refresh_session: compiling\n%!";
    if updates_per_run <= 1 then
      session_step_update_compiled := compile_proc ~name ~verbose ~for_step_update:true !session_step_update
    else
      let traced_store, compiled = compile_proc ~name ~verbose ~for_step_update:true !session_step_update in
      session_step_update_compiled :=
        ( traced_store,
          Low_level.(
            For_loop
              {
                index = Shape.get_sym_for_axis Shape.Dim;
                from_ = 0;
                to_ = updates_per_run - 1;
                body = Lines [| Comment "Update sub-step"; compiled |];
                trace_it = false;
              }) ));
  if generating || reinit || roots_changed then (
    let num_inits = List.length !session_initializations in
    let to_init = num_inits - !session_initialized in
    if verbose then Stdio.printf "refresh_session: initializing host tensors\n%!";
    initialize_host_tensors (fst !session_step_update_compiled) @@ List.take !session_initializations to_init;
    session_initialized := num_inits);
  if (not force_no_init) && (generating || reinit) then
    session_step_update_routine := !exec ~name ~verbose !session_step_update_compiled;
  if run && updates_per_run > 0 then (
    if verbose then Stdio.printf "refresh_session: running\n%!";
    !session_step_update_routine ();
    if verbose then Stdio.printf "refresh_session: finished\n%!")

(** Discards global roots, advances [Tensor.first_session_id] to [Node.state.unique_id].
    Discards all computations (forward, backward, update params, data fetches), but keeps
    the already computed data / parameters. *)
let close_session () =
  Tensor.first_session_id := session_state.next_session_id;
  Tensor.global_roots := Map.empty (module Int);
  Tensor.session_shape_updates := [];
  Tensor.session_initializations := [];
  Tensor.session_initialized := 0;
  session_step_update := Noop;
  session_step_update_compiled := (Hashtbl.Poly.create (), Comment "Noop");
  (session_step_update_routine := fun () -> ());
  minus_learning_rate := None;
  !cleanup_executor_session ()

(** Discards global roots, rolls back [Node.state.unique_id] to [Tensor.first_session_id], discards
    the corresponding elements from [Node.state.node_store]. *)
let drop_session () =
  let beginning_of_session = !Tensor.first_session_id in
  close_session ();
  Tensor.first_session_id := beginning_of_session;
  for i = !Tensor.first_session_id to session_state.next_session_id - 1 do
    Hashtbl.remove Low_level.global_node_store i
  done;
  Node.unique_id := !Tensor.first_session_id

(** Discards all global state, rolls back [Node.state.unique_id] and [Tensor.first_session_id]
    to 1. *)
let drop_all_sessions () =
  Tensor.first_session_id := 1;
  drop_session ();
  Hashtbl.clear Low_level.global_node_store;
  Node.unique_id := 1

let save_all_tensors ~name =
  let out = Npy.Npz.open_out (name ^ ".npz") in
  Hashtbl.iter Low_level.global_node_store ~f:(fun v ->
      let save field arr = Npy.Npz.write out Node.(tensor_ptr_name { id = v.id; field }) arr in
      let f arr = save Value arr in
      Ndarray.map { f } v.node.value;
      let f arr = save Grad arr in
      Option.iter v.node.grad ~f:(Ndarray.map { f }))

(** Restores the content of already-existing tensors from the file [name ^ ".npz"]. With [~partially:true],
    does not complain about tensors missing in the file. *)
let restore_tensors ?(partially = false) f_name =
  let inp = Npy.Npz.open_in (f_name ^ ".npz") in
  Hashtbl.iteri Low_level.global_node_store ~f:(fun ~key:id ~data:n ->
      let restore field =
        let ptr = Node.{ id; field } in
        match Low_level.get_tensor ptr with
        | None -> ()
        | Some arr ->
            let t_name = Node.(tensor_ptr_name { id = v.id; field }) in
            let src = Npy.Npz.read inp t_name in
            let f prec dst =
              match Npy.to_bigarray Bigarray.c_layout (Ndarray.precision_to_bigarray_kind prec) src with
              | None -> if not partially then failwith ("Session.restore_tensors: missing tensor " ^ t_name)
              | Some src -> Ndarray.A.blit src dst
            in
            Ndarray.map_with_prec { f } arr
      in
      restore Value;
      restore Node.Grad)

let value_1d_points ?from_axis ~xdim t = Ndarray.retrieve_1d_points ?from_axis ~xdim t.Tensor.node.node.value

let value_2d_points ?from_axis ~xdim ~ydim t =
  Ndarray.retrieve_2d_points ?from_axis ~xdim ~ydim t.Tensor.node.node.value

let grad_1d_points ?from_axis ~xdim t =
  match t.Tensor.node.node.grad with None -> [||] | Some a -> Ndarray.retrieve_1d_points ?from_axis ~xdim a

let grad_2d_points ?from_axis ~xdim ~ydim t =
  match t.Tensor.node.node.grad with
  | None -> [||]
  | Some a -> Ndarray.retrieve_2d_points ?from_axis ~xdim ~ydim a

let set_value t = Ndarray.set_from_float t.Tensor.node.node.value
let get_value t = Ndarray.get_as_float t.Tensor.node.node.value
let set_grad t = Ndarray.set_from_float (Option.value_exn t.Tensor.node.node.grad)
let get_grad t = Ndarray.get_as_float (Option.value_exn t.Tensor.node.node.grad)

module O = struct
  (** Get the value at the given indices. *)
  let ( .@{} ) = get_value

  (** Set the value at the given indices. *)
  let ( .@{}<- ) = set_value

  (** Get the gradient at the given indices. *)
  let ( .@%{} ) = get_grad

  (** Set the gradient at the given indices. *)
  let ( .@%{}<- ) = set_grad

  (** Get the value at the given index from a single-axis shape tensor. *)
  let ( .@[] ) t indx = get_value t [| indx |]

  (** Set the value at the given index for a single-axis shape tensor. *)
  let ( .@[]<- ) t indx = set_value t [| indx |]

  (** Get the gradient at the given index from a single-axis shape tensor. *)
  let ( .@%[] ) t indx = get_grad t [| indx |]

  (** Set the gradient at the given index for a single-axis shape tensor. *)
  let ( .@%[]<- ) t indx = set_grad t [| indx |]
end

module SDSL = struct
  type nonrec backend = backend = Interpreter | Gccjit | Cuda

  module O = O

  let set_executor = set_executor
  let refresh_session = refresh_session
  let drop_session = drop_session
  let drop_all_sessions = drop_all_sessions
  let close_session = close_session
  let compile_routine = compile_routine
  let session_params = session_params
  let minus_learning_rate = minus_learning_rate

  let print_node_tree ?entries_per_axis ?(with_backend_info = false) ?with_id ?with_value ~with_grad ~depth id
      =
    let extra_prefix = if with_backend_info then Some (fun annot -> annot.Low_level.backend_info) else None in
    try
      let v = Low_level.get id in
      PrintBox_text.output Stdio.stdout
      @@ Node.to_printbox ?entries_per_axis ?with_id ?with_value ~with_grad ?extra_prefix ~depth v
    with Not_found_s _ | Caml.Not_found -> Caml.Format.printf "Node #%d does not exist.\n%!" id

  let max_sublabel_length = Tensor.max_sublabel_length
  let print_tensor = print_tensor
  let print_global_roots = print_global_roots
  let print_preamble = print_preamble
  let print_session_code = print_session_code
  let print_decimals_precision = Ndarray.print_decimals_precision
  let get_root = get_root
  let get_node = get_node
  let set_values t cs = Ndarray.(init (Constant_fill cs) t.Tensor.node.node.value)
  let set_grads t cs = Ndarray.(init (Constant_fill cs) (Option.value_exn t.Tensor.node.node.grad))

  let set_fully_on_host t =
    t.Tensor.node.annot.value_never_virtual <- true;
    t.node.annot.grad_never_virtual <- true;
    t.Tensor.node.annot.value_never_device_only <- true;
    t.node.annot.grad_never_device_only <- true

  let everything_fully_on_host () =
    for id = !Tensor.first_session_id to session_state.next_session_id - 1 do
      let v = Low_level.get id in
      v.annot.value_never_virtual <- true;
      v.annot.grad_never_virtual <- true;
      v.annot.value_never_device_only <- true;
      v.annot.grad_never_device_only <- true
    done

  let everything_on_host_or_inlined () =
    for id = !Tensor.first_session_id to session_state.next_session_id - 1 do
      let v = Low_level.get id in
      v.annot.value_never_device_only <- true;
      v.annot.grad_never_device_only <- true
    done

  let value_1d_points ?from_axis ~xdim t =
    Ndarray.retrieve_1d_points ?from_axis ~xdim t.Tensor.node.node.value

  let value_2d_points ?from_axis ~xdim ~ydim t =
    Ndarray.retrieve_2d_points ?from_axis ~xdim ~ydim t.Tensor.node.node.value

  let grad_1d_points ?from_axis ~xdim t =
    match t.Tensor.node.node.grad with
    | None -> [||]
    | Some a -> Ndarray.retrieve_1d_points ?from_axis ~xdim a

  let grad_2d_points ?from_axis ~xdim ~ydim t =
    match t.Tensor.node.node.grad with
    | None -> [||]
    | Some a -> Ndarray.retrieve_2d_points ?from_axis ~xdim ~ydim a

  let enable_all_debugs ?(trace_interpreter = false) ?(hosted_only = true) () =
    Low_level.CDSL.with_debug := true;
    Low_level.CDSL.keep_files_in_run_directory := true;
    if hosted_only then Low_level.CDSL.virtualize_settings.enable_device_only <- false;
    if trace_interpreter then Low_level.CDSL.debug_verbose_trace := true

  let disable_all_debugs ?(restore_defaults = false) () =
    Low_level.CDSL.debug_verbose_trace := false;
    Low_level.CDSL.with_debug := false;
    Low_level.CDSL.keep_files_in_run_directory := false;
    if restore_defaults then Low_level.CDSL.virtualize_settings.enable_device_only <- true

  let default_value_prec = Tensor.default_value_prec
  let default_grad_prec = Tensor.default_grad_prec
  let global_host_size_in_bytes () = Low_level.global_host_size_in_bytes ()
  let num_domains = Node.num_domains
end
