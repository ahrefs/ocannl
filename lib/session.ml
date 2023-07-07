(** Managing a computation session. *)

open Base

let get_root id =
  let open Formula in
  match Map.find !global_roots id with
  | Some r -> r
  | None ->
      let msg =
        if id >= !first_session_id && id < Node.global.unique_id then
          "get_root: Node " ^ Int.to_string id ^ " is a subformula"
        else if id >= Node.global.unique_id then
          "get_root: Node " ^ Int.to_string id ^ " has not been created yet"
        else if id < 1 then "get_root: Node IDs start from 1"
        else "get_root: Node " ^ Int.to_string id ^ " is outside the current session"
      in
      raise @@ Session_error (msg, None)

let get_node id =
  let open Node in
  match Hashtbl.find NodeUI.global_node_store id with
  | Some r -> r
  | None ->
      let msg =
        if id >= global.unique_id then "get_node: Node " ^ Int.to_string id ^ " has not been created yet"
        else if id < 1 then "get_root: Node IDs start from 1"
        else "get_node: Node " ^ Int.to_string id ^ " has been removed or lives on a different machine"
      in
      raise @@ Formula.Session_error (msg, None)

(** *** Printing. *** *)

let print_formula ~with_grad ~with_code ?(with_low_level = false) (style : NodeUI.array_print_style) m =
  let open Formula in
  let sh = m.shape in
  let label =
    (match m.node.desc_label with None -> "" | Some l -> l ^ " ")
    ^ match m.node.op_label with "" -> "" | l -> "<" ^ l ^ "> "
  in
  let prefix =
    "[" ^ Int.to_string m.id ^ "]: " ^ label ^ "shape "
    ^ Shape.to_string_hum ~style:`Axis_number_and_size sh
    ^ " "
  in
  let indices =
    match style with
    | `Default -> NodeUI.default_display_indices sh
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
          | `Duplicate_key l -> raise @@ Session_error ("`Label_layout found a repeating label: " ^ l, Some m)
          | `Ok inv_labels -> inv_labels
        in
        let idcs =
          List.map label_idcs ~f:(fun (l, i) ->
              match Map.find inv_labels l with
              | Some axis -> (axis, i)
              | None -> raise @@ Session_error ("`Label_layout label not found in shape: " ^ l, Some m))
        in
        Shape.axis_map_to_dims_index @@ Map.of_alist_exn (module Shape.AxisKey) idcs
    | `Inline -> [||]
  in
  let needs_spec =
    Fn.non Map.is_empty sh.axis_labels
    || Shape.(List.exists ~f:Shape.dim_1 @@ list_of_dims @@ dims_of_kind Input sh)
  in
  let labels = Shape.axis_map_to_dims_index ~default:"" sh.axis_labels in
  let labels_spec = if needs_spec then Some (Shape.to_string_hum ~style:`Only_labels sh) else None in
  let num_axes kind = List.length Shape.(list_of_dims @@ dims_of_kind kind sh) in
  let num_batch_axes = num_axes Shape.AxisKey.Batch in
  let num_input_axes = num_axes Shape.AxisKey.Input in
  let num_output_axes = num_axes Shape.AxisKey.Output in
  (match style with
  | `Inline ->
      NodeUI.pp_tensor_inline Caml.Format.std_formatter ~num_batch_axes ~num_input_axes ~num_output_axes
        ?labels_spec m.node.node.value
  | _ ->
      NodeUI.pp_tensor Caml.Format.std_formatter ~prefix ~labels ~indices m.node.node.value;
      Caml.Format.print_newline ());
  (if with_grad then
     match (style, m.node.node.grad) with
     | `Inline, Some grad ->
         NodeUI.pp_tensor_inline Caml.Format.std_formatter ~num_batch_axes ~num_input_axes ~num_output_axes
           ?labels_spec grad;
         Caml.Format.print_newline ()
     | _, Some grad ->
         NodeUI.pp_tensor Caml.Format.std_formatter ~prefix:(prefix ^ " Gradient ") ~labels ~indices grad;
         Caml.Format.print_newline ()
     | _ -> ());
  if with_code then (
    (match m.forward_body with
    | Noop -> ()
    | fwd_code -> Caml.Format.printf "Current forward body:@ %a@ " Code.fprint_code fwd_code);
    match m.form with
    | Some { backprop_body = Noop; _ } -> ()
    | Some { backprop_body = bwd_code; _ } ->
        Caml.Format.printf "Current backprop body:@ %a@ " Code.fprint_code bwd_code
    | None -> ());
  if with_low_level then (
    (match m.forward_body with
    | Noop -> ()
    | fwd_code -> Caml.Format.printf "Current forward low-level body:@ %a@ " Code.fprint_low_level fwd_code);
    match m.form with
    | Some { backprop_body = Noop; _ } -> ()
    | Some { backprop_body = bwd_code; _ } ->
        Caml.Format.printf "Current backprop low-level body:@ %a@ " Code.fprint_low_level bwd_code
    | None -> ());
  Stdio.printf "\n%!"

let print_global_roots ~with_grad ~with_code (style : NodeUI.array_print_style) =
  let open Formula in
  List.iter (Map.to_alist ~key_order:`Increasing !global_roots) ~f:(fun (id, root) ->
      assert (id = root.id);
      print_formula ~with_grad ~with_code style root)

let print_preamble ?(full_shape = false) () =
  (* Stdio.printf "%s\n%!" (Formula.prefix_with_preamble "") *)
  NodeUI.print_preamble ~full_shape ()

(** *** Session management. *** *)
type backend = Interpreter | Gccjit | Cuda [@@deriving sexp, equal]

let num_parallel_tasks = ref 1
let num_domains = Caml.Domain.recommended_domain_count ()
let task_pool = Domainslib.Task.setup_pool ~name:"session_task_pool" ~num_domains ()

let multicore_step_func jit_task_id_func ~name ?verbose jit_args =
  let task_id_func = jit_task_id_func ~name ?verbose jit_args in
  fun () ->
    if !num_parallel_tasks = 1 then task_id_func ~task_id:0
    else
      Domainslib.Task.run task_pool (fun () ->
          Domainslib.Task.parallel_for task_pool ~start:0 ~finish:(!num_parallel_tasks - 1)
            ~body:(fun task_id -> task_id_func ~task_id))

let gccjit_jit_step_func = multicore_step_func Exec_as_gccjit.jit_task_id_func
let exec_step_func = ref gccjit_jit_step_func
let exec_unit_func = ref Exec_as_gccjit.jit_unit_func
let executor_error_message = ref Exec_as_gccjit.error_message
let cleanup_executor_session = ref Exec_as_gccjit.cleanup_session

let set_executor = function
  | Interpreter ->
      exec_step_func := multicore_step_func Code.interpret_task_id_func;
      exec_unit_func := Code.interpret_unit_func;
      executor_error_message := Code.interpreter_error_message;
      cleanup_executor_session := fun () -> ()
  | Gccjit ->
      exec_step_func := multicore_step_func Exec_as_gccjit.jit_task_id_func;
      exec_unit_func := Exec_as_gccjit.jit_unit_func;
      executor_error_message := Exec_as_gccjit.error_message;
      cleanup_executor_session := Exec_as_gccjit.cleanup_session
  | Cuda ->
      exec_step_func := Exec_as_cuda.jit_step_func;
      exec_unit_func := Exec_as_cuda.jit_unit_func;
      executor_error_message := Exec_as_cuda.error_message;
      cleanup_executor_session := Exec_as_cuda.cleanup_session

let initialize_host_tensors traced_store =
  List.iter ~f:(function
    | { Code.tensor = { id; field = Value } as ptr; dims; init_op } ->
        let tn = Code.get_node traced_store ptr in
        if tn.non_virtual && tn.non_device_only then
          (NodeUI.get id).node.value <- NodeUI.create_ndarray !Formula.default_value_prec (dims ()) init_op
    | { tensor = { id; field = Grad } as ptr; dims; init_op } ->
        let tn = Code.get_node traced_store ptr in
        let n = (NodeUI.get id).node in
        if tn.non_virtual && tn.non_device_only then
          n.grad <- Some (NodeUI.create_ndarray !Formula.default_grad_prec (dims ()) init_op)
        else assert (Option.is_some n.grad))

let compile_routine ~name code =
  let open Formula in
  let num_inits = List.length !session_initializations in
  let to_init = num_inits - !session_initialized in
  session_initialized := num_inits;
  let traced_store, compiled = Code.compile_proc ~name ~for_step_update:false code in
  (* Only initialize after compilation, to know which nodes are virtual. *)
  initialize_host_tensors traced_store @@ List.take !session_initializations to_init;
  !exec_unit_func ~name (traced_store, compiled)

let session_params () = NodeUI.param_nodes ~from_id:!Formula.first_session_id ()
let minus_learning_rate : Formula.t option ref = ref None
let last_refresh_roots = ref !Formula.global_roots
let last_with_backprop = ref false
let last_update_params = ref false
let last_run_for_steps = ref 1
let session_step_update = ref Code.Noop
let session_step_update_compiled = ref (Hashtbl.Poly.create (), Code.(Comment "Noop"))
let session_step_update_routine = ref (fun () -> ())

let generate_params_update ~(minus_lr : Formula.t) ?params () =
  let params = match params with Some p -> p | None -> Hashtbl.data @@ session_params () in
  let module CDSL = Code.CDSL in
  let module NFDSL = Operation.NFDSL in
  List.map params ~f:(fun n -> [%nn_cd n =+ minus_lr * n.grad ~logic:"."])

let print_session_code ?(compiled = false) () =
  let open Code in
  (* FIXME: figure out if / why this isn't idempotent. *)
  Caml.Format.set_margin !Code.code_sexp_margin;
  if compiled then
    Caml.Format.printf "Compiled session step update code:@ %a" Sexp.pp_hum
      (sexp_of_unit_low_level @@ snd !session_step_update_compiled)
  else Caml.Format.printf "Session step update code:@ %a" fprint_code !session_step_update;
  Caml.Format.print_newline ()

let refresh_session ?(regenerate = false) ?(with_backprop = true) ?update_params ?(reinit = false)
    ?(run_for_steps = 1) ?(run = true) ?(force_no_init = false) ?(verbose = false) () =
  let open Formula in
  let update_params = Option.value update_params ~default:with_backprop in
  if verbose then
    Stdio.printf "refresh_session: regenerate=%b, update_params=%b, reinit=%b, run=%b, force_no_init=%b\n%!"
      regenerate update_params reinit run force_no_init;
  if force_no_init && reinit then invalid_arg "refresh_session: ~force_no_init conflicts with ~reinit";
  if update_params && not with_backprop then
    invalid_arg "refresh_session: ~update_params:true requires ~with_backprop:true";
  (* Initialization and the forward processing. *)
  let roots_changed = not @@ phys_equal !last_refresh_roots !Formula.global_roots in
  last_refresh_roots := !Formula.global_roots;
  let backprop_changed = Bool.(!last_with_backprop <> with_backprop) in
  last_with_backprop := with_backprop;
  let update_params_changed = Bool.(!last_update_params <> update_params) in
  last_update_params := update_params;
  let run_for_steps_changed = !last_run_for_steps <> run_for_steps in
  last_run_for_steps := run_for_steps;
  if regenerate || roots_changed then List.iter !session_shape_updates ~f:Shape.propagate_shapes;
  if regenerate then session_initialized := 0;
  let generating =
    regenerate || roots_changed || backprop_changed || update_params_changed || run_for_steps_changed
  in
  let name = "session_step_update" in
  if generating then (
    let open Code in
    let preparation =
      Block_comment
        ( "Preparation",
          Par
            ( Block_comment ("Prepare forward pass", all_parallel !Formula.session_prepare_forward),
              Block_comment ("Prepare backprop pass", all_parallel !Formula.session_prepare_backprop) ) )
    in
    let forward =
      Block_comment
        ( "Forward pass",
          sequential
          @@ List.map (Map.to_alist ~key_order:`Increasing !global_roots) ~f:(fun (_node_id, root) ->
                 get_toplevel_forward root) )
    in
    let backprop =
      if not with_backprop then Noop
      else
        Block_comment
          ( "Backprop pass",
            sequential
            @@ List.filter_map (Map.to_alist ~key_order:`Decreasing !global_roots) ~f:(fun (_node_id, root) ->
                   Option.some_if (Option.value_exn root.form).needs_gradient @@ get_toplevel_backprop root)
          )
    in
    let update_params_code =
      match (update_params, !minus_learning_rate) with
      | true, Some minus_lr -> generate_params_update ~minus_lr ()
      | _ -> []
    in
    let params_update =
      if List.is_empty update_params_code then Noop
      else Block_comment ("Params update", all_parallel update_params_code)
    in
    let postprocess =
      if List.is_empty !Formula.session_postprocess then Noop
      else Block_comment ("Postprocess", all_parallel !Formula.session_postprocess)
    in
    (* Roots at the time of compilation are hosted, so that they can be consumed downstream. *)
    Map.iter_keys !Formula.global_roots ~f:(fun id ->
        let n = NodeUI.get id in
        n.value_never_virtual <- true;
        n.value_never_device_only <- true);
    (* Params are hosted also, so they can be updated over multiple steps, stored, updated etc.
       Params would typically be automatically found non-virtual and non-device-only, but there
       are corner cases we prevent here. *)
    Hashtbl.iter ~f:(fun n ->
        n.NodeUI.value_never_virtual <- true;
        n.value_never_device_only <- true)
    @@ session_params ();
    session_step_update := sequential [ preparation; forward; backprop; params_update; postprocess ];
    if verbose then Stdio.printf "refresh_session: compiling\n%!";
    if run_for_steps <= 1 then
      session_step_update_compiled := compile_proc ~name ~verbose ~for_step_update:true !session_step_update
    else
      let traced_store, compiled = compile_proc ~name ~verbose ~for_step_update:true !session_step_update in
      session_step_update_compiled :=
        ( traced_store,
          Code.(
            For_loop
              {
                index = Shape.get_sym_for_axis Shape.Dim;
                from_ = 0;
                to_ = run_for_steps - 1;
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
    session_step_update_routine := !exec_step_func ~name ~verbose !session_step_update_compiled;
  if run && run_for_steps > 0 then (
    if verbose then Stdio.printf "refresh_session: running\n%!";
    !session_step_update_routine ();
    if verbose then Stdio.printf "refresh_session: finished\n%!")

(** Discards global roots, advances [Formula.first_session_id] to [Node.state.unique_id].
    Discards all computations (forward, backward, update params, data fetches), but keeps
    the already computed data / parameters. *)
let close_session () =
  Formula.first_session_id := Node.global.unique_id;
  Formula.global_roots := Map.empty (module Int);
  Formula.session_shape_updates := [];
  Formula.session_initializations := [];
  Formula.session_initialized := 0;
  Formula.session_prepare_forward := [];
  Formula.session_prepare_backprop := [];
  Formula.session_postprocess := [];
  session_step_update := Noop;
  session_step_update_compiled := (Hashtbl.Poly.create (), Comment "Noop");
  (session_step_update_routine := fun () -> ());
  minus_learning_rate := None;
  !cleanup_executor_session ()

(** Discards global roots, rolls back [Node.state.unique_id] to [Formula.first_session_id], discards
    the corresponding elements from [Node.state.node_store]. *)
let drop_session () =
  let beginning_of_session = !Formula.first_session_id in
  close_session ();
  Formula.first_session_id := beginning_of_session;
  for i = !Formula.first_session_id to Node.global.unique_id - 1 do
    Hashtbl.remove NodeUI.global_node_store i;
    Hashtbl.remove Node.global.node_store i
  done;
  Node.global.unique_id <- !Formula.first_session_id

(** Discards all global state, rolls back [Node.state.unique_id] and [Formula.first_session_id]
    to 1. *)
let drop_all_sessions () =
  Formula.first_session_id := 1;
  drop_session ();
  Hashtbl.clear NodeUI.global_node_store;
  Hashtbl.clear Node.global.node_store;
  Node.global.unique_id <- 1

let value_1d_points ?from_axis ~xdim m = NodeUI.retrieve_1d_points ?from_axis ~xdim m.Formula.node.node.value

let value_2d_points ?from_axis ~xdim ~ydim m =
  NodeUI.retrieve_2d_points ?from_axis ~xdim ~ydim m.Formula.node.node.value

let grad_1d_points ?from_axis ~xdim m =
  match m.Formula.node.node.grad with None -> [||] | Some a -> NodeUI.retrieve_1d_points ?from_axis ~xdim a

let grad_2d_points ?from_axis ~xdim ~ydim m =
  match m.Formula.node.node.grad with
  | None -> [||]
  | Some a -> NodeUI.retrieve_2d_points ?from_axis ~xdim ~ydim a

let set_value m = Node.set_from_float m.Formula.node.node.value
let get_value m = Node.get_as_float m.Formula.node.node.value
let set_grad m = Node.set_from_float (Option.value_exn m.Formula.node.node.grad)
let get_grad m = Node.get_as_float (Option.value_exn m.Formula.node.node.grad)

module O = struct
  (** Get the value at the given indices. *)
  let ( .@{} ) = get_value

  (** Set the value at the given indices. *)
  let ( .@{}<- ) = set_value

  (** Get the gradient at the given indices. *)
  let ( .@%{} ) = get_grad

  (** Set the gradient at the given indices. *)
  let ( .@%{}<- ) = set_grad

  (** Get the value at the given index from a single-axis shape formula. *)
  let ( .@[] ) m indx = get_value m [| indx |]

  (** Set the value at the given index for a single-axis shape formula. *)
  let ( .@[]<- ) m indx = set_value m [| indx |]

  (** Get the gradient at the given index from a single-axis shape formula. *)
  let ( .@%[] ) m indx = get_grad m [| indx |]

  (** Set the gradient at the given index for a single-axis shape formula. *)
  let ( .@%[]<- ) m indx = set_grad m [| indx |]
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

  let print_node_tree ?entries_per_axis ?with_backend_info ?with_id ?with_value ~with_grad ~depth id =
    try
      PrintBox_text.output Stdio.stdout
      @@ NodeUI.to_printbox ?entries_per_axis ?with_backend_info ?with_id ?with_value ~with_grad ~depth id
    with Not_found_s _ | Caml.Not_found -> Caml.Format.printf "Node #%d does not exist.\n%!" id

  let max_sublabel_length = Formula.max_sublabel_length
  let print_formula = print_formula
  let print_global_roots = print_global_roots
  let print_preamble = print_preamble
  let print_session_code = print_session_code
  let print_decimals_precision = NodeUI.print_decimals_precision
  let get_root = get_root
  let get_node = get_node
  let set_values m cs = Node.(init_ndarray (Constant_fill cs) m.Formula.node.node.value)
  let set_grads m cs = Node.(init_ndarray (Constant_fill cs) (Option.value_exn m.Formula.node.node.grad))

  let set_fully_on_host m =
    m.Formula.node.value_never_virtual <- true;
    m.node.grad_never_virtual <- true;
    m.Formula.node.value_never_device_only <- true;
    m.node.grad_never_device_only <- true

  let everything_fully_on_host () =
    for id = !Formula.first_session_id to Node.global.unique_id - 1 do
      let n = NodeUI.get id in
      n.value_never_virtual <- true;
      n.grad_never_virtual <- true;
      n.value_never_device_only <- true;
      n.grad_never_device_only <- true
    done

  let everything_on_host_or_inlined () =
    for id = !Formula.first_session_id to Node.global.unique_id - 1 do
      let n = NodeUI.get id in
      n.value_never_device_only <- true;
      n.grad_never_device_only <- true
    done

  let value_1d_points ?from_axis ~xdim m =
    NodeUI.retrieve_1d_points ?from_axis ~xdim m.Formula.node.node.value

  let value_2d_points ?from_axis ~xdim ~ydim m =
    NodeUI.retrieve_2d_points ?from_axis ~xdim ~ydim m.Formula.node.node.value

  let grad_1d_points ?from_axis ~xdim m =
    match m.Formula.node.node.grad with
    | None -> [||]
    | Some a -> NodeUI.retrieve_1d_points ?from_axis ~xdim a

  let grad_2d_points ?from_axis ~xdim ~ydim m =
    match m.Formula.node.node.grad with
    | None -> [||]
    | Some a -> NodeUI.retrieve_2d_points ?from_axis ~xdim ~ydim a

  let enable_all_debugs ?(trace_interpreter = false) ?(hosted_only = true) () =
    Code.CDSL.with_debug := true;
    Code.CDSL.keep_files_in_run_directory := true;
    if hosted_only then Code.CDSL.virtualize_settings.enable_device_only <- false;
    if trace_interpreter then Code.CDSL.debug_trace_interpretation := true

  let disable_all_debugs ?(restore_defaults = false) () =
    Code.CDSL.debug_trace_interpretation := false;
    Code.CDSL.with_debug := false;
    Code.CDSL.keep_files_in_run_directory := false;
    if restore_defaults then Code.CDSL.virtualize_settings.enable_device_only <- true

  let default_value_prec = Formula.default_value_prec
  let default_grad_prec = Formula.default_grad_prec
  let global_host_size_in_bytes () = Node.global_host_size_in_bytes ()
  let num_parallel_tasks = num_parallel_tasks
  let num_domains = num_domains
end
