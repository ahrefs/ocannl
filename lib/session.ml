(** Managing a computation session. *)

open Base

let get_root id =
  let open Formula in
  match Map.find !global_roots id with
  | Some r -> r
  | None ->
      let msg =
        if id >= !first_session_id && id < Ocannl_runtime.Node.global.unique_id then
          "get_root: Node " ^ Int.to_string id ^ " is a subformula"
        else if id >= Ocannl_runtime.Node.global.unique_id then
          "get_root: Node " ^ Int.to_string id ^ " has not been created yet"
        else if id < 1 then "get_root: Node IDs start from 1"
        else "get_root: Node " ^ Int.to_string id ^ " is outside the current session"
      in
      raise @@ Session_error (msg, None)

let get_node id =
  let open Ocannl_runtime.Node in
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
        Shape.axis_map_to_dims_index p_labels
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
    || Shape.(List.exists ~f:(( = ) 1) @@ list_of_dims @@ dims_of_kind Input sh)
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
     match (style, m.node.node.form) with
     | `Inline, Some cform ->
         NodeUI.pp_tensor_inline Caml.Format.std_formatter ~num_batch_axes ~num_input_axes ~num_output_axes
           ?labels_spec cform.grad;
         Caml.Format.print_newline ()
     | _, Some cform ->
         NodeUI.pp_tensor Caml.Format.std_formatter ~prefix:(prefix ^ " Gradient ") ~labels ~indices
           cform.grad;
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

let print_global_root ~with_grad ~with_code (style : NodeUI.array_print_style) root =
  let open Formula in
  print_formula ~with_grad ~with_code:false style root.formula;
  if with_code then (
    (match root.forward_code with
    | None -> ()
    | Some fwd -> Caml.Format.printf "Forward:@ %a@ " Code.fprint_program fwd);
    match root.backprop_code with
    | None -> ()
    | Some bwd -> Caml.Format.printf "Backprop:@ %a@ " Code.fprint_program bwd);
  Stdio.printf "\n%!"

let print_global_roots ~with_grad ~with_code (style : NodeUI.array_print_style) =
  let open Formula in
  List.iter (Map.to_alist ~key_order:`Increasing !global_roots) ~f:(fun (id, root) ->
      assert (id = root.formula.id);
      print_global_root ~with_grad ~with_code style root)

let print_preamble () = Stdio.printf "%s\n%!" (Formula.prefix_with_preamble "")

let print_session_code () =
  let open Code in
  Caml.Format.printf "Step preparation:@ %a" fprint_code (all_parallel !Formula.session_prepare_step);
  Caml.Format.print_newline ()

(** *** Session management. *** *)
type backend = Interpreter | OCaml | Gccjit [@@deriving sexp, equal]

let node_fetch_callbacks : (int, Code.fetch_op) Hashtbl.t = Hashtbl.create (module Int)
let executor = ref Exec_as_gccjit.jit_program
let executor_error_message = ref Exec_as_gccjit.error_message
let cleanup_executor_session = ref Exec_as_gccjit.cleanup_session

let set_executor = function
  | Interpreter ->
      executor := Code.interpret_program;
      executor_error_message := Code.interpreter_error_message;
      cleanup_executor_session := fun () -> ()
  | OCaml ->
      executor := Exec_as_OCaml.load_native;
      executor_error_message := Exec_as_OCaml.error_message;
      cleanup_executor_session := fun () -> ()
  | Gccjit ->
      executor := Exec_as_gccjit.jit_program;
      executor_error_message := Exec_as_gccjit.error_message;
      cleanup_executor_session := Exec_as_gccjit.cleanup_session

let dynload_with_handler ~runtime_store code =
  let name = Code.get_name code in
  let contents = !executor code in
  match (contents, !runtime_store) with
  | Some contents, Some routine ->
      runtime_store :=
        Some
          (fun () ->
            try routine ()
            with error ->
              Formula.handle_error @@ !executor_error_message ~name ~prefix:"Runtime error:" ~contents error)
  | Some contents, None ->
      let msg = "refresh_session: error loading initialization: routine not set in code:\n" ^ contents in
      raise @@ Formula.Session_error (msg, None)
  | _, None ->
      failwith
        ("refresh_session: error loading initialization: routine not set"
        ^ if !Code.with_debug then "" else " (set `Code.CDSL.with_debug := true` for more information)")
  | _ -> ()

let compile_and_run code =
  (* Since we use [Initialization], this is just to satisfy [dynload_with_handler]. *)
  let dummy = ref @@ Some (fun () -> ()) in
  dynload_with_handler ~runtime_store:dummy (Code.Initialization code)

let compile_routine code =
  let open Formula in
  let num_inits = List.length !session_initializations in
  let to_init = num_inits - !session_initialized in
  Code.interpret_initialization @@ List.take !session_initializations to_init;
  session_initialized := num_inits;
  Ocannl_runtime.Node.most_recent_suspension := None;
  dynload_with_handler ~runtime_store:Ocannl_runtime.Node.most_recent_suspension Code.(Suspension code);
  let routine = Option.value_exn !Ocannl_runtime.Node.most_recent_suspension in
  Ocannl_runtime.Node.most_recent_suspension := None;
  routine

let session_params () = NodeUI.param_nodes ~from_id:!Formula.first_session_id ()

let generate_params_update ~(minus_lr : Formula.t) ?params () =
  let params = match params with Some p -> p | None -> Hashtbl.data @@ session_params () in
  let module CDSL = Code.CDSL in
  let module NFDSL = Operation.NFDSL in
  compile_routine @@ Code.all_parallel
  @@ List.map params ~f:(fun n -> [%nn_cd n =+ minus_lr * n.grad ~logic:"."])

let minus_learning_rate : Formula.t option ref = ref None
let update_params = ref None

let refresh_session ?(regenerate = false) ?(reinit = false) ?(run = true) ?(force_no_init = false) () =
  let open Formula in
  if force_no_init && (regenerate || reinit || run) then
    invalid_arg "refresh_session: set other triggers to false when using force_no_init";
  let root_changed =
    Map.exists !global_roots ~f:(fun root ->
        Option.is_none root.forward_code || Option.is_none root.backprop_code)
  in
  (* Initialization and the forward processing. *)
  if regenerate || root_changed then List.iter !session_shape_updates ~f:Shape.propagate_shapes;
  if regenerate then session_initialized := 0;
  if (not force_no_init) && (regenerate || reinit || root_changed) then (
    let num_inits = List.length !session_initializations in
    let to_init = num_inits - !session_initialized in
    Code.interpret_initialization @@ List.take !session_initializations to_init;
    session_initialized := num_inits);
  if regenerate || root_changed then (
    for i = !Formula.first_session_id to Ocannl_runtime.Node.global.unique_id - 1 do
      Hashtbl.remove node_fetch_callbacks i
    done;
    Ocannl_runtime.Node.global.session_prepare_step := None;
    dynload_with_handler ~runtime_store:Ocannl_runtime.Node.global.session_prepare_step
      Code.(Session_prepare_step (all_parallel !session_prepare_step)));
  (if regenerate || Option.is_none !update_params then
     match !minus_learning_rate with
     | None -> ()
     | Some minus_lr -> update_params := Some (generate_params_update ~minus_lr ()));
  List.iter (Map.to_alist ~key_order:`Increasing !global_roots) ~f:(fun (_node_id, root) ->
      let m = root.formula in
      let cform = match m.node.node.form with Some form -> form | None -> assert false in
      if regenerate || Option.is_none root.forward_code || Option.is_none root.backprop_code then (
        let forward_prog, backprop_prog = get_toplevel m in
        root.forward_code <- Some forward_prog;
        cform.forward := None;
        root.backprop_code <- Some backprop_prog;
        cform.backprop := None);
      (if (not force_no_init) && (reinit || Option.is_none !(cform.forward)) then
         try
           cform.forward := None;
           dynload_with_handler ~runtime_store:cform.forward (Option.value_exn root.forward_code)
         with Session_error (msg, None) ->
           let msg = "Forward init error: " ^ msg in
           raise @@ Session_error (msg, Some m));
      if (not force_no_init) && (reinit || Option.is_none !(cform.backprop)) then (
        try
          cform.backprop := None;
          dynload_with_handler ~runtime_store:cform.backprop (Option.value_exn root.backprop_code)
        with Session_error (msg, None) ->
          Caml.Format.printf "Forward code (context for backprop init error):@ %a@\n" Code.fprint_program
          @@ Option.value_exn root.forward_code;
          Formula.handle_error ~formula:m @@ "Backprop init error: " ^ msg));
  if run then (
    (* Preparation. *)
    (match !(Ocannl_runtime.Node.global.session_prepare_step) with
    | None -> assert false
    | Some prepare -> prepare ());
    (* Forward propagation. *)
    List.iter (Map.to_alist ~key_order:`Increasing !global_roots) ~f:(fun (_node_id, root) ->
        match !((Option.value_exn root.formula.node.node.form).forward) with
        | Some forward -> forward ()
        | None -> assert false);
    (* The backpropagation. *)
    List.iter (Map.to_alist ~key_order:`Decreasing !global_roots) ~f:(fun (_node_id, root) ->
        let cform = match root.formula.node.node.form with Some form -> form | None -> assert false in
        Option.value_exn !(cform.backprop) ()))

(** Discards global roots, rolls back [Node.state.unique_id] to [Formula.first_session_id], discards
    the corresponding elements from [Node.state.node_store]. *)
let drop_session () =
  Formula.global_roots := Map.empty (module Int);
  Formula.session_shape_updates := [];
  Formula.session_initializations := [];
  Formula.session_initialized := 0;
  Formula.session_prepare_step := [];
  Hashtbl.clear node_fetch_callbacks;
  minus_learning_rate := None;
  update_params := None;
  !cleanup_executor_session ();
  for i = !Formula.first_session_id to Ocannl_runtime.Node.global.unique_id - 1 do
    Hashtbl.remove NodeUI.global_node_store i;
    Hashtbl.remove Ocannl_runtime.Node.global.node_store i;
    Hashtbl.remove node_fetch_callbacks i
  done;
  Ocannl_runtime.Node.most_recent_suspension := None;
  Ocannl_runtime.Node.global.session_prepare_step := None;
  Ocannl_runtime.Node.global.unique_id <- !Formula.first_session_id

(** Discards all global state, rolls back [Node.state.unique_id] and [Formula.first_session_id]
    to 1. *)
let drop_all_sessions () =
  Formula.global_roots := Map.empty (module Int);
  Formula.session_shape_updates := [];
  Formula.session_initializations := [];
  Formula.session_initialized := 0;
  Formula.session_prepare_step := [];
  Formula.first_session_id := 1;
  Hashtbl.clear node_fetch_callbacks;
  minus_learning_rate := None;
  update_params := None;
  !cleanup_executor_session ();
  Hashtbl.clear NodeUI.global_node_store;
  Hashtbl.clear Ocannl_runtime.Node.global.node_store;
  Hashtbl.clear node_fetch_callbacks;
  Ocannl_runtime.Node.most_recent_suspension := None;
  Ocannl_runtime.Node.global.session_prepare_step := None;
  Ocannl_runtime.Node.global.unique_id <- 1

(** Discards global roots, advances [Formula.first_session_id] to [Node.state.unique_id].
    Discards all computations (forward, backward, update params, data fetches), but keeps
    the already computed data / parameters. *)
let close_session () =
  Formula.first_session_id := Ocannl_runtime.Node.global.unique_id;
  Formula.global_roots := Map.empty (module Int);
  Formula.session_shape_updates := [];
  Formula.session_initializations := [];
  Formula.session_initialized := 0;
  Formula.session_prepare_step := [];
  Hashtbl.clear node_fetch_callbacks;
  minus_learning_rate := None;
  update_params := None;
  !cleanup_executor_session ();
  Ocannl_runtime.Node.most_recent_suspension := None;
  Ocannl_runtime.Node.global.session_prepare_step := None

module SDSL = struct
  type nonrec backend = backend = Interpreter | OCaml | Gccjit

  let set_executor = set_executor
  let refresh_session = refresh_session
  let drop_session = drop_session
  let drop_all_sessions = drop_all_sessions
  let close_session = close_session
  let compile_routine = compile_routine
  let session_params = session_params
  let minus_learning_rate = minus_learning_rate
  let update_params = update_params
  let print_global_root = print_global_root

  let print_node_tree ?entries_per_axis ?with_id ?with_value ~with_grad ~depth id =
    PrintBox_text.output Stdio.stdout
    @@ NodeUI.to_printbox ?entries_per_axis ?with_id ?with_value ~with_grad ~depth id

  let max_sublabel_length = Formula.max_sublabel_length
  let print_formula = print_formula
  let print_global_roots = print_global_roots
  let print_preamble = print_preamble
  let print_session_code = print_session_code
  let print_decimals_precision = NodeUI.print_decimals_precision
  let get_root = get_root
  let get_node = get_node

  let value_1d_points ?from_axis ~xdim m =
    NodeUI.retrieve_1d_points ?from_axis ~xdim m.Formula.node.node.value

  let value_2d_points ?from_axis ~xdim ~ydim m =
    NodeUI.retrieve_2d_points ?from_axis ~xdim ~ydim m.Formula.node.node.value

  let grad_1d_points ?from_axis ~xdim m =
    let form = m.Formula.node.node.form in
    match form with None -> [||] | Some f -> NodeUI.retrieve_1d_points ?from_axis ~xdim f.grad

  let grad_2d_points ?from_axis ~xdim ~ydim m =
    let form = m.Formula.node.node.form in
    match form with None -> [||] | Some f -> NodeUI.retrieve_2d_points ?from_axis ~xdim ~ydim f.grad
end
