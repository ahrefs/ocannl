(** Managing a computation session. *)

open Base
module Nd = Arrayjit.Ndarray
module LA = Arrayjit.Low_level.Lazy_array
module CDSL = Arrayjit.Low_level.CDSL
open Arrayjit

(** *** Printing. *** *)

let la_dims_to_string ?(with_axis_numbers = false) arr =
  let dims_s =
    if Lazy.is_val arr.LA.dims then Nd.int_dims_to_string ~with_axis_numbers @@ Lazy.force arr.dims
    else "<not-in-yet>"
  in
  Nd.prec_string arr.prec ^ " prec " ^ dims_s

(** Converts ID, label and the dimensions of a node to a string. *)
let tensor_header t =
  let v_dims_s = la_dims_to_string t.Tensor.value in
  let g_dims_s = match t.diff with None -> "<no-grad>" | Some diff -> la_dims_to_string diff.grad in
  let dims_s =
    if String.equal v_dims_s g_dims_s then "dims " ^ v_dims_s
    else "dims val " ^ v_dims_s ^ " grad " ^ g_dims_s
  in
  "#" ^ Int.to_string t.id ^ " " ^ t.value.label ^ " " ^ dims_s ^ " ["
  ^ String.concat ~sep:"," (List.map t.children ~f:(fun { subtensor = { id; _ }; _ } -> Int.to_string id))
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

let to_dag ?(single_node = false) ?entries_per_axis ~with_id ~with_value ~with_grad t =
  (* let la_to_box la = in *)
  let rec to_dag { Tensor.subtensor = t; embedded } : PrintBox_utils.dag =
    let id = Int.to_string t.id in
    let children = if single_node then [] else List.map ~f:to_dag t.children in
    let prefix = "[" ^ id ^ "] " ^ t.value.label in
    let labels = Shape.axis_map_to_dims_index t.shape.axis_labels in
    let indices = Shape.default_display_indices t.shape in
    let txt = if with_id then prefix else t.value.label in
    match (not embedded, with_value, with_grad, t.value.array, t.diff) with
    | true, _, _, _, _ -> `Embed_subtree_ID (Int.to_string t.id)
    | _, false, false, _, _ | _, false, true, _, None -> `Subtree_with_ID (id, `Tree (`Text txt, children))
    | _, true, false, (lazy (Some v_array)), _ | _, true, true, (lazy (Some v_array)), None ->
        let node = `Box (Nd.render_array ~brief:true ~prefix ?entries_per_axis ~labels ~indices @@ v_array) in
        `Subtree_with_ID (id, `Tree (node, children))
    | _, true, false, (lazy None), _ | _, true, true, (lazy None), None ->
        let node = `Text (txt ^ " <virtual>") in
        `Subtree_with_ID (id, `Tree (node, children))
    | _, false, true, _, Some diff ->
        let prefix = prefix ^ " Gradient" in
        let node =
          match Lazy.force diff.grad.array with
          | Some g_array ->
              `Box (Nd.render_array ~brief:true ~prefix ?entries_per_axis ~labels ~indices g_array)
          | None -> `Text (diff.grad.label ^ " <virtual>")
        in
        `Subtree_with_ID (id, `Tree (node, children))
    | _, true, true, (lazy (Some v_array)), Some diff ->
        let node =
          let value = Nd.render_array ~brief:true ~prefix ?entries_per_axis ~labels ~indices v_array in
          let grad =
            match Lazy.force diff.grad.array with
            | Some g_array ->
                `Box
                  (Nd.render_array ~brief:true ~prefix:"Gradient" ?entries_per_axis ~labels ~indices g_array)
            | None -> `Text (diff.grad.label ^ "Gradient <virtual>")
          in
          `Vlist (false, [ `Box value; grad ])
        in
        `Subtree_with_ID (id, `Tree (node, children))
    | _, true, true, (lazy None), Some diff ->
        let node =
          let value = `Text (prefix ^ " " ^ t.value.label ^ " <virtual>") in
          let grad =
            match Lazy.force diff.grad.array with
            | Some g_array ->
                `Box
                  (Nd.render_array ~brief:true ~prefix:"Gradient" ?entries_per_axis ~labels ~indices g_array)
            | None -> `Text (diff.grad.label ^ "Gradient <virtual>")
          in
          `Vlist (false, [ value; grad ])
        in
        `Subtree_with_ID (id, `Tree (node, children))
  in
  to_dag { subtensor = t; embedded = true }

let to_printbox ?single_node ?entries_per_axis ?(with_id = false) ?(with_value = true) ~with_grad ~depth t =
  to_dag ?single_node ?entries_per_axis ~with_id ~with_value ~with_grad t |> PrintBox_utils.reformat_dag depth

let print_tensor_preamble t =
  let prefix = tensor_header t in
  Caml.Format.printf "Tensor %s" prefix;
  Caml.Format.printf "\n%!"

let print_tensor ~with_grad ~with_code ?(with_low_level = false) (style : array_print_style) t =
  let open Tensor in
  let sh = t.shape in
  let label = t.value.label in
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
        Shape.axis_map_to_dims_index p_labels
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
    || Shape.(List.exists ~f:(( = ) 1) @@ list_of_dims @@ dims_of_kind Input sh)
  in
  let labels = Shape.axis_map_to_dims_index ~default:"" sh.axis_labels in
  let axes_spec = if needs_spec then Some (Shape.to_string_hum ~style:`Only_labels sh) else None in
  let num_axes kind = List.length Shape.(list_of_dims @@ dims_of_kind kind sh) in
  let num_batch_axes = num_axes Shape.AxisKey.Batch in
  let num_input_axes = num_axes Shape.AxisKey.Input in
  let num_output_axes = num_axes Shape.AxisKey.Output in
  (* TODO: code sharing with [to_dag] *)
  (if not @@ Lazy.is_val t.value.array then Caml.Format.printf "<not-in-yet>@ "
   else
     match (style, t.value.array) with
     | `Inline, (lazy None) -> Caml.Format.printf "<virtual>@ "
     | `Inline, (lazy (Some arr)) ->
         Nd.pp_array_inline Caml.Format.std_formatter ~num_batch_axes ~num_input_axes ~num_output_axes
           ?axes_spec arr
     | _, (lazy None) -> Caml.Format.printf "<virtual>@ "
     | _, (lazy (Some arr)) ->
         Nd.pp_array Caml.Format.std_formatter ~prefix ~labels ~indices arr;
         Caml.Format.print_newline ());
  if with_grad then
    Option.iter t.diff ~f:(fun diff ->
        if not @@ Lazy.is_val diff.grad.array then Caml.Format.printf "Gradient <not-in-yet>@ "
        else
          match (style, diff.grad.array) with
          | `Inline, (lazy (Some arr)) ->
              Nd.pp_array_inline Caml.Format.std_formatter ~num_batch_axes ~num_input_axes ~num_output_axes
                ?axes_spec arr;
              Caml.Format.print_newline ()
          | _, (lazy (Some arr)) ->
              Nd.pp_array Caml.Format.std_formatter ~prefix:(prefix ^ " Gradient ") ~labels ~indices arr;
              Caml.Format.print_newline ()
          | _, (lazy None) -> Caml.Format.printf "Gradient <virtual>@ ");
  if with_code then (
    (match t.forward_body with
    | Noop -> ()
    | fwd_code -> Caml.Format.printf "Current forward body:@ %a@ " High_level.fprint_code fwd_code);
    match t.diff with
    | Some { backprop_body = Noop; _ } -> ()
    | Some { backprop_body = bwd_code; _ } ->
        Caml.Format.printf "Current backprop body:@ %a@ " High_level.fprint_code bwd_code
    | None -> ());
  if with_low_level then (
    (match t.forward_body with
    | Noop -> ()
    | fwd_code ->
        Caml.Format.printf "Current forward low-level body:@ %a@ " Low_level.fprint_code
        @@ High_level.to_low_level fwd_code);
    match t.diff with
    | Some { backprop_body = Noop; _ } -> ()
    | Some { backprop_body = bwd_code; _ } ->
        Caml.Format.printf "Current backprop low-level body:@ %a@ " Low_level.fprint_code
        @@ High_level.to_low_level bwd_code
    | None -> ());
  Stdio.printf "\n%!"

let print_forward_roots ~with_grad ~with_code (style : array_print_style) =
  let open Tensor in
  List.iter (Map.to_alist ~key_order:`Increasing Tensor.session_state.forward_roots) ~f:(fun (id, root) ->
      assert (id = root.id);
      print_tensor ~with_grad ~with_code style root)

(** *** Session management. *** *)
type backend = Gccjit | Cuda [@@deriving sexp, equal]

let exec = ref Exec_as_gccjit.jit
let cleanup_executor_session = ref Exec_as_gccjit.cleanup_session

let set_executor = function
  | Gccjit ->
      exec := Exec_as_gccjit.jit;
      Low_level.virtualize_settings.sequential_minibatch <- true;
      cleanup_executor_session := Exec_as_gccjit.cleanup_session
  | Cuda ->
      exec := Exec_as_cuda.jit;
      Low_level.virtualize_settings.sequential_minibatch <- false;
      cleanup_executor_session := Exec_as_cuda.cleanup_session

let compile_routine ~name code =
  !exec ~name @@ Low_level.compile_proc ~name ~for_step_update:false @@ High_level.to_low_level code

(*
let minus_learning_rate : Tensor.t option ref = ref None
let last_refresh_roots = ref Tensor.session_state.forward_roots
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
  List.map params ~f:(fun v -> [%nn_cd t =+ minus_lr * t.grad ~logic:"."])

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
*)

(*
let save_tensors ~name =
  let out = Npy.Npz.open_out (name ^ ".npz") in
  Hashtbl.iter Low_level.global_node_store ~f:(fun v ->
      let save field arr = Npy.Npz.write out Node.(tensor_ptr_name { id = v.id; field }) arr in
      let f arr = save Value arr in
      Nd.map { f } v.node.value;
      let f arr = save Grad arr in
      Option.iter v.node.grad ~f:(Nd.map { f }))

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
              match Npy.to_bigarray Bigarray.c_layout (Nd.precision_to_bigarray_kind prec) src with
              | None -> if not partially then failwith ("Session.restore_tensors: missing tensor " ^ t_name)
              | Some src -> Nd.A.blit src dst
            in
            Nd.map_with_prec { f } arr
      in
      restore Value;
      restore Node.Grad)
*)
let value_1d_points ?from_axis ~xdim t =
  Option.value_map ~default:[||] ~f:(fun arr -> Nd.retrieve_1d_points ?from_axis ~xdim arr)
  @@ Lazy.force t.Tensor.value.array

let value_2d_points ?from_axis ~xdim ~ydim t =
  Option.value_map ~default:[||] ~f:(fun arr -> Nd.retrieve_2d_points ?from_axis ~xdim ~ydim arr)
  @@ Lazy.force t.Tensor.value.array

let grad_1d_points ?from_axis ~xdim t =
  match t.Tensor.diff with
  | None -> [||]
  | Some diff ->
      Option.value_map ~default:[||] ~f:(fun arr -> Nd.retrieve_1d_points ?from_axis ~xdim arr)
      @@ Lazy.force diff.grad.array

let grad_2d_points ?from_axis ~xdim ~ydim t =
  match t.Tensor.diff with
  | None -> [||]
  | Some diff ->
      Option.value_map ~default:[||] ~f:(fun arr -> Nd.retrieve_2d_points ?from_axis ~xdim ~ydim arr)
      @@ Lazy.force diff.grad.array

let set_value t = Nd.set_from_float @@ Option.value_exn @@ Lazy.force t.Tensor.value.array
let get_value t = Nd.get_as_float @@ Option.value_exn @@ Lazy.force t.Tensor.value.array

module O = struct
  (** Get the value at the given indices. *)
  let ( .@{} ) = get_value

  (** Set the value at the given indices. *)
  let ( .@{}<- ) = set_value

  (** Get the value at the given index from a single-axis shape tensor. *)
  let ( .@[] ) t indx = get_value t [| indx |]

  (** Set the value at the given index for a single-axis shape tensor. *)
  let ( .@[]<- ) t indx = set_value t [| indx |]
end

module SDSL = struct
  type nonrec backend = backend = Gccjit | Cuda

  module O = O

  let set_executor = set_executor

  (*
  let refresh_session = refresh_session
  let drop_session = drop_session
  let drop_all_sessions = drop_all_sessions
  let close_session = close_session
  let session_params = session_params
  let minus_learning_rate = minus_learning_rate
  *)

  let compile_routine = compile_routine

  let print_tree ?entries_per_axis ?(with_backend_info = false) ?(with_id = true) ?(with_value = true)
      ~with_grad ~depth t =
    (* FIXME: print backend info *)
    ignore with_backend_info;
    PrintBox_text.output Stdio.stdout @@ PrintBox_utils.dag_to_box @@ PrintBox_utils.boxify depth
    @@ to_dag ?entries_per_axis ~with_id ~with_value ~with_grad t

  let max_sublabel_length = Tensor.max_sublabel_length
  let print_tensor = print_tensor
  let print_forward_roots = print_forward_roots
  let print_tensor_preamble = print_tensor_preamble
  let print_decimals_precision = Nd.print_decimals_precision
  let set_values t cs = Nd.(init (Constant_fill cs) @@ Option.value_exn @@ Lazy.force t.Tensor.value.array)

  let set_fully_on_host t =
    t.Tensor.value.never_virtual <- true;
    t.Tensor.value.never_device_only <- true;
    Option.iter t.diff ~f:(fun diff ->
        diff.grad.never_virtual <- true;
        diff.grad.never_device_only <- true)

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
end
