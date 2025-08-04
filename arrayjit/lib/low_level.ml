open Base
module Lazy = Utils.Lazy
module Nd = Ndarray
module Tn = Tnode

let _get_local_debug_runtime = Utils.get_local_debug_runtime

[%%global_debug_log_level 9]
[%%global_debug_log_level_from_env_var "OCANNL_LOG_LEVEL"]

module Scope_id = struct
  type t = { tn : Tn.t; scope_id : int } [@@deriving sexp_of, equal, hash, compare]

  include Comparator.Make (struct
    type nonrec t = t

    let compare = compare
    let sexp_of_t = sexp_of_t
  end)
end

type scope_id = Scope_id.t = { tn : Tn.t; scope_id : int }
[@@deriving sexp_of, equal, hash, compare]

let get_scope =
  let uid = ref 0 in
  fun tn ->
    Int.incr uid;
    { tn; scope_id = !uid }

type t =
  | Noop
  | Comment of string
  | Staged_compilation of ((unit -> PPrint.document)[@equal.ignore] [@compare.ignore])
  | Seq of t * t
  | For_loop of { index : Indexing.symbol; from_ : int; to_ : int; body : t; trace_it : bool }
  | Zero_out of Tn.t
  | Set of { tn : Tn.t; idcs : Indexing.axis_index array; llsc : scalar_t; mutable debug : string }
  | Set_from_vec of {
      tn : Tn.t;
      idcs : Indexing.axis_index array;
      length : int;
      vec_unop : Ops.vec_unop;
      arg : scalar_t;
      mutable debug : string;
    }
  | Set_local of scope_id * scalar_t
[@@deriving sexp_of, equal]

and scalar_t =
  | Local_scope of { id : scope_id; body : t; orig_indices : Indexing.axis_index array }
  | Get_local of scope_id
  | Get of Tn.t * Indexing.axis_index array
  | Get_merge_buffer of Tn.t * Indexing.axis_index array
  | Ternop of Ops.ternop * scalar_t * scalar_t * scalar_t
  | Binop of Ops.binop * scalar_t * scalar_t
  | Unop of Ops.unop * scalar_t
  | Constant of float
  | Embed_index of Indexing.axis_index
[@@deriving sexp_of, equal, compare]

let apply_op op args =
  match (op, args) with
  | Ops.Binop Ops.Arg1, [| rhs1; _ |] -> rhs1
  | Binop Arg2, [| _; rhs2 |] -> rhs2
  | Unop Identity, [| rhs |] -> rhs
  | Ternop op, [| rhs1; rhs2; rhs3 |] -> Ternop (op, rhs1, rhs2, rhs3)
  | Binop op, [| rhs1; rhs2 |] -> Binop (op, rhs1, rhs2)
  | Unop op, [| rhs |] -> Unop (op, rhs)
  | _ -> invalid_arg "Low_level.op: invalid number of arguments"

let rec flat_lines ts =
  List.concat_map ts ~f:(function Seq (t1, t2) -> flat_lines [ t1; t2 ] | t -> [ t ])

let rec unflat_lines = function
  | [] -> Noop
  | [ llc ] -> llc
  | Noop :: tl -> unflat_lines tl
  | llc :: tl -> Seq (llc, unflat_lines tl)

type virtualize_settings = {
  mutable enable_device_only : bool;
  mutable max_visits : int;
  mutable max_tracing_dim : int;
  mutable inline_scalar_constexprs : bool;
  mutable inline_simple_computations : bool;
}

let virtualize_settings =
  let max_visits =
    Int.of_string @@ Utils.get_global_arg ~arg_name:"virtualize_max_visits" ~default:"1"
  in
  let max_tracing_dim =
    Int.of_string @@ Utils.get_global_arg ~arg_name:"virtualize_max_tracing_dim" ~default:"5"
  in
  let enable_device_only = Utils.get_global_flag ~default:true ~arg_name:"enable_device_only" in
  let inline_scalar_constexprs =
    Utils.get_global_flag ~default:true ~arg_name:"inline_scalar_constexprs"
  in
  let inline_simple_computations =
    Utils.get_global_flag ~default:true ~arg_name:"inline_simple_computations"
  in
  {
    enable_device_only;
    max_visits;
    max_tracing_dim;
    inline_scalar_constexprs;
    inline_simple_computations;
  }

type visits = Visits of int | Recurrent [@@deriving sexp, equal, variants]

type traced_array = {
  tn : Tn.t;
  assignments : int array Hash_set.t;
  accesses : (int array, visits) Hashtbl.t;
  mutable zero_initialized : bool;
  mutable zeroed_out : bool;
  mutable read_before_write : bool;
  mutable read_only : bool;
  mutable is_scalar_constexpr : bool;
  mutable is_complex : bool;
}
[@@deriving sexp_of]

type optimize_ctx = {
  computations : (Tnode.t, (Indexing.axis_index array option * t) list) Base.Hashtbl.t;
}
[@@deriving sexp_of]

type traced_store = (Tn.t, traced_array) Base.Hashtbl.t [@@deriving sexp_of]

type optimized = {
  traced_store : traced_store;
  optimize_ctx : optimize_ctx;
  llc : t;
  merge_node : Tnode.t option;
}
[@@deriving sexp_of]

let get_node store tn =
  Hashtbl.find_or_add store tn ~default:(fun () ->
      {
        tn;
        assignments = Hash_set.Poly.create ();
        accesses = Hashtbl.Poly.create ();
        zero_initialized = false;
        zeroed_out = false;
        read_before_write = false;
        read_only = false;
        is_scalar_constexpr = false;
        is_complex = false;
      })

let visit ~is_assigned old =
  if not is_assigned then Recurrent
  else
    match old with
    | None -> Visits 1
    | Some (Visits i) -> Visits (i + 1)
    | Some Recurrent -> Recurrent

let is_constexpr_comp traced_store llsc =
  let rec loop llsc =
    match llsc with
    | Get_local { tn; _ } | Local_scope { id = { tn; _ }; _ } ->
        let traced = get_node traced_store tn in
        traced.is_scalar_constexpr
    | Get (tn, _) ->
        let traced = get_node traced_store tn in
        traced.is_scalar_constexpr
    | Get_merge_buffer (tn, _) ->
        let traced = get_node traced_store tn in
        traced.is_scalar_constexpr
    | Ternop (_, v1, v2, v3) -> loop v1 && loop v2 && loop v3
    | Binop (_, v1, v2) -> loop v1 && loop v2
    | Unop (_, v) -> loop v
    | Constant _ -> true
    | Embed_index _ -> false
  in
  loop llsc

let is_complex_comp traced_store llsc =
  let rec loop llsc =
    match llsc with
    | Get_local { tn; _ } | Local_scope { id = { tn; _ }; _ } ->
        let traced = get_node traced_store tn in
        traced.is_complex
    | Get (tn, _) ->
        let traced = get_node traced_store tn in
        not traced.is_scalar_constexpr
    | Get_merge_buffer (tn, _) ->
        let traced = get_node traced_store tn in
        not traced.is_scalar_constexpr
    | Ternop (_, v1, v2, v3) -> loop v1 || loop v2 || loop v3
    | Binop (_, v1, v2) -> loop v1 || loop v2
    | Unop (_, v) -> loop v
    | Constant _ -> false
    | Embed_index _ -> false
  in
  loop llsc

let is_scalar_dims tn = Array.for_all ~f:(( = ) 1) @@ Lazy.force tn.Tn.dims

let visit_llc traced_store ~merge_node_id reverse_node_map ~max_visits llc =
  let is_too_many = function Visits i -> i > max_visits | Recurrent -> true in
  (* FIXME: migrate hashtable to use offsets instead of indices *)
  let lookup env indices =
    Array.map indices ~f:(function
      | Indexing.Fixed_idx i -> i
      | Indexing.Sub_axis -> 0
      | Iterator s -> Option.value ~default:(* static index *) 0 @@ Map.find env s
      | Indexing.Affine { symbols; offset } ->
          List.fold symbols ~init:offset ~f:(fun acc (coeff, s) ->
              acc + (coeff * (Option.value ~default:0 @@ Map.find env s))))
  in
  let rec loop_proc ~first_visit env llc =
    let loop = loop_proc ~first_visit env in
    match llc with
    | Noop -> ()
    | (Seq (c1, c2) : t) ->
        loop c1;
        loop c2
    | For_loop { index; from_; to_ = _; body; trace_it = false } ->
        loop_proc ~first_visit (Map.add_exn ~key:index ~data:from_ env) body
    | For_loop { index; from_; to_; body; trace_it = true } ->
        for data = from_ to min to_ (from_ + virtualize_settings.max_tracing_dim) do
          loop_proc
            ~first_visit:(first_visit && data = from_)
            (Map.add_exn ~key:index ~data env)
            body
        done
    | Zero_out tn ->
        let traced : traced_array = get_node traced_store tn in
        if Hash_set.is_empty traced.assignments && Hashtbl.is_empty traced.accesses then (
          traced.zero_initialized <- true;
          traced.is_complex <- false;
          if is_scalar_dims tn then traced.is_scalar_constexpr <- true);
        traced.zeroed_out <- true
    | Set { tn; idcs; llsc; debug = _ } ->
        loop_float env llsc;
        let traced : traced_array = get_node traced_store tn in
        if
          Hash_set.is_empty traced.assignments
          && Hashtbl.is_empty traced.accesses && is_scalar_dims tn
        then traced.is_scalar_constexpr <- is_constexpr_comp traced_store llsc
          (* Note: this prevents detection if the same constant is assigned inside a loop. *)
        else if not @@ Hash_set.is_empty traced.assignments then traced.is_scalar_constexpr <- false;
        if first_visit then
          traced.is_complex <- traced.is_complex || is_complex_comp traced_store llsc;
        Hash_set.add traced.assignments (lookup env idcs);
        Array.iter idcs ~f:(function
          | Fixed_idx _ -> ()
          | Sub_axis -> ()
          | Iterator s ->
              let old_tn = Hashtbl.find_or_add reverse_node_map s ~default:(fun () -> tn) in
              (* TODO(#134): this prevents multiple virtual arrays from sharing for loops. *)
              assert (Tn.equal old_tn tn)
          | Indexing.Affine { symbols; _ } ->
              List.iter symbols ~f:(fun (_, s) ->
                  let old_tn = Hashtbl.find_or_add reverse_node_map s ~default:(fun () -> tn) in
                  assert (Tn.equal old_tn tn)))
    | Set_from_vec { tn; idcs; length; vec_unop = _; arg; debug = _ } ->
        loop_float env arg;
        let traced : traced_array = get_node traced_store tn in
        (* Vector operations cannot be scalar constexpr *)
        traced.is_scalar_constexpr <- false;
        if first_visit then traced.is_complex <- false;
        (* Mark all positions that will be written to *)
        for i = 0 to length - 1 do
          let pos_idcs = Array.copy idcs in
          (match pos_idcs.(Array.length pos_idcs - 1) with
          | Fixed_idx idx -> pos_idcs.(Array.length pos_idcs - 1) <- Fixed_idx (idx + i)
          | _ ->
              (* For non-Fixed_idx, we need to increment through the dimension *)
              let dims = Tn.dims_without_padding tn in
              let base_pos = lookup env idcs in
              (* Compute the flat position from base_pos *)
              let flat_pos = ref 0 in
              let stride = ref 1 in
              for j = Array.length base_pos - 1 downto 0 do
                flat_pos := !flat_pos + (base_pos.(j) * !stride);
                stride := !stride * dims.(j)
              done;
              flat_pos := !flat_pos + i;
              (* Convert back to multi-dimensional indices *)
              for j = Array.length pos_idcs - 1 downto 0 do
                pos_idcs.(j) <- Fixed_idx (!flat_pos % dims.(j));
                flat_pos := !flat_pos / dims.(j)
              done);
          Hash_set.add traced.assignments (lookup env pos_idcs)
        done;
        Array.iter idcs ~f:(function
          | Fixed_idx _ -> ()
          | Sub_axis -> ()
          | Iterator s ->
              let old_tn = Hashtbl.find_or_add reverse_node_map s ~default:(fun () -> tn) in
              assert (Tn.equal old_tn tn)
          | Indexing.Affine { symbols; _ } ->
              List.iter symbols ~f:(fun (_, s) ->
                  let old_tn = Hashtbl.find_or_add reverse_node_map s ~default:(fun () -> tn) in
                  assert (Tn.equal old_tn tn)))
    | Set_local (_, llsc) -> loop_float env llsc
    | Comment _ -> ()
    | Staged_compilation _ -> ()
  and loop_float env llsc =
    let loop = loop_float env in
    match llsc with
    | Constant _ -> ()
    | Get (ptr, indices) ->
        let traced : traced_array = get_node traced_store ptr in
        let at_pos = lookup env indices in
        Hashtbl.update traced.accesses at_pos
          ~f:(visit ~is_assigned:(traced.zeroed_out || Hash_set.mem traced.assignments at_pos))
    | Local_scope { body; _ } -> loop_proc ~first_visit:true env body
    | Get_local _ -> ()
    | Get_merge_buffer (source, _) ->
        let source_node_id = source.Tn.id in
        Option.iter !merge_node_id ~f:(fun merge_node_id ->
            if merge_node_id <> source_node_id then
              raise
              @@ Utils.User_error
                   [%string
                     "Low_evel.optimize_proc: currently only one merge buffer per routine is \
                      allowed, found node ids %{source_node_id#Int} and %{merge_node_id#Int}"]);
        merge_node_id := Some source_node_id
    | Embed_index _ -> ()
    | Binop (Arg1, llv1, _llv2) -> loop llv1
    | Binop (Arg2, _llv1, llv2) -> loop llv2
    | Ternop (_, llv1, llv2, llv3) ->
        loop llv1;
        loop llv2;
        loop llv3
    | Binop (_, llv1, llv2) ->
        loop llv1;
        loop llv2
    | Unop (_, llsc) -> loop llsc
  in
  loop_proc ~first_visit:true Indexing.empty_env llc;
  Hashtbl.iter traced_store ~f:(fun traced ->
      let tn = traced.tn in
      if
        virtualize_settings.inline_scalar_constexprs && traced.is_scalar_constexpr
        && not (Tn.known_non_virtual tn)
      then Tn.update_memory_mode tn Virtual 40;
      let skip_simple =
        virtualize_settings.inline_simple_computations && (not traced.is_complex)
        && not (Tn.known_non_virtual tn)
      in
      if
        (not skip_simple) && Option.is_none tn.memory_mode
        && Hashtbl.exists traced.accesses ~f:is_too_many
      then Tn.update_memory_mode tn Never_virtual 1;
      if (not traced.zeroed_out) && Hash_set.is_empty traced.assignments then (
        (* The tensor node is read-only/recurrent for this computation, but maybe computed or
           specified as virtual by another routine. However, if the memory mode is unspecified, we
           assume this will be the first computation involving the tensor node. *)
        traced.read_only <- true;
        if Tn.mode_is_unspecified tn then Tn.update_memory_mode tn (Hosted Unset_hosted) 37
        else if Tn.known_not_materialized tn then (
          if Tn.known_non_virtual tn then
            raise
              (Utils.User_error
                 [%string
                   "Mark %{Tn.debug_name tn} as materialized (e.g. via Train.set_materialized) \
                    before the first routine using it gets compiled; another routine re-uses that \
                    computation. Debug: %{Tn.debug_memory_mode tn.Tn.memory_mode}"]))
        else if Tn.known_non_virtual tn then Tn.update_memory_mode tn Materialized 35);
      (* We allow sharing virtual nodes across routines. *)
      if Hashtbl.exists traced.accesses ~f:is_recurrent && not (Tn.known_virtual tn) then (
        traced.read_before_write <- true;
        if Tn.mode_is_unspecified tn then
          Tn.update_memory_mode tn (Hosted (Changed_on_devices Unset)) 38
        else Tn.update_memory_mode tn Materialized 36))

let%diagn2_sexp check_and_store_virtual computations_table traced static_indices top_llc =
  let exception Non_virtual of int in
  let static_indices =
    Set.of_list (module Indexing.Symbol)
    @@ List.map ~f:(fun s -> s.Indexing.static_symbol) static_indices
  in
  let at_idcs = ref None in
  let has_setter = ref false in
  let top_tn = traced.tn in
  let check_idcs indices =
    (match !at_idcs with
    | None -> at_idcs := Some indices
    | Some at ->
        if not @@ [%equal: Indexing.axis_index array] at indices then raise @@ Non_virtual 4);
    (* TODO(#133): For non-recursive accesses, non-linearity is not supported yet. *)
    let syms =
      Set.of_array (module Indexing.Symbol)
      @@ Array.filter_map indices
           ~f:
             Indexing.(
               function
               | Fixed_idx _ -> None
               | Sub_axis -> None
               | Iterator s -> Option.some_if (not @@ Set.mem static_indices s) s
               | Affine { symbols; offset = _ } -> (
                   (* For affine indices, collect all symbols that are not static *)
                   List.filter_map symbols ~f:(fun (_, s) ->
                       Option.some_if (not @@ Set.mem static_indices s) s)
                   |> function
                   | [] -> None
                   | [ s ] -> Some s
                   | _ -> failwith "check_idcs: multiple non-static symbols in affine index"))
    in
    let num_syms =
      Array.count indices ~f:(function Iterator s -> not @@ Set.mem static_indices s | _ -> false)
    in
    if Set.length syms <> num_syms then raise @@ Non_virtual 5
  in
  (* Traverse the float code too, for completeness / future use-cases. *)
  let rec loop_proc ~env_dom llc =
    let loop = loop_proc ~env_dom in
    match llc with
    | Noop -> ()
    | (Seq (c1, c2) : t) ->
        loop c1;
        loop c2
    | For_loop { trace_it = false; _ } -> raise @@ Non_virtual 6
    | For_loop { index; body; from_ = _; to_ = _; trace_it = true } ->
        loop_proc ~env_dom:(Set.add env_dom index) body
    | Zero_out tn -> if Tn.equal tn top_tn then has_setter := true
    | Set { tn; idcs; llsc; debug = _ } ->
        if Tn.equal tn top_tn then (
          check_idcs idcs;
          has_setter := true)
        else
          (* Check for escaping variables. *)
          Array.iter idcs ~f:(function
            | Iterator s as _idx when not (Set.mem static_indices s) ->
                if not @@ Set.mem env_dom s then
                  [%log2
                    "INFO: Inlining candidate has an escaping variable",
                    (_idx : Indexing.axis_index),
                    (top_llc : t)];
                raise @@ Non_virtual 7
            | _ -> ());
        loop_float ~env_dom llsc
    | Set_from_vec { tn; idcs; length = _; vec_unop = _; arg; debug = _ } ->
        if Tn.equal tn top_tn then (
          check_idcs idcs;
          has_setter := true)
        else
          (* Check for escaping variables. *)
          Array.iter idcs ~f:(function
            | Iterator s as _idx when not (Set.mem static_indices s) ->
                if not @@ Set.mem env_dom s then
                  [%log2
                    "INFO: Inlining candidate has an escaping variable",
                    (_idx : Indexing.axis_index),
                    (top_llc : t)];
                raise @@ Non_virtual 7
            | _ -> ());
        loop_float ~env_dom arg
    | Set_local (_, llsc) -> loop_float ~env_dom llsc
    | Comment _ -> ()
    | Staged_compilation _ -> raise @@ Non_virtual 8
  and loop_float ~env_dom llsc =
    match llsc with
    | Constant _ -> ()
    | Get (tn, idcs) ->
        if Tn.equal tn top_tn then check_idcs idcs
        else
          (* Check for escaping variables. *)
          Array.iter idcs ~f:(function
            | Iterator s when not (Set.mem static_indices s) ->
                if not @@ Set.mem env_dom s then (
                  [%log2
                    "Inlining candidate has an escaping variable",
                    (s : Indexing.symbol),
                    (top_llc : t)];
                  raise @@ Non_virtual 9)
            | _ -> ())
    | Local_scope { body; _ } -> loop_proc ~env_dom body
    | Get_local _ -> ()
    | Get_merge_buffer (tn, idcs) ->
        if Tn.equal tn top_tn then check_idcs idcs
        else
          (* Check for escaping variables. *)
          Array.iter idcs ~f:(function
            | Iterator s when not (Set.mem static_indices s) ->
                if not @@ Set.mem env_dom s then (
                  [%log2
                    "Inlining candidate has an escaping variable",
                    (s : Indexing.symbol),
                    (top_llc : t)];
                  raise @@ Non_virtual 9)
            | _ -> ())
    | Embed_index (Fixed_idx _ | Sub_axis) -> ()
    | Embed_index (Iterator s) ->
        if not @@ Set.mem env_dom s then (
          if not (Set.mem static_indices s) then
            [%log2
              "Inlining candidate has an escaping variable", (s : Indexing.symbol), (top_llc : t)];
          raise @@ Non_virtual 10)
    | Embed_index (Affine { symbols; _ }) ->
        List.iter symbols ~f:(fun (_, s) ->
            if not @@ Set.mem env_dom s then (
              if not (Set.mem static_indices s) then
                [%log2
                  "Inlining candidate has an escaping variable",
                  (s : Indexing.symbol),
                  (top_llc : t)];
              raise @@ Non_virtual 10))
    | Ternop (_, llv1, llv2, llv3) ->
        loop_float ~env_dom llv1;
        loop_float ~env_dom llv2;
        loop_float ~env_dom llv3
    | Binop (_, llv1, llv2) ->
        loop_float ~env_dom llv1;
        loop_float ~env_dom llv2
    | Unop (_, llsc) -> loop_float ~env_dom llsc
  in
  try
    if Tn.known_non_virtual traced.tn then raise @@ Non_virtual 11;
    loop_proc ~env_dom:static_indices top_llc;
    if not !has_setter then raise @@ Non_virtual 12;
    let current_computations =
      Hashtbl.find computations_table traced.tn |> Option.value ~default:[]
    in
    Hashtbl.set computations_table ~key:traced.tn ~data:((!at_idcs, top_llc) :: current_computations)
  with Non_virtual i -> Tn.update_memory_mode traced.tn Never_virtual i

let%track7_sexp inline_computation ~id
    (computations_table : (Tn.t, (Indexing.axis_index array option * t) list) Hashtbl.t)
    (traced : traced_array) (static_indices : Indexing.static_symbol list)
    (call_args : Indexing.axis_index array) : t option =
  let exception Non_virtual of int in
  let static_indices =
    Set.of_list (module Indexing.Symbol)
    @@ List.map ~f:(fun s -> s.Indexing.static_symbol) static_indices
  in
  let make_subst (i : int) (lhs_ind : Indexing.axis_index) =
    if i >= Array.length call_args then
      failwith
        [%string
          "make_subst: call_args too short, maybe stale optimization context? Tnode: \
           %{Tn.debug_name traced.tn} #%{traced.tn.Tn.id#Int} i: %{i#Int}"]
    else
      let rhs_ind = call_args.(i) in
      match lhs_ind with
      | Indexing.Iterator lhs_s when not (Set.mem static_indices lhs_s) -> Some (lhs_s, rhs_ind)
      | _ when Indexing.equal_axis_index lhs_ind rhs_ind -> None
      | _ -> raise @@ Non_virtual 13
  in
  (* In the order of computation. *)
  let loop_proc ((def_args : Indexing.axis_index array option), (def : t)) : t option =
    let env =
      match def_args with
      | None -> Map.empty (module Indexing.Symbol)
      | Some def_args ->
          Map.of_alist_exn (module Indexing.Symbol)
          @@ Array.to_list
          @@ Array.filter_mapi def_args ~f:make_subst
    in
    let subst env (idx : Indexing.axis_index) : Indexing.axis_index =
      match idx with
      | Indexing.Iterator s when Map.mem env s -> Map.find_exn env s
      | Indexing.Affine { symbols; offset } ->
          (* We need to substitute each symbol in the affine expression. If a symbol maps to a
             non-Iterator, we need to handle it specially. *)
          let expand_symbol (coeff, s) =
            match Map.find env s with
            | Some (Indexing.Iterator new_s) -> [ (coeff, new_s) ]
            | Some (Indexing.Fixed_idx _ | Indexing.Sub_axis) ->
                [] (* Fixed index contributes to offset *)
            | Some (Indexing.Affine { symbols = inner_symbols; offset = _ }) ->
                (* Expand nested affine: coeff * (inner_symbols + inner_offset) *)
                List.map inner_symbols ~f:(fun (inner_coeff, inner_s) ->
                    (coeff * inner_coeff, inner_s))
            | None -> [ (coeff, s) ]
          in
          let all_terms = List.concat_map symbols ~f:expand_symbol in
          (* Calculate the new offset by adding contributions from Fixed_idx substitutions *)
          let offset_additions =
            List.fold symbols ~init:0 ~f:(fun acc (coeff, s) ->
                match Map.find env s with
                | Some (Indexing.Fixed_idx i) -> acc + (coeff * i)
                | Some (Indexing.Affine { offset = inner_offset; _ }) -> acc + (coeff * inner_offset)
                | _ -> acc)
          in
          let new_offset = offset + offset_additions in
          Indexing.Affine { symbols = all_terms; offset = new_offset }
      | idx -> idx
    in
    let rec loop env llc : t option =
      match llc with
      | Noop -> None
      | Seq _ ->
          let body = List.filter_map ~f:(loop env) @@ flat_lines [ llc ] in
          if List.is_empty body then None else Some (unflat_lines body)
      | For_loop { trace_it = false; _ } -> assert false
      | For_loop { index; body; _ } when Map.mem env index -> loop env body
      | For_loop { index; from_; to_; body; trace_it } ->
          (* Freshen the binding. *)
          let fresh = Indexing.get_symbol () in
          let env = Map.add_exn ~key:index ~data:(Indexing.Iterator fresh) env in
          Option.map ~f:(fun body : t -> For_loop { index = fresh; from_; to_; body; trace_it })
          @@ loop env body
      | Zero_out tn when Tn.equal tn traced.tn -> Some (Set_local (id, Constant 0.0))
      | Set { tn; idcs; llsc; debug = _ } when Tn.equal tn traced.tn ->
          assert ([%equal: Indexing.axis_index array option] (Some idcs) def_args);
          Some (Set_local (id, loop_float env llsc))
      | Set_from_vec { tn; idcs; length = _; vec_unop = _; arg = _; debug = _ }
        when Tn.equal tn traced.tn ->
          assert ([%equal: Indexing.axis_index array option] (Some idcs) def_args);
          (* For vector operations, we cannot inline them as scalar operations *)
          raise @@ Non_virtual 140
      | Zero_out _ -> None
      | Set _ -> None
      | Set_from_vec _ -> None
      | Set_local (id, llsc) -> Some (Set_local (id, loop_float env llsc))
      | Comment _ -> Some llc
      | Staged_compilation _ -> Some llc
    and loop_float env llsc : scalar_t =
      match llsc with
      | Constant _ -> llsc
      | Get (tn, indices) when Tn.equal tn traced.tn ->
          assert ([%equal: Indexing.axis_index array option] (Some indices) def_args);
          Get_local id
      | Get (tn, indices) -> Get (tn, Array.map ~f:(subst env) indices)
      | Local_scope { id; body; orig_indices } ->
          Local_scope
            {
              id;
              body = Option.value_exn ~here:[%here] @@ loop env body;
              orig_indices = Array.map ~f:(subst env) orig_indices;
            }
      | Get_local _ -> llsc
      | Get_merge_buffer (tn, indices) -> Get_merge_buffer (tn, Array.map ~f:(subst env) indices)
      | Embed_index idx -> Embed_index (subst env idx)
      | Ternop (op, llv1, llv2, llv3) ->
          Ternop (op, loop_float env llv1, loop_float env llv2, loop_float env llv3)
      | Binop (op, llv1, llv2) -> Binop (op, loop_float env llv1, loop_float env llv2)
      | Unop (op, llsc) -> Unop (op, loop_float env llsc)
    in
    loop env def
  in
  try
    let computations =
      Hashtbl.find computations_table traced.tn
      |> Option.value_or_thunk ~default:(fun () ->
             raise
             @@ Utils.User_error
                  [%string
                    "Stale optimize_ctx: No computations found for #%{traced.tn.Tn.id#Int}: \
                     %{Tn.debug_name traced.tn}"])
    in
    let body = List.rev_filter_map ~f:loop_proc computations in
    if List.is_empty body then raise @@ Non_virtual 14 else Some (unflat_lines body)
  with Non_virtual i ->
    Tn.update_memory_mode traced.tn Never_virtual i;
    None

let optimize_integer_pow = ref true

let rec unroll_pow ~(base : scalar_t) ~(exp : int) : scalar_t =
  if exp < 0 then unroll_pow ~base:(Binop (Div, Constant 1., base)) ~exp:(Int.neg exp)
  else if exp = 0 then Constant 1.
  else Fn.apply_n_times ~n:(exp - 1) (fun accu -> Binop (Mul, base, accu)) base

let virtual_llc computations_table traced_store reverse_node_map static_indices (llc : t) : t =
  (* The current position is within scope of the definitions of the process_for virtual arrays. *)
  let rec loop_proc ~process_for (llc : t) : t =
    let loop = loop_proc ~process_for in
    match llc with
    | Noop -> Noop
    | Seq (c1, c2) ->
        let c1 = loop c1 in
        let c2 = loop c2 in
        Seq (c1, c2)
    | For_loop ({ index; body; _ } as for_config) -> (
        match Hashtbl.find reverse_node_map index with
        | Some tn when not @@ Set.mem process_for tn ->
            let node : traced_array = get_node traced_store tn in
            let result = loop_proc ~process_for:(Set.add process_for tn) llc in
            if not @@ Tn.known_non_virtual node.tn then
              check_and_store_virtual computations_table node static_indices result;
            result
        | _ -> For_loop { for_config with body = loop body })
    | Zero_out tn ->
        let traced : traced_array = get_node traced_store tn in
        if (not @@ Set.mem process_for tn) && (not @@ Tn.known_non_virtual traced.tn) then
          check_and_store_virtual computations_table traced static_indices llc;
        llc
    | Set { tn; idcs; llsc; debug } ->
        let traced : traced_array = get_node traced_store tn in
        let next = if Tn.known_non_virtual traced.tn then process_for else Set.add process_for tn in
        let result = Set { tn; idcs; llsc = loop_float ~process_for:next llsc; debug } in
        if (not @@ Set.mem process_for tn) && (not @@ Tn.known_non_virtual traced.tn) then
          check_and_store_virtual computations_table traced static_indices result;
        result
    | Set_from_vec { tn; idcs; length; vec_unop; arg; debug } ->
        let traced : traced_array = get_node traced_store tn in
        let next = if Tn.known_non_virtual traced.tn then process_for else Set.add process_for tn in
        let result =
          Set_from_vec { tn; idcs; length; vec_unop; arg = loop_float ~process_for:next arg; debug }
        in
        if (not @@ Set.mem process_for tn) && (not @@ Tn.known_non_virtual traced.tn) then
          check_and_store_virtual computations_table traced static_indices result;
        result
    | Set_local (id, llsc) -> Set_local (id, loop_float ~process_for llsc)
    | Comment _ -> llc
    | Staged_compilation _ -> llc
  and loop_float ~process_for (llsc : scalar_t) : scalar_t =
    match llsc with
    | Constant _ -> llsc
    | Get (tn, _) when Set.mem process_for tn ->
        (* [Get_local] will replace this [Get] during [inline_computation] if [tn] remains
           virtual. *)
        llsc
    | Get (tn, indices) ->
        let traced = get_node traced_store tn in
        if Tn.known_non_virtual traced.tn then llsc
        else
          let id = get_scope tn in
          Option.value ~default:llsc
          @@ Option.map (inline_computation ~id computations_table traced static_indices indices)
               ~f:(fun body -> Local_scope { id; body; orig_indices = indices })
    | Local_scope opts ->
        Local_scope
          { opts with body = loop_proc ~process_for:(Set.add process_for opts.id.tn) opts.body }
    | Get_local _ -> llsc
    | Get_merge_buffer (_, _) -> llsc
    | Embed_index _ -> llsc
    | Ternop (op, llv1, llv2, llv3) ->
        Ternop
          ( op,
            loop_float ~process_for llv1,
            loop_float ~process_for llv2,
            loop_float ~process_for llv3 )
    | Binop (op, llv1, llv2) ->
        Binop (op, loop_float ~process_for llv1, loop_float ~process_for llv2)
    | Unop (op, llsc) -> Unop (op, loop_float ~process_for llsc)
  in
  loop_proc ~process_for:(Set.empty (module Tnode)) llc

let cleanup_virtual_llc reverse_node_map ~static_indices (llc : t) : t =
  (* The current position is within scope of the definitions of the process_for virtual arrays. *)
  let rec loop_proc ~balanced ~env_dom (llc : t) : t option =
    let loop = loop_proc ~balanced ~env_dom in
    match llc with
    | Noop -> None
    | Seq _ ->
        let body = List.filter_map ~f:loop @@ flat_lines [ llc ] in
        if List.is_empty body then None else Some (unflat_lines body)
    | For_loop ({ index; body; _ } as for_config) -> (
        let env_dom = Set.add env_dom index in
        match Hashtbl.find reverse_node_map index with
        | Some a ->
            if not @@ Tn.known_non_virtual a then (
              (* FIXME(#296): *)
              Tn.update_memory_mode a Virtual 15;
              None)
            else
              Option.map ~f:(fun body : t -> For_loop { for_config with body })
              @@ loop_proc ~balanced ~env_dom body
        | None ->
            Option.map ~f:(fun body : t -> For_loop { for_config with body })
            @@ loop_proc ~balanced ~env_dom body)
    | Zero_out tn ->
        if not @@ Tn.known_non_virtual tn then (
          (* FIXME(#296): *)
          Tn.update_memory_mode tn Virtual 151;
          None)
        else Some llc
    | Set { tn; idcs; llsc; debug } ->
        if not @@ Tn.known_non_virtual tn then (
          (* FIXME(#296): *)
          Tn.update_memory_mode tn Virtual 152;
          None)
        else (
          assert (
            Array.for_all idcs ~f:(function Indexing.Iterator s -> Set.mem env_dom s | _ -> true));
          Some (Set { tn; idcs; llsc = loop_float ~balanced ~env_dom llsc; debug }))
    | Set_from_vec { tn; idcs; length; vec_unop; arg; debug } ->
        if not @@ Tn.known_non_virtual tn then (
          (* FIXME(#296): *)
          Tn.update_memory_mode tn Virtual 152;
          None)
        else (
          assert (
            Array.for_all idcs ~f:(function Indexing.Iterator s -> Set.mem env_dom s | _ -> true));
          Some
            (Set_from_vec
               { tn; idcs; length; vec_unop; arg = loop_float ~balanced ~env_dom arg; debug }))
    | Set_local (id, llsc) ->
        assert (not @@ Tn.known_non_virtual id.tn);
        Tn.update_memory_mode id.tn Virtual 16;
        Some (Set_local (id, loop_float ~balanced ~env_dom llsc))
    | Comment _ -> Some llc
    | Staged_compilation _ -> Some llc
  and loop_float ~balanced ~env_dom (llsc : scalar_t) : scalar_t =
    let loop = loop_float ~balanced ~env_dom in
    match llsc with
    | Constant _ -> llsc
    | Get (a, indices) ->
        (* TODO(#296): this should probably already be Never_virtual, we could assert it. *)
        Tn.update_memory_mode a Never_virtual 17;
        assert (
          Array.for_all indices ~f:(function Indexing.Iterator s -> Set.mem env_dom s | _ -> true));
        llsc
    | Local_scope { id; body; orig_indices } ->
        assert (
          Array.for_all orig_indices ~f:(function
            | Indexing.Iterator s -> Set.mem env_dom s
            | _ -> true));
        if Tn.known_non_virtual id.tn then Get (id.tn, orig_indices)
        else
          let body = Option.value_exn ~here:[%here] @@ loop_proc ~balanced ~env_dom body in
          Tn.update_memory_mode id.tn Virtual 18;
          Local_scope { id; orig_indices; body }
    | Get_local id ->
        assert (not @@ Tn.known_non_virtual id.tn);
        Tn.update_memory_mode id.tn Virtual 16;
        llsc
    | Get_merge_buffer (_, _) -> llsc
    | Embed_index (Fixed_idx _ | Sub_axis) -> llsc
    | Embed_index (Iterator s) ->
        assert (Set.mem env_dom s);
        llsc
    | Embed_index (Affine { symbols; _ }) ->
        List.iter symbols ~f:(fun (_, s) -> assert (Set.mem env_dom s));
        llsc
    | Ternop (op, llv1, llv2, llv3) -> Ternop (op, loop llv1, loop llv2, loop llv3)
    | Binop (op, llv1, llv2) -> Binop (op, loop llv1, loop llv2)
    | Unop (op, llsc) -> Unop (op, loop llsc)
  in
  let static_indices =
    Set.of_list (module Indexing.Symbol)
    @@ List.map ~f:(fun s -> s.Indexing.static_symbol) static_indices
  in
  Option.value_exn ~here:[%here] @@ loop_proc ~balanced:false ~env_dom:static_indices llc

let rec substitute_float ~var ~value llsc =
  let loop_float = substitute_float ~var ~value in
  let loop_proc = substitute_proc ~var ~value in
  if equal_scalar_t var llsc then value
  else
    match llsc with
    | Constant _ -> llsc
    | Get (_ptr, _indices) -> llsc
    | Local_scope opts -> Local_scope { opts with body = loop_proc opts.body }
    | Get_local _ -> llsc
    | Get_merge_buffer (_, _) -> llsc
    | Embed_index _ -> llsc
    | Ternop (op, llv1, llv2, llv3) -> Ternop (op, loop_float llv1, loop_float llv2, loop_float llv3)
    | Binop (op, llv1, llv2) -> Binop (op, loop_float llv1, loop_float llv2)
    | Unop (op, llsc) -> Unop (op, loop_float llsc)

and substitute_proc ~var ~value llc =
  let loop_float = substitute_float ~var ~value in
  let loop_proc = substitute_proc ~var ~value in
  match llc with
  | Noop -> Noop
  | Seq (c1, c2) ->
      let c1 = loop_proc c1 in
      let c2 = loop_proc c2 in
      Seq (c1, c2)
  | For_loop for_config -> For_loop { for_config with body = loop_proc for_config.body }
  | Zero_out _ -> llc
  | Set { tn; idcs; llsc; debug } -> Set { tn; idcs; llsc = loop_float llsc; debug }
  | Set_from_vec { tn; idcs; length; vec_unop; arg; debug } ->
      Set_from_vec { tn; idcs; length; vec_unop; arg = loop_float arg; debug }
  | Set_local (id, llsc) -> Set_local (id, loop_float llsc)
  | Comment _ -> llc
  | Staged_compilation _ -> llc

let simplify_llc llc =
  (* Implements top-down rewriting. *)
  let rec loop_proc (llc : t) : t =
    let loop = loop_proc in
    match llc with
    | Noop -> Noop
    | Seq (c1, c2) ->
        let c1 = loop c1 in
        let c2 = loop c2 in
        Seq (c1, c2)
    | For_loop for_config -> For_loop { for_config with body = loop for_config.body }
    | Zero_out _ -> llc
    | Set { tn; idcs; llsc; debug } -> Set { tn; idcs; llsc = loop_float llsc; debug }
    | Set_from_vec { tn; idcs; length; vec_unop; arg; debug } ->
        Set_from_vec { tn; idcs; length; vec_unop; arg = loop_float arg; debug }
    | Set_local (id, llsc) -> Set_local (id, loop_float llsc)
    | Comment _ -> llc
    | Staged_compilation _ -> llc
  and loop_float (llsc : scalar_t) : scalar_t =
    let local_scope_body, llsc' =
      match llsc with
      | Local_scope opts ->
          ( opts.body,
            Local_scope
              {
                opts with
                body =
                  unflat_lines
                  @@ List.filter ~f:(function Comment _ -> false | _ -> true)
                  @@ flat_lines [ opts.body ];
              } )
      | _ -> (Noop, llsc)
    in
    match llsc' with
    | Constant _ -> llsc
    | Get (_ptr, _indices) -> llsc
    | Local_scope { id; body = Set_local (id2, v); _ } when equal_scope_id id id2 -> loop_float v
    | Local_scope { id; body = Seq (Set_local (id1, v1), Set_local (id2, v2)); _ }
      when equal_scope_id id id1 && equal_scope_id id id2 ->
        loop_float @@ substitute_float ~var:(Get_local id) ~value:v1 v2
    | Local_scope opts -> Local_scope { opts with body = loop_proc local_scope_body }
    | Get_local _ -> llsc
    | Get_merge_buffer (_, _) -> llsc
    | Embed_index (Fixed_idx i) -> Constant (Float.of_int i)
    | Embed_index Sub_axis -> Constant 0.
    | Embed_index (Iterator _) -> llsc
    | Embed_index (Affine _) -> llsc (* Cannot simplify affine expressions to constants *)
    | Binop (Arg1, llv1, _) -> loop_float llv1
    | Binop (Arg2, _, llv2) -> loop_float llv2
    | Binop (Threefry4x32, _, _) -> llsc
    | Binop (op, Constant c1, Constant c2) -> Constant (Ops.interpret_binop op c1 c2)
    | Binop (Add, llsc, Constant 0.)
    | Binop (Sub, llsc, Constant 0.)
    | Binop (Add, Constant 0., llsc) ->
        loop_float llsc
    | Binop (Sub, Constant 0., llsc) -> loop_float @@ Binop (Mul, Constant (-1.), llsc)
    | Binop (Mul, llsc, Constant 1.)
    | Binop (Div, llsc, Constant 1.)
    | Binop (Mul, Constant 1., llsc) ->
        loop_float llsc
    | Binop (Mul, _, Constant 0.) | Binop (Div, Constant 0., _) | Binop (Mul, Constant 0., _) ->
        Constant 0.
    | Binop (Add, (Binop (Add, Constant c2, llsc) | Binop (Add, llsc, Constant c2)), Constant c1)
    | Binop (Add, Constant c1, (Binop (Add, Constant c2, llsc) | Binop (Add, llsc, Constant c2))) ->
        loop_float @@ Binop (Add, Constant (c1 +. c2), llsc)
    | Binop (Sub, (Binop (Add, Constant c2, llsc) | Binop (Add, llsc, Constant c2)), Constant c1) ->
        loop_float @@ Binop (Add, Constant (c2 -. c1), llsc)
    | Binop (Sub, Constant c1, (Binop (Add, Constant c2, llsc) | Binop (Add, llsc, Constant c2))) ->
        loop_float @@ Binop (Sub, Constant (c1 -. c2), llsc)
    | Binop (Add, llv1, Binop (Sub, llv2, llv3)) | Binop (Add, Binop (Sub, llv2, llv3), llv1) ->
        loop_float @@ Binop (Sub, Binop (Add, llv1, llv2), llv3)
    | Binop (Sub, llv1, Binop (Sub, llv2, llv3)) ->
        loop_float @@ Binop (Sub, Binop (Add, llv1, llv3), llv2)
    | Binop (Sub, Binop (Sub, llv1, llv2), llv3) ->
        loop_float @@ Binop (Sub, llv1, Binop (Add, llv2, llv3))
    | Binop (Mul, (Binop (Mul, Constant c2, llsc) | Binop (Mul, llsc, Constant c2)), Constant c1)
    | Binop (Mul, Constant c1, (Binop (Mul, Constant c2, llsc) | Binop (Mul, llsc, Constant c2))) ->
        loop_float @@ Binop (Mul, Constant (c1 *. c2), llsc)
    | Binop (Div, (Binop (Mul, Constant c2, llsc) | Binop (Mul, llsc, Constant c2)), Constant c1) ->
        loop_float @@ Binop (Mul, Constant (c2 /. c1), llsc)
    | Binop (Div, Constant c1, (Binop (Mul, Constant c2, llsc) | Binop (Mul, llsc, Constant c2))) ->
        (* TODO: this might worsen the conditioning in hand-designed formula cases. *)
        loop_float @@ Binop (Div, Constant (c1 /. c2), llsc)
    | Binop (Mul, llv1, Binop (Div, llv2, llv3)) | Binop (Mul, Binop (Div, llv2, llv3), llv1) ->
        loop_float @@ Binop (Div, Binop (Mul, llv1, llv2), llv3)
    | Binop (Div, llv1, Binop (Div, llv2, llv3)) ->
        loop_float @@ Binop (Div, Binop (Mul, llv1, llv3), llv2)
    | Binop (Div, Binop (Div, llv1, llv2), llv3) ->
        loop_float @@ Binop (Div, llv1, Binop (Mul, llv2, llv3))
    | Binop (ToPowOf, llv1, llv2) -> (
        let v1 : scalar_t = loop_float llv1 in
        let v2 : scalar_t = loop_float llv2 in
        let result : scalar_t = Binop (ToPowOf, v1, v2) in
        if not !optimize_integer_pow then result
        else
          match v2 with
          | Constant c when Float.is_integer c ->
              loop_float @@ unroll_pow ~base:v1 ~exp:(Float.to_int c)
          | _ -> result)
    | Binop (Add, Binop (Mul, llv1, llv2), llv3) | Binop (Add, llv3, Binop (Mul, llv1, llv2)) ->
        (* TODO: this is tentative. *)
        loop_float @@ Ternop (FMA, llv1, llv2, llv3)
    | Binop (op, llv1, llv2) ->
        let v1 = loop_float llv1 in
        let v2 = loop_float llv2 in
        let result = Binop (op, v1, v2) in
        if equal_scalar_t llv1 v1 && equal_scalar_t llv2 v2 then result else loop_float result
    | Ternop (op, llv1, llv2, llv3) ->
        let v1 = loop_float llv1 in
        let v2 = loop_float llv2 in
        let v3 = loop_float llv3 in
        let result = Ternop (op, v1, v2, v3) in
        if equal_scalar_t llv1 v1 && equal_scalar_t llv2 v2 then result else loop_float result
    | Unop (Identity, llsc) -> loop_float llsc
    | Unop (op, Constant c) -> Constant (Ops.interpret_unop op c)
    | Unop (op, llsc) ->
        let v = loop_float llsc in
        let result = Unop (op, v) in
        if equal_scalar_t llsc v then result else loop_float result
  in
  let check_constant tn c =
    if Tn.exceeds_fp16_cutoff tn c then
      raise
      @@ Utils.User_error
           ("Constant " ^ Float.to_string c
          ^ " is too big for FP16 aka. half precision, risk of overflow; increase precision of \
             tensor node " ^ Tn.debug_name tn)
  in
  let rec check_proc llc =
    let loop = check_proc in
    match llc with
    | Seq (c1, c2) ->
        loop c1;
        loop c2
    | For_loop { body; _ } -> loop body
    | Zero_out _ -> ()
    | Set { tn; llsc; _ } -> check_float tn llsc
    | Set_from_vec { tn; arg; _ } -> check_float tn arg
    | Set_local (id, llsc) -> check_float id.tn llsc
    | Noop | Comment _ | Staged_compilation _ -> ()
  and check_float tn llsc =
    let loop = check_float tn in
    match llsc with
    | Constant c -> check_constant tn c
    | Local_scope { body; _ } -> check_proc body
    | Ternop (_, v1, v2, v3) ->
        loop v1;
        loop v2;
        loop v3
    | Binop (_, v1, v2) ->
        loop v1;
        loop v2
    | Unop (_, v) -> loop v
    | Embed_index (Indexing.Fixed_idx i) -> check_constant tn (Float.of_int i)
    | Embed_index _ | Get_local _ | Get_merge_buffer (_, _) | Get (_, _) -> ()
  in
  let result = loop_proc llc in
  if Option.is_some Utils.settings.check_half_prec_constants_cutoff then check_proc result;
  result

let input_and_output_nodes optimized =
  ( Hashtbl.fold optimized.traced_store
      ~init:(Set.empty (module Tn), Set.empty (module Tn))
      ~f:(fun ~key ~data (inputs, outputs) ->
        let materialized = Tn.is_materialized_force key 50 in
        let inputs =
          if
            materialized
            && (not (Tn.known_constant key))
            && (data.read_only || data.read_before_write)
          then Set.add inputs key
          else inputs
        in
        let outputs =
          if materialized && (data.zeroed_out || not (Hash_set.is_empty data.assignments)) then
            Set.add outputs key
          else outputs
        in
        (inputs, outputs)),
    optimized.merge_node )

let%diagn2_sexp optimize_proc (input_ctx : optimize_ctx) static_indices llc =
  let traced_store = Hashtbl.create (module Tnode) in
  (* Identifies the computations that the code block associated with the symbol belongs to. *)
  let reverse_node_map = Hashtbl.create (module Indexing.Symbol) in
  [%log "tracing"];
  let merge_node_id = ref None in
  visit_llc traced_store ~merge_node_id reverse_node_map ~max_visits:virtualize_settings.max_visits
    llc;
  [%log "optimizing"];
  let virtual_llc_result =
    virtual_llc input_ctx.computations traced_store reverse_node_map static_indices llc
  in
  let llc =
    simplify_llc @@ cleanup_virtual_llc reverse_node_map ~static_indices @@ virtual_llc_result
  in
  let merge_node =
    Option.map !merge_node_id ~f:(fun id -> Option.value_exn ~here:[%here] @@ Tnode.find ~id)
  in
  let optimize_ctx = input_ctx in
  { traced_store; optimize_ctx; llc; merge_node }

let code_hum_margin = ref 100

open Indexing.Doc_helpers

let function_header_doc ?name ?static_indices () =
  let open PPrint in
  match (name, static_indices) with
  | Some name, Some static_indices ->
      !^name ^^ space
      ^^ parens (separate comma_sep (List.map ~f:pp_static_symbol static_indices))
      ^^ colon ^^ space
  | Some name, None -> !^name ^^ colon ^^ space
  | _ -> empty

let get_ident_within_code ?no_dots ?(blacklist = []) llcs =
  let ident_style = Tn.get_style ~arg_name:"ll_ident_style" ?no_dots () in
  let nograd_idents = Hashtbl.create (module String) in
  let grad_idents = Hashtbl.create (module String) in
  List.iter blacklist ~f:(fun b_ident ->
      (* Consider blacklisted items as already seen with a placeholder ID like -1 to avoid
         clashes *)
      Hashtbl.set nograd_idents ~key:b_ident ~data:(Set.singleton (module Int) (-1));
      Hashtbl.set grad_idents ~key:b_ident ~data:(Set.singleton (module Int) (-1)));
  let visit tn =
    let is_grad, ident = Tn.no_grad_ident_label tn in
    let idents = if is_grad then grad_idents else nograd_idents in
    Option.iter ident
      ~f:
        (Hashtbl.update idents ~f:(fun old ->
             Set.add (Option.value ~default:Utils.no_ints old) tn.id))
  in
  let rec loop (c : t) =
    match c with
    | Noop | Comment _ | Staged_compilation _ -> ()
    | Seq (c1, c2) ->
        loop c1;
        loop c2
    | For_loop { body; _ } -> loop body
    | Zero_out la -> visit la
    | Set { tn; llsc; _ } ->
        visit tn;
        loop_float llsc
    | Set_from_vec { tn; arg; _ } ->
        visit tn;
        loop_float arg
    | Set_local ({ tn; _ }, llsc) ->
        visit tn;
        loop_float llsc
  and loop_float fc =
    match fc with
    | Local_scope { id = { tn; _ }; body; orig_indices = _ } ->
        visit tn;
        loop body
    | Get_merge_buffer (la, _) -> visit la
    | Get (la, _) -> visit la
    | Ternop (_, f1, f2, f3) ->
        loop_float f1;
        loop_float f2;
        loop_float f3
    | Binop (_, f1, f2) ->
        loop_float f1;
        loop_float f2
    | Unop (_, f) -> loop_float f
    | Get_local { tn; _ } -> visit tn
    | Constant _ | Embed_index _ -> ()
  in
  Array.iter ~f:loop llcs;
  let repeating_nograd_idents =
    Hashtbl.filter nograd_idents ~f:(fun ids -> List.length (Set.to_list ids) > 1)
  in
  let repeating_grad_idents =
    Hashtbl.filter grad_idents ~f:(fun ids -> List.length (Set.to_list ids) > 1)
  in
  fun tn ->
    let ident = Tn.styled_ident ~repeating_nograd_idents ~repeating_grad_idents ident_style tn in
    Tn.update_code_name tn ident;
    ident

let to_doc_cstyle ?name ?static_indices () llc =
  let ident_label = get_ident_within_code [| llc |] in
  let open PPrint in
  let doc_ident la = string (ident_label la) in
  let doc_local { tn; scope_id } = string ("v" ^ Int.to_string scope_id ^ "_") ^^ doc_ident tn in

  let rec doc_of_code c =
    match c with
    | Noop -> empty
    | Seq (c1, c2) ->
        let docs =
          List.filter_map [ c1; c2 ] ~f:(function Noop -> None | c -> Some (doc_of_code c))
        in
        separate hardline docs
    | For_loop { index = i; from_; to_; body; trace_it = _ } ->
        let header =
          string "for " ^^ pp_symbol i ^^ string " = " ^^ int from_ ^^ string " to " ^^ int to_
          ^^ string " {"
        in
        let body_doc = nest 2 (break 1 ^^ doc_of_code body) in
        group (header ^^ body_doc ^^ break 1 ^^ string "}")
    | Zero_out tn -> string "zero_out " ^^ doc_ident tn ^^ string ";"
    | Set p ->
        let prec = Lazy.force p.tn.prec in
        let result =
          group
            (doc_ident p.tn
            ^^ brackets (pp_indices p.idcs)
            ^^ string " := " ^^ doc_of_float prec p.llsc ^^ string ";")
        in
        if not (String.is_empty p.debug) then (
          let b = Buffer.create 100 in
          PPrint.ToBuffer.pretty 0.7 100 b result;
          p.debug <- Buffer.contents b);
        result
    | Set_from_vec p ->
        let prec = Lazy.force p.tn.prec in
        let prefix, postfix = Ops.vec_unop_c_syntax prec p.vec_unop in
        (* TODO: this assumes argument is generated from the high-level code, which means it is
           either Get or Local_scope -- they don't need precision. *)
        let vec_result = string prefix ^^ doc_of_float Ops.Void_prec p.arg ^^ string postfix in
        let length_doc = string ("<" ^ Int.to_string p.length ^ ">") in
        let result =
          group
            (doc_ident p.tn
            ^^ brackets (pp_indices p.idcs)
            ^^ length_doc ^^ string " := " ^^ vec_result ^^ string ";")
        in
        if not (String.is_empty p.debug) then (
          let b = Buffer.create 100 in
          PPrint.ToBuffer.pretty 0.7 100 b result;
          p.debug <- Buffer.contents b);
        result
    | Comment message -> string ("/* " ^ message ^ " */")
    | Staged_compilation callback -> callback ()
    | Set_local (id, llsc) ->
        let prec = Lazy.force id.tn.prec in
        group (doc_local id ^^ string " := " ^^ doc_of_float prec llsc ^^ string ";")
  and doc_of_float prec value =
    match value with
    | Local_scope { id; body; _ } ->
        group
          (doc_local id ^^ string " {"
          ^^ nest 2 (break 1 ^^ doc_of_code body)
          ^^ break 1 ^^ string "}")
    | Get_local id -> doc_local id
    | Get_merge_buffer (source, idcs) ->
        group (doc_ident source ^^ string ".merge" ^^ brackets (pp_indices idcs))
    | Get (tn, idcs) -> group (doc_ident tn ^^ brackets (pp_indices idcs))
    | Constant c -> string (Printf.sprintf "%.16g" c)
    | Embed_index idx ->
        let idx_doc = pp_axis_index idx in
        if PPrint.is_empty idx_doc then string "0" else idx_doc
    | Ternop (op, v1, v2, v3) ->
        let prefix, comma1, comma2, postfix = Ops.ternop_c_syntax prec op in
        group
          (string prefix ^^ doc_of_float prec v1 ^^ string comma1 ^^ space ^^ doc_of_float prec v2
         ^^ string comma2 ^^ space ^^ doc_of_float prec v3 ^^ string postfix)
    | Binop (Arg1, v1, _v2) -> doc_of_float prec v1
    | Binop (Arg2, _v1, v2) -> doc_of_float prec v2
    | Binop (op, v1, v2) ->
        let prefix, infix, postfix = Ops.binop_c_syntax prec op in
        group
          (string prefix ^^ doc_of_float prec v1 ^^ string infix ^^ space ^^ doc_of_float prec v2
         ^^ string postfix)
    | Unop (Identity, v) -> doc_of_float prec v
    | Unop (op, v) ->
        let prefix, postfix = Ops.unop_c_syntax prec op in
        string prefix ^^ doc_of_float prec v ^^ string postfix
  in
  hardline ^^ nest 2 (function_header_doc ?name ?static_indices () ^^ doc_of_code llc)

let to_doc ?name ?static_indices () llc =
  let ident_label = get_ident_within_code [| llc |] in
  let open PPrint in
  let doc_ident la = string (ident_label la) in
  let doc_local { tn; scope_id } = string ("v" ^ Int.to_string scope_id ^ "_") ^^ doc_ident tn in

  let rec doc_of_code c =
    match c with
    | Noop -> empty
    | Seq (c1, c2) ->
        let docs =
          List.filter_map [ c1; c2 ] ~f:(function Noop -> None | c -> Some (doc_of_code c))
        in
        separate hardline docs
    | For_loop { index = i; from_; to_; body; trace_it = _ } ->
        let header =
          string "for " ^^ pp_symbol i ^^ string " = " ^^ int from_ ^^ string " to " ^^ int to_
          ^^ string " {"
        in
        let body_doc = nest 2 (break 1 ^^ doc_of_code body) in
        group (header ^^ body_doc ^^ break 1 ^^ string "}")
    | Zero_out tn -> string "zero_out " ^^ doc_ident tn ^^ string ";"
    | Set p ->
        let result =
          group
            (doc_ident p.tn
            ^^ brackets (pp_indices p.idcs)
            ^^ string " := " ^^ doc_of_float p.llsc ^^ string ";")
        in
        let b = Buffer.create 100 in
        PPrint.ToBuffer.pretty 0.7 100 b result;
        p.debug <- Buffer.contents b;
        result
    | Set_from_vec p ->
        let length_doc = string ("<" ^ Int.to_string p.length ^ ">") in
        let result =
          group
            (doc_ident p.tn
            ^^ brackets (pp_indices p.idcs)
            ^^ length_doc ^^ string " := "
            ^^ string (Ops.vec_unop_cd_syntax p.vec_unop)
            ^^ string "(" ^^ doc_of_float p.arg ^^ string ", " ^^ length_doc ^^ string ");")
        in
        let b = Buffer.create 100 in
        PPrint.ToBuffer.pretty 0.7 100 b result;
        p.debug <- Buffer.contents b;
        result
    | Comment message -> string ("/* " ^ message ^ " */")
    | Staged_compilation callback -> callback ()
    | Set_local (id, llsc) ->
        group (doc_local id ^^ string " := " ^^ doc_of_float llsc ^^ string ";")
  and doc_of_float value =
    match value with
    | Local_scope { id; body; _ } ->
        group
          (doc_local id ^^ string " {"
          ^^ nest 2 (break 1 ^^ doc_of_code body)
          ^^ break 1 ^^ string "}")
    | Get_local id -> doc_local id
    | Get_merge_buffer (source, idcs) ->
        group (doc_ident source ^^ string ".merge" ^^ brackets (pp_indices idcs))
    | Get (tn, idcs) -> group (doc_ident tn ^^ brackets (pp_indices idcs))
    | Constant c -> string (Printf.sprintf "%.16g" c)
    | Embed_index idx ->
        let idx_doc = pp_axis_index idx in
        if PPrint.is_empty idx_doc then string "0" else idx_doc
    | Ternop (op, v1, v2, v3) ->
        let prefix = Ops.ternop_cd_syntax op in
        group
          (string prefix
          ^^ parens
               (doc_of_float v1 ^^ string "," ^^ space ^^ doc_of_float v2 ^^ string "," ^^ space
              ^^ doc_of_float v3))
    | Binop (Arg1, v1, _v2) -> doc_of_float v1
    | Binop (Arg2, _v1, v2) -> doc_of_float v2
    | Binop (op, v1, v2) ->
        if Ops.is_binop_nice_infix op then
          let infix = Ops.binop_cd_syntax op in
          group (parens (doc_of_float v1 ^^ space ^^ string infix ^^ space ^^ doc_of_float v2))
        else
          let prefix = Ops.binop_cd_fallback_syntax op in
          group (string prefix ^^ parens (doc_of_float v1 ^^ string "," ^^ space ^^ doc_of_float v2))
    | Unop (Identity, v) -> doc_of_float v
    | Unop (op, v) ->
        let prefix = Ops.unop_cd_syntax op in
        string prefix ^^ parens (doc_of_float v)
  in

  hardline ^^ nest 2 (function_header_doc ?name ?static_indices () ^^ doc_of_code llc)

let%diagn2_sexp optimize (input_ctx : optimize_ctx) ~unoptim_ll_source ~ll_source ~(name : string)
    (static_indices : Indexing.static_symbol list) (llc : t) : optimized =
  Option.iter unoptim_ll_source ~f:(fun callback -> callback (to_doc ~name ~static_indices () llc));
  let result = optimize_proc input_ctx static_indices llc in
  Option.iter ll_source ~f:(fun callback -> callback (to_doc ~name ~static_indices () result.llc));
  result

let loop_over_dims dims ~body =
  let rec for_loop rev_idcs : _ -> t = function
    | [] -> body @@ Array.of_list_rev rev_idcs
    | d :: product when not @@ Indexing.iterated d ->
        for_loop (Indexing.Fixed_idx 0 :: rev_idcs) product
    | d :: product ->
        let index = Indexing.get_symbol () in
        For_loop
          {
            index;
            from_ = 0;
            to_ = d - 1;
            body = for_loop (Indexing.Iterator index :: rev_idcs) product;
            trace_it = true;
          }
  in
  for_loop [] (Array.to_list dims)

let unroll_dims dims ~body =
  if Array.is_empty dims then body [||] ~offset:0
  else
    (* Calculate strides for each dimension (rightmost changes fastest) *)
    let strides = Array.create ~len:(Array.length dims) 1 in
    for i = Array.length dims - 2 downto 0 do
      strides.(i) <- strides.(i + 1) * dims.(i + 1)
    done;

    (* Generate all combinations of indices *)
    let rec generate_all_combinations indices_so_far offset dim_index =
      if dim_index >= Array.length dims then
        (* We have a complete combination, call the body *)
        body (Array.of_list_rev indices_so_far) ~offset
      else
        (* Generate all values for current dimension *)
        let results = ref [] in
        for i = 0 to dims.(dim_index) - 1 do
          let new_offset = offset + (i * strides.(dim_index)) in
          let result =
            generate_all_combinations
              (Indexing.Fixed_idx i :: indices_so_far)
              new_offset (dim_index + 1)
          in
          results := result :: !results
        done;
        unflat_lines (List.rev !results)
    in
    generate_all_combinations [] 0 0
