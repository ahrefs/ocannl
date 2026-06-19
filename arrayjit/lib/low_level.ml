open Base
module Lazy = Utils.Lazy
module Nd = Ndarray
module Tn = Tnode

let _get_local_debug_runtime = Utils.get_local_debug_runtime

[%%global_debug_log_level 0]

(* export OCANNL_LOG_LEVEL_LOW_LEVEL=9 to enable debugging into the log_files/ directory. *)
[%%global_debug_log_level_from_env_var "OCANNL_LOG_LEVEL_LOW_LEVEL"]

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
      arg : scalar_arg;
      mutable debug : string;
    }
  | Set_local of scope_id * scalar_t
  | Declare_local of { id : scope_id; needs_init : bool }
[@@deriving sexp_of, equal]

and scalar_t =
  | Local_scope of { id : scope_id; body : t; orig_indices : Indexing.axis_index array }
  | Get_local of scope_id
  | Get of Tn.t * Indexing.axis_index array
  | Get_dynamic of {
      tn : Tn.t;  (** The gathered table; treated as a read of [tn], like [Get]. *)
      idcs : Indexing.axis_index array;  (** Static everywhere except [dyn_axis]. *)
      dyn_axis : int;  (** Which [idcs] slot is replaced by [dyn_value] at codegen time. *)
      dyn_value : scalar_arg;
          (** Integer-valued index spliced into the row-major offset at [dyn_axis]. A nested scalar:
              all recursive scalar traversals must descend into it. gh-343: produced only by
              [rewrite_one_hot_reductions], never escapes [Low_level] / backend codegen. *)
    }
  | Get_merge_buffer of Tn.t * Indexing.axis_index array
  | Ternop of Ops.ternop * scalar_arg * scalar_arg * scalar_arg
  | Binop of Ops.binop * scalar_arg * scalar_arg
  | Unop of Ops.unop * scalar_arg
  | Constant of float
  | Constant_bits of int64  (** Direct bit representation, primarily for uint4x32 *)
  | Embed_index of Indexing.axis_index
[@@deriving sexp_of, equal, compare]

and scalar_arg = scalar_t * Ops.prec [@@deriving sexp_of, equal, compare]

(** Extract the precision from a scalar value by examining its origin tensor node *)
let scalar_precision = function
  | Get (tn, _) -> Lazy.force tn.Tn.prec
  | Get_dynamic { tn; _ } -> Lazy.force tn.Tn.prec
  | Get_merge_buffer (tn, _) -> Lazy.force tn.Tn.prec
  | Get_local { tn; _ } -> Lazy.force tn.Tn.prec
  | Local_scope { id; _ } -> Lazy.force id.tn.Tn.prec
  | Constant _ ->
      (* Single is the most widely supported precision, so we use it as the default. *)
      Ops.single
  | Constant_bits _ -> Ops.int64
  | Embed_index _ -> Ops.index_prec ()
  | Ternop (_, (_, prec), _, _) -> prec
  | Binop (_, (_, prec), _) -> prec
  | Unop (_, (_, prec)) -> prec

(** Helper to construct binary/ternary/unary ops with proper precision *)
let mk_binop op arg1 arg2 = Binop (op, (arg1, scalar_precision arg1), (arg2, scalar_precision arg2))

let mk_ternop op arg1 arg2 arg3 =
  Ternop
    (op, (arg1, scalar_precision arg1), (arg2, scalar_precision arg2), (arg3, scalar_precision arg3))

let mk_unop op arg = Unop (op, (arg, scalar_precision arg))

let apply_op op args =
  match (op, args) with
  | Ops.Binop Ops.Arg1, [| rhs1; _ |] -> rhs1
  | Binop Arg2, [| _; rhs2 |] -> rhs2
  | Unop Identity, [| rhs |] -> rhs
  | Ternop op, [| rhs1; rhs2; rhs3 |] -> mk_ternop op rhs1 rhs2 rhs3
  | Binop op, [| rhs1; rhs2 |] -> mk_binop op rhs1 rhs2
  | Unop op, [| rhs |] -> mk_unop op rhs
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
  mutable inline_complex_computations : bool;
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
  let inline_complex_computations =
    Utils.get_global_flag ~default:true ~arg_name:"inline_complex_computations"
  in
  {
    enable_device_only;
    max_visits;
    max_tracing_dim;
    inline_scalar_constexprs;
    inline_simple_computations;
    inline_complex_computations;
  }

type visits = Visits of int | Recurrent [@@deriving sexp, equal, variants]

type traced_array = {
  tn : Tn.t;
  assignments : int array Hash_set.t;
  accesses : (int array, visits) Hashtbl.t;
  mutable zero_initialized_by_code : bool;
  mutable zeroed_out : bool;
  mutable read_before_write : bool;
  mutable read_only : bool;
  mutable is_scalar_constexpr : bool;
  mutable is_accessing : bool;
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
        zero_initialized_by_code = false;
        zeroed_out = false;
        read_before_write = false;
        read_only = false;
        is_scalar_constexpr = false;
        is_accessing = false;
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
    | Get_dynamic _ -> false (* a runtime gather is never a scalar constexpr *)
    | Get_merge_buffer (tn, _) ->
        let traced = get_node traced_store tn in
        traced.is_scalar_constexpr
    | Ternop (_, (v1, _), (v2, _), (v3, _)) -> loop v1 && loop v2 && loop v3
    | Binop (_, (v1, _), (v2, _)) -> loop v1 && loop v2
    | Unop (_, (v, _)) -> loop v
    | Constant _ | Constant_bits _ -> true
    | Embed_index _ -> false
  in
  loop llsc

let is_accessing_comp traced_store llsc =
  let rec loop llsc =
    match llsc with
    | Get_local { tn; _ } | Local_scope { id = { tn; _ }; _ } ->
        let traced = get_node traced_store tn in
        traced.is_accessing
    | Get (tn, _) ->
        let traced = get_node traced_store tn in
        not traced.is_scalar_constexpr
    | Get_dynamic _ -> true (* accesses the table at a runtime index *)
    | Get_merge_buffer (tn, _) ->
        let traced = get_node traced_store tn in
        traced.is_accessing <- true;
        true
    | Ternop (_, (v1, _), (v2, _), (v3, _)) -> loop v1 || loop v2 || loop v3
    | Binop (_, (v1, _), (v2, _)) -> loop v1 || loop v2
    | Unop (_, (v, _)) -> loop v
    | Constant _ | Constant_bits _ -> false
    | Embed_index _ -> false
  in
  loop llsc

let is_complex_comp traced_store llsc =
  let accessing = is_accessing_comp traced_store in
  match llsc with
  | Get_local { tn; _ } | Local_scope { id = { tn; _ }; _ } ->
      let traced = get_node traced_store tn in
      traced.is_complex
  | Get _ -> false
  | Get_dynamic { dyn_value = v, _; _ } -> accessing v
  | Get_merge_buffer _ -> false
  | Ternop (_, (v1, _), (v2, _), (v3, _)) -> accessing v1 || accessing v2 || accessing v3
  | Binop (_, (v1, _), (v2, _)) -> accessing v1 || accessing v2
  | Unop (_, (v, _)) -> accessing v
  | Constant _ | Constant_bits _ -> false
  | Embed_index _ -> false

let is_scalar_dims tn = Array.for_all ~f:(( = ) 1) @@ Lazy.force tn.Tn.dims

(* Records [tn] as a candidate owner of each loop symbol appearing in its assignment indices.
   Multiple tensors may share a symbol (e.g. Block/concat lowering): all are recorded, in
   first-seen (trace) order and deduplicated, so that [virtual_llc] can store one computation per
   candidate. See #134. *)
let track_symbol reverse_node_map tn idcs =
  let add s =
    let existing = Hashtbl.find reverse_node_map s |> Option.value ~default:[] in
    if not (List.mem existing tn ~equal:Tn.equal) then
      Hashtbl.set reverse_node_map ~key:s ~data:(existing @ [ tn ])
  in
  Array.iter idcs ~f:(function
    | Indexing.Fixed_idx _ -> ()
    | Indexing.Sub_axis -> ()
    | Indexing.Iterator s -> add s
    | Indexing.Affine { symbols; _ } -> List.iter symbols ~f:(fun (_, s) -> add s)
    | Indexing.Concat syms -> List.iter syms ~f:add)

let visit_llc traced_store ~merge_node_id reverse_node_map ~max_visits llc =
  (* FIXME(#351): avoid excessive inlining while CSE is not implemented *)
  let is_too_many = function Visits i -> i > max_visits | Recurrent -> true in
  (* FIXME: migrate hashtable to use offsets instead of indices *)
  let lookup env indices =
    Array.map indices ~f:(function
      | Indexing.Fixed_idx i -> i
      | Indexing.Sub_axis -> 0
      | Iterator s -> Option.value ~default:(* static index *) 0 @@ Map.find env s
      | Indexing.Affine { symbols; offset } ->
          List.fold symbols ~init:offset ~f:(fun acc (coeff, s) ->
              acc + (coeff * (Option.value ~default:0 @@ Map.find env s)))
      | Indexing.Concat _syms ->
          (* Concat should be eliminated during lowering before we get here *)
          invalid_arg
            "BUG: Concat index encountered during virtualization - should have been eliminated \
             during lowering")
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
          traced.zero_initialized_by_code <- true;
          traced.is_accessing <- false;
          traced.is_complex <- false;
          if is_scalar_dims tn then traced.is_scalar_constexpr <- true);
        traced.zeroed_out <- true
    | Set { tn; idcs; llsc; debug = _ } ->
        loop_scalar env (Some (lookup env idcs)) llsc;
        let traced : traced_array = get_node traced_store tn in
        if
          Hash_set.is_empty traced.assignments
          && Hashtbl.is_empty traced.accesses && is_scalar_dims tn
        then traced.is_scalar_constexpr <- is_constexpr_comp traced_store llsc
          (* Note: this prevents detection if the same constant is assigned inside a loop. *)
        else if not @@ Hash_set.is_empty traced.assignments then traced.is_scalar_constexpr <- false;
        if first_visit then (
          traced.is_accessing <- traced.is_accessing || is_accessing_comp traced_store llsc;
          traced.is_complex <- traced.is_complex || is_complex_comp traced_store llsc);
        Hash_set.add traced.assignments (lookup env idcs);
        (* Track which tensors use which loop symbol. Multiple tensors may legitimately share a
           symbol (e.g. Block/concat lowering); all of them are recorded as candidate owners in
           first-seen (trace) order. Sharing a symbol no longer marks tensors [is_complex] -- only
           genuine computation complexity does (see #134). *)
        track_symbol reverse_node_map tn idcs
    | Set_from_vec { tn; idcs; length; vec_unop = _; arg = arg, _; debug = _ } ->
        loop_scalar env (Some (lookup env idcs)) arg;
        let traced : traced_array = get_node traced_store tn in
        (* Vector operations cannot be scalar constexpr *)
        traced.is_scalar_constexpr <- false;
        if first_visit then (
          traced.is_accessing <- traced.is_accessing || is_accessing_comp traced_store arg;
          traced.is_complex <- traced.is_complex || not (is_constexpr_comp traced_store arg));
        (* Mark all positions that will be written to *)
        for i = 0 to length - 1 do
          let pos_idcs = Array.copy idcs in
          (* Robustness against empty axes shapes *)
          let pos_idcs = if Array.is_empty pos_idcs then [| Indexing.Fixed_idx 0 |] else pos_idcs in
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
        track_symbol reverse_node_map tn idcs
    | Set_local (_, llsc) -> loop_scalar env None llsc
    | Declare_local _ -> ()
    | Comment _ -> ()
    | Staged_compilation _ -> ()
  and loop_scalar env (access_pos : int array option) llsc =
    let loop = loop_scalar env access_pos in
    match llsc with
    | Constant _ | Constant_bits _ -> ()
    | Get (ptr, indices) ->
        let traced : traced_array = get_node traced_store ptr in
        let at_pos = lookup env indices in
        if
          (not virtualize_settings.inline_complex_computations)
          || Option.value_map access_pos ~default:true ~f:(fun pos ->
              not ([%equal: int array] pos at_pos))
        then
          Hashtbl.update traced.accesses at_pos
            ~f:(visit ~is_assigned:(traced.zeroed_out || Hash_set.mem traced.assignments at_pos))
    | Get_dynamic { dyn_value = v, _; _ } ->
        (* gh-343: [Get_dynamic] is produced after this tracing pass, so this arm is defensive; the
           dynamic index sub-expression is still traversed for completeness. *)
        loop v
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
    | Binop (Arg1, (llv1, _), _llv2) -> loop llv1
    | Binop (Arg2, _llv1, (llv2, _)) -> loop llv2
    | Ternop (_, (llv1, _), (llv2, _), (llv3, _)) ->
        loop llv1;
        loop llv2;
        loop llv3
    | Binop (_, (llv1, _), (llv2, _)) ->
        loop llv1;
        loop llv2
    | Unop (_, (llsc, _)) -> loop llsc
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
        if Tn.mode_is_unspecified tn then Tn.update_memory_mode tn Materialized 37
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
        Tn.update_memory_mode tn Materialized 36))

let%diagn2_sexp check_and_store_virtual computations_table traced static_indices top_llc =
  let exception Non_virtual of int in
  let static_indices =
    Set.of_list (module Indexing.Symbol)
    @@ List.map ~f:(fun s -> s.Indexing.static_symbol) static_indices
  in
  let at_idcs = ref None in
  let has_setter = ref false in
  let top_tn = traced.tn in
  let check_idcs loop_ranges indices =
    (match !at_idcs with
    | None -> at_idcs := Some indices
    | Some at ->
        if not @@ [%equal: Indexing.axis_index array] at indices then raise @@ Non_virtual 4);
    (* gh-133 Stage A: repeated non-static symbols (diagonal [i;i] / partially-diagonal [i;j;i]) and
       covered single-symbol affine positions are supported. gh-133 Stage B: multi-symbol affine
       positions ([stride*oh+kh], [K*i+k], triangular [(s1, s1+s2)]) are also supported, but only when
       the whole LHS index map is proven injective over the producer loop widths -- otherwise dropping
       the producer loops in [inline_computation] would lose fold contributions. [Concat]
       ([Non_virtual 52]) remains rejected. *)
    let symbol_range s = Map.find loop_ranges s |> Option.value ~default:1 in
    (* Non-static symbols per position; a position with more than one non-static affine symbol is only
       admissible when the whole vector is injective. *)
    let has_multi_affine = ref false in
    let syms =
      Array.fold indices
        ~init:(Set.empty (module Indexing.Symbol))
        ~f:(fun acc -> function
          | Indexing.Fixed_idx _ | Indexing.Sub_axis -> acc
          | Indexing.Iterator s -> if Set.mem static_indices s then acc else Set.add acc s
          | Indexing.Affine { symbols; offset = _ } ->
              let nonstatic =
                List.filter_map symbols ~f:(fun (_, s) ->
                    Option.some_if (not @@ Set.mem static_indices s) s)
              in
              (match nonstatic with [] | [ _ ] -> () | _ -> has_multi_affine := true);
              List.fold nonstatic ~init:acc ~f:Set.add
          | Indexing.Concat _syms ->
              (* Concat indices should be eliminated before virtualization *)
              raise @@ Non_virtual 52)
    in
    let injective = lazy (Indexing.affine_injective ~symbol_range indices) in
    (* A multi-symbol affine position is sound to drop only if the LHS map is injective. *)
    if !has_multi_affine && not (Lazy.force injective) then raise @@ Non_virtual 51;
    (* Non-static symbols appearing in a bare [Iterator] position. [inline_computation]'s pass 1 binds
       only such positions from the call args; pass 2 grounds any affine occurrence via [subst]. *)
    let iter_syms =
      Set.of_array (module Indexing.Symbol)
      @@ Array.filter_map indices ~f:(function
           | Indexing.Iterator s when not @@ Set.mem static_indices s -> Some s
           | _ -> None)
    in
    (* Coverage check (replaces the old uniqueness check): every non-static symbol used must be
       groundable by [inline_computation]. A symbol that appears in a bare [Iterator] position is bound
       from the call args; otherwise (a symbol that occurs only inside affine positions, Stage B) the
       whole map must be affine-injective so its symbols are pinned/solvable. Repeated [Iterator]
       positions are allowed and produce equality guards. [inline_computation] re-validates per call
       site, so over-accepting here only risks a later [Non_virtual 13] fall back to materialization. *)
    if (not (Lazy.force injective)) && not (Set.is_subset syms ~of_:iter_syms) then
      raise @@ Non_virtual 5
  in
  (* A sibling tensor's access is fine as long as its indices are bound within the candidate's
     computation; only an escaping (unbound) non-static symbol disqualifies (see #134). gh-133 Stage B:
     inspect symbols hidden inside [Affine]/[Concat] too, not just bare [Iterator], so the relaxed
     affine path cannot admit an escaping symbol concealed in an affine index. *)
  let check_sibling_escaping ~env_dom ~code idcs =
    Array.iter idcs ~f:(fun idx ->
        let syms =
          match idx with
          | Indexing.Iterator s -> [ s ]
          | Indexing.Affine { symbols; _ } -> List.map symbols ~f:snd
          | Indexing.Concat syms -> syms
          | Indexing.Fixed_idx _ | Indexing.Sub_axis -> []
        in
        List.iter syms ~f:(fun s ->
            if (not (Set.mem static_indices s)) && not (Set.mem env_dom s) then (
              [%log2
                "INFO: Inlining candidate has an escaping variable",
                (idx : Indexing.axis_index),
                (top_llc : t)];
              raise @@ Non_virtual code)))
  in
  (* Traverse the float code too, for completeness / future use-cases. *)
  let rec loop_proc ~env_dom ~loop_ranges llc =
    let loop = loop_proc ~env_dom ~loop_ranges in
    match llc with
    | Noop -> ()
    | (Seq (c1, c2) : t) ->
        loop c1;
        loop c2
    | For_loop { trace_it = false; _ } -> raise @@ Non_virtual 6
    | For_loop { index; body; from_; to_; trace_it = true } ->
        loop_proc
          ~env_dom:(Set.add env_dom index)
          ~loop_ranges:(Map.set loop_ranges ~key:index ~data:(to_ - from_ + 1))
          body
    | Zero_out tn -> if Tn.equal tn top_tn then has_setter := true
    | Set { tn; idcs; llsc; debug = _ } ->
        if Tn.equal tn top_tn then (
          check_idcs loop_ranges idcs;
          has_setter := true)
        else check_sibling_escaping ~env_dom ~code:7 idcs;
        loop_scalar ~env_dom ~loop_ranges llsc
    | Set_from_vec { tn; idcs; length = _; vec_unop = _; arg = arg, _; debug = _ } ->
        if Tn.equal tn top_tn then (
          check_idcs loop_ranges idcs;
          has_setter := true)
        else check_sibling_escaping ~env_dom ~code:7 idcs;
        loop_scalar ~env_dom ~loop_ranges arg
    | Set_local (_, llsc) -> loop_scalar ~env_dom ~loop_ranges llsc
    | Declare_local _ -> raise @@ Non_virtual 19
    | Comment _ -> ()
    | Staged_compilation _ -> raise @@ Non_virtual 8
  and loop_scalar ~env_dom ~loop_ranges llsc =
    match llsc with
    | Constant _ | Constant_bits _ -> ()
    | Get (tn, idcs) ->
        if Tn.equal tn top_tn then check_idcs loop_ranges idcs
        else check_sibling_escaping ~env_dom ~code:9 idcs
    | Get_dynamic { dyn_value = v, _; _ } ->
        (* gh-343: defensive -- [Get_dynamic] is produced after virtualization analysis. *)
        loop_scalar ~env_dom ~loop_ranges v
    | Local_scope { body; _ } -> loop_proc ~env_dom ~loop_ranges body
    | Get_local _ -> ()
    | Get_merge_buffer (tn, idcs) ->
        if Tn.equal tn top_tn then check_idcs loop_ranges idcs
        else check_sibling_escaping ~env_dom ~code:9 idcs
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
    | Embed_index (Concat syms) ->
        List.iter syms ~f:(fun s ->
            if not @@ Set.mem env_dom s then (
              if not (Set.mem static_indices s) then
                [%log2
                  "Inlining candidate has an escaping variable",
                  (s : Indexing.symbol),
                  (top_llc : t)];
              raise @@ Non_virtual 10))
    | Ternop (_, (llv1, _), (llv2, _), (llv3, _)) ->
        loop_scalar ~env_dom ~loop_ranges llv1;
        loop_scalar ~env_dom ~loop_ranges llv2;
        loop_scalar ~env_dom ~loop_ranges llv3
    | Binop (_, (llv1, _), (llv2, _)) ->
        loop_scalar ~env_dom ~loop_ranges llv1;
        loop_scalar ~env_dom ~loop_ranges llv2
    | Unop (_, (llsc, _)) -> loop_scalar ~env_dom ~loop_ranges llsc
  in
  try
    if Tn.known_non_virtual traced.tn then raise @@ Non_virtual 11;
    loop_proc ~env_dom:static_indices ~loop_ranges:(Map.empty (module Indexing.Symbol)) top_llc;
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
  let computations =
    Hashtbl.find computations_table traced.tn
    |> Option.value_or_thunk ~default:(fun () ->
        raise
        @@ Utils.User_error
             [%string
               "Stale optimize_ctx: No computations found for #%{traced.tn.Tn.id#Int}: \
                %{Tn.debug_name traced.tn}"])
  in
  (* gh-133 Stage A: a guard's else-branch reads [Get_local id] -- the zero/init local produced by a
     [Zero_out] computation. If the producer has no [Zero_out] (e.g. a surjective producer that was
     previously materialized), that local is never initialized, so we must NOT emit a guard; such
     reads fall back to materialization via [Non_virtual 13] below, preserving prior behavior. *)
  let has_zero_init =
    let rec contains = function
      | Zero_out tn -> Tn.equal tn traced.tn
      | Seq (a, b) -> contains a || contains b
      | For_loop { body; _ } -> contains body
      | _ -> false
    in
    List.exists computations ~f:(fun (_, def) -> contains def)
  in
  (* In the order of computation. *)
  let loop_proc ((def_args : Indexing.axis_index array option), (def : t)) : t option =
    (* One substitution step: replace env-bound symbols, folding nested affine/fixed contributions. *)
    let subst_step env (idx : Indexing.axis_index) : Indexing.axis_index =
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
            | Some (Indexing.Concat _) ->
                (* Concat should not appear in affine substitution *)
                failwith "BUG: Concat in affine substitution not supported"
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
    (* gh-133 Stage B: a solved producer symbol can be bound to an affine expression that still
       mentions other producer symbols (e.g. [wh := t - 2*oh] when [oh]'s loop is kept). Those inner
       symbols are themselves env-bound (to a freshened loop variable), so substitution must be
       applied transitively to a fixpoint. Bindings are acyclic and finite, so this terminates. *)
    let rec subst env idx =
      let stepped = subst_step env idx in
      if Indexing.equal_axis_index stepped idx then stepped else subst env stepped
    in
    (* Canonical [(coeff, symbol) list, offset] view of an affine-like position. *)
    let canon idx =
      match idx with
      | Indexing.Iterator s -> Some ([ (1, s) ], 0)
      | Indexing.Affine { symbols; offset } -> Some (Indexing.coalesce_affine_terms symbols, offset)
      | Indexing.Fixed_idx i -> Some ([], i)
      | Indexing.Sub_axis | Indexing.Concat _ -> None
    in
    (* Per-symbol loop width of this producer computation, for range guards on solved symbols. *)
    let def_loop_ranges =
      let rec scan acc = function
        | For_loop { index; from_; to_; body; _ } ->
            scan (Map.set acc ~key:index ~data:(to_ - from_ + 1)) body
        | Seq (a, b) -> scan (scan acc a) b
        | _ -> acc
      in
      scan (Map.empty (module Indexing.Symbol)) def
    in
    let symbol_range s = Map.find def_loop_ranges s |> Option.value ~default:1 in
    (* gh-133 Stage A/B: build the substitution environment and guards in several passes, so a producer
       index vector that repeats a non-static symbol (diagonal [i;i] / partially-diagonal [i;j;i]) does
       not crash on duplicate keys, and so multi-symbol affine positions (Stage B) can be solved. *)
    let def_args_arr = Option.value def_args ~default:[||] in
    let n = Array.length def_args_arr in
    if n > Array.length call_args then
      failwith
        [%string
          "inline_computation: call_args too short, maybe stale optimization context? Tnode: \
           %{Tn.debug_name traced.tn} #%{traced.tn.Tn.id#Int} n: %{n#Int}"];
    (* [bound_pos.(i)] marks positions that DEFINE bindings (no consistency guard for them).
       [range_guards] collects [(solved_symbol, range)] pairs whose value must fall in [0, range). *)
    let bound_pos = Array.create ~len:n false in
    let env = ref (Map.empty (module Indexing.Symbol)) in
    let range_guards = ref [] in
    (* Pass 1: bind the first bare non-static [Iterator] occurrence of each symbol. *)
    Array.iteri def_args_arr ~f:(fun i lhs_ind ->
        match lhs_ind with
        | Indexing.Iterator s when (not (Set.mem static_indices s)) && not (Map.mem !env s) ->
            bound_pos.(i) <- true;
            env := Map.add_exn !env ~key:s ~data:call_args.(i)
        | _ -> ());
    (* gh-133 Stage B: structural affine match -- producer [Σ cₖ·sₖ + off] read at a call-site affine
       with the same canonical (distinct) coefficient list and equal offset binds producer symbols
       pairwise to the call-site symbols, no inversion and no guard. *)
    let try_structural_match i lhs_ind =
      if bound_pos.(i) then false
      else
        match (lhs_ind, canon call_args.(i)) with
        | Indexing.Affine { symbols = psyms; offset = poff }, Some (cterms, coff) when poff = coff ->
            let pterms = Indexing.coalesce_affine_terms psyms in
            let pcoeffs = List.map pterms ~f:fst in
            let ccoeffs = List.map cterms ~f:fst in
            let distinct l =
              List.length (List.dedup_and_sort ~compare:Int.compare l) = List.length l
            in
            let unbound (_, s) = (not (Set.mem static_indices s)) && not (Map.mem !env s) in
            if
              (not (List.is_empty pterms))
              && List.for_all pterms ~f:unbound && distinct pcoeffs
              && List.equal Int.equal
                   (List.sort ~compare:Int.compare pcoeffs)
                   (List.sort ~compare:Int.compare ccoeffs)
            then (
              List.iter pterms ~f:(fun (c, ps) ->
                  let _, cs = List.find_exn cterms ~f:(fun (cc, _) -> cc = c) in
                  env := Map.set !env ~key:ps ~data:(Indexing.Iterator cs));
              bound_pos.(i) <- true;
              true)
            else false
        | _ -> false
    in
    (* gh-133 Stage B: unit-coefficient solving -- after substituting already-bound symbols, if exactly
       one unbound producer symbol has coefficient ±1, bind it to the residual affine expression and
       emit a range guard. Other unbound symbols stay free; their producer loops are kept (and
       range-guarded indirectly via injectivity), guaranteeing exactly one matching iteration. *)
    let try_unit_solve i lhs_ind =
      if bound_pos.(i) then false
      else
        match lhs_ind with
        | Indexing.Affine { symbols; offset } -> (
            let terms = Indexing.coalesce_affine_terms symbols in
            let unbound =
              List.filter terms ~f:(fun (_, s) ->
                  (not (Set.mem static_indices s)) && not (Map.mem !env s))
            in
            match List.filter unbound ~f:(fun (c, _) -> abs c = 1) with
            | [ (uc, us) ] -> (
                match canon call_args.(i) with
                | None -> false
                | Some (rterms, roff) ->
                    (* us = uc * (rhs − offset − Σ_{other terms} c·s). uc = ±1 so uc⁻¹ = uc. Other
                       producer terms are left symbolic and resolved by [subst] (transitively). *)
                    let value_terms =
                      List.map rterms ~f:(fun (rc, rs) -> (uc * rc, rs))
                      @ List.filter_map terms ~f:(fun (c, s) ->
                            if Indexing.equal_symbol s us then None else Some (-uc * c, s))
                    in
                    let value =
                      Indexing.Affine
                        { symbols = value_terms; offset = uc * (roff - offset) }
                    in
                    env := Map.set !env ~key:us ~data:value;
                    (* Range guard [0 <= us < range], reformulated with NON-NEGATIVE operands so it is
                       correct under unsigned index precision (a negative [rhs - rest] would underflow):
                       [rest := Σ_{s≠us} c·s + offset], [rhs := call index]. uc=+1 needs
                       [rest <= rhs < rest+range]; uc=-1 needs [rhs <= rest < rhs+range]. *)
                    let rest_axis =
                      Indexing.Affine
                        {
                          symbols =
                            List.filter terms ~f:(fun (_, s) -> not (Indexing.equal_symbol s us));
                          offset;
                        }
                    in
                    range_guards :=
                      (uc, rest_axis, call_args.(i), symbol_range us) :: !range_guards;
                    bound_pos.(i) <- true;
                    true)
            | _ -> false)
        | _ -> false
    in
    (* Run the Stage B binding rounds to a fixpoint, in pinning order (structural match before
       unit-coefficient solving). *)
    let progress = ref true in
    while !progress do
      progress := false;
      Array.iteri def_args_arr ~f:(fun i lhs_ind ->
          if (not bound_pos.(i)) && (try_structural_match i lhs_ind || try_unit_solve i lhs_ind) then
            progress := true)
    done;
    let env = !env in
    (* Remaining non-binding positions become consistency guards: [subst(producer_pos) = call_site].
       [Fixed_idx]/[Sub_axis] carry no symbol, so a static mismatch there is a genuine sparse-producer
       mismatch and stays materialized via [Non_virtual 13]. Guards are materialized at the [Set] node
       with the live (freshened) env. *)
    let depends_on_symbol (idx : Indexing.axis_index) =
      match idx with
      | Indexing.Iterator s -> not (Set.mem static_indices s)
      | Indexing.Affine { symbols; _ } ->
          List.exists symbols ~f:(fun (_, s) -> not (Set.mem static_indices s))
      | Indexing.Fixed_idx _ | Indexing.Sub_axis -> false
      | Indexing.Concat _ -> false (* unreachable: rejected by check_idcs *)
    in
    let guards =
      Array.foldi def_args_arr ~init:[] ~f:(fun i guards lhs_ind ->
          if bound_pos.(i) then guards
          else
            let rhs_ind = call_args.(i) in
            let lhs' = subst env lhs_ind in
            if Indexing.equal_axis_index lhs' rhs_ind then guards
            else if depends_on_symbol lhs_ind then (lhs_ind, rhs_ind) :: guards
            else raise @@ Non_virtual 13)
    in
    let guards = List.rev guards in
    let range_guards = List.rev !range_guards in
    (* Set when a guard is introduced but the producer emitted no [Zero_out]: an explicit init local is
       then prepended before the (possibly loop-nested) guarded updates. *)
    let needs_init = ref false in
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
          let inlined = loop_scalar env llsc in
          let value_prec = Lazy.force traced.tn.Tn.prec in
          let index_prec = Ops.index_prec () in
          (* gh-133 Stage A: consistency (equality) guards for repeated / covered single-symbol affine
             positions -- the substituted producer index must equal the call-site index. Indices are
             resolved with the live (freshened) env so kept-loop symbols match the loop body. Index
             comparison uses index precision (Cmpeq is homogeneous); the [Where] keeps value precision
             on its then/else arms. *)
          let eq_conds =
            List.map guards ~f:(fun (lhs_ind, rhs_ind) ->
                Binop
                  ( Ops.Cmpeq,
                    (Embed_index (subst env lhs_ind), index_prec),
                    (Embed_index rhs_ind, index_prec) ))
          in
          (* gh-133 Stage B: range guards -- a unit-solved symbol's value must fall within its producer
             loop range [0, range). Index precision is UNSIGNED, so the guard must never form a negative
             intermediate: we compare [rest] and [rhs] (both non-negative) rather than [rhs - rest].
             uc=+1: [rest <= rhs] & [rhs < rest+range]; uc=-1: [rhs <= rest] & [rest < rhs+range].
             [a <= b] is encoded as [a < b+1]. *)
          let add_offset (idx : Indexing.axis_index) d : Indexing.axis_index =
            if d = 0 then idx
            else
              match idx with
              | Indexing.Iterator s -> Indexing.Affine { symbols = [ (1, s) ]; offset = d }
              | Indexing.Affine { symbols; offset } -> Indexing.Affine { symbols; offset = offset + d }
              | Indexing.Fixed_idx i -> Indexing.Fixed_idx (i + d)
              | Indexing.Sub_axis | Indexing.Concat _ -> idx
          in
          let lt a b =
            Binop (Ops.Cmplt, (Embed_index a, index_prec), (Embed_index b, index_prec))
          in
          let range_conds =
            List.map range_guards ~f:(fun (uc, rest_axis, rhs_axis, range) ->
                let rest = subst env rest_axis and rhs = subst env rhs_axis in
                let lower, upper =
                  if uc >= 0 then (lt rest (add_offset rhs 1), lt rhs (add_offset rest range))
                  else (lt rhs (add_offset rest 1), lt rest (add_offset rhs range))
                in
                Binop (Ops.And, (lower, index_prec), (upper, index_prec)))
          in
          let conds = eq_conds @ range_conds in
          (* Off-condition reads fall back to [Get_local id] -- the init local emitted by the producer's
             [Zero_out] ([has_zero_init]) or, when absent (an injective+surjective scatter that skipped
             neutral init -- Stage B), the explicit init prepended below. The no-[Zero_out] case implies
             the map is surjective, so every read cell IS written by exactly one iteration (injectivity)
             and the init value is always overwritten -- 0. is a safe neutral. *)
          if (not (List.is_empty conds)) && not has_zero_init then needs_init := true;
          let guarded =
            List.fold conds ~init:inlined ~f:(fun acc cond ->
                Ternop (Ops.Where, (cond, index_prec), (acc, value_prec), (Get_local id, value_prec)))
          in
          Some (Set_local (id, guarded))
      | Set_from_vec { tn; idcs; length = _; vec_unop = _; arg = _; debug = _ }
        when Tn.equal tn traced.tn ->
          assert ([%equal: Indexing.axis_index array option] (Some idcs) def_args);
          (* For vector operations, we cannot inline them as scalar operations *)
          raise @@ Non_virtual 140
      | Zero_out _ -> None
      | Set _ -> None
      | Set_from_vec _ -> None
      | Set_local (id, llsc) -> Some (Set_local (id, loop_scalar env llsc))
      | Declare_local _ -> None
      | Comment _ -> Some llc
      | Staged_compilation _ -> Some llc
    and loop_scalar env llsc : scalar_t =
      match llsc with
      | Constant _ | Constant_bits _ -> llsc
      | Get (tn, indices) when Tn.equal tn traced.tn ->
          assert ([%equal: Indexing.axis_index array option] (Some indices) def_args);
          Get_local id
      | Get (tn, indices) -> Get (tn, Array.map ~f:(subst env) indices)
      | Get_dynamic { tn; idcs; dyn_axis; dyn_value = v, prec } ->
          (* gh-343: defensive -- [Get_dynamic] is produced after virtualization. *)
          Get_dynamic
            { tn; idcs = Array.map ~f:(subst env) idcs; dyn_axis; dyn_value = (loop_scalar env v, prec) }
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
      | Ternop (op, (llv1, prec1), (llv2, prec2), (llv3, prec3)) ->
          Ternop
            ( op,
              (loop_scalar env llv1, prec1),
              (loop_scalar env llv2, prec2),
              (loop_scalar env llv3, prec3) )
      | Binop (op, (llv1, prec1), (llv2, prec2)) ->
          Binop (op, (loop_scalar env llv1, prec1), (loop_scalar env llv2, prec2))
      | Unop (op, (llsc, prec)) -> Unop (op, (loop_scalar env llsc, prec))
    in
    match loop env def with
    | Some body when !needs_init ->
        (* Prepend the init local before any (possibly loop-nested) guarded updates. *)
        Some (Seq (Set_local (id, Constant 0.0), body))
    | other -> other
  in
  try
    let body = List.rev_filter_map ~f:loop_proc computations in
    if List.is_empty body then raise @@ Non_virtual 14 else Some (unflat_lines body)
  with Non_virtual i ->
    Tn.update_memory_mode traced.tn Never_virtual i;
    None

let optimize_integer_pow = ref true

let rec unroll_pow ~(base : scalar_t) ~(exp : int) : scalar_t =
  if exp < 0 then
    unroll_pow
      ~base:(Binop (Div, (Constant 1., Ops.single), (base, scalar_precision base)))
      ~exp:(Int.neg exp)
  else if exp = 0 then Constant 1.
  else
    Fn.apply_n_times ~n:(exp - 1)
      (fun accu -> Binop (Mul, (base, scalar_precision base), (accu, scalar_precision accu)))
      base

let virtual_llc computations_table traced_store reverse_node_map static_indices (llc : t) : t =
  (* [process_for] holds tensors whose [Get]s must be left untouched (self/recursive references,
     replaced by [Get_local] during the tensor's own [inline_computation]). [owned] holds tensors
     whose whole-loop computation is captured at an enclosing [For_loop]: their per-statement
     auto-store is suppressed and they are excluded from nested candidate lists, but -- unlike
     [process_for] -- reads of them are still inlined, so surviving sibling readers can inline a
     virtualized provider. [in_storage_pass] is set within a per-candidate storage sub-pass so it
     does not recursively re-store nested-loop candidates. See #134. *)
  let rec loop_proc ~process_for ~owned ~in_storage_pass (llc : t) : t =
    let loop = loop_proc ~process_for ~owned ~in_storage_pass in
    match llc with
    | Noop -> Noop
    | Seq (c1, c2) ->
        let c1 = loop c1 in
        let c2 = loop c2 in
        Seq (c1, c2)
    | For_loop ({ index; body; _ } as for_config) ->
        if in_storage_pass then
          For_loop { for_config with body = loop_proc ~process_for ~owned ~in_storage_pass:true body }
        else
          let tns = Hashtbl.find reverse_node_map index |> Option.value ~default:[] in
          let candidates =
            (* First-seen (trace) order is preserved by [track_symbol], so a forward provider is
               stored before its consumer below. *)
            List.filter tns ~f:(fun tn ->
                (not @@ Set.mem process_for tn)
                && (not @@ Set.mem owned tn)
                && (not @@ Tn.known_non_virtual tn))
          in
          (match candidates with
          | [] ->
              For_loop
                { for_config with body = loop_proc ~process_for ~owned ~in_storage_pass:false body }
          | _ ->
              let owned' = List.fold candidates ~init:owned ~f:Set.add in
              (* Phase 1 -- store, sequentially in source order. For candidate [k], its stored loop is
                 processed with [process_for] containing [k] AND every later (not-yet-stored)
                 candidate. Keeping the not-yet-stored candidates in [process_for] leaves their [Get]s
                 intact, so a sibling setter (e.g. an in-loop materialized consumer) that reads a
                 later candidate does NOT trigger [inline_computation] before that candidate is stored
                 (which would raise the stale optimize_ctx error). Earlier candidates are already
                 stored and are left OUT, so [k]'s own setter can inline them (forward provider
                 chains). [owned'] suppresses per-statement auto-store for every shared-loop candidate;
                 [in_storage_pass] stops nested re-storage. [check_and_store]/[inline_computation]
                 filter the stored body to [k]'s own setters, so the irrelevant sibling setters left
                 un-rewritten here are dropped. *)
              List.iteri candidates ~f:(fun k tn ->
                  let node : traced_array = get_node traced_store tn in
                  let store_pf =
                    List.fold (List.drop candidates k) ~init:process_for ~f:Set.add
                  in
                  let stored =
                    For_loop
                      {
                        for_config with
                        body =
                          loop_proc ~process_for:store_pf ~owned:owned' ~in_storage_pass:true body;
                      }
                  in
                  check_and_store_virtual computations_table node static_indices stored);
              (* Phase 2 -- emit. Candidates are NOT in [process_for], so surviving readers
                 (materialized siblings, and later virtual siblings, all now stored) inline the
                 provider; [owned'] still suppresses candidate auto-store; each candidate setter
                 keeps its own self-references via the per-Set [next]. Candidate setters are emitted
                 intact and removed later by [cleanup_virtual_llc]. *)
              For_loop
                { for_config with body = loop_proc ~process_for ~owned:owned' ~in_storage_pass:false body })
    | Zero_out tn ->
        let traced : traced_array = get_node traced_store tn in
        if
          (not @@ Set.mem process_for tn)
          && (not @@ Set.mem owned tn)
          && (not @@ Tn.known_non_virtual traced.tn)
        then check_and_store_virtual computations_table traced static_indices llc;
        llc
    | Set { tn; idcs; llsc; debug } ->
        let traced : traced_array = get_node traced_store tn in
        let next = if Tn.known_non_virtual traced.tn then process_for else Set.add process_for tn in
        let result =
          Set { tn; idcs; llsc = loop_scalar ~process_for:next ~owned ~in_storage_pass llsc; debug }
        in
        if
          (not @@ Set.mem process_for tn)
          && (not @@ Set.mem owned tn)
          && (not @@ Tn.known_non_virtual traced.tn)
        then check_and_store_virtual computations_table traced static_indices result;
        result
    | Set_from_vec { tn; idcs; length; vec_unop; arg = arg_scalar, arg_prec; debug } ->
        let traced : traced_array = get_node traced_store tn in
        let next = if Tn.known_non_virtual traced.tn then process_for else Set.add process_for tn in
        let result =
          Set_from_vec
            {
              tn;
              idcs;
              length;
              vec_unop;
              arg = (loop_scalar ~process_for:next ~owned ~in_storage_pass arg_scalar, arg_prec);
              debug;
            }
        in
        if
          (not @@ Set.mem process_for tn)
          && (not @@ Set.mem owned tn)
          && (not @@ Tn.known_non_virtual traced.tn)
        then check_and_store_virtual computations_table traced static_indices result;
        result
    | Set_local (id, llsc) -> Set_local (id, loop_scalar ~process_for ~owned ~in_storage_pass llsc)
    | Declare_local _ -> llc
    | Comment _ -> llc
    | Staged_compilation _ -> llc
  and loop_scalar ~process_for ~owned ~in_storage_pass (llsc : scalar_t) : scalar_t =
    let loop = loop_scalar ~process_for ~owned ~in_storage_pass in
    match llsc with
    | Constant _ -> llsc
    | Constant_bits _ -> llsc
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
    | Get_dynamic { tn; idcs; dyn_axis; dyn_value = v, prec } ->
        Get_dynamic { tn; idcs; dyn_axis; dyn_value = (loop v, prec) }
    | Local_scope opts ->
        Local_scope
          {
            opts with
            body =
              loop_proc ~process_for:(Set.add process_for opts.id.tn) ~owned ~in_storage_pass
                opts.body;
          }
    | Get_local _ -> llsc
    | Get_merge_buffer (_, _) -> llsc
    | Embed_index _ -> llsc
    | Ternop (op, (llv1, prec1), (llv2, prec2), (llv3, prec3)) ->
        Ternop (op, (loop llv1, prec1), (loop llv2, prec2), (loop llv3, prec3))
    | Binop (op, (llv1, prec1), (llv2, prec2)) -> Binop (op, (loop llv1, prec1), (loop llv2, prec2))
    | Unop (op, (llsc, prec)) -> Unop (op, (loop llsc, prec))
  in
  loop_proc
    ~process_for:(Set.empty (module Tnode))
    ~owned:(Set.empty (module Tnode))
    ~in_storage_pass:false llc

let cleanup_virtual_llc ~static_indices (llc : t) : t =
  (* The current position is within scope of the definitions of the process_for virtual arrays. *)
  let rec loop_proc ~balanced ~env_dom (llc : t) : t option =
    let loop = loop_proc ~balanced ~env_dom in
    match llc with
    | Noop -> None
    | Seq _ ->
        let body = List.filter_map ~f:loop @@ flat_lines [ llc ] in
        if List.is_empty body then None else Some (unflat_lines body)
    | For_loop ({ index; body; _ } as for_config) ->
        (* Recurse into the loop body. A shared loop may compute several tensors: the per-statement
           cases below drop (and force [Virtual]) the setters of virtual tensors and keep those of
           non-virtual tensors, so we must not drop the whole loop just because its index has a
           virtual owner. The loop is elided only when its cleaned body is empty. See #134. *)
        let env_dom = Set.add env_dom index in
        Option.map ~f:(fun body : t -> For_loop { for_config with body })
        @@ loop_proc ~balanced ~env_dom body
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
          Some (Set { tn; idcs; llsc = loop_scalar ~balanced ~env_dom llsc; debug }))
    | Set_from_vec { tn; idcs; length; vec_unop; arg = arg_scalar, arg_prec; debug } ->
        if not @@ Tn.known_non_virtual tn then (
          (* FIXME(#296): *)
          Tn.update_memory_mode tn Virtual 152;
          None)
        else (
          assert (
            Array.for_all idcs ~f:(function Indexing.Iterator s -> Set.mem env_dom s | _ -> true));
          Some
            (Set_from_vec
               {
                 tn;
                 idcs;
                 length;
                 vec_unop;
                 arg = (loop_scalar ~balanced ~env_dom arg_scalar, arg_prec);
                 debug;
               }))
    | Set_local (id, llsc) ->
        assert (not @@ Tn.known_non_virtual id.tn);
        Tn.update_memory_mode id.tn Virtual 16;
        Some (Set_local (id, loop_scalar ~balanced ~env_dom llsc))
    | Declare_local _ -> Some llc
    | Comment _ -> Some llc
    | Staged_compilation _ -> Some llc
  and loop_scalar ~balanced ~env_dom (llsc : scalar_t) : scalar_t =
    let loop = loop_scalar ~balanced ~env_dom in
    match llsc with
    | Constant _ -> llsc
    | Constant_bits _ -> llsc
    | Get (a, indices) ->
        (* TODO(#296): this should probably already be Never_virtual, we could assert it. *)
        Tn.update_memory_mode a Never_virtual 17;
        assert (
          Array.for_all indices ~f:(function Indexing.Iterator s -> Set.mem env_dom s | _ -> true));
        llsc
    | Get_dynamic { tn; idcs; dyn_axis; dyn_value = v, prec } ->
        (* gh-343: defensive -- the table is a materialized read; recurse into the dynamic index. *)
        Tn.update_memory_mode tn Never_virtual 17;
        Get_dynamic { tn; idcs; dyn_axis; dyn_value = (loop v, prec) }
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
    | Embed_index (Concat syms) ->
        List.iter syms ~f:(fun s -> assert (Set.mem env_dom s));
        llsc
    | Ternop (op, (llv1, prec1), (llv2, prec2), (llv3, prec3)) ->
        Ternop (op, (loop llv1, prec1), (loop llv2, prec2), (loop llv3, prec3))
    | Binop (op, (llv1, prec1), (llv2, prec2)) -> Binop (op, (loop llv1, prec1), (loop llv2, prec2))
    | Unop (op, (llsc, prec)) -> Unop (op, (loop llsc, prec))
  in
  let static_indices =
    Set.of_list (module Indexing.Symbol)
    @@ List.map ~f:(fun s -> s.Indexing.static_symbol) static_indices
  in
  Option.value_exn ~here:[%here] @@ loop_proc ~balanced:false ~env_dom:static_indices llc

let rec substitute_float ~var ~value llsc =
  let loop_scalar = substitute_float ~var ~value in
  let loop_proc = substitute_proc ~var ~value in
  if equal_scalar_t var llsc then value
  else
    match llsc with
    | Constant _ -> llsc
    | Constant_bits _ -> llsc
    | Get (_ptr, _indices) -> llsc
    | Get_dynamic { tn; idcs; dyn_axis; dyn_value = v, prec } ->
        Get_dynamic { tn; idcs; dyn_axis; dyn_value = (loop_scalar v, prec) }
    | Local_scope opts -> Local_scope { opts with body = loop_proc opts.body }
    | Get_local _ -> llsc
    | Get_merge_buffer (_, _) -> llsc
    | Embed_index _ -> llsc
    | Ternop (op, (llv1, prec1), (llv2, prec2), (llv3, prec3)) ->
        Ternop (op, (loop_scalar llv1, prec1), (loop_scalar llv2, prec2), (loop_scalar llv3, prec3))
    | Binop (op, (llv1, prec1), (llv2, prec2)) ->
        Binop (op, (loop_scalar llv1, prec1), (loop_scalar llv2, prec2))
    | Unop (op, (llsc, prec)) -> Unop (op, (loop_scalar llsc, prec))

and substitute_proc ~var ~value llc =
  let loop_scalar = substitute_float ~var ~value in
  let loop_proc = substitute_proc ~var ~value in
  match llc with
  | Noop -> Noop
  | Seq (c1, c2) ->
      let c1 = loop_proc c1 in
      let c2 = loop_proc c2 in
      Seq (c1, c2)
  | For_loop for_config -> For_loop { for_config with body = loop_proc for_config.body }
  | Zero_out _ -> llc
  | Set { tn; idcs; llsc; debug } -> Set { tn; idcs; llsc = loop_scalar llsc; debug }
  | Set_from_vec { tn; idcs; length; vec_unop; arg = arg_scalar, arg_prec; debug } ->
      Set_from_vec { tn; idcs; length; vec_unop; arg = (loop_scalar arg_scalar, arg_prec); debug }
  | Set_local (id, llsc) -> Set_local (id, loop_scalar llsc)
  | Declare_local _ -> llc
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
    | Set { tn; idcs; llsc; debug } ->
        Set { tn; idcs; llsc = fst (loop_scalar (llsc, Lazy.force tn.Tn.prec)); debug }
    | Set_from_vec { tn; idcs; length; vec_unop; arg; debug } ->
        Set_from_vec { tn; idcs; length; vec_unop; arg = loop_scalar arg; debug }
    | Set_local (id, llsc) -> Set_local (id, fst (loop_scalar (llsc, Lazy.force id.tn.Tn.prec)))
    | Declare_local _ -> llc
    | Comment _ -> llc
    | Staged_compilation _ -> llc
  and loop_scalar ((llsc, prec) : scalar_t * Ops.prec) : scalar_t * Ops.prec =
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
    | Constant _ -> (llsc, prec)
    | Constant_bits _ -> (llsc, prec)
    | Get (tn, _indices) -> (llsc, Lazy.force tn.Tn.prec)
    | Get_dynamic { tn; idcs; dyn_axis; dyn_value = v, vprec } ->
        (* gh-343: defensive -- simplify runs before the one-hot rewrite, so this is unreachable in
           practice; still simplify the dynamic index sub-expression and never fold to a constant. *)
        let v', vprec' = loop_scalar (v, vprec) in
        (Get_dynamic { tn; idcs; dyn_axis; dyn_value = (v', vprec') }, Lazy.force tn.Tn.prec)
    | Local_scope { id; body = Set_local (id2, v); _ } when equal_scope_id id id2 ->
        ignore (Lazy.force id.tn.Tn.dims);
        loop_scalar (v, Lazy.force id.tn.Tn.prec)
    | Local_scope { id; body = Seq (Set_local (id1, v1), Set_local (id2, v2)); _ }
      when equal_scope_id id id1 && equal_scope_id id id2 ->
        ignore (Lazy.force id.tn.Tn.dims);
        let result = substitute_float ~var:(Get_local id) ~value:v1 v2 in
        loop_scalar (result, Lazy.force id.tn.Tn.prec)
    | Local_scope opts ->
        (Local_scope { opts with body = loop_proc local_scope_body }, Lazy.force opts.id.tn.Tn.prec)
    | Get_local id -> (llsc, Lazy.force id.tn.Tn.prec)
    | Get_merge_buffer (tn, _) -> (llsc, Lazy.force tn.Tn.prec)
    | Embed_index (Fixed_idx i) -> (Constant (Float.of_int i), prec)
    | Embed_index Sub_axis -> (Constant 0., prec)
    | Embed_index (Iterator _) -> (llsc, prec)
    | Embed_index (Affine _) -> (llsc, prec) (* Cannot simplify affine expressions to constants *)
    | Embed_index (Concat _) -> (llsc, prec) (* Cannot simplify concat to constants *)
    | Binop (Arg1, (llv1, prec1), _) -> loop_scalar (llv1, prec1)
    | Binop (Arg2, _, (llv2, prec2)) -> loop_scalar (llv2, prec2)
    | Binop ((Threefry4x32_crypto | Threefry4x32_light), _, _) -> (llsc, prec)
    | Binop (op, (Constant c1, prec1), (Constant c2, prec2)) ->
        (Constant (Ops.interpret_binop op c1 c2), Ops.promote_prec prec1 prec2)
    | Binop (Add, (llsc, prec1), (Constant 0., _))
    | Binop (Sub, (llsc, prec1), (Constant 0., _))
    | Binop (Add, (Constant 0., _), (llsc, prec1)) ->
        loop_scalar (llsc, prec1)
    | Binop (Sub, (Constant 0., _), (llsc, prec1)) ->
        loop_scalar (Binop (Mul, (Constant (-1.), prec1), (llsc, prec1)), prec1)
    | Binop (Mul, (llsc, prec1), (Constant 1., _))
    | Binop (Div, (llsc, prec1), (Constant 1., _))
    | Binop (Mul, (Constant 1., _), (llsc, prec1)) ->
        loop_scalar (llsc, prec1)
    | Binop (Mul, (_, prec1), (Constant 0., _))
    | Binop (Div, (Constant 0., _), (_, prec1))
    | Binop (Mul, (Constant 0., _), (_, prec1)) ->
        (Constant 0., prec1)
    | Binop
        ( Add,
          ( Binop (Add, (Constant c2, prec2), llsc), prec3
          | Binop (Add, llsc, (Constant c2, prec2)), prec3 ),
          (Constant c1, prec1) )
    | Binop
        ( Add,
          (Constant c1, prec1),
          ( Binop (Add, (Constant c2, prec2), llsc), prec3
          | Binop (Add, llsc, (Constant c2, prec2)), prec3 ) ) ->
        loop_scalar (Binop (Add, (Constant (c1 +. c2), Ops.promote_prec prec1 prec2), llsc), prec3)
    | Binop
        ( Sub,
          ( Binop (Add, (Constant c2, prec2), llsc), prec3
          | Binop (Add, llsc, (Constant c2, prec2)), prec3 ),
          (Constant c1, prec1) ) ->
        loop_scalar (Binop (Add, (Constant (c2 -. c1), Ops.promote_prec prec2 prec1), llsc), prec3)
    | Binop
        ( Sub,
          (Constant c1, prec1),
          ( Binop (Add, (Constant c2, prec2), llsc), prec3
          | Binop (Add, llsc, (Constant c2, prec2)), prec3 ) ) ->
        loop_scalar (Binop (Add, (Constant (c1 -. c2), Ops.promote_prec prec1 prec2), llsc), prec3)
    | Binop (Add, llv1, (Binop (Sub, llv2, llv3), prec3))
    | Binop (Add, (Binop (Sub, llv2, llv3), prec3), llv1) ->
        loop_scalar (Binop (Sub, (Binop (Add, llv1, llv2), prec), llv3), prec3)
    | Binop (Sub, llv1, (Binop (Sub, llv2, llv3), prec3)) ->
        loop_scalar (Binop (Sub, (Binop (Add, llv1, llv3), prec), llv2), prec3)
    | Binop (Sub, (Binop (Sub, llv1, llv2), prec1), llv3) ->
        loop_scalar (Binop (Sub, llv1, (Binop (Add, llv2, llv3), prec1)), prec1)
    | Binop
        ( Mul,
          ( Binop (Mul, (Constant c2, prec2), llsc), prec3
          | Binop (Mul, llsc, (Constant c2, prec2)), prec3 ),
          (Constant c1, prec1) )
    | Binop
        ( Mul,
          (Constant c1, prec1),
          ( Binop (Mul, (Constant c2, prec2), llsc), prec3
          | Binop (Mul, llsc, (Constant c2, prec2)), prec3 ) ) ->
        loop_scalar (Binop (Mul, (Constant (c1 *. c2), Ops.promote_prec prec1 prec2), llsc), prec3)
    | Binop
        ( Div,
          ( Binop (Mul, (Constant c2, prec2), llsc), prec3
          | Binop (Mul, llsc, (Constant c2, prec2)), prec3 ),
          (Constant c1, prec1) )
      when Ops.is_float prec ->
        loop_scalar (Binop (Mul, (Constant (c2 /. c1), Ops.promote_prec prec2 prec1), llsc), prec3)
    | Binop (Div, (Constant c1, prec1), (Binop (Mul, (Constant c2, prec2), llsc), prec3))
    | Binop (Div, (Constant c1, prec1), (Binop (Mul, llsc, (Constant c2, prec2)), prec3))
      when Ops.is_float prec ->
        (* TODO: this might worsen the conditioning in hand-designed formula cases. *)
        loop_scalar (Binop (Div, (Constant (c1 /. c2), Ops.promote_prec prec1 prec2), llsc), prec3)
    | Binop (Mul, llv1, (Binop (Div, llv2, llv3), prec23))
    | Binop (Mul, (Binop (Div, llv2, llv3), prec23), llv1)
      when Ops.is_float prec ->
        loop_scalar (Binop (Div, (Binop (Mul, llv1, llv2), prec), llv3), prec23)
    | Binop (Div, llv1, (Binop (Div, llv2, llv3), prec23)) when Ops.is_float prec ->
        loop_scalar (Binop (Div, (Binop (Mul, llv1, llv3), prec), llv2), prec23)
    | Binop (Div, (Binop (Div, llv1, llv2), prec12), llv3) when Ops.is_float prec ->
        loop_scalar (Binop (Div, (Binop (Mul, llv1, llv3), prec), llv2), prec12)
    | Binop (ToPowOf, llv1, llv2) -> (
        let ((v1_scalar, _) as v1) = loop_scalar llv1 in
        let v2 = loop_scalar llv2 in
        let result = (Binop (ToPowOf, v1, v2), prec) in
        if not !optimize_integer_pow then result
        else
          match v2 with
          | Constant c, _ when Float.is_integer c ->
              loop_scalar (unroll_pow ~base:v1_scalar ~exp:(Float.to_int c), prec)
          | _ -> result)
    | Binop (Add, (Binop (Mul, llv1, llv2), prec12), llv3)
    | Binop (Add, llv3, (Binop (Mul, llv1, llv2), prec12)) ->
        (* TODO: this is tentative. *)
        loop_scalar @@ (Ternop (FMA, llv1, llv2, llv3), Ops.promote_prec prec12 prec)
    | Binop (op, llv1, llv2) ->
        let v1 = loop_scalar llv1 in
        let v2 = loop_scalar llv2 in
        let result = (Binop (op, v1, v2), prec) in
        if equal_scalar_arg llv1 v1 && equal_scalar_arg llv2 v2 then result else loop_scalar result
    | Ternop
        (Where, (Binop (Cmpeq, (Embed_index a, _), (Embed_index b, _)), _), then_, _)
      when Indexing.equal_axis_index a b ->
        (* gh-133 Stage A: a repeated-symbol equality guard whose two embedded indices are
           syntactically identical is always taken; fold it to its then-branch. *)
        loop_scalar then_
    | Ternop (op, llv1, llv2, llv3) ->
        let v1 = loop_scalar llv1 in
        let v2 = loop_scalar llv2 in
        let v3 = loop_scalar llv3 in
        let result = (Ternop (op, v1, v2, v3), prec) in
        if equal_scalar_arg llv1 v1 && equal_scalar_arg llv2 v2 then result else loop_scalar result
    | Unop (Identity, llsc) -> loop_scalar llsc
    | Unop (op, (Constant c, _)) -> (Constant (Ops.interpret_unop op c), prec)
    | Unop (op, llsc) ->
        let v = loop_scalar llsc in
        let result = (Unop (op, v), prec) in
        if equal_scalar_arg llsc v then result else loop_scalar result
  in
  let check_constant tn c =
    (* Prevent triggering over-eager guard against forcing precision. *)
    ignore (Lazy.force tn.Tn.dims);
    if Ops.exceeds_fp16_cutoff c && Ops.is_up_to_fp16 (Lazy.force tn.Tn.prec) then
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
    | Set_from_vec { tn; arg = arg_scalar, _; _ } -> check_float tn arg_scalar
    | Set_local (id, llsc) -> check_float id.tn llsc
    | Declare_local _ | Noop | Comment _ | Staged_compilation _ -> ()
  and check_float tn llsc =
    let loop = check_float tn in
    match llsc with
    | Constant c -> check_constant tn c
    | Constant_bits _ -> () (* No check needed for bit constants *)
    | Local_scope { body; _ } -> check_proc body
    | Ternop (_, (v1, _), (v2, _), (v3, _)) ->
        loop v1;
        loop v2;
        loop v3
    | Binop (_, (v1, _), (v2, _)) ->
        loop v1;
        loop v2
    | Unop (_, (v, _)) -> loop v
    | Embed_index (Indexing.Fixed_idx i) -> check_constant tn (Float.of_int i)
    | Get_dynamic { dyn_value = v, _; _ } -> loop v
    | Embed_index _ | Get_local _ | Get_merge_buffer (_, _) | Get (_, _) -> ()
  in
  let result = loop_proc llc in
  if Option.is_some Utils.settings.check_half_prec_constants_cutoff then check_proc result;
  result

(** Alpha-equivalence comparison for CSE: compare two [scalar_t] trees ignoring concrete [scope_id]
    integers and fresh iterator symbols, but verifying cross-reference consistency via renaming
    maps. *)
let cse_equal_scalar s1 s2 =
  (* The renaming maps must be partial bijections, not just functions: alpha-equivalence requires an
     injective correspondence between bound variables. We therefore keep a reverse map alongside each
     forward map and reject when a target is already claimed by a different source (Bug 1: a
     forward-only map judged [t[i;j]] equal to [t[i;i]]). The maps are persistent for the whole
     comparison (no scope push/pop on entering [For_loop] / [Local_scope] binders): [Indexing.get_symbol]
     and [get_scope] are global counters, so symbols and scope ids are globally unique and no binder
     shadows another within a single tree. If the IR ever starts reusing symbol/scope ids, this
     assumption breaks and the maps would need scoping. *)
  let scope_renaming = Hashtbl.create (module Int) in
  let scope_renaming_rev = Hashtbl.create (module Int) in
  let sym_renaming = Hashtbl.create (module Indexing.Symbol) in
  let sym_renaming_rev = Hashtbl.create (module Indexing.Symbol) in
  let ids_equal (id1 : scope_id) (id2 : scope_id) =
    Tn.equal id1.tn id2.tn
    &&
    match
      (Hashtbl.find scope_renaming id1.scope_id, Hashtbl.find scope_renaming_rev id2.scope_id)
    with
    | Some mapped, _ -> Int.equal mapped id2.scope_id
    | None, Some _ -> false (* id2 already claimed by a different source scope id *)
    | None, None ->
        Hashtbl.set scope_renaming ~key:id1.scope_id ~data:id2.scope_id;
        Hashtbl.set scope_renaming_rev ~key:id2.scope_id ~data:id1.scope_id;
        true
  in
  let sym_equal (s1 : Indexing.symbol) (s2 : Indexing.symbol) =
    match (Hashtbl.find sym_renaming s1, Hashtbl.find sym_renaming_rev s2) with
    | Some mapped, _ -> Indexing.equal_symbol mapped s2
    | None, Some _ -> false (* s2 already claimed by a different source symbol *)
    | None, None ->
        Hashtbl.set sym_renaming ~key:s1 ~data:s2;
        Hashtbl.set sym_renaming_rev ~key:s2 ~data:s1;
        true
  in
  let idx_equal (i1 : Indexing.axis_index) (i2 : Indexing.axis_index) =
    match (i1, i2) with
    | Iterator s1, Iterator s2 -> sym_equal s1 s2
    | Fixed_idx n1, Fixed_idx n2 -> Int.equal n1 n2
    | Sub_axis, Sub_axis -> true
    | Affine { symbols = syms1; offset = o1 }, Affine { symbols = syms2; offset = o2 } ->
        Int.equal o1 o2
        && List.equal (fun (c1, s1) (c2, s2) -> Int.equal c1 c2 && sym_equal s1 s2) syms1 syms2
    | Concat ss1, Concat ss2 -> List.equal sym_equal ss1 ss2
    | _ -> false
  in
  let rec equal_t (a : t) (b : t) : bool =
    match (a, b) with
    | Noop, Noop -> true
    | Comment s1, Comment s2 -> String.equal s1 s2
    | Seq (a1, a2), Seq (b1, b2) -> equal_t a1 b1 && equal_t a2 b2
    | ( For_loop { index = i1; from_ = f1; to_ = t1; body = bd1; trace_it = tr1 },
        For_loop { index = i2; from_ = f2; to_ = t2; body = bd2; trace_it = tr2 } ) ->
        Int.equal f1 f2 && Int.equal t1 t2 && Bool.equal tr1 tr2 && sym_equal i1 i2
        && equal_t bd1 bd2
    | Zero_out tn1, Zero_out tn2 -> Tn.equal tn1 tn2
    | Set { tn = tn1; idcs = i1; llsc = s1; _ }, Set { tn = tn2; idcs = i2; llsc = s2; _ } ->
        Tn.equal tn1 tn2 && Array.equal idx_equal i1 i2 && equal_scalar s1 s2
    | Set_local (id1, s1), Set_local (id2, s2) -> ids_equal id1 id2 && equal_scalar s1 s2
    | Declare_local { id = id1; _ }, Declare_local { id = id2; _ } -> ids_equal id1 id2
    | _ -> false
  and equal_scalar (a : scalar_t) (b : scalar_t) : bool =
    match (a, b) with
    | ( Local_scope { id = id1; body = b1; orig_indices = oi1 },
        Local_scope { id = id2; body = b2; orig_indices = oi2 } ) ->
        (* Record the binder mapping through the checked path (Bug 3) before comparing the body, so
           the binder and its nested [Set_local] / [Get_local] uses all agree via [ids_equal]. *)
        ids_equal id1 id2 && Array.equal idx_equal oi1 oi2 && equal_t b1 b2
    | Get_local id1, Get_local id2 -> ids_equal id1 id2
    | Get (tn1, i1), Get (tn2, i2) -> Tn.equal tn1 tn2 && Array.equal idx_equal i1 i2
    | ( Get_dynamic { tn = tn1; idcs = i1; dyn_axis = da1; dyn_value = v1 },
        Get_dynamic { tn = tn2; idcs = i2; dyn_axis = da2; dyn_value = v2 } ) ->
        Tn.equal tn1 tn2 && Int.equal da1 da2 && Array.equal idx_equal i1 i2 && equal_arg v1 v2
    | Get_merge_buffer (tn1, i1), Get_merge_buffer (tn2, i2) ->
        Tn.equal tn1 tn2 && Array.equal idx_equal i1 i2
    | Ternop (op1, a1, a2, a3), Ternop (op2, b1, b2, b3) ->
        Ops.equal_ternop op1 op2 && equal_arg a1 b1 && equal_arg a2 b2 && equal_arg a3 b3
    | Binop (op1, a1, a2), Binop (op2, b1, b2) ->
        Ops.equal_binop op1 op2 && equal_arg a1 b1 && equal_arg a2 b2
    | Unop (op1, a1), Unop (op2, b1) -> Ops.equal_unop op1 op2 && equal_arg a1 b1
    | Constant c1, Constant c2 -> Float.equal c1 c2
    | Constant_bits i1, Constant_bits i2 -> Int64.equal i1 i2
    | Embed_index idx1, Embed_index idx2 -> idx_equal idx1 idx2
    | _ -> false
  and equal_arg ((s1, p1) : scalar_arg) ((s2, p2) : scalar_arg) : bool =
    Ops.equal_prec p1 p2 && equal_scalar s1 s2
  in
  equal_scalar s1 s2

(** Eliminates common subexpressions within each statement's scalar expression tree. Replaces
    duplicate [Local_scope] nodes (structurally identical modulo [scope_id]) with [Get_local]
    references to the first occurrence. *)
let eliminate_common_subexpressions llc =
  (* CSE operates within a single scalar expression tree per statement. *)
  let cse_scalar llsc =
    (* Association list: (representative Local_scope scalar, its scope_id) *)
    let seen : (scalar_t * scope_id) list ref = ref [] in
    let rec loop_scalar (llsc : scalar_t) : scalar_t =
      match llsc with
      | Local_scope { id; body; orig_indices } -> (
          (* Save seen list: inner definitions must not leak to sibling subtrees *)
          let saved_seen = !seen in
          (* First CSE within the body (bottom-up: inner scopes first) *)
          let body = loop_proc body in
          (* Restore: discard inner definitions, keep only those visible at this level *)
          seen := saved_seen;
          let result = Local_scope { id; body; orig_indices } in
          (* Search for an alpha-equivalent Local_scope already seen at this level *)
          let found =
            List.find_map !seen ~f:(fun (prev_scalar, prev_id) ->
                if cse_equal_scalar prev_scalar result then Some prev_id else None)
          in
          match found with
          | Some existing_id -> Get_local existing_id
          | None ->
              seen := (result, id) :: !seen;
              result)
      | Get_dynamic { tn; idcs; dyn_axis; dyn_value = v, prec } ->
          Get_dynamic { tn; idcs; dyn_axis; dyn_value = (loop_scalar v, prec) }
      | Get_local _ | Get _ | Get_merge_buffer _ | Constant _ | Constant_bits _ | Embed_index _ ->
          llsc
      | Ternop (op, (s1, p1), (s2, p2), (s3, p3)) ->
          Ternop (op, (loop_scalar s1, p1), (loop_scalar s2, p2), (loop_scalar s3, p3))
      | Binop (op, (s1, p1), (s2, p2)) -> Binop (op, (loop_scalar s1, p1), (loop_scalar s2, p2))
      | Unop (op, (s1, p1)) -> Unop (op, (loop_scalar s1, p1))
    and loop_proc (llc : t) : t =
      match llc with
      | Noop -> Noop
      | Comment _ | Staged_compilation _ | Zero_out _ -> llc
      | Seq (c1, c2) -> Seq (loop_proc c1, loop_proc c2)
      | For_loop for_config -> For_loop { for_config with body = loop_proc for_config.body }
      | Set { tn; idcs; llsc; debug } ->
          (* Each statement gets its own scope: codegen wraps in { } when local defs exist, so
             sibling statements can't reference each other's Local_scope declarations. *)
          let saved = !seen in
          let llsc = loop_scalar llsc in
          seen := saved;
          Set { tn; idcs; llsc; debug }
      | Set_from_vec { tn; idcs; length; vec_unop; arg = arg_scalar, arg_prec; debug } ->
          let saved = !seen in
          let arg_scalar = loop_scalar arg_scalar in
          seen := saved;
          Set_from_vec { tn; idcs; length; vec_unop; arg = (arg_scalar, arg_prec); debug }
      | Set_local (id, llsc) ->
          let saved = !seen in
          let llsc = loop_scalar llsc in
          seen := saved;
          Set_local (id, llsc)
      | Declare_local _ -> llc
    in
    loop_scalar llsc
  in
  let rec loop_proc (llc : t) : t =
    match llc with
    | Noop -> Noop
    | Comment _ | Staged_compilation _ | Zero_out _ | Declare_local _ -> llc
    | Seq (c1, c2) -> Seq (loop_proc c1, loop_proc c2)
    | For_loop for_config -> For_loop { for_config with body = loop_proc for_config.body }
    | Set { tn; idcs; llsc; debug } -> Set { tn; idcs; llsc = cse_scalar llsc; debug }
    | Set_from_vec { tn; idcs; length; vec_unop; arg = arg_scalar, arg_prec; debug } ->
        Set_from_vec { tn; idcs; length; vec_unop; arg = (cse_scalar arg_scalar, arg_prec); debug }
    | Set_local (id, llsc) -> Set_local (id, cse_scalar llsc)
  in
  loop_proc llc

(** Collect all top-level [Local_scope] nodes from a scalar expression tree. Returns a list of
    [(local_scope_scalar, scope_id)] pairs. Does not recurse into nested [Local_scope] bodies. *)
let collect_local_scopes_in_scalar (llsc : scalar_t) : (scalar_t * scope_id) list =
  let acc = ref [] in
  let rec loop (llsc : scalar_t) =
    match llsc with
    | Local_scope { id; _ } -> acc := (llsc, id) :: !acc
    | Get_dynamic { dyn_value = v, _; _ } -> loop v
    | Get_local _ | Get _ | Get_merge_buffer _ | Constant _ | Constant_bits _ | Embed_index _ -> ()
    | Ternop (_, (s1, _), (s2, _), (s3, _)) ->
        loop s1;
        loop s2;
        loop s3
    | Binop (_, (s1, _), (s2, _)) ->
        loop s1;
        loop s2
    | Unop (_, (s, _)) -> loop s
  in
  loop llsc;
  List.rev !acc

(** Collect all [Local_scope] candidates from a statement's scalar trees. *)
let collect_local_scopes_in_stmt (stmt : t) : (scalar_t * scope_id) list =
  match stmt with
  | Set { llsc; _ } -> collect_local_scopes_in_scalar llsc
  | Set_from_vec { arg = arg_scalar, _; _ } -> collect_local_scopes_in_scalar arg_scalar
  | Set_local (_, llsc) -> collect_local_scopes_in_scalar llsc
  | _ -> []

(** Replace all [Local_scope] nodes alpha-equivalent to [target] with [Get_local replacement] in a
    scalar expression tree. Also remaps [Get_local] nodes whose [scope_id] is in [stale_ids] to
    point to [replacement], since their original [Local_scope] is being hoisted. *)
let replace_local_scope_in_scalar ~target ~(replacement : scope_id) ~(stale_ids : scope_id list)
    (llsc : scalar_t) : scalar_t =
  let rec loop (llsc : scalar_t) : scalar_t =
    match llsc with
    | Local_scope _ -> if cse_equal_scalar llsc target then Get_local replacement else llsc
    | Get_local id ->
        if List.exists stale_ids ~f:(fun stale -> equal_scope_id id stale) then
          Get_local replacement
        else llsc
    | Get_dynamic { tn; idcs; dyn_axis; dyn_value = v, prec } ->
        Get_dynamic { tn; idcs; dyn_axis; dyn_value = (loop v, prec) }
    | Get _ | Get_merge_buffer _ | Constant _ | Constant_bits _ | Embed_index _ -> llsc
    | Ternop (op, (s1, p1), (s2, p2), (s3, p3)) ->
        Ternop (op, (loop s1, p1), (loop s2, p2), (loop s3, p3))
    | Binop (op, (s1, p1), (s2, p2)) -> Binop (op, (loop s1, p1), (loop s2, p2))
    | Unop (op, (s, p)) -> Unop (op, (loop s, p))
  in
  loop llsc

(** Replace matching [Local_scope] nodes in a statement's scalar children, and remap stale
    [Get_local] references. *)
let replace_local_scope_in_stmt ~target ~replacement ~stale_ids (stmt : t) : t =
  let repl = replace_local_scope_in_scalar ~target ~replacement ~stale_ids in
  match stmt with
  | Set { tn; idcs; llsc; debug } -> Set { tn; idcs; llsc = repl llsc; debug }
  | Set_from_vec { tn; idcs; length; vec_unop; arg = arg_scalar, arg_prec; debug } ->
      Set_from_vec { tn; idcs; length; vec_unop; arg = (repl arg_scalar, arg_prec); debug }
  | Set_local (id, llsc) -> Set_local (id, repl llsc)
  | other -> other

(** Collect all tensor nodes read via [Get(tn, _)] in a statement tree. *)
let reads_of_body (body : t) : Set.M(Tn).t =
  let acc = ref (Set.empty (module Tn)) in
  let rec loop_proc (llc : t) =
    match llc with
    | Noop | Comment _ | Staged_compilation _ | Zero_out _ | Declare_local _ -> ()
    | Seq (c1, c2) ->
        loop_proc c1;
        loop_proc c2
    | For_loop { body; _ } -> loop_proc body
    | Set { llsc; _ } -> loop_scalar llsc
    | Set_from_vec { arg = arg_scalar, _; _ } -> loop_scalar arg_scalar
    | Set_local (_, llsc) -> loop_scalar llsc
  and loop_scalar (llsc : scalar_t) =
    match llsc with
    | Get (tn, _) -> acc := Set.add !acc tn
    | Get_dynamic { tn; dyn_value = v, _; _ } ->
        (* gh-343: the table is read at [tn]; the dynamic index reads its own tensor inside
           [dyn_value], so recurse to count it too. *)
        acc := Set.add !acc tn;
        loop_scalar v
    | Get_merge_buffer (tn, _) -> acc := Set.add !acc tn
    | Local_scope { body; _ } -> loop_proc body
    | Get_local _ | Constant _ | Constant_bits _ | Embed_index _ -> ()
    | Ternop (_, (s1, _), (s2, _), (s3, _)) ->
        loop_scalar s1;
        loop_scalar s2;
        loop_scalar s3
    | Binop (_, (s1, _), (s2, _)) ->
        loop_scalar s1;
        loop_scalar s2
    | Unop (_, (s, _)) -> loop_scalar s
  in
  loop_proc body;
  !acc

(** Collect all tensor nodes written by a statement, recursing into [Seq] and [For_loop] bodies.

    The recursion into [For_loop] is load-bearing for hoisting safety (Bug 2): [flat_lines] keeps
    [For_loop] opaque, so [hoist_shared_locals]'s hazard check relies on this function to see writes
    performed *inside* a sibling loop sitting between two users of a hoisted [Local_scope]. A
    non-recursive version reported no writes for such a loop, which could permit an unsound hoist
    above it (later users would then read the pre-loop value). Recursing can only enlarge the hazard
    set, so it only ever narrows what is hoisted -- safe by construction. [Set_local] writes a
    [scope_id] local rather than a materialized [Tn], so it contributes nothing here. *)
let writes_of_stmt (stmt : t) : Set.M(Tn).t =
  let acc = ref (Set.empty (module Tn)) in
  let rec loop (s : t) =
    match s with
    | Set { tn; _ } | Set_from_vec { tn; _ } -> acc := Set.add !acc tn
    | Zero_out tn -> acc := Set.add !acc tn
    | Seq (a, b) ->
        loop a;
        loop b
    | For_loop { body; _ } -> loop body
    | Noop | Comment _ | Staged_compilation _ | Declare_local _ | Set_local _ -> ()
  in
  loop stmt;
  !acc

(** Returns [true] if the given [scope_id] is read (via [Get_local]) before the first definitely
    executed [Set_local] to that id in [body]. Used to decide whether a [Local_scope] or hoisted
    [Declare_local] needs a zero initializer. A loop body write is only considered definite when the
    loop bounds guarantee at least one iteration ([from_ <= to_]); reads inside loops always count
    conservatively. Nested [Local_scope] binders introduce distinct [scope_id]s, so there is no
    shadowing to handle. *)
let reads_scope_before_set (target : scope_id) (body : t) : bool =
  let rec scalar_has_read (llsc : scalar_t) : bool =
    match llsc with
    | Get_local id -> equal_scope_id id target
    | Local_scope { body; _ } -> proc_has_read body
    | Ternop (_, (s1, _), (s2, _), (s3, _)) ->
        scalar_has_read s1 || scalar_has_read s2 || scalar_has_read s3
    | Binop (_, (s1, _), (s2, _)) -> scalar_has_read s1 || scalar_has_read s2
    | Unop (_, (s, _)) -> scalar_has_read s
    | Get_dynamic { dyn_value = v, _; _ } -> scalar_has_read v
    | Get _ | Get_merge_buffer _ | Constant _ | Constant_bits _ | Embed_index _ -> false
  and proc_has_read (llc : t) : bool =
    match llc with
    | Noop | Comment _ | Staged_compilation _ | Zero_out _ | Declare_local _ -> false
    | Seq (a, b) -> proc_has_read a || proc_has_read b
    | For_loop { body; _ } -> proc_has_read body
    | Set { llsc; _ } -> scalar_has_read llsc
    | Set_from_vec { arg = s, _; _ } -> scalar_has_read s
    | Set_local (_, llsc) -> scalar_has_read llsc
  in
  (* Three-valued scan: Read (found a get before first definite set),
     Written (found a definite set before any get), Neither. *)
  let rec scan (llc : t) : [ `Read | `Written | `Neither ] =
    match llc with
    | Noop | Comment _ | Staged_compilation _ | Zero_out _ | Declare_local _ -> `Neither
    | Set { llsc; _ } -> if scalar_has_read llsc then `Read else `Neither
    | Set_from_vec { arg = s, _; _ } -> if scalar_has_read s then `Read else `Neither
    | Set_local (id, llsc) ->
        if scalar_has_read llsc then `Read
        else if equal_scope_id id target then `Written
        else `Neither
    | Seq (a, b) -> (
        match scan a with `Read -> `Read | `Written -> `Written | `Neither -> scan b)
    | For_loop { body; from_; to_; _ } -> (
        match scan body with
        | `Read -> `Read
        | `Written -> if from_ <= to_ then `Written else `Neither
        | `Neither -> `Neither)
  in
  match scan body with `Written -> false | `Read | `Neither -> true

(** Hoists shared [Local_scope] computations from sibling statements to the enclosing scope.
    Operates on a flat list of sibling statements. *)
let hoist_shared_locals (stmts : t list) : t list =
  (* Step 1: Collect all Local_scope candidates with their statement indices *)
  let candidates =
    List.concat_mapi stmts ~f:(fun stmt_idx stmt ->
        List.map (collect_local_scopes_in_stmt stmt) ~f:(fun (scalar, id) -> (stmt_idx, scalar, id)))
  in
  (* Step 2: Group by alpha-equivalence *)
  (* Each group is: (representative_scalar, representative_id, list of stmt indices,
     all scope_ids in the group) *)
  let groups : (scalar_t * scope_id * int list * scope_id list) list ref = ref [] in
  List.iter candidates ~f:(fun (stmt_idx, scalar, cand_id) ->
      let found =
        List.find_mapi !groups ~f:(fun group_idx (rep_scalar, _rep_id, _indices, _all_ids) ->
            if cse_equal_scalar rep_scalar scalar then Some group_idx else None)
      in
      match found with
      | Some group_idx ->
          groups :=
            List.mapi !groups ~f:(fun i (s, id, idxs, all_ids) ->
                if i = group_idx then (s, id, stmt_idx :: idxs, cand_id :: all_ids)
                else (s, id, idxs, all_ids))
      | None -> groups := (scalar, cand_id, [ stmt_idx ], [ cand_id ]) :: !groups);
  (* Keep only groups with 2+ members *)
  let shared_groups =
    List.filter_map !groups ~f:(fun (scalar, id, indices, all_ids) ->
        let indices = List.dedup_and_sort indices ~compare:Int.compare in
        if List.length indices >= 2 then Some (scalar, id, indices, all_ids) else None)
  in
  if List.is_empty shared_groups then stmts
  else
    (* Step 3: Safety check + rewrite *)
    let stmts = Array.of_list stmts in
    let insertions : (int * t list) list ref = ref [] in
    List.iter shared_groups ~f:(fun (target_scalar, canonical_id, user_indices, all_ids) ->
        let first_user = List.hd_exn user_indices in
        let last_user = List.last_exn user_indices in
        (* Collect reads of the Local_scope body *)
        let body_reads =
          match target_scalar with
          | Local_scope { body; _ } -> reads_of_body body
          | _ -> Set.empty (module Tn)
        in
        (* Check for writes between first_user and last_user that could invalidate hoisting. Include
           writes from ALL statements (including user statements) from first_user up to but not
           including last_user. User statements perform tensor writes after evaluating their
           Local_scope, so earlier users' writes can affect what later users would read. *)
        let safe =
          let hazard_writes = ref (Set.empty (module Tn)) in
          for i = first_user to last_user - 1 do
            hazard_writes := Set.union !hazard_writes (writes_of_stmt stmts.(i))
          done;
          Set.is_empty (Set.inter body_reads !hazard_writes)
        in
        if safe then (
          (* Extract body from canonical Local_scope *)
          let body =
            match target_scalar with Local_scope { body; _ } -> body | _ -> assert false
          in
          (* Replace all occurrences in all user statements, also remapping stale Get_local
             references that were created by intra-statement CSE pointing at Local_scope nodes that
             are now being hoisted away. *)
          List.iter user_indices ~f:(fun idx ->
              stmts.(idx) <-
                replace_local_scope_in_stmt ~target:target_scalar ~replacement:canonical_id
                  ~stale_ids:all_ids stmts.(idx));
          (* Record insertion: Declare_local + body before first user *)
          let needs_init = reads_scope_before_set canonical_id body in
          insertions :=
            (first_user, [ Declare_local { id = canonical_id; needs_init }; body ]) :: !insertions));
    (* Apply insertions (sorted by position, last first to preserve indices) *)
    let insertions = List.sort !insertions ~compare:(fun (a, _) (b, _) -> Int.descending a b) in
    let result = Array.to_list stmts in
    let result =
      List.fold insertions ~init:result ~f:(fun acc (pos, prefix) ->
          let before = List.take acc pos in
          let after = List.drop acc pos in
          before @ prefix @ after)
    in
    result

(** Hoists shared [Local_scope] computations from sibling statements to the enclosing scope. When
    two or more sibling statements share an alpha-equivalent [Local_scope] node, the computation is
    extracted as a [Declare_local] + body preceding the first user, and all occurrences are replaced
    with [Get_local]. *)
let hoist_cross_statement_cse llc =
  let rec loop_proc (llc : t) : t =
    match llc with
    | Seq _ ->
        let stmts = flat_lines [ llc ] in
        let stmts = List.map stmts ~f:loop_proc in
        hoist_shared_locals stmts |> unflat_lines
    | For_loop fc -> For_loop { fc with body = loop_proc fc.body }
    | _ -> llc
  in
  loop_proc llc

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

(* gh-343: helpers for the one-hot reduction rewrite. *)

let axis_index_mentions_symbol (s : Indexing.symbol) (idx : Indexing.axis_index) : bool =
  match idx with
  | Indexing.Iterator s' -> Indexing.equal_symbol s s'
  | Indexing.Affine { symbols; _ } ->
      List.exists symbols ~f:(fun (_, s') -> Indexing.equal_symbol s s')
  | Indexing.Concat syms -> List.exists syms ~f:(Indexing.equal_symbol s)
  | Indexing.Fixed_idx _ | Indexing.Sub_axis -> false

let rec scalar_mentions_symbol (s : Indexing.symbol) (llsc : scalar_t) : bool =
  match llsc with
  | Embed_index idx -> axis_index_mentions_symbol s idx
  | Get (_, idcs) | Get_merge_buffer (_, idcs) ->
      Array.exists idcs ~f:(axis_index_mentions_symbol s)
  | Get_dynamic { idcs; dyn_value = v, _; _ } ->
      Array.exists idcs ~f:(axis_index_mentions_symbol s) || scalar_mentions_symbol s v
  | Local_scope { orig_indices; body; _ } ->
      Array.exists orig_indices ~f:(axis_index_mentions_symbol s) || proc_mentions_symbol s body
  | Ternop (_, (v1, _), (v2, _), (v3, _)) ->
      scalar_mentions_symbol s v1 || scalar_mentions_symbol s v2 || scalar_mentions_symbol s v3
  | Binop (_, (v1, _), (v2, _)) -> scalar_mentions_symbol s v1 || scalar_mentions_symbol s v2
  | Unop (_, (v, _)) -> scalar_mentions_symbol s v
  | Get_local _ | Constant _ | Constant_bits _ -> false

and proc_mentions_symbol (s : Indexing.symbol) (llc : t) : bool =
  match llc with
  | Set { idcs; llsc; _ } ->
      Array.exists idcs ~f:(axis_index_mentions_symbol s) || scalar_mentions_symbol s llsc
  | Set_from_vec { idcs; arg = v, _; _ } ->
      Array.exists idcs ~f:(axis_index_mentions_symbol s) || scalar_mentions_symbol s v
  | Set_local (_, llsc) -> scalar_mentions_symbol s llsc
  | Seq (a, b) -> proc_mentions_symbol s a || proc_mentions_symbol s b
  | For_loop { body; _ } -> proc_mentions_symbol s body
  | Zero_out _ | Declare_local _ | Noop | Comment _ | Staged_compilation _ -> false

(* Count occurrences of [Iterator s] in [idcs], and report whether every occurrence is a plain
   [Iterator s] (no [Affine]/[Concat] use). Returns [(count, axis_of_last_plain_occurrence)]. *)
let count_plain_iterator (s : Indexing.symbol) (idcs : Indexing.axis_index array) :
    int * int option * bool =
  let count = ref 0 and axis = ref None and only_plain = ref true in
  Array.iteri idcs ~f:(fun i idx ->
      if axis_index_mentions_symbol s idx then (
        Int.incr count;
        match idx with
        | Indexing.Iterator _ -> axis := Some i
        | _ -> only_plain := false));
  (!count, !axis, !only_plain)

(* gh-343: recognize the in-range guard's reduction body. Matches the two semantically-equivalent
   one-hot selectors over loop variable [k]:
   - [Where (Cmpeq (Embed_index (Iterator k), index_expr), table_get, Constant 0.)] (either operand
     order of [Cmpeq]);
   - the multiply form [Binop (Mul, <cmpeq 0/1>, table_get)] (either factor order).
   On success returns [Some (table, table_idcs, index_expr)] where [index_expr] is the scalar value
   used as the dynamic index. *)
let match_one_hot_contribution (k : Indexing.symbol) (contribution : scalar_t) :
    (Tn.t * Indexing.axis_index array * scalar_arg) option =
  (* The range index appears either as a plain [Iterator k] or, after shape inference / reflection,
     as the unit affine [1*k + 0]. Accept both. *)
  let is_iter_k = function
    | Embed_index (Indexing.Iterator k') -> Indexing.equal_symbol k k'
    | Embed_index (Indexing.Affine { symbols = [ (1, k') ]; offset = 0 }) ->
        Indexing.equal_symbol k k'
    | _ -> false
  in
  (* Match a Cmpeq comparing [Embed_index (Iterator k)] against an index expression free of [k].
     Returns the index expression (with its precision). *)
  let match_cmpeq = function
    | Binop (Ops.Cmpeq, (a, pa), (b, pb)) ->
        if is_iter_k a && not (scalar_mentions_symbol k b) then Some (b, pb)
        else if is_iter_k b && not (scalar_mentions_symbol k a) then Some (a, pa)
        else None
    | _ -> None
  in
  let as_table_get = function
    | Get (table, table_idcs) -> Some (table, table_idcs)
    | _ -> None
  in
  match contribution with
  | Ternop (Ops.Where, cond, (then_, _), (Constant 0., _)) -> (
      match (match_cmpeq (fst cond), as_table_get then_) with
      | Some index_expr, Some (table, table_idcs) -> Some (table, table_idcs, index_expr)
      | _ -> None)
  | Binop (Ops.Mul, (a, _), (b, _)) -> (
      (* one factor is the comparison, the other is the table read *)
      match (match_cmpeq a, as_table_get b) with
      | Some index_expr, Some (table, table_idcs) -> Some (table, table_idcs, index_expr)
      | _ -> (
          match (match_cmpeq b, as_table_get a) with
          | Some index_expr, Some (table, table_idcs) -> Some (table, table_idcs, index_expr)
          | _ -> None))
  | _ -> None

(* gh-343: build the guarded dynamic gather replacing a matched one-hot reduction. [class_count] is
   the size of the table axis being gathered; [value_prec] is the table (result) precision. *)
let build_guarded_gather ~table ~table_idcs ~dyn_axis ~(index_expr : scalar_arg) ~class_count
    ~value_prec : scalar_t =
  let iv, _iprec = index_expr in
  (* The bounds comparison must NOT be evaluated in the unsigned index precision: [-1] would wrap to
     UINT_MAX and the guard would always be false (gh: unsigned-index-precision). We do the whole
     guard in a signed precision ([double], exact for the integer-valued indices in scope). [0 <= idx]
     is encoded as [-1 < idx] (there is no [Cmple]); the upper bound is [idx < class_count]. *)
  let guard_prec = Ops.double in
  let lower = Binop (Ops.Cmplt, (Constant (-1.), guard_prec), (iv, guard_prec)) in
  let upper = Binop (Ops.Cmplt, (iv, guard_prec), (Constant (Float.of_int class_count), guard_prec)) in
  let in_range = Binop (Ops.And, (lower, guard_prec), (upper, guard_prec)) in
  let index_prec = guard_prec in
  let gather = Get_dynamic { tn = table; idcs = table_idcs; dyn_axis; dyn_value = index_expr } in
  Ternop (Ops.Where, (in_range, index_prec), (gather, value_prec), (Constant 0., value_prec))

(* gh-343: peel a possible zero-initializer off a reduction body, returning the inner [For_loop]. *)
let strip_zero_init_for_local (id : scope_id) (body : t) : t option =
  match body with
  | Seq (Set_local (id', Constant 0.), (For_loop _ as fl)) when equal_scope_id id id' -> Some fl
  | For_loop _ -> Some body
  | _ -> None

(* gh-343: extract the per-iteration one-hot contribution from an accumulation [acc] in which the
   running total is recognized by [acc_is]. Handles the [Binop (Add, total, contribution)] form
   (either operand order) and the fused [Ternop (FMA, a, b, total)] form, where FMA(a,b,total) =
   a*b + total so the contribution is the product [a*b]. *)
let accumulation_contribution ~(acc_is : scalar_t -> bool) (acc : scalar_t) : scalar_t option =
  match acc with
  | Binop (Ops.Add, (total, _), (contribution, _)) when acc_is total -> Some contribution
  | Binop (Ops.Add, (contribution, _), (total, _)) when acc_is total -> Some contribution
  | Ternop (Ops.FMA, (a, pa), (b, pb), (total, _)) when acc_is total ->
      Some (Binop (Ops.Mul, (a, pa), (b, pb)))
  | _ -> None

(* gh-343: shared core -- given a reduction over [k] with bounds [\[from_, to_\]] and a per-iteration
   [contribution], check the narrow one-hot side conditions and build the guarded gather. *)
let gather_of_reduction ~(k : Indexing.symbol) ~from_ ~to_ (contribution : scalar_t) :
    scalar_t option =
  Option.bind (match_one_hot_contribution k contribution)
    ~f:(fun (table, table_idcs, index_expr) ->
      let count, axis, only_plain = count_plain_iterator k table_idcs in
      match if count = 1 && only_plain then axis else None with
      | None -> None
      | Some dyn_axis ->
          let class_count = from_ + (Lazy.force table.Tn.dims).(dyn_axis) in
          (* the loop must span exactly [0, class_count) over the gathered axis, and the index
             expression must be free of the reduction variable *)
          if (not (from_ = 0)) || to_ <> class_count - 1 then None
          else if scalar_mentions_symbol k (fst index_expr) then None
          else
            let value_prec = Lazy.force table.Tn.prec in
            (* Neutralize the now-dead loop symbol at the dynamic axis. *)
            let table_idcs = Array.copy table_idcs in
            table_idcs.(dyn_axis) <- Indexing.Fixed_idx 0;
            Some (build_guarded_gather ~table ~table_idcs ~dyn_axis ~index_expr ~class_count ~value_prec))

(* gh-343: scalar-local form -- Local_scope { id; body = [init;] For k { Set_local (id, acc) } }. *)
let try_rewrite_local_scope (id : scope_id) (body : t) : scalar_t option =
  match strip_zero_init_for_local id body with
  | Some (For_loop { index = k; from_; to_; body = Set_local (id', acc); _ })
    when equal_scope_id id id' ->
      let acc_is = function Get_local id' -> equal_scope_id id id' | _ -> false in
      Option.bind (accumulation_contribution ~acc_is acc)
        ~f:(fun contribution -> gather_of_reduction ~k ~from_ ~to_ contribution)
  | _ -> None

(* gh-343: materialized form -- a reduction loop [For k { Set lhs idcs acc }] at any nesting depth,
   where [acc] accumulates the one-hot contribution into [lhs\[idcs\]] (and [k] does not index
   [lhs]). The loop is replaced with a single read-accumulate of the guarded gather, dropping the
   vocabulary loop. Reading-and-adding [lhs\[idcs\]] keeps the rewrite sound regardless of any
   preceding zero-init: sum_k contribution == gather, so [lhs += gather] equals the original
   [lhs + sum_k contribution]. *)
let try_rewrite_materialized_loop (llc : t) : t option =
  match llc with
  | For_loop { index = k; from_; to_; body = Set { tn; idcs; llsc; _ }; _ }
    when not (Array.exists idcs ~f:(axis_index_mentions_symbol k)) ->
      let acc_is = function
        | Get (g, gi) -> Tn.equal g tn && [%equal: Indexing.axis_index array] gi idcs
        | _ -> false
      in
      Option.bind (accumulation_contribution ~acc_is llsc) ~f:(fun contribution ->
          Option.map (gather_of_reduction ~k ~from_ ~to_ contribution) ~f:(fun gather ->
              let value_prec = scalar_precision gather in
              Set
                {
                  tn;
                  idcs;
                  llsc = Binop (Ops.Add, (Get (tn, idcs), value_prec), (gather, value_prec));
                  debug = "";
                }))
  | _ -> None

let rewrite_one_hot_reductions (llc : t) : t =
  let rec loop_proc (llc : t) : t =
    match try_rewrite_materialized_loop llc with
    | Some replacement -> replacement
    | None -> (
        match llc with
        | Seq (a, b) -> Seq (loop_proc a, loop_proc b)
        | For_loop fc -> For_loop { fc with body = loop_proc fc.body }
        | Set { tn; idcs; llsc; debug } -> Set { tn; idcs; llsc = loop_scalar llsc; debug }
        | Set_from_vec { tn; idcs; length; vec_unop; arg = s, p; debug } ->
            Set_from_vec { tn; idcs; length; vec_unop; arg = (loop_scalar s, p); debug }
        | Set_local (id, llsc) -> Set_local (id, loop_scalar llsc)
        | (Noop | Comment _ | Staged_compilation _ | Zero_out _ | Declare_local _) as other -> other)
  and loop_scalar (llsc : scalar_t) : scalar_t =
    match llsc with
    | Local_scope { id; body; orig_indices } -> (
        (* Recurse into the body first so inner reductions are handled, then try to collapse this
           scope itself. *)
        let body = loop_proc body in
        match try_rewrite_local_scope id body with
        | Some gather -> gather
        | None -> Local_scope { id; body; orig_indices })
    | Get_dynamic { tn; idcs; dyn_axis; dyn_value = v, p } ->
        Get_dynamic { tn; idcs; dyn_axis; dyn_value = (loop_scalar v, p) }
    | Ternop (op, (a, pa), (b, pb), (c, pc)) ->
        Ternop (op, (loop_scalar a, pa), (loop_scalar b, pb), (loop_scalar c, pc))
    | Binop (op, (a, pa), (b, pb)) -> Binop (op, (loop_scalar a, pa), (loop_scalar b, pb))
    | Unop (op, (a, pa)) -> Unop (op, (loop_scalar a, pa))
    | (Get_local _ | Get _ | Get_merge_buffer _ | Constant _ | Constant_bits _ | Embed_index _) as
      other ->
        other
  in
  loop_proc llc

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
    hoist_cross_statement_cse @@ eliminate_common_subexpressions @@ rewrite_one_hot_reductions
    @@ simplify_llc
    @@ cleanup_virtual_llc ~static_indices
    @@ virtual_llc_result
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
        loop_scalar llsc
    | Set_from_vec { tn; arg = arg_scalar, _; _ } ->
        visit tn;
        loop_scalar arg_scalar
    | Set_local ({ tn; _ }, llsc) ->
        visit tn;
        loop_scalar llsc
    | Declare_local { id = { tn; _ }; _ } -> visit tn
  and loop_scalar fc =
    match fc with
    | Local_scope { id = { tn; _ }; body; orig_indices = _ } ->
        visit tn;
        loop body
    | Get_merge_buffer (la, _) -> visit la
    | Get (la, _) -> visit la
    | Get_dynamic { tn; dyn_value = v, _; _ } ->
        visit tn;
        loop_scalar v
    | Ternop (_, (f1, _), (f2, _), (f3, _)) ->
        loop_scalar f1;
        loop_scalar f2;
        loop_scalar f3
    | Binop (_, (f1, _), (f2, _)) ->
        loop_scalar f1;
        loop_scalar f2
    | Unop (_, (f, _)) -> loop_scalar f
    | Get_local { tn; _ } -> visit tn
    | Constant _ | Constant_bits _ | Embed_index _ -> ()
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
  let doc_ident la =
    let base = string (ident_label la) in
    if Utils.get_global_flag ~default:false ~arg_name:"output_prec_in_ll_files" then
      let prec_str = Ops.prec_string (Lazy.force la.prec) in
      base ^^ string ("<" ^ prec_str ^ ">")
    else base
  in
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
        let arg_scalar, _arg_prec = p.arg in
        let vec_result = string prefix ^^ doc_of_float Ops.Void_prec arg_scalar ^^ string postfix in
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
    | Declare_local { id; _ } -> group (string "declare " ^^ doc_local id ^^ string ";")
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
    | Get_dynamic { tn; idcs; dyn_axis; dyn_value = v, vprec } ->
        group
          (doc_ident tn ^^ brackets (pp_indices idcs)
          ^^ string (Printf.sprintf "@dyn[%d]=" dyn_axis)
          ^^ parens (doc_of_float vprec v))
    | Constant c -> string (Printf.sprintf "%.16g" c)
    | Constant_bits i -> string (Printf.sprintf "0x%LX" i)
    | Embed_index idx ->
        let idx_doc = pp_axis_index idx in
        if PPrint.is_empty idx_doc then string "0" else idx_doc
    | Ternop (op, (v1, _), (v2, _), (v3, _)) ->
        let prefix, comma1, comma2, postfix = Ops.ternop_c_syntax prec op in
        group
          (string prefix ^^ doc_of_float prec v1 ^^ string comma1 ^^ space ^^ doc_of_float prec v2
         ^^ string comma2 ^^ space ^^ doc_of_float prec v3 ^^ string postfix)
    | Binop (Arg1, (v1, _), _v2) -> doc_of_float prec v1
    | Binop (Arg2, _v1, (v2, _)) -> doc_of_float prec v2
    | Binop (op, (v1, _), (v2, _)) ->
        let prefix, infix, postfix = Ops.binop_c_syntax prec op in
        group
          (string prefix ^^ doc_of_float prec v1 ^^ string infix ^^ space ^^ doc_of_float prec v2
         ^^ string postfix)
    | Unop (Identity, (v, _)) -> doc_of_float prec v
    | Unop (op, (v, _)) ->
        let prefix, postfix = Ops.unop_c_syntax prec op in
        string prefix ^^ doc_of_float prec v ^^ string postfix
  in
  hardline ^^ nest 2 (function_header_doc ?name ?static_indices () ^^ doc_of_code llc)

let to_doc ?name ?static_indices () llc =
  let ident_label = get_ident_within_code [| llc |] in
  let open PPrint in
  let doc_ident la =
    let base = string (ident_label la) in
    if Utils.get_global_flag ~default:false ~arg_name:"output_prec_in_ll_files" then
      let prec_str = Ops.prec_string (Lazy.force la.prec) in
      base ^^ string ("<" ^ prec_str ^ ">")
    else base
  in
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
            ^^ string "("
            ^^ doc_of_float (fst p.arg)
            ^^ string ", " ^^ length_doc ^^ string ");")
        in
        let b = Buffer.create 100 in
        PPrint.ToBuffer.pretty 0.7 100 b result;
        p.debug <- Buffer.contents b;
        result
    | Comment message -> string ("/* " ^ message ^ " */")
    | Staged_compilation callback -> callback ()
    | Set_local (id, llsc) ->
        group (doc_local id ^^ string " := " ^^ doc_of_float llsc ^^ string ";")
    | Declare_local { id; _ } -> group (string "declare " ^^ doc_local id ^^ string ";")
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
    | Get_dynamic { tn; idcs; dyn_axis; dyn_value = v, _ } ->
        group
          (doc_ident tn ^^ brackets (pp_indices idcs)
          ^^ string (Printf.sprintf "@dyn[%d]=" dyn_axis)
          ^^ parens (doc_of_float v))
    | Constant c -> string (Printf.sprintf "%.16g" c)
    | Constant_bits i -> string (Printf.sprintf "0x%LX" i)
    | Embed_index idx ->
        let idx_doc = pp_axis_index idx in
        if PPrint.is_empty idx_doc then string "0" else idx_doc
    | Ternop (op, (v1, _), (v2, _), (v3, _)) ->
        let prefix = Ops.ternop_cd_syntax op in
        group
          (string prefix
          ^^ parens
               (doc_of_float v1 ^^ string "," ^^ space ^^ doc_of_float v2 ^^ string "," ^^ space
              ^^ doc_of_float v3))
    | Binop (Arg1, (v1, _), _v2) -> doc_of_float v1
    | Binop (Arg2, _v1, (v2, _)) -> doc_of_float v2
    | Binop (op, (v1, _), (v2, _)) ->
        if Ops.is_binop_nice_infix op then
          let infix = Ops.binop_cd_syntax op in
          group (parens (doc_of_float v1 ^^ space ^^ string infix ^^ space ^^ doc_of_float v2))
        else
          let prefix = Ops.binop_cd_fallback_syntax op in
          group (string prefix ^^ parens (doc_of_float v1 ^^ string "," ^^ space ^^ doc_of_float v2))
    | Unop (Identity, (v, _)) -> doc_of_float v
    | Unop (op, (v, _)) ->
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

let loop_over_padding_region ~dims ~(padding : Ops.axis_padding array) ~body =
  (* Generate loops that iterate ONLY over the padding margins (NOT the data region).

     The padding region is the union of "strips" where at least one dimension's index is in the
     padding range [0, left) or [dim-right, dim).

     For each dimension with padding, we generate: 1. Left padding strip: index in [0, left) -
     iterate ALL remaining dims 2. Middle: index in [left, dim-right) - recurse to find padding in
     other dims 3. Right padding strip: index in [dim-right, dim) - iterate ALL remaining dims

     For dimensions with NO padding, we just iterate the full range while recursing.

     The recursion stops when we've processed all dimensions. If we reach the end without any
     dimension having contributed padding, we DON'T call body (that's data). *)
  let rec build_loops ~any_padding_so_far dim_idx rev_idcs =
    if dim_idx >= Array.length dims then
      (* Only generate body if we're actually in a padding region *)
      if any_padding_so_far then body @@ Array.of_list_rev rev_idcs else Noop
    else
      let dim = dims.(dim_idx) in
      let pad = padding.(dim_idx) in
      let index = Indexing.get_symbol () in
      let has_padding = pad.left > 0 || pad.right > 0 in
      if not has_padding then
        (* No padding on this dimension - iterate full range, keep looking for padding *)
        For_loop
          {
            index;
            from_ = 0;
            to_ = dim - 1;
            body =
              build_loops ~any_padding_so_far (dim_idx + 1) (Indexing.Iterator index :: rev_idcs);
            trace_it = true;
          }
      else
        (* Has padding - generate left strip, middle (recurse), right strip *)
        let left_loop =
          if pad.left > 0 then
            For_loop
              {
                index;
                from_ = 0;
                to_ = pad.left - 1;
                body =
                  (* In left padding - iterate ALL remaining dims (they're all in padding region) *)
                  loop_over_dims
                    (Array.sub dims ~pos:(dim_idx + 1) ~len:(Array.length dims - dim_idx - 1))
                    ~body:(fun rest_idcs ->
                      body
                      @@ Array.concat
                           [ Array.of_list_rev rev_idcs; [| Indexing.Iterator index |]; rest_idcs ]);
                trace_it = true;
              }
          else Noop
        in
        let middle_loop =
          let middle_from = pad.left in
          let middle_to = dim - pad.right - 1 in
          if middle_from <= middle_to then
            For_loop
              {
                index;
                from_ = middle_from;
                to_ = middle_to;
                body =
                  (* In middle - NOT in padding for this dim, recurse to find other padded dims *)
                  build_loops ~any_padding_so_far (dim_idx + 1) (Indexing.Iterator index :: rev_idcs);
                trace_it = true;
              }
          else Noop
        in
        let right_loop =
          if pad.right > 0 then
            let right_index = Indexing.get_symbol () in
            For_loop
              {
                index = right_index;
                from_ = dim - pad.right;
                to_ = dim - 1;
                body =
                  (* In right padding - iterate ALL remaining dims *)
                  loop_over_dims
                    (Array.sub dims ~pos:(dim_idx + 1) ~len:(Array.length dims - dim_idx - 1))
                    ~body:(fun rest_idcs ->
                      body
                      @@ Array.concat
                           [
                             Array.of_list_rev rev_idcs;
                             [| Indexing.Iterator right_index |];
                             rest_idcs;
                           ]);
                trace_it = true;
              }
          else Noop
        in
        unflat_lines [ left_loop; middle_loop; right_loop ]
  in
  build_loops ~any_padding_so_far:false 0 []
