open Base

module Lazy = Utils.Lazy
(** The code for operating on n-dimensional arrays. *)

module Nd = Ndarray
module Tn = Tnode
module Debug_runtime = Utils.Debug_runtime

[%%global_debug_log_level Nothing]
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

(** *** Low-level representation. *)

let get_scope =
  let uid = ref 0 in
  fun tn ->
    Int.incr uid;
    { tn; scope_id = !uid }

(** Cases: [t] -- code, [float_t] -- single number at some precision. *)
type t =
  | Noop
  | Comment of string
  | Staged_compilation of ((unit -> unit)[@equal.ignore] [@compare.ignore])
  | Seq of t * t
  | For_loop of { index : Indexing.symbol; from_ : int; to_ : int; body : t; trace_it : bool }
  | Zero_out of Tn.t
  | Set of { tn : Tn.t; idcs : Indexing.axis_index array; llv : float_t; mutable debug : string }
  | Set_local of scope_id * float_t
[@@deriving sexp_of, equal]

and float_t =
  | Local_scope of { id : scope_id; body : t; orig_indices : Indexing.axis_index array }
  | Get_local of scope_id
  | Get_global of Ops.global_identifier * Indexing.axis_index array option
  | Get of Tn.t * Indexing.axis_index array
  | Binop of Ops.binop * float_t * float_t
  | Unop of Ops.unop * float_t
  | Constant of float
  | Embed_index of Indexing.axis_index
[@@deriving sexp_of, equal, compare]

let binop ~op ~rhs1 ~rhs2 =
  match op with Ops.Arg1 -> rhs1 | Arg2 -> rhs2 | _ -> Binop (op, rhs1, rhs2)

let unop ~op ~rhs = match op with Ops.Identity -> rhs | _ -> Unop (op, rhs)

let rec flat_lines ts =
  List.concat_map ts ~f:(function Seq (t1, t2) -> flat_lines [ t1; t2 ] | t -> [ t ])

let rec unflat_lines = function
  | [] -> Noop
  | [ llc ] -> llc
  | Noop :: tl -> unflat_lines tl
  | llc :: tl -> Seq (llc, unflat_lines tl)

let comment_to_name =
  let nonliteral = Str.regexp {|[^a-zA-Z0-9_]|} in
  Str.global_replace nonliteral "_"

(** *** Optimization *** *)

type virtualize_settings = {
  mutable enable_device_only : bool;
  mutable max_visits : int;
  mutable max_tracing_dim : int;
  mutable inline_scalar_constexprs : bool;
}

let virtualize_settings =
  let max_visits =
    Int.of_string @@ Utils.get_global_arg ~arg_name:"virtualize_max_visits" ~default:"1"
  in
  let max_tracing_dim =
    Int.of_string @@ Utils.get_global_arg ~arg_name:"virtualize_max_tracing_dim" ~default:"5"
  in
  let enable_device_only =
    Bool.of_string @@ Utils.get_global_arg ~arg_name:"enable_device_only" ~default:"true"
  in
  let inline_scalar_constexprs =
    Bool.of_string @@ Utils.get_global_arg ~arg_name:"inline_scalar_constexprs" ~default:"true"
  in
  { enable_device_only; max_visits; max_tracing_dim; inline_scalar_constexprs }

type visits =
  | Visits of int
  | Recurrent
      (** A [Recurrent] visit is when there is an access prior to any assignment in an update. *)
[@@deriving sexp, equal, variants]

type traced_array = {
  tn : Tn.t;
  mutable computations : (Indexing.axis_index array option * t) list;
      (** The computations (of the tensor node) are retrieved for optimization just as they are
          populated, so that the inlined code corresponds precisely to the changes to the arrays
          that would happen up till that point. Within the code blocks paired with an index tuple,
          all assignments and accesses must happen via the index tuple; if this is not the case for
          some assignment, the node cannot be virtual. Currently, we only allow for-loop symbols in
          assignment indices of virtual nodes. *)
  assignments : int array Hash_set.t;
  accesses : (int array, visits) Hashtbl.t;
      (** For dynamic indexes, we take a value of 0. This leads to an overestimate of visits, which
          is safe. *)
  mutable zero_initialized : bool;
  mutable zeroed_out : bool;
  mutable read_before_write : bool;
      (** The node is read before it is written (i.e. it is recurrent). *)
  mutable read_only : bool;
  mutable is_scalar_constexpr : bool;
      (** True only if the tensor node has all axes of dimension 1, is either zeroed-out or assigned
          before accessed, is assigned at most once, and from an expression involving only constants
          or tensor nodes that were at the time is_scalar_constexpr. *)
}
[@@deriving sexp_of]

let get_node store tn =
  Hashtbl.find_or_add store tn ~default:(fun () ->
      {
        tn;
        computations = [];
        assignments = Hash_set.Poly.create ();
        accesses = Hashtbl.Poly.create ();
        zero_initialized = false;
        zeroed_out = false;
        read_before_write = false;
        read_only = false;
        is_scalar_constexpr = false;
      })

let partition_tf_with_comment cs ~f =
  let both = Array.map cs ~f:(fun c -> if f c then Either.First c else Either.Second c) in
  let trues =
    Array.filter_map both ~f:(function
      | First x -> Some x
      | Second (Comment _ as x) -> Some x
      | Second _ -> None)
  in
  let falses =
    Array.filter_map both ~f:(function
      | First (Comment _ as x) -> Some x
      | First _ -> None
      | Second x -> Some x)
  in
  (trues, falses)

let visit ~is_assigned old =
  if not is_assigned then Recurrent
  else
    match old with
    | None -> Visits 1
    | Some (Visits i) -> Visits (i + 1)
    | Some Recurrent -> Recurrent

let is_constexpr_comp traced_store llv =
  let rec loop llv =
    match llv with
    | Get_local { tn; _ } | Local_scope { id = { tn; _ }; _ } ->
        let traced = get_node traced_store tn in
        traced.is_scalar_constexpr
    | Get_global (_, _) -> false
    | Get (tn, _idcs) ->
        let traced = get_node traced_store tn in
        traced.is_scalar_constexpr
    | Binop (_, v1, v2) -> loop v1 && loop v2
    | Unop (_, v) -> loop v
    | Constant _ -> true
    | Embed_index _ -> false
  in
  loop llv

let is_scalar_dims tn = Array.for_all ~f:(( = ) 1) @@ Lazy.force tn.Tn.dims

let visit_llc traced_store ~merge_node_id reverse_node_map ~max_visits llc =
  let is_too_many = function Visits i -> i > max_visits | Recurrent -> true in
  let lookup env indices =
    Array.map indices ~f:(function
      | Indexing.Fixed_idx i -> i
      | Iterator s -> Option.value ~default:(* static index *) 0 @@ Map.find env s)
  in
  let rec loop_proc env llc =
    let loop = loop_proc env in
    match llc with
    | Noop -> ()
    | (Seq (c1, c2) : t) ->
        loop c1;
        loop c2
    | For_loop { index; from_; to_ = _; body; trace_it = false } ->
        loop_proc (Map.add_exn ~key:index ~data:from_ env) body
    | For_loop { index; from_; to_; body; trace_it = true } ->
        for data = from_ to min to_ (from_ + virtualize_settings.max_tracing_dim) do
          loop_proc (Map.add_exn ~key:index ~data env) body
        done
    | Zero_out tn ->
        let traced : traced_array = get_node traced_store tn in
        if Hash_set.is_empty traced.assignments && Hashtbl.is_empty traced.accesses then (
          traced.zero_initialized <- true;
          if is_scalar_dims tn then traced.is_scalar_constexpr <- true);
        traced.zeroed_out <- true
    | Set { tn; idcs; llv; debug = _ } ->
        loop_float env llv;
        let traced : traced_array = get_node traced_store tn in
        if
          Hash_set.is_empty traced.assignments
          && Hashtbl.is_empty traced.accesses && is_scalar_dims tn
        then traced.is_scalar_constexpr <- is_constexpr_comp traced_store llv
          (* Note: this prevents detection if the same constant is assigned inside a loop. *)
        else if not @@ Hash_set.is_empty traced.assignments then traced.is_scalar_constexpr <- false;
        Hash_set.add traced.assignments (lookup env idcs);
        Array.iter idcs ~f:(function
          | Fixed_idx _ -> ()
          | Iterator s ->
              let old_tn = Hashtbl.find_or_add reverse_node_map s ~default:(fun () -> tn) in
              (* TODO(#134): this prevents multiple virtual arrays from sharing for loops. *)
              assert (Tn.equal old_tn tn))
    | Set_local (_, llv) -> loop_float env llv
    | Comment _ -> ()
    | Staged_compilation _ -> ()
  and loop_float env llv =
    let loop = loop_float env in
    match llv with
    | Constant _ -> ()
    | Get (ptr, indices) ->
        let traced : traced_array = get_node traced_store ptr in
        let at_pos = lookup env indices in
        Hashtbl.update traced.accesses at_pos
          ~f:(visit ~is_assigned:(traced.zeroed_out || Hash_set.mem traced.assignments at_pos))
    | Local_scope { body; _ } -> loop_proc env body
    | Get_local _ -> ()
    | Get_global (Ops.Merge_buffer { source_node_id }, _) -> merge_node_id := Some source_node_id
    | Get_global _ -> ()
    | Embed_index _ -> ()
    | Binop (Arg1, llv1, _llv2) -> loop llv1
    | Binop (Arg2, _llv1, llv2) -> loop llv2
    | Binop (_, llv1, llv2) ->
        loop llv1;
        loop llv2
    | Unop (_, llv) -> loop llv
  in
  loop_proc Indexing.empty_env llc;
  Hashtbl.iter traced_store ~f:(fun traced ->
      let tn = traced.tn in
      if
        virtualize_settings.inline_scalar_constexprs && traced.is_scalar_constexpr
        && not (Tn.known_non_virtual tn)
      then Tn.update_memory_mode tn Virtual 40;
      if Option.is_none tn.memory_mode && Hashtbl.exists traced.accesses ~f:is_too_many then
        Tn.update_memory_mode tn Never_virtual 1
        (* The tensor node is read-only/recurrent for this computation, but maybe computed by
           another one. However, if the memory mode is unspecified, we assume this will be the first
           computation involving the tensor node. *);
      if (not traced.zeroed_out) && Hash_set.is_empty traced.assignments then (
        traced.read_only <- true;
        if Tn.mode_is_unspecified tn then Tn.update_memory_mode tn (Hosted Constant) 37
        else Tn.update_memory_mode tn Materialized 35);
      if Hashtbl.exists traced.accesses ~f:is_recurrent then (
        traced.read_before_write <- true;
        if Tn.mode_is_unspecified tn then Tn.update_memory_mode tn (Hosted Changed_on_devices) 38
        else Tn.update_memory_mode tn Materialized 36))

let%diagn_sexp check_and_store_virtual traced static_indices top_llc =
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
               | Iterator s -> Option.some_if (not @@ Set.mem static_indices s) s)
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
    | Set { tn; idcs; llv; debug = _ } ->
        if Tn.equal tn top_tn then (
          check_idcs idcs;
          has_setter := true)
        else
          (* Check for escaping variables. *)
          Array.iter idcs ~f:(function
            | Iterator s as _idx when not (Set.mem static_indices s) ->
                if not @@ Set.mem env_dom s then
                  if Utils.settings.with_debug_level > 1 then
                    [%log
                      "INFO: Inlining candidate has an escaping variable",
                        (_idx : Indexing.axis_index),
                        (top_llc : t)];
                raise @@ Non_virtual 7
            | _ -> ());
        loop_float ~env_dom llv
    | Set_local (_, llv) -> loop_float ~env_dom llv
    | Comment _ -> ()
    | Staged_compilation _ -> raise @@ Non_virtual 8
  and loop_float ~env_dom llv =
    match llv with
    | Constant _ -> ()
    | Get (tn, idcs) ->
        if Tn.equal tn top_tn then check_idcs idcs
        else
          (* Check for escaping variables. *)
          Array.iter idcs ~f:(function
            | Iterator s when not (Set.mem static_indices s) ->
                if not @@ Set.mem env_dom s then (
                  if Utils.settings.with_debug_level > 1 then
                    [%log
                      "Inlining candidate has an escaping variable",
                        (s : Indexing.symbol),
                        (top_llc : t)];
                  raise @@ Non_virtual 9)
            | _ -> ())
    | Local_scope { body; _ } -> loop_proc ~env_dom body
    | Get_local _ -> ()
    | Get_global _ -> ()
    | Embed_index (Fixed_idx _) -> ()
    | Embed_index (Iterator s) ->
        if not @@ Set.mem env_dom s then (
          if Utils.settings.with_debug_level > 1 && not (Set.mem static_indices s) then
            [%log
              "Inlining candidate has an escaping variable", (s : Indexing.symbol), (top_llc : t)];
          raise @@ Non_virtual 10)
    | Binop (_, llv1, llv2) ->
        loop_float ~env_dom llv1;
        loop_float ~env_dom llv2
    | Unop (_, llv) -> loop_float ~env_dom llv
  in
  try
    if Tn.known_non_virtual traced.tn then raise @@ Non_virtual 11;
    loop_proc ~env_dom:static_indices top_llc;
    if not !has_setter then raise @@ Non_virtual 12;
    traced.computations <- (!at_idcs, top_llc) :: traced.computations
  with Non_virtual i -> Tn.update_memory_mode traced.tn Never_virtual i

let inline_computation ~id traced static_indices call_args =
  let exception Non_virtual of int in
  let static_indices =
    Set.of_list (module Indexing.Symbol)
    @@ List.map ~f:(fun s -> s.Indexing.static_symbol) static_indices
  in
  let make_subst i lhs_ind =
    let rhs_ind = call_args.(i) in
    match lhs_ind with
    | Indexing.Iterator lhs_s when not (Set.mem static_indices lhs_s) -> Some (lhs_s, rhs_ind)
    | _ when Indexing.equal_axis_index lhs_ind rhs_ind -> None
    | _ -> raise @@ Non_virtual 13
  in
  (* In the order of computation. *)
  let loop_proc (def_args, def) : t option =
    let env =
      match def_args with
      | None -> Map.empty (module Indexing.Symbol)
      | Some def_args ->
          Map.of_alist_exn (module Indexing.Symbol)
          @@ Array.to_list
          @@ Array.filter_mapi def_args ~f:make_subst
    in
    let subst env = function
      | Indexing.Iterator s when Map.mem env s -> Map.find_exn env s
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
      | Set { tn; idcs; llv; debug = _ } when Tn.equal tn traced.tn ->
          assert ([%equal: Indexing.axis_index array option] (Some idcs) def_args);
          Some (Set_local (id, loop_float env llv))
      | Zero_out _ -> None
      | Set _ -> None
      | Set_local (id, llv) -> Some (Set_local (id, loop_float env llv))
      | Comment _ -> Some llc
      | Staged_compilation _ -> Some llc
    and loop_float env llv : float_t =
      match llv with
      | Constant _ -> llv
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
      | Get_local _ -> llv
      | Get_global _ -> llv
      | Embed_index idx -> Embed_index (subst env idx)
      | Binop (op, llv1, llv2) -> Binop (op, loop_float env llv1, loop_float env llv2)
      | Unop (op, llv) -> Unop (op, loop_float env llv)
    in
    loop env def
  in
  try
    let body = List.rev_filter_map ~f:loop_proc traced.computations in
    if List.is_empty body then raise @@ Non_virtual 14 else Some (unflat_lines body)
  with Non_virtual i ->
    Tn.update_memory_mode traced.tn Never_virtual i;
    None

let optimize_integer_pow = ref true

let rec unroll_pow ~(base : float_t) ~(exp : int) : float_t =
  if exp < 0 then unroll_pow ~base:(Binop (Div, Constant 1., base)) ~exp:(Int.neg exp)
  else if exp = 0 then Constant 1.
  else Fn.apply_n_times ~n:(exp - 1) (fun accu -> Binop (Mul, base, accu)) base

let virtual_llc traced_store reverse_node_map static_indices (llc : t) : t =
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
              check_and_store_virtual node static_indices result;
            result
        | _ -> For_loop { for_config with body = loop body })
    | Zero_out tn ->
        let traced : traced_array = get_node traced_store tn in
        if (not @@ Set.mem process_for tn) && (not @@ Tn.known_non_virtual traced.tn) then
          check_and_store_virtual traced static_indices llc;
        llc
    | Set { tn; idcs; llv; debug = _ } ->
        let traced : traced_array = get_node traced_store tn in
        let next = if Tn.known_non_virtual traced.tn then process_for else Set.add process_for tn in
        let result = Set { tn; idcs; llv = loop_float ~process_for:next llv; debug = "" } in
        if (not @@ Set.mem process_for tn) && (not @@ Tn.known_non_virtual traced.tn) then
          check_and_store_virtual traced static_indices result;
        result
    | Set_local (id, llv) -> Set_local (id, loop_float ~process_for llv)
    | Comment _ -> llc
    | Staged_compilation _ -> llc
  and loop_float ~process_for (llv : float_t) : float_t =
    match llv with
    | Constant _ -> llv
    | Get (tn, _) when Set.mem process_for tn ->
        (* [Get_local] will replace this [Get] during [inline_computation] if [tn] remains
           virtual. *)
        llv
    | Get (tn, indices) ->
        let traced = get_node traced_store tn in
        if Tn.known_non_virtual traced.tn then llv
        else
          let id = get_scope tn in
          Option.value ~default:llv
          @@ Option.map (inline_computation ~id traced static_indices indices) ~f:(fun body ->
                 Local_scope { id; body; orig_indices = indices })
    | Local_scope opts ->
        Local_scope
          { opts with body = loop_proc ~process_for:(Set.add process_for opts.id.tn) opts.body }
    | Get_local _ -> llv
    | Get_global _ -> llv
    | Embed_index _ -> llv
    | Binop (op, llv1, llv2) ->
        Binop (op, loop_float ~process_for llv1, loop_float ~process_for llv2)
    | Unop (op, llv) -> Unop (op, loop_float ~process_for llv)
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
              (* FIXME: *)
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
          (* FIXME: *)
          Tn.update_memory_mode tn Virtual 151;
          None)
        else Some llc
    | Set { tn; idcs; llv; debug = _ } ->
        if not @@ Tn.known_non_virtual tn then (
          (* FIXME: *)
          Tn.update_memory_mode tn Virtual 152;
          None)
        else (
          assert (
            Array.for_all idcs ~f:(function Indexing.Iterator s -> Set.mem env_dom s | _ -> true));
          Some (Set { tn; idcs; llv = loop_float ~balanced ~env_dom llv; debug = "" }))
    | Set_local (id, llv) ->
        assert (not @@ Tn.known_non_virtual id.tn);
        Tn.update_memory_mode id.tn Virtual 16;
        Some (Set_local (id, loop_float ~balanced ~env_dom llv))
    | Comment _ -> Some llc
    | Staged_compilation _ -> Some llc
  and loop_float ~balanced ~env_dom (llv : float_t) : float_t =
    let loop = loop_float ~balanced ~env_dom in
    match llv with
    | Constant _ -> llv
    | Get (a, indices) ->
        (* DEBUG: *)
        Tn.update_memory_mode a Never_virtual 17;
        assert (
          Array.for_all indices ~f:(function Indexing.Iterator s -> Set.mem env_dom s | _ -> true));
        llv
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
        llv
    | Get_global _ -> llv
    | Embed_index (Fixed_idx _) -> llv
    | Embed_index (Iterator s) ->
        assert (Set.mem env_dom s);
        llv
    | Binop (op, llv1, llv2) -> Binop (op, loop llv1, loop llv2)
    | Unop (op, llv) -> Unop (op, loop llv)
  in
  let static_indices =
    Set.of_list (module Indexing.Symbol)
    @@ List.map ~f:(fun s -> s.Indexing.static_symbol) static_indices
  in
  Option.value_exn ~here:[%here] @@ loop_proc ~balanced:false ~env_dom:static_indices llc

let rec substitute_float ~var ~value llv =
  let loop_float = substitute_float ~var ~value in
  let loop_proc = substitute_proc ~var ~value in
  if equal_float_t var llv then value
  else
    match llv with
    | Constant _ -> llv
    | Get (_ptr, _indices) -> llv
    | Local_scope opts -> Local_scope { opts with body = loop_proc opts.body }
    | Get_local _ -> llv
    | Get_global _ -> llv
    | Embed_index _ -> llv
    | Binop (op, llv1, llv2) -> Binop (op, loop_float llv1, loop_float llv2)
    | Unop (op, llv) -> Unop (op, loop_float llv)

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
  | Set { tn; idcs; llv; debug = _ } -> Set { tn; idcs; llv = loop_float llv; debug = "" }
  | Set_local (id, llv) -> Set_local (id, loop_float llv)
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
    | Set { tn; idcs; llv; debug = _ } -> Set { tn; idcs; llv = loop_float llv; debug = "" }
    | Set_local (id, llv) -> Set_local (id, loop_float llv)
    | Comment _ -> llc
    | Staged_compilation _ -> llc
  and loop_float (llv : float_t) : float_t =
    let local_scope_body, llv' =
      match llv with
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
      | _ -> (Noop, llv)
    in
    match llv' with
    | Constant _ -> llv
    | Get (_ptr, _indices) -> llv
    | Local_scope { id; body = Set_local (id2, v); _ } when equal_scope_id id id2 -> loop_float v
    | Local_scope { id; body = Seq (Set_local (id1, v1), Set_local (id2, v2)); _ }
      when equal_scope_id id id1 && equal_scope_id id id2 ->
        loop_float @@ substitute_float ~var:(Get_local id) ~value:v1 v2
    | Local_scope opts -> Local_scope { opts with body = loop_proc local_scope_body }
    | Get_local _ -> llv
    | Get_global _ -> llv
    | Embed_index (Fixed_idx i) -> Constant (Float.of_int i)
    | Embed_index (Iterator _) -> llv
    | Binop (Arg1, llv1, _) -> loop_float llv1
    | Binop (Arg2, _, llv2) -> loop_float llv2
    | Binop (op, Constant c1, Constant c2) -> Constant (Ops.interpret_binop op c1 c2)
    | Binop (Add, llv, Constant 0.) | Binop (Sub, llv, Constant 0.) | Binop (Add, Constant 0., llv)
      ->
        loop_float llv
    | Binop (Sub, Constant 0., llv) -> loop_float @@ Binop (Mul, Constant (-1.), llv)
    | Binop (Mul, llv, Constant 1.) | Binop (Div, llv, Constant 1.) | Binop (Mul, Constant 1., llv)
      ->
        loop_float llv
    | Binop (Mul, _, Constant 0.) | Binop (Div, Constant 0., _) | Binop (Mul, Constant 0., _) ->
        Constant 0.
    | Binop (Add, (Binop (Add, Constant c2, llv) | Binop (Add, llv, Constant c2)), Constant c1)
    | Binop (Add, Constant c1, (Binop (Add, Constant c2, llv) | Binop (Add, llv, Constant c2))) ->
        loop_float @@ Binop (Add, Constant (c1 +. c2), llv)
    | Binop (Sub, (Binop (Add, Constant c2, llv) | Binop (Add, llv, Constant c2)), Constant c1) ->
        loop_float @@ Binop (Add, Constant (c2 -. c1), llv)
    | Binop (Sub, Constant c1, (Binop (Add, Constant c2, llv) | Binop (Add, llv, Constant c2))) ->
        loop_float @@ Binop (Sub, Constant (c1 -. c2), llv)
    | Binop (Add, llv1, Binop (Sub, llv2, llv3)) | Binop (Add, Binop (Sub, llv2, llv3), llv1) ->
        loop_float @@ Binop (Sub, Binop (Add, llv1, llv2), llv3)
    | Binop (Sub, llv1, Binop (Sub, llv2, llv3)) ->
        loop_float @@ Binop (Sub, Binop (Add, llv1, llv3), llv2)
    | Binop (Sub, Binop (Sub, llv1, llv2), llv3) ->
        loop_float @@ Binop (Sub, llv1, Binop (Add, llv2, llv3))
    | Binop (Mul, (Binop (Mul, Constant c2, llv) | Binop (Mul, llv, Constant c2)), Constant c1)
    | Binop (Mul, Constant c1, (Binop (Mul, Constant c2, llv) | Binop (Mul, llv, Constant c2))) ->
        loop_float @@ Binop (Mul, Constant (c1 *. c2), llv)
    | Binop (Div, (Binop (Mul, Constant c2, llv) | Binop (Mul, llv, Constant c2)), Constant c1) ->
        loop_float @@ Binop (Mul, Constant (c2 /. c1), llv)
    | Binop (Div, Constant c1, (Binop (Mul, Constant c2, llv) | Binop (Mul, llv, Constant c2))) ->
        (* TODO: this might worsen the conditioning in hand-designed formula cases. *)
        loop_float @@ Binop (Div, Constant (c1 /. c2), llv)
    | Binop (Mul, llv1, Binop (Div, llv2, llv3)) | Binop (Mul, Binop (Div, llv2, llv3), llv1) ->
        loop_float @@ Binop (Div, Binop (Mul, llv1, llv2), llv3)
    | Binop (Div, llv1, Binop (Div, llv2, llv3)) ->
        loop_float @@ Binop (Div, Binop (Mul, llv1, llv3), llv2)
    | Binop (Div, Binop (Div, llv1, llv2), llv3) ->
        loop_float @@ Binop (Div, llv1, Binop (Mul, llv2, llv3))
    | Binop (ToPowOf, llv1, llv2) -> (
        let v1 : float_t = loop_float llv1 in
        let v2 : float_t = loop_float llv2 in
        let result : float_t = Binop (ToPowOf, v1, v2) in
        if not !optimize_integer_pow then result
        else
          match v2 with
          | Constant c when Float.is_integer c ->
              loop_float @@ unroll_pow ~base:v1 ~exp:(Float.to_int c)
          | _ -> result)
    | Binop (op, llv1, llv2) ->
        let v1 = loop_float llv1 in
        let v2 = loop_float llv2 in
        let result = Binop (op, v1, v2) in
        if equal_float_t llv1 v1 && equal_float_t llv2 v2 then result else loop_float result
    | Unop (Identity, llv) -> loop_float llv
    | Unop (op, Constant c) -> Constant (Ops.interpret_unop op c)
    | Unop (op, llv) ->
        let v = loop_float llv in
        let result = Unop (op, v) in
        if equal_float_t llv v then result else loop_float result
  in
  loop_proc llc

type traced_store = (Tn.t, traced_array) Base.Hashtbl.t [@@deriving sexp_of]

type optimized = { traced_store : traced_store; llc : t; merge_node : Tn.t option }
[@@deriving sexp_of]

let%debug_sexp optimize_proc static_indices llc =
  let traced_store = Hashtbl.create (module Tnode) in
  (* Identifies the computations that the code block associated with the symbol belongs to. *)
  let reverse_node_map = Hashtbl.create (module Indexing.Symbol) in
  [%log "tracing"];
  let merge_node_id = ref None in
  visit_llc traced_store ~merge_node_id reverse_node_map ~max_visits:virtualize_settings.max_visits
    llc;
  [%log "optimizing"];
  let llc =
    simplify_llc
    @@ cleanup_virtual_llc reverse_node_map ~static_indices
    @@ virtual_llc traced_store reverse_node_map static_indices llc
  in
  let merge_node =
    Option.map !merge_node_id ~f:(fun id -> Option.value_exn ~here:[%here] @@ Tnode.find ~id)
  in
  { traced_store; llc; merge_node }

let code_hum_margin = ref 100
let pp_comma ppf () = Stdlib.Format.fprintf ppf ",@ "
let pp_symbol ppf sym = Stdlib.Format.fprintf ppf "%s" @@ Indexing.symbol_ident sym

let pp_static_symbol ppf { Indexing.static_symbol; static_range } =
  match static_range with
  | None -> pp_symbol ppf static_symbol
  | Some range -> Stdlib.Format.fprintf ppf "%a : [0..%d]" pp_symbol static_symbol (range - 1)

let pp_index ppf idx =
  match idx with
  | Indexing.Iterator sym -> pp_symbol ppf sym
  | Fixed_idx i -> Stdlib.Format.fprintf ppf "%d" i

let pp_indices ppf idcs =
  Stdlib.Format.pp_print_list ~pp_sep:pp_comma pp_index ppf @@ Array.to_list idcs

let fprint_function_header ?name ?static_indices () ppf =
  let open Stdlib.Format in
  match (name, static_indices) with
  | Some name, Some static_indices ->
      fprintf ppf "%s @[(%a)@]:@ " name
        (pp_print_list ~pp_sep:pp_comma pp_static_symbol)
        static_indices
  | Some name, None -> fprintf ppf "%s:@ " name
  | _ -> ()

let get_ident_within_code ?no_dots llcs =
  let ident_style = Tn.get_style ~arg_name:"ll_ident_style" ?no_dots () in
  let nograd_idents = Hashtbl.create (module String) in
  let grad_idents = Hashtbl.create (module String) in
  let visit tn =
    let idents =
      if List.mem ~equal:String.equal tn.Tn.label "grad" then grad_idents else nograd_idents
    in
    Option.iter (Tn.ident_label tn)
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
    | Set { tn; llv; _ } ->
        visit tn;
        loop_float llv
    | Set_local ({ tn; _ }, llv) ->
        visit tn;
        loop_float llv
  and loop_float fc =
    match fc with
    | Local_scope { id = { tn; _ }; body; orig_indices = _ } ->
        visit tn;
        loop body
    | Get_global (_, _) -> ()
    | Get (la, _) -> visit la
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

let fprint_hum ?name ?static_indices () ppf llc =
  let ident_label = get_ident_within_code [| llc |] in
  let open Stdlib.Format in
  pp_set_margin ppf !code_hum_margin;
  let pp_ident ppf la = fprintf ppf "%s" @@ ident_label la in
  let pp_local ppf { tn; scope_id } = fprintf ppf "v%d_%a" scope_id pp_ident tn in
  let rec pp_ll ppf c : unit =
    match c with
    | Noop -> ()
    | Seq (c1, c2) ->
        fprintf ppf "@[<v 0>%a@]" (pp_print_list pp_ll)
          (List.filter [ c1; c2 ] ~f:(function Noop -> false | _ -> true))
    | For_loop { index = i; from_; to_; body; trace_it = _ } ->
        fprintf ppf "@[<v 2>for %a = %d to %d {@ %a@]@,}" pp_symbol i from_ to_ pp_ll body
    | Zero_out tn -> fprintf ppf "zero_out %a;" pp_ident tn
    | Set p ->
        p.debug <-
          asprintf "@[<2>%a[@,%a] :=@ %a;@]" pp_ident p.tn pp_indices p.idcs (pp_float p.tn.prec)
            p.llv;
        fprintf ppf "@[<2>%a[@,%a] :=@ %a;@]" pp_ident p.tn pp_indices p.idcs (pp_float p.tn.prec)
          p.llv
    | Comment message -> fprintf ppf "/* %s */" message
    | Staged_compilation _ -> fprintf ppf "STAGED_COMPILATION_CALLBACK()"
    | Set_local (id, llv) -> fprintf ppf "@[<2>%a :=@ %a;@]" pp_local id (pp_float id.tn.prec) llv
  and pp_float prec ppf value =
    match value with
    | Local_scope { id; body; _ } -> fprintf ppf "@[<2>%a {@ %a@]@ }@," pp_local id pp_ll body
    | Get_local id -> pp_local ppf id
    | Get_global (Ops.C_function s, None) -> fprintf ppf "%s()" s
    | Get_global (Ops.C_function s, Some idcs) -> fprintf ppf "%s(%a)" s pp_indices idcs
    | Get_global (Ops.External_unsafe { ptr; prec; dims = _ }, None) ->
        fprintf ppf "%s" @@ Ops.ptr_to_string ptr prec
    | Get_global (Ops.External_unsafe { ptr; prec; dims = _ }, Some idcs) ->
        fprintf ppf "%s[%a]" (Ops.ptr_to_string ptr prec) pp_indices idcs
    | Get_global (Ops.Merge_buffer { source_node_id }, None) ->
        let tn = Option.value_exn ~here:[%here] @@ Tnode.find ~id:source_node_id in
        fprintf ppf "merge %a" pp_ident tn
    | Get_global (Ops.Merge_buffer { source_node_id }, Some idcs) ->
        let tn = Option.value_exn ~here:[%here] @@ Tnode.find ~id:source_node_id in
        fprintf ppf "@[<2>%a[@,%a]@]" pp_ident tn pp_indices idcs
    | Get (tn, idcs) -> fprintf ppf "@[<2>%a[@,%a]@]" pp_ident tn pp_indices idcs
    | Constant c -> fprintf ppf "%f" c
    | Embed_index idx -> pp_index ppf idx
    | Binop (Arg1, v1, _v2) -> pp_float prec ppf v1
    | Binop (Arg2, _v1, v2) -> pp_float prec ppf v2
    | Binop (op, v1, v2) ->
        let prefix, infix, postfix = Ops.binop_C_syntax prec op in
        fprintf ppf "@[<1>%s%a%s@ %a@]%s" prefix (pp_float prec) v1 infix (pp_float prec) v2 postfix
    | Unop (Identity, v) -> (pp_float prec) ppf v
    | Unop (Relu, v) -> fprintf ppf "@[<1>relu(%a@])" (pp_float prec) v
  in
  fprintf ppf "@,@[<v 2>";
  fprint_function_header ?name ?static_indices () ppf;
  pp_ll ppf llc;
  fprintf ppf "@]"

let%debug_sexp optimize_proc ~unoptim_ll_source ~ll_source ~(name : string)
    (static_indices : Indexing.static_symbol list) (llc : t) : optimized =
  Option.iter unoptim_ll_source ~f:(fun ppf ->
      Stdlib.Format.fprintf ppf "%a%!" (fprint_hum ~name ~static_indices ()) llc);
  let result = optimize_proc static_indices llc in
  Option.iter ll_source ~f:(fun ppf ->
      Stdlib.Format.fprintf ppf "%a%!" (fprint_hum ~name ~static_indices ()) result.llc);
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
