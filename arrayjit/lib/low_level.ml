open Base
(** The code for operating on n-dimensional arrays. *)

module Nd = Ndarray
module LA = Lazy_array

type scope_id = { nd : LA.t; scope_id : int } [@@deriving sexp_of, equal, hash, compare]
(** *** Low-level representation. *)

let get_scope =
  let uid = ref 0 in
  fun nd ->
    Int.incr uid;
    { nd; scope_id = !uid }

(** Cases: [t] -- code, [float_t] -- single number at some precision. *)
type t =
  | Noop
  | Comment of string
  | Staged_compilation of ((unit -> unit)[@equal.ignore] [@compare.ignore])
  | Seq of t * t
  | For_loop of { index : Indexing.symbol; from_ : int; to_ : int; body : t; trace_it : bool }
  | Zero_out of LA.t
  | Set of LA.t * Indexing.axis_index array * float_t
  | Set_local of scope_id * float_t
[@@deriving sexp_of, equal]

and float_t =
  | Local_scope of {
      id : scope_id;
      prec : (Ops.prec[@equal.ignore] [@compare.ignore]);
      body : t;
      orig_indices : Indexing.axis_index array;
    }
  | Get_local of scope_id
  | Get_global of Ops.global_identifier * Indexing.axis_index array option
  | Get of LA.t * Indexing.axis_index array
  | Binop of Ops.binop * float_t * float_t
  | Unop of Ops.unop * float_t
  | Constant of float
[@@deriving sexp_of, equal, compare]

let binop ~op ~rhs1 ~rhs2 = match op with Ops.Arg1 -> rhs1 | Arg2 -> rhs2 | _ -> Binop (op, rhs1, rhs2)
let unop ~op ~rhs = match op with Ops.Identity -> rhs | _ -> Unop (op, rhs)
let rec flat_lines ts = List.concat_map ts ~f:(function Seq (t1, t2) -> flat_lines [ t1; t2 ] | t -> [ t ])

let rec unflat_lines = function
  | [] -> Noop
  | [ llc ] -> llc
  | Noop :: tl -> unflat_lines tl
  | llc :: tl -> Seq (llc, unflat_lines tl)

let comment_to_name =
  let nonliteral = Str.regexp {|[^a-zA-Z0-9_]|} in
  Str.global_replace nonliteral "_"

let extract_block_name llc = match flat_lines llc with Comment s :: _ -> comment_to_name s | _ -> ""
let executor_print_comments = ref false
let keep_files_in_run_directory = ref false
let with_debug = ref false
let debug_verbose_trace = ref false
let code_sexp_margin = ref 200

let fprint_code ppf c =
  (* TODO: something nicely concise. *)
  Caml.Format.pp_set_margin ppf !code_sexp_margin;
  Caml.Format.fprintf ppf "%s" @@ Sexp.to_string_hum @@ sexp_of_t c

(** *** Optimization *** *)

type virtualize_settings = {
  mutable enable_device_only : bool;
  mutable max_visits : int;
  mutable max_tracing_dim : int;
  mutable inline_constants : bool;
}

let virtualize_settings =
  { enable_device_only = true; max_visits = 3; max_tracing_dim = 5; inline_constants = true }

type visits =
  | Visits of int
  | Recurrent  (** A [Recurrent] visit is when there is an access prior to any assignment in an update. *)
[@@deriving sexp, equal, variants]

type traced_array = {
  nd : LA.t;
  mutable computations : (Indexing.axis_index array option * t) list;
      (** The computations (of the data node) are retrieved for optimization just as they are populated,
          so that the inlined code corresponds precisely to the changes to the arrays that would happen
          up till that point. Within the code blocks paired with an index tuple, all assignments and accesses
          must happen via the index tuple; if this is not the case for some assignment, the node cannot
          be virtual. Currently, we only allow for-loop symbols in assignment indices of virtual nodes. *)
  assignments : int array Hash_set.t;
  accesses : (int array, visits) Hashtbl.t;
      (** For dynamic indexes, we take a value of 0. This leads to an overestimate of visits, which is safe. *)
  mutable non_virtual : bool;
      (** If false, this array is never materialized, its computations are inlined on a per-scalar basis.
          A array that is hosted will not be virtual. *)
  mutable non_device_only : bool;
      (** If false, this node is only materialized on the devices it is computed on, it is not persisted
          outside of a step update. It is marked as [not !(nd.hosted)]. *)
  mutable scalar : float option;
  mutable zero_initialized : bool;
  mutable zeroed_out : bool;
  mutable read_before_write : bool;  (** The node is read before it is written (i.e. it is recurrent). *)
  mutable read_only : bool;
  mutable last_write_non_update : bool;
  mutable rhses : float_t list;
}
[@@deriving sexp_of]

let get_node store nd =
  Hashtbl.find_or_add store nd ~default:(fun () ->
      let non_virtual = nd.LA.never_virtual in
      let non_device_only = nd.never_device_only in
      {
        nd;
        computations = [];
        assignments = Hash_set.Poly.create ();
        accesses = Hashtbl.Poly.create ();
        non_virtual;
        non_device_only;
        scalar = None;
        zero_initialized = false;
        zeroed_out = false;
        read_before_write = false;
        read_only = false;
        last_write_non_update = false;
        rhses = [];
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

let precompute_constants ?idcs traced_store top_ptr llv =
  let exception Non_literal of int in
  let rec loop llv =
    match llv with
    | Constant c -> c
    | Get (nd, indices) ->
        let node = get_node traced_store nd in
        Array.iter indices ~f:(function Indexing.Fixed_idx 0 -> () | _ -> raise @@ Non_literal 1);
        Option.value_or_thunk node.scalar ~default:(fun () -> raise @@ Non_literal 2)
    | Local_scope { id; orig_indices; _ } ->
        let node = get_node traced_store id.nd in
        Array.iter orig_indices ~f:(function Indexing.Fixed_idx 0 -> () | _ -> raise @@ Non_literal 3);
        Option.value_or_thunk node.scalar ~default:(fun () -> raise @@ Non_literal 4)
    | Get_local scope_id ->
        let node = get_node traced_store scope_id.nd in
        Option.value_or_thunk node.scalar ~default:(fun () -> raise @@ Non_literal 5)
    | Get_global _ -> raise @@ Non_literal 9
    | Binop (Arg1, llv1, _llv2) -> loop llv1
    | Binop (Arg2, _llv1, llv2) -> loop llv2
    | Binop (Add, llv1, llv2) -> Float.(loop llv1 + loop llv2)
    | Binop (Sub, llv1, llv2) -> Float.(loop llv1 - loop llv2)
    | Binop (Mul, llv1, llv2) -> Float.(loop llv1 * loop llv2)
    | Binop (Div, llv1, llv2) -> Float.(loop llv1 / loop llv2)
    | Binop (ToPowOf, llv1, llv2) ->
        let v1 = loop llv1 in
        let v2 = loop llv2 in
        Float.(if is_integer v2 then int_pow v1 @@ to_int v2 else v1 ** v2)
    | Binop (Relu_gate, llv1, llv2) -> Float.(if loop llv1 > 0.0 then loop llv2 else 0.0)
    | Unop (Identity, llv) -> loop llv
    | Unop (Relu, llv) ->
        let v = loop llv in
        Float.(if v > 0.0 then v else 0.0)
  in
  let top_n = get_node traced_store top_ptr in
  try
    if top_n.non_virtual then raise @@ Non_literal 8;
    if (not top_n.nd.literal) && Hashtbl.exists top_n.accesses ~f:is_recurrent then raise @@ Non_literal 6;
    (match idcs with
    | None -> ()
    | Some idcs ->
        if Array.exists idcs ~f:(function Indexing.Fixed_idx 0 -> false | _ -> true) then
          raise @@ Non_literal 7);
    top_n.scalar <- Some (loop llv)
  with Non_literal _i ->
    (* if !with_debug then
       Caml.Format.printf "TRACE: Array #%d is non-literal because no. %d\n%!" v.id i; *)
    (* In principle we might conclude again that the node is to be inlined as scalar, that's OK. *)
    top_n.scalar <- None

let visit is_assigned old =
  if not is_assigned then Recurrent
  else match old with None -> Visits 1 | Some (Visits i) -> Visits (i + 1) | Some Recurrent -> Recurrent

let visit_llc traced_store reverse_node_map ~max_visits llc =
  let is_too_many = function Visits i -> i > max_visits | Recurrent -> true in
  let lookup env indices =
    Array.map indices ~f:(function
      | Indexing.Fixed_idx i -> i
      | Iterator s (* when Map.mem env s *) -> Map.find_exn env s)
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
    | Zero_out array ->
        let traced : traced_array = get_node traced_store array in
        if Hash_set.is_empty traced.assignments && Hashtbl.is_empty traced.accesses then
          traced.zero_initialized <- true;
        traced.zeroed_out <- true
    | Set (array, idcs, llv) ->
        loop_float env llv;
        let traced : traced_array = get_node traced_store array in
        Hash_set.add traced.assignments (lookup env idcs);
        traced.rhses <- List.dedup_and_sort ~compare:[%compare: float_t] @@ (llv :: traced.rhses);
        if virtualize_settings.inline_constants then precompute_constants ~idcs traced_store array llv;
        (match llv with
        | Get (_array2, _idcs2) -> traced.last_write_non_update <- true
        | Binop (_, Get (array2, idcs2), _)
          when LA.equal array array2 && [%equal: Indexing.axis_index array] idcs idcs2 ->
            traced.last_write_non_update <- false
        | Binop (_, _, Get (array2, idcs2))
          when LA.equal array array2 && [%equal: Indexing.axis_index array] idcs idcs2 ->
            traced.last_write_non_update <- false
        | Constant _ -> traced.last_write_non_update <- true
        | _ -> traced.last_write_non_update <- true);
        Array.iter idcs ~f:(function
          | Fixed_idx _ -> ()
          | Iterator s ->
              let old_array = Hashtbl.find_or_add reverse_node_map s ~default:(fun () -> array) in
              (* TODO(#134): this prevents multiple virtual arrays from sharing for loops. *)
              assert (LA.equal old_array array))
    | Set_local (_, llv) -> loop_float env llv
    | Comment _ -> ()
    | Staged_compilation _ -> ()
  and loop_float env llv =
    let loop = loop_float env in
    match llv with
    | Constant _ -> ()
    | Get (ptr, indices) ->
        let array : traced_array = get_node traced_store ptr in
        let at_pos = lookup env indices in
        Hashtbl.update array.accesses at_pos
          ~f:(visit (array.zeroed_out || Hash_set.mem array.assignments at_pos))
    | Local_scope { body; _ } -> loop_proc env body
    | Get_local _ -> ()
    | Get_global _ -> ()
    | Binop (Arg1, llv1, _llv2) -> loop llv1
    | Binop (Arg2, _llv1, llv2) -> loop llv2
    | Binop (_, llv1, llv2) ->
        loop llv1;
        loop llv2
    | Unop (_, llv) -> loop llv
  in
  loop_proc Indexing.empty_env llc;
  Hashtbl.iter traced_store ~f:(fun traced ->
      if Hashtbl.exists traced.accesses ~f:is_too_many then traced.non_virtual <- true;
      if (not traced.zeroed_out) && Hash_set.is_empty traced.assignments then traced.read_only <- true;
      if Hashtbl.exists traced.accesses ~f:is_recurrent then (
        traced.non_virtual <- true;
        traced.non_device_only <- true;
        traced.read_before_write <- true))

let process_computation traced top_llc =
  let exception Non_virtual in
  let at_idcs = ref None in
  let has_setter = ref false in
  let top_array = traced.nd in
  let check_idcs indices =
    (match !at_idcs with
    | None -> at_idcs := Some indices
    | Some at -> if not @@ [%equal: Indexing.axis_index array] at indices then raise Non_virtual);
    (* TODO(#133): For non-recursive accesses, non-linearity is not supported yet. *)
    let syms =
      Set.of_array (module Indexing.Symbol)
      @@ Array.filter_map indices ~f:Indexing.(function Fixed_idx _ -> None | Iterator s -> Some s)
    in
    let num_syms = Array.count indices ~f:(function Iterator _ -> true | _ -> false) in
    if Set.length syms <> num_syms then raise Non_virtual
  in
  (* Traverse the float code too, for completeness / future use-cases. *)
  let rec loop_proc ~(env_dom : unit Indexing.environment) llc =
    let loop = loop_proc ~env_dom in
    match llc with
    | Noop -> ()
    | (Seq (c1, c2) : t) ->
        loop c1;
        loop c2
    | For_loop { trace_it = false; _ } -> raise Non_virtual
    | For_loop { index; body; from_ = _; to_ = _; trace_it = true } ->
        loop_proc ~env_dom:(Map.add_exn ~key:index ~data:() env_dom) body
    | Zero_out array -> if LA.equal array top_array then has_setter := true
    | Set (array, indices, llv) ->
        if LA.equal array top_array then (
          check_idcs indices;
          has_setter := true)
        else
          (* Check for escaping variables. *)
          Array.iter indices ~f:(function
            | Iterator s as idx ->
                if not @@ Map.mem env_dom s then (
                  if !with_debug then
                    Caml.Format.printf "INFO: Inlining candidate has an escaping variable %a:@ %a\n%!"
                      Sexp.pp_hum
                      ([%sexp_of: Indexing.axis_index] idx)
                      Sexp.pp_hum
                      ([%sexp_of: t] top_llc);
                  raise Non_virtual)
            | _ -> ());
        loop_float ~env_dom llv
    | Set_local (_, llv) -> loop_float ~env_dom llv
    | Comment _ -> ()
    | Staged_compilation _ -> raise Non_virtual
  and loop_float ~env_dom llv =
    match llv with
    | Constant _ -> ()
    | Get (array, idcs) ->
        if LA.equal array top_array then check_idcs idcs
        else
          (* Check for escaping variables. *)
          Array.iter idcs ~f:(function
            | Iterator s as idx ->
                if not @@ Map.mem env_dom s then (
                  if !with_debug then
                    Caml.Format.printf "INFO: Inlining candidate has an escaping variable %a:@ %a\n%!"
                      Sexp.pp_hum
                      ([%sexp_of: Indexing.axis_index] idx)
                      Sexp.pp_hum
                      ([%sexp_of: t] top_llc);
                  raise Non_virtual)
            | _ -> ())
    | Local_scope { body; _ } -> loop_proc ~env_dom body
    | Get_local _ -> ()
    | Get_global _ -> ()
    | Binop (_, llv1, llv2) ->
        loop_float ~env_dom llv1;
        loop_float ~env_dom llv2
    | Unop (_, llv) -> loop_float ~env_dom llv
  in
  try
    if traced.non_virtual then raise Non_virtual;
    loop_proc ~env_dom:Indexing.empty_env top_llc;
    if not !has_setter then raise Non_virtual;
    traced.computations <- (!at_idcs, top_llc) :: traced.computations
  with Non_virtual -> traced.non_virtual <- true

let inline_computation ~id traced call_args =
  let exception Non_virtual in
  let make_subst i lhs_ind =
    let rhs_ind = call_args.(i) in
    match (lhs_ind, rhs_ind) with
    | Indexing.Iterator lhs_s, Indexing.Iterator rhs_s -> Some (lhs_s, rhs_s)
    | _ when Indexing.equal_axis_index lhs_ind rhs_ind -> None
    | _ -> raise Non_virtual
  in
  (* In the order of computation. *)
  let loop_proc (def_args, def) : t option =
    let env =
      match def_args with
      | None -> Map.Poly.empty
      | Some def_args -> Map.Poly.of_alist_exn @@ Array.to_list @@ Array.filter_mapi def_args ~f:make_subst
    in
    let subst env = function
      | Indexing.Iterator s when Map.mem env s -> Indexing.Iterator (Map.find_exn env s)
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
          let env = Map.Poly.add_exn ~key:index ~data:fresh env in
          Option.map ~f:(fun body : t -> For_loop { index = fresh; from_; to_; body; trace_it })
          @@ loop env body
      | Zero_out array when LA.equal array traced.nd -> Some (Set_local (id, Constant 0.0))
      | Set (array, indices, llv) when LA.equal array traced.nd ->
          assert ([%equal: Indexing.axis_index array option] (Some indices) def_args);
          Some (Set_local (id, loop_float env llv))
      | Zero_out _ -> None
      | Set _ -> None
      | Set_local (id, llv) -> Some (Set_local (id, loop_float env llv))
      | Comment _ -> Some llc
      | Staged_compilation _ -> Some llc
    and loop_float env llv : float_t =
      match llv with
      | Constant _ -> llv
      | Get (array, indices) when LA.equal array traced.nd ->
          assert ([%equal: Indexing.axis_index array option] (Some indices) def_args);
          Get_local id
      | Get (array, indices) -> Get (array, Array.map ~f:(subst env) indices)
      | Local_scope { id; prec; body; orig_indices } ->
          Local_scope
            {
              id;
              prec;
              body = Option.value_exn @@ loop env body;
              orig_indices = Array.map ~f:(subst env) orig_indices;
            }
      | Get_local _ -> llv
      | Get_global _ -> llv
      | Binop (op, llv1, llv2) -> Binop (op, loop_float env llv1, loop_float env llv2)
      | Unop (op, llv) -> Unop (op, loop_float env llv)
    in
    loop env def
  in
  try Some (unflat_lines (List.rev_filter_map ~f:loop_proc traced.computations) : t)
  with Non_virtual ->
    traced.non_virtual <- true;
    None

let optimize_integer_pow = ref true

let rec unroll_pow ~(base : float_t) ~(exp : int) : float_t =
  if exp < 0 then unroll_pow ~base:(Binop (Div, Constant 1., base)) ~exp:(Int.neg exp)
  else if exp = 0 then Constant 1.
  else Fn.apply_n_times ~n:(exp - 1) (fun accu -> Binop (Mul, base, accu)) base

let virtual_llc traced_store reverse_node_map (llc : t) : t =
  (* The current position is within scope of the definitions of the process_for virtual arrays. *)
  let rec loop_proc ~process_for (llc : t) : t =
    let loop = loop_proc ~process_for in
    match llc with
    | Noop -> Noop
    | Seq (c1, c2) -> Seq (loop c1, loop c2)
    | For_loop ({ index; body; _ } as for_config) -> (
        match Hashtbl.find reverse_node_map index with
        | Some array when not @@ Set.mem process_for array ->
            let node : traced_array = get_node traced_store array in
            let result = loop_proc ~process_for:(Set.add process_for array) llc in
            if not node.non_virtual then process_computation node result;
            result
        | _ -> For_loop { for_config with body = loop body })
    | Zero_out array ->
        let traced : traced_array = get_node traced_store array in
        if (not @@ Set.mem process_for array) && not traced.non_virtual then process_computation traced llc;
        llc
    | Set (array, indices, llv) ->
        let traced : traced_array = get_node traced_store array in
        let next = if traced.non_virtual then process_for else Set.add process_for array in
        let result = Set (array, indices, loop_float ~process_for:next llv) in
        if (not @@ Set.mem process_for array) && not traced.non_virtual then process_computation traced result;
        result
    | Set_local (id, llv) -> Set_local (id, loop_float ~process_for llv)
    | Comment _ -> llc
    | Staged_compilation _ -> llc
  and loop_float ~process_for (llv : float_t) : float_t =
    match llv with
    | Constant _ -> llv
    | Get (array, _) when Set.mem process_for array ->
        (* [Get_local] will replace this [Get] during [inline_computation] if [array] remains virtual. *)
        llv
    | Get (array, indices) ->
        let traced = get_node traced_store array in
        if traced.non_virtual then llv
        else
          let id = get_scope array in
          Option.value ~default:llv
          @@ Option.map (inline_computation ~id traced indices) ~f:(fun body ->
                 Local_scope { id; prec = traced.nd.prec; body; orig_indices = indices })
    | Local_scope opts ->
        Local_scope { opts with body = loop_proc ~process_for:(Set.add process_for opts.id.nd) opts.body }
    | Get_local _ -> llv
    | Get_global _ -> llv
    | Binop (op, llv1, llv2) -> Binop (op, loop_float ~process_for llv1, loop_float ~process_for llv2)
    | Unop (op, llv) -> Unop (op, loop_float ~process_for llv)
  in
  loop_proc ~process_for:(Set.empty (module Lazy_array)) llc

let cleanup_virtual_llc traced_store reverse_node_map (llc : t) : t =
  let is_inline array =
    let node = get_node traced_store array in
    (virtualize_settings.inline_constants && Option.is_some node.scalar) || not node.non_virtual
  in
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
        | Some array ->
            if is_inline array then None
            else
              Option.map ~f:(fun body : t -> For_loop { for_config with body })
              @@ loop_proc ~balanced ~env_dom body
        | None ->
            Option.map ~f:(fun body : t -> For_loop { for_config with body })
            @@ loop_proc ~balanced ~env_dom body)
    | Zero_out array -> if is_inline array then None else Some llc
    | Set (array, indices, llv) ->
        if is_inline array then None
        else (
          assert (Array.for_all indices ~f:(function Indexing.Iterator s -> Set.mem env_dom s | _ -> true));
          Some (Set (array, indices, loop_float ~balanced ~env_dom llv)))
    | Set_local (id, llv) ->
        let node = get_node traced_store id.nd in
        if virtualize_settings.inline_constants && Option.is_some node.scalar then None
        else (
          assert (not node.non_virtual);
          Some (Set_local (id, loop_float ~balanced ~env_dom llv)))
    | Comment _ -> Some llc
    | Staged_compilation _ -> Some llc
  and loop_float ~balanced ~env_dom (llv : float_t) : float_t =
    let loop = loop_float ~balanced ~env_dom in
    match llv with
    | Constant _ -> llv
    | Get (array, indices) -> (
        let node = get_node traced_store array in
        match node.scalar with
        | Some c when virtualize_settings.inline_constants -> Constant c
        | _ ->
            if not node.non_virtual then
              Caml.Format.printf "WARNING: unexpected Get of a virtual array, details:@ %a\n%!" Sexp.pp_hum
                (sexp_of_traced_array node);
            assert (Array.for_all indices ~f:(function Indexing.Iterator s -> Set.mem env_dom s | _ -> true));
            llv)
    | Local_scope { id; prec; body; orig_indices } -> (
        let traced = get_node traced_store id.nd in
        match traced.scalar with
        | Some c when virtualize_settings.inline_constants -> Constant c
        | _ ->
            assert (
              Array.for_all orig_indices ~f:(function Indexing.Iterator s -> Set.mem env_dom s | _ -> true));
            if traced.non_virtual then Get (id.nd, orig_indices)
            else
              Option.value_or_thunk ~default:(fun () ->
                  Caml.Format.printf
                    "WARNING: unexpected non-eliminable virtual array:@ %a@ Compilation data:@ %a@ \n%!"
                    Sexp.pp_hum (LA.sexp_of_t id.nd) Sexp.pp_hum (sexp_of_traced_array traced);
                  Get (id.nd, orig_indices))
              @@ Option.map ~f:(fun body -> Local_scope { id; prec; orig_indices; body })
              @@ loop_proc ~balanced ~env_dom body)
    | Get_local id -> (
        let node = get_node traced_store id.nd in
        match node.scalar with
        | Some c when virtualize_settings.inline_constants -> Constant c
        | _ ->
            assert (not node.non_virtual);
            llv)
    | Get_global _ -> llv
    | Binop (op, llv1, llv2) -> Binop (op, loop llv1, loop llv2)
    | Unop (op, llv) -> Unop (op, loop llv)
  in
  Option.value_exn @@ loop_proc ~balanced:false ~env_dom:Set.Poly.empty llc

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
    | Binop (op, llv1, llv2) -> Binop (op, loop_float llv1, loop_float llv2)
    | Unop (op, llv) -> Unop (op, loop_float llv)

and substitute_proc ~var ~value llc =
  let loop_float = substitute_float ~var ~value in
  let loop_proc = substitute_proc ~var ~value in
  match llc with
  | Noop -> Noop
  | Seq (c1, c2) -> Seq (loop_proc c1, loop_proc c2)
  | For_loop for_config -> For_loop { for_config with body = loop_proc for_config.body }
  | Zero_out _ -> llc
  | Set (array, indices, llv) -> Set (array, indices, loop_float llv)
  | Set_local (id, llv) -> Set_local (id, loop_float llv)
  | Comment _ -> llc
  | Staged_compilation _ -> llc

let simplify_llc traced_store llc =
  let rec loop_proc (llc : t) : t =
    let loop = loop_proc in
    match llc with
    | Noop -> Noop
    | Seq (c1, c2) -> Seq (loop c1, loop c2)
    | For_loop for_config -> For_loop { for_config with body = loop for_config.body }
    | Zero_out _ -> llc
    | Set (array, indices, llv) -> Set (array, indices, loop_float llv)
    | Set_local (id, llv) -> Set_local (id, loop_float llv)
    | Comment _ -> llc
    | Staged_compilation _ -> llc
  and loop_float (llv : float_t) : float_t =
    (* FIXME: consider merging scalar simplification to here? *)
    let llv =
      match llv with
      | Local_scope opts -> Local_scope { opts with body = unflat_lines @@ flat_lines [ opts.body ] }
      | _ -> llv
    in
    match llv with
    | Constant _ -> llv
    | Get (_ptr, _indices) -> llv
    | Local_scope { id; body = Set_local (id2, v) | Seq (Comment _, Set_local (id2, v)); _ }
      when equal_scope_id id id2 ->
        loop_float v
    | Local_scope
        {
          id;
          body =
            ( Seq (Set_local (id1, v1), Set_local (id2, v2))
            | Seq (Comment _, Seq (Set_local (id1, v1), Set_local (id2, v2))) );
          _;
        }
      when equal_scope_id id id1 && equal_scope_id id id2 ->
        loop_float @@ substitute_float ~var:(Get_local id) ~value:v1 v2
    | Local_scope opts -> Local_scope { opts with body = loop_proc opts.body }
    | Get_local _ -> llv
    | Get_global _ -> llv
    | Binop (Arg1, llv1, _) -> loop_float llv1
    | Binop (Arg2, _, llv2) -> loop_float llv2
    | Binop (op, Constant c1, Constant c2) -> Constant (Ops.interpret_binop op c1 c2)
    | Binop (Add, llv, Constant 0.) | Binop (Sub, llv, Constant 0.) | Binop (Add, Constant 0., llv) ->
        loop_float llv
    | Binop (Sub, Constant 0., llv) -> loop_float @@ Binop (Mul, Constant (-1.), llv)
    | Binop (Mul, llv, Constant 1.) | Binop (Div, llv, Constant 1.) | Binop (Mul, Constant 1., llv) ->
        loop_float llv
    | Binop (Mul, _, Constant 0.) | Binop (Div, Constant 0., _) | Binop (Mul, Constant 0., _) -> Constant 0.
    | Binop (Add, (Binop (Add, Constant c2, llv) | Binop (Add, llv, Constant c2)), Constant c1)
    | Binop (Add, Constant c1, (Binop (Add, Constant c2, llv) | Binop (Add, llv, Constant c2))) ->
        loop_float @@ Binop (Add, Constant (c1 +. c2), llv)
    | Binop (Sub, (Binop (Add, Constant c2, llv) | Binop (Add, llv, Constant c2)), Constant c1) ->
        loop_float @@ Binop (Add, Constant (c2 -. c1), llv)
    | Binop (Sub, Constant c1, (Binop (Add, Constant c2, llv) | Binop (Add, llv, Constant c2))) ->
        loop_float @@ Binop (Sub, Constant (c1 -. c2), llv)
    | Binop (Add, llv1, Binop (Sub, llv2, llv3)) | Binop (Add, Binop (Sub, llv2, llv3), llv1) ->
        loop_float @@ Binop (Sub, Binop (Add, llv1, llv2), llv3)
    | Binop (Sub, llv1, Binop (Sub, llv2, llv3)) -> loop_float @@ Binop (Sub, Binop (Add, llv1, llv3), llv2)
    | Binop (Sub, Binop (Sub, llv1, llv2), llv3) -> loop_float @@ Binop (Sub, llv1, Binop (Add, llv2, llv3))
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
    | Binop (Div, llv1, Binop (Div, llv2, llv3)) -> loop_float @@ Binop (Div, Binop (Mul, llv1, llv3), llv2)
    | Binop (Div, Binop (Div, llv1, llv2), llv3) -> loop_float @@ Binop (Div, llv1, Binop (Mul, llv2, llv3))
    | Binop (ToPowOf, llv1, llv2) -> (
        let v1 : float_t = loop_float llv1 in
        let v2 : float_t = loop_float llv2 in
        let result : float_t = Binop (ToPowOf, v1, v2) in
        if not !optimize_integer_pow then result
        else
          match v2 with
          | Constant c when Float.is_integer c -> loop_float @@ unroll_pow ~base:v1 ~exp:(Float.to_int c)
          | Get (ptr, _) -> (
              let node : traced_array = get_node traced_store ptr in
              match node.scalar with
              | Some c when Float.is_integer c -> loop_float @@ unroll_pow ~base:v1 ~exp:(Float.to_int c)
              | _ -> result)
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

type traced_store = (LA.t, traced_array) Base.Hashtbl.t
type optimized = traced_store * t

let optimize_proc ?(verbose = false) llc : optimized =
  let traced_store : traced_store = Hashtbl.create (module Lazy_array) in
  (* Identifies the computations that the code block associated with the symbol belongs to. *)
  let reverse_node_map = Hashtbl.Poly.create () in
  if verbose then Stdio.printf "Low_level.optimize_proc: tracing\n%!";
  visit_llc traced_store reverse_node_map ~max_visits:virtualize_settings.max_visits llc;
  if verbose then Stdio.printf "Low_level.optimize_proc: optimizing\n%!";
  let result =
    simplify_llc traced_store
    @@ cleanup_virtual_llc traced_store reverse_node_map
    @@ virtual_llc traced_store reverse_node_map llc
  in
  (traced_store, result)

let compile_proc ~name ?(verbose = false) llc : optimized =
  if verbose then Stdio.printf "Low_level.compile_proc: generating the initial low-level code\n%!";
  if !with_debug && !keep_files_in_run_directory then (
    let fname = name ^ "-unoptimized.llc" in
    let f = Stdio.Out_channel.create fname in
    let ppf = Caml.Format.formatter_of_out_channel f in
    Caml.Format.pp_set_margin ppf !code_sexp_margin;
    Caml.Format.fprintf ppf "%a%!" Sexp.pp_hum (sexp_of_t llc));
  let result = optimize_proc ~verbose llc in
  if !with_debug && !keep_files_in_run_directory then (
    let fname = name ^ ".llc" in
    let f = Stdio.Out_channel.create fname in
    let ppf = Caml.Format.formatter_of_out_channel f in
    Caml.Format.pp_set_margin ppf !code_sexp_margin;
    Caml.Format.fprintf ppf "%a%!" Sexp.pp_hum (sexp_of_t @@ snd result));
  Hashtbl.iter (fst result) ~f:(fun v -> if v.non_virtual && v.non_device_only then v.nd.hosted := true);
  if verbose then Stdio.printf "Low_level.compile_proc: finished\n%!";
  result

let loop_over_dims dims ~body =
  let rec for_loop rev_idcs : _ -> t = function
    | [] -> body @@ Array.of_list_rev rev_idcs
    | d :: product when not @@ Indexing.iterated d -> for_loop (Indexing.Fixed_idx 0 :: rev_idcs) product
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

module CDSL = struct
  let single = Ops.single
  let double = Ops.double
  let executor_print_comments = executor_print_comments
  let keep_files_in_run_directory = keep_files_in_run_directory
  let with_debug = with_debug
  let debug_verbose_trace = debug_verbose_trace
  let virtualize_settings = virtualize_settings
  let code_sexp_margin = code_sexp_margin
  let fixed_state_for_init = Ndarray.fixed_state_for_init

  let enable_all_debugs ?(trace_interpreter = false) ?(hosted_only = true) () =
    with_debug := true;
    keep_files_in_run_directory := true;
    if hosted_only then virtualize_settings.enable_device_only <- false;
    if trace_interpreter then debug_verbose_trace := true

  let disable_all_debugs ?(restore_defaults = false) () =
    debug_verbose_trace := false;
    with_debug := false;
    keep_files_in_run_directory := false;
    if restore_defaults then virtualize_settings.enable_device_only <- true
end
