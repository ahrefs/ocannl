open Base
(** The code for operating on n-dimensional arrays. *)

(** *** High-level representation. *** *)
type binop = Add | Mul | ToPowOf | Relu_gate | Arg2 | Arg1 [@@deriving sexp]

type unop = Identity | Relu [@@deriving sexp]

module N = Node

type global_identifier =
  | Task_id  (** Retrieves the identifier (a non-negative integer) of the task running the computation. *)
  | C_function of string  (** Calls a no-argument C function. *)
[@@deriving sexp, equal, compare]

(** Initializes a tensor by filling in the corresponding numbers, at the appropriate precision. *)
type init_op = N.init_op =
  | Constant_fill of float array
      (** Fills in the numbers where the rightmost axis is contiguous, looping over the provided values
      if necessary. *)
  | Range_over_offsets
      (** Fills in the offset number of each cell (i.e. how many cells away it is from the beginning). *)
  | Standard_uniform  (** Draws the values from U(0,1). *)
[@@deriving sexp]

(** Resets a tensor by performing the specified computation or data fetching. *)
type fetch_op = Constant of float | Synthetic of t | Imported of global_identifier [@@deriving sexp]

and t =
  | Par of t * t  (** These tasks can proceed in parallel, there is no interaction. *)
  | ParHint of t * t
      (** Computing [ParHint (c1, c2)] can proceed in parallel on [c1] and [c2], but when [c2] reads values
      that [c1] writes, the writes in [c1] must occur before the reads in [c2]. *)
  | Seq of t * t
      (** These tasks can only benefit from mutual parallelism via operator fusion / loop fusion. *)
  | Accum_binop of {
      zero_out : bool;
      accum : binop;
      op : binop;
      lhs : NodeUI.tensor_ptr;
      rhs1 : NodeUI.tensor_ptr;
      rhs2 : NodeUI.tensor_ptr;
      projections : unit -> Shape.projections;
    }
  | Accum_unop of {
      zero_out : bool;
      accum : binop;
      op : unop;
      lhs : NodeUI.tensor_ptr;
      rhs : NodeUI.tensor_ptr;
      projections : unit -> Shape.projections;
    }
  | Fetch of { tensor : NodeUI.tensor_ptr; fetch_op : fetch_op }
  | Block_comment of string * t
  | Noop
[@@deriving sexp]

(** If a backend does not support detection of when [ParHint (c1, c2)] is safe to parallelize,
    one can try setting [force_unsafe_parhint] to always parallelize if the particular code
    does not have a form of computation sharing that would get broken. *)
let force_unsafe_parhint = ref false

type create = { tensor : NodeUI.tensor_ptr; dims : unit -> Shape.dim array; init_op : init_op }
(** Information to create a tensor, once its shape is inferred. *)

let remove_updates tensor c =
  let rec rm check = function
    | ( Par ((Accum_binop { lhs; _ } | Accum_unop { lhs; _ }), t)
      | ParHint ((Accum_binop { lhs; _ } | Accum_unop { lhs; _ }), t)
      | Seq ((Accum_binop { lhs; _ } | Accum_unop { lhs; _ }), t)
      | Par (t, (Accum_binop { lhs; _ } | Accum_unop { lhs; _ }))
      | ParHint (t, (Accum_binop { lhs; _ } | Accum_unop { lhs; _ }))
      | Seq (t, (Accum_binop { lhs; _ } | Accum_unop { lhs; _ })) ) as c
      when check ->
        if NodeUI.equal_tensor_ptr tensor lhs then rm true t else rm false c
    | Par (t1, t2) -> Par (rm true t1, rm true t2)
    | ParHint (t1, t2) -> ParHint (rm true t1, rm true t2)
    | Seq (t1, t2) -> Seq (rm true t1, rm true t2)
    | (Accum_binop { lhs; _ } | Accum_unop { lhs; _ }) when NodeUI.equal_tensor_ptr tensor lhs -> Noop
    | c -> c
  in
  rm true c

let all_parallel = List.fold ~init:Noop ~f:(fun sts st -> Par (st, sts))
let sequential = List.fold_right ~init:Noop ~f:(fun st sts -> Seq (st, sts))

let rec flat_parallel ~force_hints ts =
  Array.concat_map ts ~f:(function
    | Par (t1, t2) -> flat_parallel ~force_hints [| t1; t2 |]
    | ParHint (t1, t2) when force_hints -> flat_parallel ~force_hints [| t1; t2 |]
    | Block_comment (_, Noop) as t -> [| t |]
    | Block_comment (s, t) -> flat_parallel ~force_hints [| Block_comment (s, Noop); t |]
    | Noop -> [||]
    | t -> [| t |])

type scope_id = { tensor : NodeUI.tensor_ptr; scope_id : int } [@@deriving sexp, equal, hash]
(** *** Low-level representation. *)

let get_scope =
  let uid = ref 0 in
  fun tensor ->
    Int.incr uid;
    { tensor; scope_id = !uid }

type loop_symbol = Iterator of Shape.symbol | Special_iterator of Shape.dedicated_axis
[@@deriving sexp, equal, compare]

type loop_index = { loop_sym : loop_symbol; uid : int } [@@deriving sexp, equal, compare]
type sym_index = { sym : Shape.symbol; uid : int } [@@deriving sexp, equal, compare]
type index = sym_index Shape.axis_index [@@deriving sexp, equal, compare]

let global_task_id_idx = { loop_sym = Special_iterator Shape.Task_id; uid = -1 }
let global_sample_num_idx = { loop_sym = Special_iterator Shape.Sample_num; uid = -1 }

let loop_index_lident = function
  | { loop_sym = Iterator (Symbol s); uid } -> "i" ^ Int.to_string s ^ "_" ^ Int.to_string uid
  | { loop_sym = Special_iterator Task_id; uid } -> "i_task_id_" ^ Int.to_string uid
  | { loop_sym = Special_iterator Sample_num; uid } -> "i_sample_num_" ^ Int.to_string uid

let new_loop_uid, new_sym_uid =
  let uid = ref 0 in
  ( (fun loop_sym ->
      Int.incr uid;
      { loop_sym; uid = !uid }),
    fun sym ->
      Int.incr uid;
      { sym; uid = !uid } )

let new_loop_index it = function
  | Shape.{ special = Dedicated idx; _ } -> new_loop_uid (Special_iterator idx)
  | _ -> new_loop_uid (Iterator it)

let to_shape_index = function
  | { loop_sym = Iterator sym; uid } -> Shape.Iterator { sym; uid }
  | { loop_sym = Special_iterator idx; _ } -> Shape.Special_iterator idx

let to_loop_index = function
  | Shape.Iterator { sym; uid } -> { loop_sym = Iterator sym; uid }
  | Special_iterator idx -> { loop_sym = Special_iterator idx; uid = -1 }
  | _ -> invalid_arg "not a loop index"

(** Cases: [unit low_level] -- code, [float low_level] -- single number at some precision. *)
type _ low_level =
  | Comment : string -> unit low_level
  | Lines : unit low_level array -> unit low_level
  | For_loop : {
      index : loop_index;
      from_ : int;
      to_ : int;
      body : unit low_level;
      trace_it : bool;
    }
      -> unit low_level
  | If_task_id_is : { for_task_id : int; body : unit low_level } -> unit low_level
  | Rebalance : string option * unit low_level array -> unit low_level
  | Dynamic_indices : {
      tensor : NodeUI.tensor_ptr;
      tensor_idcs : index array;
      dynamic_idcs : Shape.symbol array;
      target_dims : Shape.dim array;
      body : unit low_level;
      slice : NodeUI.tensor_ptr option;
          (** Provided when we know the dynamic indexing was used to define this tensor. *)
    }
      -> unit low_level
  | Zero_out : NodeUI.tensor_ptr -> unit low_level
  | Set : NodeUI.tensor_ptr * index array * float low_level -> unit low_level
  | Set_local : scope_id * float low_level -> unit low_level
  | Local_scope : {
      id : scope_id;
      prec : NodeUI.prec;
      body : unit low_level;
      orig_indices : index array;
    }
      -> float low_level
  | Get_local : scope_id -> float low_level
  | Get_global : global_identifier -> float low_level
  | Get : NodeUI.tensor_ptr * index array -> float low_level
  | Binop : binop * float low_level * float low_level -> float low_level
  | Unop : unop * float low_level -> float low_level
  | Constant : float -> float low_level
[@@deriving sexp_of]

let check_not_replicable ~cached_replicable llc =
  let rec loop : type a. a low_level -> bool = function
    | Comment _ -> false
    | Lines ls -> Array.exists ~f:loop ls
    | For_loop { body; _ } -> loop body
    | Rebalance (_, cs) -> Array.exists ~f:loop cs
    | If_task_id_is { body; _ } -> loop body
    | Dynamic_indices { tensor = _; tensor_idcs; dynamic_idcs = _; target_dims; body; slice = _ } ->
        Array.exists tensor_idcs ~f:(function Shape.Special_iterator Task_id -> true | _ -> false)
        || Array.exists
             ~f:Shape.(function { special = Dedicated Task_id; _ } -> true | _ -> false)
             target_dims
        || loop body
    | Zero_out _ -> false
    | Set (_, indices, llv) ->
        Array.exists indices ~f:(function Shape.Special_iterator Task_id -> true | _ -> false) || loop llv
    | Set_local (_, llv) -> loop llv
    | Local_scope { body; orig_indices; _ } ->
        Array.exists orig_indices ~f:(function Shape.Special_iterator Task_id -> true | _ -> false)
        || loop body
    | Get_local _ -> false
    | Get_global Task_id -> true
    | Get_global _ -> false
    | Get (ptr, indices) ->
        (not (cached_replicable ptr))
        || Array.exists indices ~f:(function Shape.Special_iterator Task_id -> true | _ -> false)
    | Binop (_, llv1, llv2) -> loop llv1 || loop llv2
    | Unop (_, llv) -> loop llv
    | Constant _ -> false
  in
  loop llc

let binop ~op ~rhs1 ~rhs2 = match op with Arg1 -> rhs1 | Arg2 -> rhs2 | _ -> Binop (op, rhs1, rhs2)
let unop ~op ~rhs = match op with Identity -> rhs | _ -> Unop (op, rhs)
let rec flat_lines ts = Array.concat_map ts ~f:(function Lines ts -> flat_lines ts | t -> [| t |])

let to_low_level (code : t) : unit low_level =
  let rec loop code =
    match code with
    | Accum_binop { zero_out; accum; op; lhs; rhs1; rhs2; projections } ->
        let lhs_n = NodeUI.get lhs.id in
        (match (accum, op) with
        | Add, _ -> lhs_n.value_distributes_over_sum <- true
        | Arg2, Mul ->
            let rhs1_n = NodeUI.get rhs1.id in
            let rhs2_n = NodeUI.get rhs2.id in
            lhs_n.value_distributes_over_sum <-
              (rhs1_n.value_distributes_over_sum && not rhs2_n.value_distributes_over_sum)
              || (rhs2_n.value_distributes_over_sum && not rhs1_n.value_distributes_over_sum)
        | Arg2, Add ->
            let rhs1_n = NodeUI.get rhs1.id in
            let rhs2_n = NodeUI.get rhs2.id in
            lhs_n.value_distributes_over_sum <-
              rhs1_n.value_distributes_over_sum || rhs2_n.value_distributes_over_sum
        | _ -> lhs_n.value_distributes_over_sum <- false);
        let projections = projections () in
        let lhs_idx = Shape.(derive_index projections.product_iterators projections.project_lhs) in
        let rhs1_idx = Shape.(derive_index projections.product_iterators projections.project_rhs1) in
        let rhs2_idx =
          match projections.project_rhs2 with
          | None -> invalid_arg "accum_binop: projections missing project_rhs2"
          | Some rhs2 -> Shape.(derive_index projections.product_iterators rhs2)
        in
        let basecase rev_iters =
          let iters = Array.of_list_rev_map rev_iters ~f:to_shape_index in
          let rhs1_idcs = rhs1_idx iters in
          let rhs2_idcs = rhs2_idx iters in
          let lhs_idcs = lhs_idx iters in
          let lhs_ll = Get (lhs, lhs_idcs) in
          let rhs1_ll = Get (rhs1, rhs1_idcs) in
          let rhs2_ll = Get (rhs2, rhs2_idcs) in
          let body =
            Set (lhs, lhs_idcs, binop ~op:accum ~rhs1:lhs_ll ~rhs2:(binop ~op ~rhs1:rhs1_ll ~rhs2:rhs2_ll))
          in
          let slice = Some lhs in
          match Array.find rhs2_idcs ~f:Shape.is_dynamic_provider with
          | Some (Dynamic_provider { idcs = dynamic_idcs; target_dims }) ->
              Dynamic_indices
                { tensor = rhs2; tensor_idcs = rhs2_idcs; dynamic_idcs; target_dims; body; slice }
          | _ -> (
              match Array.find rhs1_idcs ~f:Shape.is_dynamic_provider with
              | Some (Dynamic_provider { idcs = dynamic_idcs; target_dims }) ->
                  Dynamic_indices
                    { tensor = rhs1; tensor_idcs = rhs1_idcs; dynamic_idcs; target_dims; body; slice }
              | _ -> (
                  match Array.find lhs_idcs ~f:Shape.is_dynamic_provider with
                  | Some (Dynamic_provider { idcs = dynamic_idcs; target_dims }) ->
                      Dynamic_indices
                        { tensor = lhs; tensor_idcs = lhs_idcs; dynamic_idcs; target_dims; body; slice }
                  | _ -> body))
        in
        let rec for_loop rev_iters = function
          | [], [] -> basecase rev_iters
          | d :: product, it :: iters ->
              let index = new_loop_index it d in
              For_loop
                {
                  index;
                  from_ = 0;
                  to_ = d.dim - 1;
                  body = for_loop (index :: rev_iters) (product, iters);
                  trace_it = true;
                }
          | _ -> invalid_arg "Code.to_low_level: Accum_binop projections dims-iterators mismatch"
        in
        let for_loops =
          for_loop [] (Array.to_list projections.product_space, Array.to_list projections.product_iterators)
        in
        let s = Comment ("Computing node " ^ NodeUI.tensor_ptr_name lhs) in
        (* Note: it might be invalid to replicate computation across tasks. *)
        if zero_out then Lines [| s; loop (Fetch { tensor = lhs; fetch_op = Constant 0. }); for_loops |]
        else Lines [| s; for_loops |]
    | Accum_unop { zero_out; accum; op; lhs; rhs; projections } ->
        let projections = projections () in
        let lhs_idx = Shape.(derive_index projections.product_iterators projections.project_lhs) in
        let rhs_idx = Shape.(derive_index projections.product_iterators projections.project_rhs1) in
        let basecase rev_iters =
          let iters = Array.of_list_rev_map rev_iters ~f:to_shape_index in
          let lhs_idcs = lhs_idx iters in
          let lhs_ll = Get (lhs, lhs_idcs) in
          let rhs_ll = Get (rhs, rhs_idx iters) in
          Set (lhs, lhs_idcs, binop ~op:accum ~rhs1:lhs_ll ~rhs2:(unop ~op ~rhs:rhs_ll))
        in
        let rec for_loop rev_iters = function
          | [], [] -> basecase rev_iters
          | d :: product, it :: iters ->
              let index = new_loop_index it d in
              For_loop
                {
                  index;
                  from_ = 0;
                  to_ = d.dim - 1;
                  body = for_loop (index :: rev_iters) (product, iters);
                  trace_it = true;
                }
          | _ -> invalid_arg "Code.to_low_level: Accum_unop projections dims-iterators mismatch"
        in
        let for_loops =
          for_loop [] (Array.to_list projections.product_space, Array.to_list projections.product_iterators)
        in
        let s = Comment ("Computing node " ^ NodeUI.tensor_ptr_name lhs) in
        (* Note: it might be invalid to replicate computation across tasks. *)
        if zero_out then Lines [| s; loop (Fetch { tensor = lhs; fetch_op = Constant 0. }); for_loops |]
        else Lines [| s; for_loops |]
    | Noop -> Lines [||]
    | Block_comment (s, (Par _ as c)) -> loop_par ~s c
    | Block_comment (s, (ParHint _ as c)) when !force_unsafe_parhint -> loop_par ~s c
    | Par _ -> loop_par code
    | ParHint _ when !force_unsafe_parhint -> loop_par code
    | Block_comment (s, c) -> Lines [| Comment s; loop c |]
    | ParHint (c1, c2) | Seq (c1, c2) -> (
        (* TODO: this ignores parallelization altogether, don't! *)
        let ll1 = loop c1 in
        let ll2 = loop c2 in
        match (ll1, ll2) with
        | Lines ls1, Lines ls2 -> Lines (Array.append ls1 ls2)
        | _, Lines ls2 -> Lines (Array.append [| ll1 |] ls2)
        | Lines ls1, _ -> Lines (Array.append ls1 [| ll2 |])
        | _ -> Lines [| ll1; ll2 |])
    | Fetch { tensor; fetch_op = Constant 0.0 } -> Zero_out tensor
    | Fetch { tensor; fetch_op = Constant c } ->
        let product_space : Shape.dim array = Shape.to_dims (NodeUI.get tensor.id).shape in
        let rec loop rev_idcs = function
          | [] -> Set (tensor, Array.of_list_rev rev_idcs, Constant c)
          | d :: product when Shape.dim_1 d -> loop (Fixed_idx 0 :: rev_idcs) product
          | d :: product ->
              let index = new_loop_index (Shape.get_symbol ()) d in
              For_loop
                {
                  index;
                  from_ = 0;
                  to_ = d.dim - 1;
                  body = loop (to_shape_index index :: rev_idcs) product;
                  trace_it = true;
                }
        in
        let for_loops = loop [] (Array.to_list product_space) in
        for_loops
    | Fetch { tensor = _; fetch_op = Synthetic gen } -> loop gen
    | Fetch { tensor = _; fetch_op = Imported _ } -> failwith "to_low_level: Imported NOT IMPLEMENTED YET"
  and loop_par ?s c =
    let ts = flat_parallel ~force_hints:!force_unsafe_parhint [| c |] in
    Rebalance (s, Array.map ts ~f:loop)
  in

  loop code

let executor_print_comments = ref false
let keep_files_in_run_directory = ref false
let with_debug = ref false

(*
type int_env = int env

let sexp_of_int_env env =
  [%sexp_of: (sym_index * int) list * (Shape.symbol * int) list] (Map.to_alist env, Map.to_alist dyn_env)
*)

let set_from_float ptr idcs value =
  match ptr.NodeUI.field with
  | Value -> N.set_from_float (N.get ptr.id).value idcs value
  | Grad -> N.set_from_float (Option.value_exn (N.get ptr.id).grad) idcs value

let fill_from_float ptr value =
  match ptr.NodeUI.field with
  | Value -> N.fill_from_float (N.get ptr.id).value value
  | Grad -> N.fill_from_float (Option.value_exn (N.get ptr.id).grad) value

let get_as_float ptr idcs =
  match ptr.NodeUI.field with
  | Value -> N.get_as_float (N.get ptr.id).value idcs
  | Grad -> N.get_as_float (Option.value_exn (N.get ptr.id).grad) idcs

let debug_trace_interpretation = ref false

let interpret_binop op v1 v2 =
  let open Float in
  match op with
  | Arg1 -> v1
  | Arg2 -> v2
  | Add -> v1 + v2
  | Mul -> v1 * v2
  | ToPowOf -> if is_integer v2 then int_pow v1 @@ to_int v2 else v1 ** v2
  | Relu_gate -> if v1 > 0.0 then v2 else 0.0

type 'a environment = {
  env : (sym_index, 'a) Map.Poly.t;
  dyn_env : (Shape.symbol, 'a) Map.Poly.t;
  task_id : (loop_index * 'a) option;  (** We keep track of [loop_index] for debugging purposes. *)
  sample_num : (loop_index * 'a) option;
}

let environment_keys env =
  (List.map ~f:[%sexp_of: sym_index] @@ Map.Poly.keys env.env)
  @ (List.map ~f:[%sexp_of: Shape.symbol] @@ Map.Poly.keys env.dyn_env)
  @ (List.map ~f:(fun (sym, _) -> [%sexp_of: string * loop_index] ("task_env", sym))
    @@ Option.to_list env.task_id)
  @ List.map ~f:(fun (sym, _) -> [%sexp_of: string * loop_index] ("sample_env", sym))
  @@ Option.to_list env.sample_num

let environment_add idx data env =
  match idx with
  | { loop_sym = Iterator sym; uid } -> { env with env = Map.add_exn ~key:{ sym; uid } ~data env.env }
  | { loop_sym = Special_iterator Task_id; _ } ->
      (* assert (Option.is_none env.task_id); *)
      { env with task_id = Option.first_some env.task_id @@ Some (idx, data) }
  | { loop_sym = Special_iterator Sample_num; _ } ->
      (* assert (Option.is_none env.sample_num); *)
      { env with sample_num = Option.first_some env.sample_num @@ Some (idx, data) }

let environment_mem key env =
  match key with
  | { loop_sym = Iterator sym; uid } -> Map.mem env.env { sym; uid }
  | { loop_sym = Special_iterator Task_id; _ } -> Option.is_some env.task_id
  | { loop_sym = Special_iterator Sample_num; _ } -> Option.is_some env.sample_num

let interpret_code ?task_id llc =
  (* Local scope ids can be non-unique due to inlining. *)
  let locals = ref Map.Poly.empty in
  let lookup ?provider_dim env indices =
    try
      Array.map indices ~f:(function
        | Shape.Fixed_idx i -> i
        | Special_iterator Task_id -> snd @@ Option.value_exn env.task_id
        | Special_iterator Sample_num -> snd @@ Option.value_exn env.sample_num
        | Iterator it -> Map.find_exn env.env it
        | Dynamic_recipient s -> Map.find_exn env.dyn_env s
        | Frozen_recipient s -> Map.find_exn env.dyn_env s
        | Dynamic_provider _ -> Option.value_exn provider_dim)
    with Caml.Not_found | Not_found_s _ ->
      if !debug_trace_interpretation then
        Caml.Format.printf "TRACE: lookup error for env keys=@ %a\n%!" Sexp.pp_hum
          (* FIXME: missing full env *)
          ([%sexp_of: Sexp.t list] @@ environment_keys env);
      failwith "interpret_code: index lookup error, set CDSL.debug_trace_interpretation for details"
  in
  let rec loop_proc env llc : unit =
    let loop = loop_proc env in
    match llc with
    | Lines body -> Array.iter ~f:loop body
    | For_loop { index; from_; to_; body; trace_it = _ } ->
        for data = from_ to to_ do
          loop_proc (environment_add index data env) body
        done
    | Rebalance (_, cs) ->
        (* FIXME: NOT IMPLEMENTED YET *)
        Array.iter ~f:loop cs
    | If_task_id_is { for_task_id; body } ->
        if Option.fold ~init:for_task_id ~f:(fun _ (_, i) -> i) env.task_id = for_task_id then loop body
    | Zero_out ptr -> Node.fill_from_float (Option.value_exn @@ NodeUI.get_tensor ptr) 0.0
    | Set (ptr, indices, Binop (op, Get (ptr2, indices2), c2))
      when NodeUI.equal_tensor_ptr ptr ptr2 && [%equal: index array] indices indices2 ->
        if !debug_trace_interpretation then
          Caml.Format.printf "{TRACE: update %a [%a] <- ...\n%!" Sexp.pp_hum
            ([%sexp_of: NodeUI.tensor_ptr] ptr)
            Sexp.pp_hum
            ([%sexp_of: index array] indices);
        let idcs = lookup env indices in
        (* Order computation to reduce prevalence of race conditions. *)
        let v2 = loop_float env c2 in
        let v1 =
          try get_as_float ptr idcs
          with e ->
            Caml.Format.printf "ERROR: %a [%a -> %a] -- indices out of bounds\n%!" Sexp.pp_hum
              ([%sexp_of: NodeUI.tensor_ptr] ptr)
              Sexp.pp_hum
              ([%sexp_of: index array] indices)
              Sexp.pp_hum
              ([%sexp_of: int array] idcs);
            if Option.fold ~init:0 ~f:(fun _ (_, i) -> i) env.task_id = 0 then
              NodeUI.print_node_preamble ~full_shape:false ptr.id;
            raise e
        in
        let result = interpret_binop op v1 v2 in
        if !debug_trace_interpretation then
          Caml.Format.printf "TRACE: %a [%a -> %a] (%f) <- %f}\n%!" Sexp.pp_hum
            ([%sexp_of: NodeUI.tensor_ptr] ptr)
            Sexp.pp_hum
            ([%sexp_of: index array] indices)
            Sexp.pp_hum
            ([%sexp_of: int array] idcs)
            v1 result;
        set_from_float ptr idcs result
    | Set (ptr, indices, llv) -> (
        if !debug_trace_interpretation then
          Caml.Format.printf "{TRACE: %a [%a] <- ...\n%!" Sexp.pp_hum
            ([%sexp_of: NodeUI.tensor_ptr] ptr)
            Sexp.pp_hum
            ([%sexp_of: index array] indices);
        let idcs = lookup env indices in
        let result = loop_float env llv in
        if !debug_trace_interpretation then
          Caml.Format.printf "TRACE: %a [%a -> %a] (%f) <- %f}\n%!" Sexp.pp_hum
            ([%sexp_of: NodeUI.tensor_ptr] ptr)
            Sexp.pp_hum
            ([%sexp_of: index array] indices)
            Sexp.pp_hum
            ([%sexp_of: int array] idcs)
            (get_as_float ptr idcs) result;
        try set_from_float ptr idcs result
        with e ->
          Caml.Format.printf "ERROR: %a [%a -> %a] <- %f -- indices out of bounds\n%!" Sexp.pp_hum
            ([%sexp_of: NodeUI.tensor_ptr] ptr)
            Sexp.pp_hum
            ([%sexp_of: index array] indices)
            Sexp.pp_hum
            ([%sexp_of: int array] idcs)
            result;
          if Option.fold ~init:0 ~f:(fun _ (_, i) -> i) env.task_id = 0 then
            NodeUI.print_node_preamble ~full_shape:false ptr.id;
          raise e)
    | Set_local (id, llv) -> locals := Map.update !locals id ~f:(fun _ -> loop_float env llv)
    | Comment message when !with_debug && !executor_print_comments -> Stdio.printf "%s\n%!" message
    | Dynamic_indices
        { tensor = { id; field = Value }; tensor_idcs; dynamic_idcs; target_dims; body; slice = _ } ->
        dynamic_indices env (N.get id).value ~tensor_idcs ~dynamic_idcs ~target_dims body
    | Dynamic_indices
        { tensor = { id; field = Grad }; tensor_idcs; dynamic_idcs; target_dims; body; slice = _ } ->
        dynamic_indices env (Option.value_exn (N.get id).grad) ~tensor_idcs ~dynamic_idcs ~target_dims body
    | Comment c ->
        if !debug_trace_interpretation then (
          Caml.Format.printf "TRACE: %s -- prior state of nodes: {\n%!" c;
          NodeUI.print_decimals_precision := 9;
          for i = 1 to Node.global.unique_id - 1 do
            Caml.Format.printf "TRACE: %a\n%!" PrintBox_text.pp
              (NodeUI.to_printbox ~single_node:true ~with_grad:true ~depth:9 i)
          done;
          Caml.Format.printf "}\n%!")
  and loop_float env llv =
    let open Float in
    let loop = loop_float env in
    match llv with
    | Constant c -> c
    | Get (ptr, indices) ->
        if !debug_trace_interpretation then
          Caml.Format.printf "{TRACE: %a [%a] -> ...\n%!" Sexp.pp_hum
            ([%sexp_of: NodeUI.tensor_ptr] ptr)
            Sexp.pp_hum
            ([%sexp_of: index array] indices);
        let idcs = lookup env indices in
        let result =
          try get_as_float ptr idcs
          with e ->
            Caml.Format.printf "ERROR: %a [%a -> %a] -- indices out of bounds\n%!" Sexp.pp_hum
              ([%sexp_of: NodeUI.tensor_ptr] ptr)
              Sexp.pp_hum
              ([%sexp_of: index array] indices)
              Sexp.pp_hum
              ([%sexp_of: int array] idcs);
            (* if Int.(task_id () = 0) then *) NodeUI.print_node_preamble ~full_shape:false ptr.id;
            raise e
        in
        if !debug_trace_interpretation then
          Caml.Format.printf "TRACE: %a [%a -> %a] -> %f}\n%!" Sexp.pp_hum
            ([%sexp_of: NodeUI.tensor_ptr] ptr)
            Sexp.pp_hum
            ([%sexp_of: index array] indices)
            Sexp.pp_hum
            ([%sexp_of: int array] idcs)
            result;
        result
    | Local_scope { id; prec = _; body; orig_indices } ->
        if !debug_trace_interpretation then
          Caml.Format.printf "{TRACE: %a [%a] <-> ...\n%!" Sexp.pp_hum
            ([%sexp_of: NodeUI.tensor_ptr] id.tensor)
            Sexp.pp_hum
            ([%sexp_of: index array] orig_indices);
        let old_locals = !locals in
        locals := Map.update !locals id ~f:(fun _ -> 0.0);
        loop_proc env body;
        let result = Map.find_exn !locals id in
        locals := old_locals;
        let idcs = lookup env orig_indices in
        if !debug_trace_interpretation then
          Caml.Format.printf "TRACE: %a [%a / %a] (%f) <-> %f}\n%!" Sexp.pp_hum
            ([%sexp_of: NodeUI.tensor_ptr] id.tensor)
            Sexp.pp_hum
            ([%sexp_of: index array] orig_indices)
            Sexp.pp_hum
            ([%sexp_of: int array] idcs)
            (get_as_float id.tensor idcs) result;
        result
    | Get_local id -> Map.find_exn !locals id
    | Get_global Task_id -> Float.of_int @@ snd @@ Option.value_exn env.task_id
    | Get_global (C_function _) -> failwith "NOT IMPLEMENTED YET: jit-dynloading C calls in the interpreter"
    | Binop (Arg1, llv1, _llv2) -> loop llv1
    | Binop (Arg2, _llv1, llv2) -> loop llv2
    | Binop (op, llv1, llv2) -> interpret_binop op (loop llv1) (loop llv2)
    | Unop (Identity, llv) -> loop llv
    | Unop (Relu, llv) ->
        let v = loop llv in
        if v > 0.0 then v else 0.0
  and dynamic_indices env tensor ~tensor_idcs ~dynamic_idcs ~target_dims body =
    let env =
      Array.foldi dynamic_idcs ~init:env ~f:(fun provider_dim env key ->
          let idcs = lookup ~provider_dim env tensor_idcs in
          let actual =
            try N.get_as_int tensor idcs
            with e ->
              Caml.Format.printf "ERROR: dynamic index at [%a -> %a] -- indices out of bounds\n%!" Sexp.pp_hum
                ([%sexp_of: index array] tensor_idcs)
                Sexp.pp_hum
                ([%sexp_of: int array] idcs);
              raise e
          in
          let target_dim = target_dims.(provider_dim).dim in
          { env with dyn_env = Map.add_exn ~key ~data:(actual % target_dim) env.dyn_env })
    in
    loop_proc env body
  in
  loop_proc
    {
      env = Map.Poly.empty;
      dyn_env = Map.Poly.empty;
      task_id = Option.map task_id ~f:(fun i -> (global_task_id_idx, i));
      sample_num = None;
    }
    llc

let code_sexp_margin = ref 200

let fprint_code ppf c =
  (* TODO: something nicely concise. *)
  Caml.Format.pp_set_margin ppf !code_sexp_margin;
  Caml.Format.fprintf ppf "%s" @@ Sexp.to_string_hum @@ sexp_of_t c

let fprint_low_level ppf c =
  (* TODO: something nicely concise. *)
  Caml.Format.pp_set_margin ppf !code_sexp_margin;
  Caml.Format.fprintf ppf "%s" @@ Sexp.to_string_hum @@ sexp_of_low_level Unit.sexp_of_t (to_low_level c)

let interpreter_error_message ~name ~prefix ?extra_error_msg ~contents exc =
  let backtrace = Caml.Printexc.get_backtrace () in
  let exc_str = Caml.Printexc.to_string exc in
  let message = Buffer.create (String.length contents + String.length backtrace + String.length exc_str) in
  let msg = Buffer.add_string message in
  msg name;
  msg ": ";
  msg prefix;
  msg exc_str;
  msg "\n";
  msg backtrace;
  (match extra_error_msg with
  | None -> ()
  | Some extra ->
      msg "\nIn the context of:\n";
      msg extra);
  msg contents;
  Buffer.contents message

(** *** Optimization *** *)

type virtualize_settings = {
  mutable enable_device_only : bool;
  mutable max_visits : int;
  mutable inline_constants : bool;
  mutable always_inline_dynamic_indexing : bool;
}

let virtualize_settings =
  {
    enable_device_only = true;
    max_visits = 3;
    inline_constants = true;
    always_inline_dynamic_indexing = true;
  }

type visits =
  | Visits of int
  | Recurrent  (** A [Recurrent] visit is when there is an access prior to any assignment in an update. *)
[@@deriving sexp, equal, variants]

type traced_tensor = {
  id : int;
  kind : NodeUI.data_kind;
  prec : NodeUI.prec;
  mutable computations : (index array option * unit low_level) list;
      (** The computations (of the data node) are retrieved for optimization just as they are populated,
          so that the inlined code corresponds precisely to the changes to the tensors that would happen
          up till that point. Within the code blocks paired with an index tuple, all assignments and accesses
          must happen via the index tuple; if this is not the case for some assignment, the node cannot
          be virtual. Currently, we only allow for-loop symbols in assignment indices of virtual nodes. *)
  assignments : int array Hash_set.t;
  accesses : (int array, visits) Hashtbl.t;
      (** For dynamic indexes, we take a value of 0. This leads to an overestimate of visits, which is safe. *)
  mutable non_virtual : bool;
      (** If false, this tensor is never materialized, its computations are inlined on a per-scalar basis.
          A tensor that already exists (has size > 0) will not be virtual. *)
  mutable non_device_only : bool;
      (** If false, this node is only materialized on the devices it is computed on, it is not persisted
          outside of a step update. *)
  mutable scalar : float option;
  mutable zero_initialized : bool;
  mutable zeroed_out : bool;
  mutable read_before_write : bool;  (** The node is read before it is written (i.e. it is recurrent). *)
  mutable reduced_racyness : bool;
      (** If true, the only non-constant writes into the tensor are updates, and a constant write is never
          the last write. An update is a read immediately followed by a write of the same cell, as in
          accumulation operations [=+], [=*]. *)
  mutable last_write_non_update : bool;
  mutable is_dynamic_slice : bool;
      (** If true, the tensor is a dynamic slice (i.e. a result of dynamic indexing). *)
  mutable is_replicable : bool;
      (** If true, in case of parallelization, the tensor's update is identical on all tasks, and the final
          update of the host should only be performed from one task rather than accumulated across tasks. *)
}
[@@deriving sexp_of]

let get_node store (uid : NodeUI.tensor_ptr) =
  Hashtbl.find_or_add store uid ~default:(fun () ->
      let n = NodeUI.get uid.id in
      let never_virtual =
        match uid.field with NodeUI.Value -> n.value_never_virtual | NodeUI.Grad -> n.grad_never_virtual
      in
      let never_device_only =
        match uid.field with
        | NodeUI.Value -> n.value_never_device_only
        | NodeUI.Grad -> n.grad_never_device_only
      in
      let non_virtual = never_virtual || NodeUI.host_size_in_bytes uid > 0 in
      let non_device_only = never_device_only || NodeUI.host_size_in_bytes uid > 0 in
      {
        id = uid.id;
        kind = uid.field;
        prec = NodeUI.node_prec uid;
        computations = [];
        assignments = Hash_set.Poly.create ();
        accesses = Hashtbl.Poly.create ();
        non_virtual;
        non_device_only;
        scalar = None;
        zero_initialized = false;
        zeroed_out = false;
        read_before_write = false;
        reduced_racyness = true;
        last_write_non_update = false;
        is_dynamic_slice = false;
        is_replicable = true;
      })

let get_other_node traced_store ptr =
  get_node traced_store
    { ptr with field = (if NodeUI.equal_data_kind ptr.NodeUI.field Value then Grad else Value) }

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

let precompute_constants ?idcs traced_store top_node llv =
  let exception Non_literal of int in
  let rec loop llv =
    match llv with
    | Constant c -> c
    | Get (ptr, indices) ->
        let node = get_node traced_store ptr in
        Array.iter indices ~f:(function Shape.Fixed_idx 0 -> () | _ -> raise @@ Non_literal 1);
        Option.value_or_thunk node.scalar ~default:(fun () -> raise @@ Non_literal 2)
    | Local_scope { id; orig_indices; _ } ->
        let node = get_node traced_store id.tensor in
        Array.iter orig_indices ~f:(function Shape.Fixed_idx 0 -> () | _ -> raise @@ Non_literal 3);
        Option.value_or_thunk node.scalar ~default:(fun () -> raise @@ Non_literal 4)
    | Get_local scope_id ->
        let node = get_node traced_store scope_id.tensor in
        Option.value_or_thunk node.scalar ~default:(fun () -> raise @@ Non_literal 5)
    | Get_global _ -> raise @@ Non_literal 9
    | Binop (Arg1, llv1, _llv2) -> loop llv1
    | Binop (Arg2, _llv1, llv2) -> loop llv2
    | Binop (Add, llv1, llv2) -> Float.(loop llv1 + loop llv2)
    | Binop (Mul, llv1, llv2) -> Float.(loop llv1 * loop llv2)
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
  let n = NodeUI.get top_node.id in
  let never_virtual =
    match top_node.kind with NodeUI.Value -> n.value_never_virtual | NodeUI.Grad -> n.grad_never_virtual
  in
  let never_device_only =
    match top_node.kind with
    | NodeUI.Value -> n.value_never_device_only
    | NodeUI.Grad -> n.grad_never_device_only
  in
  try
    if never_virtual || never_device_only then raise @@ Non_literal 8;
    if (not n.literal) && Hashtbl.exists top_node.accesses ~f:is_recurrent then raise @@ Non_literal 6;
    (match idcs with
    | None -> ()
    | Some idcs ->
        if Array.exists idcs ~f:(function Shape.Fixed_idx 0 -> false | _ -> true) then raise @@ Non_literal 7);
    top_node.scalar <- Some (loop llv)
  with Non_literal i ->
    if !with_debug && !debug_trace_interpretation then
      Caml.Format.printf "TRACE: Node #%d is non-literal because no. %d\n%!" n.id i;
    (* In principle we might conclude again that the node is to be inlined as scalar, that's OK. *)
    top_node.scalar <- None

let visit is_assigned old =
  if not is_assigned then Recurrent
  else match old with None -> Visits 1 | Some (Visits i) -> Visits (i + 1) | Some Recurrent -> Recurrent

let visit_llc traced_store reverse_node_map ~max_visits llc =
  let is_too_many = function Visits i -> i > max_visits | Recurrent -> true in
  let lookup ?provider_dim dem_env indices =
    (* For dynamic indexes, we take a value of 0. This leads to an overestimate of visits, which is safe. *)
    Array.map indices ~f:(function
      | Shape.Fixed_idx i -> i
      | Iterator s -> Map.find_exn dem_env.env s
      (* | Dynamic_recipient s | Frozen_recipient s -> Map.find_exn dem_env.dyn_env s *)
      | Dynamic_recipient _ | Frozen_recipient _ -> 0
      | Dynamic_provider _ -> Option.value_exn provider_dim
      | Special_iterator Task_id -> snd @@ Option.value_exn dem_env.task_id
      | Special_iterator Sample_num -> snd @@ Option.value_exn dem_env.sample_num)
  in
  let cached_replicable ptr = (get_node traced_store ptr).is_replicable in
  let rec loop_proc env llc =
    let loop = loop_proc env in
    match llc with
    | Lines body -> Array.iter ~f:loop body
    | For_loop { index = { loop_sym = Iterator sym; uid }; from_; to_ = _; body; trace_it = false } ->
        loop_proc { env with env = Map.add_exn ~key:{ sym; uid } ~data:from_ env.env } body
    | For_loop { index = { loop_sym = Iterator _; _ } as index; from_; to_; body; trace_it = true } ->
        for data = from_ to to_ do
          loop_proc (environment_add index data env) body
        done
    | For_loop { index; from_ = _; to_ = _; body; trace_it = _ } ->
        loop_proc (environment_add index 0 env) body
    | Rebalance (_, cs) -> Array.iter ~f:loop cs
    | If_task_id_is { for_task_id; body } ->
        if Option.fold ~init:for_task_id ~f:(fun _ (_, i) -> i) env.task_id = for_task_id then loop body
    | Zero_out ptr ->
        let traced : traced_tensor = get_node traced_store ptr in
        if Hash_set.is_empty traced.assignments && Hashtbl.is_empty traced.accesses then
          traced.zero_initialized <- true;
        traced.zeroed_out <- true
    | Set (tensor, idcs, llv) ->
        loop_float env llv;
        (* get_node will initialize reduced_racyness to true.  *)
        let traced : traced_tensor = get_node traced_store tensor in
        Hash_set.add traced.assignments (lookup env idcs);
        if virtualize_settings.inline_constants then precompute_constants ~idcs traced_store traced llv;
        if check_not_replicable ~cached_replicable llc then traced.is_replicable <- false;
        (match llv with
        | Get (tensor2, idcs2) ->
            traced.last_write_non_update <- true;
            if Array.exists idcs2 ~f:Shape.(fun i -> is_dynamic_recipient i || is_frozen_recipient i) then
              assert traced.is_dynamic_slice;
            if (NodeUI.get tensor2.id).literal then () else traced.reduced_racyness <- false
        | Binop (_, Get (tensor2, idcs2), _)
          when NodeUI.equal_tensor_ptr tensor tensor2 && [%equal: index array] idcs idcs2 ->
            traced.last_write_non_update <- false
        | Binop (_, _, Get (tensor2, idcs2))
          when NodeUI.equal_tensor_ptr tensor tensor2 && [%equal: index array] idcs idcs2 ->
            traced.last_write_non_update <- false
        | Constant _ -> traced.last_write_non_update <- true
        | _ ->
            traced.reduced_racyness <- false;
            traced.last_write_non_update <- true);
        Array.iter idcs ~f:(function
          | Shape.Dynamic_provider _ ->
              if not virtualize_settings.always_inline_dynamic_indexing then traced.non_virtual <- true
          | Dynamic_recipient _ | Frozen_recipient _ ->
              (* TODO(#133): We don't support inlining tensors with complex write patterns. *)
              traced.non_virtual <- true
          | Fixed_idx _ | Special_iterator _ -> ()
          | Iterator s ->
              let old_tensor = Hashtbl.find_or_add reverse_node_map s ~default:(fun () -> tensor) in
              (* TODO(#134): this prevents multiple virtual tensors from sharing for loops. *)
              assert (NodeUI.equal_tensor_ptr old_tensor tensor))
    | Set_local (_, llv) -> loop_float env llv
    | Comment _ -> ()
    | Dynamic_indices { tensor; tensor_idcs; dynamic_idcs; target_dims; body; slice } ->
        let traced_tensor = get_node traced_store tensor in
        (* if not virtualize_settings.always_inline_dynamic_indexing then (
           traced_tensor.non_virtual <- true;
           traced_tensor.scalar <- None); *)
        Option.iter slice ~f:(fun ptr -> (get_node traced_store ptr).is_dynamic_slice <- true);
        dynamic_indices traced_tensor env ~tensor_idcs ~dynamic_idcs ~target_dims body
  and loop_float env llv =
    let loop = loop_float env in
    match llv with
    | Constant _ -> ()
    | Get (ptr, indices) ->
        let tensor : traced_tensor = get_node traced_store ptr in
        let at_pos = lookup env indices in
        Hashtbl.update tensor.accesses at_pos
          ~f:(visit (tensor.zeroed_out || Hash_set.mem tensor.assignments at_pos))
    | Local_scope { body; _ } -> loop_proc env body
    | Get_local _ -> ()
    | Get_global _ -> ()
    | Binop (Arg1, llv1, _llv2) -> loop llv1
    | Binop (Arg2, _llv1, llv2) -> loop llv2
    | Binop (_, llv1, llv2) ->
        loop llv1;
        loop llv2
    | Unop (_, llv) -> loop llv
  and dynamic_indices node env ~tensor_idcs ~dynamic_idcs ~target_dims:_ body =
    let env =
      Array.foldi dynamic_idcs ~init:env ~f:(fun provider_dim env key ->
          let at_pos = lookup ~provider_dim env tensor_idcs in
          Hashtbl.update node.accesses at_pos
            ~f:(visit (node.zeroed_out || Hash_set.mem node.assignments at_pos));
          { env with dyn_env = Map.add_exn ~key ~data:0 env.dyn_env })
    in
    loop_proc env body
  in
  loop_proc
    {
      env = Map.Poly.empty;
      dyn_env = Map.Poly.empty;
      task_id = Some (global_task_id_idx, 0);
      sample_num = None;
    }
    llc;
  Hashtbl.iter traced_store ~f:(fun traced ->
      if traced.last_write_non_update then traced.reduced_racyness <- false;
      if
        (not (virtualize_settings.always_inline_dynamic_indexing && traced.is_dynamic_slice))
        && Hashtbl.exists traced.accesses ~f:is_too_many
      then traced.non_virtual <- true;
      if Hashtbl.exists traced.accesses ~f:is_recurrent then (
        traced.non_virtual <- true;
        traced.non_device_only <- true;
        traced.read_before_write <- true))

type tensor_ptrs = NodeUI.tensor_ptr Set.Poly.t

let sexp_of_tensor_ptrs ts = Fn.compose [%sexp_of: NodeUI.tensor_ptr list] Set.to_list ts

let process_computation node top_llc =
  let exception Non_virtual in
  let top_data = { NodeUI.id = node.id; field = node.kind } in
  let at_idcs = ref None in
  let has_setter = ref false in
  let check_idcs indices =
    (match !at_idcs with
    | None -> at_idcs := Some indices
    | Some at -> if not @@ [%equal: index array] at indices then raise Non_virtual);
    (* TODO(#133): For non-recursive accesses, non-linearity is not supported yet. *)
    let syms =
      Set.Poly.of_array
      @@ Array.filter_map indices
           ~f:
             Shape.(
               function
               | Special_iterator _ | Fixed_idx _ | Dynamic_recipient _ | Frozen_recipient _
               | Dynamic_provider _ ->
                   None
               | Iterator s -> Some s)
    in
    let num_syms = Array.count indices ~f:(function Iterator _ -> true | _ -> false) in
    if Set.length syms <> num_syms then raise Non_virtual
  in
  (* Traverse the float code too, for completeness / future use-cases. *)
  let rec loop_proc ~(env_dom : unit environment) llc =
    let loop = loop_proc ~env_dom in
    match llc with
    | Lines body -> Array.iter ~f:loop body
    | For_loop { trace_it = false; _ } ->
        (* assert false *)
        raise Non_virtual
    | For_loop { index; body; from_ = _; to_ = _; trace_it = true } ->
        loop_proc ~env_dom:(environment_add index () env_dom) body
    | Rebalance (_, cs) -> Array.iter ~f:loop cs
    | If_task_id_is { for_task_id = _; body } -> loop body
    | Zero_out tensor -> if NodeUI.equal_tensor_ptr tensor top_data then has_setter := true
    | Set (tensor, indices, llv) ->
        if NodeUI.equal_tensor_ptr tensor top_data then (
          check_idcs indices;
          has_setter := true)
        else
          (* Check for escaping variables. *)
          Array.iter indices ~f:(function
            | (Iterator _ | Special_iterator _) as idx ->
                if not @@ environment_mem (to_loop_index idx) env_dom then (
                  if !with_debug then
                    Caml.Format.printf "INFO: Inlining candidate has an escaping variable %a:@ %a\n%!"
                      Sexp.pp_hum
                      (sexp_of_loop_index @@ to_loop_index idx)
                      Sexp.pp_hum
                      ([%sexp_of: unit low_level] top_llc);
                  raise Non_virtual)
            | _ -> ());
        loop_float ~env_dom llv
    | Set_local (_, llv) -> loop_float ~env_dom llv
    | Comment _ -> ()
    | Dynamic_indices { body; _ } -> loop body
  and loop_float ~env_dom llv =
    match llv with
    | Constant _ -> ()
    | Get (tensor, idcs) ->
        if NodeUI.equal_tensor_ptr tensor top_data then check_idcs idcs
        else
          (* Check for escaping variables. *)
          Array.iter idcs ~f:(function
            | (Iterator _ | Special_iterator _) as idx ->
                if not @@ environment_mem (to_loop_index idx) env_dom then (
                  if !with_debug then
                    Caml.Format.printf "INFO: Inlining candidate has an escaping variable %a:@ %a\n%!"
                      Sexp.pp_hum
                      (sexp_of_loop_index @@ to_loop_index idx)
                      Sexp.pp_hum
                      ([%sexp_of: unit low_level] top_llc);
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
    if node.non_virtual then raise Non_virtual;
    (* FIXME: we allow task_id, but not sample_num, that's not consistent (shouldn't allow task_id?) *)
    loop_proc
      ~env_dom:
        {
          env = Map.Poly.empty;
          dyn_env = Map.Poly.empty;
          task_id = Some (global_task_id_idx, ());
          sample_num = None;
        }
      top_llc;
    if not !has_setter then raise Non_virtual;
    node.computations <- (!at_idcs, top_llc) :: node.computations
  with Non_virtual -> node.non_virtual <- true

let inline_computation ~id node call_args =
  let exception Non_virtual in
  let make_subst i lhs_ind =
    let rhs_ind = call_args.(i) in
    match lhs_ind, rhs_ind with
    | Shape.Iterator lhs_s, Shape.Iterator rhs_s -> Some (lhs_s, rhs_s)
    | _ when Shape.equal_axis_index equal_sym_index lhs_ind rhs_ind -> None
    | _ -> raise Non_virtual
  in
  let at_data = { id = node.id; NodeUI.field = node.kind } in
  (* In the order of computation. *)
  let loop_proc (def_args, def) : unit low_level option =
    let env =
      match def_args with
      | None -> Map.Poly.empty
      | Some def_args -> Map.Poly.of_alist_exn @@ Array.to_list @@ Array.filter_mapi def_args ~f:make_subst
    in
    let subst env = function Shape.Iterator s when Map.mem env s -> Shape.Iterator (Map.find_exn env s) | idx -> idx in
    let rec loop env llc : unit low_level option =
      match llc with
      | Lines body ->
          let body = Array.filter_map ~f:(loop env) body in
          if Array.is_empty body then None else Some (Lines body)
      | For_loop { trace_it = false; _ } -> assert false
      | For_loop { index = { loop_sym = Iterator sym; uid }; body; _ } when Map.mem env { sym; uid } ->
          loop env body
      | For_loop { index = { loop_sym = Iterator sym; uid }; from_; to_; body; trace_it } ->
          (* Freshen the binding. *)
          let fresh = new_sym_uid sym in
          let env = Map.Poly.add_exn ~key:{ sym; uid } ~data:fresh env in
          Option.map ~f:(fun body ->
              For_loop
                { index = { loop_sym = Iterator fresh.sym; uid = fresh.uid }; from_; to_; body; trace_it })
          @@ loop env body
      | For_loop { index; from_; to_; body; trace_it } ->
          Option.map ~f:(fun body -> For_loop { index; from_; to_; body; trace_it }) @@ loop env body
      | Rebalance (s, cs) ->
          (* FIXME: NOT IMPLEMENTED YET *)
          let cs = Array.filter_map ~f:(loop env) cs in
          if Array.is_empty cs then None else Some (Rebalance (s, cs))
      | If_task_id_is { for_task_id = _; body } ->
          (* Inlined computations are governed by the inlining context, and cannot interfere across tasks.
             Undo parallelization. *)
          loop env body
      | Zero_out tensor when NodeUI.equal_tensor_ptr tensor at_data -> Some (Set_local (id, Constant 0.0))
      | Set (tensor, indices, llv) when NodeUI.equal_tensor_ptr tensor at_data ->
          assert ([%equal: index array option] (Some indices) def_args);
          Some (Set_local (id, loop_float env llv))
      | Zero_out _ -> None
      | Set _ -> None
      | Set_local (id, llv) -> Some (Set_local (id, loop_float env llv))
      | Comment _ -> Some llc
      | Dynamic_indices dyn_idcs ->
          (* Dynamic_indices is introduced by to_low_level in the innermost scope. *)
          Option.map ~f:(fun body -> Dynamic_indices { dyn_idcs with body }) @@ loop env dyn_idcs.body
    and loop_float env llv : float low_level =
      match llv with
      | Constant _ -> llv
      | Get (tensor, indices) when NodeUI.equal_tensor_ptr tensor at_data ->
          assert ([%equal: index array option] (Some indices) def_args);
          Get_local id
      | Get (tensor, indices) -> Get (tensor, Array.map ~f:(subst env) indices)
      | Local_scope { id; prec; body; orig_indices } ->
          Local_scope
            { id; prec; body = Option.value_exn @@ loop env body; orig_indices = Array.map ~f:(subst env) orig_indices }
      | Get_local _ -> llv
      | Get_global _ -> llv
      | Binop (op, llv1, llv2) -> Binop (op, loop_float env llv1, loop_float env llv2)
      | Unop (op, llv) -> Unop (op, loop_float env llv)
    in
    loop env def
  in
  try Some (Lines (Array.filter_opt @@ Array.of_list_rev_map ~f:loop_proc node.computations))
  with Non_virtual ->
    node.non_virtual <- true;
    None

let virtual_llc traced_store reverse_node_map (llc : unit low_level) : unit low_level =
  (* The current position is within scope of the definitions of the process_for virtual tensors. *)
  let rec loop_proc ~(process_for : tensor_ptrs) (llc : unit low_level) : unit low_level =
    let loop = loop_proc ~process_for in
    match llc with
    | Lines body -> Lines (Array.map ~f:loop body)
    | For_loop ({ index = { loop_sym = Iterator sym; uid }; body; _ } as for_config) -> (
        match Hashtbl.find reverse_node_map { sym; uid } with
        | Some tensor when not @@ Set.mem process_for tensor ->
            let node : traced_tensor = Hashtbl.find_exn traced_store tensor in
            let result = loop_proc ~process_for:(Set.add process_for tensor) llc in
            if not node.non_virtual then process_computation node result;
            result
        | _ -> For_loop { for_config with body = loop body })
    | For_loop ({ body; _ } as for_config) -> For_loop { for_config with body = loop body }
    | Rebalance (s, cs) ->
        (* FIXME: NOT IMPLEMENTED YET *)
        Rebalance (s, Array.map ~f:loop cs)
    | If_task_id_is { for_task_id; body } -> If_task_id_is { for_task_id; body = loop body }
    | Zero_out ptr ->
        let tensor : traced_tensor = Hashtbl.find_exn traced_store ptr in
        if (not @@ Set.mem process_for ptr) && not tensor.non_virtual then process_computation tensor llc;
        llc
    | Set (ptr, indices, llv) ->
        let tensor : traced_tensor = Hashtbl.find_exn traced_store ptr in
        let next = if tensor.non_virtual then process_for else Set.add process_for ptr in
        let result = Set (ptr, indices, loop_float ~process_for:next llv) in
        if (not @@ Set.mem process_for ptr) && not tensor.non_virtual then process_computation tensor result;
        result
    | Set_local (id, llv) -> Set_local (id, loop_float ~process_for llv)
    | Comment _ -> llc
    | Dynamic_indices dyn_idcs -> (
        match dyn_idcs.slice with
        | None -> Dynamic_indices { dyn_idcs with body = loop dyn_idcs.body }
        | Some ptr ->
            let tensor : traced_tensor = Hashtbl.find_exn traced_store ptr in
            let next = if tensor.non_virtual then process_for else Set.add process_for ptr in
            let result = Dynamic_indices { dyn_idcs with body = loop_proc ~process_for:next dyn_idcs.body } in
            if (not @@ Set.mem process_for ptr) && not tensor.non_virtual then
              process_computation tensor result;
            result)
  and loop_float ~(process_for : tensor_ptrs) (llv : float low_level) : float low_level =
    match llv with
    | Constant _ -> llv
    | Get (tensor, _) when Set.mem process_for tensor ->
        (* [Get_local] will replace this [Get] during [inline_computation] if [tensor] remains virtual. *)
        llv
    | Get (tensor, indices) ->
        let node : traced_tensor = get_node traced_store tensor in
        if node.non_virtual then llv
        else
          let id = get_scope tensor in
          Option.value ~default:llv
          @@ Option.map (inline_computation ~id node indices) ~f:(fun body ->
                 Local_scope { id; prec = node.prec; body; orig_indices = indices })
    | Local_scope opts ->
        Local_scope { opts with body = loop_proc ~process_for:(Set.add process_for opts.id.tensor) opts.body }
    | Get_local _ -> llv
    | Get_global _ -> llv
    | Binop (op, llv1, llv2) -> Binop (op, loop_float ~process_for llv1, loop_float ~process_for llv2)
    | Unop (op, llv) -> Unop (op, loop_float ~process_for llv)
  in
  loop_proc ~process_for:Set.Poly.empty llc

let cleanup_virtual_llc traced_store reverse_node_map (llc : unit low_level) : unit low_level =
  let is_inline tensor =
    let node = Hashtbl.find_exn traced_store tensor in
    (virtualize_settings.inline_constants && Option.is_some node.scalar) || not node.non_virtual
  in
  (* The current position is within scope of the definitions of the process_for virtual tensors. *)
  let rec loop_proc ~balanced ~env_dom (llc : unit low_level) : unit low_level option =
    let loop = loop_proc ~balanced ~env_dom in
    match llc with
    | Lines body ->
        let body = Array.filter_map ~f:loop body in
        if Array.is_empty body then None else Some (Lines body)
    | For_loop ({ index = { loop_sym = Iterator sym; uid }; body; _ } as for_config) -> (
        let env_dom = Set.add env_dom { sym; uid } in
        match Hashtbl.find reverse_node_map { sym; uid } with
        | Some tensor ->
            if is_inline tensor then None
            else
              Option.map ~f:(fun body -> For_loop { for_config with body })
              @@ loop_proc ~balanced ~env_dom body
        | None ->
            Option.map ~f:(fun body -> For_loop { for_config with body }) @@ loop_proc ~balanced ~env_dom body
        )
    | For_loop ({ index = { loop_sym = Special_iterator _; _ }; body; _ } as for_config) ->
        Option.map ~f:(fun body -> For_loop { for_config with body }) @@ loop_proc ~balanced ~env_dom body
    | Rebalance (s, cs) ->
        let cs = Array.filter_map cs ~f:loop in
        if Array.is_empty cs then None else Some (Rebalance (s, cs))
    | If_task_id_is { for_task_id; body } ->
        Option.map ~f:(fun body -> If_task_id_is { for_task_id; body }) @@ loop body
    | Zero_out tensor -> if is_inline tensor then None else Some llc
    | Set (tensor, indices, llv) ->
        if is_inline tensor then None
        else (
          assert (Array.for_all indices ~f:(function Shape.Iterator s -> Set.mem env_dom s | _ -> true));
          Some (Set (tensor, indices, loop_float ~balanced ~env_dom llv)))
    | Set_local (id, llv) ->
        let node = Hashtbl.find_exn traced_store id.tensor in
        if virtualize_settings.inline_constants && Option.is_some node.scalar then None
        else (
          assert (not node.non_virtual);
          Some (Set_local (id, loop_float ~balanced ~env_dom llv)))
    | Comment _ -> Some llc
    | Dynamic_indices dyn_idcs ->
        assert (
          Array.for_all dyn_idcs.tensor_idcs ~f:(function Shape.Iterator s -> Set.mem env_dom s | _ -> true));
        (* Dynamic indices use a separate environment. Note that dynamic indices are do not appear
           in the LHSes of slice definitions, so are not erased when inlining. *)
        Option.map ~f:(fun body -> Dynamic_indices { dyn_idcs with body }) @@ loop dyn_idcs.body
  and loop_float ~balanced ~env_dom (llv : float low_level) : float low_level =
    let loop = loop_float ~balanced ~env_dom in
    match llv with
    | Constant _ -> llv
    | Get (tensor, indices) -> (
        let node = get_node traced_store tensor in
        match node.scalar with
        | Some c when virtualize_settings.inline_constants -> Constant c
        | _ ->
            if not node.non_virtual then
              Caml.Format.printf "WARNING: unexpected Get of a virtual tensor, details:@ %a\n%!" Sexp.pp_hum
                (sexp_of_traced_tensor node);
            assert (Array.for_all indices ~f:(function Shape.Iterator s -> Set.mem env_dom s | _ -> true));
            llv)
    | Local_scope { id; prec; body; orig_indices } -> (
        let node = get_node traced_store id.tensor in
        match node.scalar with
        | Some c when virtualize_settings.inline_constants -> Constant c
        | _ ->
            assert (
              Array.for_all orig_indices ~f:(function Shape.Iterator s -> Set.mem env_dom s | _ -> true));
            if node.non_virtual then Get (id.tensor, orig_indices)
            else
              Option.value_or_thunk ~default:(fun () ->
                  Caml.Format.printf
                    "WARNING: unexpected non-eliminable virtual tensor:@ %a@ Compilation data:@ %a@ \
                     Compilation for the other tensor:@ %a@ Node:@ %a\n\
                     %!"
                    Sexp.pp_hum
                    (NodeUI.sexp_of_tensor_ptr id.tensor)
                    Sexp.pp_hum (sexp_of_traced_tensor node) Sexp.pp_hum
                    (sexp_of_traced_tensor @@ get_other_node traced_store id.tensor)
                    Sexp.pp_hum
                    (NodeUI.sexp_of_t @@ NodeUI.get id.tensor.id);
                  Get (id.tensor, orig_indices))
              @@ Option.map ~f:(fun body -> Local_scope { id; prec; orig_indices; body })
              @@ loop_proc ~balanced ~env_dom body)
    | Get_local id -> (
        let node = get_node traced_store id.tensor in
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

type traced_store = (NodeUI.tensor_ptr, traced_tensor) Base.Hashtbl.t

let optimize_proc llc : traced_store * unit low_level =
  let traced_store : (NodeUI.tensor_ptr, traced_tensor) Hashtbl.t = Hashtbl.Poly.create () in
  (* Identifies the computations that the code block associated with the symbol belongs to. *)
  let reverse_node_map = Hashtbl.Poly.create () in
  let result =
    visit_llc traced_store reverse_node_map ~max_visits:virtualize_settings.max_visits llc;
    cleanup_virtual_llc traced_store reverse_node_map @@ virtual_llc traced_store reverse_node_map llc
  in
  (traced_store, result)

let compile_proc ~name ~for_step_update:_ proc =
  let llc = to_low_level proc in
  if !with_debug && !keep_files_in_run_directory then (
    let fname = name ^ "-unoptimized.llc" in
    let f = Stdio.Out_channel.create fname in
    let ppf = Caml.Format.formatter_of_out_channel f in
    Caml.Format.pp_set_margin ppf !code_sexp_margin;
    Caml.Format.fprintf ppf "%a%!" Sexp.pp_hum (sexp_of_low_level Unit.sexp_of_t llc);
    let fname = name ^ ".hlc" in
    let f = Stdio.Out_channel.create fname in
    let ppf = Caml.Format.formatter_of_out_channel f in
    Caml.Format.pp_set_margin ppf !code_sexp_margin;
    Caml.Format.fprintf ppf "%a%!" Sexp.pp_hum (sexp_of_t proc));
  let result = optimize_proc llc in
  if !with_debug && !keep_files_in_run_directory then (
    let fname = name ^ ".llc" in
    let f = Stdio.Out_channel.create fname in
    let ppf = Caml.Format.formatter_of_out_channel f in
    Caml.Format.pp_set_margin ppf !code_sexp_margin;
    Caml.Format.fprintf ppf "%a%!" Sexp.pp_hum (sexp_of_low_level Unit.sexp_of_t @@ snd result));
  (* if for_step_update then
     Hashtbl.iter (fst result) ~f:(fun n -> if n.read_before_write then (NodeUI.get n.id).is_recurrent <- true); *)
  result

let interpret_task_id_func ~name:_ ((_traced_store : traced_store), compiled) ~task_id =
  if !debug_trace_interpretation && task_id = 0 then (
    Caml.Format.set_margin !code_sexp_margin;
    Caml.Format.printf "TRACE: Interpreted program:@ %a\n%!" Sexp.pp_hum
    @@ sexp_of_low_level Unit.sexp_of_t compiled);
  interpret_code ~task_id compiled

let interpret_unit_func ~name:_ ((_ : traced_store), compiled) () = interpret_code compiled

module CDSL = struct
  let dim = Shape.dim
  let parallel = Shape.parallel
  let minibatch = Shape.minibatch
  let value_of_id id : NodeUI.tensor_ptr = { id; field = Value }
  let grad_of_id id : NodeUI.tensor_ptr = { id; field = Grad }
  let data_of_node field (n : NodeUI.t) : NodeUI.tensor_ptr = { id = n.id; field }
  let single = NodeUI.single
  let double = NodeUI.double
  let executor_print_comments = executor_print_comments
  let keep_files_in_run_directory = keep_files_in_run_directory
  let with_debug = with_debug
  let virtualize_settings = virtualize_settings
  let debug_trace_interpretation = debug_trace_interpretation
  let code_sexp_margin = code_sexp_margin
end
