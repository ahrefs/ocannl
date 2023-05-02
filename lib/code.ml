open Base
(** The code for operating on n-dimensional arrays. *)

(** *** High-level representation. *** *)
type data_kind = Value | Grad [@@deriving sexp, equal, hash]

type tensor_ptr = { id : int; field : data_kind } [@@deriving sexp, equal, hash]
type binop = Add | Mul | ToPowOf | Relu_gate | Arg2 | Arg1 [@@deriving sexp]
type unop = Identity | Relu [@@deriving sexp]

module N = Ocannl_runtime.Node

let get_tensor tensor =
  let n = N.get tensor.id in
  match tensor.field with Value -> Some n.value | Grad -> n.grad

(** Initializes a tensor by filling in the corresponding numbers, at the appropriate precision. *)
type init_op = N.init_op =
  | Constant_fill of float array
      (** Fills in the numbers where the rightmost axis is contiguous, looping over the provided values
      if necessary. *)
  | Range_over_offsets
      (** Fills in the offset number of each cell (i.e. how many cells away it is from the beginning). *)
  | Standard_uniform  (** Draws the values from U(0,1). *)
[@@deriving sexp]

type prec =
  | Void_prec : prec
  (* | Bit_as_bool: (bool, bit_as_bool_nd) precision *)
  | Byte_as_int_prec : (int, N.byte_as_int_nd) N.precision -> prec
  | Half_as_int_prec : (int, N.half_as_int_nd) N.precision -> prec
  (* | Bit_prec: (float, (bool, Bigarray.bool_elt, Bigarray.c_layout) bigarray) N.precision -> prec*)
  (* | Byte_prec: (float, (float, Bigarray.float8_elt, Bigarray.c_layout) bigarray) N.precision -> prec *)
  (* | Half_prec: (float, (float, Bigarray.float16_elt, Bigarray.c_layout) bigarray) N.precision -> prec*)
  | Single_prec : (float, N.single_nd) N.precision -> prec
  | Double_prec : (float, N.double_nd) N.precision -> prec

let byte_as_int = Byte_as_int_prec N.Byte_as_int
let half_as_int = Half_as_int_prec N.Half_as_int
let single = Single_prec N.Single
let double = Double_prec N.Double

let sexp_of_prec = function
  | Void_prec -> Sexp.Atom "Void_prec"
  | Byte_as_int_prec _ -> Sexp.Atom "Byte_as_int_prec"
  | Half_as_int_prec _ -> Sexp.Atom "Half_as_int_prec"
  | Single_prec _ -> Sexp.Atom "Single_prec"
  | Double_prec _ -> Sexp.Atom "Double_prec"

let prec_of_sexp = function
  | Sexp.Atom "Void" -> Void_prec
  | Sexp.Atom "Byte_as_int_prec" -> byte_as_int
  | Sexp.Atom "Half_as_int_prec" -> half_as_int
  | Sexp.Atom "Single_prec" -> single
  | Sexp.Atom "Double_prec" -> double
  | Sexp.List _ -> invalid_arg "prec_of_sexp: expected atom, found list"
  | Sexp.Atom s -> invalid_arg @@ "prec_of_sexp: unknown precision " ^ s

let node_prec tensor =
  match get_tensor tensor with
  | None -> Void_prec
  | Some (N.Byte_as_int_nd _) -> byte_as_int
  | Some (N.Half_as_int_nd _) -> half_as_int
  | Some (N.Single_nd _) -> single
  | Some (N.Double_nd _) -> double

(** Resets a tensor by performing the specified computation or data fetching. *)
type fetch_op =
  | Zeros
  | Ones
  | Synthetic of t
  | Imported of { func : string (* params: Gccjit.rvalue list *) }
[@@deriving sexp]

and t =
  | Par of t * t  (** These tasks can proceed in parallel, there is no interaction. *)
  | ParHint of t * t
      (** Computing [ParHint (c1, c2)] can proceed in parallel on [c1] and [c2], but when [c2] reads values
      that [c1] writes, the writes in [c1] must occur before the reads in [c2]. If a backend does not
      support detection of when [ParHint (c1, c2)] is safe to parallelize, it should provide an option
      [force_unsafe_parhint] which always parallelizes. *)
  | Seq of t * t
      (** These tasks can only benefit from mutual parallelism via operator fusion / loop fusion. *)
  | Accum_binop of {
      zero_out : bool;
      accum : binop;
      op : binop;
      lhs : tensor_ptr;
      rhs1 : tensor_ptr;
      rhs2 : tensor_ptr;
      projections : unit -> Shape.projections;
    }
  | Accum_unop of {
      zero_out : bool;
      accum : binop;
      op : unop;
      lhs : tensor_ptr;
      rhs : tensor_ptr;
      projections : unit -> Shape.projections;
    }
  | Fetch of { tensor : tensor_ptr; fetch_op : fetch_op }
  | Block_comment of string * t
  | Noop
[@@deriving sexp]

(** Dynamically loading a program bounds a callback to one of the two global routine slots:
    the [session_step_update], or the temporary slot to be read by the caller right after compilation. *)
type program = Suspension of t | Session_step_update of t [@@deriving sexp]

(** Name of a program that can be used as part of a file name. *)
let get_name = function Suspension _ -> "suspension" | Session_step_update _ -> "session_step_update"

type create = { tensor : tensor_ptr; dims : unit -> int array; init_op : init_op }
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
        if equal_tensor_ptr tensor lhs then rm true t else rm false c
    | Par (t1, t2) -> Par (rm true t1, rm true t2)
    | ParHint (t1, t2) -> ParHint (rm true t1, rm true t2)
    | Seq (t1, t2) -> Seq (rm true t1, rm true t2)
    | (Accum_binop { lhs; _ } | Accum_unop { lhs; _ }) when equal_tensor_ptr tensor lhs -> Noop
    | c -> c
  in
  rm true c

let all_parallel = List.fold ~init:Noop ~f:(fun sts st -> Par (st, sts))
let sequential = List.fold_right ~init:Noop ~f:(fun st sts -> Seq (st, sts))

type scope_id = { tensor : tensor_ptr; scope_id : int } [@@deriving sexp, equal, hash]
(** *** Low-level representation. *)

let get_scope =
  let uid = ref 0 in
  fun tensor ->
    Int.incr uid;
    { tensor; scope_id = !uid }

type sym_index = { sym : Shape.symbol; uid : int } [@@deriving sexp, equal, compare]
type index = sym_index Shape.axis_index [@@deriving sexp, equal, compare]

let new_sym_index =
  let uid = ref 0 in
  fun sym ->
    Int.incr uid;
    { sym; uid = !uid }

(** Cases: [unit low_level] -- code, [float low_level] -- single number at some precision. *)
type _ low_level =
  | Comment : string -> unit low_level
  | Lines : unit low_level array -> unit low_level
  | For_loop : { index : sym_index; from_ : int; to_ : int; body : unit low_level } -> unit low_level
  | Fill : { tensor : tensor_ptr; value : float low_level } -> unit low_level
  | Dynamic_indices : {
      tensor : tensor_ptr;
      tensor_idcs : index array;
      dynamic_idcs : Shape.symbol array;
      target_dims : int array;
      body : unit low_level;
    }
      -> unit low_level
  | Set : tensor_ptr * index array * float low_level -> unit low_level
  | Set_local : scope_id * float low_level -> unit low_level
  | Local_scope : {
      id : scope_id;
      prec : prec;
      body : unit low_level;
      idcs_for_debug : index array;
    }
      -> float low_level
  | Get_local : scope_id -> float low_level
  | Get : tensor_ptr * index array -> float low_level
  | Binop : binop * float low_level * float low_level -> float low_level
  | Unop : unop * float low_level -> float low_level
  | Constant : float -> float low_level
[@@deriving sexp_of]

type low_level_program = Assign_suspension of unit low_level | Assign_session_step_update of unit low_level
[@@deriving sexp_of]

let rec to_low_level (code : t) : unit low_level =
  match code with
  | Accum_binop { zero_out; accum; op; lhs; rhs1; rhs2; projections } ->
      let projections = projections () in
      let lhs_idx = Shape.(derive_index projections.product_iterators projections.project_lhs) in
      let rhs1_idx = Shape.(derive_index projections.product_iterators projections.project_rhs1) in
      let rhs2_idx =
        match projections.project_rhs2 with
        | None -> invalid_arg "accum_binop: projections missing project_rhs2"
        | Some rhs2 -> Shape.(derive_index projections.product_iterators rhs2)
      in
      let basecase rev_iters =
        let iters = Array.of_list_rev rev_iters in
        let rhs1_idcs = rhs1_idx iters in
        let rhs2_idcs = rhs2_idx iters in
        let lhs_idcs = lhs_idx iters in
        let lhs_ll = Get (lhs, lhs_idcs) in
        let rhs1_ll = Get (rhs1, rhs1_idcs) in
        let rhs2_ll = Get (rhs2, rhs2_idcs) in
        let body = Set (lhs, lhs_idcs, Binop (accum, lhs_ll, Binop (op, rhs1_ll, rhs2_ll))) in
        match Array.find rhs2_idcs ~f:Shape.is_dynamic_provider with
        | Some (Dynamic_provider { idcs = dynamic_idcs; target_dims }) ->
            Dynamic_indices { tensor = rhs2; tensor_idcs = rhs2_idcs; dynamic_idcs; target_dims; body }
        | _ -> (
            match Array.find rhs1_idcs ~f:Shape.is_dynamic_provider with
            | Some (Dynamic_provider { idcs = dynamic_idcs; target_dims }) ->
                Dynamic_indices { tensor = rhs1; tensor_idcs = rhs1_idcs; dynamic_idcs; target_dims; body }
            | _ -> (
                match Array.find lhs_idcs ~f:Shape.is_dynamic_provider with
                | Some (Dynamic_provider { idcs = dynamic_idcs; target_dims }) ->
                    Dynamic_indices { tensor = lhs; tensor_idcs = lhs_idcs; dynamic_idcs; target_dims; body }
                | _ -> body))
      in
      let rec loop rev_iters = function
        | [], [] -> basecase rev_iters
        | dim :: product, it :: iters ->
            let it = new_sym_index it in
            For_loop { index = it; from_ = 0; to_ = dim - 1; body = loop (it :: rev_iters) (product, iters) }
        | _ -> invalid_arg "Code.to_low_level: Accum_binop projections dims-iterators mismatch"
      in
      let for_loops =
        loop [] (Array.to_list projections.product_space, Array.to_list projections.product_iterators)
      in
      if zero_out then Lines [| to_low_level (Fetch { tensor = lhs; fetch_op = Zeros }); for_loops |]
      else for_loops
  | Accum_unop { zero_out; accum; op; lhs; rhs; projections } ->
      let projections = projections () in
      let lhs_idx = Shape.(derive_index projections.product_iterators projections.project_lhs) in
      let rhs_idx = Shape.(derive_index projections.product_iterators projections.project_rhs1) in
      let basecase rev_iters =
        let iters = Array.of_list_rev rev_iters in
        let lhs_idcs = lhs_idx iters in
        let lhs_ll = Get (lhs, lhs_idcs) in
        let rhs_ll = Get (rhs, rhs_idx iters) in
        Set (lhs, lhs_idcs, Binop (accum, lhs_ll, Unop (op, rhs_ll)))
      in
      let rec loop rev_iters = function
        | [], [] -> basecase rev_iters
        | dim :: product, it :: iters ->
            let it = new_sym_index it in
            For_loop { index = it; from_ = 0; to_ = dim - 1; body = loop (it :: rev_iters) (product, iters) }
        | _ -> invalid_arg "Code.to_low_level: Accum_unop projections dims-iterators mismatch"
      in
      let for_loops =
        loop [] (Array.to_list projections.product_space, Array.to_list projections.product_iterators)
      in
      if zero_out then Lines [| to_low_level (Fetch { tensor = lhs; fetch_op = Zeros }); for_loops |]
      else for_loops
  | Block_comment (s, c) -> Lines [| Comment s; to_low_level c |]
  | Noop -> Lines [||]
  | Par (c1, c2) | ParHint (c1, c2) | Seq (c1, c2) -> (
      (* TODO: this ignores parallelization altogether, don't! *)
      let ll1 = to_low_level c1 in
      let ll2 = to_low_level c2 in
      match (ll1, ll2) with
      | Lines ls1, Lines ls2 -> Lines (Array.append ls1 ls2)
      | _, Lines ls2 -> Lines (Array.append [| ll1 |] ls2)
      | Lines ls1, _ -> Lines (Array.append ls1 [| ll2 |])
      | _ -> Lines [| ll1; ll2 |])
  | Fetch { tensor; fetch_op = Zeros } -> Fill { tensor; value = Constant 0. }
  | Fetch { tensor; fetch_op = Ones } -> Fill { tensor; value = Constant 1. }
  | Fetch { tensor = _; fetch_op = Synthetic gen } -> to_low_level gen
  | Fetch { tensor = _; fetch_op = Imported { func = _ } } ->
      (* FIXME: NOT IMPLEMENTED YET *)
      failwith "NOT IMPLEMENTED YET"

let to_low_level_program prog : low_level_program =
  match prog with
  | Suspension proc -> Assign_suspension (to_low_level proc)
  | Session_step_update proc -> Assign_session_step_update (to_low_level proc)

let interpreter_print_comments = ref false
let keep_files_in_run_directory = ref false
let with_debug = ref true
let debug_virtual_nodes = ref false

type int_env = (sym_index, int) Base.Map.Poly.t * (Shape.symbol, int) Base.Map.Poly.t

let sexp_of_int_env (env, dyn_env) =
  [%sexp_of: (sym_index * int) list * (Shape.symbol * int) list] (Map.to_alist env, Map.to_alist dyn_env)

let set_from_float ptr idcs value =
  match ptr.field with
  | Value -> N.set_from_float (N.get ptr.id).value idcs value
  | Grad -> N.set_from_float (Option.value_exn (N.get ptr.id).grad) idcs value

let interpret_llc llc =
  (* Local scope ids can be non-unique due to inlining. *)
  let locals = ref Map.Poly.empty in
  let lookup ?provider_dim (env, dyn_env) indices =
    Array.map indices
      ~f:
        Shape.(
          function
          | Fixed_idx i -> i
          | Iterator s -> Map.find_exn env s
          | Dynamic_recipient s -> Map.find_exn dyn_env s
          | Dynamic_provider _ -> Option.value_exn provider_dim)
  in
  let rec loop_proc env llc : unit =
    let loop = loop_proc env in
    match llc with
    | Lines body -> Array.iter ~f:loop body
    | For_loop { index = key; from_; to_; body } ->
        for data = from_ to to_ do
          loop_proc (Map.add_exn ~key ~data @@ fst env, snd env) body
        done
    | Fill { tensor = { id; field = Value }; value } ->
        N.fill_from_float (N.get id).value @@ loop_float env value
    | Fill { tensor = { id; field = Grad }; value } ->
        N.fill_from_float (Option.value_exn (N.get id).grad) @@ loop_float env value
    | Set (ptr, indices, llv) -> set_from_float ptr (lookup env indices) @@ loop_float env llv
    | Set_local (id, llv) -> locals := Map.update !locals id ~f:(fun _ -> loop_float env llv)
    | Comment message when !with_debug && !interpreter_print_comments -> Stdio.printf "%s\n%!" message
    | Dynamic_indices { tensor = { id; field = Value }; tensor_idcs; dynamic_idcs; target_dims; body } ->
        dynamic_indices env (N.get id).value ~tensor_idcs ~dynamic_idcs ~target_dims body
    | Dynamic_indices { tensor = { id; field = Grad }; tensor_idcs; dynamic_idcs; target_dims; body } ->
        dynamic_indices env (Option.value_exn (N.get id).grad) ~tensor_idcs ~dynamic_idcs ~target_dims body
    | Comment _ -> ()
  and loop_float env llv =
    let open Float in
    let loop = loop_float env in
    match llv with
    | Constant c -> c
    | Get ({ id; field = Value }, indices) -> N.get_as_float (N.get id).value @@ lookup env indices
    | Get ({ id; field = Grad }, indices) ->
        N.get_as_float (Option.value_exn (N.get id).grad) @@ lookup env indices
    | Local_scope { id; prec = _; body; idcs_for_debug } ->
        let old_locals = !locals in
        locals := Map.update !locals id ~f:(fun _ -> 0.0);
        loop_proc env body;
        let result = Map.find_exn !locals id in
        locals := old_locals;
        if !debug_virtual_nodes then set_from_float id.tensor (lookup env idcs_for_debug) result;
        result
    | Get_local id -> Map.find_exn !locals id
    | Binop (Arg1, llv1, _llv2) -> loop llv1
    | Binop (Arg2, _llv1, llv2) -> loop llv2
    | Binop (Add, llv1, llv2) -> loop llv1 + loop llv2
    | Binop (Mul, llv1, llv2) -> loop llv1 * loop llv2
    | Binop (ToPowOf, llv1, llv2) ->
        let v1 = loop llv1 in
        let v2 = loop llv2 in
        Float.(if is_integer v2 then int_pow v1 @@ to_int v2 else v1 ** v2)
    | Binop (Relu_gate, llv1, llv2) -> if loop llv1 > 0.0 then loop llv2 else 0.0
    | Unop (Identity, llv) -> loop llv
    | Unop (Relu, llv) ->
        let v = loop llv in
        if v > 0.0 then v else 0.0
  and dynamic_indices env tensor ~tensor_idcs ~dynamic_idcs ~target_dims body =
    let env =
      Array.foldi dynamic_idcs ~init:env ~f:(fun provider_dim env key ->
          let actual = N.get_as_int tensor @@ lookup ~provider_dim env tensor_idcs in
          (fst env, Map.add_exn ~key ~data:(actual % target_dims.(provider_dim)) @@ snd env))
    in
    loop_proc env body
  in
  loop_proc (Map.Poly.empty, Map.Poly.empty) llc

let interpret_llprog = function
  | Assign_suspension proc ->
      Ocannl_runtime.Node.most_recent_suspension := Some (fun () -> interpret_llc proc)
  | Assign_session_step_update proc ->
      Ocannl_runtime.Node.global.session_step_update := Some (fun () -> interpret_llc proc)

let fprint_code ppf c =
  (* TODO: something nicely concise. *)
  Caml.Format.fprintf ppf "%s" @@ Sexp.to_string_hum @@ sexp_of_t c

let fprint_low_level ppf c =
  (* TODO: something nicely concise. *)
  Caml.Format.fprintf ppf "%s" @@ Sexp.to_string_hum @@ sexp_of_low_level Unit.sexp_of_t (to_low_level c)

let fprint_program ppf prog =
  (* TODO: something nicely concise. *)
  Caml.Format.fprintf ppf "%s" @@ Sexp.to_string_hum @@ sexp_of_program prog

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

(*
let substitute_llc env llc =
  let rec loop (type a) (llc : a low_level) =
    match (llc : a low_level) with
    | Comment _ as c -> c
    | Lines cs -> Lines (Array.map cs ~f:loop)
    | For_loop { index; from_; to_; body } -> For_loop { index; from_; to_; body }
    | Fill _ -> _
    | Value_at_node_id _ -> _
    | Gradient_at_node_id _ -> _
    | Dynamic_indices _ -> _
    | Set (_, _, _) -> _
    | Get (_, _) -> _
    | Binop (_, _, _) -> _
    | Unop (_, _) -> _
    | Constant _ -> _
  in
  loop llc
*)

(** *** Optimization *** *)
type visits =
  | Visits of int
  | Recurrent  (** A [Recurrent] visit is when there is an access prior to any assignment in an update. *)
[@@deriving sexp, equal]

type data_node = {
  id : int;
  kind : data_kind;
  prec : prec;
  assign_index_bag : sym_index Hash_set.t;
  mutable computations : (index array option * unit low_level) list;
      (** The computations (of the data node) are retrieved for optimization just as they are populated,
          so that the inlined code corresponds precisely to the changes to the tensors that would happen
          up till that point. Within the code blocks paired with an index tuple, all assignments and accesses
          must happen via the index tuple; if this is not the case for some assignment, the node cannot
          be virtual. Currently, we only allow for-loop symbols in assignment indices of virtual nodes. *)
  accesses : (int array, visits) Hashtbl.t;
      (** For dynamic indexes, we take a value of 0. This leads to an overestimate of visits, which is safe. *)
  mutable fill : float low_level option;
  mutable non_virtual : bool;
}
[@@deriving sexp_of]

let global_node_store : (tensor_ptr, data_node) Hashtbl.t = Hashtbl.Poly.create ()
let reverse_node_map : (sym_index, tensor_ptr) Hashtbl.t = Hashtbl.Poly.create ()
(* Identifies the computations that the code block associated with the symbol belongs to. *)

let cleanup_session () =
  Hashtbl.clear global_node_store;
  Hashtbl.clear reverse_node_map

let get_node uid =
  Hashtbl.find_or_add global_node_store uid ~default:(fun () ->
      {
        id = uid.id;
        kind = uid.field;
        prec = node_prec uid;
        assign_index_bag = Hash_set.Poly.create ();
        computations = [];
        accesses = Hashtbl.Poly.create ();
        fill = None;
        non_virtual = false;
      })

let visit fill assign_bag old =
  if Option.is_none fill && Hash_set.is_empty assign_bag then Recurrent
  else match old with None -> Visits 1 | Some (Visits i) -> Visits (i + 1) | Some Recurrent -> Recurrent

let visit_llc ~max_visits ~consider_grads llc =
  let is_too_many = function Visits i -> i > max_visits | Recurrent -> true in
  let nodes = Hash_set.create (module Int) in
  let lookup ?provider_dim (env, dyn_env) indices =
    Array.map indices
      ~f:
        Shape.(
          function
          | Fixed_idx i -> i
          | Iterator s -> Map.find_exn env s
          | Dynamic_recipient s -> Map.find_exn dyn_env s
          | Dynamic_provider _ -> Option.value_exn provider_dim)
  in
  let rec loop_proc env llc : unit =
    let loop = loop_proc env in
    match llc with
    | Lines body -> Array.iter ~f:loop body
    | For_loop { index = key; from_; to_; body } ->
        for data = from_ to to_ do
          loop_proc (Map.add_exn ~key ~data @@ fst env, snd env) body
        done
    | Fill { tensor; value } ->
        loop_float env value;
        let data_node = get_node tensor in
        Hash_set.add nodes data_node.id;
        if not @@ Hashtbl.is_empty data_node.accesses then data_node.non_virtual <- true
        else data_node.fill <- Some value
    | Set (tensor, idcs, llv) ->
        loop_float env llv;
        Hash_set.add nodes tensor.id;
        let node = get_node tensor in
        Array.iter idcs ~f:(function
          | Shape.Fixed_idx _ | Shape.Dynamic_provider _ | Shape.Dynamic_recipient _ ->
              node.non_virtual <- true
          | Shape.Iterator s ->
              let old_tensor = Hashtbl.find_or_add reverse_node_map s ~default:(fun () -> tensor) in
              assert (equal_tensor_ptr old_tensor tensor);
              Hash_set.add node.assign_index_bag s)
    | Set_local (_, llv) -> loop_float env llv
    | Comment _ -> ()
    | Dynamic_indices { tensor; tensor_idcs; dynamic_idcs; target_dims; body } ->
        let data_node = get_node tensor in
        (* FIXME(132): implement virtual dynamic indices. *)
        data_node.non_virtual <- true;
        Hash_set.add nodes data_node.id;
        dynamic_indices data_node env ~tensor_idcs ~dynamic_idcs ~target_dims body
  and loop_float env llv =
    let loop = loop_float env in
    match llv with
    | Constant _ -> ()
    | Get (tensor, indices) ->
        let data_node = get_node tensor in
        Hash_set.add nodes data_node.id;
        Hashtbl.update data_node.accesses (lookup env indices)
          ~f:(visit data_node.fill data_node.assign_index_bag)
    | Local_scope { body; _ } -> loop_proc env body
    | Get_local _ -> ()
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
          Hashtbl.update node.accesses at_pos ~f:(visit node.fill node.assign_index_bag);
          (fst env, Map.add_exn ~key ~data:0 @@ snd env))
    in
    loop_proc env body
  in
  loop_proc (Map.Poly.empty, Map.Poly.empty) llc;
  Hash_set.iter nodes ~f:(fun node_id ->
      let value_node = get_node { id = node_id; field = Value } in
      if Hashtbl.exists value_node.accesses ~f:is_too_many then value_node.non_virtual <- true;
      if consider_grads then
        Option.iter
          (Hashtbl.find global_node_store { id = node_id; field = Grad })
          ~f:(fun grad_node ->
            if Hashtbl.exists grad_node.accesses ~f:is_too_many then grad_node.non_virtual <- true;
            (* Issue #135: For now, value and gradient are non-virtual reciprocically. *)
            if value_node.non_virtual then grad_node.non_virtual <- true;
            if grad_node.non_virtual then value_node.non_virtual <- true))

let process_computation node top_llc =
  let exception Non_virtual in
  let top_data = { id = node.id; field = node.kind } in
  let at_idcs = ref None in
  let has_setter = ref false in
  let check_idcs indices =
    (match !at_idcs with
    | None -> at_idcs := Some indices
    | Some at -> if not @@ [%equal: index array] at indices then raise Non_virtual);
    let syms =
      Set.Poly.of_array
      @@ Array.filter_map indices
           ~f:
             Shape.(
               function
               | Fixed_idx _ | Dynamic_recipient _ | Dynamic_provider _ -> None | Iterator s -> Some s)
    in
    if Set.length syms <> Array.length indices then raise Non_virtual
  in
  (* Traverse the float code too, for completeness / future use-cases. *)
  let rec loop_proc llc =
    match llc with
    | Lines body -> Array.iter ~f:loop_proc body
    | For_loop { index = _; from_ = _; to_ = _; body } -> loop_proc body
    | Fill { tensor; value } ->
        if equal_tensor_ptr tensor top_data then has_setter := true;
        loop_float value
    | Set (tensor, indices, llv) ->
        if equal_tensor_ptr tensor top_data then (
          check_idcs indices;
          has_setter := true);
        loop_float llv
    | Set_local (_, llv) -> loop_float llv
    | Comment _ -> ()
    | Dynamic_indices { body; _ } -> loop_proc body
  and loop_float llv =
    match llv with
    | Constant _ -> ()
    | Get (tensor, idcs) -> if equal_tensor_ptr tensor top_data then check_idcs idcs
    | Local_scope { body; _ } -> loop_proc body
    | Get_local _ -> ()
    | Binop (Arg1, llv1, _llv2) -> loop_float llv1
    | Binop (Arg2, _llv1, llv2) -> loop_float llv2
    | Binop (_, llv1, llv2) ->
        loop_float llv1;
        loop_float llv2
    | Unop (_, llv) -> loop_float llv
  in
  (* Issue #135: For now, value and gradient are non-virtual reciprocically. *)
  let other_node =
    get_node { id = node.id; field = (if equal_data_kind node.kind Value then Grad else Value) }
  in
  try
    if node.non_virtual then raise Non_virtual;
    if other_node.non_virtual then raise Non_virtual;
    loop_proc top_llc;
    if not !has_setter then raise Non_virtual;
    node.computations <- (!at_idcs, top_llc) :: node.computations;
    (NodeUI.get node.id).virtual_ <- true
  with Non_virtual ->
    (NodeUI.get node.id).virtual_ <- false;
    node.non_virtual <- true;
    other_node.non_virtual <- true

let inline_computation ~id node call_args =
  let make_subst lhs_ind rhs_ind =
    match lhs_ind with Shape.Iterator s -> (s, rhs_ind) | _ -> assert false
  in
  let at_data = { id = node.id; field = node.kind } in
  (* In the order of computation. *)
  let loop_proc (def_args, def) : unit low_level =
    let env =
      match def_args with
      | None -> Map.Poly.empty
      | Some def_args ->
          Map.Poly.of_alist_exn @@ Array.to_list @@ Array.map2_exn def_args call_args ~f:make_subst
    in
    let subst = function Shape.Iterator s when Map.mem env s -> Map.find_exn env s | idx -> idx in
    let rec loop llc : unit low_level =
      match llc with
      | Lines body -> Lines (Array.map ~f:loop body)
      | For_loop { index; body; _ } when Map.mem env index -> loop body
      | For_loop { index; from_; to_; body } -> For_loop { index; from_; to_; body = loop body }
      | Fill { tensor; value } when equal_tensor_ptr tensor at_data -> Set_local (id, loop_float value)
      | Fill { tensor; value } -> Fill { tensor; value = loop_float value }
      | Set (tensor, indices, llv) when equal_tensor_ptr tensor at_data ->
          assert ([%equal: index array option] (Some indices) def_args);
          Set_local (id, loop_float llv)
      | Set (tensor, indices, llv) ->
          (* Check we are not leaking binders. *)
          assert (Array.for_all indices ~f:(function Iterator i -> not @@ Map.mem env i | _ -> true));
          Set (tensor, indices, loop_float llv)
      | Set_local (id, llv) -> Set_local (id, loop_float llv)
      | Comment _ -> llc
      | Dynamic_indices ({ body; _ } as dyn_idcs) ->
          (* FIXME(132): implement virtual dynamic indices. *)
          Dynamic_indices { dyn_idcs with body = loop body }
    and loop_float llv : float low_level =
      match llv with
      | Constant _ -> llv
      | Get (tensor, indices) when equal_tensor_ptr tensor at_data ->
          assert ([%equal: index array option] (Some indices) def_args);
          Get_local id
      | Get (tensor, indices) -> Get (tensor, Array.map ~f:subst indices)
      | Local_scope { id; prec; body; idcs_for_debug } ->
          Local_scope { id; prec; body = loop body; idcs_for_debug = Array.map ~f:subst idcs_for_debug }
      | Get_local _ -> llv
      | Binop (Arg1, llv1, _llv2) -> loop_float llv1
      | Binop (Arg2, _llv1, llv2) -> loop_float llv2
      | Binop (op, llv1, llv2) -> Binop (op, loop_float llv1, loop_float llv2)
      | Unop (op, llv) -> Unop (op, loop_float llv)
    in
    loop def
  in
  Lines (Array.of_list_rev_map ~f:loop_proc node.computations)

type tensor_ptrs = tensor_ptr Set.Poly.t

let sexp_of_tensor_ptrs ts = [%sexp_of: tensor_ptr list] @@ Set.to_list ts
  (* The current position is within scope of the definitions of the process_for virtual tensors. *)
  let rec loop_proc ~(process_for : tensor_ptrs) (llc : unit low_level) : unit low_level option =
    let loop = loop_proc ~process_for in
    match llc with
    | Lines body ->
        let body = Array.filter_map ~f:loop body in
        if Array.is_empty body then None else Some (Lines body)
    | For_loop { index; from_; to_; body } -> (
        match Hashtbl.find reverse_node_map index with
        | Some tensor when not @@ Set.mem process_for tensor ->
            let node : data_node = Hashtbl.find_exn global_node_store tensor in
            if node.non_virtual then
              Option.map ~f:(fun body -> For_loop { index; from_; to_; body }) @@ loop body
            else (
              Option.iter
                (loop_proc ~process_for:(Set.add process_for tensor) llc)
                ~f:(process_computation node);
              None)
        | _ -> Option.map ~f:(fun body -> For_loop { index; from_; to_; body }) @@ loop body)
    | Fill { tensor; value } ->
        let node : data_node = Hashtbl.find_exn global_node_store tensor in
        if (not node.non_virtual) && (not @@ Set.mem process_for tensor) then (
          Option.iter (loop_proc ~process_for:(Set.add process_for tensor) llc) ~f:(process_computation node);
          None)
        else Some (Fill { tensor; value = loop_float ~process_for value })
    | Set (tensor, indices, llv) ->
        let node : data_node = Hashtbl.find_exn global_node_store tensor in
        if (not @@ Set.mem process_for tensor) && not node.non_virtual then (
          process_computation node llc;
          None)
        else Some (Set (tensor, indices, loop_float ~process_for llv))
    | Set_local (id, llv) -> Some (Set_local (id, loop_float ~process_for llv))
    | Comment _ -> Some llc
    | Dynamic_indices { tensor; tensor_idcs; dynamic_idcs; target_dims; body } ->
        (* FIXME(132): implement virtual dynamic indices. *)
        Option.map ~f:(fun body -> Dynamic_indices { tensor; tensor_idcs; dynamic_idcs; target_dims; body })
        @@ loop body
  and loop_float ~(process_for : tensor_ptrs) (llv : float low_level) : float low_level =
    match llv with
    | Constant _ -> llv
    | Get (tensor, _) when Set.mem process_for tensor -> llv
    | Get (tensor, indices) ->
        let node : data_node = get_node tensor in
        if node.non_virtual then llv
        else
          let id = get_scope tensor in
          let body = inline_computation ~id node indices in
          Local_scope { id; prec = node.prec; body; idcs_for_debug = indices }
    | Local_scope opts ->
        Local_scope
          {
            opts with
            body =
              Option.value_or_thunk ~default:(fun () -> opts.body)
              @@ loop_proc ~process_for:(Set.add process_for opts.id.tensor) opts.body;
          }
    | Get_local _ -> llv
    | Binop (Arg1, llv1, _llv2) -> loop_float ~process_for llv1
    | Binop (Arg2, _llv1, llv2) -> loop_float ~process_for llv2
    | Binop (op, llv1, llv2) -> Binop (op, loop_float ~process_for llv1, loop_float ~process_for llv2)
    | Unop (op, llv) -> Unop (op, loop_float ~process_for llv)
  in
  Option.value_exn @@ loop_proc ~process_for:Set.Poly.empty llc

type virtualize_settings = {
  mutable virtualize : bool;
  mutable max_visits : int;
  mutable consider_grads : bool;
}

let virtualize_settings = { virtualize = true; max_visits = 3; consider_grads = false }

let compile_program prog =
  let prog = to_low_level_program prog in
  if not virtualize_settings.virtualize then prog
  else
    match prog with
    | Assign_suspension proc ->
        visit_llc ~max_visits:virtualize_settings.max_visits
          ~consider_grads:virtualize_settings.consider_grads proc;
        Assign_suspension (virtual_llc proc)
    | Assign_session_step_update proc ->
        visit_llc ~max_visits:virtualize_settings.max_visits
          ~consider_grads:virtualize_settings.consider_grads proc;
        Assign_session_step_update (virtual_llc proc)

let interpret_program prog : string option =
  let llp = compile_program prog in
  let () = interpret_llprog llp in
  Some (Sexp.to_string_hum @@ sexp_of_low_level_program llp)

module CDSL = struct
  let value_of_id id : tensor_ptr = { id; field = Value }
  let grad_of_id id : tensor_ptr = { id; field = Grad }
  let data_of_node field n : tensor_ptr = { id = n.NodeUI.id; field }
  let interpreter_print_comments = interpreter_print_comments
  let keep_files_in_run_directory = keep_files_in_run_directory
  let with_debug = with_debug
  let virtualize_settings = virtualize_settings
  let debug_virtual_nodes = debug_virtual_nodes
end
