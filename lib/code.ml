open Base
(** The code for operating on n-dimensional arrays. *)

(** *** High-level representation. *** *)
type data_kind = Value | Grad [@@deriving sexp, equal, hash]

type data = { id : int; field : data_kind } [@@deriving sexp, equal, hash]
type binop = Add | Mul | ToPowOf | Relu_gate | Arg2 | Arg1 [@@deriving sexp]
type unop = Identity | Relu [@@deriving sexp]

module N = Ocannl_runtime.Node

let get_tensor data =
  let n = N.get data.id in
  match data.field with Value -> n.value | Grad -> Option.value_exn n.grad

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
  | Byte_as_int_prec _ -> Sexp.Atom "Byte_as_int_prec"
  | Half_as_int_prec _ -> Sexp.Atom "Half_as_int_prec"
  | Single_prec _ -> Sexp.Atom "Single_prec"
  | Double_prec _ -> Sexp.Atom "Double_prec"

let prec_of_sexp = function
  | Sexp.Atom "Byte_as_int_prec" -> byte_as_int
  | Sexp.Atom "Half_as_int_prec" -> half_as_int
  | Sexp.Atom "Single_prec" -> single
  | Sexp.Atom "Double_prec" -> double
  | Sexp.List _ -> invalid_arg "prec_of_sexp: expected atom, found list"
  | Sexp.Atom s -> invalid_arg @@ "prec_of_sexp: unknown precision " ^ s

let node_prec data =
  match get_tensor data with
  | N.Byte_as_int_nd _ -> byte_as_int
  | N.Half_as_int_nd _ -> half_as_int
  | N.Single_nd _ -> single
  | N.Double_nd _ -> double

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
      lhs : data;
      rhs1 : data;
      rhs2 : data;
      projections : unit -> Shape.projections;
    }
  | Accum_unop of {
      zero_out : bool;
      accum : binop;
      op : unop;
      lhs : data;
      rhs : data;
      projections : unit -> Shape.projections;
    }
  | Fetch of { tensor : data; fetch_op : fetch_op }
  | Block_comment of string * t
  | Noop
[@@deriving sexp]

(** Dynamically loading a program bounds a callback to one of the two global routine slots:
    the [session_step_update], or the temporary slot to be read by the caller right after compilation. *)
type program = Suspension of t | Session_step_update of t [@@deriving sexp]

(** Name of a program that can be used as part of a file name. *)
let get_name = function Suspension _ -> "suspension" | Session_step_update _ -> "session_step_update"

type create = { tensor : data; dims : unit -> int array; init_op : init_op }
(** Information to create a tensor, once its shape is inferred. *)

let remove_updates data c =
  let rec rm check = function
    | ( Par ((Accum_binop { lhs; _ } | Accum_unop { lhs; _ }), t)
      | ParHint ((Accum_binop { lhs; _ } | Accum_unop { lhs; _ }), t)
      | Seq ((Accum_binop { lhs; _ } | Accum_unop { lhs; _ }), t)
      | Par (t, (Accum_binop { lhs; _ } | Accum_unop { lhs; _ }))
      | ParHint (t, (Accum_binop { lhs; _ } | Accum_unop { lhs; _ }))
      | Seq (t, (Accum_binop { lhs; _ } | Accum_unop { lhs; _ })) ) as c
      when check ->
        if equal_data data lhs then rm true t else rm false c
    | Par (t1, t2) -> Par (rm true t1, rm true t2)
    | ParHint (t1, t2) -> ParHint (rm true t1, rm true t2)
    | Seq (t1, t2) -> Seq (rm true t1, rm true t2)
    | (Accum_binop { lhs; _ } | Accum_unop { lhs; _ }) when equal_data data lhs -> Noop
    | c -> c
  in
  rm true c

let all_parallel = List.fold ~init:Noop ~f:(fun sts st -> Par (st, sts))
let sequential = List.fold_right ~init:Noop ~f:(fun st sts -> Seq (st, sts))

(** *** Low-level representation. *)
type scope_id = Scope_id of int [@@deriving sexp, equal, hash]

let get_scope =
  let uid = ref 0 in
  fun () ->
    Int.incr uid;
    Scope_id !uid

(** Cases: [unit low_level] -- code, [float low_level] -- single number at some precision,
    [data low_level] -- a tensor. *)
type _ low_level =
  | Comment : string -> unit low_level
  | Lines : unit low_level array -> unit low_level
  | For_loop : { index : Shape.symbol; from_ : int; to_ : int; body : unit low_level } -> unit low_level
  | Fill : { tensor : data; value : float low_level } -> unit low_level
  | Dynamic_indices : {
      tensor : data;
      tensor_idcs : Shape.Symbolic_idcs.t;
      dynamic_idcs : Shape.Symbols.t;
      target_dims : Shape.Indices.t;
      body : unit low_level;
    }
      -> unit low_level
  | Set : data * Shape.Symbolic_idcs.t * float low_level -> unit low_level
  | Set_local : scope_id * float low_level -> unit low_level
  | Local_scope : scope_id * prec * unit low_level -> float low_level
  | Get_local : scope_id -> float low_level
  | Get : data * Shape.Symbolic_idcs.t -> float low_level
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

module CDSL = struct
  let value_of_id id : data = { id; field = Value }
  let grad_of_id id : data = { id; field = Grad }
  let data_of_node field n : data = { id = n.NodeUI.id; field }
  let interpreter_print_comments = interpreter_print_comments
  let keep_files_in_run_directory = keep_files_in_run_directory
  let with_debug = with_debug
end

let interpret_llc llc =
  let locals = Hashtbl.Poly.create () in
  let lookup ?provider_dim env indices =
    Array.map indices
      ~f:
        Shape.(
          function
          | Fixed_idx i -> i
          | Iterator s | Dynamic_recipient s -> Map.find_exn env s
          | Dynamic_provider _ -> Option.value_exn provider_dim)
  in
  let open Ocannl_runtime.Node in
  let rec loop_proc env llc : unit =
    let loop = loop_proc env in
    match llc with
    | Lines body -> Array.iter ~f:loop body
    | For_loop { index = key; from_; to_; body } ->
        for data = from_ to to_ do
          loop_proc (Map.add_exn ~key ~data env) body
        done
    | Fill { tensor = { id; field = Value }; value } -> fill_from_float (get id).value @@ loop_float env value
    | Fill { tensor = { id; field = Grad }; value } ->
        fill_from_float (Option.value_exn (get id).grad) @@ loop_float env value
    | Set ({ id; field = Value }, indices, llv) ->
        set_from_float (get id).value (lookup env indices) @@ loop_float env llv
    | Set ({ id; field = Grad }, indices, llv) ->
        set_from_float (Option.value_exn (get id).grad) (lookup env indices) @@ loop_float env llv
    | Set_local (id, llv) -> Hashtbl.update locals id ~f:(fun _ -> loop_float env llv)
    | Comment message when !with_debug && !interpreter_print_comments -> Stdio.printf "%s\n%!" message
    | Dynamic_indices { tensor = { id; field = Value }; tensor_idcs; dynamic_idcs; target_dims; body } ->
        dynamic_indices env (get id).value ~tensor_idcs ~dynamic_idcs ~target_dims body
    | Dynamic_indices { tensor = { id; field = Grad }; tensor_idcs; dynamic_idcs; target_dims; body } ->
        dynamic_indices env (Option.value_exn (get id).grad) ~tensor_idcs ~dynamic_idcs ~target_dims body
    | Comment _ -> ()
  and loop_float env llv =
    let open Float in
    let loop = loop_float env in
    match llv with
    | Constant c -> c
    | Get ({ id; field = Value }, indices) -> get_as_float (get id).value @@ lookup env indices
    | Get ({ id; field = Grad }, indices) ->
        get_as_float (Option.value_exn (get id).grad) @@ lookup env indices
    | Local_scope (id, _prec, body) ->
        Hashtbl.add_exn locals ~key:id ~data:0.0;
        loop_proc env body;
        Hashtbl.find_exn locals id
    | Get_local id -> Hashtbl.find_exn locals id
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
          let actual = get_as_int tensor @@ lookup ~provider_dim env tensor_idcs in
          Map.add_exn ~key ~data:(actual % target_dims.(provider_dim)) env)
    in
    loop_proc env body
  in
  loop_proc (Map.empty (module Shape.Symbol)) llc

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

let interpret_program prog : string option =
  let llp = to_low_level_program prog in
  let () = interpret_llprog llp in
  Some (Sexp.to_string_hum @@ sexp_of_low_level_program llp)

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
  assign_index_bag : Shape.symbol Hash_set.t;
  mutable computations : (Shape.Symbolic_idcs.t * unit low_level) list;
      (** The computations (of the data node) are retrieved for optimization just as they are populated,
          so that the inlined code corresponds precisely to the changes to the tensors that would happen
          up till that point. Within the code blocks paired with an index tuple, all assignments and accesses
          must happen via the index tuple; if this is not the case for some assignment, the node cannot
          be virtual. Currently, we only allow for-loop symbols in assignment indices of virtual nodes. *)
  accesses : (Shape.Indices.t, visits) Hashtbl.t;
      (** For dynamic indexes, we take a value of 0. This leads to an overestimate of visits, which is safe. *)
  mutable fill : float low_level option;
  mutable non_virtual : bool;
}
[@@deriving sexp_of]

let global_node_store : (data, data_node) Hashtbl.t = Hashtbl.Poly.create ()
let reverse_node_map : (Shape.symbol, data Hash_set.t) Hashtbl.t = Hashtbl.Poly.create ()
(* Identifies the computations that the code block associated with the symbol belongs to. *)

let cleanup_session () =
  Hashtbl.clear global_node_store;
  Hashtbl.clear reverse_node_map

let get uid =
  Hashtbl.find_or_add global_node_store uid ~default:(fun () ->
      {
        id = uid.id;
        kind = uid.field;
        prec = node_prec uid;
        assign_index_bag = Hash_set.Poly.create ();
        computations = [];
        accesses = Hashtbl.create (module Shape.Indices);
        fill = None;
        non_virtual = false;
      })

let visit fill assign_bag old =
  if Option.is_none fill && Hash_set.is_empty assign_bag then Recurrent
  else match old with None -> Visits 0 | Some (Visits i) -> Visits (i + 1) | Some Recurrent -> Recurrent

let visit_llc ~max_visits ~consider_grads llc =
  let is_too_many = function Visits i -> i > max_visits | Recurrent -> true in
  let nodes = Hash_set.create (module Int) in
  let lookup ?provider_dim env indices =
    Array.map indices
      ~f:
        Shape.(
          function
          | Fixed_idx i -> i
          | Iterator s | Dynamic_recipient s -> Map.find_exn env s
          | Dynamic_provider _ -> Option.value_exn provider_dim)
  in
  let rec loop_proc env llc : unit =
    let loop = loop_proc env in
    match llc with
    | Lines body -> Array.iter ~f:loop body
    | For_loop { index = key; from_; to_; body } ->
        for data = from_ to to_ do
          loop_proc (Map.add_exn ~key ~data env) body
        done
    | Fill { tensor; value } ->
        loop_float env value;
        let data_node = get tensor in
        Hash_set.add nodes data_node.id;
        if not @@ Hashtbl.is_empty data_node.accesses then data_node.non_virtual <- true
        else data_node.fill <- Some value
    | Set (data, idcs, llv) ->
        loop_float env llv;
        Hash_set.add nodes data.id;
        let node = get data in
        Array.iter idcs ~f:(function
          | Shape.Fixed_idx _ | Shape.Dynamic_provider _ | Shape.Dynamic_recipient _ ->
              node.non_virtual <- true
          | Shape.Iterator s ->
              let ns = Hashtbl.find_or_add reverse_node_map s ~default:Hash_set.Poly.create in
              Hash_set.add ns data;
              Hash_set.add node.assign_index_bag s)
    | Set_local (_, llv) -> loop_float env llv
    | Comment _ -> ()
    | Dynamic_indices { tensor; tensor_idcs; dynamic_idcs; target_dims; body } ->
        let data_node = get tensor in
        (* FIXME(132): implement virtual dynamic indices. *)
        data_node.non_virtual <- true;
        Hash_set.add nodes data_node.id;
        dynamic_indices data_node env ~tensor_idcs ~dynamic_idcs ~target_dims body
  and loop_float env llv =
    let loop = loop_float env in
    match llv with
    | Constant _ -> ()
    | Get (data, indices) ->
        let data_node = get data in
        Hash_set.add nodes data_node.id;
        Hashtbl.update data_node.accesses (lookup env indices)
          ~f:(visit data_node.fill data_node.assign_index_bag)
    | Local_scope (_, _, llc) -> loop_proc env llc
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
          Map.add_exn ~key ~data:0 env)
    in
    loop_proc env body
  in
  loop_proc (Map.empty (module Shape.Symbol)) llc;
  Hash_set.iter nodes ~f:(fun node_id ->
      let value_node = Hashtbl.find_exn global_node_store { id = node_id; field = Value } in
      if Hashtbl.exists value_node.accesses ~f:is_too_many then value_node.non_virtual <- true;
      if consider_grads then
        let grad_node = Hashtbl.find_exn global_node_store { id = node_id; field = Grad } in
        if Hashtbl.exists grad_node.accesses ~f:is_too_many then grad_node.non_virtual <- true)

let visit_llprog ?(max_visits = 3) ?(consider_grads = false) = function
  | Assign_suspension proc | Assign_session_step_update proc -> visit_llc ~max_visits ~consider_grads proc

let process_computation node top_llc : unit =
  let exception Non_virtual in
  let top_data = { id = node.id; field = node.kind } in
  let at_idcs = ref None in
  let check_idcs indices =
    let syms =
      Set.of_array (module Shape.Symbol)
      @@ Array.filter_map indices
           ~f:
             Shape.(
               function
               | Fixed_idx _ | Dynamic_recipient _ | Dynamic_provider _ -> None | Iterator s -> Some s)
    in
    if Set.length syms <> Array.length indices then raise Non_virtual;
    Option.iter !at_idcs ~f:(fun at -> if not @@ Shape.Symbolic_idcs.equal at indices then raise Non_virtual)
  in
  (* Traverse the float code too, for completeness / future use-cases. *)
  let rec loop_proc llc : unit =
    match llc with
    | Lines body -> Array.iter ~f:loop_proc body
    | For_loop { index = _; from_ = _; to_ = _; body } -> loop_proc body
    | Fill { tensor = _; value } -> loop_float value
    | Set (data, indices, llv) ->
        if equal_data data top_data then (
          check_idcs indices;
          at_idcs := Some indices);
        loop_float llv
    | Set_local (_, llv) -> loop_float llv
    | Comment _ -> ()
    | Dynamic_indices { body; _ } -> loop_proc body
  and loop_float llv : unit =
    match llv with
    | Constant _ -> ()
    | Get (data, idcs) -> if equal_data data top_data then check_idcs idcs
    | Local_scope (_, _, llc) -> loop_proc llc
    | Get_local _ -> ()
    | Binop (Arg1, llv1, _llv2) -> loop_float llv1
    | Binop (Arg2, _llv1, llv2) -> loop_float llv2
    | Binop (_, llv1, llv2) ->
        loop_float llv1;
        loop_float llv2
    | Unop (_, llv) -> loop_float llv
  in
  try
    loop_proc top_llc;
    match !at_idcs with
    | None -> raise Non_virtual
    | Some idcs -> node.computations <- (idcs, top_llc) :: node.computations
  with Non_virtual -> node.non_virtual <- true

let inline_computation ~id node call_args =
  let make_subst lhs_ind rhs_ind =
    match lhs_ind with Shape.Iterator s -> (s, rhs_ind) | _ -> assert false
  in
  let at_data = { id = node.id; field = node.kind } in
  (* In the order of computation. *)
  let loop_proc (def_args, def) : unit low_level =
    let env =
      Map.of_alist_exn (module Shape.Symbol)
      @@ Array.to_list
      @@ Array.map2_exn def_args call_args ~f:make_subst
    in
    let subst = function Shape.Iterator s when Map.mem env s -> Map.find_exn env s | idx -> idx in
    let rec loop llc : unit low_level =
      match llc with
      | Lines body -> Lines (Array.map ~f:loop body)
      | For_loop { index; body; _ } when Map.mem env index -> body
      | For_loop { index; from_; to_; body } -> For_loop { index; from_; to_; body = loop body }
      | Fill { tensor; value } when equal_data tensor at_data -> Set_local (id, loop_float value)
      | Fill { tensor; value } -> Fill { tensor; value = loop_float value }
      | Set (data, indices, llv) when equal_data data at_data ->
          assert (Shape.Symbolic_idcs.equal indices def_args);
          Set_local (id, loop_float llv)
      | Set (data, indices, llv) -> Set (data, indices, loop_float llv)
      | Set_local (id, llv) -> Set_local (id, loop_float llv)
      | Comment _ -> llc
      | Dynamic_indices ({ body; _ } as dyn_idcs) ->
          (* FIXME(132): implement virtual dynamic indices. *)
          Dynamic_indices { dyn_idcs with body = loop body }
    and loop_float llv : float low_level =
      match llv with
      | Constant _ -> llv
      | Get (data, indices) when equal_data data at_data ->
          assert (Shape.Symbolic_idcs.equal indices def_args);
          Get_local id
      | Get (data, indices) -> Get (data, Array.map ~f:subst indices)
      | Local_scope (id, prec, llc) -> Local_scope (id, prec, loop llc)
      | Get_local _ -> llv
      | Binop (Arg1, llv1, _llv2) -> loop_float llv1
      | Binop (Arg2, _llv1, llv2) -> loop_float llv2
      | Binop (op, llv1, llv2) -> Binop (op, loop_float llv1, loop_float llv2)
      | Unop (op, llv) -> Unop (op, loop_float llv)
    in
    loop def
  in
  Lines (Array.of_list_rev_map ~f:loop_proc node.computations)

let virtual_llc llc : unit low_level =
  let rec loop_proc llc : unit low_level option =
    match llc with
    | Lines body ->
        let body = Array.filter_map ~f:loop_proc body in
        if Array.is_empty body then None else Some (Lines body)
    | For_loop { index = key; from_; to_; body } ->
        let for_nodes = Hashtbl.find_or_add reverse_node_map ~default:Hash_set.Poly.create key in
        (* Keep the computation in the output if it has some non-virtual assignment. *)
        let should_keep =
          Hash_set.exists for_nodes ~f:(fun data ->
              let node = Hashtbl.find_exn global_node_store data in
              node.non_virtual)
        in
        Hash_set.iter for_nodes ~f:(fun data ->
            let node = Hashtbl.find_exn global_node_store data in
            if not node.non_virtual then process_computation node llc);
        if should_keep then
          Option.map ~f:(fun body -> For_loop { index = key; from_; to_; body }) @@ loop_proc body
        else None
    | Fill { tensor; value } ->
        let node = Hashtbl.find_exn global_node_store tensor in
        if not node.non_virtual then (
          process_computation node llc;
          None)
        else Some (Fill { tensor; value = loop_float value })
    | Set (data, indices, llv) -> Some (Set (data, indices, loop_float llv))
    | Set_local (id, llv) -> Some (Set_local (id, loop_float llv))
    | Comment _ -> Some llc
    | Dynamic_indices { tensor; tensor_idcs; dynamic_idcs; target_dims; body } ->
        (* FIXME(132): implement virtual dynamic indices. *)
        Option.map ~f:(fun body -> Dynamic_indices { tensor; tensor_idcs; dynamic_idcs; target_dims; body })
        @@ loop_proc body
  and loop_float llv : float low_level =
    match llv with
    | Constant _ -> llv
    | Get (data, indices) ->
        let node = get data in
        if node.non_virtual then llv
        else
          let id = get_scope () in
          let body = Option.value_exn @@ loop_proc @@ inline_computation ~id node indices in
          Local_scope (id, node.prec, body)
    | Local_scope (id, prec, llc) ->
        Local_scope (id, prec, Option.value_or_thunk ~default:(fun () -> llc) @@ loop_proc llc)
    | Get_local _ -> llv
    | Binop (Arg1, llv1, _llv2) -> loop_float llv1
    | Binop (Arg2, _llv1, llv2) -> loop_float llv2
    | Binop (op, llv1, llv2) -> Binop (op, loop_float llv1, loop_float llv2)
    | Unop (op, llv) -> Unop (op, loop_float llv)
  in
  Option.value_exn @@ loop_proc llc

let virtual_llprog = function Assign_suspension proc | Assign_session_step_update proc -> virtual_llc proc
