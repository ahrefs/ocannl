open Base
(** The code for operating on n-dimensional arrays. *)

(** *** High-level representation. *** *)

type data = { id : int; field : [ `Value | `Grad ] } [@@deriving sexp, equal, hash]
type binop = Add | Mul | ToPowOf | Relu_gate | Arg2 | Arg1 [@@deriving sexp]
type unop = Identity | Relu [@@deriving sexp]

module N = Ocannl_runtime.Node

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
    !uid

(** Cases: [unit low_level] -- code, [float low_level] -- single number at some precision,
    [data low_level] -- a tensor. *)
type _ low_level =
  | Comment : string -> unit low_level
  | Lines : unit low_level array -> unit low_level
  | For_loop : { index : Shape.symbol; from_ : int; to_ : int; body : unit low_level } -> unit low_level
  | Fill : { tensor : data low_level; value : float low_level } -> unit low_level
  | Value_at_node_id : int -> data low_level
  | Gradient_at_node_id : int -> data low_level
  | Dynamic_indices : {
      tensor : data low_level;
      tensor_idcs : Shape.Symbolic_idcs.t;
      dynamic_idcs : Shape.Symbols.t;
      target_dims : Shape.Indices.t;
      body : unit low_level;
    }
      -> unit low_level
  | Set : data low_level * Shape.Symbolic_idcs.t * float low_level -> unit low_level
  | Set_local : scope_id * float low_level -> unit low_level
  | Local_scope : scope_id * prec * unit low_level -> float low_level
  | Get_local : scope_id -> float low_level
  | Get : data low_level * Shape.Symbolic_idcs.t -> float low_level
  | Binop : binop * float low_level * float low_level -> float low_level
  | Unop : unop * float low_level -> float low_level
  | Constant : float -> float low_level
[@@deriving sexp_of]

let is_value_at_node_id = function Value_at_node_id _ -> true | _ -> false

type low_level_program = Assign_suspension of unit low_level | Assign_session_step_update of unit low_level
[@@deriving sexp_of]

let data_pointer (xhs : data) =
  match xhs.field with `Value -> Value_at_node_id xhs.id | `Grad -> Gradient_at_node_id xhs.id

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
      let lhs_tensor = data_pointer lhs in
      let lhs_it iters = Get (lhs_tensor, lhs_idx iters) in
      let basecase rev_iters =
        let iters = Array.of_list_rev rev_iters in
        let rhs1_idcs = rhs1_idx iters in
        let rhs2_idcs = rhs2_idx iters in
        let rhs1_tensor = data_pointer rhs1 in
        let rhs2_tensor = data_pointer rhs2 in
        let rhs1 = Get (rhs1_tensor, rhs1_idcs) in
        let rhs2 = Get (rhs2_tensor, rhs2_idcs) in
        let lhs_idcs = lhs_idx iters in
        let body = Set (lhs_tensor, lhs_idcs, Binop (accum, lhs_it iters, Binop (op, rhs1, rhs2))) in
        match Array.find rhs2_idcs ~f:Shape.is_dynamic_provider with
        | Some (Dynamic_provider { idcs = dynamic_idcs; target_dims }) ->
            Dynamic_indices { tensor = rhs2_tensor; tensor_idcs = rhs2_idcs; dynamic_idcs; target_dims; body }
        | _ -> (
            match Array.find rhs1_idcs ~f:Shape.is_dynamic_provider with
            | Some (Dynamic_provider { idcs = dynamic_idcs; target_dims }) ->
                Dynamic_indices
                  { tensor = rhs1_tensor; tensor_idcs = rhs1_idcs; dynamic_idcs; target_dims; body }
            | _ -> (
                match Array.find lhs_idcs ~f:Shape.is_dynamic_provider with
                | Some (Dynamic_provider { idcs = dynamic_idcs; target_dims }) ->
                    Dynamic_indices
                      { tensor = lhs_tensor; tensor_idcs = lhs_idcs; dynamic_idcs; target_dims; body }
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
      let lhs_ptr = data_pointer lhs in
      let lhs_it iters = Get (lhs_ptr, lhs_idx iters) in
      let rhs iters = Get (data_pointer rhs, rhs_idx iters) in
      let basecase rev_iters =
        let iters = Array.of_list_rev rev_iters in
        Set (lhs_ptr, lhs_idx iters, Binop (accum, lhs_it iters, Unop (op, rhs iters)))
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
  | Fetch { tensor; fetch_op = Zeros } -> Fill { tensor = data_pointer tensor; value = Constant 0. }
  | Fetch { tensor; fetch_op = Ones } -> Fill { tensor = data_pointer tensor; value = Constant 1. }
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
  let value_of_id id : data = { id; field = `Value }
  let grad_of_id id : data = { id; field = `Grad }
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
    | Fill { tensor = Value_at_node_id id; value } -> fill_from_float (get id).value @@ loop_float env value
    | Fill { tensor = Gradient_at_node_id id; value } ->
        fill_from_float (Option.value_exn (get id).grad) @@ loop_float env value
    | Set (Value_at_node_id id, indices, llv) ->
        set_from_float (get id).value (lookup env indices) @@ loop_float env llv
    | Set (Gradient_at_node_id id, indices, llv) ->
        set_from_float (Option.value_exn (get id).grad) (lookup env indices) @@ loop_float env llv
    | Set_local (id, llv) -> Hashtbl.update locals id ~f:(fun _ -> loop_float env llv)
    | Comment message when !with_debug && !interpreter_print_comments -> Stdio.printf "%s\n%!" message
    | Dynamic_indices { tensor = Value_at_node_id id; tensor_idcs; dynamic_idcs; target_dims; body } ->
        dynamic_indices env (get id).value ~tensor_idcs ~dynamic_idcs ~target_dims body
    | Dynamic_indices { tensor = Gradient_at_node_id id; tensor_idcs; dynamic_idcs; target_dims; body } ->
        dynamic_indices env (Option.value_exn (get id).grad) ~tensor_idcs ~dynamic_idcs ~target_dims body
    | Comment _ -> ()
  and loop_float env llv =
    let open Float in
    let loop = loop_float env in
    match llv with
    | Constant c -> c
    | Get (Value_at_node_id id, indices) -> get_as_float (get id).value @@ lookup env indices
    | Get (Gradient_at_node_id id, indices) ->
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

type node = {
  id : int;
  value_assign_index_bag : Shape.symbol Hash_set.t;
  value_computations : (Shape.Symbolic_idcs.t, unit low_level list) Hashtbl.t;
      (** The computations are retrieved for optimization just as they are populated, so that
          the inlined code corresponds precisely to the changes to the tensors that would happen up till
          that point. Within the code blocks paired with an index tuple, all assignments and accesses
          must happen via the index tuple; if this is not the case for some assignment, the node cannot
          be virtual. *)
  value_accesses : (Shape.Indices.t, visits) Hashtbl.t;
      (** For dynamic indexes, we take a value of 0. This leads to an overestimate of visits, which is safe. *)
  grad_assign_index_bag : Shape.symbol Hash_set.t;
  grad_computations : (Shape.Symbolic_idcs.t, unit low_level list) Hashtbl.t;
  grad_accesses : (Shape.Indices.t, visits) Hashtbl.t;
  mutable value_fill : float low_level option;
  mutable grad_fill : float low_level option;
  mutable non_virtual : bool;
}
[@@deriving sexp_of]

let global_node_store : (int, node) Hashtbl.t = Hashtbl.create (module Int)
let cleanup_session () = Hashtbl.clear global_node_store

let get uid =
  Hashtbl.find_or_add global_node_store uid ~default:(fun () ->
      {
        id = uid;
        value_assign_index_bag = Hash_set.Poly.create ();
        value_computations = Hashtbl.create (module Shape.Symbolic_idcs);
        value_accesses = Hashtbl.create (module Shape.Indices);
        grad_assign_index_bag = Hash_set.Poly.create ();
        grad_accesses = Hashtbl.create (module Shape.Indices);
        grad_computations = Hashtbl.create (module Shape.Symbolic_idcs);
        value_fill = None;
        grad_fill = None;
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
    | Fill { tensor = Value_at_node_id id; value } ->
        loop_float env value;
        let node = get id in
        Hash_set.add nodes id;
        if not @@ Hashtbl.is_empty node.value_accesses then node.non_virtual <- true
    | Fill { tensor = Gradient_at_node_id id; value } ->
        loop_float env value;
        let node = get id in
        Hash_set.add nodes id;
        if not @@ Hashtbl.is_empty node.grad_accesses then node.non_virtual <- true
    | Set (Value_at_node_id id, idcs, llv) ->
        loop_float env llv;
        Hash_set.add nodes id;
        let node = get id in
        Array.iter idcs ~f:(function
          | Shape.Fixed_idx _ | Shape.Dynamic_provider _ -> ()
          | Shape.Iterator s | Shape.Dynamic_recipient s -> Hash_set.add node.value_assign_index_bag s)
    | Set (Gradient_at_node_id id, idcs, llv) ->
        loop_float env llv;
        Hash_set.add nodes id;
        let node = get id in
        Array.iter idcs ~f:(function
          | Shape.Fixed_idx _ | Shape.Dynamic_provider _ -> ()
          | Shape.Iterator s | Shape.Dynamic_recipient s -> Hash_set.add node.grad_assign_index_bag s)
    | Set_local (_, llv) -> loop_float env llv
    | Comment _ -> ()
    | Dynamic_indices { tensor = Value_at_node_id id; tensor_idcs; dynamic_idcs; target_dims; body } ->
        let node = get id in
        Hash_set.add nodes id;
        dynamic_indices node.value_accesses node.value_fill node.value_assign_index_bag env ~tensor_idcs
          ~dynamic_idcs ~target_dims body
    | Dynamic_indices { tensor = Gradient_at_node_id id; tensor_idcs; dynamic_idcs; target_dims; body } ->
        let node = get id in
        Hash_set.add nodes id;
        dynamic_indices node.grad_accesses node.grad_fill node.grad_assign_index_bag env ~tensor_idcs
          ~dynamic_idcs ~target_dims body
  and loop_float env llv =
    let loop = loop_float env in
    match llv with
    | Constant _ -> ()
    | Get (Value_at_node_id id, indices) ->
        let node = get id in
        Hash_set.add nodes id;
        Hashtbl.update node.value_accesses (lookup env indices)
          ~f:(visit node.value_fill node.value_assign_index_bag)
    | Get (Gradient_at_node_id id, indices) ->
        let node = get id in
        Hash_set.add nodes id;
        Hashtbl.update node.grad_accesses (lookup env indices)
          ~f:(visit node.grad_fill node.grad_assign_index_bag)
    | Local_scope (_, _, llc) -> loop_proc env llc
    | Get_local _ -> ()
    | Binop (Arg1, llv1, _llv2) -> loop llv1
    | Binop (Arg2, _llv1, llv2) -> loop llv2
    | Binop (_, llv1, llv2) ->
        loop llv1;
        loop llv2
    | Unop (_, llv) -> loop llv
  and dynamic_indices accesses fill assign_bag env ~tensor_idcs ~dynamic_idcs ~target_dims:_ body =
    let env =
      Array.foldi dynamic_idcs ~init:env ~f:(fun provider_dim env key ->
          let at_pos = lookup ~provider_dim env tensor_idcs in
          Hashtbl.update accesses at_pos ~f:(visit fill assign_bag);
          Map.add_exn ~key ~data:0 env)
    in
    loop_proc env body
  in
  loop_proc (Map.empty (module Shape.Symbol)) llc;
  Hash_set.iter nodes ~f:(fun node_id ->
      let node = Hashtbl.find_exn global_node_store node_id in
      if
        Hashtbl.exists node.value_accesses ~f:is_too_many
        || (consider_grads && Hashtbl.exists node.grad_accesses ~f:is_too_many)
      then node.non_virtual <- true)

let visit_llprog ?(max_visits = 3) ?(consider_grads = false) = function
  | Assign_suspension proc | Assign_session_step_update proc -> visit_llc ~max_visits ~consider_grads proc

(*
let analyze_llc llc =
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
    | Fill { tensor = Value_at_node_id id; value } ->
        loop_float env value;
        let node = get id in
        if not @@ Hashtbl.is_empty node.value_accesses then node.non_virtual <- true;
        Hashtbl.clear node.value_assignments;
        node.value_fill <- Some value
    | Fill { tensor = Gradient_at_node_id id; value } ->
        loop_float env value;
        let node = get id in
        if not @@ Hashtbl.is_empty node.grad_accesses then node.non_virtual <- true;
        Hashtbl.clear node.grad_assignments;
        node.grad_fill <- Some value
    | Set (Value_at_node_id id, indices, llv) ->
        loop_float env llv;
        Hashtbl.add_multi (get id).value_assignments ~key:indices ~data:llv
    | Set (Gradient_at_node_id id, indices, llv) ->
        loop_float env llv;
        Hashtbl.add_multi (get id).grad_assignments ~key:indices ~data:llv
    | Comment _ -> ()
    | Dynamic_indices { tensor = Value_at_node_id id; tensor_idcs; dynamic_idcs; target_dims; body } ->
        let node = get id in
        dynamic_indices node.value_accesses node.value_assignments env ~tensor_idcs ~dynamic_idcs ~target_dims
          body
    | Dynamic_indices { tensor = Gradient_at_node_id id; tensor_idcs; dynamic_idcs; target_dims; body } ->
        let node = get id in
        dynamic_indices node.grad_accesses node.grad_assignments env ~tensor_idcs ~dynamic_idcs ~target_dims
          body
  and loop_float env llv =
    let loop = loop_float env in
    match llv with
    | Constant _ -> ()
    | Get (Value_at_node_id id, indices) ->
        let node = get id in
        Hashtbl.update node.value_accesses (lookup env indices) ~f:(visit node.value_assignments indices)
    | Get (Gradient_at_node_id id, indices) ->
        let node = get id in
        Hashtbl.update node.grad_accesses (lookup env indices) ~f:(visit node.grad_assignments indices)
    | Binop (Arg1, llv1, _llv2) -> loop llv1
    | Binop (Arg2, _llv1, llv2) -> loop llv2
    | Binop (_, llv1, llv2) ->
        loop llv1;
        loop llv2
    | Unop (_, llv) -> loop llv
  and dynamic_indices accesses assignments env ~tensor_idcs ~dynamic_idcs ~target_dims:_ body =
    let env =
      Array.foldi dynamic_idcs ~init:env ~f:(fun provider_dim env key ->
          let at_pos = lookup ~provider_dim env tensor_idcs in
          Hashtbl.update accesses at_pos ~f:(visit assignments tensor_idcs);
          Map.add_exn ~key ~data:0 env)
    in
    loop_proc env body
  in
  loop_proc (Map.empty (module Shape.Symbol)) llc

let analyze_llprog = function Assign_suspension proc | Assign_session_step_update proc -> analyze_llc proc
*)
