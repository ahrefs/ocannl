(** The code for operating on n-dimensional arrays. *)
open Base

(** *** High-level representation. *** *)

type data = {node_id: int; field: [`Value | `Grad]}
[@@deriving sexp, equal]

type routine = {node_id: int; field: [`Forward | `Backprop]}
[@@deriving sexp, equal]

type binop =
  | Skip_arg
  | Add
  | Mul
  | ToPowOf
  | Relu_gate
[@@deriving sexp]

type unop =
  | Identity
  | Relu
[@@deriving sexp]

(** Initializes or resets a tensor by filling in the corresponding numbers, at the appropriate precision. *)
type init_op =
  [ `Unspecified
  (** Uninitialized. On reset, values may remain unchanged, but are not guaranteed to. *)
  | `Constant_of_value of float
  (** Puts the value in all cells. *)
  | `Fixed_constant of float array
  (** Fills in the numbers where the rightmost axis is contiguous. *)
  | `Range_over_axis_from_end of int
  (** Fills in the index number of the specified axis counting from end.
      [`Range_over_axis_from_end 1] is the range over the last axis. *)
  | `Range_over_offsets
  (** Fills in the offset number of each cell (i.e. how many cells away it is from the beginning). *)
  | `Standard_uniform
  (** Draws the values from U(0,1). *)
  | `Standard_gaussian
  (** Draws the values from N(0,1). *)
  ]
[@@deriving sexp]

type t =
  | Par of t * t
  (** These tasks can proceed in parallel, there is no interaction. *)
  | ParHint of t * t
  (** Computing [ParHint (c1, c2)] can proceed in parallel on [c1] and [c2], but when [c2] reads values
      that [c1] writes, the writes in [c1] must occur before the reads in [c2]. If a backend does not
      support detection of when [ParHint (c1, c2)] is safe to parallelize, it should provide an option
      [force_unsafe_parhint] which always parallelizes. *)
  | Seq of t * t
  (** These tasks can only benefit from mutual parallelism via operator fusion / loop fusion. *)
  | Accum_binop of {
      zero_out: bool;
      accum: binop; op: binop;
      lhs: data; rhs1: data; rhs2: data;
      projections: unit -> Shape.projections }
  | Accum_unop of {
      zero_out: bool;
      accum: binop; op: unop;
      lhs: data; rhs: data;
      projections: unit -> Shape.projections }
  | Create of { tensor: data; dims: unit -> int array; init_op: init_op }
  | Reset of { tensor: data; reset_op: init_op }
  | Noop
[@@deriving sexp]

(** Dynamically loading a program executes the [Initialization] code, or bounds the [procedure]
    to [routine] for a node, or bounds a callback to the global routine slot. *)
type program =
  | Node_specific of {procedure: t; routine: routine; label: string}
  | Initialization of t
  | Session_prepare_step of t
[@@deriving sexp]

let remove_updates data c =
  let rec rm check = function
    | ( Par ((Accum_binop {lhs; _} | Accum_unop {lhs; _}), t)
      | ParHint ((Accum_binop {lhs; _} | Accum_unop {lhs; _}), t)
      | Seq ((Accum_binop {lhs; _} | Accum_unop {lhs; _}), t)
      | Par (t, (Accum_binop {lhs; _} | Accum_unop {lhs; _}))
      | ParHint (t, (Accum_binop {lhs; _} | Accum_unop {lhs; _}))
      | Seq (t, (Accum_binop {lhs; _} | Accum_unop {lhs; _}))) as c when check ->
        if equal_data data lhs then rm true t else rm false c
    | Par (t1, t2) -> Par (rm true t1, rm true t2)
    | ParHint (t1, t2) -> ParHint (rm true t1, rm true t2)
    | Seq (t1, t2) -> Seq (rm true t1, rm true t2)
    | (Accum_binop {lhs; _} | Accum_unop {lhs; _}) when equal_data data lhs -> Noop
    | c -> c in
  rm true c



(** *** Low-level representation. *)

(** Cases: [unit low_level] -- code, [float low_level] -- single number at some precision,
    [data low_level] -- a tensor. *)
type _ low_level =
  | Lines: unit low_level array -> unit low_level
  | For_loop: {index: Shape.symbol; from_: int; to_: int; body: unit low_level} -> unit low_level
  | Value_at_node_id: int -> data low_level
  | Gradient_at_node_id: int -> data low_level
  | LLCreate: {
      tensor: data low_level; dims: int array; init_op: init_op;
    } -> unit low_level
  | LLReset: {
      tensor: data low_level; reset_op: init_op;
    } -> unit low_level
  | Unoptimized_set: data low_level * Shape.symbolic_axis array * float low_level -> unit low_level
  | Unoptimized_get: data low_level * Shape.symbolic_axis array -> float low_level
  | Unoptimized_binop: binop * float low_level * float low_level -> float low_level
  | Unoptimized_unop: unop * float low_level -> float low_level
  | Assign_routine: routine * unit low_level -> unit low_level
  | Assign_session_prepare_step: unit low_level -> unit low_level
  | Comment: string -> unit low_level
(* [@@deriving sexp] *)

let data_pointer (xhs: data) =
  match xhs.field with
  | `Value -> Value_at_node_id xhs.node_id | `Grad -> Gradient_at_node_id xhs.node_id

let all_parallel = List.fold ~init:Noop ~f:(fun sts st -> Par (st, sts))

let rec unoptimized (code: t): unit low_level =
  match code with
  | Accum_binop {zero_out; accum; op; lhs; rhs1; rhs2; projections} ->
    let projections = projections() in
    let lhs_idx = Shape.(derive_index projections.product_iterators projections.project_lhs) in
    let rhs1_idx = Shape.(derive_index projections.product_iterators projections.project_rhs1) in
    let rhs2_idx = match projections.project_rhs2 with
      | None -> invalid_arg "accum_binop: projections missing project_rhs2"
      | Some rhs2 -> Shape.(derive_index projections.product_iterators rhs2) in
    let lhs_ptr = data_pointer lhs in
    let lhs iters = Unoptimized_get (lhs_ptr, lhs_idx iters) in
    let rhs1 iters = Unoptimized_get (data_pointer rhs1, rhs1_idx iters) in
    let rhs2 iters = Unoptimized_get (data_pointer rhs2, rhs2_idx iters) in
    let basecase rev_iters =
      let iters = Array.of_list_rev rev_iters in
      Unoptimized_set (lhs_ptr, lhs_idx iters,
                       Unoptimized_binop (accum, lhs iters, Unoptimized_binop (op, rhs1 iters, rhs2 iters))) in
    let rec loop rev_iters = function
      | ([], []) -> basecase rev_iters
      | (dim::product, it::iters) ->
        For_loop {index=it; from_=0; to_=dim - 1; body=loop (it::rev_iters) (product, iters)}
      | _ -> invalid_arg "Code.unoptimized: Accum_binop projections dims-iterators mismatch" in
    let for_loops = 
      loop [] (Array.to_list projections.product_space, Array.to_list projections.product_iterators) in
    if zero_out
    then Lines [|LLReset {tensor=lhs_ptr; reset_op=`Constant_of_value 0.0}; for_loops|]
    else for_loops

  | Accum_unop {zero_out; accum; op; lhs; rhs; projections} ->
    let projections = projections() in
    let lhs_idx = Shape.(derive_index projections.product_iterators projections.project_lhs) in
    let rhs_idx = Shape.(derive_index projections.product_iterators projections.project_rhs1) in
    let lhs_ptr = data_pointer lhs in
    let lhs iters = Unoptimized_get (lhs_ptr, lhs_idx iters) in
    let rhs iters = Unoptimized_get (data_pointer rhs, rhs_idx iters) in
    let basecase rev_iters =
      let iters = Array.of_list_rev rev_iters in
      Unoptimized_set (lhs_ptr, lhs_idx iters,
                       Unoptimized_binop (accum, lhs iters, Unoptimized_unop (op, rhs iters))) in
    let rec loop rev_iters = function
      | ([], []) -> basecase rev_iters
      | (dim::product, it::iters) ->
        For_loop {index=it; from_=0; to_=dim - 1; body=loop (it::rev_iters) (product, iters)}
      | _ -> invalid_arg "Code.unoptimized: Accum_unop projections dims-iterators mismatch" in
    let for_loops = 
      loop [] (Array.to_list projections.product_space, Array.to_list projections.product_iterators) in
    if zero_out
    then Lines [|LLReset {tensor=lhs_ptr; reset_op=`Constant_of_value 0.0}; for_loops|]
    else for_loops

  | Noop -> Lines [||]

  | Par (c1, c2) | ParHint (c1, c2) | Seq (c1, c2) ->
    let ll1 = unoptimized c1 in
    let ll2 = unoptimized c2 in
    (match ll1, ll2 with
     | Lines ls1, Lines ls2 -> Lines (Array.append ls1 ls2)
     | _, Lines ls2 -> Lines (Array.append [|ll1|] ls2)
     | Lines ls1, _ -> Lines (Array.append ls1 [|ll2|])
     | _ -> Lines [|ll1; ll2|])

  | Create {tensor; dims; init_op} ->
    LLCreate {tensor=data_pointer tensor; dims=dims(); init_op}
  | Reset {tensor; reset_op} ->
    LLReset {tensor=data_pointer tensor; reset_op}

let unoptimized_program prog: unit low_level =
  match prog with
  | Initialization proc -> unoptimized proc
  | Node_specific {procedure; routine; label} ->
    Lines [|Comment label; Assign_routine (routine, unoptimized procedure)|]
  | Session_prepare_step proc -> Assign_session_prepare_step (unoptimized proc)

module DSL = struct
  let value_of_node n: data = {node_id=n.Ocannl_runtime.Node.id; field=`Value}
  let grad_of_node n: data = {node_id=n.Ocannl_runtime.Node.id; field=`Grad}
  let value_of_id node_id: data = {node_id; field=`Value}
  let grad_of_id node_id: data = {node_id; field=`Grad}
  
end

let interpret_llc ?(with_debug=true) llc =
  let lookup env indices =
    Array.map indices ~f:Shape.(function
    | Fixed_idx i -> i
    | Iterator s -> Map.find_exn env s) in
  let open Ocannl_runtime.Node in
  let rec loop_proc env llc: unit =
    let loop = loop_proc env in
    match llc with
    | Lines body -> Array.iter ~f:loop body
    | For_loop {index=key; from_; to_; body} ->
      for data = from_ to to_ do
        loop_proc (Map.add_exn ~key ~data env) body
      done
    | LLCreate { tensor=Value_at_node_id id; dims; init_op } ->
      (get id).value <- create_ndarray Single dims init_op
    | LLCreate { tensor=Gradient_at_node_id id; dims; init_op } ->
      (get_form id).grad <- create_ndarray Single dims init_op
    | LLReset { tensor=Value_at_node_id id; reset_op } ->
      reset_ndarray reset_op ((get id).value)
    | LLReset { tensor=Gradient_at_node_id id; reset_op } ->
      reset_ndarray reset_op ((get_form id).grad)
    | Unoptimized_set (Value_at_node_id id, indices, llv) ->
      set_from_float (get id).value (lookup env indices) @@ loop_float env llv
    | Unoptimized_set (Gradient_at_node_id id, indices, llv) ->
      set_from_float (get_form id).grad (lookup env indices) @@ loop_float env llv
    | Assign_routine ({node_id; field=`Forward}, proc) ->
      (get_form node_id).forward := Some (fun () -> loop proc)
    | Assign_routine ({node_id; field=`Backprop}, proc) ->
      (get_form node_id).backprop := Some (fun () -> loop proc)
    | Assign_session_prepare_step (proc) ->
      global.session_prepare_step := Some (fun () -> loop proc)
    | Comment message when with_debug -> Stdio.printf "%s\n%!" message
    | Comment _ -> ()
    and loop_float env llv =
    let open Float in
    let loop = loop_float env in
    match llv with
    | Unoptimized_get (Value_at_node_id id, indices) ->
      get_as_float (get id).value @@ lookup env indices
    | Unoptimized_get (Gradient_at_node_id id, indices) ->
      get_as_float (get_form id).grad @@ lookup env indices
    | Unoptimized_binop (Skip_arg, _llv1, llv2) -> loop llv2
    | Unoptimized_binop (Add, llv1, llv2) -> loop llv1 + loop llv2
    | Unoptimized_binop (Mul, llv1, llv2) -> loop llv1 * loop llv2
    | Unoptimized_binop (ToPowOf, llv1, llv2) ->
      let v1 = loop llv1 in
      let v2 = loop llv2 in
      Float.(if is_integer v2 then int_pow v1 @@ to_int v2 else v1 ** v2)
    | Unoptimized_binop (Relu_gate, llv1, llv2) -> if loop llv1 > 0.0 then loop llv2 else 0.0
    | Unoptimized_unop (Identity, llv) -> loop llv
    | Unoptimized_unop (Relu, llv) -> let v = loop llv in if v > 0.0 then v else 0.0 in
  loop_proc (Map.empty (module Shape.Symbol)) llc

let fprint_code ppf c =
  (* TODO: something nicely concise. *)
  Caml.Format.fprintf ppf "%s" @@ Sexp.to_string_hum @@ sexp_of_t c  

let fprint_program ppf prog =
  (* TODO: something nicely concise. *)
  Caml.Format.fprintf ppf "%s" @@ Sexp.to_string_hum @@ sexp_of_program prog  

let interpret_program ?(with_debug=true) prog: string option =
  let llc = unoptimized_program prog in
  let () = interpret_llc ~with_debug llc in
  (* If we were interpreting bytecode, we would return the bytecode for debugging purposes. *)
  (* Some (Sexp.to_string_hum @@ sexp_of_low_level llc) *)
  Some (Caml.Format.asprintf "%a" fprint_program prog)

let interpreter_error_message prefix ?extra_error_msg ~contents exc =
  let backtrace = Caml.Printexc.get_backtrace() in
  let exc_str = Caml.Printexc.to_string exc in
  let message =
    Buffer.create (String.length contents + String.length backtrace + String.length exc_str) in
  let msg = Buffer.add_string message in
  msg prefix; msg exc_str; msg "\n"; msg backtrace;
  (match extra_error_msg with None -> () | Some extra ->
    msg "\nIn the context of:\n"; msg extra);
  msg contents;
  Buffer.contents message
