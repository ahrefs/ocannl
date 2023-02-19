(** The code for operating on n-dimensional arrays. *)
open Base

(** *** High-level representation. *** *)

type data = {node_id: int; field: [`Value | `Grad]}
[@@deriving sexp]

type routine = {node_id: int; field: [`Forward | `Backprop]}
[@@deriving sexp]

type binop =
  | Skip_arg
  | Add
  | Mul
  | Relu_gate

type unop =
  | Identity
  | Relu

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

type t =
  | Par of t * t
  (** These tasks can proceed in parallel, there is no interaction. *)
  | ParHint of t * t
  (** Computing [ParHint (c1, c2)] can proceed in parallel on [c1] and [c2], but when [c2] reads values
      that [c1] writes, the writes in [c1] must occur before the reads in [c2]. *)
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

(** Dynamically loading a program executes [initialization] and bounds the [procedure] to [routine]. *)
type program = {initialization: t; procedure: t; routine: routine; label: string}

let sprint_code (c: t): string = ignore c; failwith "NOT IMPLEMENTED YET [1]"
let sprint_program (c: program): string = ignore c; failwith "NOT IMPLEMENTED YET [2]"

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
  | Unoptimized_set: data low_level * Shape.symbol array * float low_level -> unit low_level
  | Unoptimized_get: data low_level * Shape.symbol array -> float low_level
  | Unoptimized_binop: binop * float low_level * float low_level -> float low_level
  | Unoptimized_unop: unop * float low_level -> float low_level
  | Assign_routine: routine * unit low_level -> unit low_level
  | Comment: string -> unit low_level

let data_pointer (xhs: data) =
  match xhs.field with
  | `Value -> Value_at_node_id xhs.node_id | `Grad -> Gradient_at_node_id xhs.node_id

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

let unoptimized_program (prog: program): unit low_level =
  let init = unoptimized prog.initialization in
  let proc = Assign_routine (prog.routine, unoptimized prog.procedure) in
  let comment = Comment prog.label in
  match init with
  | Lines init_lines ->  Lines (Array.concat [[|comment|]; init_lines; [|proc|]])
  | _ -> Lines [|comment; init; proc|]

(*
let skip_arg (_n1: float Codelib.code) (n2: float Codelib.code) = n2
let num_id (n: float Codelib.code) = n

let identity (n: float Codelib.code) = n

let add n1 n2 = [%c Float.([%e n1] + [%e n2]) ]

let mul n1 n2 = [%c Float.([%e n1] * [%e n2]) ]

let relu n = [%c Float.(if [%e n] > 0.0 then [%e n] else 0.0) ]

let relu_gate n1 n2 = [%c Float.(if [%e n1] > 0.0 then [%e n2] else 0.0) ]

let value (v: float) = Lifts.Lift_float.lift v

let uniform ~low ~high = [%c Random.float_range low high ]
*)