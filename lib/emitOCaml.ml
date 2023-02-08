(** Compiles [Code.t] into OCaml. *)
open Base

open Code

(** Cases: [unit low_level] -- code, [float low_level] -- single number at some precision,
    [bool low_level] -- single boolean, [data low_level] -- a tensor. *)
type _ low_level =
  | Lines: unit low_level array -> unit low_level
  | For_loop: Shape.symbol * int * int * unit low_level -> unit low_level
  | While_loop: bool low_level * unit low_level -> unit low_level
  | Float_const: precision * float -> float low_level
  | Value_at_node_id: int -> data low_level
  | Gradient_at_node_id: int -> data low_level
  | Initialize: {
      tensor: data low_level; precision: precision; dims: int array;
      init_values: float array;
      (** [init_values] can be empty -- no initialization, single number -- initialize the whole tensor,
          the length of the tensor -- initialize from numbers where the rightmost axis is contiguous. *)
    } -> unit low_level
  | Unoptimized_set: Shape.symbol array * data low_level * float low_level -> unit low_level
  | Unoptimized_get: Shape.symbol array * data low_level -> float low_level
  | All_greater_0: data low_level -> bool low_level
  | Exists_greater_0: data low_level -> bool low_level

    (* TODO(41): [@@deriving fold_sig] *)
  
let unoptimized (code: t): unit low_level =
  match code with
  | Accum_binop {
      zero_out;
      accum; op;
      lhs; rhs1; rhs2;
      projections;
      precision;
    } -> ignore (zero_out,
                 accum, op,
                 lhs, rhs1, rhs2,
                 projections, precision); failwith "NOT IMPLEMENTED YET"
  | Accum_unop {
      zero_out;
      accum; op;
      lhs; rhs;
      projections;
      precision;
    } -> ignore (
      zero_out,
      accum, op,
      lhs, rhs,
      projections, precision
    ); failwith "NOT IMPLEMENTED YET"
  | (Noop|Par (_, _)|ParHint (_, _)|Seq (_, _)|Create _|Reset _) -> failwith "NOT IMPLEMENTED YET"

(* TODO(41): this could be automatically derived. *)
module type FOLD_CODE = sig
  type 'a result
  type 'a low_level_result
  val accum_binop:
      zero_out:bool ->
      accum:binop -> op:binop ->
      lhs:data result option -> rhs1:data result option -> rhs2:data result option ->
      projections:Shape.projections -> data result
  val accum_unop:
      zero_out:bool ->
      accum:binop -> op:binop ->
      lhs:data result option -> rhs:data result option ->
      projections:Shape.projections -> data result
  val embed_low_level: 'a low_level_result -> 'a result

  val seq: unit low_level_result -> 'a low_level_result -> 'a low_level_result
  val for_loop:
    Shape.symbol -> int low_level_result -> int low_level_result -> unit low_level_result -> unit low_level_result
  val while_loop: bool low_level_result -> unit low_level_result -> unit low_level_result
  val int_const: int -> int low_level_result
  val float_const: precision -> float -> float low_level_result
  val tensor_at_node_id: int -> data low_level_result
  val initialize_tensor:
      node_id:int -> precision:precision -> dims:int array ->
      init_values:float array ->
      (** [init_values] can be empty -- no initialization, single number -- initialize the whole tensor,
          the length of the tensor -- initialize from numbers where the rightmost axis is contiguous. *)
      unit low_level_result

  val unoptimized_set:
    Shape.symbol array -> data low_level_result -> float low_level_result -> unit low_level_result
  val unoptimized_get: Shape.symbol array -> data low_level_result -> float low_level_result
  val all_greater_0: data low_level_result -> bool low_level_result
  val exists_greater_0: data low_level_result -> bool low_level_result
end

let emit = unoptimized

let format_code fmt c: unit = ignore (fmt, c); failwith "NOT IMPLEMENTED YET"

(*
 let zero = [%c 0.0 ]

 let one = [%c 1.0 ]
 
(** Accumulates the results of the operation: [lhs = accum lhs (op rhs1 rhs2)]. *)
let accum_binop ?(zero_out=false) ~accum ~op ?lhs ?rhs1 ?rhs2 projections =
  let lhs_idx = Shape.(derive_index projections.product_iterators projections.project_lhs) in
  let rhs1_idx = Shape.(derive_index projections.product_iterators projections.project_rhs1) in
  let rhs2_idx = match projections.project_rhs2 with
    | None -> invalid_arg "accum_binop: projections missing project_rhs2"
    | Some rhs2 -> Shape.(derive_index projections.product_iterators rhs2) in
  let rhs1 iters =
    match rhs1 with
    | None -> zero
    | Some rhs1 ->
      let rhs1_idx = Lifts.lift_array @@ rhs1_idx iters in
      [%c Bigarray.Genarray.get [%e rhs1] [%e rhs1_idx]] in
  let rhs2 iters =
    match rhs2 with
    | None -> zero
    | Some rhs2 -> 
      let rhs2_idx = Lifts.lift_array @@ rhs2_idx iters in
      [%c Bigarray.Genarray.get [%e rhs2] [%e rhs2_idx]] in
  match lhs with
  | None -> [%c ()]
  | Some lhs ->
    let basecase rev_iters =
      let iters = Array.of_list_rev rev_iters in
      let lhs_idx = Lifts.lift_array @@ lhs_idx iters in
      [%c Bigarray.Genarray.set [%e lhs] [%e lhs_idx]
           [%e accum [%c Bigarray.Genarray.get [%e lhs] [%e lhs_idx]] @@ op (rhs1 iters) (rhs2 iters) ] ] in
    let rec loop rev_iters = function
      | [] -> basecase rev_iters
      | dim::product ->
        [%c for i = 0 to [%e Lifts.Lift_int.lift dim] - 1 do
          [%e loop ([%c i] ::rev_iters) product]
        done ] in
    if zero_out then
      [%c Bigarray.Genarray.fill [%e lhs] [%e zero]; [%e loop [] @@ Array.to_list projections.product_space] ]
    else
      loop [] @@ Array.to_list projections.product_space

(** Accumulates the results of the operation: [lhs = accum lhs (op rhs)]. *)
let accum_unop ?(zero_out=false) ~accum ~op ?lhs ?rhs projections =
  let lhs_idx = Shape.(derive_index projections.product_iterators projections.project_lhs) in
  let rhs1_idx = Shape.(derive_index projections.product_iterators projections.project_rhs1) in
  let rhs iters =
    match rhs with
    | None -> zero
    | Some rhs ->
      let rhs1_idx = Lifts.lift_array @@ rhs1_idx iters in
      [%c Bigarray.Genarray.get [%e rhs] [%e rhs1_idx]] in
  match lhs with
  | None -> [%c ()]
  | Some lhs ->
    let basecase rev_iters =
      let iters = Array.of_list_rev rev_iters in
      let lhs_idx = Lifts.lift_array @@ lhs_idx iters in
      [%c Bigarray.Genarray.set [%e lhs] [%e lhs_idx]
          [%e accum [%c Bigarray.Genarray.get [%e lhs] [%e lhs_idx]] @@ op (rhs iters) ] ] in
    let rec loop rev_iters = function
      | [] -> basecase rev_iters
      | dim::product ->
        [%c for i = 0 to [%e Lifts.Lift_int.lift dim] - 1 do
          [%e loop ([%c i] ::rev_iters) product]
        done ] in
    if zero_out then
      [%c Bigarray.Genarray.fill [%e lhs] [%e zero]; [%e loop [] @@ Array.to_list projections.product_space] ]
    else
      loop [] @@ Array.to_list projections.product_space

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