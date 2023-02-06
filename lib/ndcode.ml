(** The code for operating on n-dimensional arrays. *)
open Base

type precision =
  | Half
  | Single
  | Double
  (* FIXME(28): implement precision setting and precision-specific code generation. *)
  
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
