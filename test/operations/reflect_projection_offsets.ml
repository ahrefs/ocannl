(* gh-ocannl-773: [Indexing.reflect_projection] turns an access's per-axis indices into the single
   affine index of the flat, row-major offset that access reads. It carried a [Concat] arm that gave
   every segment symbol of a concatenated axis the SAME stride, under a [FIXME] saying concatenation
   was not handled -- which it was not: a concatenation's segments partition the axis into disjoint
   sub-ranges, so its contribution is not one stride times one symbol and has no single-term affine
   form. The arm is now an explicit refusal.

   A refusal is only half of what the change needs pinned. The other half is that the arms that DO
   run still compute the offset formula the function is named for, so each leg below evaluates the
   returned affine index at concrete symbol values and compares it against row-major arithmetic
   computed independently -- over every point of the index space, not one sample. The last leg is
   executed rather than symbolic: [reflect_projection]'s sole caller is the [Range_over_offsets]
   fetch, so a materialized offsets tensor is the function's answer at every cell, run through the
   backend. *)

open Base
open Ocannl.Operation.DSL_modules
module Idx = Ir.Indexing
open Verdict.Claims

(* The value of an affine axis index under an assignment of the symbols. Deliberately partial: a
   [Sub_axis] or [Concat] never appears in what [reflect_projection] RETURNS (it always returns one
   [Affine]), so reaching those arms would be a bug in this test. *)
let eval (env : Idx.symbol -> int) (idx : Idx.axis_index) : int =
  match idx with
  | Idx.Fixed_idx i -> i
  | Idx.Iterator s -> env s
  | Idx.Affine { symbols; offset } ->
      List.fold symbols ~init:offset ~f:(fun acc (c, s) -> acc + (c * env s))
  | Idx.Sub_axis | Idx.Concat _ -> failwith "reflect_projection returned a non-affine index"

let env_of alist s =
  match List.Assoc.find alist s ~equal:Idx.equal_symbol with
  | Some v -> v
  | None -> failwith "unbound symbol in the reflected offset"

(* Every point of a rectangular index space, in row-major order. *)
let points dims =
  Array.fold_right dims ~init:[ [] ] ~f:(fun d acc ->
      List.concat_map (List.init d ~f:Fn.id) ~f:(fun i -> List.map acc ~f:(fun rest -> i :: rest)))

let () =
  (* --- All-iterator projection: the offset must be plain row-major arithmetic. The dims are
     pairwise distinct and none is 1, so a dropped axis, a swapped stride or a stride computed from
     the wrong neighbour all change some point's answer. --- *)
  let dims = [| 2; 3; 4 |] in
  let i = Idx.get_symbol () and j = Idx.get_symbol () and k = Idx.get_symbol () in
  let reflected =
    Idx.reflect_projection ~dims ~projection:[| Idx.Iterator i; Idx.Iterator j; Idx.Iterator k |]
  in
  p_all "the all-iterator offset is row-major at every point" (points dims) ~f:(function
    | [ a; b; c ] -> eval (env_of [ (i, a); (j, b); (k, c) ]) reflected = (((a * 3) + b) * 4) + c
    | _ -> false);

  (* --- Mixed projection: a [Fixed_idx] contributes its own stride-scaled constant, an [Affine]
     scales both its coefficient and its offset by the axis stride, and a [Sub_axis] (an axis folded
     into its neighbour) contributes nothing. Each of the four constructors reaches a different fold
     arm here, and each carries a value no other arm would produce. --- *)
  let dims = [| 2; 3; 4; 5 |] in
  let a_sym = Idx.get_symbol () and b_sym = Idx.get_symbol () in
  let reflected =
    Idx.reflect_projection ~dims
      ~projection:
        [|
          Idx.Iterator a_sym;
          Idx.Fixed_idx 2;
          Idx.Sub_axis;
          Idx.affine ~symbols:[ (2, b_sym) ] ~offset:1;
        |]
  in
  p_all "the mixed-constructor offset agrees with row-major arithmetic"
    (points [| 2; 2 |])
    ~f:(function
      | [ a; b ] ->
          (* Strides for [| 2; 3; 4; 5 |] are 60, 20, 5, 1. [Sub_axis] contributes 0. *)
          eval (env_of [ (a_sym, a); (b_sym, b) ]) reflected
          = (a * 60) + (2 * 20) + 0 + ((2 * b) + 1)
      | _ -> false);

  (* --- The refusal. A concatenated axis reaches no caller (its segments are iterated one loop
     each, and [Concat] indices are eliminated during lowering), so this is the impossibility stated
     rather than approximated. --- *)
  let refused =
    try
      ignore
        (Idx.reflect_projection ~dims:[| 2; 4 |]
           ~projection:[| Idx.Iterator i; Idx.Concat [ j; k ] |]
          : Idx.axis_index);
      false
    with Utils.User_error msg -> String.is_substring msg ~substring:"concatenated axis"
  in
  p "a concatenated axis is refused rather than mis-strided" refused;

  (* --- Executed: the sole caller. A [Range_over_offsets] tensor fills every cell with
     [reflect_projection] of that cell's indices, so a materialized one must read 0, 1, 2, ... in
     row-major order across batch, output and input axes alike. --- *)
  let ctx = Context.auto () in
  let offsets =
    TDSL.range_of_shape ~label:[ "off" ] ~batch_dims:[ 2 ] ~output_dims:[ 3 ] ~input_dims:[ 4 ] ()
  in
  let ctx = Ocannl.Train.forward_once ctx offsets in
  let values = Context.get_values ctx offsets.Tensor.value in
  p "the offsets tensor is fully materialized" (Array.length values = 24);
  p_all "every offsets cell holds its own flat position"
    (List.init (Array.length values) ~f:Fn.id)
    ~f:(fun n -> Float.equal values.(n) (Float.of_int n))
