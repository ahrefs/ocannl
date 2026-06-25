(* gh-ocannl-133 Stage A: end-to-end numeric correctness of inlining repeated-symbol (diagonal /
   partially-diagonal) producers.

   Each case builds a diagonal/partially-diagonal producer with an einsum and consumes it WITHOUT
   materializing the producer, so the producer virtualizes and inlines (with an equality guard) into
   the consumer. The printed values must match materialized execution: off-diagonal cells stay at
   the zero/init value and the diagonal cells carry the producer value. The optimized-IR structure
   (guard present for distinct symbols, folded for equal indices) is pinned separately by
   test/operations/virtual_diagonal. *)

open Base
open Ocannl
module IDX = Train.IDX
module CDSL = Train.CDSL
open Nn_blocks.DSL_modules

(* Diagonal producer read by a generic consumer: diagonal i=>ii, then (diagonal + 1). Expected 5x5:
   ones everywhere except the diagonal, which is [1;2;3;4;5]. *)
let test_diagonal_generic () =
  Stdio.printf "\nDiagonal i=>ii inlined into a generic consumer (diagonal + 1):\n";
  Tensor.unsafe_reinitialize ();
  let ctx = Context.auto () in
  let input = TDSL.range 5 in
  let%op diagonal = input ++ "i=>ii" in
  let%op consumer = diagonal + 1 in
  let ctx = Train.forward_once ctx consumer in
  Train.printf ~here:[%here] ~with_code:false ~with_grad:false ctx consumer;
  Stdio.printf "\n"

(* Reference: the same diagonal, but materialized, to confirm the inlined values above match the
   materialized tensor (its diagonal is [0;1;2;3;4], zeros off-diagonal; consumer adds 1). *)
let test_diagonal_materialized_reference () =
  Stdio.printf "\nReference: materialized diagonal i=>ii (consumer reads diagonal+1):\n";
  Tensor.unsafe_reinitialize ();
  let ctx = Context.auto () in
  let input = TDSL.range 5 in
  let%op diagonal = input ++ "i=>ii" in
  let%op consumer = diagonal + 1 in
  Train.set_materialized diagonal.value;
  let ctx = Train.forward_once ctx consumer in
  Train.printf ~here:[%here] ~with_code:false ~with_grad:false ctx consumer;
  Stdio.printf "\n"

(* Partially-diagonal producer ij=>iji read generically, then (+ 1). Off the i==i diagonal the value
   is the init (so +1 = 1); on it the value is the producer value + 1. *)
let test_partial_diagonal_generic () =
  Stdio.printf "\nPartially diagonal ij=>iji inlined into a generic consumer (+ 1):\n";
  Tensor.unsafe_reinitialize ();
  let ctx = Context.auto () in
  let input = TDSL.range_of_shape ~output_dims:[ 3; 4 ] () in
  let%op partial = input ++ "ij=>iji" in
  let%op consumer = partial + 1 in
  let ctx = Train.forward_once ctx consumer in
  Train.printf ~here:[%here] ~with_code:false ~with_grad:false ctx consumer;
  Stdio.printf "\n"

let () =
  test_diagonal_generic ();
  test_diagonal_materialized_reference ();
  test_partial_diagonal_generic ();
  Stdio.printf "All virtual-diagonal numeric tests completed.\n"
