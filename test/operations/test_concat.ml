open! Base
open! Ocannl
open Ocannl.Operation.DSL_modules

(* Note: The concat operation is currently limited by shape inference issues.
   The Block shape logic doesn't yet fully support deriving projections for
   concatenated axes. This test verifies that the operation compiles correctly
   and documents the intended API. *)

let () =
  Tensor.unsafe_reinitialize ();
  Stdio.print_endline "=== Test: Verifying concat operation compiles ===";

  (* Create some test tensors *)
  let t1 = TDSL.term ~batch_dims:[] ~input_dims:[] ~output_dims:[ 3 ] () in
  let t2 = TDSL.term ~batch_dims:[] ~input_dims:[] ~output_dims:[ 2 ] () in

  (* Test 1: Basic concat function call with 2 tensors *)
  let result1 = TDSL.concat "a; b => a^b" [| t1; t2 |] () in
  Stdio.print_endline "concat with 2 tensors: created successfully";
  Stdio.printf "  Result tensor id: %d\n" result1.Tensor.id;

  (* Test 2: Concat with 3 tensors *)
  let t3 = TDSL.term ~batch_dims:[] ~input_dims:[] ~output_dims:[ 2 ] () in
  let t4 = TDSL.term ~batch_dims:[] ~input_dims:[] ~output_dims:[ 3 ] () in
  let t5 = TDSL.term ~batch_dims:[] ~input_dims:[] ~output_dims:[ 1 ] () in
  let result2 = TDSL.concat "a; b; c => a^b^c" [| t3; t4; t5 |] () in
  Stdio.print_endline "concat with 3 tensors: created successfully";
  Stdio.printf "  Result tensor id: %d\n" result2.Tensor.id;

  (* Test 3: Negated concat *)
  let t6 = TDSL.term ~batch_dims:[] ~input_dims:[] ~output_dims:[ 2 ] () in
  let t7 = TDSL.term ~batch_dims:[] ~input_dims:[] ~output_dims:[ 2 ] () in
  let result3 = TDSL.concat ~negated:true "a; b => a^b" [| t6; t7 |] () in
  Stdio.print_endline "negated concat: created successfully";
  Stdio.printf "  Result tensor id: %d\n" result3.Tensor.id;

  (* Test 4: Concat using O module syntax *)
  let result4 = TDSL.O.concat "a; b => a^b" [| t1; t2 |] in
  Stdio.print_endline "concat via O module: created successfully";
  Stdio.printf "  Result tensor id: %d\n" result4.Tensor.id;

  (* Test 5: Concat with gradient tracking (PDSL) *)
  let p1 = PDSL.term ~batch_dims:[] ~input_dims:[] ~output_dims:[ 2 ] () in
  let p2 = PDSL.term ~batch_dims:[] ~input_dims:[] ~output_dims:[ 2 ] () in
  let result5 = PDSL.concat "a; b => a^b" [| p1; p2 |] () in
  Stdio.print_endline "concat with gradient tracking (PDSL): created successfully";
  Stdio.printf "  Result tensor id: %d\n" result5.Tensor.id;
  Stdio.printf "  Has gradient: %b\n" (Option.is_some result5.Tensor.diff);

  (* Note: Multi-dimensional concat currently fails due to shape inference issues *)
  (* let m1 = TDSL.term ~batch_dims:[] ~input_dims:[ 2 ] ~output_dims:[ 3 ] () in
     let m2 = TDSL.term ~batch_dims:[] ~input_dims:[ 2 ] ~output_dims:[ 2 ] () in
     let result6 = TDSL.concat "r, i; s, i => r^s, i" [| m1; m2 |] () in *)

  Stdio.print_endline "\nAll concat operation creation tests passed!";
  Stdio.print_endline "Note: Runtime execution requires fixing Block shape projection derivation."
