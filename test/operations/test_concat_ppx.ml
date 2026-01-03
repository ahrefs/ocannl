open Base
open Ocannl.Operation.DSL_modules

(* Concatenation runtime smoke tests using %op syntax. *)

let%op concat2 a b = (a, b) ++^ "a; b => a^b"
let%op concat3 a b c = (a, b, c) ++^ "a; b; c => a^b^c"
let%op concat_capture a b = (a, b) ++^ "i; j => i^j" [ "i"; "j" ]
let%op sum_all x = x ++ "...|... => 0"

let () =
  Tensor.unsafe_reinitialize ();
  Stdio.print_endline "=== Test: concat via %op syntax ===";

  let t1 = TDSL.term ~batch_dims:[] ~input_dims:[] ~output_dims:[ 3 ] () in
  let t2 = TDSL.term ~batch_dims:[] ~input_dims:[] ~output_dims:[ 2 ] () in
  let t3 = TDSL.term ~batch_dims:[] ~input_dims:[] ~output_dims:[ 1 ] () in

  let r1 = concat2 t1 t2 in
  Stdio.printf "concat2 id: %d\n" r1.Tensor.id;

  let r2 = concat3 t1 t2 t3 in
  Stdio.printf "concat3 id: %d\n" r2.Tensor.id;

  let r3 = concat_capture t1 t2 in
  Stdio.printf "concat_capture id: %d\n" r3.Tensor.id;

  let p1 =
    PDSL.ndarray [| 1.0; 2.0; 3.0 |] ~batch_dims:[] ~input_dims:[] ~output_dims:[ 3 ] ()
  in
  let p2 =
    PDSL.ndarray [| 4.0; 5.0 |] ~batch_dims:[] ~input_dims:[] ~output_dims:[ 2 ] ()
  in
  let loss = sum_all (concat2 p1 p2) in
  let ctx = Ocannl.Context.auto () in
  Ocannl.Train.run ctx (Ocannl.Train.grad_update loss);
  Stdio.printf "concat gradients present: %b\n"
    (Option.is_some p1.Tensor.diff && Option.is_some p2.Tensor.diff);

  Stdio.print_endline "concat %op smoke tests done"
