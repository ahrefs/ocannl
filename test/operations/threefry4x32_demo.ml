(* A simple demo of using the Threefry4x32 PRNG in OCANNL *)

open Base
open Ocannl.Nn_blocks.DSL_modules
module O = TDSL.O

let () =
  Tensor.unsafe_reinitialize ();
  let ctx = Context.auto () in
  (* This is a stress-test for shape inference, usually we would use %op or %cd syntax *)
  let key =
    Ocannl.Tensor.term ~label:[ "random_seed" ] ~grad_spec:Prohibit_grad
      ~fetch_op:(Ir.Assignments.Constant_bits 42L) ()
  in
  Ir.Tnode.update_prec key.value Ir.Ops.uint4x32;
  let range = TDSL.range 5 in
  let%op random_bits = threefry4x32 key range in

  (* Convert to uniform floats in [0, 1) *)
  let%op uniform_floats = uint4x32_to_prec_uniform random_bits in
  Ir.Tnode.update_prec uniform_floats.value Ir.Ops.half;

  (* Compile and run *)
  Ocannl.Train.set_materialized uniform_floats.value;
  let ctx = Ocannl.Train.forward_once ctx uniform_floats in
  let result = Context.get_values ctx uniform_floats.value in

  let print_check name passed =
    Stdio.printf "  %s: %s\n" name (if passed then "PASS" else "FAIL")
  in
  let expected =
    [|
      0.443;
      0.464;
      0.673;
      0.732;
      0.333;
      0.561;
      0.852;
      0.507;
      0.378;
      0.502;
      0.656;
      0.966;
      0.356;
      0.803;
      0.334;
      0.995;
      0.812;
      0.519;
      0.234;
      0.165;
      0.852;
      0.101;
      0.181;
      0.312;
      0.962;
      0.418;
      0.0297;
      0.448;
      0.907;
      0.344;
      0.820;
      0.469;
      0.226;
      0.178;
      0.740;
      0.840;
      0.373;
      0.546;
      0.858;
      0.547;
    |]
  in
  let mean = Array.fold result ~init:0.0 ~f:( +. ) /. Float.of_int (Array.length result) in
  let min_val = Array.min_elt result ~compare:Float.compare |> Option.value ~default:0.0 in
  let max_val = Array.max_elt result ~compare:Float.compare |> Option.value ~default:0.0 in
  let max_abs_error =
    if Int.equal (Array.length result) (Array.length expected) then
      Array.map2_exn result expected ~f:(fun actual expected -> Float.abs (actual -. expected))
      |> Array.max_elt ~compare:Float.compare
      |> Option.value ~default:0.0
    else Float.infinity
  in

  Stdio.printf "Generated %d uniform random numbers:\n" (Array.length result);
  Stdio.printf "  Stats: mean=%.2f min=%.2f max=%.2f max_abs_error=%.3f\n" mean min_val max_val
    max_abs_error;
  print_check "Count is 40" (Int.equal (Array.length result) 40);
  print_check "All values in [0, 1)"
    (Array.for_all result ~f:(fun x -> Float.(x >= 0.0 && x < 1.0)));
  print_check "Matches expected stream within 0.01" Float.(max_abs_error <= 0.01);
  print_check "Mean within 0.40..0.60" Float.(mean >= 0.40 && mean <= 0.60)
