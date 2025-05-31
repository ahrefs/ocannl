open Base
module Ndarray = Ir.Ndarray
module Ops = Ir.Ops

let test_bfloat16_conversions () =
  Stdio.printf "Testing BFloat16 conversions:\n";

  (* Test some specific values *)
  let test_values = [ 0.0; 1.0; -1.0; 3.14159; 1e-3; 1e3; Float.infinity; Float.neg_infinity ] in

  List.iter test_values ~f:(fun orig ->
      let bf16 = Ndarray.float_to_bfloat16 orig in
      let back = Ndarray.bfloat16_to_float bf16 in
      Stdio.printf "  %.6f -> 0x%04x -> %.6f\n" orig bf16 back);

  (* Test round-trip through ndarray *)
  let arr =
    Ndarray.create_array ~debug:"test" Ops.bfloat16 ~dims:[| 3; 2 |]
      (Ops.Constant_fill { values = [| 1.0; 2.0; 3.14; -1.5; 0.125; 1000.0 |]; strict = true })
  in

  Stdio.printf "\nBFloat16 array values:\n";
  let flat_values = Ndarray.retrieve_flat_values arr in
  Array.iteri flat_values ~f:(fun i v -> Stdio.printf "  [%d] = %.6f\n" i v)

let test_fp8_conversions () =
  Stdio.printf "\n\nTesting FP8 conversions:\n";

  (* Test some specific values *)
  let test_values = [ 0.0; 1.0; -1.0; 0.5; 2.0; 0.125; 16.0; -0.25 ] in

  List.iter test_values ~f:(fun orig ->
      let fp8 = Ndarray.float_to_fp8 orig in
      let back = Ndarray.fp8_to_float fp8 in
      Stdio.printf "  %.6f -> 0x%02x -> %.6f\n" orig fp8 back);

  (* Test round-trip through ndarray *)
  let arr =
    Ndarray.create_array ~debug:"test" Ops.fp8 ~dims:[| 2; 2 |]
      (Ops.Constant_fill { values = [| 1.0; 0.5; 2.0; -1.0 |]; strict = true })
  in

  Stdio.printf "\nFP8 array values:\n";
  let flat_values = Ndarray.retrieve_flat_values arr in
  Array.iteri flat_values ~f:(fun i v -> Stdio.printf "  [%d] = %.6f\n" i v)

let () =
  test_bfloat16_conversions ();
  test_fp8_conversions ()
