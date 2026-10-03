open Base
module Ndarray = Ir.Ndarray
module Ops = Ir.Ops

let test_bfloat16_conversions () =
  Stdio.printf "Testing BFloat16 conversions:\n";

  (* Test some specific values *)
  let test_values = [ 0.0; 1.0; -1.0; 3.14159; 1e-3; 1e3; Float.infinity; Float.neg_infinity ] in

  List.iter test_values ~f:(fun orig ->
      let bf16 = Ops.single_to_bfloat16 orig in
      let back = Ops.bfloat16_to_single bf16 in
      Stdio.printf "  %.6f -> 0x%04x -> %.6f\n" orig bf16 back);

  (* Test round-trip through ndarray *)
  let arr = Ndarray.create_array ~debug:"test" Ops.bfloat16 ~dims:[| 3; 2 |] ~padding:None in
  Ndarray.set_flat_values arr (Array.of_list test_values);

  Stdio.printf "\nBFloat16 array values:\n";
  let flat_values = Ndarray.retrieve_flat_values arr in
  Array.iteri flat_values ~f:(fun i v -> Stdio.printf "  [%d] = %.6f\n" i v)

let test_fp8_conversions () =
  Stdio.printf "\n\nTesting FP8 conversions:\n";

  (* Test some specific values *)
  let test_values = [ 0.0; 1.0; -1.0; 0.5; 2.0; 0.125; 16.0; -0.25 ] in

  List.iter test_values ~f:(fun orig ->
      let fp8 = Ops.single_to_fp8 orig in
      let back = Ops.fp8_to_single fp8 in
      Stdio.printf "  %.6f -> 0x%02x -> %.6f\n" orig fp8 back);

  (* Test round-trip through ndarray *)
  let arr = Ndarray.create_array ~debug:"test" Ops.fp8 ~dims:[| 2; 2 |] ~padding:None in
  Ndarray.set_flat_values arr (Array.of_list test_values);

  Stdio.printf "\nFP8 array values:\n";
  let flat_values = Ndarray.retrieve_flat_values arr in
  Array.iteri flat_values ~f:(fun i v -> Stdio.printf "  [%d] = %.6f\n" i v)

let test_padding () =
  Stdio.printf "\n\nTesting padding functionality:\n";

  (* Test padding with float32 array *)
  let padding_config = [| { Ops.left = 1; right = 1 }; { left = 2; right = 1 } |] in
  (* left=1,right=1 for first dim; left=2,right=1 for second dim *)
  let padding_value = -999.0 in

  let padded_dims = [| 4; 6 |] in
  (* (2+1+1) x (3+2+1) *)

  let arr =
    Ndarray.create_array ~debug:"padded_test" Ops.single ~dims:padded_dims
      ~padding:(Some (padding_config, padding_value))
  in
  Ndarray.set_flat_values ~padding:padding_config arr [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0 |];
  Stdio.printf "Padded array (dims 4x6, unpadded region 2x3):\n";
  let dims = Ndarray.dims arr in
  for i = 0 to dims.(0) - 1 do
    for j = 0 to dims.(1) - 1 do
      let idx = [| i; j |] in
      let value = Ndarray.get_as_float arr idx in
      Stdio.printf "%8.1f " value
    done;
    Stdio.printf "\n"
  done;

  Stdio.printf
    "\nExpected: padding value (-999.0) in margins, data values (1.0-6.0) in center region\n"

(* [uint32] and [uint64] are stored in signed int32/int64 bigarrays, so every float-facing
   conversion has to reinterpret the bits: read through the host's signed conversion, the u32
   0xffffffff is -1., and it cannot be written back at all. The values below straddle the sign bit,
   and each is exactly representable as a double, so a round-trip that changes anything is the
   conversion's doing. *)
let test_unsigned_extremes () =
  Stdio.printf "\n\nTesting unsigned values across the sign bit:\n";
  let cases =
    [
      ("uint32", Ops.uint32, [| 0.0; 1.0; 2147483647.0; 2147483648.0; 4294967295.0 |]);
      ( "uint64",
        Ops.uint64,
        [| 0.0; 1.0; 4294967296.0; 9223372036854775808.0; 18446744073709549568.0 |] );
    ]
  in
  List.iter cases ~f:(fun (name, prec, values) ->
      let arr =
        Ndarray.create_array ~debug:name prec ~dims:[| Array.length values |] ~padding:None
      in
      Ndarray.set_flat_values arr values;
      let back = Ndarray.retrieve_flat_values arr in
      (* %h, not %.1f: these exceed the ~17 significant digits the Windows C runtime formats, so a
         decimal print of 2^63 diverges between platforms; OCaml's hex-float formatting does not. *)
      Stdio.printf "  %s: [%s]\n" name
        (String.concat ~sep:"; " (Array.to_list (Array.map back ~f:(Printf.sprintf "%h"))));
      Verdict.p (name ^ " round-trips through float") (Array.equal Float.equal back values);
      (* The same conversion the ndarray-precision APIs go through when a caller asks for floats. *)
      let widened = Ndarray.retrieve_flat_values (Ndarray.convert Ops.double arr) in
      Verdict.p (name ^ " widens to double unchanged") (Array.equal Float.equal widened values))

(* A u64 whose value is not representable as a double must round ONCE, to the nearest double. The
   values below are raw bit patterns rather than floats -- a test built from already-representable
   floats cannot see a rounding bug at all -- and the oracle is [Float.of_string] on the exact
   decimal, i.e. the C library's correctly-rounded strtod rather than a hand-computed constant. *)
let test_unsigned_rounding () =
  Stdio.printf "\n\nTesting u64 values that a double cannot hold exactly:\n";
  let cases =
    [
      (* 2^63 + 1025: halving it lands exactly on a tie, so a conversion that rounds the half and
         then doubles gives 2^63, an ulp below the nearest double. *)
      (0x8000000000000401L, "9223372036854776833");
      (0xFFFFFFFFFFFFFFFFL, "18446744073709551615");
      (0x8000000000000801L, "9223372036854777857");
      (0x7FFFFFFFFFFFFFFFL, "9223372036854775807");
    ]
  in
  let arr =
    Ndarray.create_array ~debug:"u64_bits" Ops.uint64 ~dims:[| List.length cases |] ~padding:None
  in
  (match arr with
  | Ndarray.Uint64_nd ba ->
      List.iteri cases ~f:(fun i (bits, _) -> Bigarray.Genarray.set ba [| i |] bits)
  | _ -> Verdict.fail "uint64 array is not Uint64_nd");
  let got = Ndarray.retrieve_flat_values arr in
  List.iteri cases ~f:(fun i (_, decimal) ->
      let expected = Float.of_string decimal in
      let ok = Float.equal got.(i) expected in
      Stdio.printf "  %s -> %s\n" decimal (if ok then "nearest" else "OFF");
      Verdict.p ("u64 " ^ decimal ^ " converts to the nearest double") ok)

let () =
  Verdict.case "test_bfloat16_conversions" test_bfloat16_conversions;
  Verdict.case "test_fp8_conversions" test_fp8_conversions;
  Verdict.case "test_padding" test_padding;
  Verdict.case "test_unsigned_extremes" test_unsigned_extremes;
  Verdict.case "test_unsigned_rounding" test_unsigned_rounding
