open Base
module Nd = Ir.Ndarray
module Ops = Ir.Ops

let tmp_file = Stdlib.Filename.temp_file "ndarray_binary_io_test" ".bin"

let test_round_trip_prec prec_name prec init_f =
  let dims = [| 3; 4 |] in
  let nd1 = Nd.create_array ~debug:"test" prec ~dims ~padding:None in
  (* Initialize with known values *)
  let idx = Array.create ~len:2 0 in
  for i = 0 to 2 do
    for j = 0 to 3 do
      idx.(0) <- i;
      idx.(1) <- j;
      init_f nd1 idx (i * 4 + j)
    done
  done;
  (* Write payload to file *)
  let oc = Stdlib.open_out_bin tmp_file in
  let n_bytes = Nd.write_payload_to_channel nd1 oc in
  Stdlib.close_out oc;
  (* Read payload into fresh ndarray *)
  let nd2 = Nd.create_array ~debug:"test2" prec ~dims ~padding:None in
  let ic = Stdlib.open_in_bin tmp_file in
  Nd.read_payload_from_channel nd2 ic n_bytes;
  Stdlib.close_in ic;
  (* Compare using exact byte comparison *)
  if Nd.payloads_equal nd1 nd2 then Stdio.printf "PASS: %s\n" prec_name
  else Stdio.printf "FAIL: %s\n" prec_name

let test_padded () =
  let prec = Ops.single in
  let padding =
    Some ([| Ops.{ left = 1; right = 1 }; Ops.{ left = 0; right = 2 } |], Some 0.0)
  in
  (* Padded dims: 2+1+1=4 x 3+0+2=5, logical: 2x3 *)
  let dims = [| 4; 5 |] in
  let nd1 = Nd.create_array ~debug:"padded" prec ~dims ~padding in
  let padding_arr =
    [| Ops.{ left = 1; right = 1 }; Ops.{ left = 0; right = 2 } |]
  in
  (* Set logical values *)
  let idx = Array.create ~len:2 0 in
  for i = 0 to 1 do
    for j = 0 to 2 do
      idx.(0) <- i;
      idx.(1) <- j;
      Nd.set_from_float ~padding:padding_arr nd1 idx (Float.of_int (i * 3 + j + 1))
    done
  done;
  (* Write payload with padding *)
  let oc = Stdlib.open_out_bin tmp_file in
  let n_bytes = Nd.write_payload_to_channel ~padding:padding_arr nd1 oc in
  Stdlib.close_out oc;
  (* Read into fresh padded ndarray *)
  let nd2 = Nd.create_array ~debug:"padded2" prec ~dims ~padding in
  let ic = Stdlib.open_in_bin tmp_file in
  Nd.read_payload_from_channel ~padding:padding_arr nd2 ic n_bytes;
  Stdlib.close_in ic;
  (* Compare logical payloads *)
  if Nd.payloads_equal ~padding:padding_arr nd1 nd2 then
    Stdio.printf "PASS: padded\n"
  else Stdio.printf "FAIL: padded\n"

let () =
  (* Test each precision type *)
  test_round_trip_prec "Byte" Ops.byte (fun nd idx i ->
      Nd.set_from_float nd idx (Float.of_int (i % 256)));
  test_round_trip_prec "Uint16" Ops.uint16 (fun nd idx i ->
      Nd.set_from_float nd idx (Float.of_int (i * 1000)));
  test_round_trip_prec "Int32" Ops.int32 (fun nd idx i ->
      Nd.set_from_float nd idx (Float.of_int (i * 100000 - 500000)));
  test_round_trip_prec "Uint32" Ops.uint32 (fun nd idx i ->
      Nd.set_from_float nd idx (Float.of_int (i * 100000)));
  test_round_trip_prec "Int64" Ops.int64 (fun nd idx i ->
      Nd.set_from_float nd idx (Float.of_int ((i + 1) * 1_000_000_000_000)));
  test_round_trip_prec "Uint64" Ops.uint64 (fun nd idx i ->
      Nd.set_from_float nd idx (Float.of_int ((i + 1) * 1_000_000_000_000)));
  test_round_trip_prec "Half" Ops.half (fun nd idx i ->
      Nd.set_from_float nd idx (Float.of_int i *. 0.5));
  test_round_trip_prec "Bfloat16" Ops.bfloat16 (fun nd idx i ->
      Nd.set_from_float nd idx (Float.of_int i *. 0.25));
  test_round_trip_prec "Fp8" Ops.fp8 (fun nd idx i ->
      Nd.set_from_float nd idx (Float.of_int (i % 128) *. 0.1));
  test_round_trip_prec "Single" Ops.single (fun nd idx i ->
      Nd.set_from_float nd idx (Float.of_int i *. 3.14));
  test_round_trip_prec "Double" Ops.double (fun nd idx i ->
      Nd.set_from_float nd idx (Float.of_int i *. 2.71828));
  (* Note: Uint4x32 uses Complex.t carrier with raw byte access *)
  test_round_trip_prec "Uint4x32" Ops.uint4x32 (fun nd idx i ->
      Nd.set_from_float nd idx (Float.of_int (i * 7 + 3)));
  (* Test padded tensor *)
  test_padded ();
  (* Clean up *)
  Stdlib.Sys.remove tmp_file
