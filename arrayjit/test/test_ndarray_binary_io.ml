open Base
module Nd = Ir.Ndarray
module Ops = Ir.Ops

(* One temp file PER test, not one shared path: each mapped read below leaves a live mapping of the
   file, and Windows refuses to truncate a file some section still maps (the section pins the file's
   size), so reusing the path would fail the next test's [open_out_bin] there with EINVAL
   (gh-ocannl-588). Deleting the file under a live mapping is fine on every platform -- modern
   Windows uses POSIX delete semantics, the same fact that lets a checkpoint save rename over a
   mapped file -- so each test removes its own file while its mapping is still alive. *)
let fresh_tmp_file () = Stdlib.Filename.temp_file "ndarray_binary_io_test" ".bin"

let test_round_trip_prec prec_name prec init_f =
  let tmp_file = fresh_tmp_file () in
  let dims = [| 3; 4 |] in
  let nd1 = Nd.create_array ~debug:"test" prec ~dims ~padding:None in
  (* Initialize with known values *)
  let idx = Array.create ~len:2 0 in
  for i = 0 to 2 do
    for j = 0 to 3 do
      idx.(0) <- i;
      idx.(1) <- j;
      init_f nd1 idx ((i * 4) + j)
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
  Verdict.pass_fail (Printf.sprintf "%s round trip" prec_name) (Nd.payloads_equal nd1 nd2);
  (* The mapped read (gh-ocannl-467) reinterprets the payload bytes in place instead of decoding
     them element by element. That the two agree is the whole premise of mapped checkpoint loading:
     each precision's payload encoding has to be its in-memory representation. *)
  let fd = Unix.openfile tmp_file [ Unix.O_RDONLY ] 0 in
  let nd3 = Nd.map_file_array prec ~dims ~byte_offset:0 fd in
  Unix.close fd;
  Stdlib.Sys.remove tmp_file;
  Verdict.pass_fail (Printf.sprintf "%s mapped" prec_name) (Nd.payloads_equal nd1 nd3)

let test_padded () =
  let tmp_file = fresh_tmp_file () in
  let prec = Ops.single in
  let padding = Some ([| Ops.{ left = 1; right = 1 }; Ops.{ left = 0; right = 2 } |], 0.0) in
  (* Padded dims: 2+1+1=4 x 3+0+2=5, logical: 2x3 *)
  let dims = [| 4; 5 |] in
  let nd1 = Nd.create_array ~debug:"padded" prec ~dims ~padding in
  let padding_arr = [| Ops.{ left = 1; right = 1 }; Ops.{ left = 0; right = 2 } |] in
  (* Set logical values *)
  let idx = Array.create ~len:2 0 in
  for i = 0 to 1 do
    for j = 0 to 2 do
      idx.(0) <- i;
      idx.(1) <- j;
      Nd.set_from_float ~padding:padding_arr nd1 idx (Float.of_int ((i * 3) + j + 1))
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
  Stdlib.Sys.remove tmp_file;
  Verdict.pass_fail "padded round trip" (Nd.payloads_equal ~padding:padding_arr nd1 nd2)

(* A payload does not start at the beginning of the file, and its offset is not page aligned: the
   runtime maps from the enclosing page boundary and offsets the data pointer, so any offset works
   (which is why checkpoint alignment is about SIMD-friendly pointers, not about mappability). *)
let test_mapped_at_offset () =
  let prec = Ops.double in
  let dims = [| 5 |] in
  let nd1 = Nd.create_array ~debug:"offset" prec ~dims ~padding:None in
  for i = 0 to 4 do
    Nd.set_from_float nd1 [| i |] (Float.of_int i *. 1.5)
  done;
  let path = Stdlib.Filename.temp_file "ndarray_map_offset" ".bin" in
  let oc = Stdlib.open_out_bin path in
  let byte_offset = 7 in
  Stdlib.output_string oc (String.make byte_offset 'x');
  let _n : int = Nd.write_payload_to_channel nd1 oc in
  Stdlib.close_out oc;
  let fd = Unix.openfile path [ Unix.O_RDONLY ] 0 in
  let nd2 = Nd.map_file_array prec ~dims ~byte_offset fd in
  Unix.close fd;
  (* The mapping outlives the descriptor it was taken from, and the file it was taken from. *)
  Stdlib.Sys.remove path;
  Verdict.pass_fail "mapped at unaligned offset" (Nd.payloads_equal nd1 nd2)

(* The mappable-or-decode act itself (gh-ocannl-667). Both readers -- checkpoints and safetensors --
   ingest through [Nd.ingest_payload], and each can only A/B whole files: a checkpoint load takes
   [~mmap:false] for all of its payloads or none, and safetensors offers no way to decline a mapping
   at all. What is checked here is per payload, of one file whose layout takes BOTH paths: the
   reported path, the counter it bumped, and -- for a payload that is mappable -- that declining the
   mapping decodes the same bytes into an equal array.

   The layout is packed on purpose, which is where the mixture comes from: a one-byte payload ahead
   of a double leaves that double at an offset no multiple of 8, and it decodes. *)
let test_ingestion () =
  let path = Stdlib.Filename.temp_file "ndarray_ingest" ".bin" in
  let flat prec ~debug n ~f =
    let nd = Nd.create_array ~debug prec ~dims:[| n |] ~padding:None in
    for i = 0 to n - 1 do
      Nd.set_from_float nd [| i |] (f i)
    done;
    nd
  in
  let a = flat Ops.byte ~debug:"a" 8 ~f:(fun i -> Float.of_int (i + 1)) in
  let b = flat Ops.double ~debug:"b" 4 ~f:(fun i -> (Float.of_int i *. 1.5) +. 0.25) in
  let c = flat Ops.byte ~debug:"c" 1 ~f:(fun _ -> 9.0) in
  let d = flat Ops.double ~debug:"d" 4 ~f:(fun i -> (Float.of_int i *. -2.5) -. 0.125) in
  let padding_arr = [| Ops.{ left = 1; right = 1 }; Ops.{ left = 0; right = 2 } |] in
  let e =
    Nd.create_array ~debug:"e" Ops.single ~dims:[| 4; 5 |] ~padding:(Some (padding_arr, 0.0))
  in
  for i = 0 to 1 do
    for j = 0 to 2 do
      Nd.set_from_float ~padding:padding_arr e [| i; j |] (Float.of_int ((i * 3) + j + 1))
    done
  done;
  let oc = Stdlib.open_out_bin path in
  (* Sequential lets, not `and`: the offsets are the channel positions, so the order matters. *)
  let put ?padding nd =
    let at = Stdlib.pos_out oc in
    let n = Nd.write_payload_to_channel ?padding nd oc in
    (at, n)
  in
  let a_at = put a in
  let b_at = put b in
  let c_at = put c in
  let d_at = put d in
  let e_at = put ~padding:padding_arr e in
  Stdlib.close_out oc;
  let ic = Stdlib.open_in_bin path in
  let ingest ?padding ?mmap ~debug prec ~dims (byte_offset, nbytes) =
    let m0, d0 = Nd.ingestion_counts () in
    let nd, taken = Nd.ingest_payload ?padding ?mmap ~debug prec ~dims ~byte_offset ~nbytes ic in
    let m1, d1 = Nd.ingestion_counts () in
    let counted =
      match taken with
      | Nd.Mapped -> m1 - m0 = 1 && d1 - d0 = 0
      | Nd.Decoded -> d1 - d0 = 1 && m1 - m0 = 0
    in
    (nd, taken, counted)
  in
  let a_nd, a_path, a_counted = ingest ~debug:"a" Ops.byte ~dims:[| 8 |] a_at in
  let b_nd, b_path, b_counted = ingest ~debug:"b" Ops.double ~dims:[| 4 |] b_at in
  let c_nd, c_path, c_counted = ingest ~debug:"c" Ops.byte ~dims:[| 1 |] c_at in
  let d_nd, d_path, d_counted = ingest ~debug:"d" Ops.double ~dims:[| 4 |] d_at in
  let e_nd, e_path, e_counted =
    ingest ~padding:(padding_arr, 0.0) ~debug:"e" Ops.single ~dims:[| 4; 5 |] e_at
  in
  (* The same payload, same offset, the other act. *)
  let b_dec, b_dec_path, b_dec_counted =
    ingest ~mmap:false ~debug:"b_dec" Ops.double ~dims:[| 4 |] b_at
  in
  Stdlib.close_in ic;
  (* The mappings outlive the descriptor they were taken from, and the file itself. *)
  Stdlib.Sys.remove path;
  let mapped = function Nd.Mapped -> true | Nd.Decoded -> false in
  let name_of p = if mapped p then "mapped" else "decoded" in
  Stdio.printf "  packed layout: a@%d b@%d c@%d d@%d e@%d\n" (fst a_at) (fst b_at) (fst c_at)
    (fst d_at) (fst e_at);
  Stdio.printf "  paths: a %s, b %s, c %s, d %s, e %s\n" (name_of a_path) (name_of b_path)
    (name_of c_path) (name_of d_path) (name_of e_path);
  Verdict.p "every ingestion bumps exactly the counter for the path it reports"
    (a_counted && b_counted && c_counted && d_counted && e_counted && b_dec_counted);
  Verdict.p "an element-aligned unpadded payload of a packed file is mapped"
    (mapped a_path && mapped b_path && mapped c_path);
  Verdict.p "a payload at an offset its element size does not divide is decoded"
    (not (mapped d_path));
  Verdict.p "a padded payload is decoded, whatever its offset" (not (mapped e_path));
  Verdict.p "declining the mapping decodes the same bytes into an equal array"
    ((not (mapped b_dec_path)) && Nd.payloads_equal b_nd b_dec);
  Verdict.p "both paths reproduce the payloads they ingested"
    (Nd.payloads_equal a a_nd && Nd.payloads_equal b b_nd && Nd.payloads_equal c c_nd
   && Nd.payloads_equal d d_nd
    && Nd.payloads_equal ~padding:padding_arr e e_nd)

let () =
  (* Test each precision type *)
  Verdict.case "Byte" (fun () ->
      test_round_trip_prec "Byte" Ops.byte (fun nd idx i ->
          Nd.set_from_float nd idx (Float.of_int (i % 256))));
  Verdict.case "Uint16" (fun () ->
      test_round_trip_prec "Uint16" Ops.uint16 (fun nd idx i ->
          Nd.set_from_float nd idx (Float.of_int (i * 1000))));
  Verdict.case "Int32" (fun () ->
      test_round_trip_prec "Int32" Ops.int32 (fun nd idx i ->
          Nd.set_from_float nd idx (Float.of_int ((i * 100000) - 500000))));
  Verdict.case "Uint32" (fun () ->
      test_round_trip_prec "Uint32" Ops.uint32 (fun nd idx i ->
          Nd.set_from_float nd idx (Float.of_int (i * 100000))));
  Verdict.case "Int64" (fun () ->
      test_round_trip_prec "Int64" Ops.int64 (fun nd idx i ->
          Nd.set_from_float nd idx (Float.of_int ((i + 1) * 1_000_000_000_000))));
  Verdict.case "Uint64" (fun () ->
      test_round_trip_prec "Uint64" Ops.uint64 (fun nd idx i ->
          Nd.set_from_float nd idx (Float.of_int ((i + 1) * 1_000_000_000_000))));
  Verdict.case "Half" (fun () ->
      test_round_trip_prec "Half" Ops.half (fun nd idx i ->
          Nd.set_from_float nd idx (Float.of_int i *. 0.5)));
  Verdict.case "Bfloat16" (fun () ->
      test_round_trip_prec "Bfloat16" Ops.bfloat16 (fun nd idx i ->
          Nd.set_from_float nd idx (Float.of_int i *. 0.25)));
  Verdict.case "Fp8" (fun () ->
      test_round_trip_prec "Fp8" Ops.fp8 (fun nd idx i ->
          Nd.set_from_float nd idx (Float.of_int (i % 128) *. 0.1)));
  Verdict.case "Single" (fun () ->
      test_round_trip_prec "Single" Ops.single (fun nd idx i ->
          Nd.set_from_float nd idx (Float.of_int i *. 3.14)));
  Verdict.case "Double" (fun () ->
      test_round_trip_prec "Double" Ops.double (fun nd idx i ->
          Nd.set_from_float nd idx (Float.of_int i *. 2.71828)));
  (* Note: Uint4x32 uses Complex.t carrier with raw byte access *)
  Verdict.case "Uint4x32" (fun () ->
      test_round_trip_prec "Uint4x32" Ops.uint4x32 (fun nd idx i ->
          Nd.set_from_float nd idx (Float.of_int ((i * 7) + 3))));
  (* Test padded tensor *)
  Verdict.case "test_padded" test_padded;
  Verdict.case "test_mapped_at_offset" test_mapped_at_offset;
  Verdict.case "test_ingestion" test_ingestion
