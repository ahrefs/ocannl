(* Roundtrip test for the Safetensors reader: writes a small file in the safetensors format (8-byte
   LE header length, JSON header, little-endian payloads), reads it back, and pushes one tensor
   through a device roundtrip via TDSL.wrap. *)

open Base
open Ocannl
open Stdio
open Nn_blocks.DSL_modules

let le_bytes_of_floats values =
  let b = Bytes.create (4 * Array.length values) in
  Array.iteri values ~f:(fun i v ->
      Stdlib.Bytes.set_int32_le b (4 * i) (Stdlib.Int32.bits_of_float v));
  Bytes.to_string b

let () =
  let a_vals = [| 0.; 1.; 2.; 3.; 4.; 5. |] in
  let b_vals = [| 10.; 20.; 30.; 40. |] in
  let header =
    {|{"__metadata__":{"format":"pt"},"a":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]},"b":{"dtype":"F32","shape":[4],"data_offsets":[24,40]}}|}
  in
  let len_bytes = Bytes.create 8 in
  Stdlib.Bytes.set_int64_le len_bytes 0 (Int64.of_int (String.length header));
  let path = "roundtrip.safetensors" in
  Out_channel.write_all path
    ~data:
      (Bytes.to_string len_bytes ^ header ^ le_bytes_of_floats a_vals ^ le_bytes_of_floats b_vals);

  let st = Safetensors.read path in
  printf "names: %s\n" (String.concat ~sep:", " (Safetensors.names st));
  List.iter (Safetensors.metadata st) ~f:(fun (k, v) -> printf "metadata %s = %s\n" k v);
  (match Safetensors.info st "a" with
  | Some { dtype; shape; offset; nbytes } ->
      printf "a: dtype %s, shape [%s], offset %d, nbytes %d\n" dtype
        (String.concat ~sep:"; " (List.map shape ~f:Int.to_string))
        offset nbytes
  | None -> printf "a: missing!\n");
  let a = Safetensors.to_float32 st "a" in
  Verdict.p_alli "a roundtrips (2x3)" (Array.to_list a_vals) ~f:(fun i v ->
      Float.equal (Bigarray.Genarray.get a [| i / 3; i % 3 |]) v);

  (* Device roundtrip through an OCANNL tensor. *)
  let b = TDSL.wrap ~l:"b" ~b:[] ~o:[ 4 ] (Safetensors.to_ndarray st "b") () in
  let ctx = Train.forward_once (Context.auto ()) b in
  let got = Context.get_values ctx b.Tensor.value in
  printf "b via tensor: [%s]\n"
    (String.concat ~sep:"; " (Array.to_list (Array.map got ~f:(fun v -> Printf.sprintf "%.0f" v))));

  (* Error paths. *)
  match Safetensors.to_float32 st "missing" with
  | exception Failure msg -> printf "missing tensor: %s\n" msg
  | _ -> printf "missing tensor: unexpectedly succeeded\n"

(* The payload ranges must tile the byte buffer exactly (safetensors invariant): overlapping or
   non-contiguous data_offsets, and trailing uncovered bytes, are rejected. *)
let () =
  let write_file path header payload =
    let len_bytes = Bytes.create 8 in
    Stdlib.Bytes.set_int64_le len_bytes 0 (Int64.of_int (String.length header));
    Out_channel.write_all path ~data:(Bytes.to_string len_bytes ^ header ^ payload)
  in
  let payload = le_bytes_of_floats [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let overlapping =
    {|{"a":{"dtype":"F32","shape":[4],"data_offsets":[0,16]},"b":{"dtype":"F32","shape":[4],"data_offsets":[8,24]}}|}
  in
  write_file "overlap.safetensors" overlapping payload;
  (match Safetensors.read "overlap.safetensors" with
  | exception Failure msg -> printf "overlap rejected: %s\n" msg
  | _ -> printf "overlap rejected: unexpectedly accepted\n");
  let trailing = {|{"a":{"dtype":"F32","shape":[4],"data_offsets":[0,16]}}|} in
  write_file "trailing.safetensors" trailing payload;
  match Safetensors.read "trailing.safetensors" with
  | exception Failure msg -> printf "trailing rejected: %s\n" msg
  | _ -> printf "trailing rejected: unexpectedly accepted\n"

(* Payload ingestion (gh-ocannl-587): payloads are mapped rather than decoded element by element,
   for every dtype that names an OCANNL precision. The format fixes little-endian order and
   contiguous unpadded payloads, so the only thing left to check per payload is the alignment of its
   file offset -- which the format does not guarantee, and which the copy path covers. *)
let payload_bytes prec values =
  let dims = [| Array.length values |] in
  let nd = Ir.Ndarray.create_array ~debug:"gen" prec ~dims ~padding:None in
  Ir.Ndarray.set_flat_values nd values;
  let tmp = "payload_gen.tmp" in
  Out_channel.with_file tmp ~binary:true ~f:(fun oc ->
      let _n : int = Ir.Ndarray.write_payload_to_channel nd oc in
      ());
  let data = In_channel.read_all tmp in
  Stdlib.Sys.remove tmp;
  data

(* [extra_pad] shifts the byte buffer off the 8-byte boundary the reference implementation pads the
   JSON header to -- the header length is the producer's choice, not a validity rule. *)
let build_file path ~extra_pad entries =
  let payloads = Buffer.create 256 in
  let json = Buffer.create 256 in
  Buffer.add_char json '{';
  List.iteri entries ~f:(fun i (name, dtype, shape, payload) ->
      if i > 0 then Buffer.add_char json ',';
      let start = Buffer.length payloads in
      Buffer.add_string payloads payload;
      Buffer.add_string json
        (Printf.sprintf {|"%s":{"dtype":"%s","shape":[%s],"data_offsets":[%d,%d]}|} name dtype
           (String.concat ~sep:"," (List.map shape ~f:Int.to_string))
           start (Buffer.length payloads)));
  Buffer.add_char json '}';
  let header = Buffer.contents json in
  let align_pad = (8 - ((8 + String.length header) % 8)) % 8 in
  let header = header ^ String.make (align_pad + extra_pad) ' ' in
  let len_bytes = Bytes.create 8 in
  Stdlib.Bytes.set_int64_le len_bytes 0 (Int64.of_int (String.length header));
  Out_channel.write_all path ~data:(Bytes.to_string len_bytes ^ header ^ Buffer.contents payloads);
  (* The byte buffer's file position, which is what the payload offsets are relative to. *)
  8 + String.length header

let () =
  printf "=== Payload ingestion: mapped and decoded ===\n";
  let values = [| 1.0; 2.0; 4.0; 8.0 |] in
  let dtypes =
    [
      ("f32", "F32", Ir.Ops.single);
      ("f64", "F64", Ir.Ops.double);
      ("f16", "F16", Ir.Ops.half);
      ("bf16", "BF16", Ir.Ops.bfloat16);
      ("fp8", "F8_E5M2", Ir.Ops.fp8);
      ("i32", "I32", Ir.Ops.int32);
      ("i64", "I64", Ir.Ops.int64);
      ("u16", "U16", Ir.Ops.uint16);
      ("u8", "U8", Ir.Ops.byte);
    ]
  in
  (* A zero-element tensor has nothing to map -- [Unix.map_file] has no empty mapping -- so it takes
     the decoding path at any alignment. *)
  let empty = ("empty", "F32", [ 0 ], "") in
  let entries =
    empty
    :: List.map dtypes ~f:(fun (name, dtype, prec) ->
        (name, dtype, [ 4 ], payload_bytes prec values))
  in
  (* The same payloads twice: once with the byte buffer on the 8-byte boundary the reference
     implementation pads to, once shifted a byte off it. Neither file is uniform: payload offsets
     accumulate the preceding payloads' sizes, so a wide dtype behind narrow ones can land unaligned
     even in the padded file. *)
  let aligned_start = build_file "aligned.safetensors" ~extra_pad:0 entries in
  let unaligned_start = build_file "unaligned.safetensors" ~extra_pad:1 entries in
  let read_all path buffer_start =
    let st = Safetensors.read path in
    let read =
      List.map (("empty", "F32", Ir.Ops.single) :: dtypes) ~f:(fun (name, _, prec) ->
          let m0, c0 = Ir.Ndarray.ingestion_counts () in
          let values = Ir.Ndarray.retrieve_flat_values (Safetensors.to_ndarray st name) in
          let m1, c1 = Ir.Ndarray.ingestion_counts () in
          let { Safetensors.offset; nbytes; _ } = Option.value_exn (Safetensors.info st name) in
          (* The decision has to be exactly "the payload's file position is element-aligned". *)
          let expected = nbytes > 0 && (buffer_start + offset) % Ir.Ops.prec_in_bytes prec = 0 in
          (name, values, m1 - m0 = 1 && c1 - c0 = 0, c1 - c0 = 1 && m1 - m0 = 0, expected))
    in
    Safetensors.close st;
    read
  in
  let aligned = read_all "aligned.safetensors" aligned_start in
  let unaligned = read_all "unaligned.safetensors" unaligned_start in
  let describe file entries =
    List.iter entries ~f:(fun (name, _, mapped, decoded, _) ->
        printf "  %s %s: %s\n" file name
          (if mapped then "mapped" else if decoded then "decoded" else "NEITHER"))
  in
  describe "aligned" aligned;
  describe "unaligned" unaligned;
  let decides_by_alignment entries =
    (not (List.is_empty entries))
    && List.for_all entries ~f:(fun (_, _, mapped, decoded, expected) ->
        Bool.equal mapped expected && Bool.equal decoded (not expected))
  in
  Verdict.p "ingestion maps exactly the element-aligned payloads"
    (decides_by_alignment aligned && decides_by_alignment unaligned);
  (* Shifting the buffer by a byte leaves only the one-byte precisions mappable. *)
  let one_byte = List.count dtypes ~f:(fun (_, _, prec) -> Ir.Ops.prec_in_bytes prec = 1) in
  Verdict.p "the shifted file maps only its one-byte payloads"
    (List.count unaligned ~f:(fun (_, _, mapped, _, _) -> mapped) = one_byte);
  List.iter2_exn aligned unaligned ~f:(fun (name, mapped, _, _, _) (_, decoded, _, _, _) ->
      printf "  %s: [%s]%s\n" name
        (String.concat ~sep:"; " (Array.to_list (Array.map mapped ~f:(Printf.sprintf "%.1f"))))
        (if Array.equal Float.equal mapped decoded then "" else " DIFFERS WHEN DECODED"));
  Verdict.p_all2 "mapping and decoding agree on values" (Array.of_list aligned)
    (Array.of_list unaligned) ~f:(fun (_, m, _, _, _) (_, d, _, _, _) ->
      Array.equal Float.equal m d)

let () =
  printf "=== Precision conversion, descriptor lifetime, rejections ===\n";
  let st = Safetensors.read "aligned.safetensors" in
  (* [?prec] converts on ingestion, for callers that want one precision regardless of the file's. *)
  let as_single = Safetensors.to_ndarray ~prec:Ir.Ops.single st "bf16" in
  printf "bf16 as single: [%s]\n"
    (String.concat ~sep:"; "
       (Array.to_list
          (Array.map (Ir.Ndarray.retrieve_flat_values as_single) ~f:(Printf.sprintf "%.1f"))));
  Verdict.p "converted payload has the requested precision"
    (Ir.Ops.equal_prec (Ir.Ndarray.get_prec as_single) Ir.Ops.single);
  (* A mapping outlives the descriptor it was taken through. *)
  let mapped = Safetensors.to_ndarray st "f32" in
  Safetensors.close st;
  Verdict.p "a mapped payload is readable after the reader is closed"
    (Array.equal Float.equal (Ir.Ndarray.retrieve_flat_values mapped) [| 1.0; 2.0; 4.0; 8.0 |]);
  Safetensors.close st;
  (match Safetensors.to_float32 st "f32" with
  | exception Failure msg -> printf "closed reader: %s\n" msg
  | _ -> printf "closed reader: unexpectedly succeeded\n");
  (* The upper half of an unsigned range survives ingestion and conversion: a U32 payload is stored
     in a signed int32 bigarray, so both the mapping's reads and [?prec] have to reinterpret it. *)
  let big = [| 0.0; 1.0; 2147483648.0; 4294967295.0 |] in
  let _ : int =
    build_file "u32.safetensors" ~extra_pad:0
      [ ("big", "U32", [ 4 ], payload_bytes Ir.Ops.uint32 big) ]
  in
  let st32 = Safetensors.read "u32.safetensors" in
  let mapped_u32 = Ir.Ndarray.retrieve_flat_values (Safetensors.to_ndarray st32 "big") in
  let as_double =
    Ir.Ndarray.retrieve_flat_values (Safetensors.to_ndarray ~prec:Ir.Ops.double st32 "big")
  in
  printf "u32 payload: [%s]\n"
    (String.concat ~sep:"; " (Array.to_list (Array.map mapped_u32 ~f:(Printf.sprintf "%.0f"))));
  Verdict.p "a mapped u32 payload keeps the values above 2^31"
    (Array.equal Float.equal mapped_u32 big);
  Verdict.p "converting a u32 payload to double keeps them" (Array.equal Float.equal as_double big);
  Safetensors.close st32;
  (* A dtype OCANNL has no precision for is refused, rather than reinterpreted. *)
  Verdict.p "I8 has no OCANNL precision" (Option.is_none (Safetensors.prec_of_dtype "I8"));
  Verdict.p "BF16 maps to bfloat16"
    (Option.value_map (Safetensors.prec_of_dtype "BF16") ~default:false ~f:(fun p ->
         Ir.Ops.equal_prec p Ir.Ops.bfloat16));
  let _ : int =
    build_file "i8.safetensors" ~extra_pad:0 [ ("q", "I8", [ 4 ], "\000\001\002\003") ]
  in
  let st8 = Safetensors.read "i8.safetensors" in
  (match Safetensors.to_ndarray st8 "q" with
  | exception Failure msg -> printf "I8 tensor: %s\n" msg
  | _ -> printf "I8 tensor: unexpectedly succeeded\n");
  Safetensors.close st8
