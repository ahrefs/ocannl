open Base
module Tn = Ir.Tnode
module Nd = Ir.Ndarray
module Ops = Ir.Ops
module Tensor = Ocannl_tensor.Tensor
module Persistence = Ocannl.Persistence

(* Regression test for gh-ocannl-333 / gh-ocannl-373: tensor persistence is context-mediated. There
   is no host array on a tensor node; [save] reads each node's values from its device buffer via the
   context, and [load]/[restore] upload file data into the context, returning the updated context.
   The round-trips below assert that saved values reload/restore exactly. *)

let tmp_dir = Stdlib.Filename.get_temp_dir_name ()
let tmp_file name = Stdlib.Filename.concat tmp_dir ("test_persistence_" ^ name ^ ".ckpt")

let cleanup name =
  let path = tmp_file name in
  if Stdlib.Sys.file_exists path then Stdlib.Sys.remove path

(* Create a tnode with given values and upload it into [ctx], returning the updated context. *)
let make_tn ctx ~id ~label ?(padding = None) prec dims values =
  let nd = Nd.create_array ~debug:"test" prec ~dims ~padding in
  Nd.set_flat_values nd values;
  let tn, _init = Tn.create_from_padded ~id ~label ~ndarray:nd ~padding () in
  (Context.from_host ctx tn nd, tn)

let show ctx tn =
  String.concat ~sep:"; "
    (Array.to_list
       (Array.map (Context.get_values ctx tn) ~f:(fun v -> Stdlib.Printf.sprintf "%.1f" v)))

let () =
  (* === Test 1: Round-trip save/load === *)
  Stdio.printf "=== Test 1: Round-trip save/load ===\n";
  Tensor.unsafe_reinitialize ();
  let ctx = Context.cpu () in
  let ctx, tn1 =
    make_tn ctx ~id:0 ~label:[ "weights" ] Ops.single [| 2; 3 |] [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0 |]
  in
  let ctx, tn2 = make_tn ctx ~id:1 ~label:[ "bias" ] Ops.double [| 3 |] [| 10.0; 20.0; 30.0 |] in
  let t_set = Set.of_list (module Tn) [ tn1; tn2 ] in
  let path = tmp_file "roundtrip" in
  Persistence.save ~ctx ~appending:false t_set path;
  (* Reset and load *)
  Tensor.unsafe_reinitialize ();
  let ctx = Context.cpu () in
  let ctx, loaded = Persistence.load ~ctx path in
  let loaded_list = Set.to_list loaded in
  Stdio.printf "Loaded %d tensors\n" (List.length loaded_list);
  List.iter loaded_list ~f:(fun tn ->
      Stdio.printf "  id=%d label=%s values=[%s]\n" tn.Tn.id
        (String.concat ~sep:"." tn.Tn.label)
        (show ctx tn));
  cleanup "roundtrip";

  (* === Test 2: Restore === *)
  Stdio.printf "=== Test 2: Restore ===\n";
  Tensor.unsafe_reinitialize ();
  let ctx = Context.cpu () in
  let ctx, tn1 = make_tn ctx ~id:0 ~label:[ "w" ] Ops.single [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |] in
  let t_set = Set.of_list (module Tn) [ tn1 ] in
  let path = tmp_file "restore" in
  Persistence.save ~ctx ~appending:false t_set path;
  (* Modify values on-device *)
  let ctx = Context.set_values ctx tn1 [| 99.0; 99.0; 99.0; 99.0 |] in
  Stdio.printf "Before restore: [%s]\n" (show ctx tn1);
  (* Restore original values *)
  let ctx = Persistence.restore ~ctx t_set path in
  Stdio.printf "After restore: [%s]\n" (show ctx tn1);
  cleanup "restore";

  (* === Test 3: Append mode - disjoint sets === *)
  Stdio.printf "=== Test 3: Append mode ===\n";
  Tensor.unsafe_reinitialize ();
  let ctx = Context.cpu () in
  let ctx, tn_a = make_tn ctx ~id:0 ~label:[ "a" ] Ops.single [| 2 |] [| 1.0; 2.0 |] in
  let ctx, tn_b = make_tn ctx ~id:1 ~label:[ "b" ] Ops.single [| 2 |] [| 3.0; 4.0 |] in
  let path = tmp_file "append" in
  Persistence.save ~ctx ~appending:false (Set.of_list (module Tn) [ tn_a ]) path;
  Persistence.save ~ctx ~appending:true (Set.of_list (module Tn) [ tn_b ]) path;
  Tensor.unsafe_reinitialize ();
  let ctx = Context.cpu () in
  let ctx, loaded = Persistence.load ~ctx path in
  Stdio.printf "Loaded %d tensors after append\n" (Set.length loaded);
  Set.iter loaded ~f:(fun tn -> Stdio.printf "  id=%d values=[%s]\n" tn.Tn.id (show ctx tn));
  cleanup "append";

  (* === Test 4: Append overwrite === *)
  Stdio.printf "=== Test 4: Append overwrite ===\n";
  Tensor.unsafe_reinitialize ();
  let ctx = Context.cpu () in
  let ctx, tn_orig = make_tn ctx ~id:0 ~label:[ "x" ] Ops.single [| 2 |] [| 10.0; 20.0 |] in
  let path = tmp_file "overwrite" in
  Persistence.save ~ctx ~appending:false (Set.of_list (module Tn) [ tn_orig ]) path;
  (* Overwrite with new values *)
  let ctx = Context.set_values ctx tn_orig [| 77.0; 88.0 |] in
  Persistence.save ~ctx ~appending:true (Set.of_list (module Tn) [ tn_orig ]) path;
  Tensor.unsafe_reinitialize ();
  let ctx = Context.cpu () in
  let ctx, loaded = Persistence.load ~ctx path in
  Set.iter loaded ~f:(fun tn -> Stdio.printf "  id=%d values=[%s]\n" tn.Tn.id (show ctx tn));
  cleanup "overwrite";

  (* === Test 5: Empty checkpoint === *)
  Stdio.printf "=== Test 5: Empty checkpoint ===\n";
  Tensor.unsafe_reinitialize ();
  let ctx = Context.cpu () in
  let empty = Set.empty (module Tn) in
  let path = tmp_file "empty" in
  Persistence.save ~ctx ~appending:false empty path;
  Tensor.unsafe_reinitialize ();
  let ctx = Context.cpu () in
  let ctx, loaded = Persistence.load ~ctx path in
  Stdio.printf "Loaded %d tensors from empty checkpoint\n" (Set.length loaded);
  let _ctx = Persistence.restore ~ctx empty path in
  Stdio.printf "Restore on empty set succeeded\n";
  cleanup "empty";

  (* === Test 6: Error - missing tensor on restore === *)
  Stdio.printf "=== Test 6: Missing tensor on restore ===\n";
  Tensor.unsafe_reinitialize ();
  let ctx = Context.cpu () in
  let ctx, tn1 = make_tn ctx ~id:0 ~label:[ "present" ] Ops.single [| 2 |] [| 1.0; 2.0 |] in
  let ctx, tn2 = make_tn ctx ~id:1 ~label:[ "absent" ] Ops.single [| 2 |] [| 3.0; 4.0 |] in
  ignore tn1;
  let path = tmp_file "missing" in
  Persistence.save ~ctx ~appending:false (Set.of_list (module Tn) [ tn1 ]) path;
  (try
     let _ctx = Persistence.restore ~ctx (Set.of_list (module Tn) [ tn2 ]) path in
     Verdict.fail "should have raised"
   with Failure msg -> Stdio.printf "Caught: %s\n" msg);
  cleanup "missing";

  (* === Test 7: Error - dimension mismatch on restore === *)
  Stdio.printf "=== Test 7: Dimension mismatch on restore ===\n";
  Tensor.unsafe_reinitialize ();
  let ctx = Context.cpu () in
  let ctx, tn_save =
    make_tn ctx ~id:0 ~label:[ "d" ] Ops.single [| 2; 3 |] [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0 |]
  in
  let path = tmp_file "dimfail" in
  Persistence.save ~ctx ~appending:false (Set.of_list (module Tn) [ tn_save ]) path;
  (* Create a tnode with different dims but same id *)
  Tensor.unsafe_reinitialize ();
  let ctx = Context.cpu () in
  let ctx, tn_wrong =
    make_tn ctx ~id:0 ~label:[ "d" ] Ops.single [| 3; 2 |] [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0 |]
  in
  (try
     let _ctx = Persistence.restore ~ctx (Set.of_list (module Tn) [ tn_wrong ]) path in
     Verdict.fail "should have raised"
   with Failure msg -> Stdio.printf "Caught: %s\n" msg);
  cleanup "dimfail";

  (* === Test 8: Error - ID clash on load === *)
  Stdio.printf "=== Test 8: ID clash on load ===\n";
  Tensor.unsafe_reinitialize ();
  let ctx = Context.cpu () in
  let ctx, tn1 = make_tn ctx ~id:0 ~label:[ "clash" ] Ops.single [| 2 |] [| 1.0; 2.0 |] in
  ignore tn1;
  let path = tmp_file "clash" in
  Persistence.save ~ctx ~appending:false (Set.of_list (module Tn) [ tn1 ]) path;
  (* Don't reinitialize - tn1 is still in registry *)
  (try
     let _ = Persistence.load ~ctx path in
     Verdict.fail "should have raised"
   with Failure msg -> Stdio.printf "Caught: %s\n" msg);
  cleanup "clash";

  (* === Test 9: ID floor after load === *)
  Stdio.printf "=== Test 9: ID floor after load ===\n";
  Tensor.unsafe_reinitialize ();
  let ctx = Context.cpu () in
  let ctx, tn1 = make_tn ctx ~id:5 ~label:[ "high_id" ] Ops.single [| 2 |] [| 1.0; 2.0 |] in
  ignore tn1;
  let path = tmp_file "idfloor" in
  Persistence.save ~ctx ~appending:false (Set.of_list (module Tn) [ tn1 ]) path;
  Tensor.unsafe_reinitialize ();
  let ctx = Context.cpu () in
  let next_before = Tensor.get_next_id () in
  let _ctx, _loaded = Persistence.load ~ctx path in
  let next_after = Tensor.get_next_id () in
  Stdio.printf "next_id before load=%d, after load=%d (should be >= 6)\n" next_before next_after;
  cleanup "idfloor";

  (* === Test 10: Error - padding mismatch on restore === *)
  Stdio.printf "=== Test 10: Padding mismatch on restore ===\n";
  Tensor.unsafe_reinitialize ();
  let ctx = Context.cpu () in
  let padding1 = Some ([| Ops.{ left = 1; right = 1 } |], 0.0) in
  let ctx, tn_padded =
    make_tn ctx ~id:0 ~label:[ "p" ] ~padding:padding1 Ops.single [| 4 |] [| 1.0; 2.0 |]
  in
  ignore tn_padded;
  let path = tmp_file "padmismatch" in
  Persistence.save ~ctx ~appending:false (Set.of_list (module Tn) [ tn_padded ]) path;
  (* Create a tnode with same dims but different padding *)
  Tensor.unsafe_reinitialize ();
  let ctx = Context.cpu () in
  let padding2 = Some ([| Ops.{ left = 0; right = 2 } |], 0.0) in
  let ctx, tn_diff_pad =
    make_tn ctx ~id:0 ~label:[ "p" ] ~padding:padding2 Ops.single [| 4 |] [| 1.0; 2.0 |]
  in
  (try
     let _ctx = Persistence.restore ~ctx (Set.of_list (module Tn) [ tn_diff_pad ]) path in
     Verdict.fail "should have raised"
   with Failure msg -> Stdio.printf "Caught: %s\n" msg);
  cleanup "padmismatch";

  (* === Test 11: Aligned payload offsets (gh-ocannl-467) === *)
  Stdio.printf "=== Test 11: Payload alignment ===\n";
  (* Parses the file layout independently of Persistence's own reader, so that the on-disk format is
     pinned rather than the writer's agreement with itself. *)
  let file_layout path =
    let ic = Stdlib.open_in_bin path in
    let header_len = Stdlib.input_binary_int ic in
    let buf = Bytes.create header_len in
    Stdlib.really_input ic buf 0 header_len;
    let data_start = Stdlib.pos_in ic in
    Stdlib.close_in ic;
    let sexp = Sexplib.Sexp.of_string (String.strip (Bytes.to_string buf)) in
    let field key sexp =
      match sexp with
      | Sexp.List fields ->
          List.find_map fields ~f:(function
            | Sexp.List [ Sexp.Atom k; v ] when String.equal k key -> Some v
            | _ -> None)
      | _ -> None
    in
    let int_field key sexp =
      match field key sexp with Some (Sexp.Atom s) -> Int.of_string s | _ -> failwith key
    in
    let tensors =
      match field "tensors" sexp with
      | Some (Sexp.List metas) ->
          List.map metas ~f:(fun m -> (int_field "id" m, int_field "offset" m))
      | _ -> failwith "tensors"
    in
    (int_field "alignment" sexp, data_start, tensors)
  in
  let report path =
    let alignment, data_start, tensors = file_layout path in
    Stdio.printf "  alignment=%d, data_start mod alignment=%d\n" alignment (data_start % alignment);
    List.iter tensors ~f:(fun (id, offset) ->
        Stdio.printf "  id=%d offset=%d, offset mod alignment=%d\n" id offset (offset % alignment))
  in
  Tensor.unsafe_reinitialize ();
  let ctx = Context.cpu () in
  (* Byte lengths (12, 5, 4) that would leave every subsequent payload misaligned if packed. *)
  let ctx, tn_a = make_tn ctx ~id:0 ~label:[ "a" ] Ops.single [| 3 |] [| 1.0; 2.0; 3.0 |] in
  let ctx, tn_b = make_tn ctx ~id:1 ~label:[ "b" ] Ops.byte [| 5 |] [| 1.0; 2.0; 3.0; 4.0; 5.0 |] in
  let ctx, tn_c = make_tn ctx ~id:2 ~label:[ "c" ] Ops.single [| 1 |] [| 7.0 |] in
  let t_set = Set.of_list (module Tn) [ tn_a; tn_b; tn_c ] in
  let path = tmp_file "aligned" in
  Persistence.save ~ctx ~appending:false t_set path;
  report path;
  (* Appending re-lays out the file; the alignment invariant must survive it. *)
  Persistence.save ~ctx ~appending:true (Set.of_list (module Tn) [ tn_b ]) path;
  report path;
  Tensor.unsafe_reinitialize ();
  let ctx = Context.cpu () in
  let ctx, loaded = Persistence.load ~ctx path in
  Set.iter loaded ~f:(fun tn -> Stdio.printf "  id=%d values=[%s]\n" tn.Tn.id (show ctx tn));
  cleanup "aligned";

  (* An explicit alignment of 1 reproduces the pre-alignment layout: payloads packed back to back,
     which is how checkpoints written before the field are read. *)
  Stdio.printf "=== Test 12: Unaligned (legacy) layout ===\n";
  Tensor.unsafe_reinitialize ();
  let ctx = Context.cpu () in
  let ctx, tn_a = make_tn ctx ~id:0 ~label:[ "a" ] Ops.single [| 3 |] [| 1.0; 2.0; 3.0 |] in
  let ctx, tn_b = make_tn ctx ~id:1 ~label:[ "b" ] Ops.byte [| 5 |] [| 1.0; 2.0; 3.0; 4.0; 5.0 |] in
  let ctx, tn_c = make_tn ctx ~id:2 ~label:[ "c" ] Ops.single [| 1 |] [| 7.0 |] in
  let path = tmp_file "packed" in
  Persistence.save ~ctx ~appending:false ~alignment:1
    (Set.of_list (module Tn) [ tn_a; tn_b; tn_c ])
    path;
  report path;
  Tensor.unsafe_reinitialize ();
  let ctx = Context.cpu () in
  let mapped_before, decoded_before = Nd.ingestion_counts () in
  let ctx, loaded = Persistence.load ~ctx ~mmap:true path in
  let mapped_after, decoded_after = Nd.ingestion_counts () in
  Set.iter loaded ~f:(fun tn -> Stdio.printf "  id=%d values=[%s]\n" tn.Tn.id (show ctx tn));
  (* Packed, nothing keeps a payload on its element's boundary: the header is not padded out, and
     each offset is just the sum of the preceding payloads. A mapping at such an offset would hand
     out a misaligned float pointer, so those payloads are decoded even with mapping on. The
     expectation is computed from the layout rather than written down, so it stays right if the
     header's length changes. *)
  let _, data_start, layout = file_layout path in
  let prec_of_id id = if id = 1 then Ops.byte else Ops.single in
  let expected_mapped =
    List.count layout ~f:(fun (id, offset) ->
        (data_start + offset) % Ops.prec_in_bytes (prec_of_id id) = 0)
  in
  Stdio.printf "  packed load: %d mapped, %d decoded\n" (mapped_after - mapped_before)
    (decoded_after - decoded_before);
  Verdict.p "a payload at an offset its precision cannot be mapped at is decoded"
    (mapped_after - mapped_before = expected_mapped
    && decoded_after - decoded_before = List.length layout - expected_mapped);
  Verdict.p "the packed layout really does leave a payload unmappable"
    (expected_mapped < List.length layout);
  (* And a checkpoint written before the field existed -- the same layout with the field deleted
     from the header -- must still read back. *)
  let legacy_path = tmp_file "legacy" in
  let ic = Stdlib.open_in_bin path in
  let header_len = Stdlib.input_binary_int ic in
  let buf = Bytes.create header_len in
  Stdlib.really_input ic buf 0 header_len;
  let data = Stdio.In_channel.input_all ic in
  Stdlib.close_in ic;
  let legacy_header =
    String.substr_replace_first
      (String.strip (Bytes.to_string buf))
      ~pattern:"(alignment 1)" ~with_:""
  in
  let oc = Stdlib.open_out_bin legacy_path in
  Stdlib.output_binary_int oc (String.length legacy_header);
  Stdlib.output_string oc legacy_header;
  Stdlib.output_string oc data;
  Stdlib.close_out oc;
  Tensor.unsafe_reinitialize ();
  let ctx = Context.cpu () in
  let ctx, loaded = Persistence.load ~ctx legacy_path in
  Stdio.printf "  no-alignment-field checkpoint:\n";
  Set.iter loaded ~f:(fun tn -> Stdio.printf "  id=%d values=[%s]\n" tn.Tn.id (show ctx tn));
  cleanup "legacy";
  cleanup "packed";

  (* === Test 13: Mapped load matches the decoding load (gh-ocannl-467) === *)
  Stdio.printf "=== Test 13: Mapped vs decoded payloads ===\n";
  (* The mapping reinterprets the payload bytes as the host buffer, so precisions whose in-memory
     representation is not the payload's would silently decode to garbage: check them all. The
     padded node exercises the fallback -- its payload holds only the logical region. *)
  let precisions = List.map Ops.scalar_precs ~f:(fun prec -> (Ops.prec_string prec, prec)) in
  let values = [| 1.0; 2.0; 32.0; 5.0 |] in
  let padding = Some ([| Ops.{ left = 1; right = 1 } |], 0.0) in
  let path = tmp_file "mapped" in
  Tensor.unsafe_reinitialize ();
  let ctx = Context.cpu () in
  let ctx, tns =
    List.foldi precisions ~init:(ctx, []) ~f:(fun i (ctx, acc) (_name, prec) ->
        let ctx, tn = make_tn ctx ~id:i ~label:[ "p" ] prec [| 4 |] values in
        (ctx, tn :: acc))
  in
  let tns = List.rev tns in
  let ctx, tn_padded =
    make_tn ctx ~id:(List.length precisions) ~label:[ "padded" ] ~padding Ops.single [| 6 |]
      [| 3.0; 4.0; 5.0; 6.0 |]
  in
  Persistence.save ~ctx ~appending:false (Set.of_list (module Tn) (tn_padded :: tns)) path;
  let load_and_show ~mmap =
    Tensor.unsafe_reinitialize ();
    let ctx = Context.cpu () in
    let mapped_before, decoded_before = Nd.ingestion_counts () in
    let ctx, loaded = Persistence.load ~ctx ~mmap path in
    let mapped_after, decoded_after = Nd.ingestion_counts () in
    Stdio.printf "  load ~mmap:%b: %d mapped, %d decoded\n" mmap (mapped_after - mapped_before)
      (decoded_after - decoded_before);
    List.map (Set.to_list loaded) ~f:(fun tn -> (tn.Tn.id, show ctx tn))
  in
  let decoded = load_and_show ~mmap:false in
  let mapped = load_and_show ~mmap:true in
  List.iter2_exn decoded mapped ~f:(fun (id, decoded) (_, mapped) ->
      let name =
        if id < List.length precisions then fst (List.nth_exn precisions id) else "padded/single"
      in
      let identical = String.equal decoded mapped in
      Stdio.printf "  %s: %s (mapped %s)\n" name decoded
        (if identical then "identical" else "DIFFERS: " ^ mapped);
      (* The row renders the comparison itself, so the claim sits beside it on the same boolean:
         "DIFFERS: ..." exits 0 and is promotable on its own (gh-ocannl-601). *)
      Verdict.claimf "%s: mapped payload identical to the decoded one" name identical);
  (* Restore takes the same path, into already-existing device buffers. *)
  Tensor.unsafe_reinitialize ();
  let ctx = Context.cpu () in
  let single_id, _ =
    List.findi precisions ~f:(fun _ (_, prec) -> Ops.equal_prec prec Ops.single) |> Option.value_exn
  in
  let ctx, tn =
    make_tn ctx ~id:single_id ~label:[ "p" ] Ops.single [| 4 |] [| 0.0; 0.0; 0.0; 0.0 |]
  in
  let t_set = Set.of_list (module Tn) [ tn ] in
  let mapped_before, _ = Nd.ingestion_counts () in
  let ctx = Persistence.restore ~ctx ~mmap:true t_set path in
  let mapped_after, _ = Nd.ingestion_counts () in
  Stdio.printf "  restored with mapping (%d mapped): [%s]\n" (mapped_after - mapped_before)
    (show ctx tn);
  cleanup "mapped";

  (* === Test 14: Truncated file === *)
  Stdio.printf "=== Test 14: Truncated checkpoint ===\n";
  Tensor.unsafe_reinitialize ();
  let ctx = Context.cpu () in
  let ctx, tn = make_tn ctx ~id:0 ~label:[ "t" ] Ops.single [| 4 |] values in
  let path = tmp_file "truncated" in
  Persistence.save ~ctx ~appending:false (Set.of_list (module Tn) [ tn ]) path;
  let full = Stdio.In_channel.read_all path in
  let oc = Stdlib.open_out_bin path in
  Stdlib.output_string oc (String.drop_suffix full 4);
  Stdlib.close_out oc;
  Tensor.unsafe_reinitialize ();
  let ctx = Context.cpu () in
  (try
     let _ = Persistence.load ~ctx ~mmap:true path in
     Verdict.fail "should have raised"
   with Failure msg ->
     (* The message embeds the machine-dependent path. *)
     Verdict.p "Caught a payload-past-end-of-file failure"
       (String.is_suffix msg ~suffix:"extends past the end of the file"));
  cleanup "truncated"

(* === Test 15: Saving over a checkpoint whose mappings are still live (gh-ocannl-588) === *)

(* [save] writes a temp file and renames it over the target. On POSIX that replaces the directory
   entry and leaves the inode the mapping was taken from alone, so a live mapping keeps reading what
   it read before. Windows was expected not to allow it at all -- a mapped view keeps the file
   object referenced, so the replacing rename should fail with a sharing violation -- which is why
   [checkpoint_load_mmap] defaulted off there. This measures both halves: whether the save succeeds,
   and, since a successful rename could equally mean the mapping now sees the new bytes, whether the
   mapping still reads the values it was taken from. The platform-specific detail goes to stderr, so
   the golden says the same thing everywhere. *)
let live_mapping_path = tmp_file "live_mapping"
let v1 = [| 1.0; 2.0; 3.0; 4.0 |]
let v2 = [| 9.0; 8.0; 7.0; 6.0 |]

let attempt what f =
  match f () with
  | () ->
      Stdio.eprintf "  %s: succeeded\n" what;
      true
  | exception exn ->
      Stdio.eprintf "  %s: raised %s\n" what (Exn.to_string exn);
      false

let () =
  Stdio.printf "=== Test 15: Save over a live mapped checkpoint ===\n";
  let path = live_mapping_path in
  Tensor.unsafe_reinitialize ();
  let ctx = Context.cpu () in
  let ctx, tn = make_tn ctx ~id:0 ~label:[ "w" ] Ops.single [| 4 |] v1 in
  Persistence.save ~ctx ~appending:false (Set.of_list (module Tn) [ tn ]) path;
  Tensor.unsafe_reinitialize ();
  let ctx = Context.cpu () in
  let mapped_before, _ = Nd.ingestion_counts () in
  let ctx, loaded = Persistence.load ~ctx ~mmap:true path in
  let mapped_after, _ = Nd.ingestion_counts () in
  Verdict.p "the reloaded payload is mapped, whatever the platform default"
    (mapped_after - mapped_before = 1);
  let tn = List.hd_exn (Set.to_list loaded) in
  (* The mapping itself, held in a local across the save below -- reading it afterwards is what
     makes this an experiment about a *live* mapping, not one the GC may already have dropped. *)
  let mapping = Lazy.force (Option.value_exn (Ir.Host_inits.find tn)) in
  Verdict.p "the mapping reads the checkpoint's values"
    (Array.equal Float.equal (Nd.retrieve_flat_values mapping) v1);
  (* A save killed between streaming and its commit leaves a staging file the size of the model, and
     no cleanup of its own can run. Stand one in, aged past the sweep's threshold, and let the save
     below be the event that reclaims it (Codex P2, round 1). A second one, belonging to a different
     checkpoint in the same directory, is what makes the scope claim: this save is not licensed to
     delete another publication's artifact. *)
  let plant name =
    let planted = Stdlib.Filename.concat (Stdlib.Filename.dirname path) name in
    Stdio.Out_channel.write_all planted ~data:"abandoned checkpoint";
    let stamp = Unix.time () -. (2. *. Utils.Atomic_file.default_max_age_seconds) in
    Unix.utimes planted stamp stamp;
    planted
  in
  (* A name the publication helper itself would have produced: stem, infix, pid, counter, nonce. The
     claim below is what keeps this test honest if that shape ever changes -- an unrecognized plant
     would otherwise make the sweep claim pass by sweeping nothing. *)
  let staged_name target =
    Printf.sprintf "%s%s%08x.%08x.%s" target Utils.Atomic_file.staging_infix 4242 0
      "00c0ffee00c0ffee"
  in
  let abandoned = plant (staged_name (Stdlib.Filename.basename path)) in
  let other = plant (staged_name "someone_elses.safetensors") in
  Verdict.p "the planted staging name is recognized as this checkpoint's"
    (Utils.Atomic_file.is_staging_file_for ~path (Stdlib.Filename.basename abandoned));
  (* Save different values over the same path. [set_values] goes through a fresh host buffer, so it
     does not disturb the mapping. *)
  let ctx = Context.set_values ctx tn v2 in
  let saved =
    attempt "save over a live mapping" (fun () ->
        Persistence.save ~ctx ~appending:false loaded path)
  in
  Verdict.p "saving over a checkpoint with a live mapping succeeds" saved;
  Verdict.p "a save reclaims this checkpoint's abandoned staging file"
    (not (Stdlib.Sys.file_exists abandoned));
  Verdict.p "a save leaves another checkpoint's staging file alone" (Stdlib.Sys.file_exists other);
  (try Stdlib.Sys.remove other with _ -> ());
  Verdict.p "the live mapping still reads the values it was taken from"
    (Array.equal Float.equal (Nd.retrieve_flat_values mapping) v1)

(* A fresh top-level block, so the previous one's mapping, context and nodes are out of scope. *)
let () =
  Tensor.unsafe_reinitialize ();
  Stdlib.Gc.full_major ();
  let ctx = Context.cpu () in
  let ctx, reloaded = Persistence.load ~ctx ~mmap:true live_mapping_path in
  let tn = List.hd_exn (Set.to_list reloaded) in
  Verdict.p "the file on disk holds what the save wrote"
    (Array.equal Float.equal (Context.get_values ctx tn) v2);
  (* [Persistence.save] publishes through [Atomic_file], which removes its own staging file on every
     failing path, so only the checkpoint itself is left to clean up. Looking for one is what proves
     that. The scan is narrowed to staging files of THIS checkpoint: the directory is the shared
     system temp directory, where another process's in-flight publication is none of this test's
     business — neither to fail on nor to delete. *)
  let dir = Stdlib.Filename.dirname live_mapping_path in
  let abandoned =
    Array.to_list (Stdlib.Sys.readdir dir)
    |> List.filter ~f:(Utils.Atomic_file.is_staging_file_for ~path:live_mapping_path)
  in
  Verdict.p_empty "no checkpoint save left a staging file behind" ~over:[ live_mapping_path ]
    abandoned;
  List.iter
    (live_mapping_path :: List.map abandoned ~f:(Stdlib.Filename.concat dir))
    ~f:(fun path -> if Stdlib.Sys.file_exists path then Stdlib.Sys.remove path);

  Stdio.printf "=== All persistence tests completed ===\n"
