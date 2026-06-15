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
    (Array.to_list (Array.map (Context.get_values ctx tn) ~f:(fun v -> Stdlib.Printf.sprintf "%.1f" v)))

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
        (String.concat ~sep:"." tn.Tn.label) (show ctx tn));
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
     Stdio.printf "ERROR: should have raised\n"
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
     Stdio.printf "ERROR: should have raised\n"
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
     Stdio.printf "ERROR: should have raised\n"
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
  let padding1 = Some ([| Ops.{ left = 1; right = 1 } |], None) in
  let ctx, tn_padded =
    make_tn ctx ~id:0 ~label:[ "p" ] ~padding:padding1 Ops.single [| 4 |] [| 1.0; 2.0 |]
  in
  ignore tn_padded;
  let path = tmp_file "padmismatch" in
  Persistence.save ~ctx ~appending:false (Set.of_list (module Tn) [ tn_padded ]) path;
  (* Create a tnode with same dims but different padding *)
  Tensor.unsafe_reinitialize ();
  let ctx = Context.cpu () in
  let padding2 = Some ([| Ops.{ left = 0; right = 2 } |], None) in
  let ctx, tn_diff_pad =
    make_tn ctx ~id:0 ~label:[ "p" ] ~padding:padding2 Ops.single [| 4 |] [| 1.0; 2.0 |]
  in
  (try
     let _ctx = Persistence.restore ~ctx (Set.of_list (module Tn) [ tn_diff_pad ]) path in
     Stdio.printf "ERROR: should have raised\n"
   with Failure msg -> Stdio.printf "Caught: %s\n" msg);
  cleanup "padmismatch";

  Stdio.printf "=== All persistence tests completed ===\n"
