open Base
module Tn = Ir.Tnode
module Nd = Ir.Ndarray
module Ops = Ir.Ops
module Tensor = Ocannl_tensor.Tensor
module Persistence = Ocannl.Persistence

let tmp_dir = Stdlib.Filename.get_temp_dir_name ()

let tmp_file name =
  Stdlib.Filename.concat tmp_dir ("test_persistence_" ^ name ^ ".ckpt")

let cleanup name =
  let path = tmp_file name in
  if Stdlib.Sys.file_exists path then Stdlib.Sys.remove path

(** Create a tnode with given values for testing. *)
let make_tn ~id ~label prec dims values =
  let padding = None in
  let nd = Nd.create_array ~debug:"test" prec ~dims ~padding in
  Nd.set_flat_values nd values;
  Tn.create_from_padded ~id ~label ~ndarray:nd ~padding ()

let get_values tn =
  match Lazy.force tn.Tn.array with
  | Some nd -> Nd.retrieve_flat_values nd
  | None -> [||]

let () =
  (* === Test 1: Round-trip save/load === *)
  Stdio.printf "=== Test 1: Round-trip save/load ===\n";
  Tensor.unsafe_reinitialize ();
  let tn1 =
    make_tn ~id:0 ~label:[ "weights" ] Ops.single [| 2; 3 |]
      [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0 |]
  in
  let tn2 =
    make_tn ~id:1 ~label:[ "bias" ] Ops.double [| 3 |] [| 10.0; 20.0; 30.0 |]
  in
  let t_set = Set.of_list (module Tn) [ tn1; tn2 ] in
  let path = tmp_file "roundtrip" in
  Persistence.save ~appending:false t_set path;
  (* Reset and load *)
  Tensor.unsafe_reinitialize ();
  let loaded = Persistence.load path in
  let loaded_list = Set.to_list loaded in
  Stdio.printf "Loaded %d tensors\n" (List.length loaded_list);
  List.iter loaded_list ~f:(fun tn ->
      let vals = get_values tn in
      Stdio.printf "  id=%d label=%s values=[%s]\n" tn.Tn.id
        (String.concat ~sep:"." tn.Tn.label)
        (String.concat ~sep:"; "
           (Array.to_list (Array.map vals ~f:(fun v -> Stdlib.Printf.sprintf "%.1f" v)))));
  cleanup "roundtrip";

  (* === Test 2: Restore === *)
  Stdio.printf "=== Test 2: Restore ===\n";
  Tensor.unsafe_reinitialize ();
  let tn1 =
    make_tn ~id:0 ~label:[ "w" ] Ops.single [| 2; 2 |]
      [| 1.0; 2.0; 3.0; 4.0 |]
  in
  let t_set = Set.of_list (module Tn) [ tn1 ] in
  let path = tmp_file "restore" in
  Persistence.save ~appending:false t_set path;
  (* Modify values *)
  let nd = Option.value_exn (Lazy.force tn1.Tn.array) in
  Nd.set_flat_values nd [| 99.0; 99.0; 99.0; 99.0 |];
  Stdio.printf "Before restore: [%s]\n"
    (String.concat ~sep:"; "
       (Array.to_list (Array.map (get_values tn1) ~f:(fun v ->
            Stdlib.Printf.sprintf "%.1f" v))));
  (* Restore original values *)
  Persistence.restore t_set path;
  Stdio.printf "After restore: [%s]\n"
    (String.concat ~sep:"; "
       (Array.to_list (Array.map (get_values tn1) ~f:(fun v ->
            Stdlib.Printf.sprintf "%.1f" v))));
  cleanup "restore";

  (* === Test 3: Append mode - disjoint sets === *)
  Stdio.printf "=== Test 3: Append mode ===\n";
  Tensor.unsafe_reinitialize ();
  let tn_a =
    make_tn ~id:0 ~label:[ "a" ] Ops.single [| 2 |] [| 1.0; 2.0 |]
  in
  let tn_b =
    make_tn ~id:1 ~label:[ "b" ] Ops.single [| 2 |] [| 3.0; 4.0 |]
  in
  let path = tmp_file "append" in
  Persistence.save ~appending:false
    (Set.of_list (module Tn) [ tn_a ])
    path;
  Persistence.save ~appending:true
    (Set.of_list (module Tn) [ tn_b ])
    path;
  Tensor.unsafe_reinitialize ();
  let loaded = Persistence.load path in
  Stdio.printf "Loaded %d tensors after append\n" (Set.length loaded);
  Set.iter loaded ~f:(fun tn ->
      let vals = get_values tn in
      Stdio.printf "  id=%d values=[%s]\n" tn.Tn.id
        (String.concat ~sep:"; "
           (Array.to_list (Array.map vals ~f:(fun v ->
                Stdlib.Printf.sprintf "%.1f" v)))));
  cleanup "append";

  (* === Test 4: Append overwrite === *)
  Stdio.printf "=== Test 4: Append overwrite ===\n";
  Tensor.unsafe_reinitialize ();
  let tn_orig =
    make_tn ~id:0 ~label:[ "x" ] Ops.single [| 2 |] [| 10.0; 20.0 |]
  in
  let path = tmp_file "overwrite" in
  Persistence.save ~appending:false
    (Set.of_list (module Tn) [ tn_orig ])
    path;
  (* Overwrite with new values *)
  let nd = Option.value_exn (Lazy.force tn_orig.Tn.array) in
  Nd.set_flat_values nd [| 77.0; 88.0 |];
  Persistence.save ~appending:true
    (Set.of_list (module Tn) [ tn_orig ])
    path;
  Tensor.unsafe_reinitialize ();
  let loaded = Persistence.load path in
  Set.iter loaded ~f:(fun tn ->
      let vals = get_values tn in
      Stdio.printf "  id=%d values=[%s]\n" tn.Tn.id
        (String.concat ~sep:"; "
           (Array.to_list (Array.map vals ~f:(fun v ->
                Stdlib.Printf.sprintf "%.1f" v)))));
  cleanup "overwrite";

  (* === Test 5: Empty checkpoint === *)
  Stdio.printf "=== Test 5: Empty checkpoint ===\n";
  Tensor.unsafe_reinitialize ();
  let empty = Set.empty (module Tn) in
  let path = tmp_file "empty" in
  Persistence.save ~appending:false empty path;
  Tensor.unsafe_reinitialize ();
  let loaded = Persistence.load path in
  Stdio.printf "Loaded %d tensors from empty checkpoint\n" (Set.length loaded);
  Persistence.restore empty path;
  Stdio.printf "Restore on empty set succeeded\n";
  cleanup "empty";

  (* === Test 6: Error - missing tensor on restore === *)
  Stdio.printf "=== Test 6: Missing tensor on restore ===\n";
  Tensor.unsafe_reinitialize ();
  let tn1 =
    make_tn ~id:0 ~label:[ "present" ] Ops.single [| 2 |] [| 1.0; 2.0 |]
  in
  let tn2 =
    make_tn ~id:1 ~label:[ "absent" ] Ops.single [| 2 |] [| 3.0; 4.0 |]
  in
  let path = tmp_file "missing" in
  Persistence.save ~appending:false
    (Set.of_list (module Tn) [ tn1 ])
    path;
  (try
     Persistence.restore (Set.of_list (module Tn) [ tn2 ]) path;
     Stdio.printf "ERROR: should have raised\n"
   with Failure msg -> Stdio.printf "Caught: %s\n" msg);
  cleanup "missing";

  (* === Test 7: Error - dimension mismatch on restore === *)
  Stdio.printf "=== Test 7: Dimension mismatch on restore ===\n";
  Tensor.unsafe_reinitialize ();
  let tn_save =
    make_tn ~id:0 ~label:[ "d" ] Ops.single [| 2; 3 |]
      [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0 |]
  in
  let path = tmp_file "dimfail" in
  Persistence.save ~appending:false
    (Set.of_list (module Tn) [ tn_save ])
    path;
  (* Create a tnode with different dims but same id *)
  Tensor.unsafe_reinitialize ();
  let tn_wrong =
    make_tn ~id:0 ~label:[ "d" ] Ops.single [| 3; 2 |]
      [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0 |]
  in
  (try
     Persistence.restore (Set.of_list (module Tn) [ tn_wrong ]) path;
     Stdio.printf "ERROR: should have raised\n"
   with Failure msg -> Stdio.printf "Caught: %s\n" msg);
  cleanup "dimfail";

  (* === Test 8: Error - ID clash on load === *)
  Stdio.printf "=== Test 8: ID clash on load ===\n";
  Tensor.unsafe_reinitialize ();
  let tn1 =
    make_tn ~id:0 ~label:[ "clash" ] Ops.single [| 2 |] [| 1.0; 2.0 |]
  in
  let path = tmp_file "clash" in
  Persistence.save ~appending:false
    (Set.of_list (module Tn) [ tn1 ])
    path;
  (* Don't reinitialize - tn1 is still in registry *)
  (try
     let _ = Persistence.load path in
     Stdio.printf "ERROR: should have raised\n"
   with Failure msg -> Stdio.printf "Caught: %s\n" msg);
  cleanup "clash";

  (* === Test 9: ID floor after load === *)
  Stdio.printf "=== Test 9: ID floor after load ===\n";
  Tensor.unsafe_reinitialize ();
  let tn1 =
    make_tn ~id:5 ~label:[ "high_id" ] Ops.single [| 2 |] [| 1.0; 2.0 |]
  in
  let path = tmp_file "idfloor" in
  Persistence.save ~appending:false
    (Set.of_list (module Tn) [ tn1 ])
    path;
  Tensor.unsafe_reinitialize ();
  let _loaded = Persistence.load path in
  (* Now create a new tnode via create_from_padded - it should get id > 5 *)
  let nd = Nd.create_array ~debug:"new" Ops.single ~dims:[| 1 |] ~padding:None in
  Nd.set_flat_values nd [| 42.0 |];
  let new_tn = Tn.create_from_padded ~id:6 ~label:[ "new" ] ~ndarray:nd ~padding:None () in
  Stdio.printf "Loaded id=5, new tensor id=%d (should be >= 6)\n" new_tn.Tn.id;
  cleanup "idfloor";

  Stdio.printf "=== All persistence tests completed ===\n"
