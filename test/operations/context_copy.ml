(* Context.copy (docs/proposals/backend-singletons-context-copy.md): same-backend copies pair-match
   Backends.wrapped_context to dispatch onto the backend's device_to_device transfer machinery;
   cross-backend copies fall back to a host round-trip. Backends are pinned explicitly (cc /
   multidev_cc, always available) so the output is deterministic regardless of the configured
   default backend.

   Scenarios:
   1. On-device copy into a context lacking the node (the init_from_device path: allocate + copy).
   2. On-device copy when both contexts hold the node (the transfer-routine path), overwriting the
      destination's values.
   3. Fallback for a node with no device buffer in the source: host-init literal data reaches the
      destination via the host round-trip.
   4. Cross-backend (cc -> multidev_cc) host round-trip.
   5. ~into_merge_buffer:Copy plus a %cd consumer of [t.merge]: the copy returns a context carrying
      the merge-buffer node, against which compiling the consumer statically verifies
      (gh-ocannl-288); compiling the same consumer against a context with no prior transfer fails
      the static check. *)

open Base
open Stdio
open Ocannl
open Operation.DSL_modules
module IDX = Train.IDX
module Tn = Ir.Tnode

let show label ctx tn =
  printf "%s:" label;
  Array.iter (Context.get_values ctx tn) ~f:(fun v -> printf " %.1f" v);
  printf "\n"

let scenario_on_device () =
  printf "=== 1+2. On-device copy: fresh destination, then overwrite ===\n";
  Tensor.unsafe_reinitialize ();
  let ctx = Context.cpu () in
  let%op t = [ 1.; 2.; 3. ] + [ 10.; 20.; 30. ] in
  let src = Train.forward_once ctx t in
  let dst = Context.cpu () in
  printf "destination has the node before copy: %b\n" (Context.mem dst t.Tensor.value);
  let dst = Context.copy ~src ~dst t.Tensor.value in
  printf "destination has the node after copy: %b\n" (Context.mem dst t.Tensor.value);
  show "copied (expect 11 22 33)" dst t.Tensor.value;
  (* Both contexts hold the node now: the second copy goes through the transfer routine. *)
  let src = Context.set_values src t.Tensor.value [| 100.; 200.; 300. |] in
  let dst = Context.copy ~src ~dst t.Tensor.value in
  show "re-copied (expect 100 200 300)" dst t.Tensor.value

let scenario_host_init_fallback () =
  printf "=== 3. Source-absent node: host-init data via the round-trip fallback ===\n";
  Tensor.unsafe_reinitialize ();
  let src = Context.cpu () in
  let dst = Context.cpu () in
  let nd =
    Ir.Ndarray.init_array ~debug:"c" Ir.Ops.single ~dims:[| 2 |] ~padding:None ~f:(fun idx ->
        5. +. Float.of_int idx.(0))
  in
  let c = TDSL.wrap ~l:"c" nd () in
  (* [c] was never computed in [src]; its value lives in host-init data only. *)
  printf "source has the node: %b\n" (Context.mem src c.Tensor.value);
  let dst = Context.copy ~src ~dst c.Tensor.value in
  show "copied (expect 5 6)" dst c.Tensor.value

let scenario_cross_backend () =
  printf "=== 4. Cross-backend copy (cc -> multidev_cc) ===\n";
  Tensor.unsafe_reinitialize ();
  let ctx = Context.cpu () in
  (* 0.4 rather than 0.25: 2.25 is a representable %.1f tie, which glibc rounds to even ("2.2")
     but the Windows CRT rounds away from zero ("2.3"), making the output platform-dependent. *)
  let%op t = [ 1.; 2. ] + [ 0.5; 0.4 ] in
  let src = Train.forward_once ctx t in
  let dst = Context.cpu ~threads:4 () in
  let dst = Context.copy ~src ~dst t.Tensor.value in
  show "copied (expect 1.5 2.4)" dst t.Tensor.value;
  printf "backends: %s -> %s\n" (Context.backend_name src) (Context.backend_name dst)

let scenario_merge_buffer () =
  printf "=== 5. Merge-buffer copy + statically verified consumer ===\n";
  Tensor.unsafe_reinitialize ();
  let%op t = [ 1.; 2.; 3. ] + [ 10.; 20.; 30. ] in
  let fwd = Train.forward t in
  let ctx_a = Train.to_routine (Context.cpu ()) IDX.empty fwd in
  let ctx_a = Context.run (Context.context ctx_a) ctx_a in
  let routine_b = Train.to_routine (Context.cpu ()) IDX.empty fwd in
  let ctx_b = Context.run (Context.context routine_b) routine_b in
  let ctx_a = Context.set_values ctx_a t.Tensor.value [| 1.; 1.; 1. |] in
  (* Consuming the merge buffer without a prior transfer must fail the static check. *)
  let consumer = [%cd t =+ t.merge] in
  let consumer = { consumer with asgns = Ir.Assignments.Block_comment ("merge_consume", consumer.asgns) } in
  (try
     let (_ : Context.t * Context.routine) = Context.compile ctx_b consumer IDX.empty in
     printf "UNEXPECTED: consumer compiled without a merge-buffer transfer\n"
   with Utils.User_error msg -> printf "consumer against a transfer-less context: %s\n" msg);
  (* Copy A's values into B's merge buffer, then consume: t_b += t_a.merge. *)
  let ctx_b = Context.copy ~into_merge_buffer:Copy ~src:ctx_a ~dst:ctx_b t.Tensor.value in
  let ctx_b, consume = Context.compile ctx_b consumer IDX.empty in
  let ctx_b = Context.run ctx_b consume in
  show "b after b += a.merge (expect 12 23 34)" ctx_b t.Tensor.value

let () =
  scenario_on_device ();
  scenario_host_init_fallback ();
  scenario_cross_backend ();
  scenario_merge_buffer ()
