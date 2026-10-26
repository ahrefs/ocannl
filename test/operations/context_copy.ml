(* Context.copy (docs/proposals/backend-singletons-context-copy.md): same-backend copies pair-match
   Backends.wrapped_context to dispatch onto the backend's device_to_device transfer machinery;
   cross-backend copies fall back to a host round-trip. Backends are pinned explicitly (cc /
   multidev_cc, always available) so the output is deterministic regardless of the configured
   default backend.

   Scenarios: 1. On-device copy into a context lacking the node (the init_from_device path: allocate
   + copy). 2. On-device copy when both contexts hold the node (the transfer-routine path),
   overwriting the destination's values. 3. Fallback for a node with no device buffer in the source:
   host-init literal data reaches the destination via the host round-trip. 4. Cross-backend (cc ->
   multidev_cc) host round-trip. 5. ~into_merge_buffer:Copy plus a %cd consumer of [t.merge]: the
   copy returns a context carrying the merge-buffer node, against which compiling the consumer
   statically verifies (gh-ocannl-288); compiling the same consumer against a context with no prior
   transfer fails the static check. 6. (gh-ocannl-1623, on the SELECTED backend) repeated copies --
   into an existing buffer, into a fresh destination followed by overwrites, and into the merge
   buffer -- keep the destination's lifecycle depth constant, and a copy's result is its
   destination's release unit, in both directions. *)

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
  Verdict.p "destination lacks the node before copy" (not (Context.mem dst t.Tensor.value));
  let dst = Context.copy ~src ~dst t.Tensor.value in
  Verdict.p "destination has the node after copy" (Context.mem dst t.Tensor.value);
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
  Verdict.p "source lacks the node (host-init data only)" (not (Context.mem src c.Tensor.value));
  let dst = Context.copy ~src ~dst c.Tensor.value in
  show "copied (expect 5 6)" dst c.Tensor.value

let scenario_cross_backend () =
  printf "=== 4. Cross-backend copy (cc -> multidev_cc) ===\n";
  Tensor.unsafe_reinitialize ();
  let ctx = Context.cpu () in
  (* 0.4 rather than 0.25: 2.25 is a representable %.1f tie, which glibc rounds to even ("2.2") but
     the Windows CRT rounds away from zero ("2.3"), making the output platform-dependent. *)
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
  let compiled_a, routine_a = Train.to_routine (Context.cpu ()) IDX.empty fwd in
  let ctx_a = Context.run compiled_a routine_a in
  let compiled_b, routine_b = Train.to_routine (Context.cpu ()) IDX.empty fwd in
  let ctx_b = Context.run compiled_b routine_b in
  let ctx_a = Context.set_values ctx_a t.Tensor.value [| 1.; 1.; 1. |] in
  (* Consuming the merge buffer without a prior transfer must fail the static check. *)
  let consumer = [%cd t =+ t.merge] in
  let consumer =
    { consumer with asgns = Ir.Assignments.Block_comment ("merge_consume", consumer.asgns) }
  in
  (try
     let (_ : Context.t * Context.routine) = Context.compile ctx_b consumer IDX.empty in
     printf "UNEXPECTED: consumer compiled without a merge-buffer transfer\n"
   with Utils.User_error msg -> printf "consumer against a transfer-less context: %s\n" msg);
  (* Copy A's values into B's merge buffer, then consume: t_b += t_a.merge. *)
  let ctx_b = Context.copy ~into_merge_buffer:Copy ~src:ctx_a ~dst:ctx_b t.Tensor.value in
  let ctx_b, consume = Context.compile ctx_b consumer IDX.empty in
  let ctx_b = Context.run ctx_b consume in
  show "b after b += a.merge (expect 12 23 34)" ctx_b t.Tensor.value

let scenario_copy_chain () =
  printf "=== 6. Repeated copies keep the lifecycle chain bounded (gh-ocannl-1623) ===\n";
  Tensor.unsafe_reinitialize ();
  let%op t = [ 1.; 2.; 3. ] + [ 10.; 20.; 30. ] in
  let fwd = Train.forward t in
  let root () =
    let compiled, routine = Train.to_routine (Context.auto ()) IDX.empty fwd in
    Context.run compiled routine
  in
  let tn = t.Tensor.value in
  let iterations = 1000 in
  let values i = [| Float.of_int i; Float.of_int (i + 1); Float.of_int (i + 2) |] in
  (* Copy [src] into [dst] [iterations] times, changing the source every 250 copies and reading the
     destination back after each such copy; returns the final destination, every sampled depth and
     whether every read-back matched. *)
  let copy_loop ?into_merge_buffer ~src ~dst () =
    let src = ref src and dst = ref dst and depths = ref [] and matched = ref true in
    for i = 1 to iterations do
      let changing = i % 250 = 0 in
      if changing then src := Context.set_values !src tn (values i);
      dst := Context.copy ?into_merge_buffer ~src:!src ~dst:!dst tn;
      depths := Context.lifecycle_depth !dst :: !depths;
      if changing && Option.is_none into_merge_buffer then
        matched := !matched && Poly.equal (Context.get_values !dst tn) (values i)
    done;
    (!src, !dst, !depths, !matched)
  in
  let ctx_a = root () and ctx_b = root () in
  let depth_b = Context.lifecycle_depth ctx_b in
  Verdict.p "a compiled destination has a lifecycle ancestor" (depth_b >= 1);
  let ctx_a, ctx_b, depths, matched = copy_loop ~src:ctx_a ~dst:ctx_b () in
  Verdict.p_all "copies into an existing buffer: every sampled depth is the initial one" depths
    ~f:(fun d -> d = depth_b);
  Verdict.p "copies into an existing buffer: each changed value reached the destination" matched;
  let ctx_c = Context.auto () in
  Verdict.p "a fresh root is at depth 0" (Context.lifecycle_depth ctx_c = 0);
  let ctx_c = Context.copy ~src:ctx_a ~dst:ctx_c tn in
  let ctx_a, ctx_c, depths, matched = copy_loop ~src:ctx_a ~dst:ctx_c () in
  Verdict.p_all "a fresh destination, then overwrites: every sampled depth is 0" (0 :: depths)
    ~f:(fun d -> d = 0);
  Verdict.p "a fresh destination, then overwrites: each changed value reached it" matched;
  let ctx_a, ctx_b, depths, _ = copy_loop ~into_merge_buffer:Copy ~src:ctx_a ~dst:ctx_b () in
  Verdict.p_all "merge-buffer fills: every sampled depth is the initial one" depths ~f:(fun d ->
      d = depth_b);
  (* The last fill carries [values iterations]; a consumer compiled afterwards adds it in. *)
  let ctx_b = Context.set_values ctx_b tn [| 1.; 1.; 1. |] in
  let consumer = [%cd t =+ t.merge] in
  let consumer =
    { consumer with asgns = Ir.Assignments.Block_comment ("chain_consume", consumer.asgns) }
  in
  let ctx_b, consume = Context.compile ctx_b consumer IDX.empty in
  let ctx_b = Context.run ctx_b consume in
  Verdict.p "the consumer compiled after the fills reads the last one"
    (Poly.equal (Context.get_values ctx_b tn) (Array.map (values iterations) ~f:(fun v -> v +. 1.)));
  (* One release unit: releasing a copy's result releases its destination, and the reverse. *)
  let dead f =
    match f () with
    | _ -> false
    | exception Utils.User_error msg -> String.is_substring msg ~substring:"was released"
  in
  let result = Context.copy ~src:ctx_a ~dst:ctx_c tn in
  Context.release result;
  Verdict.p "releasing a copy's result releases its destination"
    (dead (fun () -> Context.get_values ctx_c tn));
  Context.release ctx_c;
  Verdict.p "releasing the destination again is harmless" true;
  let result = Context.copy ~src:ctx_a ~dst:ctx_b tn in
  Context.release ctx_b;
  Verdict.p "releasing the destination releases a copy's result"
    (dead (fun () -> Context.get_values result tn));
  Context.release ctx_a

let () =
  scenario_on_device ();
  scenario_host_init_fallback ();
  scenario_cross_backend ();
  scenario_merge_buffer ();
  scenario_copy_chain ()
