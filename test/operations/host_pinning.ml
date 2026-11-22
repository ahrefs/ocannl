(* gh-ocannl-1854: device-free state-machine tests for [Ir.Host_pinning].

   Covers: threshold gating, disable via [min_bytes = 0], second-sighting registration, idempotent
   third sighting, decline without retry, GC finaliser unregistration (with a kept-witness positive
   control), unregister exception swallowing, finaliser reentrancy inside [register] without
   deadlocking, [unregister_all] teardown followed by GC (exactly-once unregister), and address
   table draining across collection cycles. *)

open Base
open Verdict.Claims
module HP = Ir.Host_pinning
module Nd = Ir.Ndarray
module Ops = Ir.Ops

let make_nd floats = Nd.create_array ~debug:"hp_test" Ops.single ~dims:[| floats |] ~padding:None

let[@inline never] build_registered_and_drop ~hp ~floats =
  let nd = make_nd floats in
  let r1 = HP.note_transfer hp nd in
  let r2 = HP.note_transfer hp nd in
  let r3 = HP.note_transfer hp nd in
  (* Witness the underlying Bigarray.Genarray.t (which owns the C storage and carries the GC
     finaliser), not the outer Ndarray.t wrapper. *)
  let w = Nd.apply { f = Gc_witness.witness } nd in
  (w, r1, r2, r3)

let[@inline never] build_once_seen_and_drop ~hp ~floats =
  let nd = make_nd floats in
  let r1 = HP.note_transfer hp nd in
  let w = Gc_witness.witness nd in
  (w, r1)

let[@inline never] build_declined_and_drop ~hp ~floats =
  let nd = make_nd floats in
  let r1 = HP.note_transfer hp nd in
  let r2 = HP.note_transfer hp nd in
  let r3 = HP.note_transfer hp nd in
  let w = Gc_witness.witness nd in
  (w, r1, r2, r3)

let[@inline never] build_for_unregister_all ~hp ~floats =
  let nd = make_nd floats in
  let r1 = HP.note_transfer hp nd in
  let r2 = HP.note_transfer hp nd in
  let w = Gc_witness.witness nd in
  (nd, w, r1, r2)

let () =
  (* 1. Below-threshold arrays are never sighted or registered. *)
  let reg_calls = ref 0 and unreg_calls = ref 0 in
  let hp =
    HP.create
      ~min_bytes:(fun () -> 4096)
      ~register:(fun ~addr ~bytes ->
        Int.incr reg_calls;
        Some (addr, bytes))
      ~unregister:(fun _ -> Int.incr unreg_calls)
  in
  let small =
    make_nd 256
    (* 1024 B < 4096 B *)
  in
  let s1 = HP.note_transfer hp small in
  let s2 = HP.note_transfer hp small in
  let s3 = HP.note_transfer hp small in
  let st = HP.stats hp in
  p "below threshold: repeated transfers never register"
    ((not s1) && (not s2) && (not s3) && !reg_calls = 0 && st.registered = 0 && st.live_bytes = 0);

  (* 2. min_bytes = 0 disables registration for arrays of any size. *)
  let dis_reg = ref 0 in
  let hp_dis =
    HP.create
      ~min_bytes:(fun () -> 0)
      ~register:(fun ~addr ~bytes ->
        Int.incr dis_reg;
        Some (addr, bytes))
      ~unregister:(fun _ -> ())
  in
  let big_dis =
    make_nd 4096
    (* 16384 B *)
  in
  let d1 = HP.note_transfer hp_dis big_dis in
  let d2 = HP.note_transfer hp_dis big_dis in
  let st_dis = HP.stats hp_dis in
  p "disabled (min_bytes = 0): large repeated transfers never register"
    ((not d1) && (not d2) && !dis_reg = 0 && st_dis.registered = 0 && st_dis.live_bytes = 0);

  (* 3. Second-sighting registration, third-sighting idempotence, and GC unregistration. *)
  let live_nd =
    make_nd 2048
    (* 8192 B *)
  in
  let live_w = Gc_witness.witness live_nd in
  p "kept control: live host array survives collection rounds" (Gc_witness.kept live_w);
  let k1 = HP.note_transfer hp live_nd in
  let k2 = HP.note_transfer hp live_nd in
  let k3 = HP.note_transfer hp live_nd in
  let st_live = HP.stats hp in
  p "above threshold: registers on second transfer and stays idempotent on third"
    ((not k1) && k2 && k3 && !reg_calls = 1 && st_live.registered = 1 && st_live.live_bytes = 8192);
  ignore (Stdlib.Sys.opaque_identity live_nd);

  (* Drop a registered array and confirm GC finaliser unregisters it exactly once. *)
  let w_drop, dr1, dr2, dr3 = build_registered_and_drop ~hp ~floats:2048 in
  let drop_reclaimed = Gc_witness.reclaimed w_drop in
  let st_after_drop = HP.stats hp in
  p "dropped registered array: registered on second sighting, reclaimed and unregistered once"
    ((not dr1) && dr2 && dr3 && drop_reclaimed && !reg_calls = 2 && !unreg_calls = 1
   && st_after_drop.registered = 2 && st_after_drop.unregistered = 1
    && st_after_drop.unregister_failures = 0
    && st_after_drop.live_bytes = 8192);

  (* Dropping a once-seen (unregistered) array must not call unregister. *)
  let w_once, or1 = build_once_seen_and_drop ~hp ~floats:2048 in
  let once_reclaimed = Gc_witness.reclaimed w_once in
  let st_after_once = HP.stats hp in
  p "dropped once-seen array: never registered and never unregistered"
    ((not or1) && once_reclaimed && !reg_calls = 2 && !unreg_calls = 1
   && st_after_once.live_bytes = 8192);

  (* Keep live_nd alive until after the drop checks, then unregister_all. *)
  let live_still = HP.note_transfer hp live_nd in
  p "live array remains registered after unrelated drops" live_still;

  (* 4. Decline without retry. *)
  let dec_reg = ref 0 and dec_unreg = ref 0 in
  let hp_dec =
    HP.create
      ~min_bytes:(fun () -> 4096)
      ~register:(fun ~addr:_ ~bytes:_ ->
        Int.incr dec_reg;
        None)
      ~unregister:(fun _ -> Int.incr dec_unreg)
  in
  let w_dec, cr1, cr2, cr3 = build_declined_and_drop ~hp:hp_dec ~floats:2048 in
  let dec_reclaimed = Gc_witness.reclaimed w_dec in
  let st_dec = HP.stats hp_dec in
  p "declined registration: not retried on third transfer and never unregistered on drop"
    ((not cr1) && (not cr2) && (not cr3) && dec_reclaimed && !dec_reg = 1 && !dec_unreg = 0
   && st_dec.registered = 0 && st_dec.declined = 1 && st_dec.live_bytes = 0);

  (* 5. Unregister failure in GC finaliser keeps token, backing array, and live_bytes retryable;
     once the driver recovers, retry unregisters cleanly and releases the backing array. *)
  let should_fail_unreg = ref true in
  let hp_fail =
    HP.create
      ~min_bytes:(fun () -> 4096)
      ~register:(fun ~addr ~bytes -> Some (addr, bytes))
      ~unregister:(fun _ -> if !should_fail_unreg then failwith "simulated driver unregister error")
  in
  let w_fail, fr1, fr2, _ = build_registered_and_drop ~hp:hp_fail ~floats:2048 in
  (* Run settle rounds so the finaliser runs once and fails; the retry entry keeps [nd] alive. *)
  Gc_witness.settle ();
  Gc_witness.settle ();
  let st_fail = HP.stats hp_fail in
  let kept_while_failed = Gc_witness.kept ~rounds:1 w_fail in
  should_fail_unreg := false;
  HP.unregister_all hp_fail;
  let st_retried = HP.stats hp_fail in
  let reclaimed_after_retry = Gc_witness.reclaimed w_fail in
  p
    "unregister failure in finaliser: stays retryable with live_bytes and storage kept until retry \
     succeeds"
    ((not fr1) && fr2 && kept_while_failed && st_fail.registered = 1 && st_fail.unregistered = 0
   && st_fail.unregister_failures = 1 && st_fail.live_bytes = 8192 && st_retried.unregistered = 1
   && st_retried.live_bytes = 0 && reclaimed_after_retry);

  (* 6. Finaliser reentrancy inside [register]: triggering GC while [note_transfer] holds the lock
     must not deadlock and must unregister the dead entry cleanly. *)
  let reent_unreg = ref 0 in
  let force_gc_in_register = ref false in
  let hp_reent =
    HP.create
      ~min_bytes:(fun () -> 4096)
      ~register:(fun ~addr ~bytes ->
        if !force_gc_in_register then Gc_witness.settle ();
        Some (addr, bytes))
      ~unregister:(fun _ -> Int.incr reent_unreg)
  in
  let w_dead, _, dr_ok, _ = build_registered_and_drop ~hp:hp_reent ~floats:2048 in
  (* Do not settle yet; trigger GC inside the next array's [register] callback. *)
  force_gc_in_register := true;
  let nd_reent = make_nd 2048 in
  let rr1 = HP.note_transfer hp_reent nd_reent in
  let rr2 = HP.note_transfer hp_reent nd_reent in
  let dead_gone = Gc_witness.reclaimed w_dead in
  let st_reent = HP.stats hp_reent in
  p "finaliser reentrancy during register: no deadlock, dead entry unregistered once"
    (dr_ok && (not rr1) && rr2 && dead_gone && !reent_unreg = 1 && st_reent.registered = 2
   && st_reent.unregistered = 1 && st_reent.live_bytes = 8192);
  ignore (Stdlib.Sys.opaque_identity nd_reent);

  (* 7. [unregister_all] unregisters every live entry once; subsequent GC finalisation is a
     no-op. *)
  let ua_unreg = ref 0 in
  let hp_ua =
    HP.create
      ~min_bytes:(fun () -> 4096)
      ~register:(fun ~addr ~bytes -> Some (addr, bytes))
      ~unregister:(fun _ -> Int.incr ua_unreg)
  in
  let w_ua =
    let nd_ua, w, ur1, ur2 = build_for_unregister_all ~hp:hp_ua ~floats:2048 in
    let st_before = HP.stats hp_ua in
    HP.unregister_all hp_ua;
    let st_mid = HP.stats hp_ua in
    p "unregister_all: releases live entries immediately"
      ((not ur1) && ur2 && st_before.live_bytes = 8192 && st_mid.live_bytes = 0 && !ua_unreg = 1
     && st_mid.unregistered = 1);
    ignore (Stdlib.Sys.opaque_identity nd_ua);
    w
  in
  let ua_reclaimed = Gc_witness.reclaimed w_ua in
  let st_after_ua = HP.stats hp_ua in
  p "unregister_all followed by GC: finaliser does not double-unregister"
    (ua_reclaimed && !ua_unreg = 1 && st_after_ua.unregistered = 1
    && st_after_ua.unregister_failures = 0
    && st_after_ua.live_bytes = 0);

  (* 8. Raising threshold after registration still reports already-registered arrays as registered
     (so CUDA still synchronizes the stream), while not registering new arrays. *)
  let dyn_min = ref 4096 in
  let dyn_reg = ref 0 in
  let hp_dyn =
    HP.create
      ~min_bytes:(fun () -> !dyn_min)
      ~register:(fun ~addr ~bytes ->
        Int.incr dyn_reg;
        Some (addr, bytes))
      ~unregister:(fun _ -> ())
  in
  let nd_a = make_nd 2048 and nd_b = make_nd 2048 in
  let _ = HP.note_transfer hp_dyn nd_a in
  let a2 = HP.note_transfer hp_dyn nd_a in
  dyn_min := 0;
  let a3 = HP.note_transfer hp_dyn nd_a in
  let b1 = HP.note_transfer hp_dyn nd_b in
  let b2 = HP.note_transfer hp_dyn nd_b in
  p "dynamic disable after registration: already-pinned array stays true, new arrays stay false"
    (a2 && a3 && (not b1) && (not b2) && !dyn_reg = 1);
  ignore (Stdlib.Sys.opaque_identity (nd_a, nd_b));

  (* 9. Overlapping views: sub-slice registered first, then parent uploaded -> slice evicted cleanly
     (no mixed-range copy), parent registered on second upload, then sub-slice covered by parent
     even when smaller than min_bytes. *)
  let ov_reg = ref 0 and ov_unreg = ref 0 in
  let hp_ov =
    HP.create
      ~min_bytes:(fun () -> 4096)
      ~register:(fun ~addr ~bytes ->
        Int.incr ov_reg;
        Some (addr, bytes))
      ~unregister:(fun _ -> Int.incr ov_unreg)
  in
  let parent_ba = Bigarray.Genarray.create Bigarray.float32 Bigarray.c_layout [| 4; 1024 |] in
  let parent_nd = Nd.as_array Ops.Single parent_ba in
  let slice_ba =
    Bigarray.Genarray.sub_left parent_ba 1 2
    (* 8 KiB inside 16 KiB parent *)
  in
  let slice_nd = Nd.as_array Ops.Single slice_ba in
  let tiny_slice_ba =
    Bigarray.Genarray.sub_left parent_ba 0 1
    (* 4 KiB = threshold *)
  in
  let tiny_half_ba =
    Bigarray.Genarray.sub_left
      (Bigarray.reshape_1 tiny_slice_ba 1024 |> Bigarray.genarray_of_array1)
      0 512
    (* 2 KiB < threshold *)
  in
  let tiny_slice_nd = Nd.as_array Ops.Single tiny_half_ba in
  let sl1 = HP.note_transfer hp_ov slice_nd in
  let sl2 = HP.note_transfer hp_ov slice_nd in
  let st_sl = HP.stats hp_ov in
  (* First transfer of parent partially overlaps the registered slice: must evict slice first. *)
  let p1 = HP.note_transfer hp_ov parent_nd in
  let st_p1 = HP.stats hp_ov in
  (* Second transfer of parent registers the whole 16 KiB parent. *)
  let p2 = HP.note_transfer hp_ov parent_nd in
  let st_p2 = HP.stats hp_ov in
  (* Sub-slices (both 8 KiB and below-threshold 2 KiB) are now covered by the registered parent. *)
  let sl3 = HP.note_transfer hp_ov slice_nd in
  let tiny_cov = HP.note_transfer hp_ov tiny_slice_nd in
  let st_cov = HP.stats hp_ov in
  p
    "overlapping slice then parent: slice evicted on parent upload, parent registers, slices \
     covered without re-registering"
    ((not sl1) && sl2 && st_sl.registered = 1 && st_sl.live_bytes = 8192 && (not p1)
   && !ov_unreg = 1 && st_p1.unregistered = 1 && st_p1.live_bytes = 0 && p2 && !ov_reg = 2
   && st_p2.registered = 2 && st_p2.live_bytes = 16384 && sl3 && tiny_cov && st_cov.registered = 2
   && st_cov.unregistered = 1 && st_cov.live_bytes = 16384);
  ignore (Stdlib.Sys.opaque_identity (parent_nd, slice_nd, tiny_slice_nd));

  (* 10. Failed overlap eviction refuses the mixed-range parent transfer with
     [Overlap_eviction_failed], keeps the slice retryable, and succeeds once unregister recovers. *)
  let fail_evict = ref true in
  let hp_ev =
    HP.create
      ~min_bytes:(fun () -> 4096)
      ~register:(fun ~addr ~bytes -> Some (addr, bytes))
      ~unregister:(fun _ -> if !fail_evict then failwith "evict error")
  in
  let ev_parent_ba = Bigarray.Genarray.create Bigarray.float32 Bigarray.c_layout [| 2; 1024 |] in
  let ev_parent_nd = Nd.as_array Ops.Single ev_parent_ba in
  let ev_slice_nd = Nd.as_array Ops.Single (Bigarray.Genarray.sub_left ev_parent_ba 0 1) in
  let _ = HP.note_transfer hp_ev ev_slice_nd in
  let ev_sl2 = HP.note_transfer hp_ev ev_slice_nd in
  let refused =
    try
      ignore (HP.note_transfer hp_ev ev_parent_nd : bool);
      false
    with HP.Overlap_eviction_failed _ -> true
  in
  let st_refused = HP.stats hp_ev in
  fail_evict := false;
  let ev_p1 = HP.note_transfer hp_ev ev_parent_nd in
  let ev_p2 = HP.note_transfer hp_ev ev_parent_nd in
  let st_ev_ok = HP.stats hp_ev in
  p
    "failed overlap eviction: refuses mixed-range parent transfer, then recovers and registers \
     parent"
    (ev_sl2 && refused
    && st_refused.unregister_failures = 1
    && st_refused.live_bytes = 4096 && (not ev_p1) && ev_p2 && st_ev_ok.unregistered = 1
    && st_ev_ok.registered = 2 && st_ev_ok.live_bytes = 8192);
  ignore (Stdlib.Sys.opaque_identity (ev_parent_nd, ev_slice_nd));

  (* 11. Failed rollback after a finaliser raced with [register]: two views share the exact same
     byte range; the first-sighted view is dropped so GC finalises it inside [register] of the
     second view, and rollback [unregister] raises. [note_transfer] must still return [true] (the
     driver still holds the pin) and keep live_bytes retryable until [unregister] recovers. *)
  let rb_fail = ref true in
  let trigger_race = ref false in
  let hp_rb =
    HP.create
      ~min_bytes:(fun () -> 4096)
      ~register:(fun ~addr ~bytes ->
        if !trigger_race then Gc_witness.settle ();
        Some (addr, bytes))
      ~unregister:(fun _ -> if !rb_fail then failwith "rollback unregister error")
  in
  let rb_ba = Bigarray.Genarray.create Bigarray.float32 Bigarray.c_layout [| 2048 |] in
  let[@inline never] sight_first_view () =
    let alias = Bigarray.reshape_1 rb_ba 2048 |> Bigarray.genarray_of_array1 in
    let nd = Nd.as_array Ops.Single alias in
    let r = HP.note_transfer hp_rb nd in
    let w = Gc_witness.witness alias in
    (w, r)
  in
  let w_alias, rb_s1 = sight_first_view () in
  let rb_nd2 = Nd.as_array Ops.Single rb_ba in
  trigger_race := true;
  let rb_r2 = HP.note_transfer hp_rb rb_nd2 in
  let alias_gone = Gc_witness.reclaimed w_alias in
  let st_rb_fail = HP.stats hp_rb in
  rb_fail := false;
  HP.unregister_all hp_rb;
  let st_rb_ok = HP.stats hp_rb in
  p "failed rollback after finaliser race during register: reports pinned and recovers on retry"
    ((not rb_s1) && alias_gone && rb_r2 && st_rb_fail.registered = 1
    && st_rb_fail.unregister_failures = 1
    && st_rb_fail.live_bytes = 8192 && st_rb_ok.unregistered = 1 && st_rb_ok.live_bytes = 0);
  ignore (Stdlib.Sys.opaque_identity rb_nd2)
