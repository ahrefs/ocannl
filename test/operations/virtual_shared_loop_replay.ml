(* gh-ocannl-2510: shared-loop read of a virtual value before a sibling setter updates it.

   1. Extent-20 reduction fixture (probe 20261126T174901Z-38521): [loc] settles [Local] (reduction
   extent 20 exceeds [max_inline_reduction = 16]); [v_loc] is virtual and fed by [loc]; a
   shared-loop setter [v_reader] reads [v_loc] before a sibling setter updates [v_loc]; a consumer
   [out] reads [v_reader] in the same routine. On base this aborts at [Low_level.virtual_llc
   decision-coverage seam] for the surviving [les_v_loc] read.

   2. Shared-versus-split comparison (deferred from gh-ocannl-2493): two distinct dependency writes
   run in shared versus split loops. Each stored reader is inspected for a retained [Get] versus a
   captured [Local_scope], and executed host reads (and same-routine reads) are compared against
   both producer-time and replay-time materialized references with discriminating producers. *)

open Base
open Stdio
open Verdict.Claims
module L = Ll_test
module LL = Ir.Low_level
module Tn = Ir.Tnode

let with_pinned_virtualize_settings f =
  let s = LL.virtualize_settings in
  let saved_enable = s.enable_device_only
  and saved_visits = s.max_visits
  and saved_red = s.max_inline_reduction
  and saved_fanin = s.max_inline_fanin in
  s.enable_device_only <- true;
  s.max_visits <- 2;
  s.max_inline_reduction <- 16;
  s.max_inline_fanin <- 8;
  Exn.protect ~f ~finally:(fun () ->
      s.enable_device_only <- saved_enable;
      s.max_visits <- saved_visits;
      s.max_inline_reduction <- saved_red;
      s.max_inline_fanin <- saved_fanin)

let n = 4
let node = L.node_factory ~first_id:2510000 ~dims:[| n |] ()
let exact = Array.equal Float.equal

(* Count [Get]s of [dep] and [Local_scope]s for [dep] inside [self]'s own setters across all stored
   components of [self]. *)
let stored_reader_shape (ctx : LL.optimize_ctx) ~self ~dep =
  let comps = Hashtbl.find_multi ctx.LL.computations self in
  let direct_gets = ref 0 and captured_scopes = ref 0 in
  let inspect_rhs llsc =
    let stub = L.set_at self (L.fixed 0) llsc in
    direct_gets :=
      !direct_gets
      + L.count_scalar stub ~f:(function
        | LL.Get (tn, _) | LL.Get_dynamic { tn; _ } -> Tn.equal tn dep
        | _ -> false);
    captured_scopes :=
      !captured_scopes
      + L.count_scalar stub ~f:(function
        | LL.Local_scope { id; _ } -> Tn.equal id.tn dep
        | _ -> false)
  in
  List.iter comps ~f:(fun (_, body) ->
      L.walk ~in_scopes:false body ~on_stmt:(function
        | LL.Set { tn; llsc; _ } when Tn.equal tn self -> inspect_rhs llsc
        | LL.Set_from_vec { tn; arg = a, _; _ } when Tn.equal tn self -> inspect_rhs a
        | _ -> ()));
  (!direct_gets, !captured_scopes)

(* === Part 1: extent-20 reduction fixture (probe 20261126T174901Z-38521) === *)
let case_local_shared_loop ~label ~tag ~in_loop_out ~observable_reader =
  let build ~materialized_extra =
    let loc = node (tag ^ "_loc") in
    let v_loc = node (tag ^ "_les_v_loc") in
    let v_reader = node (tag ^ "_v_reader") in
    let out = node (tag ^ "_out") in
    let out_replay = node (tag ^ "_out_replay") in
    L.materialize out;
    L.materialize out_replay;
    if observable_reader then Tn.set_observable v_reader;
    let i1 = L.sym () and r = L.sym () and i2 = L.sym () and i3 = L.sym () in
    let i4 = L.sym () and i5 = L.sym () in
    let shared_body =
      let read_v = L.set_at v_reader (L.iter i3) (L.add (L.get v_loc [| L.iter i3 |]) (L.c 10.)) in
      let update_v =
        L.set_at v_loc (L.iter i3)
          (L.add (L.get loc [| L.iter i3 |]) (L.add (L.c 2.) (L.mul (L.c 2.) (L.embed i3))))
      in
      if in_loop_out then
        L.seq read_v (L.seq (L.set_at out (L.iter i3) (L.get v_reader [| L.iter i3 |])) update_v)
      else L.seq read_v update_v
    in
    let llc =
      LL.unflat_lines
      @@ [
           L.loop ~upto:(n - 1) i1
             (L.seq
                (L.set_at loc (L.iter i1) (L.c 0.))
                (L.loop ~upto:19 r
                   (L.set_at loc (L.iter i1)
                      (L.add (L.get loc [| L.iter i1 |]) (L.add (L.tick i1) (L.embed r))))));
           L.loop ~upto:(n - 1) i2
             (L.set_at v_loc (L.iter i2) (L.add (L.get loc [| L.iter i2 |]) (L.tick i2)));
           L.loop ~upto:(n - 1) i3 shared_body;
         ]
      @ (if in_loop_out then []
         else
           [ L.loop ~upto:(n - 1) i4 (L.set_at out (L.iter i4) (L.get v_reader [| L.iter i4 |])) ])
      @ [
          L.loop ~upto:(n - 1) i5
            (L.set_at out_replay (L.iter i5) (L.add (L.get v_loc [| L.iter i5 |]) (L.c 10.)));
        ]
    in
    let materialized =
      List.filter_map materialized_extra ~f:(function
        | `V_loc -> Some v_loc
        | `V_reader -> Some v_reader)
    in
    (loc, v_loc, v_reader, out, out_replay, materialized, llc)
  in
  let loc, v_loc, v_reader, out, out_replay, _, llc = build ~materialized_extra:[] in
  let o = with_pinned_virtualize_settings (fun () -> L.optimize ~name:("vslr_" ^ tag) llc) in
  let ctx = L.run ~ctx:(Context.auto ()) ~name:("vslr_" ^ tag) o ~seed:[] in
  (* Expected values: loc[i] = 210 + 20*i W1: v_loc[i] = loc[i] + (1 + i) = 211 + 21*i =>
     producer-time out = 221 + 21*i W2: v_loc[i] = loc[i] + (2 + 2*i) = 212 + 22*i => replay-time
     out = 222 + 22*i *)
  let expected_producer = Array.init n ~f:(fun i -> Float.of_int (221 + (21 * i))) in
  let expected_replay = Array.init n ~f:(fun i -> Float.of_int (222 + (22 * i))) in
  let _, _, _, out_m, out_replay_m, mat, llc_m = build ~materialized_extra:[ `V_loc; `V_reader ] in
  let o_m =
    with_pinned_virtualize_settings (fun () ->
        L.optimize ~materialized:mat ~name:("vslr_" ^ tag ^ "_mat") llc_m)
  in
  let got_m = L.execute ~name:("vslr_" ^ tag ^ "_mat") o_m ~seed:[] ~read:[ out_m; out_replay_m ] in
  let mat_producer, mat_replay = match got_m with [ a; b ] -> (a, b) | _ -> assert false in
  p
    (label ^ ": materialized producer-time reference matches closed form")
    (exact mat_producer expected_producer);
  p
    (label ^ ": materialized replay-time reference matches closed form")
    (exact mat_replay expected_replay);
  p (label ^ ": loc settles Local") (L.known_local o loc);
  p (label ^ ": v_loc stays Virtual") (L.known_virtual o v_loc);
  let direct_gets, captured_scopes =
    stored_reader_shape o.LL.optimize_ctx ~self:v_reader ~dep:v_loc
  in
  p
    (label ^ ": stored v_reader retains Get(v_loc) and no Local_scope(v_loc)")
    (direct_gets = 1 && captured_scopes = 0);
  if observable_reader then (
    p
      (label ^ ": observable v_reader materializes On_device")
      (match Context.observability ctx v_reader with Context.Stored -> true | _ -> false);
    p
      (label ^ ": observable v_reader matches producer-time reference (captured before sibling W2)")
      (exact (Context.get_values ctx v_reader) mat_producer))
  else (
    p (label ^ ": v_reader stays Virtual") (L.known_virtual o v_reader);
    let has_local_path ctx =
      match Context.observability ctx v_reader with
      | Context.Unobservable_local path -> List.equal Tn.equal path [ v_reader; v_loc; loc ]
      | _ -> false
    in
    p
      (label ^ ": observability reports Unobservable_local [v_reader; v_loc; loc]")
      (has_local_path ctx);
    let seeded = Context.set_values ctx v_loc (L.blank n) in
    p
      (label ^ ": seeding v_loc keeps the Unobservable_local path to loc")
      (Context.mem seeded v_loc && has_local_path seeded));
  if in_loop_out then
    p
      (label ^ ": in-loop out before sibling W2 matches producer-time reference")
      (exact (Context.get_values ctx out) mat_producer)
  else
    p
      (label ^ ": out matches replay-time reference")
      (exact (Context.get_values ctx out) mat_replay);
  p
    (label ^ ": out_replay matches replay-time reference")
    (exact (Context.get_values ctx out_replay) mat_replay)

(* === Part 2: shared vs split loops with two distinct dependency writes (deferred from #2493)
   === *)
let case_shared_vs_split () =
  let early_vals = Array.init n ~f:(fun i -> Float.of_int (1 + (3 * i))) in
  (* W1: dep[i] = early[i] + (5 + i) = 6 + 4*i => producer-time tap = 16 + 4*i W2: dep[i] = early[i]
     + (6 + 2*i) = 7 + 5*i => replay-time tap = 17 + 5*i *)
  let expected_producer = Array.init n ~f:(fun i -> Float.of_int (16 + (4 * i))) in
  let expected_replay = Array.init n ~f:(fun i -> Float.of_int (17 + (5 * i))) in
  let run_variant ~name ~shared ~shield ~stored ~same_routine_out =
    let early = node (name ^ "_early") in
    let dep = node (name ^ "_dep") in
    let tap = node (name ^ "_tap") in
    let forward = node (name ^ "_forward") in
    let out = node (name ^ "_out") in
    let tap_at_read = node (name ^ "_tap_at_read") in
    L.materialize early;
    L.materialize forward;
    if shield then Tn.set_observable dep;
    if same_routine_out then L.materialize out;
    let at tn i = L.get tn [| L.iter i |] in
    let w1 i = L.add (at early i) (L.add (L.c 5.) (L.embed i)) in
    let w2 i = L.add (at early i) (L.add (L.c 6.) (L.mul (L.c 2.) (L.embed i))) in
    let tap_expr i = L.add (at dep i) (L.c 10.) in
    let i1 = L.sym () and i2 = L.sym () and i3 = L.sym () and i4 = L.sym () in
    let i5 = L.sym () and i6 = L.sym () in
    let middle =
      if shared then
        [
          L.loop_n i2 n
            (L.seq (L.set_at tap (L.iter i2) (tap_expr i2)) (L.set_at dep (L.iter i2) (w2 i2)));
        ]
      else
        [
          L.loop_n i2 n (L.set_at tap (L.iter i2) (tap_expr i2));
          L.loop_n i3 n (L.set_at dep (L.iter i3) (w2 i3));
        ]
    in
    let forward_rhs i =
      let dep_val =
        if shield then
          L.gather ~tn:dep ~idcs:[| L.iter i |] ~dyn_axis:0 ~dyn_value:(L.embed i, L.iprec ())
        else at dep i
      in
      L.add (at early i) dep_val
    in
    let llc =
      LL.unflat_lines
      @@ [ L.loop_n i1 n (L.set_at dep (L.iter i1) (w1 i1)) ]
      @ middle
      @ (if same_routine_out then [ L.loop_n i4 n (L.set_at out (L.iter i4) (at tap i4)) ] else [])
      @ [ L.loop_n i5 n (L.set_at forward (L.iter i5) (forward_rhs i5)) ]
      @ if stored then [ L.loop_n i6 n (L.set_at tap_at_read (L.iter i6) (tap_expr i6)) ] else []
    in
    let materialized = if stored then [ dep; tap; tap_at_read ] else [] in
    let o = with_pinned_virtualize_settings (fun () -> L.optimize ~name ~materialized llc) in
    let direct_gets, captured_scopes = stored_reader_shape o.LL.optimize_ctx ~self:tap ~dep in
    let ctx = L.run ~ctx:(Context.auto ()) ~name o ~seed:[ (early, early_vals) ] in
    (o, ctx, dep, tap, out, tap_at_read, direct_gets, captured_scopes)
  in
  let _, ref_ctx, _, ref_tap_prod, _, ref_tap_replay, _, _ =
    run_variant ~name:"svs_ref" ~shared:true ~shield:false ~stored:true ~same_routine_out:false
  in
  let ref_producer = Context.get_values ref_ctx ref_tap_prod in
  let ref_replay = Context.get_values ref_ctx ref_tap_replay in
  p "shared vs split: materialized producer-time reference matches closed form"
    (exact ref_producer expected_producer);
  p "shared vs split: materialized replay-time reference matches closed form"
    (exact ref_replay expected_replay);

  (* Split loops, virtual dep: stored tap captures Local_scope(dep), host read matches
     producer-time *)
  let o_sp, ctx_sp, dep_sp, tap_sp, out_sp, _, gets_sp, scopes_sp =
    run_variant ~name:"svs_split_virt" ~shared:false ~shield:false ~stored:false
      ~same_routine_out:true
  in
  p "split (virtual dep): tap and dep stay virtual"
    (L.known_virtual o_sp tap_sp && L.known_virtual o_sp dep_sp);
  p "split (virtual dep): stored tap captures Local_scope(dep) and retains no Get(dep)"
    (gets_sp = 0 && scopes_sp = 1);
  p "split (virtual dep): host read of tap matches producer-time reference"
    (exact (Context.get_values ctx_sp tap_sp) ref_producer);
  p "split (virtual dep): same-routine out matches producer-time reference"
    (exact (Context.get_values ctx_sp out_sp) ref_producer);

  (* Split loops, materialized shield: stored tap captures Local_scope(dep) which retracts to
     buffer, so host read and same-routine out match replay-time *)
  let o_sps, ctx_sps, dep_sps, tap_sps, out_sps, _, gets_sps, scopes_sps =
    run_variant ~name:"svs_split_shield" ~shared:false ~shield:true ~stored:false
      ~same_routine_out:true
  in
  p "split (materialized shield): tap stays virtual and dep is non-virtual"
    (L.known_virtual o_sps tap_sps && L.known_non_virtual o_sps dep_sps);
  p "split (materialized shield): stored tap captured Local_scope(dep) (retracted at consumption)"
    (gets_sps = 0 && scopes_sps = 1);
  p "split (materialized shield): host read of tap matches replay-time reference"
    (exact (Context.get_values ctx_sps tap_sps) ref_replay);
  p "split (materialized shield): same-routine out matches replay-time reference"
    (exact (Context.get_values ctx_sps out_sps) ref_replay);

  (* Shared loop, virtual dep, host read only (no same-routine consumer of tap) *)
  let o_sh, ctx_sh, dep_sh, tap_sh, _, _, gets_sh, scopes_sh =
    run_variant ~name:"svs_shared_virt_host" ~shared:true ~shield:false ~stored:false
      ~same_routine_out:false
  in
  p "shared (virtual dep, host only): tap and dep stay virtual"
    (L.known_virtual o_sh tap_sh && L.known_virtual o_sh dep_sh);
  p "shared (virtual dep, host only): stored tap retains Get(dep) and no Local_scope(dep)"
    (gets_sh = 1 && scopes_sh = 0);
  p "shared (virtual dep, host only): host read of tap matches replay-time reference"
    (exact (Context.get_values ctx_sh tap_sh) ref_replay);

  (* Shared loop, virtual dep, WITH same-routine consumer of tap *)
  let o_shr, ctx_shr, dep_shr, tap_shr, out_shr, _, gets_shr, scopes_shr =
    run_variant ~name:"svs_shared_virt_same" ~shared:true ~shield:false ~stored:false
      ~same_routine_out:true
  in
  p "shared (virtual dep, same-routine read): tap and dep stay virtual"
    (L.known_virtual o_shr tap_shr && L.known_virtual o_shr dep_shr);
  p "shared (virtual dep, same-routine read): stored tap retains Get(dep) and no Local_scope(dep)"
    (gets_shr = 1 && scopes_shr = 0);
  p "shared (virtual dep, same-routine read): host read of tap matches replay-time reference"
    (exact (Context.get_values ctx_shr tap_shr) ref_replay);
  p "shared (virtual dep, same-routine read): same-routine out matches replay-time reference"
    (exact (Context.get_values ctx_shr out_shr) ref_replay);

  (* Shared loop, materialized shield *)
  let o_shs, ctx_shs, dep_shs, tap_shs, out_shs, _, gets_shs, scopes_shs =
    run_variant ~name:"svs_shared_shield" ~shared:true ~shield:true ~stored:false
      ~same_routine_out:true
  in
  p "shared (materialized shield): tap stays virtual and dep is non-virtual"
    (L.known_virtual o_shs tap_shs && L.known_non_virtual o_shs dep_shs);
  p "shared (materialized shield): stored tap retains Get(dep) and no Local_scope(dep)"
    (gets_shs = 1 && scopes_shs = 0);
  p "shared (materialized shield): host read of tap matches replay-time reference"
    (exact (Context.get_values ctx_shs tap_shs) ref_replay);
  p "shared (materialized shield): same-routine out matches replay-time reference"
    (exact (Context.get_values ctx_shs out_shs) ref_replay)

(* === Part 3: multi-hop retained chain in one shared loop === *)
let case_retained_chain () =
  let x = node "rc_x" and a = node "rc_a" and b = node "rc_b" and c = node "rc_c" in
  let out = node "rc_out" in
  L.materialize x;
  L.materialize out;
  let x_vals = Array.init n ~f:(fun i -> Float.of_int (10 + (3 * i))) in
  let i1 = L.sym () and i2 = L.sym () and i3 = L.sym () in
  let at tn i = L.get tn [| L.iter i |] in
  let llc =
    LL.unflat_lines
      [
        L.loop_n i1 n
          (L.seq
             (L.set_at b (L.iter i1) (L.add (at x i1) (L.c 1.)))
             (L.set_at c (L.iter i1) (L.add (at x i1) (L.c 2.))));
        L.loop_n i2 n
          (L.seq
             (L.set_at a (L.iter i2) (L.add (at b i2) (L.c 10.)))
             (L.seq
                (L.set_at b (L.iter i2) (L.add (at c i2) (L.c 20.)))
                (L.set_at c (L.iter i2) (L.add (at x i2) (L.c 30.)))));
        L.loop_n i3 n (L.set_at out (L.iter i3) (at a i3));
      ]
  in
  let o = with_pinned_virtualize_settings (fun () -> L.optimize ~name:"vslr_chain" llc) in
  p "retained chain: a, b, and c stay virtual"
    (L.known_virtual o a && L.known_virtual o b && L.known_virtual o c);
  p "retained chain: no buffer reads of a, b, or c survive"
    (L.count_get o a = 0 && L.count_get o b = 0 && L.count_get o c = 0);
  let got = L.execute ~name:"vslr_chain" o ~seed:[ (x, x_vals) ] ~read:[ out ] in
  (* Replay-time: c = x + 30, b = c + 20 = x + 50, a = b + 10 = x + 60 *)
  let expected = Array.map x_vals ~f:(fun v -> v +. 60.) in
  p "retained chain: executed out matches replay-time closed form"
    (exact (List.hd_exn got) expected)

(* === Part 4: mutual retained-read cycle across two shared loops === *)
let case_cyclic_retained_reads () =
  let x_vals = Array.init n ~f:(fun i -> Float.of_int (20 + (5 * i))) in
  let build ~tag ~declare_a_virtual =
    let x = node (tag ^ "_x") and a = node (tag ^ "_a") and b = node (tag ^ "_b") in
    let out = node (tag ^ "_out") in
    L.materialize x;
    L.materialize out;
    if declare_a_virtual then L.virtualize a;
    let i1 = L.sym () and i2 = L.sym () and i3 = L.sym () and i4 = L.sym () in
    let at tn i = L.get tn [| L.iter i |] in
    let llc =
      LL.unflat_lines
        [
          L.loop_n i1 n
            (L.seq
               (L.set_at a (L.iter i1) (L.add (at x i1) (L.c 1.)))
               (L.set_at b (L.iter i1) (L.add (at x i1) (L.c 2.))));
          L.loop_n i2 n
            (L.seq
               (L.set_at a (L.iter i2) (L.add (at b i2) (L.c 10.)))
               (L.set_at b (L.iter i2) (L.add (at x i2) (L.c 3.))));
          L.loop_n i3 n
            (L.seq
               (L.set_at b (L.iter i3) (L.add (at a i3) (L.c 20.)))
               (L.set_at a (L.iter i3) (L.add (at x i3) (L.c 4.))));
          L.loop_n i4 n (L.set_at out (L.iter i4) (at a i4));
        ]
    in
    (x, a, b, out, llc)
  in
  let x, a, b, out, llc = build ~tag:"cyc_u" ~declare_a_virtual:false in
  let o = with_pinned_virtualize_settings (fun () -> L.optimize ~name:"vslr_cyc_u" llc) in
  p "cyclic retained reads (undecided): cycle breaks by materializing a with 162"
    (Option.equal Tn.equal_provenance (L.rejection_code o a)
       (Some (Tn.Site "162:cyclic-virtual-replay"))
    && L.known_virtual o b);
  let got = L.execute ~name:"vslr_cyc_u" o ~seed:[ (x, x_vals) ] ~read:[ out ] in
  let xm, am, bm, outm, llcm = build ~tag:"cyc_m" ~declare_a_virtual:false in
  let o_m =
    with_pinned_virtualize_settings (fun () ->
        L.optimize ~materialized:[ am; bm ] ~name:"vslr_cyc_m" llcm)
  in
  let got_m = L.execute ~name:"vslr_cyc_m" o_m ~seed:[ (xm, x_vals) ] ~read:[ outm ] in
  let expected = Array.map x_vals ~f:(fun v -> v +. 4.) in
  p "cyclic retained reads (undecided): executed out matches materialized run and closed form"
    (exact (List.hd_exn got) expected && exact (List.hd_exn got) (List.hd_exn got_m));

  let _, av, _, _, llc_v = build ~tag:"cyc_v" ~declare_a_virtual:true in
  let refused =
    match with_pinned_virtualize_settings (fun () -> L.optimize ~name:"vslr_cyc_v" llc_v) with
    | _ -> false
    | exception Utils.User_error msg ->
        String.is_substring msg ~substring:"cyclic virtual replay"
        && String.is_substring msg ~substring:(Tn.debug_name av)
  in
  p "cyclic retained reads (declared virtual): raises named User_error" refused

(* === Part 5: cross-routine retained read when routine B also updates the dependency === *)
let case_inherited_updated_dep () =
  let early = node "iud_early" and dep = node "iud_dep" and tap = node "iud_tap" in
  let out_before = node "iud_out_before" and out_after = node "iud_out_after" in
  L.materialize early;
  L.materialize out_before;
  L.materialize out_after;
  let early_vals = Array.init n ~f:(fun i -> Float.of_int (2 + (3 * i))) in
  let at tn i = L.get tn [| L.iter i |] in
  let i1 = L.sym () and i2 = L.sym () in
  let llc_a =
    LL.unflat_lines
      [
        L.loop_n i1 n (L.set_at dep (L.iter i1) (L.add (at early i1) (L.c 5.)));
        L.loop_n i2 n
          (L.seq
             (L.set_at tap (L.iter i2) (L.add (at dep i2) (L.c 10.)))
             (L.set_at dep (L.iter i2) (L.add (at early i2) (L.c 6.))));
      ]
  in
  let i3 = L.sym () and i4 = L.sym () and i5 = L.sym () in
  let llc_b =
    LL.unflat_lines
      [
        L.loop_n i3 n (L.set_at out_before (L.iter i3) (at tap i3));
        L.loop_n i4 n (L.set_at dep (L.iter i4) (L.add (at early i4) (L.c 90.)));
        L.loop_n i5 n (L.set_at out_after (L.iter i5) (at tap i5));
      ]
  in
  let ctx = LL.empty_optimize_ctx () in
  let o_a =
    with_pinned_virtualize_settings (fun () -> L.optimize_in ctx ~name:"vslr_iud_a" llc_a)
  in
  let o_b =
    with_pinned_virtualize_settings (fun () -> L.optimize_in ctx ~name:"vslr_iud_b" llc_b)
  in
  p "inherited updated dep: routine A defers both tap and dep"
    (L.known_virtual o_a tap && L.known_virtual o_a dep);
  let got =
    L.execute ~name:"vslr_iud_b" o_b ~seed:[ (early, early_vals) ] ~read:[ out_before; out_after ]
  in
  let got_before, got_after = match got with [ a; b ] -> (a, b) | _ -> assert false in
  p "inherited updated dep: read before local update sees routine A's final dep (early + 16)"
    (exact got_before (Array.map early_vals ~f:(fun v -> v +. 16.)));
  p "inherited updated dep: read after local update sees routine B's dep (early + 100)"
    (exact got_after (Array.map early_vals ~f:(fun v -> v +. 100.)))

(* === Part 6: Codex P1 -- in-loop read before a later sibling initializes a materialized input
   === *)
let case_inloop_uninit_input_after_read () =
  let base = node "p1_base" in
  let dep = node "p1_dep" in
  let reader = node "p1_reader" in
  let late_x = node "p1_late_x" in
  let out_before = node "p1_out_before" in
  let out_after = node "p1_out_after" in
  L.materialize base;
  L.materialize late_x;
  L.materialize out_before;
  L.materialize out_after;
  let i1 = L.sym () and i2 = L.sym () and i3 = L.sym () in
  let at tn i = L.get tn [| L.iter i |] in
  let llc =
    LL.unflat_lines
      [
        L.loop_n i1 n (L.set_at dep (L.iter i1) (L.add (at base i1) (L.tick i1)));
        L.loop_n i2 n
          (LL.unflat_lines
             [
               L.set_at reader (L.iter i2) (L.add (at dep i2) (L.c 10.));
               L.set_at out_before (L.iter i2) (at reader i2);
               L.set_at late_x (L.iter i2) (L.ramp 100. i2);
               L.set_at dep (L.iter i2) (L.add (at late_x i2) (L.c 90.));
             ]);
        L.loop_n i3 n (L.set_at out_after (L.iter i3) (at reader i3));
      ]
  in
  let base_vals = Array.init n ~f:(fun i -> Float.of_int (5 + (2 * i))) in
  let o = with_pinned_virtualize_settings (fun () -> L.optimize ~name:"vslr_p1" llc) in
  p "in-loop late input: dep and reader stay virtual"
    (L.known_virtual o dep && L.known_virtual o reader);
  p "in-loop late input: late_x is not read before write" (not (L.read_before_write o late_x));
  let got =
    L.execute ~name:"vslr_p1" o
      ~seed:[ (base, base_vals); (late_x, L.blank n) ]
      ~read:[ out_before; out_after ]
  in
  let got_before, got_after = match got with [ a; b ] -> (a, b) | _ -> assert false in
  (* Producer-time before late_x: base[i] + (1 + i) + 10 = 16 + 3*i Replay-time after late_x: (100 +
     i) + 90 + 10 = 200 + i *)
  let expected_before = Array.init n ~f:(fun i -> Float.of_int (16 + (3 * i))) in
  let expected_after = Array.init n ~f:(fun i -> Float.of_int (200 + i)) in
  p "in-loop late input: out_before sees pre-loop dep and never reads uninitialized late_x"
    (exact got_before expected_before);
  p "in-loop late input: out_after sees updated dep after late_x is initialized"
    (exact got_after expected_after)

(* === Part 7: mid-write read between two setters of the same candidate in one shared loop === *)
let case_mid_write_read () =
  let late_x = node "mw_late_x" and a = node "mw_a" and out = node "mw_out" in
  let out_after = node "mw_out_after" in
  L.materialize late_x;
  L.materialize out;
  L.materialize out_after;
  let i1 = L.sym () and i2 = L.sym () in
  let at tn i = L.get tn [| L.iter i |] in
  let llc =
    LL.unflat_lines
      [
        L.loop_n i1 n
          (LL.unflat_lines
             [
               L.set_at a (L.iter i1) (L.ramp 11. i1);
               L.set_at out (L.iter i1) (at a i1);
               L.set_at late_x (L.iter i1) (L.ramp 100. i1);
               L.set_at a (L.iter i1) (L.add (at late_x i1) (L.c 50.));
             ]);
        L.loop_n i2 n (L.set_at out_after (L.iter i2) (at a i2));
      ]
  in
  let o = with_pinned_virtualize_settings (fun () -> L.optimize ~name:"vslr_mw" llc) in
  p "mid-write read: candidate materializes with 163"
    (Option.equal Tn.equal_provenance (L.rejection_code o a)
       (Some (Tn.Site "163:shared-loop-mid-write-read")));
  let got = L.execute ~name:"vslr_mw" o ~seed:[ (late_x, L.blank n) ] ~read:[ out; out_after ] in
  let got_mid, got_after = match got with [ a; b ] -> (a, b) | _ -> assert false in
  p "mid-write read: in-loop out sees first write (11 + i)"
    (exact got_mid (Array.init n ~f:(fun i -> Float.of_int (11 + i))));
  p "mid-write read: post-loop out_after sees second write (150 + i)"
    (exact got_after (Array.init n ~f:(fun i -> Float.of_int (150 + i))))

(* === Part 8: Codex P2 -- dead-arm declared-virtual cycle deferred during speculative capture
   === *)
let case_dead_arm_local_cycle () =
  let x = node "dac_x" and a = node "dac_a" and b = node "dac_b" in
  let w = node "dac_w" and out = node "dac_out" in
  L.materialize x;
  L.materialize out;
  L.virtualize a;
  let x_vals = Array.init n ~f:(fun i -> Float.of_int (30 + (4 * i))) in
  let i1 = L.sym () and i2 = L.sym () and i3 = L.sym () and i4 = L.sym () and i5 = L.sym () in
  let at tn i = L.get tn [| L.iter i |] in
  let llc =
    LL.unflat_lines
      [
        L.loop_n i1 n
          (L.seq
             (L.set_at a (L.iter i1) (L.add (at x i1) (L.c 1.)))
             (L.set_at b (L.iter i1) (L.add (at x i1) (L.c 2.))));
        L.loop_n i2 n
          (L.seq
             (L.set_at a (L.iter i2) (L.add (at b i2) (L.c 10.)))
             (L.set_at b (L.iter i2) (L.add (at x i2) (L.c 3.))));
        L.loop_n i3 n
          (L.seq
             (L.set_at b (L.iter i3) (L.add (at a i3) (L.c 20.)))
             (L.set_at a (L.iter i3) (L.add (at x i3) (L.c 4.))));
        L.loop_n i4 n
          (L.set_at w (L.iter i4)
             (L.where_ (L.lt (L.embed i4) (L.ic n)) (L.add (at x i4) (L.c 7.)) (at a i4)));
        L.loop_n i5 n (L.set_at out (L.iter i5) (at w i5));
      ]
  in
  let o = with_pinned_virtualize_settings (fun () -> L.optimize ~name:"vslr_dac" llc) in
  p "dead-arm local cycle: undecided capture falls back to 156 and compiles"
    (Option.equal Tn.equal_provenance (L.rejection_code o w)
       (Some (Tn.Site "156:capture-read-uninlinable")));
  let got = L.execute ~name:"vslr_dac" o ~seed:[ (x, x_vals) ] ~read:[ out ] in
  p "dead-arm local cycle: live arm executes to x + 7"
    (exact (List.hd_exn got) (Array.map x_vals ~f:(fun v -> v +. 7.)))

(* === Part 9: Codex P2 -- disjoint cell replay a[1,i] -> b[i] -> a[0,i] is not a cycle === *)
let case_disjoint_cell_replay () =
  let mk2 = L.node_factory ~first_id:2510900 ~dims:[| 2; n |] () in
  let x = node "dcr_x" and b = node "dcr_b" and out = node "dcr_out" in
  let a = mk2 "dcr_a" in
  L.materialize x;
  L.materialize out;
  L.virtualize a;
  let x_vals = Array.init n ~f:(fun i -> Float.of_int (40 + (3 * i))) in
  let i1 = L.sym () and i2 = L.sym () and i3 = L.sym () and i4 = L.sym () in
  let at1 tn i = L.get tn [| L.iter i |] in
  let at2 row i = L.get a [| L.fixed row; L.iter i |] in
  let set2 row i v = L.set a [| L.fixed row; L.iter i |] v in
  let llc =
    LL.unflat_lines
      [
        L.loop_n i1 n (set2 0 i1 (L.add (at1 x i1) (L.c 1.)));
        L.loop_n i2 n
          (L.seq
             (L.set_at b (L.iter i2) (L.add (at2 0 i2) (L.c 10.)))
             (set2 0 i2 (L.add (at1 x i2) (L.c 2.))));
        L.loop_n i3 n
          (L.seq
             (set2 1 i3 (L.add (at1 b i3) (L.c 20.)))
             (L.set_at b (L.iter i3) (L.add (at1 b i3) (L.c 1.))));
        L.loop_n i4 n (L.set_at out (L.iter i4) (at2 1 i4));
      ]
  in
  let o = with_pinned_virtualize_settings (fun () -> L.optimize ~name:"vslr_dcr" llc) in
  p "disjoint cell replay: both a and b stay virtual with no false cycle"
    (L.known_virtual o a && L.known_virtual o b && L.count_get o a = 0 && L.count_get o b = 0);
  let got = L.execute ~name:"vslr_dcr" o ~seed:[ (x, x_vals) ] ~read:[ out ] in
  (* a[0,i] replay = x[i] + 2; b[i] replay = (a[0,i] + 10) + 1 = x[i] + 13; a[1,i] = b[i] + 20 =
     x[i] + 33 *)
  p "disjoint cell replay: executed out matches closed form (x + 33)"
    (exact (List.hd_exn got) (Array.map x_vals ~f:(fun v -> v +. 33.)))

let () =
  case "local-shared-loop-downstream" (fun () ->
      case_local_shared_loop ~label:"local shared loop (downstream out)" ~tag:"lsd"
        ~in_loop_out:false ~observable_reader:false);
  case "local-shared-loop-inloop" (fun () ->
      case_local_shared_loop ~label:"local shared loop (in-loop out)" ~tag:"lsi" ~in_loop_out:true
        ~observable_reader:false);
  case "local-shared-loop-observable" (fun () ->
      case_local_shared_loop ~label:"local shared loop (observable reader)" ~tag:"lso"
        ~in_loop_out:false ~observable_reader:true);
  case "shared-vs-split" case_shared_vs_split;
  case "retained-chain" case_retained_chain;
  case "cyclic-retained-reads" case_cyclic_retained_reads;
  case "inherited-updated-dep" case_inherited_updated_dep;
  case "inloop-uninit-input-after-read" case_inloop_uninit_input_after_read;
  case "mid-write-read" case_mid_write_read;
  case "dead-arm-local-cycle" case_dead_arm_local_cycle;
  case "disjoint-cell-replay" case_disjoint_cell_replay;
  printf "%!"
