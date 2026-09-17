(* gh-ocannl-681: a [Local_scope] over a MATERIALIZED node is out of contract for
   [Low_level.optimize], and is REJECTED rather than silently normalized to a plain [Get].

   The shape is legal on the OTHER side of the pipeline: [Schedule]'s materializing [Unroll] and
   [Partition] mints, and [C_syntax.try_localize_serial_reduce], build exactly it over a
   materialized accumulator AFTER optimization, and codegen renders it. So one IR meant two
   different things depending on which side of [optimize] it was handed to, and nothing told a test
   author or a future transform writer which side they were on. The optimizer used to answer by
   discarding the body and reading the buffer instead, which is how two [accum_width.ml] legs ran
   kernels literally spelling [acc[0] = acc[0]] while claiming to pin an accumulation width: the
   identity copy happened to reproduce the expected value (256 stays 256), so they passed without
   executing what they claimed.

   The legs below are the contract. Materialized-accumulator localization belongs to codegen's
   accumulator peel (gh-ocannl-693), which this rejection makes the ONE route: until it lands, the
   only way to hand a backend the post-optimize scope form is past the optimizer, through the
   [?prelowered] seam that {!Ll_test.optimize_scoped} wraps -- leg 4. *)

open Base
module LL = Ir.Low_level
module Tn = Ir.Tnode
open Verdict.Claims

let () =
  let node = Ll_test.node_factory ~first_id:9800 ~dims:[| 8 |] () in
  (* Eight values that identify their iteration and stay off both the zero-init and the seed, so a
     dropped, replayed or reordered iteration changes the sum: 1 + 2 + ... + 8 = 36. *)
  let xs_values = Array.init 8 ~f:(fun k -> Float.of_int (k + 1)) in
  let seeded = 100.0 in
  let accumulated = 136.0 in
  (* === the shape that used to collapse === *)
  let acc = node ~dims:[| 1 |] "som_acc" in
  let xs = node "som_xs" in
  Ll_test.materialize acc;
  Ll_test.materialize xs;
  let i = Ll_test.sym () in
  let id = LL.get_scope acc in
  let body =
    LL.Seq
      ( LL.Set_local (id, Ll_test.get acc [| Ll_test.fixed 0 |]),
        Ll_test.loop_n i 8
          (LL.Set_local (id, Ll_test.add (LL.Get_local id) (Ll_test.get xs [| Ll_test.iter i |])))
      )
  in
  let scoped mint =
    Ll_test.set acc
      [| Ll_test.fixed 0 |]
      (LL.Local_scope { id; body; orig_indices = [| Ll_test.fixed 0 |]; mint })
  in
  let rejection_of mint =
    match Ll_test.optimize ~materialized:[ acc; xs ] ~name:"som_reject" (scoped mint) with
    | (_ : LL.optimized) -> None
    | exception Invalid_argument msg -> Some msg
  in
  let rejection = rejection_of LL.Inlined_computation in
  p "optimizing a Local_scope over a materialized node is refused" (Option.is_some rejection);
  p "the refusal names the node"
    (Option.value_map rejection ~default:false ~f:(fun msg ->
         String.is_substring msg ~substring:(Tn.debug_name acc)));
  (* gh-ocannl-687 added a mint field to [Local_scope], and it is deliberately NOT what decides this
     rejection. The mint records which pass BUILT a scope; this contract is about which side of
     [optimize] a program was handed to, a per-call fact that {!Low_level.input_scope_ids} answers.
     Claiming the schedule's provenance must not buy a program past the optimizer -- were the two
     conflated, hand-built IR (which has no honest way to spell "not mine") could label its way back
     into the silent collapse this rejection closed. *)
  p "claiming the schedule mint does not exempt a program handed to optimize"
    (Option.is_some (rejection_of LL.Schedule_minted));
  (* The refusal is about the SCOPE, not about the computation: the same accumulation spelled
     without one optimizes and executes. This is also the raw twin the prelowered leg needs. *)
  let raw =
    Ll_test.loop_n i 8
      (Ll_test.set acc
         [| Ll_test.fixed 0 |]
         (Ll_test.add (Ll_test.get acc [| Ll_test.fixed 0 |]) (Ll_test.get xs [| Ll_test.iter i |])))
  in
  let raw_vals =
    let o = Ll_test.optimize ~materialized:[ acc; xs ] ~name:"som_raw" raw in
    Ll_test.execute ~name:"som_raw" o ~seed:[ (acc, [| seeded |]); (xs, xs_values) ] ~read:[ acc ]
  in
  p "the same accumulation without the scope optimizes and executes"
    (Float.equal (List.hd_exn raw_vals).(0) accumulated);
  (* === negative control: a scope over a VIRTUAL node is the legal form === *)
  (* [out.(r) = sum_c src.(r, c)], with the reduction held in a scope local over a virtual node --
     the shape the virtualizer itself mints. The producer values identify BOTH symbols
     ([1 + 10*r + c]), so a substitution that drops [r] or [c], or a range that over- or
     under-covers, changes the row sums; [out] is seeded with the sentinel, so an uncovered row
     fails rather than reading whatever the buffer held. *)
  let out = node ~dims:[| 4 |] "som_out" in
  let src = node ~dims:[| 4; 3 |] "som_src" in
  let tmp = node ~dims:[| 1 |] "som_tmp" in
  Ll_test.materialize out;
  Ll_test.materialize src;
  Ll_test.virtualize tmp;
  let r = Ll_test.sym () in
  let c = Ll_test.sym () in
  let vid = LL.get_scope tmp in
  let vbody =
    LL.Seq
      ( LL.Set_local (vid, Ll_test.c 0.0),
        Ll_test.loop_n c 3
          (LL.Set_local
             ( vid,
               Ll_test.add (LL.Get_local vid) (Ll_test.get src [| Ll_test.iter r; Ll_test.iter c |])
             )) )
  in
  let vllc =
    Ll_test.loop_n r 4
      (Ll_test.set out
         [| Ll_test.iter r |]
         (LL.Local_scope
            {
              id = vid;
              body = vbody;
              orig_indices = [| Ll_test.fixed 0 |];
              mint = LL.Inlined_computation;
            }))
  in
  let vo = Ll_test.optimize ~materialized:[ out; src ] ~name:"som_virtual" vllc in
  p "a Local_scope over a virtual node survives optimization" (Ll_test.count_scopes vo.LL.llc = 1);
  let src_values =
    Array.init 12 ~f:(fun k -> 1.0 +. (10.0 *. Float.of_int (k / 3)) +. Float.of_int (k % 3))
  in
  let vvals =
    Ll_test.execute ~name:"som_virtual" vo
      ~seed:[ (out, Ll_test.blank 4); (src, src_values) ]
      ~read:[ out ]
  in
  p "the virtual-node scope executes to the reference row sums"
    (Ll_test.same vvals [ [| 6.0; 36.0; 66.0; 96.0 |] ]);
  (* === the supported route for the post-optimize form === *)
  (* The honest provenance for this shape: it is what a materializing [Unroll] mints. *)
  let so =
    Ll_test.optimize_scoped ~materialized:[ acc; xs ] ~name:"som_prelowered" ~raw
      (scoped LL.Schedule_minted)
  in
  let svals =
    Ll_test.execute ~name:"som_prelowered" so
      ~seed:[ (acc, [| seeded |]); (xs, xs_values) ]
      ~read:[ acc ]
  in
  p "the prelowered seam still delivers the materialized-accumulator scope"
    (Float.equal (List.hd_exn svals).(0) accumulated)

(* === the one exemption: a scope the pass MINTED, retracted by the pass itself === *)

(* gh-ocannl-704. The scope-target rejection carries exactly one exemption, and the exemption is the
   only reason [cleanup_virtual_llc] takes [~input_scopes] and [specialize_proc] computes
   [input_scope_ids]. [virtual_llc] mints a scope at a [Get] of a node still virtual at that point;
   a later refusal can commit that node [Never_virtual]; cleanup then rewrites the scope back to a
   plain [Get], which is sound because the setter that now survives writes the very value the body
   recomputed. Nothing in the suite exercised that, so the question was whether the branch is
   reachable at all -- if not, the exemption, the id threading and one concept could go.

   It IS reachable, and reachable structurally rather than by accident. [virtual_llc] walks
   statements in SOURCE ORDER while a node's placement is a single mutable cell shared by the whole
   walk, and both rejection families -- store time ([check_and_store_virtual]) and consumption time
   ([inline_computation]) -- can be triggered by a statement the walk reaches AFTER a read that has
   already minted. Nothing re-visits the minted scope in between, so cleanup meets it over a node
   that is now materialized. The two witnesses below are one from each family; with the exemption
   removed, both are refused, which is the point: the branch is not dead code.

   Both are hand-built, which is the honest standing of the exemption: it is not conservatism about
   a hypothetical, it is what keeps [LL.optimize] from refusing IR it produced itself -- but no
   ordinary [Assignments] lowering in the suite has been observed to reach it (an instrumented build
   of the gh-ocannl-681 PR found hits only on out-of-contract INPUT scopes, and one repeated here
   over the targeted virtualization tests found none at all). *)

(* Both witnesses are about LEGALITY, and every heuristic cap in [decide_placements] runs BEFORE any
   legality question is asked -- so an ambient policy setting can materialize either node ahead of
   the refusal under test (provenance 1, 39 or 41 instead of 142 or 13) and leave the legs asserting
   about a program the virtualizer never entered. The read-modify-write exemption matters the same
   way: with [inline_complex_computations] off, witness 1's accumulating producer counts its own
   self-read as a visit and trips the visit cap. So EVERY field of [virtualize_settings] is pinned
   for the duration and restored, rather than the one field a particular default happens to make
   load-bearing. Pinning at runtime is also what keeps this test independent of the configuration it
   runs under: no [OCANNL_VIRTUALIZE_*] override can change its reading, so the stanza has no
   virtualization env keys to declare. *)
let with_virtualize_settings ~max_visits f =
  let s = LL.virtualize_settings in
  let saved =
    ( s.LL.enable_device_only,
      s.LL.max_visits,
      s.LL.max_inline_reduction,
      s.LL.max_inline_fanin,
      s.LL.inline_scalar_constexprs,
      s.LL.inline_simple_computations,
      s.LL.inline_complex_computations )
  in
  s.LL.enable_device_only <- true;
  s.LL.max_visits <- max_visits;
  (* The two chain caps disabled outright: neither witness is about recompute cost, and a strict
     ambient value would materialize the producer before the refusal under test. *)
  s.LL.max_inline_reduction <- -1;
  s.LL.max_inline_fanin <- -1;
  s.LL.inline_scalar_constexprs <- true;
  (* Off, so the visit cap is always consulted: the [max_visits] pinned above is then the whole
     policy story, with no bypass deciding a leg behind its back. *)
  s.LL.inline_simple_computations <- false;
  s.LL.inline_complex_computations <- true;
  Exn.protect ~f ~finally:(fun () ->
      let edo, mv, mir, mif, isc, isi, ici = saved in
      s.LL.enable_device_only <- edo;
      s.LL.max_visits <- mv;
      s.LL.max_inline_reduction <- mir;
      s.LL.max_inline_fanin <- mif;
      s.LL.inline_scalar_constexprs <- isc;
      s.LL.inline_simple_computations <- isi;
      s.LL.inline_complex_computations <- ici)

let () =
  (* Witness 1, store time. [v] is produced, read, and only THEN conditionally overwritten. The
     guarded setter is rejected as [Non_virtual 142] (gh-ocannl-651: a candidate whose nest sits
     under an [If] would replay unguarded at every read site), which materializes [v] -- after the
     consumer's read has been inlined into a minted scope. The producer writes [1 + i], so a
     replayed, dropped or reordered iteration changes the reading, and no cell can be confused with
     the zero-init or the sentinel. Each cell is read once, so the pinned [max_visits = 1] (the
     default) is what the shape needs. *)
  with_virtualize_settings ~max_visits:1 @@ fun () ->
  let node = Ll_test.node_factory ~first_id:9850 ~dims:[| 4 |] () in
  let v = node "smr_v" in
  let out = node "smr_out" in
  let flag = node ~dims:[| 1 |] "smr_flag" in
  Ll_test.materialize out;
  Ll_test.materialize flag;
  let producer =
    let i = Ll_test.sym () in
    Ll_test.seq (Ll_test.zero v)
      (Ll_test.loop_n i 4
         (Ll_test.set v
            [| Ll_test.iter i |]
            (Ll_test.add (Ll_test.get v [| Ll_test.iter i |]) (Ll_test.tick i))))
  in
  let consumer =
    let k = Ll_test.sym () in
    Ll_test.loop_n k 4 (Ll_test.set out [| Ll_test.iter k |] (Ll_test.get v [| Ll_test.iter k |]))
  in
  let overwrite =
    let i = Ll_test.sym () in
    Ll_test.if_
      (Ll_test.get flag [| Ll_test.fixed 0 |])
      (Ll_test.loop_n i 4 (Ll_test.set v [| Ll_test.iter i |] (Ll_test.c 9.0)))
  in
  (* The control is the same program MINUS the guarded setter: it is what pins that the consumer's
     read really does mint a scope. [v] ends virtual with not one read of its buffer left, so the
     read was served by an inlined computation and by nothing else. *)
  let control =
    Ll_test.optimize ~materialized:[ out; flag ] ~name:"smr_control" (Ll_test.seq producer consumer)
  in
  p "without the later refusal the read site inlines the producer"
    (Ll_test.known_virtual control v && Ll_test.count_get control v = 0);
  let retracted =
    Ll_test.optimize ~materialized:[ out; flag ] ~name:"smr_retract"
      (Ll_test.seq producer (Ll_test.seq consumer overwrite))
  in
  (* Provenance 142 is written by [check_and_store_virtual], which only runs on a node still
     undecided -- so [v] was a live virtualization candidate for the whole of the consumer
     statement, and became materialized only at the appended one. *)
  p "appending the guarded setter commits the already-inlined node Never_virtual"
    (Option.equal Tn.equal_provenance
       (Ll_test.rejection_code retracted v)
       (Some (Tn.Site "142:guarded-computation")));
  p "the pass retracts the scope it minted rather than refusing its own program"
    (Ll_test.count_scopes retracted.LL.llc = 0 && Ll_test.count_get retracted v > 0);
  (* Executed parity against the materialized reading of the same program: [v] declared materialized
     up front, so no scope is ever minted and the consumer reads the buffer all along. Every
     execution gets a routine name of its own -- a second compile under one name overwrites the
     first's debug artifacts, so a failing leg would otherwise be debugged against another leg's
     IR. *)
  let reference =
    Ll_test.optimize ~materialized:[ out; flag; v ] ~name:"smr_reference"
      (Ll_test.seq producer (Ll_test.seq consumer overwrite))
  in
  let run_at o name f reads =
    Ll_test.execute ~name o ~seed:[ (out, Ll_test.blank 4); (flag, [| f |]) ] ~read:reads
  in
  let produced = [| 1.; 2.; 3.; 4. |] in
  let retracted_off = List.hd_exn (run_at retracted "smr_retract_guard_off" 0.0 [ out ]) in
  let retracted_on = List.hd_exn (run_at retracted "smr_retract_guard_on" 1.0 [ out ]) in
  let reference_off = List.hd_exn (run_at reference "smr_reference_guard_off" 0.0 [ out ]) in
  (* [v] is readable only in the reference arm -- the retracted arm places it [Local]. *)
  let reference_on = run_at reference "smr_reference_guard_on" 1.0 [ out; v ] in
  p "with the guard off, the retracted read agrees with the materialized reading"
    (Ll_test.close retracted_off produced && Ll_test.close reference_off produced);
  p "with the guard on, the retracted read agrees with the materialized reading"
    (Ll_test.close retracted_on produced && Ll_test.close (List.hd_exn reference_on) produced);
  (* ... and the guard is not vacuous: the overwrite really lands, after the consumer, which is why
     both readings above are the produced values rather than nines. *)
  p "the guarded setter does run when the flag is on"
    (Ll_test.close (List.nth_exn reference_on 1) [| 9.; 9.; 9.; 9. |]);
  (* The exemption keys on which SIDE of this [optimize] call a scope came from, not on its shape:
     spell the consumer's read as an equivalent [Local_scope] with the honest [Inlined_computation]
     mint, and the same program is rejected, because [input_scope_ids] recorded that scope at
     entry. *)
  let handed =
    let k = Ll_test.sym () in
    let hid = LL.get_scope v in
    let hbody =
      LL.Seq
        ( LL.Set_local (hid, Ll_test.c 0.0),
          LL.Set_local (hid, Ll_test.add (LL.Get_local hid) (Ll_test.tick k)) )
    in
    Ll_test.seq producer
      (Ll_test.seq
         (Ll_test.loop_n k 4
            (Ll_test.set out
               [| Ll_test.iter k |]
               (LL.Local_scope
                  {
                    id = hid;
                    body = hbody;
                    orig_indices = [| Ll_test.iter k |];
                    mint = LL.Inlined_computation;
                  })))
         overwrite)
  in
  p "an equivalent scope HANDED to the same optimize call is still rejected"
    (match Ll_test.optimize ~materialized:[ out; flag ] ~name:"smr_handed" handed with
    | (_ : LL.optimized) -> false
    | exception Invalid_argument msg -> String.is_substring msg ~substring:(Tn.debug_name v))

let () =
  (* Witness 2, consumption time. The other rejection family reaches the same branch: a producer
     that writes a FIXED cell can be inlined at a read of that cell and cannot be inlined at a read
     of the same cell spelled through a (width-one) iterator -- [inline_computation] has no symbol
     to bind and the static positions disagree, which is [Non_virtual 13]. Put the matching read
     first and the mismatching one second, and the second read materializes a node the first has
     already inlined into a minted scope.

     The shape needs the SAME cell read twice, so [max_visits] is pinned to 2 here: the cap is a
     policy prior decided before any legality question, not the mechanism under test. *)
  with_virtualize_settings ~max_visits:2 @@ fun () ->
  let node = Ll_test.node_factory ~first_id:9880 ~dims:[| 2 |] () in
  let w = node "smr2_w" in
  let src = node "smr2_src" in
  let oa = node "smr2_oa" in
  let ob = node "smr2_ob" in
  List.iter [ src; oa; ob ] ~f:Ll_test.materialize;
  (* [3 + 10*4 = 43] identifies both source cells, so a producer inlined against the wrong operand,
     or a read served from an uninitialized buffer, cannot reproduce it. *)
  let src_values = [| 3.; 4. |] in
  let produced = 43. in
  let producer =
    Ll_test.set w
      [| Ll_test.fixed 0 |]
      (Ll_test.add
         (Ll_test.get src [| Ll_test.fixed 0 |])
         (Ll_test.mul (Ll_test.c 10.) (Ll_test.get src [| Ll_test.fixed 1 |])))
  in
  let matching_read = Ll_test.set oa [| Ll_test.fixed 0 |] (Ll_test.get w [| Ll_test.fixed 0 |]) in
  let iterator_read =
    let j = Ll_test.sym () in
    Ll_test.loop_n j 1 (Ll_test.set ob [| Ll_test.fixed 0 |] (Ll_test.get w [| Ll_test.iter j |]))
  in
  let control =
    Ll_test.optimize ~materialized:[ src; oa; ob ] ~name:"smr2_control"
      (Ll_test.seq producer matching_read)
  in
  p "the matching read alone inlines the fixed-cell producer"
    (Ll_test.known_virtual control w && Ll_test.count_get control w = 0);
  let retracted =
    Ll_test.optimize ~materialized:[ src; oa; ob ] ~name:"smr2_retract"
      (Ll_test.seq producer (Ll_test.seq matching_read iterator_read))
  in
  p "a later unservable read commits the already-inlined node Never_virtual"
    (Option.equal Tn.equal_provenance
       (Ll_test.rejection_code retracted w)
       (Some (Tn.Site "13:call-site-index-mismatch")));
  p "consumption-time refusal retracts the minted scope too, rather than refusing the program"
    (Ll_test.count_scopes retracted.LL.llc = 0 && Ll_test.count_get retracted w > 0);
  let reference =
    Ll_test.optimize ~materialized:[ src; oa; ob; w ] ~name:"smr2_reference"
      (Ll_test.seq producer (Ll_test.seq matching_read iterator_read))
  in
  let read_both o name =
    Ll_test.execute ~name o
      ~seed:[ (src, src_values); (oa, Ll_test.blank 2); (ob, Ll_test.blank 2) ]
      ~read:[ oa; ob ]
  in
  let both_cells vals =
    (not (List.is_empty vals)) && List.for_all vals ~f:(fun a -> Float.equal a.(0) produced)
  in
  p "the retracted read and the materialized reading deliver the same value to both consumers"
    (both_cells (read_both retracted "smr2_retract_run")
    && both_cells (read_both reference "smr2_reference_run"))
