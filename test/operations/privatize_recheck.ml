(* [Schedule.Privatize]'s tile width, judged through whole schedules (follow-ups of gh-ocannl-1151).
   All legs run under [Bf16_wide], so the accumulator residency widens bf16 to f32 on every backend,
   over bf16 accumulators seeded at 256, where bf16's spacing is 2: each outer iteration adds 1.25
   in steps below 1, so a program narrowing once gives 258 (256 + 2.5), one narrowing per outer
   iteration gives 260 (257.25 rounds up, twice), and one narrowing per step stays at 256. The
   inputs are dyadic, so every partial sum is exact in f32 whatever the order.

   1. The final re-check ([Schedule.privatized_tile_widens]). A wide tile fails it only when a later
   op puts, into the scope a materializing [Unroll] minted around the tile's update, something that
   scope's grammar excludes: a hardware-typed loop ([Retype] of a loop inside it) or a tensor-node
   write ([Stage]'s loads landing inside it). Both legs drive that through [Schedule.apply] and pin
   the trigger (the tile is judged non-widening), the seam (the pair is recorded in the
   [privatize_fallbacks] of the domain that applied it, and of no other) and the drop (the tile is
   back at bf16 storage). Both programs are ones compilation refuses -- a hardware-annotated loop
   inside a [Local_scope] body (the renderer's hardware-axis table holds none, and
   [Low_level.validate_parallel] rejects it), a tensor-node write inside one
   ([Low_level.validate_scope_bodies], gh-ocannl-584) -- and so is each one's twin without
   [Privatize]: the fallback reaches no executed value. The control is the same schedule without the
   breaking op: the tile stays wide and executes to the entry program's once-narrowed 258.

   2. Provenance for loops a [Split] mints. The tile narrows once per execution of [over], which
   matches the serial rendering of the program [apply] was handed only where one execution of [over]
   covers exactly the loop that rendering localizes the write at. The outer half of a [Split] of
   that loop does (258, on an f32 tile). The inner half narrows once per outer iteration -- 260,
   matching neither the entry program (258) nor the split schedule without [Privatize] (256) -- so
   its tile stays at storage and reproduces the latter; without the record it widened. So does the
   outer half of a split of a loop INSIDE the localizing level.

   3. Provenance for loops a [Coalesce] or a [Partition] mints. The loop a [Coalesce] merges from
   the localizing loop and the one inside it covers that loop, and so does the outer half of a
   [Split] of it (258, on an f32 tile); the inner half of that [Split] covers only part of it, and
   keeps storage (256). A segment of a [Partition] of the localizing loop runs only part of it, and
   keeps storage too (256). Without the record both storage legs widened (260 and 260).

   4. Provenance for loops a [Split_reduce] mints. Its pass-1 pair is a [Split] of the reduction
   loop: one execution of the block loop runs every iteration of it, the in-block loop only one
   block's chunk. The split target's own accumulation moves into the minted partials node, which the
   program [apply] was handed lacks, so a [Privatize] of it is judged by nothing and keeps storage;
   the judgment reaches a minted loop through ANOTHER node accumulated in the same statement, whose
   write [Split_reduce] re-indexes but does not redirect. That write shares the reduction loop's
   body with the target's, so the serial rendering localizes it at a loop inside the reduction loop
   and never at the reduction loop itself: a [Privatize] over the block loop or the in-block loop
   keeps storage (256, as without [Privatize]). Without the record both widened (258 and 260). *)

open Base
module Tn = Ir.Tnode
module LL = Ir.Low_level
module Sched = Ir.Schedule
module Numerics = Ir.Numerics
open Verdict.Claims

let bf16 = Ir.Ops.bfloat16
let bin op a b = LL.Binop (op, (a, bf16), (b, bf16))
let it = Ll_test.iter
let pz ~target ~over = Sched.privatize ~accum_prec:Ll_test.backend_accum_prec ~target ~over
let floats l = String.concat ~sep:"," (List.map l ~f:Float.to_string)

(* The privatized tiles of a program: the tile-namespace nodes it accumulates into. *)
let tiles (llc : LL.t) =
  let acc = ref [] in
  Ll_test.walk llc ~on_stmt:(function
    | LL.Set { tn; llsc; _ }
      when String.equal tn.Tn.namespace "tile"
           (* [Split_reduce]'s partials share the namespace; they are no [Privatize] tile. *)
           && (not (List.mem tn.Tn.label "partials" ~equal:String.equal))
           && LL.scalar_touches_tn tn llsc
           && not (List.mem !acc tn ~equal:Tn.equal) ->
        acc := tn :: !acc
    | _ -> ());
  !acc

let only_prec prec llc =
  let precs = List.map (tiles llc) ~f:(fun tn -> Lazy.force tn.Tn.storage_prec) in
  List.equal Ir.Ops.equal_prec (List.dedup_and_sort precs ~compare:Ir.Ops.compare_prec) [ prec ]

let exec ~name ~acc ~init ~x ~xs o =
  Ll_test.execute ~name o ~seed:[ (acc, init); (x, xs) ] ~read:[ acc ]
  |> List.hd_exn |> Array.to_list

let claim_control =
  "without a breaking op the privatized tile stays f32, passes the re-check, leaves the fallback \
   log unchanged, and executes to the entry program's once-narrowed value (258)"

let claim_trigger_wgr =
  "a Workgroup_reduce Retype inside the scope a materializing Unroll minted around a privatized \
   tile's update leaves an accumulation the final re-check does not widen"

let claim_seam_wgr =
  "privatize_fallbacks is per domain: the same trigger run in a second domain records its own \
   fallback there, each domain's log holds exactly its own, and this one's is left unchanged"

let claim_drop_wgr = "that tile falls back to bf16 storage precision"

let claim_trigger_stage =
  "a Stage whose loads land inside that scope trips the re-check too: the fallback is recorded and \
   the tile is at bf16 storage"

let claim_uncompilable =
  "compilation refuses both fallen-back programs and their twins without Privatize, so the \
   fallback reaches no executed value"

let claim_split_outer =
  "a Privatize over the outer half of a Split of the localizing loop widens like that loop (258 \
   with the serial rendering, on an f32 tile)"

let claim_split_inner =
  "a Privatize over the inner half of that Split keeps a bf16 storage tile and the split \
   schedule's own narrowing (256, as without Privatize)"

let claim_split_inside =
  "a Privatize over the outer half of a Split of a loop inside the localizing level keeps a bf16 \
   storage tile and the schedule's own narrowing (256, as without Privatize)"

let claim_coalesce =
  "a Privatize over the loop a Coalesce merged from the localizing loop and the loop inside it, \
   and one over the outer half of a Split of that merged loop, widen like the localizing loop (258 \
   with the serial rendering, on an f32 tile)"

let claim_coalesce_inner =
  "a Privatize over the inner half of that Split of the merged loop keeps a bf16 storage tile and \
   the split schedule's own narrowing (256, as without Privatize)"

let claim_partition =
  "a Privatize over a segment of a Partition of the localizing loop keeps a bf16 storage tile and \
   the partitioned schedule's own narrowing (256, as without Privatize)"

let claim_split_reduce_entry =
  "the program the Split_reduce legs schedule localizes the node accumulated beside the \
   Split_reduce target at its own inner loop, widened there"

let claim_split_reduce =
  "a Privatize of that node over a Split_reduce's block loop, or over its in-block loop, keeps a \
   bf16 storage tile and the split schedule's own narrowing (256, as without Privatize)"

(* === The final re-check === *)

(* [r { s { t { acc[0] += x[r, s, t] } } }], localized at [r]; each [r] iteration adds 1.25. *)
let xs_rst = [| 0.125; 0.375; 0.5; 0.25; 0.25; 0.5; 0.375; 0.125 |]
let node = Ll_test.node_factory ~prec:bf16 ~first_id:14000 ~dims:[| 1 |] ()

let build_rst name =
  let acc = node (name ^ "_acc") and x = node ~dims:[| 2; 2; 2 |] (name ^ "_x") in
  Ll_test.materialize acc;
  Ll_test.materialize x;
  let r = Ll_test.sym () and s = Ll_test.sym () and t = Ll_test.sym () in
  let c0 = [| Ll_test.fixed 0 |] in
  let raw =
    Ll_test.loop_n r 2
      (Ll_test.loop_n s 2
         (Ll_test.loop_n t 2
            (Ll_test.set acc c0
               (bin Ir.Ops.Add (Ll_test.get acc c0) (Ll_test.get x [| it r; it s; it t |])))))
  in
  (Ll_test.optimize ~materialized:[ acc; x ] ~name raw, acc, x, r, s, t)

(* [r { s { u { t { acc[0] += x[r, s, u, t] } } } }] with [s] of extent 1: the materializing
   [Unroll] of [s] leaves the [u] loop inside the minted scope, and a [Stage] of [x] over [t] puts
   its loads in [u]'s body. *)
let build_rsut name =
  let acc = node (name ^ "_acc") and x = node ~dims:[| 2; 1; 2; 2 |] (name ^ "_x") in
  Ll_test.materialize acc;
  Ll_test.materialize x;
  let r = Ll_test.sym () and s = Ll_test.sym () and u = Ll_test.sym () and t = Ll_test.sym () in
  let c0 = [| Ll_test.fixed 0 |] in
  let raw =
    Ll_test.loop_n r 2
      (Ll_test.loop_n s 1
         (Ll_test.loop_n u 2
            (Ll_test.loop_n t 2
               (Ll_test.set acc c0
                  (bin Ir.Ops.Add (Ll_test.get acc c0) (Ll_test.get x [| it r; it s; it u; it t |]))))))
  in
  (Ll_test.optimize ~materialized:[ acc; x ] ~name raw, acc, x, r, s, t)

let stage_over x t =
  Sched.Stage
    {
      source = x;
      tile_loops = [ t ];
      shared = false;
      cooperative = None;
      hoisted = false;
      swizzle = None;
      pad_stride = None;
      pipeline_depth = 1;
      tile_prec = None;
    }

let unroll s = Sched.Unroll { axis = s; materialize = true }

(* Applies [sched], returning the program and the fallbacks this apply recorded in this domain. *)
let apply_logged sched o =
  let before = List.length (Sched.privatize_fallbacks ()) in
  let scheduled = Sched.apply sched o in
  let after = Sched.privatize_fallbacks () in
  (scheduled, List.take after (List.length after - before))

(* Compilation refuses the program for a hardware-annotated loop or a tensor-node write inside a
   scope body, by whichever check meets it first: the renderer's hardware-axis table or
   [validate_parallel] for the one, [validate_scope_bodies] for the other. *)
let refuses_compile ~name ~because o =
  match Ll_test.run ~name o ~seed:[] with
  | _ -> false
  | exception Invalid_argument msg ->
      Stdio.eprintf "privatize_recheck: %s refused at compile: %s (not part of the golden)\n%!" name
        msg;
      String.is_substring msg ~substring:because

let fallback_of ~acc log =
  match log with [ (target, tile) ] when Tn.equal target acc -> Some tile | _ -> None

let same_pairs a b = List.equal (fun (a1, t1) (a2, t2) -> Tn.equal a1 a2 && Tn.equal t1 t2) a b

(* The [Workgroup_reduce] trigger on fresh nodes: the program, its accumulator, and the fallbacks
   this apply recorded. *)
let wgr_trigger name =
  let o, acc, _, r, s, t = build_rst name in
  let scheduled, log =
    apply_logged
      [ pz ~target:acc ~over:r; unroll s; Sched.Retype { axis = t; ty = LL.Workgroup_reduce } ]
      o
  in
  (scheduled, acc, log)

let recheck_legs () =
  let wgr, wgr_acc, wgr_log = wgr_trigger "prc_wgr" in
  let wgr_tile = fallback_of ~acc:wgr_acc wgr_log in
  p claim_trigger_wgr
    (match wgr_tile with
    | Some tile ->
        List.mem (tiles wgr.LL.llc) tile ~equal:Tn.equal
        && not (Sched.privatized_tile_widens ~tile wgr.LL.llc)
    | None -> false);
  (* The seam: a second domain runs the same trigger on nodes of its own. Each domain's log holds
     exactly its own fallback, and this domain's is unchanged by the other's. *)
  (let here = Sched.privatize_fallbacks () in
   let there, there_pair =
     Stdlib.Domain.join
       (Stdlib.Domain.spawn (fun () ->
            let _, acc, log = wgr_trigger "prc_wgr_domain" in
            ( Sched.privatize_fallbacks (),
              Option.map (fallback_of ~acc log) ~f:(fun tile -> (acc, tile)) )))
   in
   p claim_seam_wgr
     (match (wgr_tile, there_pair) with
     | Some tile, Some pair ->
         same_pairs wgr_log [ (wgr_acc, tile) ]
         && same_pairs there [ pair ]
         && same_pairs (Sched.privatize_fallbacks ()) here
     | _ -> false));
  p claim_drop_wgr
    (match wgr_tile with
    | Some tile ->
        Ir.Ops.equal_prec (Lazy.force tile.Tn.storage_prec) bf16 && only_prec bf16 wgr.LL.llc
    | None -> false);
  let stg =
    let o, acc, x, r, s, t = build_rsut "prc_stage" in
    let stg, log = apply_logged [ pz ~target:acc ~over:r; unroll s; stage_over x t ] o in
    p claim_trigger_stage
      (match fallback_of ~acc log with
      | Some tile ->
          (not (Sched.privatized_tile_widens ~tile stg.LL.llc))
          && Ir.Ops.equal_prec (Lazy.force tile.Tn.storage_prec) bf16
      | None -> false);
    stg
  in
  let wgr_twin =
    let o, _, _, _, s, t = build_rst "prc_wgr_twin" in
    Sched.apply [ unroll s; Sched.Retype { axis = t; ty = LL.Workgroup_reduce } ] o
  in
  let stg_twin =
    let o, _, x, _, s, t = build_rsut "prc_stage_twin" in
    Sched.apply [ unroll s; stage_over x t ] o
  in
  let hw = "hardware-annotated loop" and scope = "validate_scope_bodies" in
  p claim_uncompilable
    (refuses_compile ~name:"prc_wgr" ~because:hw wgr
    && refuses_compile ~name:"prc_wgr_twin" ~because:hw wgr_twin
    && refuses_compile ~name:"prc_stage" ~because:scope stg
    && refuses_compile ~name:"prc_stage_twin" ~because:scope stg_twin);
  (* The control: the same tile with nothing breaking its scope, after the triggers, so the log it
     must leave unchanged is not empty. *)
  let serial =
    let o, acc, x, _, _, _ = build_rst "prc_serial" in
    exec ~name:"prc_serial" ~acc ~init:[| 256.0 |] ~x ~xs:xs_rst o
  in
  let o, acc, x, r, s, _ = build_rst "prc_ctl" in
  let before = Sched.privatize_fallbacks () in
  let ctl = Sched.apply [ pz ~target:acc ~over:r; unroll s ] o in
  let after = Sched.privatize_fallbacks () in
  let values = exec ~name:"prc_ctl" ~acc ~init:[| 256.0 |] ~x ~xs:xs_rst ctl in
  Stdio.eprintf "privatize_recheck: serial %s, control %s (not part of the golden)\n%!"
    (floats serial) (floats values);
  p claim_control
    ((match tiles ctl.LL.llc with
       | [ tile ] ->
           Sched.privatized_tile_widens ~tile ctl.LL.llc
           && Ir.Ops.equal_prec (Lazy.force tile.Tn.storage_prec) Ir.Ops.single
       | _ -> false)
    && (not (List.is_empty before))
    && same_pairs after before
    && List.equal Float.equal serial [ 258.0 ]
    && List.equal Float.equal values serial)

(* === Provenance for Split-minted loops === *)

(* [x] varies along every axis; each [k] iteration pair (and each [k] row of [build_ikm]) adds 1.25
   to its cell. *)
let xs_ik = [| 0.5; 0.75; 0.25; 1.0; 0.75; 0.5; 1.0; 0.25 |]
let node2 = Ll_test.node_factory ~prec:bf16 ~first_id:14100 ~dims:[| 2 |] ()

(* [i { k { acc[i] += x[i, k] } }], localized at [k]. *)
let build_ik name =
  let acc = node2 (name ^ "_acc") and x = node2 ~dims:[| 2; 4 |] (name ^ "_x") in
  Ll_test.materialize acc;
  Ll_test.materialize x;
  let i = Ll_test.sym () and k = Ll_test.sym () in
  let raw =
    Ll_test.loop_n i 2
      (Ll_test.loop_n k 4
         (Ll_test.set acc
            [| it i |]
            (bin Ir.Ops.Add (Ll_test.get acc [| it i |]) (Ll_test.get x [| it i; it k |]))))
  in
  (Ll_test.optimize ~materialized:[ acc; x ] ~name raw, acc, x, i, k)

(* [i { k { m { acc[i] += x[i, k, m] } } }], localized at [k]. *)
let build_ikm name =
  let acc = node2 (name ^ "_acc") and x = node2 ~dims:[| 2; 2; 2 |] (name ^ "_x") in
  Ll_test.materialize acc;
  Ll_test.materialize x;
  let i = Ll_test.sym () and k = Ll_test.sym () and m = Ll_test.sym () in
  let raw =
    Ll_test.loop_n i 2
      (Ll_test.loop_n k 2
         (Ll_test.loop_n m 2
            (Ll_test.set acc
               [| it i |]
               (bin Ir.Ops.Add (Ll_test.get acc [| it i |]) (Ll_test.get x [| it i; it k; it m |])))))
  in
  (Ll_test.optimize ~materialized:[ acc; x ] ~name raw, acc, x, i, k, m)

let split ~factor axis = Sched.split ~axis ~factor ~outer:LL.Serial ~inner:LL.Serial

let provenance_legs () =
  let run_ik name sched =
    let o, acc, x, i, k = build_ik name in
    let scheduled = Sched.apply (sched ~acc ~i ~k) o in
    (exec ~name ~acc ~init:[| 256.0; 256.0 |] ~x ~xs:xs_ik scheduled, scheduled.LL.llc)
  in
  (* The cell varies under [i], which the [Swap] puts inside the reduction loop: code generation
     localizes no level of the split program, which narrows per step. *)
  let swapped_split ~i ~k =
    let sp, ko, ki = split ~factor:2 k in
    ([ Sched.Swap { outer = i; inner = k }; sp ], ko, ki)
  in
  let serial, _ = run_ik "prc_ik_serial" (fun ~acc:_ ~i:_ ~k:_ -> []) in
  let twin, _ =
    run_ik "prc_ik_twin" (fun ~acc:_ ~i ~k ->
        let ops, _, _ = swapped_split ~i ~k in
        ops)
  in
  let outer, outer_llc =
    run_ik "prc_ik_outer" (fun ~acc ~i ~k ->
        let ops, ko, _ = swapped_split ~i ~k in
        ops @ [ pz ~target:acc ~over:ko ])
  in
  let inner, inner_llc =
    run_ik "prc_ik_inner" (fun ~acc ~i ~k ->
        let ops, _, ki = swapped_split ~i ~k in
        ops @ [ pz ~target:acc ~over:ki ])
  in
  let run_ikm name ~privatize =
    let o, acc, x, i, k, m = build_ikm name in
    let sp, mo, _ = split ~factor:1 m in
    let ops =
      [ Sched.Swap { outer = i; inner = k }; Sched.Swap { outer = i; inner = m }; sp ]
      @ if privatize then [ pz ~target:acc ~over:mo ] else []
    in
    let scheduled = Sched.apply ops o in
    (exec ~name ~acc ~init:[| 256.0; 256.0 |] ~x ~xs:xs_ik scheduled, scheduled.LL.llc)
  in
  let inside_twin, _ = run_ikm "prc_ikm_twin" ~privatize:false in
  let inside, inside_llc = run_ikm "prc_ikm_priv" ~privatize:true in
  Stdio.eprintf
    "privatize_recheck: split legs serial %s, twin %s, outer %s, inner %s; inside-level twin %s, \
     privatized %s (not part of the golden)\n\
     %!"
    (floats serial) (floats twin) (floats outer) (floats inner) (floats inside_twin) (floats inside);
  p claim_split_outer
    (List.equal Float.equal serial [ 258.0; 258.0 ]
    && List.equal Float.equal outer serial
    && only_prec Ir.Ops.single outer_llc);
  p claim_split_inner
    (List.equal Float.equal twin [ 256.0; 256.0 ]
    && List.equal Float.equal inner twin && only_prec bf16 inner_llc);
  p claim_split_inside
    (List.equal Float.equal inside_twin [ 256.0; 256.0 ]
    && List.equal Float.equal inside inside_twin
    && only_prec bf16 inside_llc)

(* === Provenance for Coalesce- and Partition-minted loops === *)

let minted_legs () =
  (* [i { k { m { acc[i] += x[i, k, m] } } }] with [x] laid out as [build_ik]'s: coalescing [k] and
     [m] gives [build_ik]'s [k] loop over the same cells in the same order, so the legs reproduce
     the split legs' values. *)
  let run_ikm name sched =
    let o, acc, x, i, k, m = build_ikm name in
    let scheduled = Sched.apply (sched ~acc ~i ~k ~m) o in
    (exec ~name ~acc ~init:[| 256.0; 256.0 |] ~x ~xs:xs_ik scheduled, scheduled.LL.llc)
  in
  (* As [swapped_split]: the [Swap] puts the cell's [i] inside the reduction, so code generation
     localizes no level of the scheduled program, which narrows per step. *)
  let coalesced ~i ~k ~m =
    let co, km = Sched.coalesce ~outer:k ~inner:m in
    ([ co; Sched.Swap { outer = i; inner = km } ], km)
  in
  let coalesced_split ~i ~k ~m =
    let ops, km = coalesced ~i ~k ~m in
    let sp, ko, ki = split ~factor:2 km in
    (ops @ [ sp ], ko, ki)
  in
  let serial, _ = run_ikm "prc_ckm_serial" (fun ~acc:_ ~i:_ ~k:_ ~m:_ -> []) in
  let twin, _ =
    run_ikm "prc_ckm_twin" (fun ~acc:_ ~i ~k ~m ->
        let ops, _, _ = coalesced_split ~i ~k ~m in
        ops)
  in
  let merged, merged_llc =
    run_ikm "prc_ckm_merged" (fun ~acc ~i ~k ~m ->
        let ops, km = coalesced ~i ~k ~m in
        ops @ [ pz ~target:acc ~over:km ])
  in
  let outer, outer_llc =
    run_ikm "prc_ckm_outer" (fun ~acc ~i ~k ~m ->
        let ops, ko, _ = coalesced_split ~i ~k ~m in
        ops @ [ pz ~target:acc ~over:ko ])
  in
  let inner, inner_llc =
    run_ikm "prc_ckm_inner" (fun ~acc ~i ~k ~m ->
        let ops, _, ki = coalesced_split ~i ~k ~m in
        ops @ [ pz ~target:acc ~over:ki ])
  in
  (* [build_ik]'s [k] partitioned at 2, with the same [Swap]. *)
  let run_part name ~privatize =
    let o, acc, x, i, k = build_ik name in
    let pt, segments = Sched.partition ~axis:k ~breakpoints:[ 2 ] in
    let ops =
      [ Sched.Swap { outer = i; inner = k }; pt ]
      @ if privatize then [ pz ~target:acc ~over:(List.hd_exn segments) ] else []
    in
    let scheduled = Sched.apply ops o in
    (exec ~name ~acc ~init:[| 256.0; 256.0 |] ~x ~xs:xs_ik scheduled, scheduled.LL.llc)
  in
  let part_twin, _ = run_part "prc_part_twin" ~privatize:false in
  let part, part_llc = run_part "prc_part_priv" ~privatize:true in
  Stdio.eprintf
    "privatize_recheck: coalesced legs serial %s, twin %s, merged %s, outer %s, inner %s; \
     partition twin %s, privatized %s (not part of the golden)\n\
     %!"
    (floats serial) (floats twin) (floats merged) (floats outer) (floats inner) (floats part_twin)
    (floats part);
  p claim_coalesce
    (List.equal Float.equal serial [ 258.0; 258.0 ]
    && List.equal Float.equal merged serial
    && List.equal Float.equal outer serial
    && only_prec Ir.Ops.single merged_llc
    && only_prec Ir.Ops.single outer_llc);
  p claim_coalesce_inner
    (List.equal Float.equal twin [ 256.0; 256.0 ]
    && List.equal Float.equal inner twin && only_prec bf16 inner_llc);
  p claim_partition
    (List.equal Float.equal part_twin [ 256.0; 256.0 ]
    && List.equal Float.equal part part_twin
    && only_prec bf16 part_llc)

(* === Provenance for Split_reduce-minted loops === *)

(* [x] varies along every axis; each [m] pair adds 0.625 to its cell, each pair of [k] iterations
   (one [Split_reduce] block) 1.25. *)
let xs_ikm =
  [|
    0.25;
    0.375;
    0.5;
    0.125;
    0.375;
    0.25;
    0.125;
    0.5;
    0.5;
    0.125;
    0.25;
    0.375;
    0.125;
    0.5;
    0.375;
    0.25;
  |]

let node_sr = Ll_test.node_factory ~prec:bf16 ~first_id:14200 ~dims:[| 2 |] ()

(* [i { k { m { acc[i] += x[i, k, m] }; tot[i] += x[i, k, 0] } }]: [tot] is the [Split_reduce]
   target, [acc] the node accumulated beside it. The serial rendering localizes [acc] at [m] (its
   level's single-statement chain; [k]'s body holds two statements, so no level enclosing [tot]'s
   write localizes it) and narrows once per [(i, k)], which stays at 256. *)
let build_ikm_tot name =
  let acc = node_sr (name ^ "_acc")
  and tot = node_sr (name ^ "_tot")
  and x = node_sr ~dims:[| 2; 4; 2 |] (name ^ "_x") in
  List.iter [ acc; tot; x ] ~f:Ll_test.materialize;
  let i = Ll_test.sym () and k = Ll_test.sym () and m = Ll_test.sym () in
  let raw =
    Ll_test.loop_n i 2
      (Ll_test.loop_n k 4
         (LL.Seq
            ( Ll_test.loop_n m 2
                (Ll_test.set acc
                   [| it i |]
                   (bin Ir.Ops.Add
                      (Ll_test.get acc [| it i |])
                      (Ll_test.get x [| it i; it k; it m |]))),
              Ll_test.set tot
                [| it i |]
                (bin Ir.Ops.Add
                   (Ll_test.get tot [| it i |])
                   (Ll_test.get x [| it i; it k; Ll_test.fixed 0 |])) )))
  in
  (Ll_test.optimize ~materialized:[ acc; tot; x ] ~name raw, acc, tot, x, m, k)

let exec_tot ~name ~acc ~tot ~x o =
  Ll_test.execute ~name o
    ~seed:[ (acc, [| 256.0; 256.0 |]); (tot, [| 0.0; 0.0 |]); (x, xs_ikm) ]
    ~read:[ acc ]
  |> List.hd_exn |> Array.to_list

let split_reduce_legs () =
  (* Two blocks of two [k] iterations each. A tile widened over the block loop would narrow once
     (258), over the in-block loop once per block (260); neither is the entry program's per-[(i, k)]
     narrowing, which both the split schedule and a storage tile keep (256). *)
  let run name ~privatize =
    let o, acc, tot, x, _, k = build_ikm_tot name in
    let sr, block, inner, _ = Sched.split_reduce ~axis:k ~target:tot ~num_blocks:2 in
    let ops =
      sr
      ::
      (match privatize with
      | `None -> []
      | `Block -> [ pz ~target:acc ~over:block ]
      | `Inner -> [ pz ~target:acc ~over:inner ])
    in
    let scheduled = Sched.apply ops o in
    (exec_tot ~name ~acc ~tot ~x scheduled, scheduled.LL.llc)
  in
  let serial, localized =
    let o, acc, tot, x, m, _ = build_ikm_tot "prc_sr_serial" in
    let widths = LL.serial_accum_widths ~serialized:(fun _ _ -> false) ~tn:acc o.LL.llc in
    ( exec_tot ~name:"prc_sr_serial" ~acc ~tot ~x o,
      match widths with
      | [ { LL.sa_level = Some level; sa_widens = true } ] -> Ir.Indexing.equal_symbol level m
      | _ -> false )
  in
  let twin, _ = run "prc_sr_twin" ~privatize:`None in
  let block, block_llc = run "prc_sr_block" ~privatize:`Block in
  let inner, inner_llc = run "prc_sr_inner" ~privatize:`Inner in
  Stdio.eprintf
    "privatize_recheck: split-reduce legs serial %s, twin %s, block %s, in-block %s (not part of \
     the golden)\n\
     %!"
    (floats serial) (floats twin) (floats block) (floats inner);
  p claim_split_reduce_entry localized;
  p claim_split_reduce
    (List.equal Float.equal serial [ 256.0; 256.0 ]
    && List.equal Float.equal twin serial
    && List.equal Float.equal block twin && List.equal Float.equal inner twin
    && only_prec bf16 block_llc && only_prec bf16 inner_llc)

let () =
  Test_utils.with_policy
    (fun pol -> { pol with Numerics.bf16_arithmetic = Numerics.Bf16_wide })
    (fun () ->
      recheck_legs ();
      provenance_legs ();
      minted_legs ();
      split_reduce_legs ())
