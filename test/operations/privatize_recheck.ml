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
   breaking op: the tile stays wide and executes to the entry program's once-narrowed 258. *)

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

let () =
  Test_utils.with_policy
    (fun pol -> { pol with Numerics.bf16_arithmetic = Numerics.Bf16_wide })
    (fun () -> recheck_legs ())
