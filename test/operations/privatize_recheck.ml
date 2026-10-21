(* [Schedule.Privatize]'s tile width, judged through whole schedules (follow-ups of gh-ocannl-1151).
   All legs run under [Bf16_wide], so the accumulator residency widens bf16 to f32 on every backend,
   over bf16 accumulators seeded at 256, where bf16's spacing is 2: each outer iteration adds 1.25
   in steps below 1, so a program narrowing once gives 258 (256 + 2.5), one narrowing per outer
   iteration gives 260 (257.25 rounds up, twice), and one narrowing per step stays at 256. The
   inputs are dyadic, so every partial sum is exact in f32 whatever the order.

   1. The scope grammar and the tile's lifetime, refused at the op (gh-ocannl-1467). A [Local_scope]
   body is hoisted out of its position, so a later op that puts into the scope a materializing
   [Unroll] minted around the tile's update a hardware-typed loop ([Retype] of a loop inside it) or
   a tensor-node write ([Stage]'s loads landing inside it) is refused at that op, typed, naming it
   -- through [Schedule.apply_classified], and through [Schedule.op_legality] asked of the program
   the ops before it left (the race queries alone call the [Grid] retype and the [Stage] legal) --
   and so is each one's twin without [Privatize]: the rule is the scope's, not the tile's. These
   were the only triggers of the final re-check ([Schedule.privatized_tile_widens]); compilation
   refused them at its exit gates, untyped. A hardware-typed loop inside the tile's per-thread
   lifetime (between its init-load and its store-back) is refused at the later op too, wide tile or
   not, whether the cell omits the loop or mentions it, and for the privatized loop itself: through
   [apply_classified] and [schedule_legality], which apply the chain; a single op judged by
   [op_legality] has no record of the tile ([Op_unknown] -- the tile is routine-local scratch the
   race queries exempt). The neighbours stay admitted: a [Vectorized] retype or an in-place [Unroll]
   of the scope-nested loop, a [Stage] whose loads stay outside the scope, a [Grid] retype of a loop
   enclosing the tile's transfers, a [Vectorized] retype of the omitted loop. The control is the
   same schedule without the breaking op: the tile stays wide and executes to the entry program's
   once-narrowed 258.

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
module Outcome = Ir.Schedule_outcome
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
  "without a breaking op the privatized tile stays f32, passes the re-check, and executes to the \
   entry program's once-narrowed value (258)"

let claim_scope_hw =
  "a Workgroup_reduce or Grid Retype of a loop inside the scope a materializing Unroll minted \
   around a privatized tile's update is refused at the Retype, typed, naming the hardware loop in \
   the scope"

let claim_scope_stage =
  "a Stage whose loads land inside that scope is refused at the Stage, typed, naming the \
   tensor-node write in the scope"

let claim_scope_twins =
  "the same three schedules without Privatize are refused the same way: the rule is the scope's"

let claim_scope_oracle =
  "op_legality asked of the program the Privatize and the Unroll left calls all three ops illegal, \
   with the refusal apply raises, and probing the Stage leaves that program's traced store as it \
   was"

let claim_scope_admitted =
  "a Vectorized retype or an in-place Unroll of that scope-nested loop, and the same Stage applied \
   ahead of the Unroll (its loads then keep the nest from minting a scope), are admitted by apply \
   and not called illegal by op_legality"

let claim_lifetime_omitted =
  "after a Privatize, a Grid or Workgroup Retype of a loop the tile's cell omits, between the \
   tile's transfers and its update, is refused at the Retype, typed, through apply_classified and \
   schedule_legality"

let claim_lifetime_single_op =
  "op_legality judging that Retype alone, on the program an earlier apply privatized, does not \
   call it illegal: it has no record of the tile"

let claim_lifetime_storage =
  "on a storage-precision tile, a Grid Retype of an inner loop its cell mentions, or of the \
   privatized loop itself, is refused at the Retype too"

let claim_lifetime_admitted =
  "a Grid Retype of a loop enclosing the tile's transfers, and a Vectorized Retype of the omitted \
   loop, are admitted"

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

(* === The scope grammar and the tile's lifetime === *)

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

(* f32 accumulators, so a [Privatize] tile is at storage precision, not wide: [r { i { acc[i] +=
   x[r, i] } }] (the cell mentions the inner loop [i]) and [i { r { acc[i] += x[i, r] } }] (the
   cell's [i] selects the element outside the privatized [r]). *)
let node32 = Ll_test.node_factory ~prec:Ir.Ops.single ~first_id:14300 ~dims:[| 2 |] ()

let build_f32 name ~inner_mentioned =
  let acc = node32 (name ^ "_acc") and x = node32 ~dims:[| 2; 2 |] (name ^ "_x") in
  Ll_test.materialize acc;
  Ll_test.materialize x;
  let r = Ll_test.sym () and i = Ll_test.sym () in
  let update =
    Ll_test.set acc
      [| it i |]
      (LL.Binop
         ( Ir.Ops.Add,
           (Ll_test.get acc [| it i |], Ir.Ops.single),
           ( Ll_test.get x (if inner_mentioned then [| it r; it i |] else [| it i; it r |]),
             Ir.Ops.single ) ))
  in
  let raw =
    if inner_mentioned then Ll_test.loop_n r 2 (Ll_test.loop_n i 2 update)
    else Ll_test.loop_n i 2 (Ll_test.loop_n r 2 update)
  in
  (Ll_test.optimize ~materialized:[ acc; x ] ~name raw, acc, r, i)

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
let retype axis ty = Sched.Retype { axis; ty }

(* The detail of the typed refusal [apply_classified] raises for [sched], [None] when it applies. *)
let refusal ~name sched o =
  match Sched.apply_classified sched o with
  | (_ : LL.optimized) -> None
  | exception Outcome.Cause_at (Outcome.Transform, Outcome.Illegal_schedule { detail; _ }) ->
      Stdio.eprintf "privatize_recheck: %s refused: %s (not part of the golden)\n%!" name detail;
      Some detail

(* Refused at the op [op], the detail saying [what]. *)
let refused ~name ~op ~what sched o =
  match refusal ~name sched o with
  | Some detail ->
      String.is_prefix detail ~prefix:("Schedule." ^ op ^ ":")
      && String.is_substring detail ~substring:what
  | None -> false

let admitted ~name sched o = Option.is_none (refusal ~name sched o)

let illegal_with ~what = function
  | Sched.Op_illegal msg -> String.is_substring msg ~substring:what
  | Sched.Op_legal | Sched.Op_unknown _ -> false

let not_illegal = function
  | Sched.Op_illegal _ -> false
  | Sched.Op_legal | Sched.Op_unknown _ -> true

let in_scope = "hardware-annotated loop"
and scope_write = "writes the tensor node"

let lifetime = "per-thread lifetime of the Privatize tile"

let scope_legs () =
  let hw ty name ~privatize =
    let o, acc, _, r, s, t = build_rst name in
    let pre = if privatize then [ pz ~target:acc ~over:r ] else [] in
    refused ~name ~op:"Retype" ~what:in_scope (pre @ [ unroll s; retype t ty ]) o
  in
  let stage name ~privatize =
    let o, acc, x, r, s, t = build_rsut name in
    let pre = if privatize then [ pz ~target:acc ~over:r ] else [] in
    refused ~name ~op:"Stage" ~what:scope_write (pre @ [ unroll s; stage_over x t ]) o
  in
  p claim_scope_hw
    (hw LL.Workgroup_reduce "prc_wgr" ~privatize:true && hw LL.Grid "prc_grid" ~privatize:true);
  p claim_scope_stage (stage "prc_stage" ~privatize:true);
  p claim_scope_twins
    (hw LL.Workgroup_reduce "prc_wgr_twin" ~privatize:false
    && hw LL.Grid "prc_grid_twin" ~privatize:false
    && stage "prc_stage_twin" ~privatize:false);
  (* The oracle, asked of the program the ops before the breaking one left. *)
  let rst_prefix name =
    let o, acc, _, r, s, t = build_rst name in
    (Sched.apply [ pz ~target:acc ~over:r; unroll s ] o, t)
  in
  let rsut_prefix name =
    let o, acc, x, r, s, t = build_rsut name in
    (Sched.apply [ pz ~target:acc ~over:r; unroll s ] o, x, r, t)
  in
  (let pre, t = rst_prefix "prc_oracle_hw" in
   let st, x, _, t' = rsut_prefix "prc_oracle_stage" in
   let keys o = List.sort (Hashtbl.keys o.LL.traced_store) ~compare:Tn.compare in
   let store = keys st in
   let stage_v = Sched.op_legality st (stage_over x t') in
   p claim_scope_oracle
     (illegal_with ~what:in_scope (Sched.op_legality pre (retype t LL.Workgroup_reduce))
     && illegal_with ~what:in_scope (Sched.op_legality pre (retype t LL.Grid))
     && illegal_with ~what:scope_write stage_v
     && List.equal Tn.equal (keys st) store));
  (let pre, t = rst_prefix "prc_adm_vec" in
   let pre_u, t_u = rst_prefix "prc_adm_unroll" in
   let staged =
     let o, acc, x, r, s, t = build_rsut "prc_adm_stage" in
     match Sched.apply_classified [ pz ~target:acc ~over:r; stage_over x t; unroll s ] o with
     | applied -> Ll_test.count_scopes applied.LL.llc = 0
     | exception Outcome.Cause_at _ -> false
   in
   let st, x, t_st =
     let o, acc, x, r, _, t = build_rsut "prc_adm_stage_oracle" in
     (Sched.apply [ pz ~target:acc ~over:r ] o, x, t)
   in
   let adm_rst name ty =
     let o, acc, _, r, s, t = build_rst name in
     admitted ~name [ pz ~target:acc ~over:r; unroll s; ty t ] o
   in
   p claim_scope_admitted
     (adm_rst "prc_adm_vec" (fun t -> retype t LL.Vectorized)
     && adm_rst "prc_adm_unroll" (fun t -> Sched.Unroll { axis = t; materialize = false })
     && staged
     && not_illegal (Sched.op_legality pre (retype t LL.Vectorized))
     && not_illegal (Sched.op_legality pre_u (Sched.Unroll { axis = t_u; materialize = false }))
     && not_illegal (Sched.op_legality st (stage_over x t_st))));
  (* The control: the same tile with nothing breaking its scope. *)
  let serial =
    let o, acc, x, _, _, _ = build_rst "prc_serial" in
    exec ~name:"prc_serial" ~acc ~init:[| 256.0 |] ~x ~xs:xs_rst o
  in
  let o, acc, x, r, s, _ = build_rst "prc_ctl" in
  let ctl = Sched.apply [ pz ~target:acc ~over:r; unroll s ] o in
  let values = exec ~name:"prc_ctl" ~acc ~init:[| 256.0 |] ~x ~xs:xs_rst ctl in
  Stdio.eprintf "privatize_recheck: serial %s, control %s (not part of the golden)\n%!"
    (floats serial) (floats values);
  p claim_control
    ((match tiles ctl.LL.llc with
       | [ tile ] ->
           Sched.privatized_tile_widens ~tile ctl.LL.llc
           && Ir.Ops.equal_prec (Lazy.force tile.Tn.storage_prec) Ir.Ops.single
       | _ -> false)
    && List.equal Float.equal serial [ 258.0 ]
    && List.equal Float.equal values serial)

let lifetime_legs () =
  let omitted name ty =
    let o, acc, _, r, _, t = build_rst name in
    let sched = [ pz ~target:acc ~over:r; retype t ty ] in
    let chain =
      let o, acc, _, r, _, t = build_rst (name ^ "_chain") in
      match List.last (Sched.schedule_legality o [ pz ~target:acc ~over:r; retype t ty ]) with
      | Some (Sched.Retype _, v) -> illegal_with ~what:lifetime v
      | _ -> false
    in
    refused ~name ~op:"Retype" ~what:lifetime sched o && chain
  in
  p claim_lifetime_omitted (omitted "prc_life_grid" LL.Grid && omitted "prc_life_wg" LL.Workgroup);
  (let o, acc, _, r, _, t = build_rst "prc_life_single" in
   let privatized = Sched.apply [ pz ~target:acc ~over:r ] o in
   p claim_lifetime_single_op (not_illegal (Sched.op_legality privatized (retype t LL.Grid))));
  let f32 name ~inner_mentioned ops =
    let o, acc, r, i = build_f32 name ~inner_mentioned in
    (o, pz ~target:acc ~over:r :: ops ~r ~i)
  in
  p claim_lifetime_storage
    (let o, sched =
       f32 "prc_life_mentioned" ~inner_mentioned:true (fun ~r:_ ~i -> [ retype i LL.Grid ])
     in
     let o', sched' =
       f32 "prc_life_over" ~inner_mentioned:true (fun ~r ~i:_ -> [ retype r LL.Grid ])
     in
     let o'', sched'' = f32 "prc_life_storage_tile" ~inner_mentioned:true (fun ~r:_ ~i:_ -> []) in
     let storage_tile =
       match tiles (Sched.apply sched'' o'').LL.llc with
       | [ tile ] -> Ir.Ops.equal_prec (Lazy.force tile.Tn.storage_prec) Ir.Ops.single
       | _ -> false
     in
     storage_tile
     && refused ~name:"prc_life_mentioned" ~op:"Retype" ~what:lifetime sched o
     && refused ~name:"prc_life_over" ~op:"Retype" ~what:lifetime sched' o');
  p claim_lifetime_admitted
    (let o, sched =
       f32 "prc_life_outer" ~inner_mentioned:false (fun ~r:_ ~i -> [ retype i LL.Grid ])
     in
     let o', acc, _, r, _, t = build_rst "prc_life_vec" in
     admitted ~name:"prc_life_outer" sched o
     && admitted ~name:"prc_life_vec" [ pz ~target:acc ~over:r; retype t LL.Vectorized ] o')

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
      scope_legs ();
      lifetime_legs ();
      provenance_legs ();
      minted_legs ();
      split_reduce_legs ())
