(* gh-ocannl-1821: the ranking score's staged-copy term and the launch tail it switches on.

   At the gh-ocannl-1479 cells the compute leg is flat across a site's staged blocktile leaves,
   while the leaves measure 1.3-2x apart by block tile (64-, 32-, 16-wide) and barely by k-block
   depth. What orders them is the volume the operand staging copies move into workgroup memory over
   every block and k-step -- [M * N * K * (BM + BN) / (BM * BN)] cells -- priced over a fitted rate
   and added to the compute leg; with it, the launch fill quantizes by the launch tail (the busiest
   multiprocessor's [ceil (B / P)] blocks), fitted jointly.

   First on hand-built kernels, at power-of-two rates and a stand-in fill model so every value
   compared is exact: - the census counts a copy's executions (serial and grid multiplied) times its
   element width in its own parallel context, a fringe-guarded copy too, an [i < c] partition guard
   counting the iterations it admits, and nothing for a zero fill, a reduction into a shared
   scratch, a staged-to-staged move, a copy into an unstaged node, or a wide sibling nest; nothing
   at all in a kernel with a [Tile_mma] statement; - the term adds exactly the filled bytes over the
   rate to the compute leg, and a rate of [None] or [0], or no copy, leaves both legs bitwise as
   without the term; - the tail equals the continuous fill at [B <= P] and at a saturated multiple
   of [P], and is below it between multiples (48 blocks over 32 multiprocessors: 3/4 exactly; 64
   over 46: 32/46); it reaches the tile-MMA leg through the same fill.

   Then at the attention out projection of gpt2_mini (b8), lowered here and scheduled by every
   scalar blocktile leaf of the GPU matmul family (hermetic, nothing compiles, so the claims run on
   every backend): the census of each leaf equals the block-tile arithmetic above -- what the
   offline fit (benchmarks/gh1821_geometry_fit.py) assumes of the code. *)

(* cc-fp-contract: not-executed -- the claims compare cost-model scores computed in OCaml *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
open Verdict.Claims
module LL = Ir.Low_level
module CM = Ir.Cost_model
module BI = Ir.Backend_intf

let fresh_tn =
  let make = Ll_test.node_factory ~first_id:972_000_000 ~dims:[||] () in
  fun label dims -> make ~dims label

let it = Ll_test.iter
let fm = { CM.fm_multiprocessors = 4; fm_fill_threads = 64; fm_tail = false; fm_chains = 1 }

(* A staged k-step nest: [nb] blocks of [nw] threads, [ns] serial k-steps; each step copies one cell
   per thread from the global [a] into the staged [ta], and the block's threads then accumulate it
   into [d]. *)
let staged_kernel ~tag ~nb ~nw ~ns =
  let d = fresh_tn (tag ^ "_d") [| nb; nw |]
  and a = fresh_tn (tag ^ "_a") [| nb; ns; nw |]
  and ta = fresh_tn (tag ^ "_ta") [| nw |] in
  let b = Ll_test.sym () and w = Ll_test.sym () and s = Ll_test.sym () in
  let step =
    Ll_test.seq
      (Ll_test.set ta [| it w |] (Ll_test.get a [| it b; it s; it w |]))
      (Ll_test.set d
         [| it b; it w |]
         (Ll_test.add (Ll_test.get d [| it b; it w |]) (Ll_test.get ta [| it w |])))
  in
  let code =
    Ll_test.loop_n ~axis:LL.Grid b nb
      (Ll_test.loop_n ~axis:LL.Workgroup w nw (Ll_test.loop_n s ns step))
  in
  (code, ta)

let peak = 1024.
let issue = 256.

let legs ?staged_copies ?staged_copy_rate ?(fill = fm) code =
  CM.ranking_legs_of ~peak_flops:peak ~peak_memory_bandwidth:peak ~issue_rate:issue
    ~launch:(fill, CM.launch_contexts code)
    ?staged_copies ?staged_copy_rate (CM.analyze code) None

let legs_equal (c1, m1) (c2, m2) = Option.equal Float.equal c1 c2 && Option.equal Float.equal m1 m2
let ctx_list copies = List.map copies ~f:(fun c -> (c.CM.sc_blocks, c.CM.sc_threads, c.CM.sc_bytes))

let () =
  let code, ta = staged_kernel ~tag:"scc_k" ~nb:2 ~nw:4 ~ns:3 in
  let is_staged tn = Ir.Tnode.equal tn ta in
  let copies = CM.staged_copies ~is_staged code in
  (* 2 blocks x 4 threads x 3 steps, 4-byte cells. *)
  p "the census counts the copy's executions times its width, in its own context"
    (List.equal Poly.equal (ctx_list copies) [ (2, 4, 96.) ]);
  p_empty "nothing is staged where no node is" ~over:copies
    (CM.staged_copies ~is_staged:(fun _ -> false) code);
  (* The context fills 2 * 4 / (4 * 64) = 1/32 of the device: 96 * 32 / 8 = 384 s. *)
  let base = legs code in
  let priced = legs ~staged_copies:copies ~staged_copy_rate:8. code in
  p "the term adds the filled bytes over the rate to the compute leg, exactly"
    (match (base, priced) with
    | (Some c0, m0), (Some c1, m1) -> Float.equal c1 (c0 +. 384.) && Option.equal Float.equal m0 m1
    | _ -> false);
  p "a rate of None leaves both legs bitwise as without the term"
    (legs_equal (legs ~staged_copies:copies code) base);
  p "a rate of 0 leaves both legs bitwise as without the term"
    (legs_equal (legs ~staged_copies:copies ~staged_copy_rate:0. code) base);
  p "no copy leaves both legs bitwise as without the term"
    (legs_equal (legs ~staged_copies:[] ~staged_copy_rate:8. code) base);
  p "the term never lowers the compute leg below the bound's"
    (match fst priced with
    | Some c -> Float.(c >= of_int (CM.analyze code).CM.flops /. peak)
    | None -> false)

let () =
  (* Two copy contexts in one kernel: the listing's per-kernel stg= (the filled bytes) moves its
     compute leg by stg * (1/R - 1/L) between rates, exactly -- what gh1600_txn_fit.py --staged-rate
     re-scores. *)
  let k1, ta1 = staged_kernel ~tag:"scc_r1" ~nb:2 ~nw:4 ~ns:3 in
  let k2, ta2 = staged_kernel ~tag:"scc_r2" ~nb:8 ~nw:16 ~ns:2 in
  let code = Ll_test.seq k1 k2 in
  let is_staged tn = Ir.Tnode.equal tn ta1 || Ir.Tnode.equal tn ta2 in
  let copies = CM.staged_copies ~is_staged code in
  p "two nests, two copy contexts" (List.length copies = 2);
  let stg = CM.filled_staged_bytes ~fill_model:fm copies in
  match
    ( fst (legs ~staged_copies:copies ~staged_copy_rate:8. code),
      fst (legs ~staged_copies:copies ~staged_copy_rate:32. code) )
  with
  | Some at_l, Some at_r ->
      p "a listing at rate L re-scores at R by stg * (1/R - 1/L), over every context"
        (Float.equal at_r (at_l +. (stg *. ((1. /. 32.) -. (1. /. 8.)))))
  | _ -> p "a listing at rate L re-scores at R by stg * (1/R - 1/L), over every context" false

let () =
  (* Nested guards on one index compose as their minimum, in either order. *)
  let nw = 8 in
  let a = fresh_tn "scc_g_a" [| nw |] and ta = fresh_tn "scc_g_ta" [| nw |] in
  let is_staged tn = Ir.Tnode.equal tn ta in
  let nested outer inner =
    let w = Ll_test.sym () in
    let guard n body = Ll_test.if_ (Ll_test.lt (Ll_test.embed w) (Ll_test.ic n)) body in
    Ll_test.loop_n ~axis:LL.Workgroup w nw
      (guard outer (guard inner (Ll_test.set ta [| it w |] (Ll_test.get a [| it w |]))))
  in
  (* 2 of 8 threads, 4-byte cells. *)
  p "an outer [w < 4] and an inner [w < 2] admit 2 threads"
    (List.equal Poly.equal (ctx_list (CM.staged_copies ~is_staged (nested 4 2))) [ (1, 8, 8.) ]);
  p "an outer [w < 2] and an inner [w < 4] admit 2 threads"
    (List.equal Poly.equal (ctx_list (CM.staged_copies ~is_staged (nested 2 4))) [ (1, 8, 8.) ])

let () =
  (* What is not a copy, and what lends a copy no context. *)
  let nw = 4 and ns = 2 in
  let a = fresh_tn "scc_n_a" [| ns; nw |]
  and ta = fresh_tn "scc_n_ta" [| nw |]
  and tb = fresh_tn "scc_n_tb" [| nw |]
  and scratch = fresh_tn "scc_n_s" [| nw |]
  and out = fresh_tn "scc_n_o" [| nw |]
  and wide = fresh_tn "scc_n_w" [| 64; 64 |] in
  let staged = [ ta; tb; scratch ] in
  let is_staged tn = List.mem staged tn ~equal:Ir.Tnode.equal in
  let w = Ll_test.sym () and s = Ll_test.sym () and gi = Ll_test.sym () and gw = Ll_test.sym () in
  let fringe =
    Ll_test.where_
      (Ll_test.lt (Ll_test.embed w) (Ll_test.ic 3))
      (Ll_test.get a [| it s; it w |])
      (Ll_test.c 0.)
  in
  let body =
    List.reduce_exn ~f:Ll_test.seq
      [
        (* Counted: a fringe-guarded copy, behind a partition guard admitting 3 of the 4 threads. *)
        Ll_test.if_ (Ll_test.lt (Ll_test.embed w) (Ll_test.ic 3)) (Ll_test.set ta [| it w |] fringe);
        (* Not counted: a zero fill, a staged-to-staged move, a reduction into shared scratch, a
           copy into an unstaged node. *)
        Ll_test.set tb [| it w |] (Ll_test.c 0.);
        Ll_test.set tb [| it w |] (Ll_test.get ta [| it w |]);
        Ll_test.set scratch
          [| it w |]
          (Ll_test.add (Ll_test.get scratch [| it w |]) (Ll_test.get a [| it s; it w |]));
        Ll_test.set out [| it w |] (Ll_test.get a [| it s; it w |]);
      ]
  in
  let nest = Ll_test.loop_n ~axis:LL.Workgroup w nw (Ll_test.loop_n s ns body) in
  let sibling =
    Ll_test.loop_n ~axis:LL.Grid gi 64
      (Ll_test.loop_n ~axis:LL.Workgroup gw 64 (Ll_test.set wide [| it gi; it gw |] (Ll_test.c 0.)))
  in
  let code = Ll_test.seq sibling nest in
  (* 3 threads x 2 steps, 4-byte cells. *)
  p "only the operand copy counts, at the iterations its index guard admits, in its own context"
    (List.equal Poly.equal (ctx_list (CM.staged_copies ~is_staged code)) [ (1, 4, 24.) ]);
  let d = fresh_tn "scc_m_d" [| 2; 2 |] and ma = fresh_tn "scc_m_a" [| 2; 2 |] in
  let mb = fresh_tn "scc_m_b" [| 2; 2 |] in
  let at tn = (tn, [| Ll_test.fixed 0; Ll_test.fixed 0 |]) in
  let mma = Ll_test.tile_mma ~d:(at d) ~a:(at ta) ~b:(at mb) LL.Noop in
  let copy =
    Ll_test.set ta [| Ll_test.fixed 0 |] (Ll_test.get ma [| Ll_test.fixed 0; Ll_test.fixed 0 |])
  in
  p_empty "a kernel with a Tile_mma statement has no counted copies"
    ~over:(CM.staged_copies ~is_staged copy)
    (CM.staged_copies ~is_staged (Ll_test.seq copy mma))

let () =
  let tail = { CM.fm_multiprocessors = 32; fm_fill_threads = 64; fm_tail = true; fm_chains = 1 } in
  let flat = { tail with fm_tail = false } in
  let both ~blocks ~threads =
    (CM.context_fill tail ~blocks ~threads, CM.context_fill flat ~blocks ~threads)
  in
  let equal_at ~blocks ~threads =
    let q, c = both ~blocks ~threads in
    Float.equal q c
  in
  p "the tail is the continuous fill at B <= P" (equal_at ~blocks:16 ~threads:64);
  p "and at B = P" (equal_at ~blocks:32 ~threads:64);
  p "and at a saturated multiple of P" (equal_at ~blocks:64 ~threads:64);
  p "48 saturated blocks over 32 multiprocessors fill 3/4 under the tail, all of it without"
    (Poly.equal (both ~blocks:48 ~threads:64) (0.75, 1.));
  p "unsaturated, the busiest multiprocessor's two blocks still fit its issue: no tail"
    (Poly.equal (both ~blocks:40 ~threads:16) (0.3125, 0.3125));
  let rog = { CM.fm_multiprocessors = 46; fm_fill_threads = 256; fm_tail = true; fm_chains = 1 } in
  p "64 blocks of 256 threads over 46 multiprocessors fill 32/46"
    Float.(abs (CM.context_fill rog ~blocks:64 ~threads:256 -. (32. /. 46.)) < 1e-12);
  (* The tile-MMA leg divides by the same fill. *)
  let d = fresh_tn "scc_t_d" [| 2; 2 |]
  and a = fresh_tn "scc_t_a" [| 2; 2 |]
  and b = fresh_tn "scc_t_b" [| 2; 2 |] in
  let g = Ll_test.sym () in
  let at tn = (tn, [| Ll_test.fixed 0; Ll_test.fixed 0 |]) in
  let mma =
    Ll_test.loop_n ~axis:LL.Grid g 48 (Ll_test.tile_mma ~d:(at d) ~a:(at a) ~b:(at b) LL.Noop)
  in
  let mma_leg fill =
    let s = CM.analyze mma in
    fst
      (CM.ranking_legs_of ~peak_flops:peak ~peak_memory_bandwidth:peak ~issue_rate:issue
         ~mma_flops:s.CM.flops
         ~launch:(fill, CM.launch_contexts mma)
         s None)
  in
  let one_lane =
    { CM.fm_multiprocessors = 32; fm_fill_threads = 1; fm_tail = true; fm_chains = 1 }
  in
  p "the tail reaches the tile-MMA leg through the same fill: 48 one-lane blocks over 32, 4/3x"
    (match (mma_leg one_lane, mma_leg { one_lane with fm_tail = false }) with
    | Some q, Some c ->
        let ops = Float.of_int (CM.analyze mma).CM.flops in
        Float.equal c (ops /. peak) && Float.equal q (ops /. 0.75 /. peak)
    | _ -> false)

(* The out projection at b8: each scalar blocktile leaf's census against its block tile. *)
let stand_in =
  {
    BI.no_hardware_limits with
    max_threads_per_workgroup = Some 1024;
    max_workgroup_memory_bytes = Some 49152;
    max_workgroup_dims = Some (1024, 1024, 64);
    max_grid_yz = Some 65535;
    simdgroup_width = Some 32;
    peak_flops = Some 1.5e13;
    peak_memory_bandwidth = Some 8.0e12;
    issue_rate = Some 3.4e12;
    multiprocessors = Some 46;
    fill_threads = Some 256;
    transaction_rate = Some 1.0e12;
    staged_copy_rate = Some 1.0e12;
  }

(* The seeding gate reads neither the copy rate nor the tail: the term was fitted among staged
   leaves, and the gate weighs a staged sketch against an unstaged preset. Through the gate's own
   ranker: the plan the untuned CUDA compile derives for a two-matmul chain (model_seeded_anchor's,
   which the term priced in the gate stopped seeding on rog's CUDA) under a CUDA-shaped record is
   the same plan, by its digest, with and without the copy rate -- and it seeds. *)
let cuda_like =
  {
    stand_in with
    cache_capacity_bytes = Some (12 * 1024 * 1024 * 1024);
    cache_bandwidth = Some 1.8e12;
    max_resident_threads = Some (46 * 1536);
    dram_bandwidth = Some 6.0e11;
  }

let () =
  let q = 128 in
  let cell idx = Float.of_int (1 + (Array.fold idx ~init:0 ~f:( + ) % 7)) *. 0.125 in
  let mat label = NTDSL.init ~l:label ~prec:Ir.Ops.single ~o:[ q ] ~i:[ q ] ~f:cell () in
  let qa = mat "scc_ca" and qb = mat "scc_cb" and qc = mat "scc_cc" and qf = mat "scc_cf" in
  let%op qd = qa + qb in
  Train.set_materialized qd.Tensor.value;
  let%op qe = qd * qc in
  Train.set_materialized qe.Tensor.value;
  let%op qg = qf * qe in
  Train.set_materialized qg.Tensor.value;
  let target = Ll_test.cuda_model_target ~limits:cuda_like () in
  let facts = target.Autotune.mt_facts and capabilities = target.Autotune.mt_capabilities in
  let plan ~tag limits =
    let opt =
      Context.lowered_for_decisions ~name:("scc_chain_" ^ tag) ~facts ~limits (Context.cpu ())
        (Train.forward qg) Ir.Indexing.Empty
    in
    Autotune.seeded_plan
      ~seed_gate:(Autotune.seeded_default_gate ~limits)
      ~capabilities ~facts ~limits ~accum_prec:capabilities.Ir.Backend_intf.accum_prec
      ~static_indices:[] opt
  in
  let seeded_count = function
    | Some { Autotune.sp_segments = Some (_, ds); _ } ->
        List.count ds ~f:(function Autotune.Seeded_sketch _ -> true | _ -> false)
    | _ -> 0
  in
  let with_rate = plan ~tag:"rate" cuda_like in
  let without = plan ~tag:"norate" { cuda_like with staged_copy_rate = None } in
  Stdio.eprintf
    "cost_model_staged_copy: chain seeds %d segment(s) with the copy rate, %d without (not part of \
     the golden)\n"
    (seeded_count with_rate) (seeded_count without);
  p "the auto gate (Ranked) is what derives the chain's plan here"
    (match Autotune.seeded_default_gate ~limits:cuda_like with
    | Autotune.Ranked _ -> true
    | Autotune.Min_reduction _ -> false);
  p "under a CUDA-shaped record the untuned plan seeds the chain" (seeded_count without >= 1);
  p "and it is the same plan with the copy rate: the gate's ranker prices no copy term"
    (match (with_rate, without) with
    | Some a, Some b -> String.equal (Autotune.seeded_plan_digest a) (Autotune.seeded_plan_digest b)
    | _ -> false)

let () =
  let ctx = Context.cpu () in
  let cell idx = Float.of_int (1 + (Array.fold idx ~init:0 ~f:( + ) % 7)) *. 0.125 in
  let nb, ns, nj, nh, ne = (8, 128, 256, 8, 32) in
  let w = NTDSL.init ~l:"scc_wout" ~prec:Ir.Ops.single ~o:[ nj ] ~i:[ nh; ne ] ~f:cell () in
  let x = NTDSL.init ~l:"scc_xout" ~prec:Ir.Ops.single ~b:[ nb; ns ] ~o:[ nh; ne ] ~f:cell () in
  let%op d = w * x in
  Train.set_materialized d.Tensor.value;
  let lowered =
    Context.lowered_for_decisions ~name:"scc_outproj" ctx (Train.forward d) Ir.Indexing.Empty
  in
  let hermetic (o : LL.optimized) =
    {
      o with
      LL.traced_store = Hashtbl.copy o.LL.traced_store;
      LL.optimize_ctx = LL.copy_optimize_ctx o.LL.optimize_ctx;
    }
  in
  let capabilities = BI.no_codegen_capabilities in
  let leaves =
    (match Autotune.matmul_sketch_tree ~is_gpu:true ~is_cpu:false ~limits:stand_in lowered with
      | None -> []
      | Some tree -> List.filter (Ir.Schedule_space.leaves tree) ~f:Autotune.scalar_blocktile)
    @ Autotune.coalesced_seed_params ~capabilities ~is_gpu:true ~is_cpu:false ~limits:stand_in
        lowered
  in
  let m = nb * ns and n = nj and k = nh * ne in
  let counted =
    List.filter_map leaves ~f:(fun (sp : Autotune.sketch_params) ->
        match
          Ir.Schedule.apply
            (Autotune.sketch_schedule ~accum_prec:Fn.id ~p:sp (hermetic lowered))
            (hermetic lowered)
        with
        | exception _ -> None
        | post ->
            let is_staged tn =
              Set.mem post.LL.workgroup_shared tn
              && (not (Set.mem post.LL.simdgroup_fragments tn))
              && not (Set.mem post.LL.masked_fragments tn)
            in
            if Set.is_empty post.LL.workgroup_shared then None
            else
              let copies = CM.staged_copies ~is_staged post.LL.llc in
              let bytes = List.sum (module Float) copies ~f:(fun c -> c.CM.sc_bytes) in
              let expected =
                Float.of_int (m * n * k / (sp.sk_bm * sp.sk_bn) * (sp.sk_bm + sp.sk_bn) * 4)
              in
              Stdio.eprintf
                "cost_model_staged_copy: %s staged %.0f bytes, block-tile arithmetic %.0f (not \
                 part of the golden)\n"
                (Autotune.sketch_params_label sp) bytes expected;
              Some (sp, bytes, expected))
  in
  p_all ~min:8 "every staged leaf's census equals M*N*K*(BM+BN)/(BM*BN) cells of 4 bytes" counted
    ~f:(fun (_, bytes, expected) -> Float.equal bytes expected);
  p_exists "a coalesced leaf is among them" counted ~f:(fun (sp, _, _) ->
      Option.is_some sp.Autotune.sk_coalesce)
