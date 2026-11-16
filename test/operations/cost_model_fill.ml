(* gh-ocannl-1479: the ranking score's launch fill.

   The compute leg used to divide a kernel's scalar operations by one issue rate, so every scalar
   leaf of a site scored the same whatever its launch: a plain leaf, its batch loops serial inside a
   few blocks, tied its [bgrid] twin, and measured 3-36x slower (gh-ocannl-1278). Each op-carrying
   statement's work is now attributed to its parallel context -- [B] blocks (its enclosing [Grid]
   loops) of [T] threads (its [Workgroup] loops) -- and priced at the issue rate times [min 1 (B *
   min T t / (P * t))].

   First on hand-built kernels, at a stand-in fill model of P = 4 multiprocessors saturating at t =
   64 threads and power-of-two rates, so every value compared is exact: the contexts and fills of a
   grid-bound nest, its serial-batch twin, a wide zero nest beside a one-block contraction (no work,
   no say: a kernel-wide launch reading would call that kernel full), a block wider than t, two
   compute nests (the leg sums per context), and accepted vs declined tile-MMA work (both filled).
   The score never drops below the roofline bound, and a fill model under which every context is
   full scores bitwise what no fill model does.

   Then at the q/k/v projection of gpt2_mini (b8), lowered here and scheduled by every scalar
   blocktile leaf of the GPU matmul family under a CUDA-like stand-in record (hermetic: nothing is
   compiled, so the claims run on every backend): with the record's launch-fill constants each plain
   leaf ranks strictly above its [bgrid] twin and the best-ranked leaf is a [bgrid] one; without
   them the twins tie, as before. Ranks go to stderr: device constants' quotients.

   gh-ocannl-1639, the per-thread ILP credit: the census of independent private accumulation chains
   (register tiles unrolled or looped, one-accumulator reductions, and what counts none: global
   accumulators, dependent or guarded updates, elementwise work), the capped normalized credit's
   exact values at the stand-in with a cap of 4 (and the launch tail), the controls that keep it
   bitwise the thread-only fill (cap 1, unknown chains, chains at the cap), the census on the
   scheduled qkv leaves ([tm * tn]), and that no backend advises a cap: under the running backend's
   own limits the ranking is the thread-only one. *)

(* cc-fp-contract: not-executed -- the bitwise claim compares cost-model scores computed in OCaml *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
open Verdict.Claims
module LL = Ir.Low_level
module CM = Ir.Cost_model
module BI = Ir.Backend_intf
module Tn = Ir.Tnode

let fresh_tn =
  let make = Ll_test.node_factory ~first_id:971_000_000 ~dims:[||] () in
  fun label dims -> make ~dims label

let it = Ll_test.iter
let fm = { CM.fm_multiprocessors = 4; fm_fill_threads = 64; fm_tail = false; fm_chains = 1 }

(* [d[...] += x[...] * y[...]]: two operations per cell. *)
let mac d didx x xidx y yidx =
  Ll_test.set d didx
    (Ll_test.add (Ll_test.get d didx) (Ll_test.mul (Ll_test.get x xidx) (Ll_test.get y yidx)))

(* A batched product's nest: [nb] batch rows, [nr] blocks of [nw] threads, a serial reduction of
   [nk]; [batch_axis] binds the batch loop to the grid ([bgrid]) or leaves it serial (plain). *)
let nest ~tag ~batch_axis ~nb ~nr ~nw ~nk =
  let d = fresh_tn (tag ^ "_d") [| nb; nr; nw |]
  and x = fresh_tn (tag ^ "_x") [| nb; nr; nk |]
  and y = fresh_tn (tag ^ "_y") [| nk; nw |] in
  let b = Ll_test.sym () and r = Ll_test.sym () and w = Ll_test.sym () and k = Ll_test.sym () in
  let body =
    Ll_test.loop_n k nk (mac d [| it b; it r; it w |] x [| it b; it r; it k |] y [| it k; it w |])
  in
  match batch_axis with
  | LL.Grid ->
      Ll_test.loop_n ~axis:LL.Grid b nb
        (Ll_test.loop_n ~axis:LL.Grid r nr (Ll_test.loop_n ~axis:LL.Workgroup w nw body))
  | _ ->
      Ll_test.loop_n ~axis:LL.Grid r nr
        (Ll_test.loop_n ~axis:LL.Workgroup w nw (Ll_test.loop_n b nb body))

let ctx_list code =
  List.map (CM.launch_contexts code) ~f:(fun c ->
      (c.CM.lc_blocks, c.CM.lc_threads, c.CM.lc_tile_mma, c.CM.lc_ops))

let peak = 1024. (* operations/s and bytes/s: powers of two keep the quotients exact *)
let issue = 256.

let compute ?launch ?(mma_flops = 0) code =
  fst
    (CM.ranking_legs_of ~peak_flops:peak ~peak_memory_bandwidth:peak ~issue_rate:issue ~mma_flops
       ?launch (CM.analyze code) None)

let filled ?mma_flops code = compute ?mma_flops ~launch:(fm, CM.launch_contexts code) code

let () =
  let grid = nest ~tag:"cf_g" ~batch_axis:LL.Grid ~nb:2 ~nr:2 ~nw:64 ~nk:8 in
  let plain = nest ~tag:"cf_p" ~batch_axis:LL.Serial ~nb:2 ~nr:2 ~nw:64 ~nk:8 in
  let ops = 2 * 2 * 2 * 64 * 8 in
  p "the grid-bound nest is one context of 4 blocks of 64 threads holding all its operations"
    (List.equal Poly.equal (ctx_list grid) [ (4, 64, false, ops) ]);
  p "the serial-batch twin is one context of 2 blocks: its batch loop adds work, not blocks"
    (List.equal Poly.equal (ctx_list plain) [ (2, 64, false, ops) ]);
  p "4 blocks of 64 threads fill P=4 at t=64; 2 blocks fill half"
    (Float.equal (CM.launch_fill fm (CM.launch_contexts grid)) 1.
    && Float.equal (CM.launch_fill fm (CM.launch_contexts plain)) 0.5);
  p "without a fill model the twins tie on the compute leg"
    (Option.equal Float.equal (compute grid) (compute plain));
  p "with it the serial-batch twin's compute leg is exactly twice the grid-bound one's"
    (match (filled grid, filled plain) with
    | Some g, Some pl -> Float.(g = of_int ops /. issue) && Float.(pl = 2. *. g)
    | _ -> false);
  p "a fill model under which every context is full scores bitwise what no fill model does"
    (Option.equal Float.equal (filled grid) (compute grid))

let () =
  (* A wide zero nest -- 16 blocks of 64 threads, no operations -- beside a one-block
     contraction. *)
  let z = fresh_tn "cf_z" [| 16; 64 |] in
  let zb = Ll_test.sym () and zw = Ll_test.sym () in
  let zero =
    Ll_test.loop_n ~axis:LL.Grid zb 16
      (Ll_test.loop_n ~axis:LL.Workgroup zw 64 (Ll_test.set z [| it zb; it zw |] (Ll_test.c 0.)))
  in
  let narrow = nest ~tag:"cf_n" ~batch_axis:LL.Serial ~nb:2 ~nr:1 ~nw:64 ~nk:8 in
  let kernel = Ll_test.seq zero narrow in
  p "the zero nest contributes no context: only the contraction's 1 block of 64 threads"
    (List.equal Poly.equal (ctx_list kernel) [ (1, 64, false, 2 * 2 * 64 * 8) ]);
  p "so the kernel's fill is the contraction's quarter, though its launch spans 16 blocks"
    (Float.equal (CM.launch_fill fm (CM.launch_contexts kernel)) 0.25
    && Array.fold (LL.launch_dims kernel).LL.grid ~init:1 ~f:( * ) >= 16)

let () =
  (* Blocks wider than t: each counts one multiprocessor's saturating share. *)
  let wide nr =
    nest ~tag:(Printf.sprintf "cf_w%d" nr) ~batch_axis:LL.Grid ~nb:1 ~nr ~nw:128 ~nk:4
  in
  p
    "2 blocks of 128 threads fill half of P=4 at t=64 (not all of it: a block is on one \
     multiprocessor)"
    (Float.equal (CM.launch_fill fm (CM.launch_contexts (wide 2))) 0.5);
  p "8 blocks of 128 threads fill it"
    (Float.equal (CM.launch_fill fm (CM.launch_contexts (wide 8))) 1.)

let () =
  (* Two compute nests of equal work, one full and one at a quarter: the leg sums per context. *)
  let full = nest ~tag:"cf_f" ~batch_axis:LL.Grid ~nb:2 ~nr:2 ~nw:64 ~nk:8 in
  let quarter = nest ~tag:"cf_q" ~batch_axis:LL.Serial ~nb:4 ~nr:1 ~nw:64 ~nk:8 in
  let both = Ll_test.seq full quarter in
  let ops = 2 * 2 * 2 * 64 * 8 in
  p "two nests of equal work, at fills 1 and 1/4, price as 1 + 4 times one nest's work"
    (match filled both with
    | Some leg -> Float.(leg = (of_int ops +. (4. *. of_int ops)) /. issue)
    | None -> false)

let () =
  (* A tile-MMA statement inside one block of one lane: accepted work runs at the ceiling, declined
     work (its scalar fallback) at the issue rate, each times the fill of its context. *)
  let d = fresh_tn "cf_md" [| 2; 2 |]
  and a = fresh_tn "cf_ma" [| 2; 2 |]
  and b = fresh_tn "cf_mb" [| 2; 2 |] in
  let g = Ll_test.sym () in
  let at tn = (tn, [| Ll_test.fixed 0; Ll_test.fixed 0 |]) in
  let mma =
    Ll_test.loop_n ~axis:LL.Grid g 1 (Ll_test.tile_mma ~d:(at d) ~a:(at a) ~b:(at b) LL.Noop)
  in
  let tile_ops = (CM.analyze mma).CM.flops in
  let fill = CM.context_fill fm ~blocks:1 ~threads:1 in
  p "the tile-MMA statement is one tile context of 1 block"
    (List.equal Poly.equal (ctx_list mma) [ (1, 1, true, tile_ops) ]);
  p "accepted tile-MMA work prices at the ceiling times its context's fill"
    (Option.equal Float.equal (filled ~mma_flops:tile_ops mma)
       (Some (Float.of_int tile_ops /. fill /. peak)));
  p "declined tile-MMA work prices at the issue rate times its context's fill"
    (Option.equal Float.equal (filled mma) (Some (Float.of_int tile_ops /. fill /. issue)))

let () =
  (* Never below the bound: the compute leg is at least the operations at the ceiling. *)
  let kernels =
    [
      nest ~tag:"cf_b1" ~batch_axis:LL.Serial ~nb:8 ~nr:1 ~nw:16 ~nk:4;
      nest ~tag:"cf_b2" ~batch_axis:LL.Grid ~nb:8 ~nr:4 ~nw:64 ~nk:4;
    ]
  in
  p_all ~min:2 "the filled compute leg is never below the roofline's compute leg" kernels
    ~f:(fun k ->
      match filled k with
      | Some leg -> Float.(leg >= of_int (CM.analyze k).CM.flops /. peak)
      | None -> false)

(* gh-ocannl-1639: the per-thread ILP credit. [rt ~chains] is a register-tiled nest: [nb] blocks of
   [nw] threads, each accumulating [chains] private cells of [acc] over a serial reduction of [nk]
   -- as [chains] sibling statements (the materializing unroll the GPU blocktile applies to its
   register tile), or with [~unrolled:false] one statement under a serial loop over the cells. With
   [~batch], a serial batch loop of that extent encloses the reduction: a plain leaf's. *)
let rt ~tag ?(unrolled = true) ?(batch = 1) ~chains ~nb ~nw ~nk () =
  let acc = fresh_tn (tag ^ "_acc") [| batch; nb; nw; chains |]
  and x = fresh_tn (tag ^ "_x") [| batch; nb; nw; nk |]
  and y = fresh_tn (tag ^ "_y") [| nk; chains |] in
  let s = Ll_test.sym () and b = Ll_test.sym () and w = Ll_test.sym () and k = Ll_test.sym () in
  let cell ci =
    mac acc [| it s; it b; it w; ci |] x [| it s; it b; it w; it k |] y [| it k; ci |]
  in
  let body =
    if unrolled then
      List.reduce_exn ~f:Ll_test.seq (List.init chains ~f:(fun i -> cell (Ll_test.fixed i)))
    else
      let ci = Ll_test.sym () in
      Ll_test.loop_n ci chains (cell (it ci))
  in
  ( acc,
    Ll_test.loop_n ~axis:LL.Grid b nb
      (Ll_test.loop_n ~axis:LL.Workgroup w nw (Ll_test.loop_n s batch (Ll_test.loop_n k nk body)))
  )

let census ?(private_ = []) code =
  List.map
    (CM.launch_contexts ~is_private:(fun tn -> List.mem private_ tn ~equal:Tn.equal) code)
    ~f:(fun c -> c.CM.lc_chains)

let chains_are ?private_ code expected =
  List.equal (Option.equal Int.equal) (census ?private_ code) expected

(* The stand-in fill model with an ILP cap of 4 chains a thread. *)
let ilp = { fm with CM.fm_chains = 4 }

(* The cap resolves on its own, so a calibration whose fill model does not yet resolve still fits
   under the cap the score will read. *)
let () =
  let only_cap = { BI.no_hardware_limits with fill_chains = Some 8 } in
  p "the ILP cap resolves without the fill model's other constants, and is 1 where none is stated"
    (Autotune.ranking_fill_chains ~limits:only_cap = 8
    && Option.is_none (Autotune.ranking_fill_model ~limits:only_cap)
    && Autotune.ranking_fill_chains ~limits:BI.no_hardware_limits = 1)

let filled_under ?(private_ = []) model code =
  compute
    ~launch:
      (model, CM.launch_contexts ~is_private:(fun tn -> List.mem private_ tn ~equal:Tn.equal) code)
    code

let () =
  let acc4, tiled = rt ~tag:"cf_r4" ~chains:4 ~nb:4 ~nw:16 ~nk:8 () in
  let acc1, single = rt ~tag:"cf_r1" ~chains:1 ~nb:4 ~nw:16 ~nk:8 () in
  let acc16, wide = rt ~tag:"cf_r16" ~chains:16 ~nb:4 ~nw:16 ~nk:8 () in
  let accl, looped = rt ~tag:"cf_rl" ~unrolled:false ~chains:4 ~nb:4 ~nw:16 ~nk:8 () in
  let accd, deep = rt ~tag:"cf_rd" ~chains:4 ~nb:4 ~nw:16 ~nk:64 () in
  let accp, plain = rt ~tag:"cf_rp" ~batch:8 ~chains:1 ~nb:4 ~nw:16 ~nk:8 () in
  p "a 2x2 register tile's four unrolled accumulators are four chains a thread"
    (chains_are ~private_:[ acc4 ] tiled [ Some 4 ]);
  p "a one-accumulator reduction is one chain, a 4x4 tile sixteen"
    (chains_are ~private_:[ acc1 ] single [ Some 1 ]
    && chains_are ~private_:[ acc16 ] wide [ Some 16 ]);
  p "a serial loop over the cells inside the reduction counts its extent"
    (chains_are ~private_:[ accl ] looped [ Some 4 ]);
  p "a deeper reduction alone does not change the count"
    (chains_are ~private_:[ accd ] deep [ Some 4 ]);
  p "a plain leaf's serial batch loop, outside the reduction, is serial work, not chains"
    (chains_are ~private_:[ accp ] plain [ Some 1 ]);
  p "a global (non-private) accumulator counts no chain" (chains_are tiled [ None ]);
  (* One cell updated four times, four cells whose updates read each other, a guarded update, and
     elementwise work. *)
  let a = fresh_tn "cf_dep_a" [| 4 |] and x = fresh_tn "cf_dep_x" [| 8 |] in
  let k = Ll_test.sym () in
  let upd i v =
    Ll_test.set a [| Ll_test.fixed i |] (Ll_test.add (Ll_test.get a [| Ll_test.fixed i |]) v)
  in
  let xk = Ll_test.get x [| it k |] in
  let same =
    Ll_test.loop_n k 8 (List.reduce_exn ~f:Ll_test.seq (List.init 4 ~f:(fun _ -> upd 0 xk)))
  in
  let linked =
    Ll_test.loop_n k 8
      (List.reduce_exn ~f:Ll_test.seq
         (List.init 4 ~f:(fun i ->
              upd i
                (if i = 0 then xk else Ll_test.mul xk (Ll_test.get a [| Ll_test.fixed (i - 1) |])))))
  in
  let guarded = Ll_test.loop_n k 8 (Ll_test.if_ (Ll_test.c 1.) (upd 0 xk)) in
  let e = fresh_tn "cf_dep_e" [| 8 |] in
  let elementwise = Ll_test.loop_n k 8 (Ll_test.set e [| it k |] (Ll_test.mul xk xk)) in
  p "four updates of one cell are one chain" (chains_are ~private_:[ a ] same [ Some 1 ]);
  p "accumulators whose updates read each other are no independent chains"
    (chains_are ~private_:[ a ] linked [ None ]);
  p "a guarded update counts no chain" (chains_are ~private_:[ a ] guarded [ None ]);
  (* A repeated target's later update reading another accumulator: [a0 += x; a1 += x; a0 += a1]. *)
  let late =
    Ll_test.loop_n k 8
      (List.reduce_exn ~f:Ll_test.seq
         [ upd 0 xk; upd 1 xk; upd 0 (Ll_test.get a [| Ll_test.fixed 1 |]) ])
  in
  p "a repeated target's later update reading another accumulator makes the chains dependent"
    (chains_are ~private_:[ a ] late [ None ]);
  (* A guarded update coupling them: [a0 += x; a1 += x; if c then a0 += a1]. It counts no chain of
     its own, and still couples the two it writes and reads. *)
  let coupled =
    Ll_test.loop_n k 8
      (List.reduce_exn ~f:Ll_test.seq
         [
           upd 0 xk;
           upd 1 xk;
           Ll_test.if_ (Ll_test.c 1.) (upd 0 (Ll_test.get a [| Ll_test.fixed 1 |]));
         ])
  in
  p "a guarded update coupling two accumulators makes the chains dependent"
    (chains_are ~private_:[ a ] coupled [ None ]);
  (* Single cells no fixed component tells apart: [a[i] += x; a[0] += x] under an enclosing i < 2 --
     at i = 0 they are one chain. *)
  let oi = Ll_test.sym () in
  let maybe_one =
    Ll_test.loop_n oi 2
      (Ll_test.loop_n k 8
         (Ll_test.seq
            (Ll_test.set a [| it oi |] (Ll_test.add (Ll_test.get a [| it oi |]) xk))
            (upd 0 xk)))
  in
  p "two single cells no fixed component tells apart may alias: no chains"
    (chains_are ~private_:[ a ] maybe_one [ None ]);
  (* An update whose contribution is a [Local_scope] reading another member: [a0 += scope(a1 * x);
     a1 += x]. *)
  let tmp = fresh_tn "cf_dep_t" [| 1 |] in
  let scoped =
    Ll_test.loop_n k 8
      (Ll_test.seq
         (upd 0 (Ll_test.scope tmp (Ll_test.mul (Ll_test.get a [| Ll_test.fixed 1 |]) xk)))
         (upd 1 xk))
  in
  p "an update reading through a scope body is not proven independent: no chains"
    (chains_are ~private_:[ a ] scoped [ None ]);
  (* A runtime-indexed read of a member node may land on any member's cell: [a0 += a[dyn]; a1 +=
     x]. *)
  let gathered =
    Ll_test.loop_n k 8
      (Ll_test.seq
         (upd 0
            (Ll_test.gather ~tn:a
               ~idcs:[| Ll_test.fixed 0 |]
               ~dyn_axis:0
               ~dyn_value:(Ll_test.c 1., Ir.Ops.int32)))
         (upd 1 xk))
  in
  p "an update gathering a member node at a runtime index is not proven independent: no chains"
    (chains_are ~private_:[ a ] gathered [ None ]);
  (* A dependence carried by a temporary the step writes: [tmp = a[1]; a0 += tmp; a1 += x]. *)
  let tv = fresh_tn "cf_dep_v" [| 1 |] in
  let through_tmp =
    Ll_test.loop_n k 8
      (List.reduce_exn ~f:Ll_test.seq
         [
           Ll_test.set tv [| Ll_test.fixed 0 |] (Ll_test.get a [| Ll_test.fixed 1 |]);
           upd 0 (Ll_test.get tv [| Ll_test.fixed 0 |]);
           upd 1 xk;
         ])
  in
  p "an update reading a temporary the reduction step writes is not proven independent: no chains"
    (chains_are ~private_:[ a; tv ] through_tmp [ None ]);
  (* A zero fill of a member inside the step: no accumulation is carried. *)
  let zeroed =
    Ll_test.loop_n k 8 (List.reduce_exn ~f:Ll_test.seq [ Ll_test.zero a; upd 0 xk; upd 1 xk ])
  in
  p "a member zeroed inside the reduction step carries no chain"
    (chains_are ~private_:[ a ] zeroed [ None ]);
  (* Inner loops that do not address distinct cells: [acc[i + j]] over i, j < 4 is 7 cells, not 16;
     and a looped member beside a single cell of the same node may overlap it. *)
  let s7 = fresh_tn "cf_dep_s" [| 7 |] in
  let i = Ll_test.sym () and j = Ll_test.sym () in
  let sum_idx = Ll_test.aff [ (1, i); (1, j) ] 0 in
  let aliased =
    Ll_test.loop_n k 8
      (Ll_test.loop_n i 4
         (Ll_test.loop_n j 4
            (Ll_test.set s7 [| sum_idx |] (Ll_test.add (Ll_test.get s7 [| sum_idx |]) xk))))
  in
  p "inner loops whose cells may alias prove no chains"
    (chains_are ~private_:[ s7 ] aliased [ None ]);
  let overlap =
    Ll_test.loop_n k 8
      (Ll_test.seq
         (Ll_test.loop_n i 4 (Ll_test.set a [| it i |] (Ll_test.add (Ll_test.get a [| it i |]) xk)))
         (upd 2 xk))
  in
  p "a looped member beside a single cell of its node may overlap it: no chains"
    (chains_are ~private_:[ a ] overlap [ None ]);
  p "elementwise work counts no chain" (chains_are ~private_:[ e ] elementwise [ None ]);
  (* The fill: 4 blocks of 16 threads, a quarter of P = 4 at t = 64. With 4 chains a thread (at the
     cap) the tile keeps that quarter; its one-accumulator twin's threads count a quarter each: 4 *
     16 / 4 threads a block, a sixteenth of the device. *)
  let ops = 4 * 16 * 8 * 2 in
  p "at the cap, the register tile fills a quarter: bitwise its thread-only fill"
    (Float.equal (CM.context_fill ~chains:4 ilp ~blocks:4 ~threads:16) 0.25
    && Float.equal
         (CM.context_fill ~chains:4 ilp ~blocks:4 ~threads:16)
         (CM.context_fill fm ~blocks:4 ~threads:16));
  p "its one-accumulator twin fills a sixteenth"
    (Float.equal (CM.context_fill ~chains:1 ilp ~blocks:4 ~threads:16) 0.0625);
  p "the compute legs are each kernel's operations over its fill: 4x the work at 4x the fill"
    (match
       (filled_under ~private_:[ acc4 ] ilp tiled, filled_under ~private_:[ acc1 ] ilp single)
     with
    | Some t4, Some t1 ->
        let ops4 = Float.of_int (4 * ops) and ops1 = Float.of_int ops in
        Float.(t4 = ops4 /. 0.25 /. issue) && Float.(t1 = ops1 /. 0.0625 /. issue)
    | _ -> false);
  p "so per operation the register-tiled context ranks strictly above its one-accumulator twin"
    (match
       (filled_under ~private_:[ acc4 ] ilp tiled, filled_under ~private_:[ acc1 ] ilp single)
     with
    | Some t4, Some t1 -> Float.(t4 /. 4. < t1)
    | _ -> false);
  p "one block of 4-chain threads still fills only its share: a sixteenth"
    (Float.equal (CM.context_fill ~chains:4 ilp ~blocks:1 ~threads:16) 0.0625);
  p_all ~min:4 "chains past the cap buy nothing more"
    [ 4; 5; 16; 1 lsl 20 ]
    ~f:(fun c -> Float.equal (CM.context_fill ~chains:c ilp ~blocks:4 ~threads:16) 0.25);
  p "two chains under the cap of 4 count half a thread each"
    (Float.equal (CM.context_fill ~chains:2 ilp ~blocks:4 ~threads:16) 0.125);
  (* Controls: the cap of 1 is the thread-only fill for every chain count; unknown chains keep it at
     every cap; and the gh-1479 kernels above score bitwise at cap 1. *)
  let shapes = List.cartesian_product [ 1; 3; 4; 9 ] [ 1; 16; 64; 128 ] in
  p_all ~min:16 "at cap 1 every chain count fills as the thread-only fill" shapes
    ~f:(fun (blocks, threads) ->
      List.for_all [ 1; 2; 4; 16 ] ~f:(fun c ->
          Float.equal
            (CM.context_fill ~chains:c fm ~blocks ~threads)
            (CM.context_fill fm ~blocks ~threads)));
  p_all ~min:16 "unknown chains fill as the thread-only fill at any cap" shapes
    ~f:(fun (blocks, threads) ->
      List.for_all [ 1; 2; 4; 16 ] ~f:(fun l ->
          Float.equal
            (CM.context_fill { fm with CM.fm_chains = l } ~blocks ~threads)
            (CM.context_fill fm ~blocks ~threads)));
  p_all ~min:16 "a context at or past the cap fills bitwise as the thread-only fill at any cap"
    shapes ~f:(fun (blocks, threads) ->
      List.for_all [ 1; 2; 4; 16 ] ~f:(fun l ->
          Float.equal
            (CM.context_fill ~chains:16 { fm with CM.fm_chains = l } ~blocks ~threads)
            (CM.context_fill fm ~blocks ~threads)));
  p "the one-accumulator kernel scores bitwise as before at cap 1"
    (Option.equal Float.equal (filled_under ~private_:[ acc1 ] fm single) (filled single));
  (* The tail: 48 blocks over 32 multiprocessors of t = 64, blocks of 64 threads: 16 of them run
     two. Thread-only the busiest is full and the launch fills 48/32/2 = 3/4; one chain under a cap
     of 4 counts 16 threads a block, the busiest runs at 32/64, and the launch fills 3/8. *)
  let tail = { CM.fm_multiprocessors = 32; fm_fill_threads = 64; fm_tail = true; fm_chains = 4 } in
  p "under the tail, 48 blocks of 64 one-chain threads fill 3/8, of four-chain ones 3/4"
    (Float.equal (CM.context_fill ~chains:1 tail ~blocks:48 ~threads:64) 0.375
    && Float.equal (CM.context_fill ~chains:4 tail ~blocks:48 ~threads:64) 0.75
    && Float.equal
         (CM.context_fill ~chains:4 tail ~blocks:48 ~threads:64)
         (CM.context_fill { tail with fm_chains = 1 } ~blocks:48 ~threads:64));
  (* Mixed chains in one launch shape: two contexts, each over its own fill. *)
  let both = Ll_test.seq tiled single in
  p "equal launch shapes of 4 and 1 chains are two contexts, priced each over its own fill"
    (List.equal (Option.equal Int.equal)
       (List.sort ~compare:(Option.compare Int.compare) (census ~private_:[ acc4; acc1 ] both))
       [ Some 1; Some 4 ]
    &&
    match filled_under ~private_:[ acc4; acc1 ] ilp both with
    | Some leg ->
        let ops4 = Float.of_int (4 * ops) and ops1 = Float.of_int ops in
        Float.(leg = ((ops4 /. 0.25) +. (ops1 /. 0.0625)) /. issue)
    | None -> false);
  (* At cap 1 the census changes nothing, bitwise, even where it splits one launch shape into
     contexts of different chains: under a non-dyadic stand-in (P = 46, t = 256), 7 blocks of 16
     threads with 2 and 3 operations a thread summed each over its own (rounded) fill differ from
     their sum over it in the last ulp -- the score merges them back first. *)
  let rog_like ~tail =
    { CM.fm_multiprocessors = 46; fm_fill_threads = 256; fm_tail = tail; fm_chains = 1 }
  in
  let acc7, one7 = rt ~tag:"cf_m7" ~chains:1 ~nb:7 ~nw:16 ~nk:1 () in
  let e = fresh_tn "cf_mix_e" [| 7; 16 |] and xe = fresh_tn "cf_mix_x" [| 7; 16 |] in
  let mb = Ll_test.sym () and mw = Ll_test.sym () in
  let xv = Ll_test.get xe [| it mb; it mw |] in
  let elementwise_beside =
    Ll_test.loop_n ~axis:LL.Grid mb 7
      (Ll_test.loop_n ~axis:LL.Workgroup mw 16
         (Ll_test.set e
            [| it mb; it mw |]
            (Ll_test.add (Ll_test.mul xv (Ll_test.add xv (Ll_test.c 0.5))) xv)))
  in
  let mixed = Ll_test.seq one7 elementwise_beside in
  let private_ = [ acc7 ] in
  let split model =
    List.fold
      (CM.launch_contexts ~is_private:(fun tn -> List.mem private_ tn ~equal:Tn.equal) mixed)
      ~init:0.
      ~f:(fun w c ->
        w
        +. Float.of_int c.CM.lc_ops
           /. CM.context_fill ?chains:c.CM.lc_chains model ~blocks:c.CM.lc_blocks
                ~threads:c.CM.lc_threads)
  in
  p "the mixed kernel's census splits one launch shape in two (1 and uncounted chains)"
    (List.length (census ~private_ mixed) = 2);
  p_all ~min:2
    "at cap 1 a split launch shape scores bitwise as with no census, where an unmerged sum would \
     not"
    [ rog_like ~tail:false; rog_like ~tail:true ]
    ~f:(fun model ->
      let unsplit = compute ~launch:(model, CM.launch_contexts mixed) mixed in
      Option.equal Float.equal (filled_under ~private_ model mixed) unsplit
      && Float.equal
           (CM.launch_fill model
              (CM.launch_contexts
                 ~is_private:(fun tn -> List.mem private_ tn ~equal:Tn.equal)
                 mixed))
           (CM.launch_fill model (CM.launch_contexts mixed))
      && not (Option.equal Float.equal (Some (split model /. issue)) unsplit));
  p "the chains change no operation count: the contexts still sum to the census"
    (List.sum
       (module Int)
       (CM.launch_contexts ~is_private:(fun tn -> Tn.equal tn acc4 || Tn.equal tn acc1) both)
       ~f:(fun c -> c.CM.lc_ops)
    = (CM.analyze both).CM.flops)

(* The q/k/v projection at b8, scored per scalar blocktile leaf under a CUDA-like record. *)
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
    cache_capacity_bytes = Some (12 * 1024 * 1024 * 1024);
    cache_bandwidth = Some 1.8e12;
    max_resident_threads = Some (70 * 1536);
    issue_rate = Some 3.4e12;
    dram_bandwidth = Some 6.0e11;
    multiprocessors = Some 70;
    fill_threads = Some 128;
  }

let without_fill = { stand_in with multiprocessors = None; fill_threads = None }

let () =
  let ctx = Context.cpu () in
  let cell idx = Float.of_int (1 + (Array.fold idx ~init:0 ~f:( + ) % 7)) *. 0.125 in
  let w = NTDSL.init ~l:"cf_wqkv" ~prec:Ir.Ops.single ~o:[ 8; 32 ] ~i:[ 256 ] ~f:cell () in
  let x = NTDSL.init ~l:"cf_xqkv" ~prec:Ir.Ops.single ~b:[ 8; 128 ] ~o:[ 256 ] ~f:cell () in
  let%op d = w * x in
  Train.set_materialized d.Tensor.value;
  let lowered =
    Context.lowered_for_decisions ~name:"cf_qkv" ctx (Train.forward d) Ir.Indexing.Empty
  in
  let hermetic (o : LL.optimized) =
    {
      o with
      LL.traced_store = Hashtbl.copy o.LL.traced_store;
      LL.optimize_ctx = LL.copy_optimize_ctx o.LL.optimize_ctx;
    }
  in
  let leaves =
    match Autotune.matmul_sketch_tree ~is_gpu:true ~is_cpu:false ~limits:stand_in lowered with
    | None -> []
    | Some tree -> List.filter (Ir.Schedule_space.leaves tree) ~f:Autotune.scalar_blocktile
  in
  let scored =
    List.filter_map leaves ~f:(fun p ->
        match
          Ir.Schedule.apply
            (Autotune.sketch_schedule ~accum_prec:Fn.id ~p (hermetic lowered))
            (hermetic lowered)
        with
        | exception _ -> None
        | post -> (
            match
              ( Autotune.ranking_ms ~limits:stand_in [ post ],
                Autotune.ranking_ms ~limits:without_fill [ post ] )
            with
            | Some (rank, bound), Some (rank_off, _) -> Some (p, rank, bound, rank_off, post)
            | _ -> None))
  in
  (* gh-ocannl-1639: the census on the leaves as scheduled -- privatized, the register tile unrolled
     -- and the cap's invariance on the 16-chain tiles the launch sweep fits [fill_threads] on. *)
  let main_chains post =
    CM.launch_contexts ~is_private:(CM.private_in post) post.LL.llc
    |> List.max_elt ~compare:(fun a b -> Int.compare a.CM.lc_ops b.CM.lc_ops)
    |> Option.bind ~f:(fun c -> c.CM.lc_chains)
  in
  List.iter scored ~f:(fun (p, _, _, _, post) ->
      Stdio.eprintf "cost_model_fill: %s chains %s (not part of the golden)\n"
        (Autotune.sketch_params_label p)
        (Option.value_map (main_chains post) ~default:"none" ~f:Int.to_string));
  p_all ~min:4 "every scheduled blocktile leaf's reduction carries its register tile's tm*tn chains"
    scored ~f:(fun (p, _, _, _, post) ->
      Option.equal Int.equal (main_chains post) (Some (p.Autotune.sk_tm * p.Autotune.sk_tn)));
  let capped = { stand_in with fill_chains = Some 8 } in
  let tiles16 =
    List.filter scored ~f:(fun (p, _, _, _, _) -> p.Autotune.sk_tm * p.Autotune.sk_tn >= 16)
  in
  p_all ~min:2 "a 16-chain leaf ranks bitwise as thread-only under a cap of 8" tiles16
    ~f:(fun (_, rank, _, _, post) ->
      match Autotune.ranking_ms ~limits:capped [ post ] with
      | Some (r, _) -> Float.equal r rank
      | None -> false);
  let tiles4 =
    List.filter scored ~f:(fun (p, _, _, _, _) -> p.Autotune.sk_tm * p.Autotune.sk_tn < 8)
  in
  p_all ~min:2 "a fewer-chain leaf ranks no lower under a cap of 8" tiles4
    ~f:(fun (_, rank, _, _, post) ->
      match Autotune.ranking_ms ~limits:capped [ post ] with
      | Some (r, _) -> Float.(r >= rank)
      | None -> false);
  p_exists "some under-filled fewer-chain leaf ranks strictly higher under a cap of 8" tiles4
    ~f:(fun (_, rank, _, _, post) ->
      match Autotune.ranking_ms ~limits:capped [ post ] with
      | Some (r, _) -> Float.(r > rank)
      | None -> false);
  (* gh-ocannl-1639 ships the census with no cap advised on any backend (the rog fit moved no order
     at the gh-1479 cells): under the running backend's own limits -- cc's, or CUDA's on its leg --
     every leaf ranks bitwise as under the thread-only fill. Setting a backend's [fill_chains] turns
     this claim red: that is the change that needs its own evidence. *)
  let own = Context.hardware_limits (Context.auto ()) in
  let thread_only = { own with fill_chains = Some 1 } in
  Stdio.eprintf "cost_model_fill: the running backend's fill model: %s (not part of the golden)\n"
    (Option.value_map (Autotune.ranking_fill_model ~limits:own) ~default:"none" ~f:(fun fm ->
         Sexp.to_string (CM.sexp_of_fill_model fm)));
  let fill_leg =
    Verdict.leg
      ~when_:(Option.is_some (Autotune.ranking_fill_model ~limits:own))
      ~on:(match Utils.get_global_arg ~arg_name:"backend" ~default:"" with "" -> "cc" | b -> b)
      ()
  in
  in_leg ~leg:fill_leg
    "the running backend advises no ILP cap: every leaf ranks bitwise as under the thread-only fill"
    (fun label ->
      p_all ~min:4 label scored ~f:(fun (_, _, _, _, post) ->
          Option.equal
            (fun (r1, b1) (r2, b2) -> Float.equal r1 r2 && Float.equal b1 b2)
            (Autotune.ranking_ms ~limits:own [ post ])
            (Autotune.ranking_ms ~limits:thread_only [ post ])));
  let scored = List.map scored ~f:(fun (p, r, b, o, _) -> (p, r, b, o)) in
  List.iter scored ~f:(fun (p, rank, bound, off) ->
      Stdio.eprintf
        "cost_model_fill: %s rank %.6f ms (no fill %.6f ms, bound %.6f ms) (not part of the golden)\n"
        (Autotune.sketch_params_label p) rank off bound);
  let twins =
    List.filter_map scored ~f:(fun ((p, _, _, _) as plain) ->
        if p.Autotune.sk_batch_grid then None
        else
          List.find scored ~f:(fun (q, _, _, _) ->
              q.Autotune.sk_batch_grid
              && Sexp.equal
                   (Autotune.sexp_of_sketch_params { q with sk_batch_grid = false })
                   (Autotune.sexp_of_sketch_params p))
          |> Option.map ~f:(fun grid -> (plain, grid)))
  in
  p_all ~min:2 "with the fill constants every plain leaf ranks strictly above its bgrid twin" twins
    ~f:(fun ((_, plain, _, _), (_, grid, _, _)) -> Float.(grid < plain));
  p_all ~min:2 "without them the twins tie, as before the launch fill" twins
    ~f:(fun ((_, _, _, plain_off), (_, _, _, grid_off)) -> Float.equal plain_off grid_off);
  p_all ~min:2 "every leaf's rank is at least its bound" scored ~f:(fun (_, rank, bound, _) ->
      Float.(rank >= bound));
  p "the best-ranked scalar leaf is a bgrid one"
    (match List.min_elt scored ~compare:(fun (_, a, _, _) (_, b, _, _) -> Float.compare a b) with
    | Some (p, _, _, _) -> p.Autotune.sk_batch_grid
    | None -> false)

(* gh-ocannl-1639: the premise behind "no refit of fill_threads": the launch sweep's own kernels --
   Calibrate.default_launch_geometries' pinned 4x4 leaves of [dw = dy +* "ko; ki => oi" x] -- carry
   16 chains a thread, so every cap up to 16 leaves their fill, and the fit, as it was. *)
let () =
  let ctx = Context.cpu () in
  let size = 128 and k = 64 in
  let cyc len modulus stride =
    Array.init len ~f:(fun c -> Float.of_int (1 + (c % modulus)) *. stride)
  in
  let dy =
    NTDSL.ndarray (cyc (k * size) 7 0.25) ~label:[ "cf_sw_dy" ] ~output_dims:[ k; size ] ()
  in
  let x = NTDSL.ndarray (cyc (k * size) 5 0.125) ~label:[ "cf_sw_x" ] ~output_dims:[ k; size ] () in
  let%op dw = dy +* "ko; ki => oi" x in
  Train.set_materialized dw.Tensor.value;
  let lowered =
    Context.lowered_for_decisions ~name:"cf_sweep" ctx (Train.forward dw) Ir.Indexing.Empty
  in
  let geometries = Ocannl.Calibrate.default_launch_geometries in
  let leaves =
    match Autotune.matmul_sketch_tree ~is_gpu:true ~is_cpu:false ~limits:stand_in lowered with
    | None -> []
    | Some tree ->
        List.filter (Ir.Schedule_space.leaves tree) ~f:(fun p ->
            Autotune.scalar_blocktile p && (not p.Autotune.sk_batch_grid)
            && List.mem geometries (p.sk_bm, p.sk_bn, p.sk_bk, p.sk_tm, p.sk_tn) ~equal:Poly.equal)
  in
  p_all ~min:(List.length geometries)
    "every launch-sweep geometry's scheduled reduction carries 16 chains a thread" leaves
    ~f:(fun p ->
      match Ir.Schedule.apply (Autotune.sketch_schedule ~accum_prec:Fn.id ~p lowered) lowered with
      | exception _ -> false
      | post ->
          CM.launch_contexts ~is_private:(CM.private_in post) post.LL.llc
          |> List.max_elt ~compare:(fun a b -> Int.compare a.CM.lc_ops b.CM.lc_ops)
          |> Option.bind ~f:(fun c -> c.CM.lc_chains)
          |> Option.equal Int.equal (Some 16))
