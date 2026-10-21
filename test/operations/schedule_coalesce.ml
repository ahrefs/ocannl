(* gh-ocannl-1165: the [Coalesce] op and the coalesced tuner branch.

   The q/k/v projections' site [d[b,s,h,e] += w[h,e,k] * x[b,s,k]] carries its heads as an interior
   batch loop right above the column role, so no column tile is wider than one head; #915's bgrid-in
   flavor recovered the heads-merged layout's launch order, and the residual is tile width. Every
   access reads [h; e] as adjacent plain iterators over unpadded axes, so the pair is ONE loop to
   the compiler: [Sched.Coalesce] re-indexes it as [Sub_axis; Iterator f], and the coalesced branch
   ([Autotune.coalesced_seed_params]) enumerates the GPU scalar blocktile family over that lowering,
   where a 64-wide column tile spans two heads of useful columns.

   Four parts:

   - The op on hand-built nests (every backend, executed): the positive control applies, is
   [Op_legal], and computes the uncoalesced nest's values bitwise; a per-head operand (one symbol of
   the pair read alone) and a padded inner axis (dim larger than the loop extent: its stride is not
   the inner extent, so the composed index is not the address) decline, each for its own reason; and
   the per-tile index extension ([Sched.add_symbol]) refuses the flattened component it produced.

   - The real projection site through [%op]: the structural prefix, the coalesced site it produces
   (one column role over every head's columns), the prefix alone executed against a serial
   reference, and the branch's seeds — offered with column tiles wider than a head, every schedule
   constructing and validating with the merged column blocks on [.x], every schedule surviving the
   schedule cache's structural round trip, and on GPU backends every seed executed against the
   serial reference.

   - The v1 boundary at seeding: a companion nest over the uncoalesced axes (an elementwise tail of
   the projection) cannot share the merged chain's geometry, so the branch offers nothing there
   while the ordinary family still does.

   - A fission segment's site, the form the real step searches: its [Zero_out] lands in a segment of
   its own, and the unzeroed site coalesces by its own [Coalesce] alone.

   Inputs vary with every index and keep every partial sum exactly representable in f32, so bitwise
   equality is required whatever order a tiling accumulates in. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module LL = Ir.Low_level
module Sched = Ir.Schedule
module SC = Ir.Schedule_cache
module Asgns = Ir.Assignments
module L = Ll_test

(* The backend's accumulator residency, which a [Privatize] tile is minted at (gh-ocannl-1116). *)
let accum_prec = Ll_test.backend_accum_prec

open Verdict.Claims

let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")
let on_gpu = Sched.backend_is_gpu backend_name

let raises_with ~substring f =
  match f () with
  | _ -> false
  | exception Invalid_argument msg -> String.is_substring msg ~substring

(* {1 The op on hand-built nests} *)

let () =
  let bb = 2 and ss = 3 and hh = 3 and ee = 4 and kk = 5 in
  let mk = L.node_factory ~first_id:116500 ~dims:[||] () in
  let w = mk ~dims:[| hh; ee; kk |] "co_w" and x = mk ~dims:[| bb; ss; kk |] "co_x" in
  let g = mk ~dims:[| hh |] "co_g" in
  List.iter [ w; x; g ] ~f:L.materialize;
  (* [d[b,s,h,e] := 0; for k: d[b,s,h,e] += w[h,e,k] * x[b,s,k] * extra] over the nest b, s, h, e,
     with the pair (h, e) returned for the op to name. *)
  let nest ~dst ~extra =
    let b = L.sym () and s = L.sym () and h = L.sym () and e = L.sym () and k = L.sym () in
    let at = [| L.iter b; L.iter s; L.iter h; L.iter e |] in
    let prod =
      L.mul
        (L.get w [| L.iter h; L.iter e; L.iter k |])
        (L.get x [| L.iter b; L.iter s; L.iter k |])
    in
    let rhs = match extra with None -> prod | Some f -> L.mul prod (f ~h) in
    let llc =
      L.loop_n b bb
        (L.loop_n s ss
           (L.loop_n h hh
              (L.loop_n e ee
                 (L.seq
                    (L.set dst at (L.c 0.))
                    (L.loop_n k kk (L.set dst at (L.add (L.get dst at) rhs)))))))
    in
    (llc, h, e)
  in
  let seed =
    [
      ( w,
        Array.init
          (hh * ee * kk)
          ~f:(L.cycle_flat ~dims:[| hh; ee; kk |] ~modulus:11 ~offset:(-5.) ~stride:0.5) );
      ( x,
        Array.init
          (bb * ss * kk)
          ~f:(L.cycle_flat ~dims:[| bb; ss; kk |] ~modulus:13 ~offset:(-6.) ~stride:0.25) );
      (g, Array.init hh ~f:(fun i -> Float.of_int (i + 1)));
    ]
  in
  (* Positive control: applies, is [Op_legal], merges the pair into one loop of extent [hh * ee]
     whose accesses read [Sub_axis; Iterator f], and computes what the uncoalesced nest computes. *)
  let d = mk ~dims:[| bb; ss; hh; ee |] "co_d" in
  L.materialize d;
  let llc, h, e = nest ~dst:d ~extra:None in
  let o = L.optimize ~name:"co_plain" llc in
  let op, merged = Sched.coalesce ~outer:h ~inner:e in
  p "control: Coalesce of the head pair is Op_legal"
    (Sched.equal_op_verdict (Sched.op_legality o op) Sched.Op_legal);
  let oc = Sched.apply [ op ] o in
  let merged_extent = ref None and flattened_writes = ref 0 and flattened_pair = ref None in
  L.walk oc.LL.llc ~on_stmt:(function
    | LL.For_loop { index; to_; _ } when Ir.Indexing.equal_symbol index merged ->
        merged_extent := Some (to_ + 1)
    | LL.Set { tn; idcs; _ }
      when Ir.Tnode.equal tn d
           && Array.equal Ir.Indexing.equal_axis_index (Array.sub idcs ~pos:2 ~len:2)
                [| Ir.Indexing.Sub_axis; Ir.Indexing.Iterator merged |] ->
        Int.incr flattened_writes;
        flattened_pair := Some (idcs.(2), idcs.(3))
    | _ -> ());
  p "control: the merged loop spans every head's columns"
    (Poly.equal !merged_extent (Some (hh * ee)));
  p "control: the writes read the merged index flattened over the pair" (!flattened_writes > 0);
  (* The per-tile index extension [Tensorize] and [Fuse_epilogue] apply to an accumulator's base
     refuses the flattened component this op produced -- extended, it would read as an ordinary
     coordinate -- and extends the merged iterator beside it. *)
  let tile = L.sym () in
  p "control: add_symbol refuses the flattened (Sub_axis) component of a coalesced write"
    (match !flattened_pair with
    | Some (flat, _) -> Option.is_none (Sched.add_symbol flat tile)
    | None -> false);
  p "control: add_symbol extends the merged iterator of a coalesced write by the tile symbol"
    (match !flattened_pair with
    | Some (_, it) ->
        Option.equal Ir.Indexing.equal_axis_index (Sched.add_symbol it tile)
          (Some (Ir.Indexing.affine ~symbols:[ (1, tile); (1, merged) ] ~offset:0))
    | None -> false);
  let run name o = List.hd_exn (L.execute ~name o ~seed:(List.take seed 2) ~read:[ d ]) in
  let want = run "co_plain_run" o in
  let got = run "co_merged_run" oc in
  p "control: the coalesced nest computes the uncoalesced one's values bitwise"
    (Array.exists want ~f:(fun v -> Float.(v <> 0.)) && Array.equal Float.equal want got);
  (* A per-head operand reads [h] alone: the composed index cannot stand for it. *)
  let dg = mk ~dims:[| bb; ss; hh; ee |] "co_dg" in
  L.materialize dg;
  let llc, h, e = nest ~dst:dg ~extra:(Some (fun ~h -> L.get g [| L.iter h |])) in
  let o = L.optimize ~name:"co_per_head" llc in
  let op = fst (Sched.coalesce ~outer:h ~inner:e) in
  p "per-head operand: Coalesce declines because a read does not hold the pair"
    (raises_with ~substring:"does not read" (fun () -> Sched.apply [ op ] o));
  p "per-head operand: the oracle proves it illegal"
    (match Sched.op_legality o op with Sched.Op_illegal _ -> true | _ -> false);
  (* A padded inner axis: the node's dim exceeds the loop's extent, so the inner stride is not the
     inner extent and [ee*h + e] is not the address. *)
  let dp = mk ~dims:[| bb; ss; hh; ee + 2 |] "co_dp" in
  L.materialize dp;
  let llc, h, e = nest ~dst:dp ~extra:None in
  let o = L.optimize ~name:"co_padded" llc in
  let op = fst (Sched.coalesce ~outer:h ~inner:e) in
  p "padded inner axis: Coalesce declines on the dims"
    (raises_with ~substring:"padded axis" (fun () -> Sched.apply [ op ] o));
  p "padded inner axis: the oracle proves it illegal"
    (match Sched.op_legality o op with Sched.Op_illegal _ -> true | _ -> false)

(* A pair right after an existing flattened run: [ds[Sub_axis; h; e]] over [[2; hh; ee]] already
   reads [h] as a flattened index over the run, and the merged one would flatten over it too, so the
   view would bound [merged] by [2 * hh * ee] rather than the [hh * ee] it ranges over. *)
let () =
  let hh = 3 and ee = 4 in
  let mk = L.node_factory ~first_id:116700 ~dims:[||] () in
  let ds = mk ~dims:[| 2; hh; ee |] "cr_ds" in
  L.materialize ds;
  let h = L.sym () and e = L.sym () in
  let llc =
    L.loop_n h hh (L.loop_n e ee (L.set ds [| Ir.Indexing.Sub_axis; L.iter h; L.iter e |] (L.c 1.)))
  in
  let o = L.optimize ~name:"cr_after_run" llc in
  let op = fst (Sched.coalesce ~outer:h ~inner:e) in
  p "after a flattened run: Coalesce declines because the pair follows a Sub_axis run"
    (raises_with ~substring:"after a flattened (Sub_axis) run" (fun () -> Sched.apply [ op ] o));
  p "after a flattened run: the oracle proves it illegal"
    (match Sched.op_legality o op with Sched.Op_illegal _ -> true | _ -> false)

(* {1 Composition: a coalesced pair under [Split_reduce]}

   The coalesced output [d[Sub_axis; f]] of a reduction [d[h,e] += x[k,h,e]] is split over [k]:
   [Split_reduce] derives the cell of its combine nest from the original cell with the enclosing
   loops renamed to the combine indices. Rebuilt instead from the cell's decomposition, where a
   [Sub_axis] has none, the marker comes back as [Fixed_idx 0]: the combine's write renders the same
   address but reads to the address queries as an ordinary in-bounds coordinate, under which the
   flattened component [f] (extent [hh * ee]) is taken to stay below [ee]. The executed leg runs the
   composition through the default (fissioned, on GPU hardware-mapped) schedules. *)

let () =
  let hh = 3 and ee = 4 and kk = 16 in
  let mk = L.node_factory ~first_id:116600 ~dims:[||] () in
  let x = mk ~dims:[| kk; hh; ee |] "cs_x" and d = mk ~dims:[| hh; ee |] "cs_d" in
  List.iter [ x; d ] ~f:L.materialize;
  let h = L.sym () and e = L.sym () and k = L.sym () in
  let at = [| L.iter h; L.iter e |] in
  let llc =
    L.seq (L.zero d)
      (L.loop_n h hh
         (L.loop_n e ee
            (L.loop_n k kk
               (L.set d at (L.add (L.get d at) (L.get x [| L.iter k; L.iter h; L.iter e |]))))))
  in
  let seed =
    [
      ( x,
        Array.init
          (kk * hh * ee)
          ~f:(L.cycle_flat ~dims:[| kk; hh; ee |] ~modulus:13 ~offset:(-6.) ~stride:0.5) );
    ]
  in
  let o = L.optimize ~name:"cs_plain" llc in
  let co, _ = Sched.coalesce ~outer:h ~inner:e in
  let sr, _, _, _ = Sched.split_reduce ~axis:k ~target:d ~num_blocks:4 in
  let os = Sched.apply [ co; sr ] o in
  let d_writes = ref [] in
  L.walk os.LL.llc ~on_stmt:(function
    | LL.Set { tn; idcs; _ } when Ir.Tnode.equal tn d -> d_writes := idcs :: !d_writes
    | _ -> ());
  p_all "split-reduce: every write of the coalesced cell, the combine's included, stays flattened"
    !d_writes ~f:(fun idcs -> (Ir.Affine.axis_extents ~dims:[| hh; ee |] idcs).(1) = hh * ee);
  let run name o transform =
    let ctx, routine =
      Context.compile ~name ~prelowered:o ~lowered_transform:transform (Context.auto ())
        Ir.Assignments.empty_comp Ir.Indexing.Empty
    in
    let ctx = List.fold seed ~init:ctx ~f:(fun ctx (tn, vs) -> Context.set_values ctx tn vs) in
    Context.get_values (Context.run ctx routine) d
  in
  let want = run "cs_plain_run" o (fun o -> [ o ]) in
  let got =
    run "cs_split_run" os
      (Sched.maybe_default_schedules ~backend_name
         ~limits:(Context.hardware_limits (Context.auto ()))
         ~static_indices:[])
  in
  p
    "split-reduce: the coalesced, split, default-scheduled reduction computes the plain one's \
     values"
    (Array.exists want ~f:(fun v -> Float.(v <> 0.)) && Array.equal Float.equal want got)

(* {1 Composition: a coalesced pair under [Stage] and [Privatize]}

   Both insert nests that access the coalesced node itself — [Stage] the source read of its load
   nest, [Privatize] the target's init-load and store-back transfers — and both derive that access
   from the original one, renaming the tile loops' symbols to the nest's fresh ones. A [Sub_axis]
   has no symbols, so the rename keeps it; derived instead by rebuilding each component from its
   affine decomposition, where a [Sub_axis] has none, it comes back as [Fixed_idx 0]: the same
   address, read by the address queries as an ordinary coordinate under which the flattened
   component after it stays below its own axis's dim. After both ops every remaining access of the
   two nodes is one of those derived accesses (the computation reads the tiles), so the structural
   claims below read exactly what the derivation wrote. Every backend runs this leg: non-shared
   [Stage] and [Privatize] are the CPU packing pair. *)

let () =
  let hh = 3 and ee = 4 and kk = 5 in
  let mk = L.node_factory ~first_id:116800 ~dims:[||] () in
  let x = mk ~dims:[| kk; hh; ee |] "cp_x" and d = mk ~dims:[| hh; ee |] "cp_d" in
  List.iter [ x; d ] ~f:L.materialize;
  let h = L.sym () and e = L.sym () and k = L.sym () in
  let at = [| L.iter h; L.iter e |] in
  let llc =
    L.seq (L.zero d)
      (L.loop_n h hh
         (L.loop_n e ee
            (L.loop_n k kk
               (L.set d at (L.add (L.get d at) (L.get x [| L.iter k; L.iter h; L.iter e |]))))))
  in
  let seed =
    [
      ( x,
        Array.init
          (kk * hh * ee)
          ~f:(L.cycle_flat ~dims:[| kk; hh; ee |] ~modulus:13 ~offset:(-6.) ~stride:0.5) );
    ]
  in
  let o = L.optimize ~name:"cp_plain" llc in
  let co, f = Sched.coalesce ~outer:h ~inner:e in
  let sp, _f_o, f_i = Sched.split ~axis:f ~factor:4 ~outer:LL.Serial ~inner:LL.Serial in
  let stage =
    Sched.Stage
      {
        source = x;
        tile_loops = [ f_i; k ];
        shared = false;
        cooperative = None;
        hoisted = false;
        swizzle = None;
        pad_stride = None;
        pipeline_depth = 1;
        tile_prec = None;
      }
  in
  let os = Sched.apply [ co; sp; stage; Sched.privatize ~accum_prec ~target:d ~over:k ] o in
  let x_reads = ref [] and d_accesses = ref [] in
  L.walk os.LL.llc
    ~on_stmt:(function
      | LL.Set { tn; idcs; _ } when Ir.Tnode.equal tn d -> d_accesses := idcs :: !d_accesses
      | _ -> ())
    ~on_scalar:(function
      | LL.Get (tn, idcs) when Ir.Tnode.equal tn x -> x_reads := idcs :: !x_reads
      | LL.Get (tn, idcs) when Ir.Tnode.equal tn d -> d_accesses := idcs :: !d_accesses
      | _ -> ());
  p_all "stage: the load nest's read of the coalesced source stays flattened" !x_reads
    ~f:(fun idcs -> (Ir.Affine.axis_extents ~dims:[| kk; hh; ee |] idcs).(2) = hh * ee);
  p_all "privatize: the transfers' accesses of the coalesced target stay flattened" !d_accesses
    ~f:(fun idcs -> (Ir.Affine.axis_extents ~dims:[| hh; ee |] idcs).(1) = hh * ee);
  let want = List.hd_exn (L.execute ~name:"cp_plain_run" o ~seed ~read:[ d ]) in
  let got = List.hd_exn (L.execute ~name:"cp_packed_run" os ~seed ~read:[ d ]) in
  p "stage+privatize: the coalesced, packed reduction computes the plain one's values bitwise"
    (Array.exists want ~f:(fun v -> Float.(v <> 0.)) && Array.equal Float.equal want got)

(* {1 The contraction pair: the out projection's [(h, e)] (gh-ocannl-1279)}

   The legs above coalesce an OUTPUT pair. Attention's out projection [d[s,j] += w[j,h,e] *
   x[s,h,e]] contracts over its pair instead: both operands read [(h, e)] as adjacent plain
   iterators and the accumulator reads neither, so [Coalesce] makes the contraction one k loop of
   extent [h*e]. What the tiling needs from the ops on that merged k, pinned on hand-built nests
   before any seed relies on it:

   - [Pad] + [Split] along the merged k with a block that does not divide the head ([e = 5], block
   4, odd [K = 15]): tiles cross head boundaries and the last one runs past [K]; - [Stage] of BOTH
   operands at the k-block loop: each load nest reads its source through the flattened address and
   guards the fringe against the merged extent [K], not against [e]; - [Privatize] over the k-block
   loop, with the pad's iteration mask on the leaf; - an unrelated second reader of the activation,
   in its own nest, keeps reading the source; - [Split_reduce] on the merged k of a small-output
   contraction; - a weight stored [w[h, j, e]] (the pair not adjacent), a per-head-only operand, and
   a padded inner axis each make [Coalesce] decline.

   Inputs are multiples of 1/4 and 1/2 with partial sums far below 2^24, so every order of
   accumulation is exact in f32 and the claims are bitwise. *)

let () =
  let ss = 4 and jj = 6 and hh = 3 and ee = 5 and bk = 4 in
  let kk = hh * ee in
  let mk = L.node_factory ~first_id:127900 ~dims:[||] () in
  let w = mk ~dims:[| jj; hh; ee |] "ck_w" and x = mk ~dims:[| ss; hh; ee |] "ck_x" in
  let d = mk ~dims:[| ss; jj |] "ck_d" and r = mk ~dims:[| ss; hh; ee |] "ck_r" in
  List.iter [ w; x; d; r ] ~f:L.materialize;
  let s = L.sym () and j = L.sym () and h = L.sym () and e = L.sym () in
  let s' = L.sym () and h' = L.sym () and e' = L.sym () in
  let at = [| L.iter s; L.iter j |] in
  let llc =
    L.seq
      (L.seq (L.zero d)
         (L.loop_n s ss
            (L.loop_n j jj
               (L.loop_n h hh
                  (L.loop_n e ee
                     (L.set d at
                        (L.add (L.get d at)
                           (L.mul
                              (L.get w [| L.iter j; L.iter h; L.iter e |])
                              (L.get x [| L.iter s; L.iter h; L.iter e |])))))))))
      (* The unrelated second reader: its own nest over [x], which nothing below schedules. *)
      (L.loop_n s' ss
         (L.loop_n h' hh
            (L.loop_n e' ee
               (L.set r
                  [| L.iter s'; L.iter h'; L.iter e' |]
                  (L.mul (L.c 2.) (L.get x [| L.iter s'; L.iter h'; L.iter e' |]))))))
  in
  let seed =
    [
      ( w,
        Array.init (jj * kk)
          ~f:(L.cycle_flat ~dims:[| jj; hh; ee |] ~modulus:11 ~offset:(-11.) ~stride:0.5) );
      ( x,
        Array.init (ss * kk)
          ~f:(L.cycle_flat ~dims:[| ss; hh; ee |] ~modulus:13 ~offset:1. ~stride:0.25) );
    ]
  in
  let o = L.optimize ~name:"ck_plain" llc in
  let co, f = Sched.coalesce ~outer:h ~inner:e in
  p "contraction: Coalesce of the contraction pair is Op_legal"
    (Sched.equal_op_verdict (Sched.op_legality o co) Sched.Op_legal);
  let oc = Sched.apply [ co ] o in
  p "contraction: the merged k loop spans the whole contraction"
    (Option.exists
       (L.find_loop ~f:(fun site -> Ir.Indexing.equal_symbol site.L.ls_index f) oc.LL.llc)
       ~f:(fun site -> site.L.ls_extent = kk));
  p "contraction: the accumulator's index is untouched by the coalescing"
    (L.count_stmt oc.LL.llc ~f:(function
       | LL.Set { tn; idcs; _ } ->
           Ir.Tnode.equal tn d && Array.equal Ir.Indexing.equal_axis_index idcs at
       | _ -> false)
    > 0);
  let sp, f_o, f_i = Sched.split ~axis:f ~factor:bk ~outer:LL.Serial ~inner:LL.Serial in
  let stage source =
    Sched.Stage
      {
        source;
        tile_loops = [ f_i ];
        shared = false;
        cooperative = None;
        hoisted = false;
        swizzle = None;
        pad_stride = None;
        pipeline_depth = 1;
        tile_prec = None;
      }
  in
  let sched =
    [
      co;
      Sched.Pad { axis = f; to_multiple_of = bk };
      sp;
      stage w;
      stage x;
      Sched.privatize ~accum_prec ~target:d ~over:f_o;
    ]
  in
  let os = Sched.apply sched o in
  (* Every read of the two operands, split by whether it reads the flattened pair: the load nests'
     reads do, the second reader's do not. *)
  let flat_reads = ref [] and plain_x_reads = ref [] in
  L.walk os.LL.llc ~on_scalar:(function
    | LL.Get (tn, idcs) when Ir.Tnode.equal tn w || Ir.Tnode.equal tn x ->
        if Array.exists idcs ~f:(Ir.Indexing.equal_axis_index Ir.Indexing.Sub_axis) then
          flat_reads := (tn, idcs) :: !flat_reads
        else if Ir.Tnode.equal tn x then plain_x_reads := idcs :: !plain_x_reads
    | _ -> ());
  p "contraction: both operands are staged through their flattened address"
    (List.exists !flat_reads ~f:(fun (tn, _) -> Ir.Tnode.equal tn w)
    && List.exists !flat_reads ~f:(fun (tn, _) -> Ir.Tnode.equal tn x));
  p_all "contraction: every staged read spans the merged extent on its flattened axis" !flat_reads
    ~f:(fun (tn, idcs) ->
      let dims = Lazy.force tn.Ir.Tnode.dims in
      (Ir.Affine.axis_extents ~dims idcs).(2) = kk);
  p "contraction: the unrelated second reader keeps reading the activation axis by axis"
    (List.exists !plain_x_reads ~f:(fun idcs ->
         Array.equal Ir.Indexing.equal_axis_index idcs [| L.iter s'; L.iter h'; L.iter e' |]));
  (* [bk = 4] divides neither [e = 5] nor [K = 15]: tiles cross heads and the last one runs past
     [K]. A fringe guarded against the head's [e] instead of [K] would zero most of every tile,
     which the bitwise claim below catches. *)
  let want_d, want_r, got_d, got_r =
    match
      ( L.execute ~name:"ck_plain_run" o ~seed ~read:[ d; r ],
        L.execute ~name:"ck_tiled_run" os ~seed ~read:[ d; r ] )
    with
    | [ wd; wr ], [ gd; gr ] -> (wd, wr, gd, gr)
    | _ -> assert false
  in
  p
    "contraction: the coalesced, padded, staged and privatized contraction computes the plain \
     one's values bitwise"
    ((not (Array.is_empty want_d))
    && Array.for_all want_d ~f:(fun v -> Float.(v <> 0.))
    && Array.equal Float.equal want_d got_d);
  p "contraction: the second reader's values are untouched"
    (Array.exists want_r ~f:(fun v -> Float.(v <> 0.)) && Array.equal Float.equal want_r got_r);
  (* Declines: the pair not adjacent in the weight, a per-head-only operand, a padded inner axis. *)
  let declines ~tag ~substring ~w_dims ~w_at ~extra =
    let w' = mk ~dims:w_dims ("ck_w_" ^ tag) and d' = mk ~dims:[| ss; jj |] ("ck_d_" ^ tag) in
    let x' =
      mk ~dims:[| ss; hh; (if String.equal tag "padded" then ee + 2 else ee) |] ("ck_x_" ^ tag)
    in
    let g = mk ~dims:[| hh |] ("ck_g_" ^ tag) in
    List.iter [ w'; d'; x'; g ] ~f:L.materialize;
    let s = L.sym () and j = L.sym () and h = L.sym () and e = L.sym () in
    let at = [| L.iter s; L.iter j |] in
    let prod = L.mul (L.get w' (w_at ~j ~h ~e)) (L.get x' [| L.iter s; L.iter h; L.iter e |]) in
    let prod = if extra then L.mul prod (L.get g [| L.iter h |]) else prod in
    let llc =
      L.loop_n s ss
        (L.loop_n j jj (L.loop_n h hh (L.loop_n e ee (L.set d' at (L.add (L.get d' at) prod)))))
    in
    let o = L.optimize ~name:("ck_" ^ tag) llc in
    let op = fst (Sched.coalesce ~outer:h ~inner:e) in
    p
      (Printf.sprintf "contraction %s: Coalesce declines" tag)
      (raises_with ~substring (fun () -> Sched.apply [ op ] o));
    p
      (Printf.sprintf "contraction %s: the oracle proves it illegal" tag)
      (match Sched.op_legality o op with Sched.Op_illegal _ -> true | _ -> false)
  in
  let plain_w ~j ~h ~e = [| L.iter j; L.iter h; L.iter e |] in
  declines ~tag:"hje" ~substring:"does not read" ~w_dims:[| hh; jj; ee |]
    ~w_at:(fun ~j ~h ~e -> [| L.iter h; L.iter j; L.iter e |])
    ~extra:false;
  declines ~tag:"per_head" ~substring:"does not read" ~w_dims:[| jj; hh; ee |] ~w_at:plain_w
    ~extra:true;
  declines ~tag:"padded" ~substring:"padded axis" ~w_dims:[| jj; hh; ee |] ~w_at:plain_w
    ~extra:false

(* [Split_reduce] on the merged k of a small-output contraction [d[j] += w[j,h,e] * x[h,e]]: the
   split's legality probe accepts the merged loop, and the coalesced, split, default-scheduled
   reduction computes the plain one's values bitwise on these exact inputs ([Split_reduce]
   reassociates, so bitwise parity is promised on exact fixtures only). *)
let () =
  let jj = 4 and hh = 4 and ee = 32 in
  let mk = L.node_factory ~first_id:127950 ~dims:[||] () in
  let w = mk ~dims:[| jj; hh; ee |] "cks_w" and x = mk ~dims:[| hh; ee |] "cks_x" in
  let d = mk ~dims:[| jj |] "cks_d" in
  List.iter [ w; x; d ] ~f:L.materialize;
  let j = L.sym () and h = L.sym () and e = L.sym () in
  let at = [| L.iter j |] in
  let llc =
    L.seq (L.zero d)
      (L.loop_n j jj
         (L.loop_n h hh
            (L.loop_n e ee
               (L.set d at
                  (L.add (L.get d at)
                     (L.mul
                        (L.get w [| L.iter j; L.iter h; L.iter e |])
                        (L.get x [| L.iter h; L.iter e |])))))))
  in
  let seed =
    [
      ( w,
        Array.init
          (jj * hh * ee)
          ~f:(L.cycle_flat ~dims:[| jj; hh; ee |] ~modulus:11 ~offset:(-11.) ~stride:0.5) );
      ( x,
        Array.init (hh * ee)
          ~f:(L.cycle_flat ~dims:[| hh; ee |] ~modulus:13 ~offset:1. ~stride:0.25) );
    ]
  in
  let o = L.optimize ~name:"cks_plain" llc in
  let co, f = Sched.coalesce ~outer:h ~inner:e in
  let oc = Sched.apply [ co ] o in
  let sr, _, _, _ = Sched.split_reduce ~axis:f ~target:d ~num_blocks:4 in
  p "contraction split-reduce: the split of the merged k is Op_legal"
    (Sched.equal_op_verdict (Sched.op_legality oc sr) Sched.Op_legal);
  let os = Sched.apply [ co; sr ] o in
  let run name o transform =
    let ctx, routine =
      Context.compile ~name ~prelowered:o ~lowered_transform:transform (Context.auto ())
        Ir.Assignments.empty_comp Ir.Indexing.Empty
    in
    let ctx = List.fold seed ~init:ctx ~f:(fun ctx (tn, vs) -> Context.set_values ctx tn vs) in
    Context.get_values (Context.run ctx routine) d
  in
  let want = run "cks_plain_run" o (fun o -> [ o ]) in
  let got =
    run "cks_split_run" os
      (Sched.maybe_default_schedules ~backend_name
         ~limits:(Context.hardware_limits (Context.auto ()))
         ~static_indices:[])
  in
  p
    "contraction split-reduce: the coalesced, split, default-scheduled contraction computes the \
     plain one's values bitwise"
    ((not (Array.is_empty want))
    && Array.for_all want ~f:(fun v -> Float.(v <> 0.))
    && Array.equal Float.equal want got)

(* {1 Composition: a coalesced pair under [Fuse_epilogue]}

   The one caller of the shared per-axis decomposition ([Schedule.terms_of_index]) that refuses a
   flattened component rather than carrying it (gh-ocannl-1293): relocating the elementwise tail
   [out[h', e'] = d[h', e'] + 1] into the store-back of the coalesced [d[Sub_axis; f]] substitutes
   the site's per-axis index for each tail symbol, and a flattened site has none. Read as the empty
   sum at offset 0 instead, the [Sub_axis] was a per-axis coordinate to the exact-radix cover check,
   which then refused the site as an incomplete cover -- the right outcome for the wrong reason, and
   nothing in the decomposition's type said a sub-axis needs an answer of its own. The claims name
   the reason, at both sites the relocation knows: the plain accumulation nest and the [Privatize]
   tile store-back. The uncoalesced nest is the control: the same tail fuses there. *)

let () =
  let hh = 3 and ee = 4 and kk = 5 in
  let mk = L.node_factory ~first_id:129300 ~dims:[||] () in
  let x = mk ~dims:[| kk; hh; ee |] "cf_x" and d = mk ~dims:[| hh; ee |] "cf_d" in
  let out = mk ~dims:[| hh; ee |] "cf_out" in
  List.iter [ x; d; out ] ~f:L.materialize;
  let h = L.sym () and e = L.sym () and k = L.sym () in
  let at = [| L.iter h; L.iter e |] in
  let h' = L.sym () and e' = L.sym () in
  let at' = [| L.iter h'; L.iter e' |] in
  let llc =
    L.seq
      (L.seq (L.zero d)
         (L.loop_n h hh
            (L.loop_n e ee
               (L.loop_n k kk
                  (L.set d at (L.add (L.get d at) (L.get x [| L.iter k; L.iter h; L.iter e |])))))))
      (L.loop_n h' hh (L.loop_n e' ee (L.set out at' (L.add (L.get d at') (L.c 1.)))))
  in
  let o = L.optimize ~name:"cf_plain" llc in
  let fuse = Sched.Fuse_epilogue { target = d; shared = false } in
  let flattened = "flattened (Sub_axis)" in
  p "fuse-epilogue: the uncoalesced reduction's tail fuses"
    (Option.is_none (Sched.fuse_epilogue_witness ~target:d o)
    && Result.is_ok (Result.try_with (fun () -> Sched.apply [ fuse ] o)));
  let co, _ = Sched.coalesce ~outer:h ~inner:e in
  let oc = Sched.apply [ co ] o in
  (* The refusal's own words, echoed to stderr so a failing claim shows what the relocation said. *)
  let refusal what f =
    let why = match f () with _ -> None | exception Invalid_argument msg -> Some msg in
    Stdio.eprintf "fuse-epilogue %s: %s (not part of the golden)\n" what
      (Option.value why ~default:"<no refusal>");
    why
  in
  let names_flattened = function
    | Some why -> String.is_substring why ~substring:flattened
    | None -> false
  in
  p "fuse-epilogue: the relocation declines the coalesced accumulation nest's flattened site"
    (names_flattened (refusal "nest" (fun () -> Sched.apply [ fuse ] oc)));
  p "fuse-epilogue: the witness names the flattened site as the reason"
    (names_flattened (Sched.fuse_epilogue_witness ~target:d oc));
  p
    "fuse-epilogue: the relocation declines the coalesced Privatize tile store-back's flattened \
     site"
    (names_flattened
       (refusal "privatized" (fun () ->
            Sched.apply [ co; Sched.privatize ~accum_prec ~target:d ~over:k; fuse ] o)))

(* {1 The real projection site} *)

let named name (comp : Asgns.comp) : Asgns.comp =
  { comp with asgns = Asgns.Block_comment (name, comp.asgns) }

let capture fwd =
  let captured = ref None in
  let _ctx, _r =
    Context.compile
      ~lowered_transform:(fun opt ->
        captured := Some opt;
        [ opt ])
      (Context.auto ()) fwd Ir.Indexing.Empty
  in
  Option.value_exn ~here:[%here] !captured

let run_with fwd tensor transform =
  let ctx, routine =
    Context.compile
      ~lowered_transform:(fun o -> [ transform o ])
      (Context.auto ()) fwd Ir.Indexing.Empty
  in
  Context.get_values (Context.run ctx routine) tensor.Tensor.value

let bb = 2
and ss = 64
and hh = 4
and ee = 32
and kk = 16

let x () =
  NTDSL.init ~l:"co_x" ~prec:Ir.Ops.single ~o:[ bb; ss; kk ]
    ~f:(Ll_test.cycle ~dims:[| bb; ss; kk |] ~modulus:13 ~offset:0. ~stride:0.25)
    ()

let w () =
  NTDSL.init ~l:"co_w" ~prec:Ir.Ops.single ~o:[ hh; ee; kk ]
    ~f:(Ll_test.cycle ~dims:[| hh; ee; kk |] ~modulus:11 ~offset:(-5.) ~stride:0.5)
    ()

(* The gpt2 q/k/v projection's own layout: [d[b,s,h,e] += x[b,s,k] * w[h,e,k]]. *)
let projection () =
  let xv = x () and wv = w () in
  let%op out = xv +* "bsk;hjk=>bshj" wv in
  out

let blocktile p = p.Autotune.sk_gpu && not p.Autotune.sk_mma

let () =
  let reference = projection () in
  let ref_fwd = named "co_ref" (Train.forward reference) in
  let want = run_with ref_fwd reference Fn.id in
  p "projection: the serial reference is not all zeros"
    (Array.exists want ~f:(fun v -> Float.(v <> 0.)));
  let cand = projection () in
  let fwd = named "co_sched" (Train.forward cand) in
  let opt = capture fwd in
  let prefix = Autotune.coalesce_prefix ~layout:Coalesce_columns opt in
  p "projection: the coalesced layout's prefix applies" (Option.is_some prefix);
  let prefix = Option.value ~default:[] prefix in
  p "projection: the prefix expands the zeroing, coalesces its pair, then the site's"
    (match prefix with
    | [ Sched.Expand_zero _; Sched.Coalesce _; Sched.Coalesce _ ] -> true
    | _ -> false);
  p_all "projection: every Coalesce of the prefix is Op_legal" (Sched.schedule_legality opt prefix)
    ~f:(fun (op, v) ->
      match op with Sched.Coalesce _ -> Sched.equal_op_verdict v Sched.Op_legal | _ -> true);
  let prepared = Sched.apply prefix opt in
  let site = Autotune.detect_matmul prepared.LL.llc in
  p "projection: the coalesced site's column role spans every head's columns"
    (Option.exists site ~f:(fun site -> site.Autotune.m_nj = hh * ee));
  Option.iter site ~f:(fun site ->
      p_empty "projection: no interior batch loop is left among the coalesced site's batch loops"
        ~over:(site.Autotune.m_bo @ site.Autotune.m_bi)
        site.Autotune.m_bi);
  p "projection: the prefix alone computes the serial reference bitwise"
    (Array.equal Float.equal want (run_with fwd cand (fun o -> Sched.apply prefix o)));
  (* The branch's seeds: synthetic no-limits keep the enumeration machine-independent. *)
  let limits = Ir.Backend_intf.no_hardware_limits in
  let seeds = Autotune.coalesced_seed_params ~is_gpu:true ~is_cpu:false ~limits opt in
  p "projection: the coalesced branch offers seeds" (not (List.is_empty seeds));
  p_all "projection: every coalesced seed is a scalar blocktile seed stamped coalesced" seeds
    ~f:(fun q ->
      Poly.equal q.Autotune.sk_coalesce (Some Autotune.Coalesce_columns)
      && blocktile q && not q.Autotune.sk_epilogue);
  p_exists "projection: some coalesced seed's column tile is wider than one head" seeds ~f:(fun q ->
      q.Autotune.sk_bn > ee);
  p_empty "projection: the coalesced branch offers nothing off GPU" ~over:seeds
    (Autotune.coalesced_seed_params ~is_gpu:false ~is_cpu:true ~limits opt);
  let canon = SC.canonicalize opt in
  let schedules = List.map seeds ~f:(fun q -> (q, Autotune.sketch_schedule ~accum_prec ~p:q opt)) in
  p_all
    "projection: every coalesced schedule constructs, validates and launches the merged column \
     blocks on .x"
    schedules ~f:(fun (q, sched) ->
      match Sched.apply sched opt with
      | o -> (
          match LL.validate_parallel o.LL.optimize_ctx.LL.placements o.LL.llc with
          | () -> (LL.launch_dims o.LL.llc).LL.grid.(0) = hh * ee / q.Autotune.sk_bn
          | exception exn ->
              Stdio.eprintf "validate_parallel FAILED: %s\n" (Exn.to_string exn);
              false)
      | exception exn ->
          Stdio.eprintf "schedule FAILED: %s\n" (Exn.to_string exn);
          false);
  (* The persisted form names the merged loop structurally: replaying the saved schedule against the
     canonical form rebuilds code with the same digest. *)
  p_all "projection: every coalesced schedule survives the cache's structural round trip" schedules
    ~f:(fun (_, sched) ->
      let saved, _ = SC.to_saved (SC.base_registry canon) sched in
      let replayed, _ = SC.of_saved canon saved in
      List.exists saved ~f:(function SC.Coalesce _ -> true | _ -> false)
      && String.equal
           (SC.digest (SC.canonicalize (Sched.apply sched opt)))
           (SC.digest (SC.canonicalize (Sched.apply replayed opt))));
  Stdio.eprintf "schedule_coalesce: backend %s, %d coalesced seed(s): %s (not part of the golden)\n"
    backend_name (List.length seeds)
    (String.concat ~sep:", "
       (List.map seeds ~f:(fun q ->
            Printf.sprintf "%dx%dx%d/%dx%d%s" q.Autotune.sk_bm q.Autotune.sk_bn q.Autotune.sk_bk
              q.Autotune.sk_tm q.Autotune.sk_tn
              (if q.Autotune.sk_batch_grid then " bgrid" else ""))));
  gated_all ~when_:on_gpu ~on:backend_name
    "projection: every coalesced seed executes to the serial reference bitwise" schedules
    ~f:(fun (q, _) ->
      match
        run_with fwd cand (fun o -> Sched.apply (Autotune.sketch_schedule ~accum_prec ~p:q o) o)
      with
      | got -> Array.equal Float.equal got want
      | exception exn ->
          Stdio.eprintf "coalesced seed FAILED: %s\n" (Exn.to_string exn);
          false)

(* {1 The v1 boundary: a companion over the uncoalesced axes} *)

let () =
  let cand =
    let xv = x () and wv = w () in
    let%op z = xv +* "bsk;hjk=>bshj" wv in
    Train.set_materialized z.Tensor.value;
    let%op y = relu z in
    y
  in
  let opt = capture (named "co_companion" (Train.forward cand)) in
  let limits = Ir.Backend_intf.no_hardware_limits in
  p "companion: the site's own pair still coalesces"
    (Option.is_some (Autotune.coalesce_prefix ~layout:Coalesce_columns opt));
  let family = Autotune.sketch_seed_params ~is_gpu:true ~is_cpu:false ~limits opt in
  p_exists "companion: the ordinary family still covers the tail" family ~f:blocktile;
  p_empty
    "companion: the elementwise tail cannot share the merged chain, so the branch offers nothing"
    ~over:family
    (Autotune.coalesced_seed_params ~is_gpu:true ~is_cpu:false ~limits opt)

(* {1 A fission segment's site}

   In a routine that fissions, the projection's [Zero_out] lands in a [`Zeros] segment of its own
   and the site's segment is unzeroed: the per-segment seeds the real step searches see the site's
   own [Coalesce] alone. A row reduction of the materialized projection is the cross-nest edge that
   cuts the routine. *)

let () =
  let cand =
    let xv = x () and wv = w () in
    let%op z = xv +* "bsk;hjk=>bshj" wv in
    Train.set_materialized z.Tensor.value;
    let%op y = z ++ "bshj=>bs" in
    y
  in
  let opt = capture (named "co_fission" (Train.forward cand)) in
  let limits = Ir.Backend_intf.no_hardware_limits in
  (* The search's own segment enumeration (GPU presets: unannotated neighbours coalesce back into
     one kernel, so an empty preset would never cut). *)
  let segments =
    Sched.fission_scheduled ~promote_locals:true
      ~preset:(Sched.default_gpu ~min_parallel:1 ~limits)
      ~zero_sched:(Sched.zero_expansion ~limits) ~static_indices:[]
      {
        opt with
        LL.traced_store = Hashtbl.copy opt.LL.traced_store;
        LL.optimize_ctx = LL.copy_optimize_ctx opt.LL.optimize_ctx;
      }
  in
  p "fission: the routine fissions" (List.length segments > 1);
  p_exists
    "fission: the site's unzeroed segment coalesces by its own Coalesce alone, and seeds the branch"
    segments ~f:(fun (kind, pre, _, _) ->
      match (kind, Autotune.coalesce_prefix ~layout:Coalesce_columns pre) with
      | `Normal, Some [ Sched.Coalesce _ ] ->
          not
            (List.is_empty (Autotune.coalesced_seed_params ~is_gpu:true ~is_cpu:false ~limits pre))
      | _ -> false)

(* {1 The folded zero of a GPU fission segment}

   The search's segmentation folds a reduction's covering zero, expanded per cell, into the
   reduction's own segment (gh-ocannl-1175): the q/k/v segment the real step searches carries the
   zero nest beside the site. The prefix coalesces that nest's pair too, so it takes the merged
   chain's geometry as a companion; left uncoalesced it would refute the branch on companion
   coverage. *)

let () =
  let cand =
    let xv = x () and wv = w () in
    let%op z = xv +* "bsk;hjk=>bshj" wv in
    Train.set_materialized z.Tensor.value;
    z
  in
  let opt = capture (named "co_folded" (Train.forward cand)) in
  let n = bb * ss * hh * ee in
  let seed = [ (cand.Tensor.value, Array.create ~len:n (-999.)) ] in
  let want =
    List.hd_exn (L.execute ~name:"co_folded_materialized" opt ~seed ~read:[ cand.Tensor.value ])
  in
  let limits = Ir.Backend_intf.no_hardware_limits in
  let segments =
    Sched.fission_scheduled ~fold_zeros:true
      ~keep_mapping:(Sched.mapping_probe (Sched.default_gpu ~limits))
      ~preset:(Sched.default_gpu ~limits) ~zero_sched:(Sched.zero_expansion ~limits)
      ~static_indices:[]
      {
        opt with
        LL.traced_store = Hashtbl.copy opt.LL.traced_store;
        LL.optimize_ctx = LL.copy_optimize_ctx opt.LL.optimize_ctx;
      }
  in
  let folded =
    List.find_map segments ~f:(fun (kind, pre, _, _) ->
        match (kind, Autotune.coalesce_prefix ~layout:Coalesce_columns pre) with
        | `Normal, Some [ Sched.Coalesce _; Sched.Coalesce _ ] -> Some pre
        | _ -> None)
  in
  p "folded: the segment's zero nest and site both coalesce" (Option.is_some folded);
  Option.iter folded ~f:(fun pre ->
      let seeds = Autotune.coalesced_seed_params ~is_gpu:true ~is_cpu:false ~limits pre in
      let apply sched =
        Sched.apply sched
          {
            pre with
            LL.traced_store = Hashtbl.copy pre.LL.traced_store;
            LL.optimize_ctx = LL.copy_optimize_ctx pre.LL.optimize_ctx;
          }
      in
      p_exists "folded: the branch seeds a column tile wider than one head" seeds ~f:(fun q ->
          q.Autotune.sk_bn > ee);
      p_all "folded: every coalesced schedule constructs and validates" seeds ~f:(fun q ->
          match apply (Autotune.sketch_schedule ~accum_prec ~p:q pre) with
          | o -> (
              match LL.validate_parallel o.LL.optimize_ctx.LL.placements o.LL.llc with
              | () -> true
              | exception exn ->
                  Stdio.eprintf "folded validate_parallel FAILED: %s\n" (Exn.to_string exn);
                  false)
          | exception exn ->
              Stdio.eprintf "folded schedule FAILED: %s\n" (Exn.to_string exn);
              false);
      gated_alli ~when_:on_gpu ~on:backend_name
        "folded: every coalesced seed executes to the materialized run" seeds ~f:(fun i q ->
          let o = apply (Autotune.sketch_schedule ~accum_prec ~p:q pre) in
          let got =
            List.hd_exn
              (L.execute
                 ~name:("co_folded_seed_" ^ Int.to_string i)
                 o ~seed ~read:[ cand.Tensor.value ])
          in
          Array.equal Float.equal got want))

(* {1 The contraction layout on the real out projection (gh-ocannl-1279)}

   [{ w_o } * attn] with the weight's two input axes [(h, e)] lowers to [d[b,s,j] += w[j,h,e] *
   x[b,s,h,e]]: a contraction nest, [m_ko = [h]] and [m_k = e]. The contraction layout's prefix is
   [Coalesce (h, e)] alone (the output coordinates, and so the zeroing, are untouched); the prepared
   code re-detects as one k of extent [h * e] with every output role unchanged, and the branch
   enumerates the GPU scalar blocktile family over it. Each seed must construct, validate, keep both
   staged loads on the flattened source address bounded by the merged extent, survive the cache's
   structural round trip and, on GPU backends, execute to the serial reference bitwise.

   Inputs: the weight strictly negative and the activation strictly positive, every product a small
   multiple of 1/8 and every partial sum far below 2^24, so no cell is zero and every accumulation
   order is exact in f32. *)

let out_proj ~tag ~bb ~ss ~jj ~hh ~ee () =
  let w =
    NTDSL.init ~l:(tag ^ "_w") ~prec:Ir.Ops.single ~o:[ jj ] ~i:[ hh; ee ]
      ~f:(Ll_test.cycle ~dims:[| jj; hh; ee |] ~modulus:11 ~offset:(-11.) ~stride:0.5)
      ()
  in
  let att =
    NTDSL.init ~l:(tag ^ "_att") ~prec:Ir.Ops.single ~b:[ bb; ss ] ~o:[ hh; ee ]
      ~f:(Ll_test.cycle ~dims:[| bb; ss; hh; ee |] ~modulus:13 ~offset:1. ~stride:0.25)
      ()
  in
  let%op out = w * att in
  (out, w, att)

let contraction_seeds ?(limits = Ir.Backend_intf.no_hardware_limits) opt =
  Autotune.coalesced_seed_params ~is_gpu:true ~is_cpu:false ~limits opt
  |> List.filter ~f:(fun q ->
      Poly.equal q.Autotune.sk_coalesce (Some Autotune.Coalesce_contraction))

(* One out-projection leg; returns the captured lowering, the contraction seeds under no limits, and
   those under the device's limits (the executed ones on GPU), for the caller's K-menu claims. *)
let contraction_leg ~tag ~bb ~ss ~jj ~hh ~ee =
  let kk = hh * ee in
  let reference, _, _ = out_proj ~tag:(tag ^ "_ref") ~bb ~ss ~jj ~hh ~ee () in
  let want = run_with (named (tag ^ "_ref") (Train.forward reference)) reference Fn.id in
  p
    (tag ^ ": no cell of the serial reference is zero")
    ((not (Array.is_empty want)) && Array.for_all want ~f:(fun v -> Float.(v <> 0.)));
  let cand, w, att = out_proj ~tag ~bb ~ss ~jj ~hh ~ee () in
  let fwd = named (tag ^ "_sched") (Train.forward cand) in
  let opt = capture fwd in
  p
    (tag ^ ": the columns layout has no interior batch loop to merge here")
    (Option.is_none (Autotune.coalesce_prefix ~layout:Coalesce_columns opt));
  let prefix = Autotune.coalesce_prefix ~layout:Coalesce_contraction opt in
  p
    (tag ^ ": the contraction prefix is the one Coalesce of the contraction pair")
    (match prefix with Some [ Sched.Coalesce _ ] -> true | _ -> false);
  let prefix = Option.value ~default:[] prefix in
  p_all (tag ^ ": the contraction prefix is Op_legal") (Sched.schedule_legality opt prefix)
    ~f:(fun (_, v) -> Sched.equal_op_verdict v Sched.Op_legal);
  let before = Autotune.detect_matmul opt.LL.llc in
  let prepared = Sched.apply prefix opt in
  let site = Autotune.detect_matmul prepared.LL.llc in
  p
    (tag ^ ": the coalesced site contracts over one k of the pair's product, every output role kept")
    (match (before, site) with
    | Some b, Some s ->
        s.Autotune.m_nk = kk
        (* The pair left [m_ko]: what remains is the original outer contraction loops minus the head
           loop the merge consumed. *)
        && (match List.rev b.Autotune.m_ko with
          | (_, nh) :: rest ->
              nh = hh
              && List.equal
                   (fun (x, n) (y, n') -> Ir.Indexing.equal_symbol x y && n = n')
                   (List.rev rest) s.Autotune.m_ko
          | [] -> false)
        && b.Autotune.m_ni = s.Autotune.m_ni && b.Autotune.m_nj = s.Autotune.m_nj
        && s.Autotune.m_nj = jj
        && List.length b.Autotune.m_bo = List.length s.Autotune.m_bo
    | _ -> false);
  p
    (tag ^ ": the prefix alone computes the serial reference bitwise")
    (Array.equal Float.equal want (run_with fwd cand (fun o -> Sched.apply prefix o)));
  let seeds = contraction_seeds opt in
  p (tag ^ ": the contraction branch offers seeds") (not (List.is_empty seeds));
  p_all (tag ^ ": every contraction seed is an unfused scalar blocktile seed") seeds ~f:(fun q ->
      blocktile q && not q.Autotune.sk_epilogue);
  let canon = SC.canonicalize opt in
  let schedules = List.map seeds ~f:(fun q -> (q, Autotune.sketch_schedule ~accum_prec ~p:q opt)) in
  p_all (tag ^ ": every contraction schedule constructs and validates") schedules
    ~f:(fun (_, sched) ->
      match Sched.apply sched opt with
      | o -> (
          match LL.validate_parallel o.LL.optimize_ctx.LL.placements o.LL.llc with
          | () -> true
          | exception exn ->
              Stdio.eprintf "%s validate_parallel FAILED: %s\n" tag (Exn.to_string exn);
              false)
      | exception exn ->
          Stdio.eprintf "%s schedule FAILED: %s\n" tag (Exn.to_string exn);
          false);
  (* After the pipeline every read of either operand is a staged load's: each reads the source
     through the flattened address, bounded by the merged extent (the edge guards' source). *)
  p_all
    (tag ^ ": every contraction schedule loads both operands through the flattened address over K")
    schedules ~f:(fun (_, sched) ->
      match Sched.apply sched opt with
      | o ->
          let reads = ref [] in
          L.walk o.LL.llc ~on_scalar:(function
            | LL.Get (tn, idcs)
              when Ir.Tnode.equal tn w.Tensor.value || Ir.Tnode.equal tn att.Tensor.value ->
                reads := (tn, idcs) :: !reads
            | _ -> ());
          let flat_ok tn =
            let mine = List.filter !reads ~f:(fun (t, _) -> Ir.Tnode.equal t tn) in
            (not (List.is_empty mine))
            && List.for_all mine ~f:(fun (t, idcs) ->
                let dims = Lazy.force t.Ir.Tnode.dims in
                let n = Array.length idcs in
                n = Array.length dims
                && Ir.Indexing.equal_axis_index idcs.(n - 2) Ir.Indexing.Sub_axis
                && (Ir.Affine.axis_extents ~dims idcs).(n - 1) = kk)
          in
          flat_ok w.Tensor.value && flat_ok att.Tensor.value
      | exception _ -> false);
  p_all (tag ^ ": every contraction schedule survives the cache's structural round trip") schedules
    ~f:(fun (_, sched) ->
      let saved, _ = SC.to_saved (SC.base_registry canon) sched in
      let replayed, _ = SC.of_saved canon saved in
      List.exists saved ~f:(function SC.Coalesce _ -> true | _ -> false)
      && String.equal
           (SC.digest (SC.canonicalize (Sched.apply sched opt)))
           (SC.digest (SC.canonicalize (Sched.apply replayed opt))));
  Stdio.eprintf "%s: backend %s, %d contraction seed(s): %s (not part of the golden)\n" tag
    backend_name (List.length seeds)
    (String.concat ~sep:", " (List.map seeds ~f:Autotune.sketch_params_label));
  (* Executed under the device's own limits: a seed the synthetic no-limits enumeration admits but
     this device's workgroup memory does not (32x32x256 in f32) is refuted there, not run. *)
  let device_seeds =
    if on_gpu then contraction_seeds ~limits:(Context.hardware_limits (Context.auto ())) opt
    else seeds
  in
  Stdio.eprintf
    "%s: %d contraction seed(s) under this device's limits: %s (not part of the golden)\n" tag
    (List.length device_seeds)
    (String.concat ~sep:", " (List.map device_seeds ~f:Autotune.sketch_params_label));
  gated_all ~when_:on_gpu ~on:backend_name
    (tag ^ ": every contraction seed this device admits executes to the serial reference bitwise")
    device_seeds ~f:(fun q ->
      match
        run_with fwd cand (fun o -> Sched.apply (Autotune.sketch_schedule ~accum_prec ~p:q o) o)
      with
      | got -> Array.equal Float.equal got want
      | exception exn ->
          Stdio.eprintf "%s contraction seed FAILED: %s\n" tag (Exn.to_string exn);
          false);
  (opt, seeds, device_seeds)

(* The K menu (gh-ocannl-1279). gpt2_mini's heads, h = 8 and e = 32: K = 256, so the deepest depth
   tiles the whole merged contraction in one k-block, and every menu depth divides K. Under Metal's
   32 KiB threadgroup limit the 16x16x256 tile (32 KiB of f32 operand tiles) is admitted exactly at
   the limit and the 32x32x256 one (64 KiB) is refuted by the workgroup-memory floor. *)
let () =
  let kk = 8 * 32 in
  let opt, seeds, device_seeds = contraction_leg ~tag:"ck_proj" ~bb:2 ~ss:64 ~jj:64 ~hh:8 ~ee:32 in
  let geom (bm, bn, bk) q =
    q.Autotune.sk_bm = bm && q.Autotune.sk_bn = bn && q.Autotune.sk_bk = bk
  in
  p_exists "ck_proj: a contraction seed tiles the whole merged k (bk = K = 256)" seeds ~f:(fun q ->
      q.Autotune.sk_bk = kk);
  p_all "ck_proj: no contraction seed's k-tile is shallower than one head" seeds ~f:(fun q ->
      q.Autotune.sk_bk >= 32);
  p_exists "ck_proj: the menu reaches below 256 too" seeds ~f:(fun q -> q.Autotune.sk_bk < kk);
  let metal_like =
    { Ir.Backend_intf.no_hardware_limits with max_workgroup_memory_bytes = Some 32768 }
  in
  let limited = contraction_seeds ~limits:metal_like opt in
  p_exists "ck_proj: 32 KiB of workgroup memory admits the 16x16x256 tile, exactly at the limit"
    limited
    ~f:(geom (16, 16, 256));
  p_none "ck_proj: 32 KiB of workgroup memory refutes the 32x32x256 tile" limited
    ~f:(geom (32, 32, 256));
  p_exists "ck_proj: no-limits enumeration offers the 32x32x256 tile the floor refutes" seeds
    ~f:(geom (32, 32, 256));
  (* Every GPU this runs on admits at least 32 KiB (Metal exactly that), so the headline tile is
     among the seeds executed above. *)
  gated_exists ~when_:on_gpu ~on:backend_name
    "ck_proj: this device admits, and so executed, the 16x16x256 tile" device_seeds
    ~f:(geom (16, 16, 256));
  (* gh-ocannl-1459: the 64x64 block of 4x4 register tiles (256 threads), the shape CUDA's
     uncoalesced winner uses. Its two f32 operand tiles need 512 bytes per unit of depth: 32 KiB at
     depth 64, so CUDA's 48 KiB admits depths 64 and 32 and refutes 128, and Metal's 32 KiB admits
     64 exactly at the limit. The row role is [s] (the batch [b] is a batch loop), so [s = 64] makes
     the 64-row block divide it: on a 32-row site it pads 2x and the pad-waste judgment excludes
     it. *)
  let is_64x64 q = q.Autotune.sk_bm = 64 && q.Autotune.sk_bn = 64 in
  p_exists "ck_proj: the menu offers a 64x64 block of 4x4 register tiles" seeds ~f:(fun q ->
      is_64x64 q && q.Autotune.sk_tm = 4 && q.Autotune.sk_tn = 4);
  let cuda_like =
    { Ir.Backend_intf.no_hardware_limits with max_workgroup_memory_bytes = Some 49152 }
  in
  let cuda_seeds = contraction_seeds ~limits:cuda_like opt in
  p_exists "ck_proj: 48 KiB of workgroup memory admits the 64x64x64 tile" cuda_seeds
    ~f:(geom (64, 64, 64));
  p_none "ck_proj: 48 KiB of workgroup memory refutes the 64x64x128 tile" cuda_seeds
    ~f:(geom (64, 64, 128));
  p_exists "ck_proj: no-limits enumeration offers the 64x64x128 tile the 48 KiB floor refutes" seeds
    ~f:(geom (64, 64, 128));
  p_exists "ck_proj: 32 KiB of workgroup memory admits the 64x64x64 tile, exactly at the limit"
    limited
    ~f:(geom (64, 64, 64));
  let thread_cap n =
    contraction_seeds
      ~limits:{ Ir.Backend_intf.no_hardware_limits with max_threads_per_workgroup = Some n }
      opt
  in
  p_exists "ck_proj: a 256-thread cap admits the 64x64 block" (thread_cap 256) ~f:is_64x64;
  p_none "ck_proj: a 255-thread cap refutes the 64x64 block" (thread_cap 255) ~f:is_64x64;
  gated_exists ~when_:on_gpu ~on:backend_name
    "ck_proj: this device admits, and so executed, a 64x64 tile" device_seeds ~f:is_64x64

(* An odd K: h = 3 heads of e = 20, K = 60. A depth whose half already covers K would only pad, so
   the menu offers 64 and 32: a 32-deep tile crosses a head boundary in every block and pads the
   last one, a 64-deep one is a single block padding K by four. *)
let () =
  let _opt, seeds, _ = contraction_leg ~tag:"ck_odd" ~bb:2 ~ss:16 ~jj:32 ~hh:3 ~ee:20 in
  p_all "ck_odd: no contraction seed's k-tile is deeper than one padded block of K" seeds
    ~f:(fun q -> q.Autotune.sk_bk <= 64);
  p_exists "ck_odd: a contraction seed's k-tile crosses heads (32 does not divide e = 20)" seeds
    ~f:(fun q -> q.Autotune.sk_bk = 32);
  p_exists "ck_odd: a contraction seed pads the merged k (64 does not divide K = 60)" seeds
    ~f:(fun q -> q.Autotune.sk_bk = 64)

(* A weight storing the pair apart ([w[h, j, e]]): the contraction prefix declines, so the branch
   offers no contraction seed, while the ordinary family still seeds the site. *)
let () =
  let bb = 2 and ss = 16 and jj = 32 and hh = 4 and ee = 16 in
  let w =
    NTDSL.init ~l:"ck_hje_w" ~prec:Ir.Ops.single ~o:[ hh; jj; ee ]
      ~f:(Ll_test.cycle ~dims:[| hh; jj; ee |] ~modulus:11 ~offset:(-11.) ~stride:0.5)
      ()
  in
  let att =
    NTDSL.init ~l:"ck_hje_att" ~prec:Ir.Ops.single ~o:[ bb; ss; hh; ee ]
      ~f:(Ll_test.cycle ~dims:[| bb; ss; hh; ee |] ~modulus:13 ~offset:1. ~stride:0.25)
      ()
  in
  let%op out = att +* "bshe;hje=>bsj" w in
  let opt = capture (named "ck_hje" (Train.forward out)) in
  p "ck_hje: the site is a multi-axis contraction"
    (Option.exists (Autotune.detect_matmul opt.LL.llc) ~f:(fun s ->
         not (List.is_empty s.Autotune.m_ko)));
  p "ck_hje: the contraction prefix declines on the weight storing the pair apart"
    (Option.is_none (Autotune.coalesce_prefix ~layout:Coalesce_contraction opt));
  let family =
    Autotune.sketch_seed_params ~is_gpu:true ~is_cpu:false
      ~limits:Ir.Backend_intf.no_hardware_limits opt
  in
  p_exists "ck_hje: the ordinary family still seeds the site" family ~f:blocktile;
  p_empty "ck_hje: the branch offers no contraction seed" ~over:family (contraction_seeds opt)
