(* Kernel fission at cross-workgroup edges (docs/proposals/schedule-ir-optops.md §7): routines whose
   top-level statements form materialized producer/consumer pairs split into segment kernels
   launched back-to-back on the routine's stream, each with its own default schedule.

   Covered here:

   - Executed: a two-nest chain with a forced-materialized intermediate — before fission the whole
   routine ran 1×1; the aligned cross-nest rule now keeps both nests in a {e single} parallel kernel
   (values checked, source checked for the absence of [__seg] splits and for per-backend parallel
   constructs). - Executed: a backward pass ([Train.grad_update]) — the [Zero_out] of the gradient
   and the bare/reduction statements segment away from the accumulation nest (gradient values
   checked). - Structural, backend-independent (analysis run for [backend_name:"metal"] on captured
   lowered code): segment count, hardware annotation, traced stores, identity on backends without
   automatic scheduling. - Structural, hand-built [Low_level.t]: replication of hoisted scope-locals
   into consuming segments (option (b) v2), the merge-back fallback when a write between the
   definition and the consumer invalidates replication, and promotion of [Local] scratch stranded
   across a cut. These need genuinely racy (misaligned) cross-nest edges to force cuts, hence the
   [Fixed_idx] reads below. - Structural, hand-built: the aligned cross-nest rule itself — aligned
   pairs merge into one annotated kernel (including aligned [Local] scratch, which then needs no
   promotion), while extent mismatches, transposed accesses, and alignment trims that would lose a
   nest's parallelism all still cut. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module LL = Ir.Low_level
module Sched = Ir.Schedule
module Asgns = Ir.Assignments
module Tn = Ir.Tnode
module Idx = Ir.Indexing
module IDX = Train.IDX

let () = Utils.settings.output_debug_files_in_build_directory <- true

open Verdict.Claims

let approx a b = Float.(abs (a - b) < 1e-4)
let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")
let on_cpu = Sched.backend_is_cpu backend_name

module Generated = Test_utils.Generated

let () = Generated.init ~backend_name

let has_hardware_regs src =
  String.is_substring src ~substring:"gid." || String.is_substring src ~substring:"blockIdx."

let has_parallel_construct src =
  String.is_substring src ~substring:"dispatch_apply"
  || String.is_substring src ~substring:"#pragma omp parallel for"

let count_substr src pattern = List.length (String.substr_index_all src ~may_overlap:false ~pattern)

let named name (comp : Asgns.comp) : Asgns.comp =
  { comp with asgns = Asgns.Block_comment (name, comp.asgns) }

let grad_of t = (Option.value_exn t.Tensor.diff).Tensor.grad

let rec has_declare_local (llc : LL.t) =
  match llc with
  | LL.Declare_local _ -> true
  | LL.Seq (a, b) -> has_declare_local a || has_declare_local b
  | LL.For_loop { body; _ } | LL.If { body; _ } -> has_declare_local body
  | _ -> false

let annotated seg = not (List.is_empty (LL.hardware_axes seg.LL.llc))

(* --- 1. Executed: materialized-intermediate chain — the aligned cross-nest rule keeps the
   pointwise producer/consumer pair in one parallel kernel (no fission cut). --- *)
let () =
  let n = 512 in
  let av = Array.init (n * n) ~f:(fun i -> Float.of_int (i % 19) *. 0.5) in
  let bv = Array.init (n * n) ~f:(fun i -> Float.of_int (i % 23) -. 11.) in
  let expected = Array.init (n * n) ~f:(fun i -> (av.(i) +. bv.(i)) *. (av.(i) +. bv.(i))) in
  let a = TDSL.ndarray av ~label:[ "a" ] ~output_dims:[ n; n ] () in
  let b = TDSL.ndarray bv ~label:[ "b" ] ~output_dims:[ n; n ] () in
  let%op d = a + b in
  Train.set_materialized d.Tensor.value;
  let%op e = d *. d in
  let comp = named "fission_chain" (Train.forward e) in
  let ctx = Context.auto () in
  let ctx, routine = Context.compile ctx comp Ir.Indexing.Empty in
  let ctx = Context.run ctx routine in
  let got = Context.get_values ctx e.Tensor.value in
  p_all2 "chain values correct" got expected ~f:approx;
  (* Run again: results must be stable. *)
  let ctx = Context.run ctx routine in
  let got2 = Context.get_values ctx e.Tensor.value in
  p "chain rerun deterministic" (Array.equal Float.( = ) got got2);
  (let src = Generated.read "fission_chain" in
   p "chain stays a single aligned kernel" (count_substr src "__seg0" = 0);
   let parallel_ok = if on_cpu then has_parallel_construct src else has_hardware_regs src in
   p "the merged chain kernel parallelizes" parallel_ok);

  (* --- 2. Structural: the same chain analyzed for the metal backend, on captured lowered code
     (runs identically under every configured backend). --- *)
  let%op d2 = a + b in
  Train.set_materialized d2.Tensor.value;
  let%op e2 = d2 *. d2 in
  let comp2 = named "fission_capture" (Train.forward e2) in
  let stash = ref None in
  let _ctx, _routine =
    Context.compile
      ~lowered_transform:(fun opt ->
        stash := Some opt;
        [ opt ])
      (Context.auto ()) comp2 Ir.Indexing.Empty
  in
  let opt = Option.value_exn ~here:[%here] !stash in
  let segs = Sched.maybe_default_schedules ~backend_name:"metal" ~static_indices:[] opt in
  p "capture: one aligned segment" (List.length segs = 1);
  p_all "capture: the merged segment is hardware-annotated" segs ~f:annotated;
  (match segs with
  | [ seg0 ] ->
      let mem seg tn = Hashtbl.mem seg.LL.traced_store tn in
      p "capture: traced store covers the whole chain"
        (mem seg0 d2.Tensor.value && mem seg0 e2.Tensor.value)
  | _ -> p "capture: traced store covers the whole chain" false);
  let identity = Sched.maybe_default_schedules ~backend_name:"interpreter" ~static_indices:[] opt in
  p "capture: identity on non-scheduling backends"
    (match identity with [ o ] -> phys_equal o opt | _ -> false)

(* --- 3. Structural, hand-built: scope-local replication, merge-back, Local promotion --- *)

let fresh_tn =
  let c = ref 950_000_000 in
  fun label dims ->
    Int.incr c;
    Tn.create (Tn.Specified Ir.Ops.single) ~id:!c ~label:[ label ]
      ~unpadded_dims:(lazy dims)
      ~padding:(lazy None)
      ()

let sp = Ir.Ops.single

let for_over ?(extent = 4096) sym body =
  LL.For_loop { index = sym; from_ = 0; to_ = extent - 1; body; axis = LL.Serial }

let hand_built ~stmts ~tns_on_device ~tns_local =
  let optimize_ctx = LL.empty_optimize_ctx () in
  let plc = optimize_ctx.LL.placements in
  List.iter tns_on_device ~f:(fun tn -> Tn.Placements.update plc tn Tn.On_device "49:test-setup");
  List.iter tns_local ~f:(fun tn -> Tn.Placements.update plc tn Tn.Local "49:test-setup");
  let traced_store = Hashtbl.create (module Tn) in
  let llc = LL.unflat_lines stmts in
  List.iter (tns_on_device @ tns_local) ~f:(fun tn ->
      ignore (LL.get_node traced_store tn : LL.traced_array));
  {
    LL.traced_store;
    optimize_ctx;
    llc;
    merge_node = None;
    workgroup_shared = Set.empty (module Tn);
    simdgroup_fragments = Set.empty (module Tn);
    swizzled = Map.empty (module Tn);
    pipelined = Map.empty (module Tn);
    zero_fringe = Set.empty (module Tn);
    flip_candidates = [];
    spliced_rbw = Base.Set.empty (module Tn);
  }

let () =
  (* Shared hoisted scope-local [v]: two nests over a materialized producer/consumer edge, both
     reading [v]. The consumer segment must receive a replica of the definition. The consumer reads
     the intermediate at [Fixed_idx 0] — a misaligned (genuinely racy) edge, so the aligned
     cross-nest rule cannot merge the pair and the cut is forced. *)
  let a = fresh_tn "ha" [| 4096 |] in
  let m1 = fresh_tn "hm1" [| 4096 |] in
  let m2 = fresh_tn "hm2" [| 4096 |] in
  let vtn = fresh_tn "hv" [| 1 |] in
  let v = LL.get_scope vtn in
  let get tn idx = LL.Get (tn, [| idx |]) in
  let def_stmts =
    [ LL.Declare_local { id = v; needs_init = false }; LL.Set_local (v, get a (Idx.Fixed_idx 0)) ]
  in
  let i = Idx.get_symbol () and j = Idx.get_symbol () in
  let nest1 =
    for_over i
      (LL.Set
         {
           tn = m1;
           idcs = [| Idx.Iterator i |];
           llsc = LL.Binop (Ir.Ops.Mul, (get a (Idx.Iterator i), sp), (LL.Get_local v, sp));
           debug = "";
         })
  in
  let nest2 ?(read_a = false) () =
    let rhs =
      if read_a then
        LL.Binop
          ( Ir.Ops.Add,
            ( LL.Binop (Ir.Ops.Add, (get m1 (Idx.Fixed_idx 0), sp), (get a (Idx.Fixed_idx 0), sp)),
              sp ),
            (LL.Get_local v, sp) )
      else LL.Binop (Ir.Ops.Add, (get m1 (Idx.Fixed_idx 0), sp), (LL.Get_local v, sp))
    in
    for_over j (LL.Set { tn = m2; idcs = [| Idx.Iterator j |]; llsc = rhs; debug = "" })
  in
  let opt =
    hand_built ~stmts:(def_stmts @ [ nest1; nest2 () ]) ~tns_on_device:[ a; m1; m2 ] ~tns_local:[]
  in
  let segs = Sched.maybe_default_schedules ~backend_name:"metal" ~static_indices:[] opt in
  p "replication: two segments" (List.length segs = 2);
  p_all "replication: both segments annotated" segs ~f:annotated;
  p "replication: consumer segment carries a replica of the local's definition"
    (match segs with [ _; seg1 ] -> has_declare_local seg1.LL.llc | _ -> false);

  (* Invalid replication: a write to [a] (the definition's read) sits between the definition and the
     consumer, in its own forced segment. The consumer range merges back and runs serially, still
     carrying a valid replica computed before the offending write. *)
  let k = Idx.get_symbol () in
  let a_writer =
    for_over k (LL.Set { tn = a; idcs = [| Idx.Iterator k |]; llsc = LL.Constant 1.; debug = "" })
  in
  let opt =
    hand_built
      ~stmts:(def_stmts @ [ nest1; a_writer; nest2 ~read_a:true () ])
      ~tns_on_device:[ a; m1; m2 ] ~tns_local:[]
  in
  let segs = Sched.maybe_default_schedules ~backend_name:"metal" ~static_indices:[] opt in
  p "merge-back: two segments" (List.length segs = 2);
  (match segs with
  | [ seg0; seg1 ] ->
      p "merge-back: producer segment annotated, merged range serial"
        (annotated seg0 && not (annotated seg1));
      p "merge-back: merged segment still carries the replica" (has_declare_local seg1.LL.llc)
  | _ ->
      p "merge-back: producer segment annotated, merged range serial" false;
      p "merge-back: merged segment still carries the replica" false);

  (* Local scratch stranded across a cut: nest1 writes Local [d]; nest2 writes materialized [m];
     nest3 reads both — the [m] edge, misaligned by the [Fixed_idx 0] read, forces the cut,
     stranding [d] (whose own edge is aligned), which must be promoted. *)
  let d = fresh_tn "hd" [| 4096 |] in
  let m = fresh_tn "hm" [| 4096 |] in
  let m3 = fresh_tn "hm3" [| 4096 |] in
  let s1 = Idx.get_symbol () and s2 = Idx.get_symbol () and s3 = Idx.get_symbol () in
  let w1 =
    for_over s1
      (LL.Set { tn = d; idcs = [| Idx.Iterator s1 |]; llsc = get a (Idx.Iterator s1); debug = "" })
  in
  let w2 =
    for_over s2
      (LL.Set
         {
           tn = m;
           idcs = [| Idx.Iterator s2 |];
           llsc = LL.Binop (Ir.Ops.Mul, (get a (Idx.Iterator s2), sp), (LL.Constant 2., sp));
           debug = "";
         })
  in
  let w3 =
    for_over s3
      (LL.Set
         {
           tn = m3;
           idcs = [| Idx.Iterator s3 |];
           llsc = LL.Binop (Ir.Ops.Add, (get d (Idx.Iterator s3), sp), (get m (Idx.Fixed_idx 0), sp));
           debug = "";
         })
  in
  let opt = hand_built ~stmts:[ w1; w2; w3 ] ~tns_on_device:[ a; m; m3 ] ~tns_local:[ d ] in
  let plc = opt.LL.optimize_ctx.LL.placements in
  p "promotion: scratch starts unmaterialized" (not (Tn.Placements.is_materialized_peek plc d));
  let segs = Sched.maybe_default_schedules ~backend_name:"metal" ~static_indices:[] opt in
  p "promotion: two segments, both annotated"
    (List.length segs = 2 && List.for_all segs ~f:annotated);
  p "promotion: stranded Local scratch promoted to On_device"
    (Tn.Placements.is_materialized_peek plc d)

(* --- 5. Structural, hand-built: the aligned cross-nest rule. Analysis for the metal backend;
   [annotated]/[grid_axes] inspect the emitted hardware loops. --- *)
let () =
  let get tn idcs = LL.Get (tn, idcs) in
  let set tn idcs llsc = LL.Set { tn; idcs; llsc; debug = "" } in
  let segs_of ~stmts ~tns_on_device ~tns_local =
    let opt = hand_built ~stmts ~tns_on_device ~tns_local in
    (opt, Sched.maybe_default_schedules ~backend_name:"metal" ~static_indices:[] opt)
  in
  let grid_axes seg =
    List.count (LL.hardware_axes seg.LL.llc) ~f:(fun ax ->
        match ax.LL.ha_kind with `Grid -> true | `Workgroup -> false)
  in
  let a = fresh_tn "pa" [| 4096 |] in
  let a2 = fresh_tn "pa2" [| 64; 64 |] in

  (* Aligned pointwise producer/consumer: one kernel, both nests annotated. *)
  let m1 = fresh_tn "pm1" [| 4096 |] in
  let m2 = fresh_tn "pm2" [| 4096 |] in
  let i = Idx.get_symbol () and j = Idx.get_symbol () in
  let n1 = for_over i (set m1 [| Idx.Iterator i |] (get a [| Idx.Iterator i |])) in
  let n2 = for_over j (set m2 [| Idx.Iterator j |] (get m1 [| Idx.Iterator j |])) in
  let _, segs = segs_of ~stmts:[ n1; n2 ] ~tns_on_device:[ a; m1; m2 ] ~tns_local:[] in
  p "aligned: pointwise pair merges into one kernel with both nests annotated"
    (match segs with [ seg ] -> annotated seg && grid_axes seg = 2 | _ -> false);

  (* Extent mismatch: identical geometry is impossible, so the pair still cuts. *)
  let m2b = fresh_tn "pm2b" [| 2048 |] in
  let j' = Idx.get_symbol () in
  let n2b =
    for_over ~extent:2048 j' (set m2b [| Idx.Iterator j' |] (get m1 [| Idx.Iterator j' |]))
  in
  let _, segs = segs_of ~stmts:[ n1; n2b ] ~tns_on_device:[ a; m1; m2b ] ~tns_local:[] in
  p "aligned: extent mismatch still cuts, both segments annotated"
    (match segs with [ s0; s1 ] -> annotated s0 && annotated s1 | _ -> false);

  (* Transposed consumer read: threads would read other threads' elements, so the pair cuts. *)
  let mt1 = fresh_tn "pmt1" [| 64; 64 |] in
  let mt2 = fresh_tn "pmt2" [| 64; 64 |] in
  let i1 = Idx.get_symbol () and i2 = Idx.get_symbol () in
  let j1 = Idx.get_symbol () and j2 = Idx.get_symbol () in
  let nt1 =
    for_over ~extent:64 i1
      (for_over ~extent:64 i2
         (set mt1
            [| Idx.Iterator i1; Idx.Iterator i2 |]
            (get a2 [| Idx.Iterator i1; Idx.Iterator i2 |])))
  in
  let nt2 =
    for_over ~extent:64 j1
      (for_over ~extent:64 j2
         (set mt2
            [| Idx.Iterator j1; Idx.Iterator j2 |]
            (get mt1 [| Idx.Iterator j2; Idx.Iterator j1 |])))
  in
  let _, segs = segs_of ~stmts:[ nt1; nt2 ] ~tns_on_device:[ a2; mt1; mt2 ] ~tns_local:[] in
  p "aligned: transposed read still cuts, both segments annotated"
    (match segs with [ s0; s1 ] -> annotated s0 && annotated s1 | _ -> false);

  (* Parallelism switch: a batch-parallel producer feeding a consumer that reduces over the batch
     (its own chain is the inner loop) can never align — cut, and the consumer parallelizes on its
     inner loop in its own kernel. *)
  let mr = fresh_tn "pmr" [| 4096 |] in
  let mo = fresh_tn "pmo" [| 4096 |] in
  let b = Idx.get_symbol () and b2 = Idx.get_symbol () and o = Idx.get_symbol () in
  let nr = for_over b (set mr [| Idx.Iterator b |] (get a [| Idx.Iterator b |])) in
  let no =
    for_over b2
      (for_over o
         (set mo [| Idx.Iterator o |]
            (LL.Binop
               (Ir.Ops.Add, (get mo [| Idx.Iterator o |], sp), (get mr [| Idx.Iterator b2 |], sp)))))
  in
  let _, segs = segs_of ~stmts:[ nr; no ] ~tns_on_device:[ a; mr; mo ] ~tns_local:[] in
  p "aligned: parallelism switch still cuts, both segments annotated"
    (match segs with [ s0; s1 ] -> annotated s0 && annotated s1 | _ -> false);

  (* Alignment by trimming would shrink the producer's parallelism (2-D chain to 1-D): the lossless
     guard prefers the cut. The consumer's own kernel is annotated as well (64 parallel iterations,
     at the default [gpu_schedule_min_parallel] = 64 — real parallelism beats the serial 1x1
     fallback). *)
  let mp = fresh_tn "pmp" [| 64; 64 |] in
  let mq = fresh_tn "pmq" [| 64 |] in
  let k1 = Idx.get_symbol () and k2 = Idx.get_symbol () and q = Idx.get_symbol () in
  let np =
    for_over ~extent:64 k1
      (for_over ~extent:64 k2
         (set mp
            [| Idx.Iterator k1; Idx.Iterator k2 |]
            (get a2 [| Idx.Iterator k1; Idx.Iterator k2 |])))
  in
  let nq =
    for_over ~extent:64 q
      (set mq [| Idx.Iterator q |] (get mp [| Idx.Iterator q; Idx.Fixed_idx 5 |]))
  in
  let _, segs = segs_of ~stmts:[ np; nq ] ~tns_on_device:[ a2; mp; mq ] ~tns_local:[] in
  p "aligned: lossy trim still cuts, producer keeps its 2-D parallelism"
    (match segs with
    | [ s0; s1 ] -> annotated s0 && List.length (LL.hardware_axes s0.LL.llc) = 2 && annotated s1
    | _ -> false);

  (* Aligned [Local] scratch: the pair shares one kernel — each thread reads back the slice of its
     private copy it wrote — so the scratch needs no promotion, unlike the stranded case of section
     3. *)
  let tmp = fresh_tn "ptmp" [| 4096 |] in
  let mm1 = fresh_tn "pmm1" [| 4096 |] in
  let mm2 = fresh_tn "pmm2" [| 4096 |] in
  let u = Idx.get_symbol () and w = Idx.get_symbol () in
  let nu =
    for_over u
      (LL.Seq
         ( set tmp [| Idx.Iterator u |] (get a [| Idx.Iterator u |]),
           set mm1 [| Idx.Iterator u |] (get a [| Idx.Iterator u |]) ))
  in
  let nw =
    for_over w
      (set mm2 [| Idx.Iterator w |]
         (LL.Binop
            (Ir.Ops.Add, (get tmp [| Idx.Iterator w |], sp), (get mm1 [| Idx.Iterator w |], sp))))
  in
  let opt, segs = segs_of ~stmts:[ nu; nw ] ~tns_on_device:[ a; mm1; mm2 ] ~tns_local:[ tmp ] in
  let plc = opt.LL.optimize_ctx.LL.placements in
  p "aligned: scratch pair merges into one kernel with both nests annotated"
    (match segs with [ seg ] -> annotated seg && grid_axes seg = 2 | _ -> false);
  p "aligned: the scratch is not promoted" (not (Tn.Placements.is_materialized_peek plc tmp))

(* --- 6. Structural: the arity_cuts (finer) segmentation mode, gh-ocannl-574. The default mode
   compares parallelism under the presets' Grid+Workgroup cap, so a materialized GEMM merges with a
   reduction over its minor output axis (the lm_head's max-logits row: the max target's [-inf]
   initialization is a [Set] nest, not a [Zero_out], so nothing separates the three statements) —
   correct for the default annotators, fatal for the full-arity GPU sketches, whose every seed then
   declines on companion coverage. Under [arity_cuts:true] the same code cuts the GEMM into its own
   kernel; shapes whose companions can follow the site's full arity (bias+relu) and shapes already
   separated by a [Zero_out] (a sum-reduce) segment identically in both modes. --- *)
let () =
  let capture comp =
    let stash = ref None in
    let _ctx, _routine =
      Context.compile
        ~lowered_transform:(fun opt ->
          stash := Some opt;
          [ opt ])
        (Context.auto ()) comp Ir.Indexing.Empty
    in
    Option.value_exn ~here:[%here] !stash
  in
  let rec writes_tn tn (llc : LL.t) =
    match llc with
    | LL.Set { tn = t; _ }
    | LL.Set_dynamic { tn = t; _ }
    | LL.Set_from_vec { tn = t; _ }
    | LL.Zero_out t
    | LL.Tile_mma { d = t, _; _ } ->
        Tn.equal t tn
    | LL.Seq (a, b) -> writes_tn tn a || writes_tn tn b
    | LL.For_loop { body; _ } | LL.If { body; _ } -> writes_tn tn body
    | _ -> false
  in
  let segments ~arity_cuts opt =
    (* A hermetic copy per query: [promote_locals] mutates the lowering's placements, and the modes
       must not observe each other's surviving promotions. *)
    let scratch =
      {
        opt with
        LL.traced_store = Hashtbl.copy opt.LL.traced_store;
        LL.optimize_ctx = LL.copy_optimize_ctx opt.LL.optimize_ctx;
      }
    in
    let limits = Ir.Backend_intf.no_hardware_limits in
    Sched.fission_scheduled ~promote_locals:true ~arity_cuts
      ~preset:(Sched.default_gpu ~min_parallel:1 ~limits)
      ~zero_sched:(Sched.zero_expansion ~min_parallel:1 ~limits)
      ~static_indices:[] scratch
  in
  let seg_kinds tuples = List.map tuples ~f:(fun (kind, _, _, _) -> kind) in
  let b = 4 and n = 32 and m = 64 and k = 16 in
  let xv = Array.init (b * n * k) ~f:(fun i -> Float.of_int (i % 7) *. 0.25) in
  let wv = Array.init (k * m) ~f:(fun i -> Float.of_int (i % 5) *. 0.125) in
  let x = TDSL.ndarray xv ~label:[ "fx" ] ~batch_dims:[ b ] ~output_dims:[ n; k ] () in
  let w = TDSL.ndarray wv ~label:[ "fw" ] ~output_dims:[ k; m ] () in
  let%op z = x +* "b|ik;kj=>b|ij" w in
  Train.set_materialized z.Tensor.value;
  let%op r = z @^^ "b|ij => b|i" in
  let opt = capture (named "arity_max" (Train.forward r)) in
  let ztn = z.Tensor.value and rtn = r.Tensor.value in
  (match seg_kinds (segments ~arity_cuts:false opt) with
  | [ `Zeros; `Normal ] ->
      p "arity: default mode merges the GEMM with its row-max companion"
        (match segments ~arity_cuts:false opt with
        | [ _; (_, pre, _, _) ] -> writes_tn ztn pre.LL.llc && writes_tn rtn pre.LL.llc
        | _ -> false)
  | _ -> p "arity: default mode merges the GEMM with its row-max companion" false);
  (match segments ~arity_cuts:true opt with
  | [ (`Zeros, _, _, _); (`Normal, gemm, _, _); (`Normal, red, _, _) ] ->
      p "arity: arity_cuts frees the GEMM from the row-max companion"
        (writes_tn ztn gemm.LL.llc
        && (not (writes_tn rtn gemm.LL.llc))
        && writes_tn rtn red.LL.llc
        && not (writes_tn ztn red.LL.llc))
  | _ -> p "arity: arity_cuts frees the GEMM from the row-max companion" false);
  (* Full-arity-compatible companion: bias+relu follows the site's chain, so both modes keep the
     merged single kernel. *)
  let x2 = TDSL.ndarray xv ~label:[ "fx2" ] ~batch_dims:[ b ] ~output_dims:[ n; k ] () in
  let w2 = TDSL.ndarray wv ~label:[ "fw2" ] ~output_dims:[ k; m ] () in
  let bv = Array.init m ~f:(fun i -> Float.of_int (i % 3) -. 1.) in
  let bias = TDSL.ndarray bv ~label:[ "fb2" ] ~output_dims:[ m ] () in
  let%op z2 = x2 +* "b|ik;kj=>b|ij" w2 in
  Train.set_materialized z2.Tensor.value;
  let%op y2 = relu (z2 + bias) in
  let opt2 = capture (named "arity_relu" (Train.forward y2)) in
  p "arity: bias+relu companion stays merged in both modes"
    (List.equal Poly.equal (seg_kinds (segments ~arity_cuts:false opt2)) [ `Zeros; `Normal ]
    && List.equal Poly.equal (seg_kinds (segments ~arity_cuts:true opt2)) [ `Zeros; `Normal ]);
  (* A sum-reduce target is zero-initialized, so its [Zero_out] already separates the statements:
     both modes agree. *)
  let x3 = TDSL.ndarray xv ~label:[ "fx3" ] ~batch_dims:[ b ] ~output_dims:[ n; k ] () in
  let w3 = TDSL.ndarray wv ~label:[ "fw3" ] ~output_dims:[ k; m ] () in
  let%op z3 = x3 +* "b|ik;kj=>b|ij" w3 in
  Train.set_materialized z3.Tensor.value;
  let%op r3 = z3 ++ "b|ij => b|i" in
  let opt3 = capture (named "arity_sum" (Train.forward r3)) in
  p "arity: Zero_out-separated sum-reduce segments identically in both modes"
    (List.equal Poly.equal
       (seg_kinds (segments ~arity_cuts:false opt3))
       (seg_kinds (segments ~arity_cuts:true opt3))
    && List.length (seg_kinds (segments ~arity_cuts:false opt3)) = 4)

(* --- 4. Executed: backward pass — Zero_out and reduction statements segment away from the gradient
   accumulation nest. --- *)
let () =
  let n = 192 in
  let xv = Array.init (n * n) ~f:(fun i -> Float.of_int (i % 29) *. 0.125) in
  let x = TDSL.ndarray xv ~label:[ "x" ] ~output_dims:[ n; n ] () in
  let%op l = ({ w = uniform (); o = [ 192; 192 ] } *. x) ++ "...|... => |->0" in
  let update = named "fission_bwd" (Train.grad_update l) in
  let ctx = Train.init_params (Context.auto ()) IDX.empty l in
  let ctx, routine = Train.to_routine ctx IDX.empty update in
  let ctx = Context.run ctx routine in
  let w =
    List.find_exn (Set.to_list l.Tensor.params) ~f:(fun t ->
        String.equal (Tn.debug_name t.Tensor.value) "w")
  in
  let wg = Context.get_values ctx (grad_of w) in
  (* dl/dw = x. *)
  p_all2 "backward gradient values correct" wg xv ~f:approx;
  Generated.assert_emits ~routine:"fission_bwd__seg" ~contains:"__seg0" "backward pass fissioned"
