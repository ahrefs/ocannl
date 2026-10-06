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
   the inner extent, so the composed index is not the address) decline, each for its own reason.

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
let accum_prec =
  let caps = lazy (Context.codegen_capabilities (Context.auto ())) in
  fun p -> (Lazy.force caps).Ir.Backend_intf.accum_prec p

open Verdict.Claims

let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")
let skipped = Verdict.skipped ~backend:backend_name
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
  let merged_extent = ref None and flattened_writes = ref 0 in
  L.walk oc.LL.llc ~on_stmt:(function
    | LL.For_loop { index; to_; _ } when Ir.Indexing.equal_symbol index merged ->
        merged_extent := Some (to_ + 1)
    | LL.Set { tn; idcs; _ }
      when Ir.Tnode.equal tn d
           && Array.equal Ir.Indexing.equal_axis_index (Array.sub idcs ~pos:2 ~len:2)
                [| Ir.Indexing.Sub_axis; Ir.Indexing.Iterator merged |] ->
        Int.incr flattened_writes
    | _ -> ());
  p "control: the merged loop spans every head's columns"
    (Poly.equal !merged_extent (Some (hh * ee)));
  p "control: the writes read the merged index flattened over the pair" (!flattened_writes > 0);
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
   [Sub_axis] decomposes to nothing, the marker comes back as [Fixed_idx 0]: the combine's write
   renders the same address but reads to the address queries as an ordinary in-bounds coordinate,
   under which the flattened component [f] (extent [hh * ee]) is taken to stay below [ee]. The
   executed leg runs the composition through the default (fissioned, on GPU hardware-mapped)
   schedules. *)

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
   affine decomposition, where a [Sub_axis] decomposes to nothing, it comes back as [Fixed_idx 0]:
   the same address, read by the address queries as an ordinary coordinate under which the flattened
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
  let prefix = Autotune.coalesce_prefix opt in
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
    ~f:(fun q -> q.Autotune.sk_coalesce && blocktile q && not q.Autotune.sk_epilogue);
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
  if on_gpu then begin
    let n_match = ref 0 in
    List.iter schedules ~f:(fun (q, _) ->
        match
          run_with fwd cand (fun o -> Sched.apply (Autotune.sketch_schedule ~accum_prec ~p:q o) o)
        with
        | got -> if Array.equal Float.equal got want then Int.incr n_match
        | exception exn -> Stdio.eprintf "coalesced seed FAILED: %s\n" (Exn.to_string exn));
    p "projection: every coalesced seed executes to the serial reference bitwise"
      (!n_match = List.length schedules && !n_match > 0)
  end
  else begin
    Stdio.eprintf "%s cannot execute workgroup-shared staging — the execution leg is skipped\n"
      backend_name;
    skipped "projection: every coalesced seed executes to the serial reference bitwise"
  end

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
  p "companion: the site's own pair still coalesces" (Option.is_some (Autotune.coalesce_prefix opt));
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
      match (kind, Autotune.coalesce_prefix pre) with
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
    Sched.fission_scheduled ~fold_zeros:true ~keep_mapping:(Sched.default_gpu ~limits)
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
        match (kind, Autotune.coalesce_prefix pre) with
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
      let label = "folded: every coalesced seed executes to the materialized run" in
      if on_gpu then
        p_all label
          (List.mapi seeds ~f:(fun i q -> (i, q)))
          ~f:(fun (i, q) ->
            let o = apply (Autotune.sketch_schedule ~accum_prec ~p:q pre) in
            let got =
              List.hd_exn
                (L.execute
                   ~name:("co_folded_seed_" ^ Int.to_string i)
                   o ~seed ~read:[ cand.Tensor.value ])
            in
            Array.equal Float.equal got want)
      else skipped label)
