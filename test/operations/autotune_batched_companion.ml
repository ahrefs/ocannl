(* gh-ocannl-569: the companion-coverage rule (gh-ocannl-521) must not decline a BATCHED site.

   The workload is the shape of gpt2's FFN up-projection: a rank-3 GEMM (batch x tokens x features,
   the `8x128x1024` geometry of the gh-531 profile) whose materialized output feeds an elementwise
   companion nest (bias + relu) in the same routine. The site's chain is three loops, but
   {!Sched.aligned_chains} used to cap every nest's chain at two — so [companion_geometry]'s
   full-arity check could never match a rank-3 site, and every GPU sketch seed (scalar AND
   tensorized) declined with "the accumulation nest's aligned chain was trimmed below its 8x128x1024
   geometry". That single decline pinned five FFN-class kernels to ~1.3% of fp32 peak — 70% of the
   gpt2_mini step on CUDA, 47% on HIP.

   Two layers of pinning: - Structural (backend-independent, runs on cc too): the scalar GPU seeds'
   schedules CONSTRUCT — [Autotune.sketch_schedule] does not raise the companion-coverage decline —
   and the constructed schedule spreads the minor output axis (j) across [Grid] blocks. - Executable
   (GPU backends): each unfused scalar GPU seed compiles and computes correctly. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module Sched = Ir.Schedule
module LL = Ir.Low_level
module Asgns = Ir.Assignments
open Verdict.Claims

(* The report's outcome as the questions this test asks of it (gh-ocannl-677): the outcome is a
   variant naming one of five mutually exclusive states, so a claim names the state it means instead
   of combining flags — [not (replayed r)] in particular does NOT say a search ran. *)
let replayed (r : Autotune.report) =
  match r.Autotune.outcome with Autotune.Cache_replay -> true | _ -> false

let completed (r : Autotune.report) =
  match r.Autotune.outcome with Autotune.Searched -> true | _ -> false

let named name (comp : Asgns.comp) : Asgns.comp =
  { comp with asgns = Asgns.Block_comment (name, comp.asgns) }

let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")
let is_gpu = Sched.backend_is_gpu backend_name

(* The family tree's unfused flavor is refuted on companion coverage and on nothing else (the fused
   flavor, the root's other child, is refuted separately — these sites feed their output to a
   reduction, not a fusable tail — and carries the fusion recognizer's reason, gh-ocannl-613). *)
let coverage_refutes_unfused tree =
  let refs =
    List.filter (Ir.Schedule_space.refutations tree) ~f:(fun (path, _) ->
        List.exists path ~f:(fun (_, decision) ->
            Autotune.Family_decision.equal decision (Autotune.Family_decision.Fusion `Unfused)))
  in
  (not (List.is_empty refs))
  && List.for_all refs ~f:(fun (_, w) ->
      String.is_substring w ~substring:"companion coverage (gh-521)")

let clean_cache dir =
  if Stdlib.Sys.file_exists dir && Stdlib.Sys.is_directory dir then
    Array.iter (Stdlib.Sys.readdir dir) ~f:(fun f ->
        Stdlib.Sys.remove (Stdlib.Filename.concat dir f))

let () =
  clean_cache "autotune_cache_batched_companion";
  let b = 4 and n = 32 and m = 64 and k = 16 in
  let xv = Array.init (b * n * k) ~f:(fun i -> Float.of_int (i % 7) *. 0.25) in
  let wv = Array.init (k * m) ~f:(fun i -> Float.of_int (i % 5) *. 0.125) in
  let bv = Array.init m ~f:(fun i -> Float.of_int (i % 3) -. 1.) in
  let expected =
    Array.init
      (b * n * m)
      ~f:(fun idx ->
        let bi = idx / (n * m) in
        let i = idx % (n * m) / m and j = idx % m in
        let acc = ref 0. in
        for kk = 0 to k - 1 do
          acc := !acc +. (xv.((bi * n * k) + (i * k) + kk) *. wv.((kk * m) + j))
        done;
        Float.max 0. (!acc +. bv.(j)))
  in
  let x = TDSL.ndarray xv ~label:[ "bc_x" ] ~batch_dims:[ b ] ~output_dims:[ n; k ] () in
  let w = TDSL.ndarray wv ~label:[ "bc_w" ] ~output_dims:[ k; m ] () in
  let bias = TDSL.ndarray bv ~label:[ "bc_b" ] ~output_dims:[ m ] () in
  let%op z = x +* "b|ik;kj=>b|ij" w in
  Train.set_materialized z.Tensor.value;
  let%op y = relu (z + bias) in
  let comp = named "bc_head" (Train.forward y) in

  (* Capture the lowered routine once; enumerate GPU seeds regardless of the running backend —
     schedule CONSTRUCTION (where the gh-521 companion-coverage decline fires) is
     backend-independent. *)
  let limits = Context.hardware_limits (Context.auto ()) in
  let captured = ref None in
  let _ctx, _r =
    Context.compile
      ~lowered_transform:(fun opt ->
        captured := Some opt;
        [ opt ])
      (Context.auto ()) comp Ir.Indexing.Empty
  in
  let opt = Option.value_exn !captured in
  let gpu_seeds opt =
    List.filter (Autotune.sketch_seed_params ~is_gpu:true ~is_cpu:false ~limits opt) ~f:(fun p ->
        p.Autotune.sk_gpu && not p.Autotune.sk_epilogue)
  in
  let seeds = gpu_seeds opt in
  p "bc: scalar GPU seeds exist for the batched site"
    (List.exists seeds ~f:(fun p -> not p.Autotune.sk_mma));
  (* The production census declined the tensorized seeds in the same proportion as the scalar ones
     (10 and 10) — both flavors route through [companion_geometry]. Tensorized seeds only exist
     where the backend advertises an mma format for f32 (Metal locally; CUDA/HIP need the tf32 arm),
     so this line is informational rather than golden-pinned. *)
  if not (List.exists seeds ~f:(fun p -> p.Autotune.sk_mma)) then
    Stdio.eprintf "bc: no tensorized seed for this site on %s — mma legs below are vacuous\n"
      backend_name;
  let constructed =
    List.map seeds ~f:(fun sp ->
        match Autotune.sketch_schedule ~p:sp opt with
        | sched -> Either.First sched
        | exception Ir.Schedule_outcome.Cause_at (_, Ir.Schedule_outcome.Unsupported { detail; _ })
          ->
            Either.Second detail
        | exception exn -> Either.Second (Exn.to_string exn))
  in
  let declines =
    List.filter_map constructed ~f:(function Either.Second e -> Some e | _ -> None)
  in
  p_empty "bc: no seed declines on companion coverage" ~over:constructed declines;
  List.iter declines ~f:(fun e -> Stdio.eprintf "bc decline: %s\n" e);
  (* The point of the fix is reach: the minor output axis (j) must actually carry [Grid] blocks in
     the constructed schedule, not merely survive construction. A schedule constructed under the old
     rule could not exist at all; the guard is against a future regression that constructs a
     j-serial form. m = 64 is unique to the minor output axis here (b = 4, i = 32, k = 16), in the
     site's nest and its companions alike.

     TWO claims, because since gh-ocannl-730 the menu reaches j's whole extent in a single block:
     padding lets bm = 64 apply to this site's 32 rows, and that geometry's bn = 64 then covers j in
     one Grid block. Measured against the pre-gh-730 seeding, that geometry is an ADDITION — every
     geometry that spread j before still does, and two more now exist — so the [for_all] that used
     to carry the whole property is split rather than weakened. The first claim is the
     anti-regression invariant proper and runs over ALL seeds: j is Grid-typed in every constructed
     schedule, never left serial. The second keeps [for_all] strength over exactly the geometries
     the property can speak about — those whose column block is narrower than j — and requires that
     set to be non-empty, so it cannot pass by going vacuous. Both are stated over the resulting
     BLOCK COUNT rather than over the split factor (gh-ocannl-744), so neither pins the provenance
     of the geometry that produced it. *)
  let bounds = LL.loop_bounds opt.LL.llc in
  let j_grid_factors bounds sched =
    List.filter_map sched ~f:(function
      | Sched.Split { axis; factor; outer = LL.Grid; _ } -> (
          match List.Assoc.find bounds axis ~equal:Ir.Indexing.equal_symbol with
          | Some (0, hi) when hi + 1 = m -> Some factor
          | _ -> None)
      | _ -> None)
  in
  (* Stated as a BLOCK COUNT, not as a comparison of the split factor against the extent
     (gh-ocannl-744): what the leg is about is that the minor output axis carries block-level
     parallelism, and any factor yielding more than one block delivers that, whatever geometry
     produced it. *)
  let j_block_counts bounds sched =
    List.map (j_grid_factors bounds sched) ~f:(fun f -> (m + f - 1) / f)
  in
  let blocks_j bounds sched = not (List.is_empty (j_block_counts bounds sched)) in
  let spreads_j sched = List.exists (j_block_counts bounds sched) ~f:(fun c -> c > 1) in
  let built =
    List.filter_map (List.zip_exn seeds constructed) ~f:(function
      | sp, Either.First sched -> Some (sp, sched)
      | _, Either.Second _ -> None)
  in
  p "bc: every constructed schedule Grid-blocks the minor output axis, never leaving it serial"
    (List.length built = List.length seeds
    && (not (List.is_empty built))
    && List.for_all built ~f:(fun (_, sched) -> blocks_j bounds sched));
  let narrower = List.filter built ~f:(fun (sp, _) -> sp.Autotune.sk_bn < m) in
  p_all "bc: the minor output axis carries block-level parallelism wherever the geometry allows"
    narrower ~f:(fun (_, s) -> spreads_j s);
  (* Negative control: the detector has teeth. The CPU pipelines block the ROW axis (under
     [sk_grid]) and never j, so [blocks_j] must reject every CPU-seeded schedule — if it answered
     true there it would be matching any Grid split at all, and both claims above would hold for a
     j-serial form. *)
  let cpu_scheds =
    List.filter_map
      (List.filter (Autotune.sketch_seed_params ~is_gpu:false ~is_cpu:true ~limits opt)
         ~f:(fun sp -> not sp.Autotune.sk_epilogue))
      ~f:(fun sp ->
        match Autotune.sketch_schedule ~p:sp opt with sched -> Some sched | exception _ -> None)
  in
  p_none "bc: the j-blocking detector rejects the CPU pipelines, which block the row axis instead"
    cpu_scheds ~f:(fun s -> blocks_j bounds s);

  (* Executable parity, seed by seed, on backends that can run shared staging. Vacuous on cc —
     announced on stderr so it is not mistaken for coverage; the golden stays
     backend-independent. *)
  if not is_gpu then
    Stdio.eprintf "bc: %s is not a GPU backend — the executable parity check below is vacuous\n"
      backend_name;
  p_all "bc: every unfused GPU seed compiles and computes correctly"
    (List.init (List.length seeds) ~f:Fn.id)
    ~f:(fun idx ->
      (not is_gpu)
      ||
      let transform opt =
        let sp = List.nth_exn (gpu_seeds opt) idx in
        Sched.apply (Autotune.sketch_schedule ~p:sp opt) opt
      in
      match
        let sctx, sroutine =
          Context.compile
            ~lowered_transform:(fun o -> [ transform o ])
            (Context.auto ()) comp Ir.Indexing.Empty
        in
        let sctx = Context.run sctx sroutine in
        Context.get_values sctx y.Tensor.value
      with
      | got -> Array.for_all2_exn got expected ~f:(fun a b -> Float.(abs (a - b) < 1e-3))
      | exception exn ->
          Stdio.eprintf "bc: scalar GPU seed %d FAILED %s\n" idx (Exn.to_string exn);
          false);

  (* Tune integration: the search itself (fission, per-segment seeding, replay) must route through
     the widened coverage — and the winner must still compute the right values. On cc the GPU
     sketches are not seeded, so the decline assertion is vacuously true there; on GPU backends it
     is the production pin: gpt2's search logged this exact decline key 20 times per arm. *)
  let reports = ref [] in
  let ctx = Context.auto () in
  let ctx, routine =
    Autotune.tune ~beam_width:2 ~rounds:0 ~repeats:1 ~cache_dir:"autotune_cache_batched_companion"
      ~timing_ctx:(Context.auto ())
      ~report:(fun r -> reports := r :: !reports)
      ctx comp Ir.Indexing.Empty
  in
  let ctx = Context.run ctx routine in
  let got = Context.get_values ctx y.Tensor.value in
  p_all2 "bc: tuned batched head matches the reference" got expected ~f:(fun a b ->
      Float.(abs (a - b) < 1e-3));
  (* Exactly one report, then its census: the callback contract is a claim of its own, and the
     census is quantified over the reports, so a claim over zero reports cannot call it clean
     without having inspected one. *)
  p "bc: the tuning search reports exactly once" (List.length !reports = 1);
  p_empty "bc: the tuning census records no companion-coverage decline" ~over:!reports
    (List.concat_map !reports ~f:(fun r ->
         List.filter r.Autotune.declines ~f:(fun d ->
             match d.Autotune.key with
             | Ir.Schedule_outcome.Unsupported_key k ->
                 String.equal k "autotune_sketch_companion_coverage"
             | _ -> false)));

  (* The safety boundary the lifted arity must NOT cross (the lm_head shape): a companion that
     REDUCES over the site's minor output axis reads cells every j-block wrote, with no intra-kernel
     synchronization — spreading j is a race there until fission separates the reduction. Since
     gh-ocannl-577 the coverage verdict is decided at tree construction (here the analysis bails
     outright on the reduction target's whole-node [Zero_out]; a pre-zeroed variant would instead
     trim the component's common prefix below the site's arity): the family refutes with the
     coverage witness before any seed is proposed, so no schedule is constructed AND none is
     enumerated. *)
  let x2 = TDSL.ndarray xv ~label:[ "bc_x2" ] ~batch_dims:[ b ] ~output_dims:[ n; k ] () in
  let w2 = TDSL.ndarray wv ~label:[ "bc_w2" ] ~output_dims:[ k; m ] () in
  let%op z2 = x2 +* "b|ik;kj=>b|ij" w2 in
  Train.set_materialized z2.Tensor.value;
  let%op r2 = z2 ++ "b|ij => b|i" in
  let comp2 = named "bc_rowsum" (Train.forward r2) in
  let captured2 = ref None in
  let _ctx, _r =
    Context.compile
      ~lowered_transform:(fun opt ->
        captured2 := Some opt;
        [ opt ])
      (Context.auto ()) comp2 Ir.Indexing.Empty
  in
  let opt2 = Option.value_exn !captured2 in
  let seeds2 = gpu_seeds opt2 in
  p "bc: reduction-over-j companion refutes the GPU family pre-proposal"
    (List.is_empty seeds2
    &&
    match Autotune.matmul_sketch_tree ~is_gpu:true ~is_cpu:false ~limits opt2 with
    | Some tree -> coverage_refutes_unfused tree
    | None -> false);
  (* The builders' raise site remains the safety net for parameters replayed against a lowering they
     were not seeded for (the fission-recombination scenario): a seed minted on the buildable
     batched head must still raise the coverage decline when applied to the reduction-companion
     routine. *)
  p "bc: a foreign seed replayed against the companion routine still raises the decline"
    (match seeds with
    | sp :: _ -> (
        match Autotune.sketch_schedule ~p:sp opt2 with
        | _ -> false
        | exception Ir.Schedule_outcome.Cause_at (_, Ir.Schedule_outcome.Unsupported { feature; _ })
          ->
            String.equal feature "autotune_sketch_companion_coverage"
        | exception _ -> false)
    | [] -> false)

(* gh-ocannl-574 (the gh-569 residual): the boundary above is respected by CUTTING, not by coverage.
   The lm_head shape proper — a materialized GEMM whose row-MAX companion follows in the same
   fission segment (a max target's [-inf] initialization is a [Set] nest, so no [Zero_out] separates
   the statements, unlike the rowsum above) — must fission apart under the finer [arity_cuts]
   segmentation: the GEMM ships alone and its seeds then spread the minor output axis (j) across
   [Grid] blocks, while the reduction runs as its own downstream kernel with the stream supplying
   the synchronization. *)
let () =
  let b = 4 and n = 32 and m = 64 and k = 16 in
  let xv = Array.init (b * n * k) ~f:(fun i -> Float.of_int (i % 7) *. 0.25) in
  let wv = Array.init (k * m) ~f:(fun i -> Float.of_int (i % 5) *. 0.125) in
  let z_expected =
    Array.init
      (b * n * m)
      ~f:(fun idx ->
        let bi = idx / (n * m) in
        let i = idx % (n * m) / m and j = idx % m in
        let acc = ref 0. in
        for kk = 0 to k - 1 do
          acc := !acc +. (xv.((bi * n * k) + (i * k) + kk) *. wv.((kk * m) + j))
        done;
        !acc)
  in
  let r_expected =
    Array.init (b * n) ~f:(fun row ->
        Array.fold
          (Array.sub z_expected ~pos:(row * m) ~len:m)
          ~init:Float.neg_infinity ~f:Float.max)
  in
  let mk_comp tag =
    let x = TDSL.ndarray xv ~label:[ tag ^ "_x" ] ~batch_dims:[ b ] ~output_dims:[ n; k ] () in
    let w = TDSL.ndarray wv ~label:[ tag ^ "_w" ] ~output_dims:[ k; m ] () in
    let%op z = x +* "b|ik;kj=>b|ij" w in
    Train.set_materialized z.Tensor.value;
    let%op r = z @^^ "b|ij => b|i" in
    (named (tag ^ "_head") (Train.forward r), z, r)
  in
  let comp, z, r = mk_comp "lm" in
  let limits = Context.hardware_limits (Context.auto ()) in
  let captured = ref None in
  let _ctx, _r =
    Context.compile
      ~lowered_transform:(fun opt ->
        captured := Some opt;
        [ opt ])
      (Context.auto ()) comp Ir.Indexing.Empty
  in
  let opt = Option.value_exn !captured in
  let gpu_seeds opt =
    List.filter (Autotune.sketch_seed_params ~is_gpu:true ~is_cpu:false ~limits opt) ~f:(fun p ->
        p.Autotune.sk_gpu && not p.Autotune.sk_epilogue)
  in
  (* The fission segmentation, exactly as the autotuner's seed enumeration runs it (GPU pipeline;
     hermetic copy per query — [promote_locals] mutates placements). *)
  let segments ~arity_cuts opt =
    let scratch =
      {
        opt with
        LL.traced_store = Hashtbl.copy opt.LL.traced_store;
        LL.optimize_ctx = LL.copy_optimize_ctx opt.LL.optimize_ctx;
      }
    in
    Sched.fission_scheduled ~promote_locals:true ~arity_cuts
      ~preset:(Sched.default_gpu ~min_parallel:1 ~limits)
      ~zero_sched:(Sched.zero_expansion ~min_parallel:1 ~limits)
      ~static_indices:[] scratch
  in
  let normals tuples =
    List.filter_map tuples ~f:(fun (kind, pre, _, _) ->
        match kind with `Normal -> Some pre | `Zeros | `Solo -> None)
  in
  (* Coarse segmentation: the GEMM shares its segment with the row-max companion (and the max
     target's initialization nest) — the production decline gh-574 sets out to relieve. Since
     gh-ocannl-577 the segment's family refutes on the companion-coverage witness at tree
     construction, so its seed list is empty rather than every proposed seed declining at build. *)
  let coarse = normals (segments ~arity_cuts:false opt) in
  p "lm: coarse fission keeps the row-max companion in the GEMM's segment"
    (match coarse with
    | [ seg ] -> (
        List.is_empty (gpu_seeds seg)
        &&
        match Autotune.matmul_sketch_tree ~is_gpu:true ~is_cpu:false ~limits seg with
        | Some tree -> coverage_refutes_unfused tree
        | None -> false)
    | _ -> false);
  (* Finer segmentation: the GEMM segment is freed, its seeds construct, and they spread j — the
     axis whose extent m = 64 is unique in this workload (b = 4, i = 32, k = 16). *)
  let bounds seg = LL.loop_bounds seg.LL.llc in
  let j_grid_factors seg sched =
    List.filter_map sched ~f:(function
      | Sched.Split { axis; factor; outer = LL.Grid; _ } -> (
          match List.Assoc.find (bounds seg) axis ~equal:Ir.Indexing.equal_symbol with
          | Some (0, hi) when hi + 1 = m -> Some factor
          | _ -> None)
      | _ -> None)
  in
  let j_block_counts seg sched =
    List.map (j_grid_factors seg sched) ~f:(fun f -> (m + f - 1) / f)
  in
  let blocks_j seg sched = not (List.is_empty (j_block_counts seg sched)) in
  let spreads_j seg sched = List.exists (j_block_counts seg sched) ~f:(fun c -> c > 1) in
  let fine_gemm_seg tuples =
    List.find (normals tuples) ~f:(fun seg -> not (List.is_empty (gpu_seeds seg)))
  in
  (* Split the same way as the [bc] leg above, and for the same reason (gh-ocannl-730): every freed
     seed must Grid-block j, and every seed whose column block is narrower than j must spread it
     across more than one block. *)
  let fine_built =
    match fine_gemm_seg (segments ~arity_cuts:true opt) with
    | None -> None
    | Some seg ->
        let seeds = gpu_seeds seg in
        let built =
          List.filter_map seeds ~f:(fun sp ->
              match Autotune.sketch_schedule ~p:sp seg with
              | sched -> Some (sp, sched)
              | exception exn ->
                  Stdio.eprintf "lm: fine seed FAILED to construct: %s\n" (Exn.to_string exn);
                  None)
        in
        if List.length built <> List.length seeds then None else Some (seg, built)
  in
  p "lm: arity_cuts fission frees the GEMM and its seeds Grid-block j"
    (match fine_built with
    | None -> false
    | Some (seg, built) ->
        (not (List.is_empty built)) && List.for_all built ~f:(fun (_, s) -> blocks_j seg s));
  p "lm: the freed GEMM's seeds carry block-level parallelism on j wherever the geometry allows"
    (match fine_built with
    | None -> false
    | Some (seg, built) ->
        let narrower = List.filter built ~f:(fun (sp, _) -> sp.Autotune.sk_bn < m) in
        (not (List.is_empty narrower)) && List.for_all narrower ~f:(fun (_, s) -> spreads_j seg s));
  (* Executed: the finer fissioned form (default per-segment presets) computes the same values — the
     cut's stream-order synchronization replaces the fused segment's serial order. Runs on every
     backend. *)
  let comp_e, z_e, r_e = mk_comp "lme" in
  let is_cpu = Sched.backend_is_cpu backend_name in
  let transforms opt =
    let preset seg =
      if is_gpu then Sched.default_gpu ~min_parallel:1 ~limits seg
      else if is_cpu then Sched.default_cpu ~min_parallel:1 seg
      else []
    in
    let zero_sched tns = if is_gpu then Sched.zero_expansion ~limits tns else [] in
    List.map
      (Sched.fission_scheduled ~promote_locals:is_gpu ~arity_cuts:true ~preset ~zero_sched
         ~static_indices:[] opt) ~f:(fun (_, _, _, post) -> post)
  in
  (* Non-empty by construction (gh-ocannl-746): two arrays emptied together compare equal. *)
  let approx got want =
    (not (Array.is_empty got))
    && Array.for_all2_exn got want ~f:(fun a c -> Float.(abs (a - c) < 1e-3))
  in
  p "lm: the finer fissioned form executes correctly"
    (match
       let ctx, routine =
         Context.compile ~lowered_transform:transforms (Context.auto ()) comp_e Ir.Indexing.Empty
       in
       let ctx = Context.run ctx routine in
       (Context.get_values ctx z_e.Tensor.value, Context.get_values ctx r_e.Tensor.value)
     with
    | got_z, got_r -> approx got_z z_expected && approx got_r r_expected
    | exception exn ->
        Stdio.eprintf "lm: finer fissioned execution FAILED: %s\n" (Exn.to_string exn);
        false);
  (* Executed per seed (GPU backends): each fine GPU seed's sketch schedule on the freed GEMM
     segment, presets elsewhere — the executable half of the spreads-j pin. Vacuous on cc, announced
     on stderr. *)
  if not is_gpu then
    Stdio.eprintf "lm: %s is not a GPU backend — the per-seed executable check below is vacuous\n"
      backend_name;
  let n_seeds =
    match fine_gemm_seg (segments ~arity_cuts:true opt) with
    | Some seg -> List.length (gpu_seeds seg)
    | None -> 0
  in
  p_all "lm: every fine GPU seed compiles and computes correctly" (List.init n_seeds ~f:Fn.id)
    ~f:(fun idx ->
      (not is_gpu)
      ||
      let comp_s, z_s, r_s = mk_comp (Printf.sprintf "lms%d" idx) in
      let transforms opt =
        let preset seg =
          (* The freed GEMM segment is the only one with GPU seeds; every other segment keeps the
             default preset. *)
          match List.nth (gpu_seeds seg) idx with
          | Some sp -> Autotune.sketch_schedule ~p:sp seg
          | None -> Sched.default_gpu ~min_parallel:1 ~limits seg
        in
        let zero_sched = Sched.zero_expansion ~limits in
        List.map
          (Sched.fission_scheduled ~promote_locals:true ~arity_cuts:true ~preset ~zero_sched
             ~static_indices:[] opt) ~f:(fun (_, _, _, post) -> post)
      in
      match
        let ctx, routine =
          Context.compile ~lowered_transform:transforms (Context.auto ()) comp_s Ir.Indexing.Empty
        in
        let ctx = Context.run ctx routine in
        (Context.get_values ctx z_s.Tensor.value, Context.get_values ctx r_s.Tensor.value)
      with
      | got_z, got_r -> approx got_z z_expected && approx got_r r_expected
      | exception exn ->
          Stdio.eprintf "lm: fine GPU seed %d FAILED %s\n" idx (Exn.to_string exn);
          false);
  (* Tune integration: the search on the lm_head shape (fine candidates seeded on GPU backends)
     crowns a winner that computes the right values, and a second tune replays it through the disk
     cache — exercising the [finer_fission] entry field when a fine candidate won. *)
  clean_cache "autotune_cache_lm_head";
  let tune_once () =
    let reports = ref [] in
    let ctx = Context.auto () in
    let ctx, routine =
      Autotune.tune ~beam_width:2 ~rounds:0 ~repeats:1 ~cache_dir:"autotune_cache_lm_head"
        ~timing_ctx:(Context.auto ())
        ~report:(fun rep -> reports := rep :: !reports)
        ctx comp Ir.Indexing.Empty
    in
    let ctx = Context.run ctx routine in
    ((Context.get_values ctx z.Tensor.value, Context.get_values ctx r.Tensor.value), !reports)
  in
  let (got_z, got_r), reports = tune_once () in
  p "lm: tuned head matches the reference" (approx got_z z_expected && approx got_r r_expected);
  (match reports with
  | [ rep ] ->
      if is_gpu && rep.Autotune.fiss_sketch_candidates = 0 then
        Stdio.eprintf "lm: NOTE: no per-segment sketch candidates were seeded on %s\n" backend_name
  | _ -> ());
  let (got_z2, got_r2), reports2 = tune_once () in
  p "lm: cache-replayed head matches the reference"
    (approx got_z2 z_expected && approx got_r2 r_expected);
  p "lm: second tune was a cache hit or full replay"
    (match reports2 with [ rep ] -> replayed rep || completed rep | _ -> false)
