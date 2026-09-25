(* Batched matmul sketch seeding and tensorization (gh-ocannl-528).

   Three lowered shapes drive the checks:

   - Leading batch: [out[b,s,j] += x[b,s,k] * w[k,j]] — the ffn/logits shape of a transformer
   forward. Detection assigns [i = s] (the deepest loop read by A and absent from B) and records [b]
   as an outer batch loop; the sketch pipelines leave it Serial. - Interior batch: [out[b,i,h,j] +=
   att[b,i,h,k] * v[b,k,h,j]] — attention's scores-times-values, with the head axis BETWEEN the tile
   roles in both the output layout and the loop nest. The pipelines hoist [h] above [i] with Swaps,
   and [Tensorize] records leading-dimension strides larger than the minor dims
   ([Tile_mma.ldd]/[lda]/[ldb]) — a wrong stride reads/writes the wrong cells, so the bitwise parity
   against the serial twin is what pins the layout. - A variance-style self-product [v[b,s] +=
   x[b,s,k] * x[b,s,k]] — the layer-norm shape whose reads mention every loop. It used to be
   mis-detected as a matmul site (its seeds were the only mma candidates gpt2_mini ever proposed,
   all failing at candidate compile); the role exclusions ([j] absent from A, [i] absent from B)
   must reject it, so it seeds nothing.

   The execution legs run the CPU tensorized sketch end-to-end through the public seeding API
   ([Autotune.sketch_seed_params] with synthetic limits, [Autotune.sketch_schedule]) on the C
   backends, where the register-tiled [Tile_mma] rendering promises bitwise parity for fused-f32
   accumulations. GPU backends keep the seeding checks (pure functions of the lowering) and skip the
   CPU-pipeline execution legs. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module Tn = Ir.Tnode
module LL = Ir.Low_level
module Sched = Ir.Schedule
module Asgns = Ir.Assignments

let () = Utils.settings.output_debug_files_in_build_directory <- true

open Verdict.Claims

(* Zeros compare equal to zeros. A fragment mapping that reads outside the staged block, a kernel
   that never ran, or a reference whose own setup silently collapsed all yield all-zeros, and a
   parity check between two zero arrays passes while covering nothing (gh-ocannl-481 item 3). Every
   reference array is pinned nonzero where it is produced, so the parity claims below have
   content. *)
let nonzero name (a : float array) =
  if not (Array.exists a ~f:(fun x -> Float.(x <> 0.))) then
    failwith (name ^ ": the reference is all zeros — the parity checks against it are vacuous");
  a

let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")
let skipped = Verdict.skipped ~backend:backend_name
let on_gpu = Sched.backend_is_gpu backend_name

module Generated = Test_utils.Generated

let () = Generated.init ~backend_name

let named name (comp : Asgns.comp) : Asgns.comp =
  { comp with asgns = Asgns.Block_comment (name, comp.asgns) }

(* Synthetic limits keep the seeding checks machine-independent: a 32-byte vector file for the CPU
   register tiling, and an f32-capable mma capability for the GPU tensorized seeds. *)
let cpu_limits = { Ir.Backend_intf.no_hardware_limits with simd_vector_bytes = 32 }

let gpu_limits =
  {
    Ir.Backend_intf.no_hardware_limits with
    mma =
      Some
        {
          Ir.Backend_intf.mma_simd_width = 32;
          mma_tile = (8, 8, 8);
          mma_format_tiles =
            [
              ( (Ir.Backend_intf.Mma_f32, Ir.Backend_intf.Mma_f32, Ir.Backend_intf.Mma_f32),
                (8, 8, 8) );
            ];
          mma_f16_wide_acc_scopes = [];
          mma_bf16_wide_acc_scopes = [];
          mma_staged_layouts = [];
          mma_pipeline_depths = [];
        };
  }

let cpu_mma_seeds opt =
  Autotune.sketch_seed_params ~is_gpu:false ~is_cpu:true ~limits:cpu_limits opt
  |> List.filter ~f:(fun q -> q.Autotune.sk_mma)

let gpu_mma_seeds opt =
  Autotune.sketch_seed_params ~is_gpu:true ~is_cpu:false ~limits:gpu_limits opt
  |> List.filter ~f:(fun q -> q.Autotune.sk_mma)

let compile_serial ~name tensor =
  let ctx, routine =
    Context.compile
      ~lowered_transform:(fun opt -> [ opt ])
      (Context.auto ())
      (named name (Train.forward tensor))
      Ir.Indexing.Empty
  in
  let ctx = Context.run ctx routine in
  nonzero name (Context.get_values ctx tensor.Tensor.value)

(* Compile [tensor] through [transform], capturing the lowering for the seeding assertions. *)
let with_lowering ~name tensor ~(transform : LL.optimized -> LL.optimized) =
  let captured = ref None in
  let ctx, routine =
    Context.compile
      ~lowered_transform:(fun opt ->
        captured := Some opt;
        [ transform opt ])
      (Context.auto ())
      (named name (Train.forward tensor))
      Ir.Indexing.Empty
  in
  (Option.value_exn !captured, ctx, routine)

(* The whole-triple CPU tensorized sketch selected from the public seeding API, executed against the
   serial twin. On GPU backends only the lowering capture runs (identity transform; the routine is
   compiled but not executed). *)
let check_leg ~tag ~serial ~tensorized =
  let want = compile_serial ~name:(tag ^ "_serial") serial in
  if on_gpu then (
    let opt, _ctx, _routine = with_lowering ~name:(tag ^ "_mma") tensorized ~transform:Fn.id in
    p (tag ^ ": cpu mma seeds present") (not (List.is_empty (cpu_mma_seeds opt)));
    skipped (tag ^ " tensorized matches the serial twin bitwise");
    skipped (tag ^ " tensorized structure as expected");
    opt)
  else
    let seed = ref None in
    let transform opt =
      let q =
        List.find_exn (cpu_mma_seeds opt) ~f:(fun q -> q.Autotune.sk_bm = 0 && q.Autotune.sk_bk = 0)
      in
      seed := Some q;
      Sched.apply (Autotune.sketch_schedule ~p:q opt) opt
    in
    let opt, ctx, routine = with_lowering ~name:(tag ^ "_mma") tensorized ~transform in
    p (tag ^ ": cpu mma seeds present") (Option.is_some !seed);
    let ctx = Context.run ctx routine in
    let got = Context.get_values ctx tensorized.Tensor.value in
    p_all2 (tag ^ " tensorized matches the serial twin bitwise") got want ~f:Float.equal;
    Generated.assert_emits ~routine:(tag ^ "_mma") ~contains:"Tile_mma register tiling"
      (tag ^ " tensorized structure as expected");
    opt

let () =
  let bt = 2 and ss = 16 and kk = 8 and jj = 32 in
  (* --- Leading batch: out[b,s,j] += x[b,s,k] * w[k,j] --- *)
  let xv =
    NTDSL.init ~l:"xv" ~prec:Ir.Ops.single ~o:[ bt; ss; kk ]
      ~f:(Ll_test.cycle ~dims:[| bt; ss; kk |] ~modulus:13 ~offset:0. ~stride:0.25)
      ()
  in
  let wv =
    NTDSL.init ~l:"wv" ~prec:Ir.Ops.single ~o:[ kk; jj ]
      ~f:(Ll_test.cycle ~dims:[| kk; jj |] ~modulus:17 ~offset:(-8.) ~stride:0.5)
      ()
  in
  let%op lb0 = xv +* "bsk;kj=>bsj" wv in
  let%op lb1 = xv +* "bsk;kj=>bsj" wv in
  let opt_lb = check_leg ~tag:"batched_lb" ~serial:lb0 ~tensorized:lb1 in
  p "leading-batch: gpu mma seeds present" (not (List.is_empty (gpu_mma_seeds opt_lb)));

  (* --- Interior batch: out[b,i,h,j] += att[b,i,h,k] * v[b,k,h,j] --- *)
  let hh = 2 in
  let att =
    NTDSL.init ~l:"att" ~prec:Ir.Ops.single ~o:[ bt; ss; hh; kk ]
      ~f:(Ll_test.cycle ~dims:[| bt; ss; hh; kk |] ~modulus:11 ~offset:0. ~stride:0.125)
      ()
  in
  let vv =
    NTDSL.init ~l:"vv" ~prec:Ir.Ops.single ~o:[ bt; kk; hh; jj ]
      ~f:(Ll_test.cycle ~dims:[| bt; kk; hh; jj |] ~modulus:7 ~offset:(-3.) ~stride:0.5)
      ()
  in
  let%op ib0 = att +* "bihk;bkhj=>bihj" vv in
  let%op ib1 = att +* "bihk;bkhj=>bihj" vv in
  let opt_ib = check_leg ~tag:"batched_ib" ~serial:ib0 ~tensorized:ib1 in
  let gpu_seeds_ib = gpu_mma_seeds opt_ib in
  p "interior-batch: gpu mma seeds present" (not (List.is_empty gpu_seeds_ib));
  p "interior-batch: gpu staged sketch builds"
    (match
       List.find gpu_seeds_ib ~f:(fun q -> q.Autotune.sk_bk > 0)
       |> Option.map ~f:(fun q -> Autotune.sketch_schedule ~p:q opt_ib)
     with
    | Some (_ :: _) -> true
    | Some [] | None -> false
    | exception _ -> false);

  (* --- Variance-style self-product: must not be detected as a matmul site --- *)
  let x2 =
    NTDSL.init ~l:"x2" ~prec:Ir.Ops.single ~o:[ bt; ss; kk ]
      ~f:(Ll_test.cycle ~dims:[| bt; ss; kk |] ~modulus:5 ~offset:0. ~stride:0.5)
      ()
  in
  let%op var = x2 +* "bsk;bsk=>bs" x2 in
  let opt_var, _ctx, _routine = with_lowering ~name:"batched_var" var ~transform:Fn.id in
  p "variance-like site: no cpu mma seeds" (List.is_empty (cpu_mma_seeds opt_var));
  p "variance-like site: no gpu mma seeds" (List.is_empty (gpu_mma_seeds opt_var));

  (* --- The tensor-core legs, at bf16, against the backend's REAL advertised capability ---

     Everything above runs the batched sites through synthetic limits, which keeps the seeding
     checks machine-independent but leaves the question this file exists for unanswered on a GPU:
     the recorded leading-dimension strides ([Tile_mma.ldd]/[lda]/[ldb]) are consumed by the
     backends' own mma hooks, and a stride that is right for the register-tiled C rendering can
     still be wrong for a fragment load. On a real device an f32 site seeds nothing on the wmma
     backends (RDNA3.5 and CUDA have no f32 operand shape), so the reachable format is a narrow one;
     bf16 is the one both wmma backends and Metal have in the uniform combination.

     So: seed against [Context.hardware_limits], apply the real pipeline, execute, and require both
     that the values match the serial twin and that the emitted source carries the backend's
     intrinsic. The interior-batch leg is the load-bearing one — its [lda]/[ldb]/[ldd] are
     [hh]-times larger than the minor dims, which is exactly the case gh-ocannl-528 introduced and
     the case a fragment load addresses differently from a scalar loop.

     Tolerance rather than bitwise on HIP, for the reason [schedule_mma_matmul] documents at length:
     gfx1151's WMMA does not return the exactly-rounded dot product in any format combination,
     reproducibly so outside OCANNL. It is loose enough to admit that (worst measured 5.86e-03 in
     the uniform-bf16 combination) and far tighter than a stride defect, which moves values by
     O(1). *)
  let real_limits = Context.hardware_limits (Context.auto ()) in
  let has_uniform_bf16_tile =
    Ir.Backend_intf.advertises_mma_format real_limits ~a:Ir.Backend_intf.Mma_bf16
      ~b:Ir.Backend_intf.Mma_bf16 ~d:Ir.Backend_intf.Mma_bf16
  in
  let close a b = Float.(abs (a - b) <= 0.05 * max 1. (abs b)) in
  (* At most this many candidates are executed per site: each one is a full backend compile, and on
     HIP the rocWMMA headers make that expensive. The budget used to go to the first
     [max_candidates] seeds, which stopped covering the site once gh-ocannl-643 added the batch-grid
     twins: the "batch" family-tree level orders batch-serial first, so every twin enumerates AFTER
     all serial-batch geometries and a first-N prefix of a batched site's seeds is uniformly
     [sk_batch_grid = false] — no twin ever reached a real device from here. Sample both flavors
     instead, in enumeration order within each, and spend any leftover budget on whichever flavor
     still has seeds (a site that seeds only one flavor keeps its old coverage). Reported on stderr,
     split by flavor, so neither the bound nor the sample is silent. *)
  let max_candidates = 4 in
  let per_flavor = 2 in
  let sample_candidates seeds =
    let serial, grid = List.partition_tf seeds ~f:(fun q -> not q.Autotune.sk_batch_grid) in
    let head = List.take serial per_flavor @ List.take grid per_flavor in
    let rest = List.drop serial per_flavor @ List.drop grid per_flavor in
    head @ List.take rest (max_candidates - List.length head)
  in
  let bf16_leg ~tag ~build =
    let ref_t = build () in
    let want =
      let ctx, routine =
        Context.compile
          ~lowered_transform:(fun opt -> [ opt ])
          (Context.auto ())
          (named (tag ^ "_bf16_serial") (Train.forward ref_t))
          Ir.Indexing.Empty
      in
      nonzero (tag ^ "_bf16_serial")
        (Context.get_values (Context.run ctx routine) ref_t.Tensor.value)
    in
    let cand = build () in
    let fwd = named (tag ^ "_bf16_mma") (Train.forward cand) in
    let captured = ref None in
    let _ctx, _r =
      Context.compile
        ~lowered_transform:(fun opt ->
          captured := Some opt;
          [ opt ])
        (Context.auto ()) fwd Ir.Indexing.Empty
    in
    let opt = Option.value_exn ~here:[%here] !captured in
    let seeds =
      Autotune.sketch_seed_params ~is_gpu:true ~is_cpu:false ~limits:real_limits opt
      |> List.filter ~f:(fun q -> q.Autotune.sk_mma)
    in
    let cands = sample_candidates seeds in
    let flavors qs = List.count qs ~f:(fun q -> not q.Autotune.sk_batch_grid) in
    if not has_uniform_bf16_tile then
      Stdio.eprintf
        "%s: %s advertises no uniform-bf16 mma tile — the tensor-core checks below are vacuous\n"
        tag backend_name
    else
      Stdio.eprintf
        "%s: executing %d of %d bf16 mma seeds (compile cost): %d of %d batch-serial, %d of %d \
         batch-grid\n"
        tag (List.length cands) (List.length seeds) (flavors cands) (flavors seeds)
        (List.length cands - flavors cands)
        (List.length seeds - flavors seeds);
    (* Counted per flavor, not in aggregate: with a mixed sample every aggregate counter reads the
       same whether both flavors were covered or one flavor was covered twice, so a batch-serial
       candidate that fails to compile, or a batch-grid candidate that runs [Tile_mma]'s scalar
       fallback while its serial neighbour renders the intrinsic, would leave the claims green over
       exactly the coverage this sampling exists to provide. *)
    let sampled = List.count cands ~f:(fun q -> q.Autotune.sk_batch_grid) in
    let n_ran = (ref 0, ref 0) and n_close = (ref 0, ref 0) and n_intrinsic = (ref 0, ref 0) in
    let of_flavor (serial, grid) q = if q.Autotune.sk_batch_grid then grid else serial in
    List.iter cands ~f:(fun q ->
        (* Every candidate compiles under the one routine name [<tag>_bf16_mma], so each overwrites
           the previous one's artifact. Arming deletes it first, and the read below then sees this
           candidate's kernel or nothing (gh-ocannl-655) -- which is what the per-flavor intrinsic
           count above rests on: a batch-grid candidate running the scalar fallback is only caught
           if the source judging it is really its own. *)
        Generated.arm (tag ^ "_bf16_mma");
        match
          let ctx, routine =
            Context.compile
              ~lowered_transform:(fun o -> [ Sched.apply (Autotune.sketch_schedule ~p:q o) o ])
              (Context.auto ()) fwd Ir.Indexing.Empty
          in
          (Context.get_values (Context.run ctx routine) cand.Tensor.value, routine.mma)
        with
        | got, mma ->
            Int.incr (of_flavor n_ran q);
            if Array.for_all2_exn got want ~f:close then Int.incr (of_flavor n_close q);
            if Ir.C_syntax.equal_tensorization mma.Ir.C_syntax.tensorization Ir.C_syntax.Tensorized
            then Int.incr (of_flavor n_intrinsic q)
        | exception _ -> ());
    p
      (tag ^ " bf16: the backend's advertised tile is seeded")
      ((not has_uniform_bf16_tile) || not (List.is_empty seeds));
    (* One set of claims per flavor. A flavor the site does not seed contributes no sampled
       candidate and its claims stand vacuously, which is what keeps the golden backend-uniform; a
       flavor that IS sampled has to compile, run, agree with the serial twin, and reach the
       backend's mma hook on its own. *)
    let flavor_claims ~label ~grid =
      let n_sampled = if grid then sampled else List.length cands - sampled in
      let count refs = !(if grid then snd refs else fst refs) in
      if has_uniform_bf16_tile then
        Stdio.eprintf "%s: %d of %d %s candidates ran, %d matched, %d rendered the intrinsic\n" tag
          (count n_ran) n_sampled label (count n_close) (count n_intrinsic);
      p
        (tag ^ " bf16: a sampled " ^ label ^ " candidate compiles and runs")
        ((not has_uniform_bf16_tile) || n_sampled = 0 || count n_ran >= 1);
      p
        (tag ^ " bf16: every running " ^ label ^ " candidate matches the serial twin")
        (count n_ran = count n_close);
      p
        (tag ^ " bf16: a running " ^ label ^ " candidate renders the tensor-core intrinsic")
        ((not has_uniform_bf16_tile) || n_sampled = 0 || count n_intrinsic >= 1)
    in
    flavor_claims ~label:"batch-serial" ~grid:false;
    flavor_claims ~label:"batch-grid" ~grid:true
  in
  let bf16_init ~l ~o ~f = NTDSL.init ~l ~prec:Ir.Ops.bfloat16 ~o ~f () in
  (* Products are multiples of 1/8 and every partial sum is bounded by 16, so the serial twin is
     exact and any deviation the tolerance admits is the tensor core's own. *)
  let ss2 = 32 and kk2 = 32 and jj2 = 32 in
  let xb =
    bf16_init ~l:"xb" ~o:[ bt; ss2; kk2 ]
      ~f:(Ll_test.cycle ~dims:[| bt; ss2; kk2 |] ~modulus:3 ~offset:0. ~stride:0.25)
  in
  let wb =
    bf16_init ~l:"wb" ~o:[ kk2; jj2 ]
      ~f:(Ll_test.cycle ~dims:[| kk2; jj2 |] ~modulus:5 ~offset:(-2.) ~stride:0.5)
  in
  bf16_leg ~tag:"batched_lb" ~build:(fun () ->
      let%op t = xb +* "bsk;kj=>bsj" wb in
      Tn.update_prec t.Tensor.value Ir.Ops.bfloat16;
      t);
  (* The interior-batch shape: [h] sits between the tile roles, so [lda]/[ldb]/[ldd] are all
     [hh]-times the minor dim (64 here, against minor dims of 32). *)
  let attb =
    bf16_init ~l:"attb" ~o:[ bt; ss2; hh; kk2 ]
      ~f:(Ll_test.cycle ~dims:[| bt; ss2; hh; kk2 |] ~modulus:3 ~offset:0. ~stride:0.25)
  in
  let vb =
    bf16_init ~l:"vb" ~o:[ bt; kk2; hh; jj2 ]
      ~f:(Ll_test.cycle ~dims:[| bt; kk2; hh; jj2 |] ~modulus:5 ~offset:(-2.) ~stride:0.5)
  in
  bf16_leg ~tag:"batched_ib" ~build:(fun () ->
      let%op t = attb +* "bihk;bkhj=>bihj" vb in
      Tn.update_prec t.Tensor.value Ir.Ops.bfloat16;
      t)
