(* gh-ocannl-1407: the seeding census, the diagnostic that makes sketch-family seeding refusals
   observable ([Autotune.seeding_census], config [autotune_log_seeding]).

   A seed a gate refuses never becomes a candidate, so the decline report never sees it, and
   [schedule_log_declines] reports only render and compile declines: gh-ocannl-503's trigger (the
   staged conv seeds declining on [c_zeroed]) could not be observed without scratch instrumentation.
   Pinned here, on synthetic device limits so the transcript is backend-independent.

   The census's seeds ARE [Autotune.sketch_seed_params]' list, and both families' refusals are their
   trees' refutations ([Autotune.matmul_sketch_tree], and since gh-ocannl-697
   [Autotune.conv_sketch_tree]). A conv leg with no leaf carries a refutation below it, and the
   named gate is the one the site's facts imply: no mma capability; no mma tile for f32 operands; a
   vector file under 8 bytes. The whole-routine lowering's conv output is zeroed in the routine, and
   since gh-ocannl-503 the GPU leg seeds it anyway (the pipeline expands the zero with the site's
   geometry); on the tuner's GPU fission segments the zero folds into a per-cell companion, and the
   GPU leg seeds there too. The GPU leg judges companion coverage, not a statement count
   (gh-ocannl-1406): an aligned-merged conv of three real statements seeds, a channel reduction of
   the conv output is refused naming the coverage verdict. With the key on (the dune action passes
   it), [Autotune.log_seeding] writes exactly the rendered lines, and [Autotune.tune] emits a
   whole-routine census line. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
open Verdict.Claims
module LL = Ir.Low_level
module Sched = Ir.Schedule

let scratch (o : LL.optimized) =
  {
    o with
    LL.traced_store = Hashtbl.copy o.LL.traced_store;
    LL.optimize_ctx = LL.copy_optimize_ctx o.LL.optimize_ctx;
  }

(* An f32 8x8x8 tile, the only one advertised. *)
let mma_limits ?(formats = [ Ir.Backend_intf.(Mma_f32, Mma_f32, Mma_f32) ]) () =
  {
    Ir.Backend_intf.no_hardware_limits with
    mma =
      Some
        {
          Ir.Backend_intf.minimal_mma_capability with
          mma_tile = (8, 8, 8);
          mma_format_tiles = List.map formats ~f:(fun f -> (f, (8, 8, 8)));
        };
  }

let gpu_limits = mma_limits ()

(* A device whose only tile takes f16 operands: an f32 site has no tile. *)
let f16_only_limits = mma_limits ~formats:[ Ir.Backend_intf.(Mma_f16, Mma_f16, Mma_f32) ] ()

let cpu_limits =
  {
    Ir.Backend_intf.no_hardware_limits with
    simd_vector = { Ir.Vector_file.bytes = 16; registers = 16 };
  }

let narrow_cpu_limits =
  { cpu_limits with Ir.Backend_intf.simd_vector = { Ir.Vector_file.bytes = 4; registers = 16 } }

(* The census lines on stderr for reading, outside the golden. *)
let show ~where c = List.iter (Autotune.seeding_census_lines ~where c) ~f:(Stdio.eprintf "%s\n")

let census ~where ~is_gpu ~limits opt =
  let c = Autotune.seeding_census ~is_gpu ~is_cpu:(not is_gpu) ~limits opt in
  show ~where c;
  c

module Sspace = Ir.Schedule_space
module FD = Autotune.Family_decision

let conv_seeding (c : Autotune.seeding_census) =
  match c.sc_family with Autotune.Seeding_conv { site; tree; _ } -> Some (site, tree) | _ -> None

let mentions s ~sub = String.is_substring s ~substring:sub
let on_path d path = List.exists path ~f:(fun (_, d') -> FD.equal d d')
let unfused_leg leg path = on_path (FD.Fusion `Unfused) path && on_path (FD.Conv_leg leg) path

(* A leg's unfused leaves, and the witnesses refuting at or below it. *)
let leg_seeds tree leg =
  List.filter_map (Sspace.enumerate tree) ~f:(fun (path, p) ->
      if unfused_leg leg path then Some p else None)

let leg_refutations tree leg =
  List.filter_map (Sspace.refutations tree) ~f:(fun (path, w) ->
      if unfused_leg leg path then Some w else None)

(* The refutation of the leg itself — its guard, before any flavor is judged. *)
let leg_refusal tree leg =
  List.find_map (Sspace.refutations tree) ~f:(fun (path, w) ->
      match List.last path with
      | Some (_, FD.Conv_leg l) when Poly.equal l leg && on_path (FD.Fusion `Unfused) path -> Some w
      | _ -> None)

let refusal tree leg = Option.value (leg_refusal tree leg) ~default:""

(* The census reports the seeder's own list, and a conv leg that seeds nothing says why: some
   refutation lies at or below it. *)
let claim_consistent label ~is_gpu ~limits opt (c : Autotune.seeding_census) =
  let seeds = Autotune.sketch_seed_params ~is_gpu ~is_cpu:(not is_gpu) ~limits opt in
  p
    (label ^ ": the census's seeds are sketch_seed_params' list")
    (List.equal Poly.equal c.sc_seeds seeds);
  match conv_seeding c with
  | None -> ()
  | Some (_, tree) ->
      p_all (label ^ ": each conv leg without a seed carries a refutation") [ `Cpu; `Gpu ]
        ~f:(fun leg ->
          (not (List.is_empty (leg_seeds tree leg)))
          || not (List.is_empty (leg_refutations tree leg)))

let make_conv tag =
  let x =
    NTDSL.init ~l:(tag ^ "x") ~prec:Ir.Ops.single ~b:[ 2 ] ~o:[ 11; 11; 4 ]
      ~f:(Ll_test.weighted ~weights:[| 1; 1; 2; 3 |] ~modulus:7 ~offset:0. ~stride:1.)
      ()
  in
  let kern =
    NTDSL.init ~l:(tag ^ "k") ~prec:Ir.Ops.single ~i:[ 3; 3; 4 ] ~o:[ 8 ]
      ~f:(Ll_test.weighted ~weights:[| 2; 1; 1; 3 |] ~modulus:5 ~offset:(-2.) ~stride:1.)
      ()
  in
  let%op y =
    x +* "...| oh<+kh, ow<+kw, ..ic..; |kh, kw, ..ic.. -> ..oc.. => ...| oh, ow, ..oc.." kern
  in
  y

(* The routine's whole lowering, captured on its way through a compile. *)
let lowering name t =
  let captured = ref None in
  let ctx = Context.auto () in
  let _ctx, _routine =
    Context.compile ~name
      ~lowered_transform:(fun opt ->
        captured := Some (scratch opt);
        [ opt ])
      ctx (Train.forward t) Ir.Indexing.Empty
  in
  Option.value_exn !captured

let () =
  (* === A conv, whole routine: its output's whole-node Zero_out is in the lowering. === *)
  let opt = lowering "census_conv" (make_conv "cc") in
  let gpu = census ~where:"conv whole-routine gpu" ~is_gpu:true ~limits:gpu_limits opt in
  claim_consistent "conv whole-routine (GPU)" ~is_gpu:true ~limits:gpu_limits opt gpu;
  (match conv_seeding gpu with
  | None -> p "conv whole-routine: the conv family claims the lowering" false
  | Some (site, tree) ->
      p "conv whole-routine: the conv family claims the lowering" true;
      p "conv whole-routine: the site's output is zeroed in the lowering" site.c_zeroed;
      p "conv whole-routine: the GPU leg seeds the zeroed site"
        (Option.is_none (leg_refusal tree `Gpu) && not (List.is_empty (leg_seeds tree `Gpu)));
      p "conv whole-routine: the CPU leg is refused as not a CPU target"
        (String.equal (refusal tree `Cpu) "not a CPU target"));
  p_exists "conv whole-routine: the conv nest is listed, marked zeroed" gpu.sc_conv_nests
    ~f:(fun (_, zeroed) -> zeroed);
  let no_mma =
    census ~where:"conv whole-routine no-mma" ~is_gpu:true
      ~limits:Ir.Backend_intf.no_hardware_limits opt
  in
  claim_consistent "conv whole-routine (no mma)" ~is_gpu:true
    ~limits:Ir.Backend_intf.no_hardware_limits opt no_mma;
  p "no mma capability: the GPU leg names it"
    (Option.exists (conv_seeding no_mma) ~f:(fun (_, tree) ->
         String.equal (refusal tree `Gpu) "the backend advertises no mma capability"));
  let f16_only =
    census ~where:"conv whole-routine f16-only" ~is_gpu:true ~limits:f16_only_limits opt
  in
  claim_consistent "conv whole-routine (f16-only tile)" ~is_gpu:true ~limits:f16_only_limits opt
    f16_only;
  p "no f32 tile: the GPU leg names the missing tile"
    (Option.exists (conv_seeding f16_only) ~f:(fun (_, tree) ->
         mentions (refusal tree `Gpu) ~sub:"advertises no mma tile for operands"));
  let cpu = census ~where:"conv whole-routine cpu" ~is_gpu:false ~limits:cpu_limits opt in
  claim_consistent "conv whole-routine (CPU)" ~is_gpu:false ~limits:cpu_limits opt cpu;
  p "conv whole-routine: the CPU leg seeds on a 16-byte vector file"
    (Option.exists (conv_seeding cpu) ~f:(fun (_, tree) ->
         not (List.is_empty (leg_seeds tree `Cpu))));
  (* The fused flavor's verdict is the root's own, judged whatever the unfused flavor seeds: the
     recognizer's reason, verbatim. *)
  p
    "conv whole-routine: a site without a fusable tail refutes the fused flavor with the \
     recognizer's reason"
    (Option.exists (conv_seeding cpu) ~f:(fun (site, tree) ->
         match Sched.fuse_epilogue_witness ~target:site.c_d opt with
         | None -> false
         | Some why ->
             List.exists (Sspace.refutations tree) ~f:(fun (path, w) ->
                 String.equal w why
                 && List.equal Poly.equal (List.map path ~f:snd) [ FD.Fusion `Fused ])));
  p "conv whole-routine: the zeroed site's row-block flavors are refuted on its zero"
    (Option.exists (conv_seeding cpu) ~f:(fun (_, tree) ->
         List.exists (Sspace.refutations tree) ~f:(fun (path, w) ->
             on_path (FD.Conv_shape (`Row_block 8)) path && mentions w ~sub:"c_zeroed")));
  let narrow =
    census ~where:"conv whole-routine narrow-cpu" ~is_gpu:false ~limits:narrow_cpu_limits opt
  in
  claim_consistent "conv whole-routine (4-byte vectors)" ~is_gpu:false ~limits:narrow_cpu_limits opt
    narrow;
  p "a 4-byte vector file: the CPU leg names it"
    (Option.exists (conv_seeding narrow) ~f:(fun (_, tree) ->
         mentions (refusal tree `Cpu) ~sub:"4-byte vector file is under 8 bytes"));
  (* === The tuner's GPU fission segments (its [enum_fiss_entries] settings): the conv output's zero
     folds into a per-cell companion of the conv nest, so the site is no longer zeroed. === *)
  let segments =
    Sched.fission_segmented ~promote_locals:true ~fold_zeros:true
      ?keep_mapping:(Sched.fission_keep_mapping ~is_gpu:true ~limits:gpu_limits)
      ~preset:(fun seg -> Sched.default_gpu ~min_parallel:1 ~limits:gpu_limits seg)
      ~zero_sched:(Sched.zero_expansion ~limits:gpu_limits)
      ~static_indices:[] (scratch opt)
    |> snd
    |> List.map ~f:(fun (_, pre, _, _) -> pre)
  in
  let conv_segments =
    List.filter_map segments ~f:(fun seg ->
        let c = census ~where:"conv segment gpu" ~is_gpu:true ~limits:gpu_limits seg in
        claim_consistent "conv segment (GPU)" ~is_gpu:true ~limits:gpu_limits seg c;
        conv_seeding c)
  in
  p_exists "conv segment: an unzeroed site the GPU leg seeds" conv_segments ~f:(fun (site, tree) ->
      (not site.c_zeroed) && not (List.is_empty (leg_seeds tree `Gpu)));
  p_exists "conv segment: row blocks too large for the 9-row site are refuted at their geometry"
    conv_segments ~f:(fun (_, tree) ->
      List.exists (Sspace.refutations tree) ~f:(fun (path, w) ->
          on_path (FD.Conv_rows 16) path && mentions w ~sub:"fewer than two blocks"));
  (* A workgroup-memory cap between the 8-row block's charge (768 bytes: its two staged tiles, and,
     the flavor being masked, its workgroup-shared accumulator fragment, gh-ocannl-1452) and the
     whole padded row's (1280): the leg still seeds, and notes the flavor the cap refused. *)
  let tight_limits =
    {
      Ir.Backend_intf.no_hardware_limits with
      mma = gpu_limits.Ir.Backend_intf.mma;
      max_workgroup_memory_bytes = Some 1000;
    }
  in
  let tight =
    List.filter_map segments ~f:(fun seg ->
        let c = census ~where:"conv segment gpu 1000B" ~is_gpu:true ~limits:tight_limits seg in
        claim_consistent "conv segment (1000-byte workgroup memory)" ~is_gpu:true
          ~limits:tight_limits seg c;
        conv_seeding c)
  in
  p_exists "a partial workgroup-memory refusal is refuted at its geometry beside the seeds" tight
    ~f:(fun (_, tree) ->
      (not (List.is_empty (leg_seeds tree `Gpu)))
      && List.exists (Sspace.refutations tree) ~f:(fun (path, w) ->
          on_path (FD.Conv_rows 0) path && mentions w ~sub:"exceeding the 1000-byte limit"));
  (* === gh-ocannl-1406: the GPU leg judges companion coverage, not a statement count. A conv with
     two materialized elementwise companions (the convnet's conv+relu+scale shape) is an
     aligned-merged segment of three real statements, which a [real_stmts <= 2] cap refused although
     the pipeline gives every companion the site's geometry; a channel reduction of the conv output
     cannot follow the site's full chain, and the leg names the coverage verdict. === *)
  let gpu_segments opt =
    Sched.fission_segmented ~promote_locals:true ~fold_zeros:true
      ?keep_mapping:(Sched.fission_keep_mapping ~is_gpu:true ~limits:gpu_limits)
      ~preset:(fun seg -> Sched.default_gpu ~min_parallel:1 ~limits:gpu_limits seg)
      ~zero_sched:(Sched.zero_expansion ~limits:gpu_limits)
      ~static_indices:[] (scratch opt)
    |> snd
    |> List.map ~f:(fun (_, pre, _, _) -> pre)
  in
  let conv_lowerings ~where opt =
    List.filter_mapi (opt :: gpu_segments opt) ~f:(fun i seg ->
        let where = where ^ if i = 0 then " whole-routine" else " segment" in
        let c = census ~where ~is_gpu:true ~limits:gpu_limits seg in
        claim_consistent where ~is_gpu:true ~limits:gpu_limits seg c;
        match c.sc_family with
        | Autotune.Seeding_conv { real_stmts; tree; _ } -> Some (real_stmts, tree)
        | _ -> None)
  in
  let merged tag =
    let pr = make_conv tag in
    Ir.Tnode.update_memory_mode pr.Tensor.value Ir.Tnode.On_device (Site "99:test-setup");
    let%op y2 = relu pr in
    Ir.Tnode.update_memory_mode y2.Tensor.value Ir.Tnode.On_device (Site "99:test-setup");
    let%op y3 = y2 *. 0.5 in
    y3
  in
  let merged_js =
    conv_lowerings ~where:"conv merged gpu" (lowering "census_merged" (merged "cm"))
  in
  p_exists "conv merged: a lowering of three or more real statements the GPU leg seeds" merged_js
    ~f:(fun (n_real, tree) -> n_real >= 3 && not (List.is_empty (leg_seeds tree `Gpu)));
  p_none "conv merged: no GPU leg refuses on companion coverage" merged_js ~f:(fun (_, tree) ->
      mentions (refusal tree `Gpu) ~sub:"companion coverage");
  let channel_sum tag =
    let pr = make_conv tag in
    Ir.Tnode.update_memory_mode pr.Tensor.value Ir.Tnode.On_device (Site "99:test-setup");
    let%op s = pr ++ "...|oh, ow, oc => ...|oh, ow" in
    s
  in
  let sum_js =
    conv_lowerings ~where:"conv channel-sum gpu" (lowering "census_chsum" (channel_sum "cs"))
  in
  p_exists "conv channel-sum: the GPU leg's refusal names companion coverage" sum_js
    ~f:(fun (_, tree) ->
      List.is_empty (leg_seeds tree `Gpu)
      && mentions (refusal tree `Gpu) ~sub:"GPU conv companion coverage (gh-521)");
  (* === A matmul: the census is the family tree's. === *)
  let a =
    TDSL.ndarray (Array.init 64 ~f:Float.of_int) ~label:[ "ma" ] ~input_dims:[ 8 ]
      ~output_dims:[ 8 ] ()
  in
  let b =
    TDSL.ndarray (Array.init 64 ~f:Float.of_int) ~label:[ "mb" ] ~input_dims:[ 8 ]
      ~output_dims:[ 8 ] ()
  in
  let%op mm = a * b in
  let mopt = lowering "census_matmul" mm in
  let mc = census ~where:"matmul whole-routine gpu" ~is_gpu:true ~limits:gpu_limits mopt in
  claim_consistent "matmul (GPU)" ~is_gpu:true ~limits:gpu_limits mopt mc;
  (match
     (mc.sc_family, Autotune.matmul_sketch_tree ~is_gpu:true ~is_cpu:false ~limits:gpu_limits mopt)
   with
  | Autotune.Seeding_matmul { tree; _ }, Some reference ->
      p "matmul: the census's refutations are the family tree's"
        (List.length (Ir.Schedule_space.refutations tree)
        = List.length (Ir.Schedule_space.refutations reference))
  | _ -> p "matmul: the census's refutations are the family tree's" false);
  let mcc =
    Autotune.seeding_census ~coalesced:true ~is_gpu:true ~is_cpu:false ~limits:gpu_limits mopt
  in
  show ~where:"matmul whole-routine gpu, coalesced" mcc;
  (match mcc.sc_coalesced with
  | None -> p "matmul: the coalesced census is the coalesced branch's seeds" false
  | Some verdicts ->
      p "matmul: the coalesced census is the coalesced branch's seeds"
        (List.equal Poly.equal
           (List.concat_map verdicts ~f:(fun v -> v.Autotune.cv_seeds))
           (Autotune.coalesced_seed_params ~is_gpu:true ~is_cpu:false ~limits:gpu_limits mopt));
      p_all "matmul: each coalesced layout names a refusal exactly when it seeds nothing" verdicts
        ~f:(fun v ->
          Bool.equal (Option.is_some v.Autotune.cv_refusal) (List.is_empty v.Autotune.cv_seeds)));
  p "matmul: the head line names the family and its refutation count"
    (match Autotune.seeding_census_lines ~where:"m" mc with
    | head :: _ -> mentions head ~sub:"family=matmul" && mentions head ~sub:" refuted="
    | [] -> false);
  (* === The key: log_seeding writes the rendered lines; tune emits its whole-routine line. === *)
  p "autotune_log_seeding is on for this run" (Autotune.log_seeding_enabled ());
  let capture f = snd (Stderr_capture.lines f) in
  let logged =
    capture (fun () ->
        Autotune.log_seeding ~where:"probe" ~is_gpu:true ~is_cpu:false ~limits:gpu_limits opt)
  in
  p "log_seeding writes exactly the rendered census"
    (List.equal String.equal logged
       (Autotune.seeding_census_lines ~where:"probe"
          (Autotune.seeding_census ~is_gpu:true ~is_cpu:false ~limits:gpu_limits opt)));
  (* The untuned seeded default (a CUDA/HIP pipeline, derived here for "cuda" on the synthetic
     limits): one census line per segment of the plan it chooses. *)
  let plan = ref None in
  let seeded =
    capture (fun () ->
        plan :=
          Autotune.seeded_plan ~min_reduction:1 ~backend:"cuda" ~limits:gpu_limits ~accum_prec:Fn.id
            ~static_indices:[] mopt)
  in
  let seeded_lines =
    List.filter seeded ~f:(fun l ->
        String.is_prefix l ~prefix:Autotune.seeding_census_prefix
        && mentions l ~sub:"where=\"seeded_default segment u")
  in
  p_exists "the seeded default prints a census line for its plan" seeded_lines ~f:(fun l ->
      mentions l ~sub:" seeded: " || mentions l ~sub:" refused: ");
  (match !plan with
  | Some { Autotune.sp_segments = Some (_, segments); _ } ->
      p "the seeded default prints one census line per segment of its plan"
        (List.length seeded_lines = List.length segments)
  | _ -> ());
  let va = TDSL.ndarray [| 1.; 2.; 3.; 4. |] ~label:[ "pa" ] ~output_dims:[ 4 ] () in
  let vb = TDSL.ndarray [| 4.; 3.; 2.; 1. |] ~label:[ "pb" ] ~output_dims:[ 4 ] () in
  let%op sum = va + vb in
  let tuned =
    capture (fun () ->
        ignore
          (Autotune.tune ~name:"census_tune" ~beam_width:1 ~rounds:0 ~repeats:1 ~cache_dir:""
             (Context.auto ()) (Train.forward sum) Ir.Indexing.Empty))
  in
  p_exists "tune emits a whole-routine census line" tuned ~f:(fun l ->
      String.is_prefix l ~prefix:Autotune.seeding_census_prefix
      && mentions l ~sub:"where=\"tune whole-routine\"")
