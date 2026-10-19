(* gh-ocannl-1407: the seeding census, the diagnostic that makes sketch-family seeding refusals
   observable ([Autotune.seeding_census], config [autotune_log_seeding]).

   A seed a gate refuses never becomes a candidate, so the decline report never sees it, and
   [schedule_log_declines] reports only render and compile declines: gh-ocannl-503's trigger (the
   staged conv seeds declining on [c_zeroed]) could not be observed without scratch instrumentation.
   Pinned here, on synthetic device limits so the transcript is backend-independent.

   The census's seeds ARE [Autotune.sketch_seed_params]' list, and the matmul family's refutations
   are its tree's ([Autotune.matmul_sketch_tree]). Each conv leg names a refusal exactly when it
   seeds nothing, and the named gate is the one the site's facts imply: [c_zeroed] on the
   whole-routine lowering, whose conv output is zeroed in the routine; no mma capability; no mma
   tile for f32 operands; a vector file under 8 bytes. On the tuner's GPU fission segments the zero
   folds into a per-cell companion, and the GPU leg seeds. With the key on (the dune action passes
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

let conv_judgment (c : Autotune.seeding_census) =
  match c.sc_family with Autotune.Seeding_conv { judgment; _ } -> Some judgment | _ -> None

let mentions s ~sub = String.is_substring s ~substring:sub
let refusal (l : Autotune.conv_leg) = Option.value l.leg_refusal ~default:""

(* The census reports the seeder's own list, and a leg refuses exactly when it seeds nothing. *)
let claim_consistent label ~is_gpu ~limits opt (c : Autotune.seeding_census) =
  let seeds = Autotune.sketch_seed_params ~is_gpu ~is_cpu:(not is_gpu) ~limits opt in
  p
    (label ^ ": the census's seeds are sketch_seed_params' list")
    (List.equal Poly.equal c.sc_seeds seeds);
  match conv_judgment c with
  | None -> ()
  | Some j ->
      p_all (label ^ ": each conv leg names a refusal exactly when it seeds nothing")
        [ j.cj_cpu; j.cj_gpu ] ~f:(fun (l : Autotune.conv_leg) ->
          Bool.equal (Option.is_some l.leg_refusal) (List.is_empty l.leg_seeds))

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
  (match conv_judgment gpu with
  | None -> p "conv whole-routine: the conv family claims the lowering" false
  | Some j ->
      p "conv whole-routine: the conv family claims the lowering" true;
      p "conv whole-routine: the site's output is zeroed in the lowering" j.cj_site.c_zeroed;
      p "conv whole-routine: the GPU leg's refusal names c_zeroed"
        (mentions (refusal j.cj_gpu) ~sub:"(c_zeroed)");
      p "conv whole-routine: the CPU leg is refused as not a CPU target"
        (String.equal (refusal j.cj_cpu) "not a CPU target"));
  p_exists "conv whole-routine: the conv nest is listed, marked zeroed" gpu.sc_conv_nests
    ~f:(fun (_, zeroed) -> zeroed);
  let no_mma =
    census ~where:"conv whole-routine no-mma" ~is_gpu:true
      ~limits:Ir.Backend_intf.no_hardware_limits opt
  in
  claim_consistent "conv whole-routine (no mma)" ~is_gpu:true
    ~limits:Ir.Backend_intf.no_hardware_limits opt no_mma;
  p "no mma capability: the GPU leg names it"
    (Option.exists (conv_judgment no_mma) ~f:(fun j ->
         String.equal (refusal j.cj_gpu) "the backend advertises no mma capability"));
  let f16_only =
    census ~where:"conv whole-routine f16-only" ~is_gpu:true ~limits:f16_only_limits opt
  in
  claim_consistent "conv whole-routine (f16-only tile)" ~is_gpu:true ~limits:f16_only_limits opt
    f16_only;
  p "no f32 tile: the GPU leg names the missing tile, before the site's own gates"
    (Option.exists (conv_judgment f16_only) ~f:(fun j ->
         mentions (refusal j.cj_gpu) ~sub:"advertises no mma tile for operands"
         && not (mentions (refusal j.cj_gpu) ~sub:"c_zeroed")));
  let cpu = census ~where:"conv whole-routine cpu" ~is_gpu:false ~limits:cpu_limits opt in
  claim_consistent "conv whole-routine (CPU)" ~is_gpu:false ~limits:cpu_limits opt cpu;
  p "conv whole-routine: the CPU leg seeds on a 16-byte vector file"
    (Option.exists (conv_judgment cpu) ~f:(fun j -> not (List.is_empty j.cj_cpu.leg_seeds)));
  p "conv whole-routine: seeds without a fusable tail name the fusion recognizer's reason"
    (match cpu.sc_family with
    | Autotune.Seeding_conv { fused_twins = false; twin_refusal = Some _; _ } -> true
    | _ -> false);
  p "conv whole-routine: the zeroed site's row-block flavors are noted withheld"
    (Option.exists (conv_judgment cpu) ~f:(fun j ->
         List.exists j.cj_cpu.leg_notes ~f:(mentions ~sub:"row-block flavors withheld")));
  let narrow =
    census ~where:"conv whole-routine narrow-cpu" ~is_gpu:false ~limits:narrow_cpu_limits opt
  in
  claim_consistent "conv whole-routine (4-byte vectors)" ~is_gpu:false ~limits:narrow_cpu_limits opt
    narrow;
  p "a 4-byte vector file: the CPU leg names it"
    (Option.exists (conv_judgment narrow) ~f:(fun j ->
         mentions (refusal j.cj_cpu) ~sub:"4-byte vector file is under 8 bytes"));
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
        conv_judgment c)
  in
  p_exists "conv segment: an unzeroed site the GPU leg seeds" conv_segments ~f:(fun j ->
      (not j.cj_site.c_zeroed) && not (List.is_empty j.cj_gpu.leg_seeds));
  p_exists "conv segment: row blocks too large for the 9-row site are noted withheld" conv_segments
    ~f:(fun j -> List.exists j.cj_gpu.leg_notes ~f:(mentions ~sub:"fewer than two blocks"));
  (* A workgroup-memory cap between the 8-row block's staged tiles (512 bytes) and the whole padded
     row's (768): the leg still seeds, and notes the flavor the cap refused. *)
  let tight_limits =
    {
      Ir.Backend_intf.no_hardware_limits with
      mma = gpu_limits.Ir.Backend_intf.mma;
      max_workgroup_memory_bytes = Some 600;
    }
  in
  let tight =
    List.filter_map segments ~f:(fun seg ->
        let c = census ~where:"conv segment gpu 600B" ~is_gpu:true ~limits:tight_limits seg in
        claim_consistent "conv segment (600-byte workgroup memory)" ~is_gpu:true
          ~limits:tight_limits seg c;
        conv_judgment c)
  in
  p_exists "a partial workgroup-memory refusal is noted beside the seeds" tight ~f:(fun j ->
      (not (List.is_empty j.cj_gpu.leg_seeds))
      && List.exists j.cj_gpu.leg_notes ~f:(mentions ~sub:"refused by workgroup memory"));
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
  let capture f =
    let file = Stdlib.Filename.temp_file "seeding_census" ".stderr" in
    Stdio.Out_channel.flush Stdio.stderr;
    let saved = Unix.dup Unix.stderr in
    let fd = Unix.openfile file [ Unix.O_WRONLY; Unix.O_TRUNC ] 0o600 in
    Unix.dup2 fd Unix.stderr;
    Unix.close fd;
    let restore () =
      Stdio.Out_channel.flush Stdio.stderr;
      Unix.dup2 saved Unix.stderr;
      Unix.close saved
    in
    (match f () with
    | () -> restore ()
    | exception exn ->
        restore ();
        raise exn);
    let lines = String.split_lines (Stdio.In_channel.read_all file) in
    Stdlib.Sys.remove file;
    List.iter lines ~f:(Stdio.eprintf "%s\n");
    lines
  in
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
