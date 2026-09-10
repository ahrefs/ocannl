(* Convolution sketch families (gh-ocannl-493), route 1: implicit GEMM. A convolution is a matmul
   over a virtual im2col operand, and conv einsums already lower to affine-indexed accumulation
   nests, so the pipeline is a re-association of loops that already exist: reorder to [outer..;
   kernel..; row; oc; ic], pack the input's strided-window [row x ic] slice and the kernel's [ic x
   oc] slice at the kernel-window anchor (the packing Stage IS im2col — same copy nest, conv index
   arithmetic — and normalizes the kernel's stored layout), then Tensorize (row, oc, ic): the
   register-tiled Tile_mma micro-kernel with the accumulator contracted to a fragment resident
   across the whole kernel-window chain (gh-ocannl-480, gh-ocannl-501).

   Pinned here: - [Autotune.detect_conv] recognizes the conv accumulation across stride/padding
   variants, with strides, dilations, and padding offsets read off the projections. - The hand-built
   implicit-GEMM pipeline (C backends): the packed+tensorized form matches the reorder-only serial
   twin BITWISE (the register tiling's fused per-element chains), and the reorder-only twin matches
   the natural form within float-reassociation tolerance (moving [ic] inside the kernel loops
   reorders each element's reduction — conv sketches are tolerance-tier against unscheduled code,
   like the GPU fragment paths). - Autotune seeding: on the C backends a conv+bias+relu graph seeds
   the serial and Grid-parallel conv pipelines, each with its fused-epilogue twin (gh-ocannl-486):
   the whole-window accumulator contraction (gh-ocannl-501) lands the fragment store-back after the
   full kernel window, so [Fuse_epilogue]'s exactly-once check passes on multi-window (2-D) convs
   too. The tuned routine matches the untuned twin, and the winning schedule round-trips through the
   saved form. - The GPU staged leg: the same re-association with Grid-typed outer loops,
   cooperative shared staging at the kernel-window anchor, and [Tensorize] at the backend lane width
   — value-pinned on metal against the natural form (tolerance tier), and seeded per fission segment
   on backends with an mma capability (intrinsic-tile divisibility gates, unzeroed segments only). -
   The Grid conv flavor on aligned-merged segments: with more than one companion nest the pipeline
   adopts the default preset's whole-segment Grid geometry ([Sched.default_cpu]'s aligned cross-nest
   analysis), value-pinned on the C backends and seeded per fission segment. - Strided rows
   (gh-ocannl-502): the compacting [Stage] packs a stride-2 window densely, so the strided site
   seeds the same flavors as a unit-stride one — pinned by the seed counts, by executing {e every}
   proposed candidate against the natural form (candidates are timed, not value-checked), and by the
   tuned stride-2 routine on both legs. - detect_conv's pattern discipline: a plain matmul is not a
   conv site. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
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

let named name (comp : Asgns.comp) : Asgns.comp =
  { comp with asgns = Asgns.Block_comment (name, comp.asgns) }

let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")
let skipped = Verdict.skipped ~backend:backend_name
let on_cpu = Sched.backend_is_cpu backend_name

let advertises_mma_format a b d =
  Ir.Backend_intf.advertises_mma_format (Context.hardware_limits (Context.auto ())) ~a ~b ~d

let has_uniform_f32_mma =
  advertises_mma_format Ir.Backend_intf.Mma_f32 Ir.Backend_intf.Mma_f32 Ir.Backend_intf.Mma_f32

let has_tf32_mma =
  advertises_mma_format Ir.Backend_intf.Mma_tf32 Ir.Backend_intf.Mma_tf32 Ir.Backend_intf.Mma_f32

let has_f32_input_mma = has_uniform_f32_mma || has_tf32_mma

module Generated = Test_utils.Generated

let () = Generated.init ~backend_name

let nest_paths (llc : LL.t) : Ir.Indexing.symbol list list =
  let strip stmts = List.filter stmts ~f:(function LL.Noop | LL.Comment _ -> false | _ -> true) in
  let rec path (llc : LL.t) : Ir.Indexing.symbol list =
    match llc with
    | LL.For_loop { index; body; _ } ->
        index :: (match strip (LL.flat_lines [ body ]) with [ single ] -> path single | _ -> [])
    | LL.If { body; _ } -> path body
    | _ -> []
  in
  List.filter_map (LL.flat_lines [ llc ]) ~f:(fun stmt ->
      match path stmt with [] -> None | p -> Some p)

(* Deterministic operands so sibling graphs compute identical values (forward code is consumed by
   compilation, so each leg builds its own graph). *)
let make_x tag =
  NTDSL.init ~l:(tag ^ "x") ~prec:Ir.Ops.single ~b:[ 2 ] ~o:[ 11; 11; 4 ]
    ~f:(fun idcs -> Float.of_int ((idcs.(0) + idcs.(1) + (2 * idcs.(2)) + (3 * idcs.(3))) % 7))
    ()

(* Same values as {!make_x}, but through the padding-aware [reshape] constructor: the buffer is
   created at the node's first compilation with the inferred padding (data copied into the
   interior), so a padded conv's halo demand is satisfiable — [init] commits an unpadded layout at
   creation, and a padded conv on it is rejected. *)
let make_x_fresh tag =
  let ndarray =
    Ir.Ndarray.init_array ~debug:(tag ^ "x") Ir.Ops.single ~dims:[| 2; 11; 11; 4 |] ~padding:None
      ~f:(fun idcs -> Float.of_int ((idcs.(0) + idcs.(1) + (2 * idcs.(2)) + (3 * idcs.(3))) % 7))
  in
  NTDSL.reshape ~l:(tag ^ "x") ~b:[ 2 ] ~o:[ 11; 11; 4 ] ndarray ()

let make_kern tag =
  NTDSL.init ~l:(tag ^ "k") ~prec:Ir.Ops.single ~i:[ 3; 3; 4 ] ~o:[ 8 ]
    ~f:(fun idcs ->
      Float.of_int (((2 * idcs.(0)) + idcs.(1) + idcs.(2) + (3 * idcs.(3))) % 5) -. 2.)
    ()

let run_plain name y =
  let ctx = Context.auto () in
  let ctx, routine =
    Context.compile
      ~lowered_transform:(fun opt -> [ opt ])
      ctx
      (named name (Train.forward y))
      Ir.Indexing.Empty
  in
  let ctx = Context.run ctx routine in
  nonzero name (Context.get_values ctx y.Tensor.value)

let () =
  (* === detect_conv across stride/padding variants (via the conv2d block: random params are fine,
     only the lowered structure is inspected) === *)
  let detect_leg tag ~stride ~use_padding ~want_stride ~want_offset ~want_row ~want_seeds =
    (* The padded leg's input goes through the padding-aware constructor: its layout commits at
       first compilation WITH the conv's halo, and the lowered access is then offset-free in buffer
       space — a valid conv over the physically padded buffer. *)
    let x = if use_padding then make_x_fresh tag else make_x tag in
    let conv =
      Nn_blocks.conv2d ~label:[ tag ] ~kernel_size:3 ~stride ~use_padding ~out_channels:8 ()
    in
    let y = conv x in
    let site = ref None in
    let n_seeds = ref (-1) in
    let ctx = Context.auto () in
    (* Pin the vector width so the C-backend seed-count assertion is backend-independent (the check
       always passes is_cpu; only [limits] would otherwise vary per device). *)
    let limits = { (Context.hardware_limits ctx) with Ir.Backend_intf.simd_vector_bytes = 16 } in
    let transform (opt : LL.optimized) =
      site := Autotune.detect_conv opt.LL.llc;
      n_seeds := List.length (Autotune.sketch_seed_params ~is_gpu:false ~is_cpu:true ~limits opt);
      opt
    in
    let ctx = Train.init_params ctx Ir.Indexing.Empty y in
    let ctx, routine =
      Context.compile
        ~lowered_transform:(fun o -> [ transform o ])
        ctx
        (named (tag ^ "_det") (Train.forward y))
        Ir.Indexing.Empty
    in
    ignore (ctx, routine);
    (* The C-backend seed count: serial + Grid, each with its fused-epilogue twin (the 3x3 conv's
       two kernel-window loops are no obstacle since the whole-window contraction, gh-ocannl-501). A
       strided row seeds the same four (gh-ocannl-502): the compacting [Stage] packs the strided
       window densely, so the pipeline applies exactly as on a unit-stride row. *)
    p (tag ^ " C-backend conv seeds") (!n_seeds = want_seeds);
    match !site with
    | None -> p (tag ^ " detected") false
    | Some s ->
        p_all (tag ^ " detected") s.Autotune.c_axes ~f:(fun cx ->
            List.length s.Autotune.c_axes = 2
            && s.Autotune.c_nrow = want_row && s.Autotune.c_noc = 8 && s.Autotune.c_nred = 4
            && s.Autotune.c_zeroed && s.Autotune.c_fma
            && cx.Autotune.cx_stride = want_stride
            && cx.Autotune.cx_dilation = 1
            && cx.Autotune.cx_offset = want_offset
            && cx.Autotune.cx_nk = 3)
  in
  detect_leg "cvd_s1v" ~stride:1 ~use_padding:false ~want_stride:1 ~want_offset:0 ~want_row:9
    ~want_seeds:4;
  detect_leg "cvd_s2v" ~stride:2 ~use_padding:false ~want_stride:2 ~want_offset:0 ~want_row:5
    ~want_seeds:4;
  detect_leg "cvd_s1p" ~stride:1 ~use_padding:true ~want_stride:1 ~want_offset:0 ~want_row:11
    ~want_seeds:4;

  (* === Pattern discipline: a matmul is not a conv site === *)
  (let ma =
     NTDSL.init ~l:"cvm_a" ~prec:Ir.Ops.single ~i:[ 16 ] ~o:[ 16 ]
       ~f:(fun idcs -> Float.of_int ((idcs.(0) + idcs.(1)) % 5))
       ()
   in
   let mb =
     NTDSL.init ~l:"cvm_b" ~prec:Ir.Ops.single ~i:[ 16 ] ~o:[ 16 ]
       ~f:(fun idcs -> Float.of_int ((idcs.(0) - idcs.(1)) % 3))
       ()
   in
   let%op mc = ma * mb in
   let site = ref (Some ()) in
   let transform (opt : LL.optimized) =
     site := Option.map (Autotune.detect_conv opt.LL.llc) ~f:(fun _ -> ());
     opt
   in
   let ctx = Context.auto () in
   let ctx, routine =
     Context.compile
       ~lowered_transform:(fun o -> [ transform o ])
       ctx
       (named "cvm_det" (Train.forward mc))
       Ir.Indexing.Empty
   in
   ignore (ctx, routine);
   p "matmul is not a conv site" (Option.is_none !site));

  (* === The hand-built implicit-GEMM pipeline (C backends; unit-lane Tensorize like the packed mma
     pipelines). Stride-2 and padded variants included: strided windows only change the packing
     Stage's index arithmetic, and a padded conv reads a physically padded source whose halo is part
     of the buffer — the lowered nest is offset-free in buffer space (a valid conv over the padded
     dims), so [Stage]'s edge guards and the tensorization apply unchanged. === *)
  let make_conv_s1v sub =
    let x = make_x sub in
    let kern = make_kern sub in
    let%op y =
      x +* "...| 1*oh<+kh, 1*ow<+kw, ..ic..; |kh, kw, ..ic.. -> ..oc.. => ...| oh, ow, ..oc.." kern
    in
    (x, kern, y)
  in
  let make_conv_s1p sub =
    (* The conv input goes through the padding-aware constructor ([init] would commit an unpadded
       layout at creation, and the padded conv on it would be rejected); it is a host-init like the
       other legs' inputs, so it is not written in the routine and the packing Stage applies. *)
    let x = make_x_fresh sub in
    let kern = make_kern sub in
    let%op y =
      x +* "...| 1*oh=+kh, 1*ow=+kw, ..ic..; |kh, kw, ..ic.. -> ..oc.. => ...| oh, ow, ..oc.." kern
    in
    (x, kern, y)
  in
  let make_conv_s2v sub =
    let x = make_x sub in
    let kern = make_kern sub in
    let%op y =
      x +* "...| 2*oh<+kh, 2*ow<+kw, ..ic..; |kh, kw, ..ic.. -> ..oc.. => ...| oh, ow, ..oc.." kern
    in
    (x, kern, y)
  in
  let run_sched name (x, kern, y) ~tensorized =
    let transform (opt : LL.optimized) =
      let paths = nest_paths opt.LL.llc in
      let _b, _oh, ow, oc, ic, kh, kw =
        match List.find_exn paths ~f:(fun q -> List.length q = 7) with
        | [ b; oh; ow; oc; ic; kh; kw ] -> (b, oh, ow, oc, ic, kh, kw)
        | _ -> assert false
      in
      let sink sym below = List.map below ~f:(fun inner -> Sched.Swap { outer = sym; inner }) in
      let reorder =
        sink ow [ oc; ic; kh; kw ]
        @ sink oc [ ic; kh; kw ]
        @ sink ic [ kh; kw ]
        @ sink ic [ oc; ow ]
        @ sink oc [ ow ]
      in
      let stage source tile_loops =
        Sched.Stage
          {
            source;
            tile_loops;
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
        if not tensorized then reorder
        else
          let tz, _lane = Sched.tensorize ~i:ow ~j:oc ~k:ic ~simd_width:1 () in
          reorder @ [ stage x.Tensor.value [ ow; ic ]; stage kern.Tensor.value [ ic; oc ]; tz ]
      in
      Sched.apply sched opt
    in
    let ctx = Context.auto () in
    let ctx, routine =
      Context.compile
        ~lowered_transform:(fun o -> [ transform o ])
        ctx
        (named name (Train.forward y))
        Ir.Indexing.Empty
    in
    let ctx = Context.run ctx routine in
    Context.get_values ctx y.Tensor.value
  in
  let pipeline_leg tag make_conv =
    let want =
      let _, _, y = make_conv (tag ^ "_r") in
      run_plain (tag ^ "_ref") y
    in
    let swapped = run_sched (tag ^ "_swap") (make_conv (tag ^ "_s")) ~tensorized:false in
    p_all2 (tag ^ ": reorder-only conv matches the natural form within tolerance") swapped want
      ~f:(fun a b -> Float.(abs (a - b) < 1e-3));
    if on_cpu then (
      let full = run_sched (tag ^ "_gemm") (make_conv (tag ^ "_g")) ~tensorized:true in
      p_all2
        (tag ^ ": packed+tensorized conv matches the reorder-only twin bitwise")
        full swapped ~f:Float.equal;
      p_all2 (tag ^ ": packed+tensorized conv matches the natural form within tolerance") full want
        ~f:(fun a b -> Float.(abs (a - b) < 1e-3)))
    else (
      skipped (tag ^ ": packed+tensorized conv matches the reorder-only twin bitwise");
      skipped (tag ^ ": packed+tensorized conv matches the natural form within tolerance"))
  in
  pipeline_leg "cvg" make_conv_s1v;
  (* The padded pipeline leg: the packing Stage's tile loads iterate the padded buffer (its edge
     guards compare against the padded dims, [Tn.dims]), so the halo values participate in the
     packed tile exactly as in the natural form — staging padded convs is sound. *)
  pipeline_leg "cvp" make_conv_s1p;
  (* A strided row used to pack a dilated tile ([Stage] packed by index range), which [Tensorize]'s
     unit-coefficient discipline rejected. The compacting Stage (gh-ocannl-502) packs the strided
     window densely — source reads keep the stride, tile store/read use coefficient 1 — so the
     packed+tensorized stride-2 pipeline now runs, matching its serial twins like the unit-stride
     legs — and the seeding wave proposes it (asserted above, executed below). *)
  pipeline_leg "cvg2" make_conv_s2v;
  if on_cpu then (
    let struct_pin routine =
      let src = Generated.read routine in
      let has s = String.is_substring src ~substring:s in
      has "Tile_mma register tiling" && has "fragment_" && has "tile_"
    in
    p "conv pipeline structure: im2col packs, register tiling, resident fragment"
      (struct_pin "cvg_gemm");
    p "cvg2 pipeline structure: compacted im2col packs, register tiling fires on the strided row"
      (struct_pin "cvg2_gemm"))
  else (
    skipped "conv pipeline structure: im2col packs, register tiling, resident fragment";
    skipped
      "cvg2 pipeline structure: compacted im2col packs, register tiling fires on the strided row");
  (* v1 fence: hoisted (host-side) packing does not compact — a strided tile part is rejected loudly
     rather than silently packing a dilated tile. *)
  (let x, _kern, y = make_conv_s2v "cvg2_h" in
   let transform (opt : LL.optimized) =
     let paths = nest_paths opt.LL.llc in
     let ow, ic =
       match List.find_exn paths ~f:(fun q -> List.length q = 7) with
       | [ _b; _oh; ow; _oc; ic; _kh; _kw ] -> (ow, ic)
       | _ -> assert false
     in
     Sched.apply
       [
         Sched.Stage
           {
             source = x.Tensor.value;
             tile_loops = [ ow; ic ];
             shared = false;
             cooperative = None;
             hoisted = true;
             swizzle = None;
             pad_stride = None;
             pipeline_depth = 1;
             tile_prec = None;
           };
       ]
       opt
   in
   match
     Context.compile
       ~lowered_transform:(fun o -> [ transform o ])
       (Context.auto ())
       (named "cvg2_hoist" (Train.forward y))
       Ir.Indexing.Empty
   with
   | _ -> p "cvg2: hoisted staging of a strided window rejected (no compaction in v1)" false
   | exception Invalid_argument msg ->
       p "cvg2: hoisted staging of a strided window rejected (no compaction in v1)"
         (String.is_substring msg ~substring:"hoisted staging does not support compacting"));

  (* === The seeded stride-2 candidates, executed (gh-ocannl-502). Sketch candidates are timed, not
     value-checked, so a seed that proposes an unsound schedule would be crowned silently: each seed
     the wave proposes on a stride-2 conv gets its own compile and is compared against the natural
     form. The conv-alone graph carries its own [Zero_out], so the proposals are the serial pipeline
     and its Grid twin (whose zero expansion adopts the same geometry). === *)
  if on_cpu then (
    let limits =
      { (Context.hardware_limits (Context.auto ())) with Ir.Backend_intf.simd_vector_bytes = 16 }
    in
    let seeds = ref [] in
    (let _, _, y = make_conv_s2v "cvs2_p" in
     let transform (opt : LL.optimized) =
       seeds := Autotune.sketch_seed_params ~is_gpu:false ~is_cpu:true ~limits opt;
       opt
     in
     ignore
       (Context.compile
          ~lowered_transform:(fun o -> [ transform o ])
          (Context.auto ())
          (named "cvs2_probe" (Train.forward y))
          Ir.Indexing.Empty));
    let want2 =
      let _, _, y = make_conv_s2v "cvs2_r" in
      run_plain "cvs2_ref" y
    in
    let run_seed i p_ =
      let tag = Printf.sprintf "cvs2_%d" i in
      let _, _, y = make_conv_s2v tag in
      let transform (opt : LL.optimized) = Sched.apply (Autotune.sketch_schedule ~p:p_ opt) opt in
      let ctx = Context.auto () in
      let ctx, routine =
        Context.compile
          ~lowered_transform:(fun o -> [ transform o ])
          ctx
          (named tag (Train.forward y))
          Ir.Indexing.Empty
      in
      let ctx = Context.run ctx routine in
      let got = Context.get_values ctx y.Tensor.value in
      (not (Array.is_empty got))
      && Array.for_all2_exn got want2 ~f:(fun a b -> Float.(abs (a - b) < 1e-3))
    in
    p_all "cvs2: the stride-2 conv seeds are the serial pipeline and its Grid twin" !seeds
      ~f:(fun s ->
        List.length !seeds = 2
        && s.Autotune.sk_conv && (not s.Autotune.sk_gpu)
        && List.count !seeds ~f:(fun t -> t.Autotune.sk_grid) = 1);
    p_alli "cvs2: every seeded stride-2 candidate matches the natural form within tolerance" !seeds
      ~f:run_seed)
  else (
    skipped "cvs2: the stride-2 conv seeds are the serial pipeline and its Grid twin";
    skipped "cvs2: every seeded stride-2 candidate matches the natural form within tolerance");

  (* === Autotune seeding on conv+bias+relu: serial + Grid conv pipelines, each with its
     fused-epilogue twin (2-D convs fuse since the whole-window contraction, gh-ocannl-501; the
     unfused Grid candidate fails validation — the tail write is uncovered — and only its twin
     survives); the tuned routine matches the untuned twin === *)
  let clean_cache dir =
    if Stdlib.Sys.file_exists dir && Stdlib.Sys.is_directory dir then
      Array.iter (Stdlib.Sys.readdir dir) ~f:(fun f ->
          Stdlib.Sys.remove (Stdlib.Filename.concat dir f))
  in
  clean_cache "autotune_cache_conv";
  let make_tail sub =
    let x = make_x sub in
    let kern = make_kern sub in
    let bias =
      NTDSL.init ~l:(sub ^ "b") ~prec:Ir.Ops.single ~o:[ 8 ]
        ~f:(fun idcs -> Float.of_int (idcs.(0) % 3) -. 1.)
        ()
    in
    let%op pr =
      x +* "...| 1*oh<+kh, 1*ow<+kw, ..ic..; |kh, kw, ..ic.. -> ..oc.. => ...| oh, ow, ..oc.." kern
    in
    let%op y = relu (pr + bias) in
    (x, kern, pr, y)
  in
  let want_t =
    let _, _, _, y = make_tail "cvt_r" in
    run_plain "cvt_ref" y
  in
  let _, _, _, y = make_tail "cvt_t" in
  let reports = ref [] in
  let ctx = Context.auto () in
  let ctx, routine =
    Autotune.tune ~beam_width:2 ~rounds:0 ~repeats:1 ~cache_dir:"autotune_cache_conv"
      ~timing_ctx:(Context.auto ())
      ~report:(fun r -> reports := r :: !reports)
      ctx
      (named "cvt_tuned" (Train.forward y))
      Ir.Indexing.Empty
  in
  let ctx = Context.run ctx routine in
  let got_t = Context.get_values ctx y.Tensor.value in
  (match !reports with
  | [ r ] ->
      p "conv sketches seeded (serial+grid, with fused-epilogue twins)"
        (if on_cpu then
           r.Autotune.sketch_candidates = 4 && r.Autotune.epilogue_sketch_candidates = 2
         else r.Autotune.sketch_candidates = 0)
  | _ -> p "conv sketches seeded (serial+grid, with fused-epilogue twins)" false);
  p_all2 "tuned conv+tail matches the untuned twin within tolerance" got_t want_t ~f:(fun a b ->
      Float.(abs (a - b) < 1e-3));

  (* === The fused conv twin, hand-built (deterministic — independent of which candidate the tuner
     crowns): the serial implicit-GEMM pipeline plus [Fuse_epilogue] on the conv output. A
     successful compile IS the whole-window pin (gh-ocannl-501): [Fuse_epilogue] raises when the
     fragment store-back sits inside a surplus kernel-window loop, so fusing a 3x3 conv requires the
     contraction to have landed the store-back after the full window. The fused routine merges the
     tail into the conv nest (one real statement beside the [Zero_out]), and elementwise tails never
     reorder the reduction — fused values match the unfused pipeline BITWISE on the C backends.
     === *)
  if on_cpu then (
    let run_tail_sched name (x, kern, pr, y) ~fused =
      let n_real = ref (-1) in
      let transform (opt : LL.optimized) =
        let paths = nest_paths opt.LL.llc in
        let _b, _oh, ow, oc, ic, kh, kw =
          match List.find_exn paths ~f:(fun q -> List.length q = 7) with
          | [ b; oh; ow; oc; ic; kh; kw ] -> (b, oh, ow, oc, ic, kh, kw)
          | _ -> assert false
        in
        let sink sym below = List.map below ~f:(fun inner -> Sched.Swap { outer = sym; inner }) in
        let reorder =
          sink ow [ oc; ic; kh; kw ]
          @ sink oc [ ic; kh; kw ]
          @ sink ic [ kh; kw ]
          @ sink ic [ oc; ow ]
          @ sink oc [ ow ]
        in
        let stage source tile_loops =
          Sched.Stage
            {
              source;
              tile_loops;
              shared = false;
              cooperative = None;
              hoisted = false;
              swizzle = None;
              pad_stride = None;
              pipeline_depth = 1;
              tile_prec = None;
            }
        in
        let tz, _lane = Sched.tensorize ~i:ow ~j:oc ~k:ic ~simd_width:1 () in
        let sched =
          reorder
          @ [ stage x.Tensor.value [ ow; ic ]; stage kern.Tensor.value [ ic; oc ]; tz ]
          @
          if fused then [ Sched.Fuse_epilogue { target = pr.Tensor.value; shared = false } ] else []
        in
        let opt = Sched.apply sched opt in
        n_real :=
          List.length
            (List.filter (LL.flat_lines [ opt.LL.llc ]) ~f:(function
              | LL.Noop | LL.Comment _ | LL.Zero_out _ -> false
              | _ -> true));
        opt
      in
      let ctx = Context.auto () in
      let ctx, routine =
        Context.compile
          ~lowered_transform:(fun o -> [ transform o ])
          ctx
          (named name (Train.forward y))
          Ir.Indexing.Empty
      in
      let ctx = Context.run ctx routine in
      (Context.get_values ctx y.Tensor.value, !n_real)
    in
    let unfused, n_unfused = run_tail_sched "cvf_unfused" (make_tail "cvf_u") ~fused:false in
    let fused, n_fused = run_tail_sched "cvf_fused" (make_tail "cvf_f") ~fused:true in
    p "cvf: fusion merges the tail into the conv nest" (n_unfused = 2 && n_fused = 1);
    p_all2 "cvf: fused conv twin matches the unfused pipeline bitwise" fused unfused ~f:Float.equal;
    p_all2 "cvf: fused conv twin matches the natural form within tolerance" fused want_t
      ~f:(fun a b -> Float.(abs (a - b) < 1e-3)))
  else (
    skipped "cvf: fusion merges the tail into the conv nest";
    skipped "cvf: fused conv twin matches the unfused pipeline bitwise";
    skipped "cvf: fused conv twin matches the natural form within tolerance");

  (* === The GPU staged leg and the aligned-merged Grid flavor (gh-ocannl-493 follow-ups). Both legs
     use micro-kernel extents divisible by Metal's 8x8x8 intrinsic tile: row = 8, oc = 16, ic = 8.
     The pipelines run through the fission seam ([Sched.fission_scheduled]) because the conv's
     [Zero_out] must land in its own [`Zeros] segment for the annotated conv kernel to validate —
     the same shape the autotune per-segment seeding targets. === *)
  let make_x8 tag =
    NTDSL.init ~l:(tag ^ "x") ~prec:Ir.Ops.single ~b:[ 2 ] ~o:[ 10; 10; 8 ]
      ~f:(fun idcs -> Float.of_int ((idcs.(0) + idcs.(1) + (2 * idcs.(2)) + (3 * idcs.(3))) % 7))
      ()
  in
  let make_kern8 tag =
    NTDSL.init ~l:(tag ^ "k") ~prec:Ir.Ops.single ~i:[ 3; 3; 8 ] ~o:[ 16 ]
      ~f:(fun idcs ->
        Float.of_int (((2 * idcs.(0)) + idcs.(1) + idcs.(2) + (3 * idcs.(3))) % 5) -. 2.)
      ()
  in
  let make_conv8 sub =
    let x = make_x8 sub in
    let kern = make_kern8 sub in
    let%op y =
      x +* "...| 1*oh<+kh, 1*ow<+kw, ..ic..; |kh, kw, ..ic.. -> ..oc.. => ...| oh, ow, ..oc.." kern
    in
    (x, kern, y)
  in
  (* Adjacent-transposition reorder (mirrors the pipeline's [reorder_swaps]). *)
  let reorder_swaps ~current ~target =
    let cur = Array.of_list current in
    let swaps = ref [] in
    List.iteri target ~f:(fun pos want ->
        match Array.findi cur ~f:(fun _ s -> Ir.Indexing.equal_symbol s want) with
        | None -> assert false
        | Some (q, _) ->
            assert (q >= pos);
            for r = q downto pos + 1 do
              swaps := Sched.Swap { outer = cur.(r - 1); inner = cur.(r) } :: !swaps;
              let tmp = cur.(r - 1) in
              cur.(r - 1) <- cur.(r);
              cur.(r) <- tmp
            done);
    List.rev !swaps
  in
  let conv_target (site : Autotune.conv_site) =
    List.map site.Autotune.c_outer ~f:fst
    @ site.Autotune.c_kernel
    @ [ site.Autotune.c_row; site.Autotune.c_oc; site.Autotune.c_red ]
  in
  (* Compile+run through the fission seam, with [conv_sched] supplying the schedule for the segment
     containing the detected conv site and every other segment staying unscheduled (zero segments
     get the default zero expansion on GPU). *)
  let run_fiss_sched name y ~conv_sched =
    let ctx = Context.auto () in
    let limits = Context.hardware_limits ctx in
    let transforms (opt : LL.optimized) =
      let preset (seg : LL.optimized) =
        match Autotune.detect_conv seg.LL.llc with Some site -> conv_sched site seg | None -> []
      in
      let zero_sched tns = if on_cpu then [] else Sched.zero_expansion ~limits tns in
      Sched.fission_scheduled ~preset ~zero_sched ~static_indices:[] opt
      |> List.map ~f:(fun (_, _, _, post) -> post)
    in
    let ctx, routine =
      Context.compile ~lowered_transform:transforms ctx
        (named name (Train.forward y))
        Ir.Indexing.Empty
    in
    let ctx = Context.run ctx routine in
    Context.get_values ctx y.Tensor.value
  in
  let want8 =
    let _, _, y = make_conv8 "cvu_r" in
    run_plain "cvu_ref" y
  in
  (* --- The GPU staged pipeline, hand-built: Grid outer loops, cooperative shared staging at the
     kernel-window anchor, Tensorize at the backend lane width (the accumulator fragment stays
     resident across the whole kernel window, gh-ocannl-480/501). Tolerance tier: the reorder and
     the fragment arithmetic reassociate each element's reduction. --- *)
  if has_uniform_f32_mma then
    let x, kern, y = make_conv8 "cvu_g" in
    let conv_sched (site : Autotune.conv_site) _seg =
      let w =
        match (Context.hardware_limits (Context.auto ())).Ir.Backend_intf.mma with
        | Some m -> m.Ir.Backend_intf.mma_simd_width
        | None -> 32
      in
      let stage source tile_loops =
        Sched.Stage
          {
            source;
            tile_loops;
            shared = true;
            cooperative = Some w;
            hoisted = false;
            swizzle = None;
            pad_stride = None;
            pipeline_depth = 1;
            tile_prec = None;
          }
      in
      let tz, _lane =
        Sched.tensorize ~i:site.Autotune.c_row ~j:site.Autotune.c_oc ~k:site.Autotune.c_red
          ~simd_width:w ()
      in
      List.map site.Autotune.c_outer ~f:(fun (s, _) -> Sched.Retype { axis = s; ty = LL.Grid })
      @ reorder_swaps ~current:site.Autotune.c_loops ~target:(conv_target site)
      @ [
          stage x.Tensor.value [ site.Autotune.c_row; site.Autotune.c_red ];
          stage kern.Tensor.value [ site.Autotune.c_red; site.Autotune.c_oc ];
          tz;
        ]
    in
    let got = run_fiss_sched "cvu_gpu" y ~conv_sched in
    p_all2 "cvu: GPU staged conv pipeline matches the natural form within tolerance" got want8
      ~f:(fun a b -> Float.(abs (a - b) < 1e-3))
  else skipped "cvu: GPU staged conv pipeline matches the natural form within tolerance";
  (* --- Seeding + tuning on the conv-alone graph: the C backends seed serial + Grid per fission
     segment (and whole-routine); metal seeds the staged GPU flavor per fission segment (the
     whole-routine site is zeroed, hence gated). --- *)
  clean_cache "autotune_cache_conv_gpu";
  let _, _, y = make_conv8 "cvu_t" in
  let reports = ref [] in
  let ctx = Context.auto () in
  let ctx, routine =
    Autotune.tune ~beam_width:2 ~rounds:0 ~repeats:1 ~cache_dir:"autotune_cache_conv_gpu"
      ~timing_ctx:(Context.auto ())
      ~report:(fun r -> reports := r :: !reports)
      ctx
      (named "cvu_tuned" (Train.forward y))
      Ir.Indexing.Empty
  in
  let ctx = Context.run ctx routine in
  let got_u = Context.get_values ctx y.Tensor.value in
  (match !reports with
  | [ r ] ->
      (* metal: the whole-extent staged flavor plus its pd2 pipelined twin (gh-ocannl-487). *)
      p "cvu: conv seeds per fission segment (GPU staged on metal; serial+grid on cc)"
        (if has_uniform_f32_mma then
           r.Autotune.sketch_candidates = 0
           && r.Autotune.fiss_sketch_candidates = 2
           && r.Autotune.fiss_sketch_timed = 2
         else if on_cpu then
           r.Autotune.sketch_candidates = 2
           && r.Autotune.fiss_sketch_candidates = 2
           && r.Autotune.fiss_sketch_timed = 2
         else true)
  | _ -> p "cvu: conv seeds per fission segment (GPU staged on metal; serial+grid on cc)" false);
  p_all2 "cvu: tuned conv matches the untuned twin within tolerance" got_u want8 ~f:(fun a b ->
      Float.(abs (a - b) < 1e-3));

  (* --- The GPU staged pipeline on a stride-2 conv (gh-ocannl-502): the compacting Stage packs the
     strided row densely through the cooperative shared tiles — the loads keep the stride, the tile
     store/read are unit-coefficient — so the tensorized micro-kernel applies unchanged. Input 17
     with k=3, s=2 keeps row = 8, oc = 16, ic = 8 (the 8x8x8 intrinsic tile divides). --- *)
  let make_conv8_s2 sub =
    let x =
      NTDSL.init ~l:(sub ^ "x") ~prec:Ir.Ops.single ~b:[ 2 ] ~o:[ 17; 17; 8 ]
        ~f:(fun idcs -> Float.of_int ((idcs.(0) + idcs.(1) + (2 * idcs.(2)) + (3 * idcs.(3))) % 7))
        ()
    in
    let kern = make_kern8 sub in
    let%op y =
      x +* "...| 2*oh<+kh, 2*ow<+kw, ..ic..; |kh, kw, ..ic.. -> ..oc.. => ...| oh, ow, ..oc.." kern
    in
    (x, kern, y)
  in
  let want8s2 =
    let _, _, y = make_conv8_s2 "cvu2_r" in
    run_plain "cvu2_ref" y
  in
  if has_f32_input_mma then
    let x, kern, y = make_conv8_s2 "cvu2_g" in
    let conv_sched (site : Autotune.conv_site) _seg =
      let w =
        match (Context.hardware_limits (Context.auto ())).Ir.Backend_intf.mma with
        | Some m -> m.Ir.Backend_intf.mma_simd_width
        | None -> 32
      in
      let stage source tile_loops =
        Sched.Stage
          {
            source;
            tile_loops;
            shared = true;
            cooperative = Some w;
            hoisted = false;
            swizzle = None;
            pad_stride = None;
            pipeline_depth = 1;
            tile_prec = None;
          }
      in
      let tz, _lane =
        Sched.tensorize ~i:site.Autotune.c_row ~j:site.Autotune.c_oc ~k:site.Autotune.c_red
          ~simd_width:w ()
      in
      List.map site.Autotune.c_outer ~f:(fun (s, _) -> Sched.Retype { axis = s; ty = LL.Grid })
      @ reorder_swaps ~current:site.Autotune.c_loops ~target:(conv_target site)
      @ [
          stage x.Tensor.value [ site.Autotune.c_row; site.Autotune.c_red ];
          stage kern.Tensor.value [ site.Autotune.c_red; site.Autotune.c_oc ];
          tz;
        ]
    in
    let got = run_fiss_sched "cvu2_gpu" y ~conv_sched in
    p_all2 "cvu2: GPU staged stride-2 conv (compacted) matches the natural form within tolerance"
      got want8s2 ~f:(fun a b -> Float.(abs (a - b) < 1e-3))
  else
    skipped "cvu2: GPU staged stride-2 conv (compacted) matches the natural form within tolerance";

  (* --- The same stride-2 site through the seeding wave (gh-ocannl-502): the per-segment counts
     match the unit-stride [cvu] leg — the stride only changes the packing Stage's load arithmetic,
     not which flavors are proposable — and the tuned routine matches the untuned twin. --- *)
  clean_cache "autotune_cache_conv_s2";
  let _, _, y = make_conv8_s2 "cvu2_t" in
  let reports = ref [] in
  let ctx = Context.auto () in
  let ctx, routine =
    Autotune.tune ~beam_width:2 ~rounds:0 ~repeats:1 ~cache_dir:"autotune_cache_conv_s2"
      ~timing_ctx:(Context.auto ())
      ~report:(fun r -> reports := r :: !reports)
      ctx
      (named "cvu2_tuned" (Train.forward y))
      Ir.Indexing.Empty
  in
  let ctx = Context.run ctx routine in
  let got_u2 = Context.get_values ctx y.Tensor.value in
  (match !reports with
  | [ r ] ->
      (* metal: the staged flavor plus its pd2 pipelined twin (gh-ocannl-487). *)
      p "cvu2: stride-2 conv seeds per fission segment (GPU staged on metal; serial+grid on cc)"
        (if has_uniform_f32_mma then
           r.Autotune.sketch_candidates = 0
           && r.Autotune.fiss_sketch_candidates = 2
           && r.Autotune.fiss_sketch_timed = 2
         else if on_cpu then
           r.Autotune.sketch_candidates = 2
           && r.Autotune.fiss_sketch_candidates = 2
           && r.Autotune.fiss_sketch_timed = 2
         else true)
  | _ ->
      p "cvu2: stride-2 conv seeds per fission segment (GPU staged on metal; serial+grid on cc)"
        false);
  p_all2 "cvu2: tuned stride-2 conv matches the untuned twin within tolerance" got_u2 want8s2
    ~f:(fun a b -> Float.(abs (a - b) < 1e-3));

  (* --- Row-block flavors on a strided row (gh-ocannl-502 x gh-ocannl-500 x gh-ocannl-485): an
     unzeroed fission segment proposes the cache-panel flavors too, and a row extent that does not
     divide the panel pads the row on top of the compaction — a stride-2 row of 12 with an 8-row
     panel. Every per-segment seed is executed against the natural form: the padded panel's fringe
     rows read the source past its extent, where [Stage]'s Where-form edge guards must supply the
     accumulation identity. --- *)
  let make_conv_s2_r12 sub =
    let x =
      NTDSL.init ~l:(sub ^ "x") ~prec:Ir.Ops.single ~b:[ 2 ] ~o:[ 25; 25; 4 ]
        ~f:(fun idcs -> Float.of_int ((idcs.(0) + idcs.(1) + (2 * idcs.(2)) + (3 * idcs.(3))) % 7))
        ()
    in
    let kern = make_kern sub in
    let%op y =
      x +* "...| 2*oh<+kh, 2*ow<+kw, ..ic..; |kh, kw, ..ic.. -> ..oc.. => ...| oh, ow, ..oc.." kern
    in
    (x, kern, y)
  in
  if on_cpu then (
    let limits =
      { (Context.hardware_limits (Context.auto ())) with Ir.Backend_intf.simd_vector_bytes = 16 }
    in
    let want12 =
      let _, _, y = make_conv_s2_r12 "cvs2b_r" in
      run_plain "cvs2b_ref" y
    in
    let seeds = ref [] in
    (* The seeding target is the unzeroed [`Normal] segment (the conv's [Zero_out] fissions into its
       own [`Zeros] segment) — the same tuples [Autotune.tune]'s per-segment seeding consumes. A
       preset that schedules nothing lets the segments coalesce and the preset re-run on the merged
       whole, so the probe keys on the unzeroed site. *)
    (let _, _, y = make_conv_s2_r12 "cvs2b_p" in
     ignore
       (run_fiss_sched "cvs2b_probe" y ~conv_sched:(fun site seg ->
            if not site.Autotune.c_zeroed then
              seeds := Autotune.sketch_seed_params ~is_gpu:false ~is_cpu:true ~limits seg;
            [])));
    let run_seed i p_ =
      let tag = Printf.sprintf "cvs2b_%d" i in
      let _, _, y = make_conv_s2_r12 tag in
      let got =
        run_fiss_sched tag y ~conv_sched:(fun site seg ->
            if site.Autotune.c_zeroed then [] else Autotune.sketch_schedule ~p:p_ seg)
      in
      (not (Array.is_empty got))
      && Array.for_all2_exn got want12 ~f:(fun a b -> Float.(abs (a - b) < 1e-3))
    in
    p_all "cvs2b: the strided non-dividing row seeds a padded row-panel flavor" !seeds ~f:(fun s ->
        List.exists !seeds ~f:(fun t -> t.Autotune.sk_bm = 8)
        && List.length !seeds = 3
        && s.Autotune.sk_conv);
    p_alli "cvs2b: every per-segment stride-2 candidate matches the natural form within tolerance"
      !seeds ~f:run_seed)
  else (
    skipped "cvs2b: the strided non-dividing row seeds a padded row-panel flavor";
    skipped "cvs2b: every per-segment stride-2 candidate matches the natural form within tolerance");

  (* === Blocked tile flavors (gh-ocannl-500): the GEMM row is split into [Grid] panels before the
     reorder — cache-blocked pool panels on the C backends, extra threadgroups per row-block on GPU
     — and the in-panel micro-kernel is tensorized. Row = 16 so an 8-row block gives two panels; oc
     = 16 and ic = 8 keep the whole-extent intrinsic-tile gates on the tensorized column/reduction.
     Dividing blocks only: the split stays guard-free, so the reorder's [Swap]s and (GPU)
     cooperative-load barriers are well-formed. === *)
  let make_x16 tag =
    NTDSL.init ~l:(tag ^ "x") ~prec:Ir.Ops.single ~b:[ 2 ] ~o:[ 18; 18; 8 ]
      ~f:(fun idcs -> Float.of_int ((idcs.(0) + idcs.(1) + (2 * idcs.(2)) + (3 * idcs.(3))) % 7))
      ()
  in
  let make_conv16 sub =
    let x = make_x16 sub in
    let kern = make_kern8 sub in
    let%op y =
      x +* "...| 1*oh<+kh, 1*ow<+kw, ..ic..; |kh, kw, ..ic.. -> ..oc.. => ...| oh, ow, ..oc.." kern
    in
    (x, kern, y)
  in
  let want16 =
    let _, _, y = make_conv16 "cvb_r" in
    run_plain "cvb_ref" y
  in
  let bm = 8 in
  (* --- Hand-built blocked pipeline through the fission seam: split the row into [Grid] panels of
     [bm], reorder to [outer..; row_o; kernel..; row_i; oc; ic], stage the in-panel slices, and
     tensorize (row_i, oc, ic). On the C backends the panels are non-shared packing Stages and the
     panel loop pool-parallelizes; on metal they are cooperative shared tiles and each panel is a
     threadgroup. --- *)
  let blocked_sched ~shared ~simd_width (x, kern) (site : Autotune.conv_site) _seg =
    let sp_row, row_o, row_i =
      Sched.split ~axis:site.Autotune.c_row ~factor:bm ~outer:LL.Grid ~inner:LL.Serial
    in
    let current =
      List.concat_map site.Autotune.c_loops ~f:(fun s ->
          if Ir.Indexing.equal_symbol s site.Autotune.c_row then [ row_o; row_i ] else [ s ])
    in
    let target =
      List.map site.Autotune.c_outer ~f:fst
      @ [ row_o ] @ site.Autotune.c_kernel
      @ [ row_i; site.Autotune.c_oc; site.Autotune.c_red ]
    in
    let cooperative = if shared then Some simd_width else None in
    let stage source tile_loops =
      Sched.Stage
        {
          source;
          tile_loops;
          shared;
          cooperative;
          hoisted = false;
          swizzle = None;
          pad_stride = None;
          pipeline_depth = 1;
          tile_prec = None;
        }
    in
    let outer_grid =
      if not shared then []
      else List.map site.Autotune.c_outer ~f:(fun (s, _) -> Sched.Retype { axis = s; ty = LL.Grid })
    in
    let tz, _lane =
      Sched.tensorize ~i:row_i ~j:site.Autotune.c_oc ~k:site.Autotune.c_red ~simd_width ()
    in
    (outer_grid @ [ sp_row ]) @ reorder_swaps ~current ~target
    @ [
        stage x.Tensor.value [ row_i; site.Autotune.c_red ];
        stage kern.Tensor.value [ site.Autotune.c_red; site.Autotune.c_oc ];
        tz;
      ]
  in
  (* Constant label across backends (the [.expected] is backend-neutral): the CPU leg uses
     non-shared packing panels, the metal leg cooperative shared tiles at the backend lane width. *)
  let blocked_claim =
    "cvb: blocked (row-panel) conv pipeline matches the natural form within tolerance"
  in
  if on_cpu then
    let x, kern, y = make_conv16 "cvb_c" in
    let got =
      run_fiss_sched "cvb_cpu" y ~conv_sched:(blocked_sched ~shared:false ~simd_width:1 (x, kern))
    in
    p_all2 blocked_claim got want16 ~f:(fun a b -> Float.(abs (a - b) < 1e-3))
  else if has_uniform_f32_mma then
    let w =
      match (Context.hardware_limits (Context.auto ())).Ir.Backend_intf.mma with
      | Some m -> m.Ir.Backend_intf.mma_simd_width
      | None -> 32
    in
    let x, kern, y = make_conv16 "cvb_g" in
    let got =
      run_fiss_sched "cvb_gpu" y ~conv_sched:(blocked_sched ~shared:true ~simd_width:w (x, kern))
    in
    p_all2 blocked_claim got want16 ~f:(fun a b -> Float.(abs (a - b) < 1e-3))
  else skipped blocked_claim;

  (* --- Seeding: the fission segment now proposes the row-block flavors alongside the whole-extent
     ones — an extra CPU cache-panel seed (bm=8, two panels) and an extra GPU threadgroup-block seed
     — and the tuned routine still matches the untuned twin. --- *)
  clean_cache "autotune_cache_conv_blocked";
  let _, _, y = make_conv16 "cvb_t" in
  let reports = ref [] in
  let ctx = Context.auto () in
  let ctx, routine =
    Autotune.tune ~beam_width:2 ~rounds:0 ~repeats:1 ~cache_dir:"autotune_cache_conv_blocked"
      ~timing_ctx:(Context.auto ())
      ~report:(fun r -> reports := r :: !reports)
      ctx
      (named "cvb_tuned" (Train.forward y))
      Ir.Indexing.Empty
  in
  let ctx = Context.run ctx routine in
  let got_b = Context.get_values ctx y.Tensor.value in
  (match !reports with
  | [ r ] ->
      (* cc per-segment: serial + grid + one row-block panel (bm=8, 16/8=2 panels) = 3. metal
         per-segment: whole-extent staged + one threadgroup-block (bm=8), each with its pd2
         pipelined twin (gh-ocannl-487) = 4. *)
      p "cvb: blocked flavors seeded per fission segment"
        (if has_uniform_f32_mma then r.Autotune.fiss_sketch_candidates = 4
         else if on_cpu then r.Autotune.fiss_sketch_candidates = 3
         else true)
  | _ -> p "cvb: blocked flavors seeded per fission segment" false);
  p_all2 "cvb: tuned blocked conv matches the untuned twin within tolerance" got_b want16
    ~f:(fun a b -> Float.(abs (a - b) < 1e-3));

  (* === Blocked flavor on an aligned-merged segment (gh-ocannl-500): the realistic convnet shape,
     conv + materialized companions. The whole-segment Grid geometry comes from the aligned
     cross-nest analysis ([Sched.default_cpu]) and the row is blocked SERIALLY for cache residency
     within each pool chunk — so the block flavor fires on fused conv+bias/relu/pool segments, not
     only conv-alone. Row = 16 (two 8-row panels). The GPU staged conv leg is gated off on
     multi-companion segments, so this leg is cc-only. === *)
  let make_merged16 sub =
    let x = make_x16 sub in
    let kern = make_kern8 sub in
    let%op pr =
      x +* "...| 1*oh<+kh, 1*ow<+kw, ..ic..; |kh, kw, ..ic.. -> ..oc.. => ...| oh, ow, ..oc.." kern
    in
    Ir.Tnode.update_memory_mode pr.Tensor.value Ir.Tnode.On_device 99;
    let%op y2 = relu pr in
    Ir.Tnode.update_memory_mode y2.Tensor.value Ir.Tnode.On_device 99;
    let%op y3 = y2 *. 0.5 in
    (x, kern, y3)
  in
  let want_mb =
    let _, _, y = make_merged16 "cvmb_r" in
    run_plain "cvmb_ref" y
  in
  if on_cpu then
    let x, kern, y = make_merged16 "cvmb_g" in
    let conv_sched (site : Autotune.conv_site) seg =
      let sp_row, row_o, row_i =
        Sched.split ~axis:site.Autotune.c_row ~factor:bm ~outer:LL.Serial ~inner:LL.Serial
      in
      let current =
        List.concat_map site.Autotune.c_loops ~f:(fun s ->
            if Ir.Indexing.equal_symbol s site.Autotune.c_row then [ row_o; row_i ] else [ s ])
      in
      let target =
        List.map site.Autotune.c_outer ~f:fst
        @ [ row_o ] @ site.Autotune.c_kernel
        @ [ row_i; site.Autotune.c_oc; site.Autotune.c_red ]
      in
      let stage source tile_loops =
        Sched.Stage
          {
            source;
            tile_loops;
            shared = false;
            cooperative = None;
            hoisted = false;
            swizzle = None;
            pad_stride = None;
            pipeline_depth = 1;
            tile_prec = None;
          }
      in
      let tz, _lane =
        Sched.tensorize ~i:row_i ~j:site.Autotune.c_oc ~k:site.Autotune.c_red ~simd_width:1 ()
      in
      Sched.default_cpu ~min_parallel:1 seg
      @ (sp_row :: reorder_swaps ~current ~target)
      @ [
          stage x.Tensor.value [ row_i; site.Autotune.c_red ];
          stage kern.Tensor.value [ site.Autotune.c_red; site.Autotune.c_oc ];
          tz;
        ]
    in
    let got = run_fiss_sched "cvmb_cpu" y ~conv_sched in
    p_all2 "cvmb: merged-segment blocked conv pipeline matches the natural form within tolerance"
      got want_mb ~f:(fun a b -> Float.(abs (a - b) < 1e-3))
  else
    skipped "cvmb: merged-segment blocked conv pipeline matches the natural form within tolerance";
  clean_cache "autotune_cache_conv_merged_blocked";
  let _, _, y = make_merged16 "cvmb_t" in
  let reports = ref [] in
  let ctx = Context.auto () in
  let ctx, routine =
    Autotune.tune ~beam_width:2 ~rounds:0 ~repeats:1 ~cache_dir:"autotune_cache_conv_merged_blocked"
      ~timing_ctx:(Context.auto ())
      ~report:(fun r -> reports := r :: !reports)
      ctx
      (named "cvmb_tuned" (Train.forward y))
      Ir.Indexing.Empty
  in
  let ctx = Context.run ctx routine in
  let got_mb = Context.get_values ctx y.Tensor.value in
  (match !reports with
  | [ r ] ->
      (* cc merged fission segment: serial + aligned-grid + one serial row-block panel, each with
         its fused-epilogue twin (gh-ocannl-501) = 6, all compiling (the aligned twins omit the
         preset [Retype] on the consumed tail nest — fuse-before-annotate). metal: GPU conv seeds
         gated off on multi-companion segments. *)
      p "cvmb: blocked flavor seeded on the merged fission segment"
        (if on_cpu then r.Autotune.fiss_sketch_candidates = 6 && r.Autotune.fiss_sketch_timed = 6
         else if has_uniform_f32_mma then r.Autotune.fiss_sketch_candidates = 0
         else true)
  | _ -> p "cvmb: blocked flavor seeded on the merged fission segment" false);
  p_all2 "cvmb: tuned merged-blocked conv matches the untuned twin within tolerance" got_mb want_mb
    ~f:(fun a b -> Float.(abs (a - b) < 1e-3));

  (* --- The aligned-merged Grid flavor: conv + two materialized elementwise companions form one
     aligned-merged segment; the Grid conv seed adopts the default preset's whole-segment geometry
     ([Sched.default_cpu]'s aligned cross-nest analysis Grid-retypes every companion nest). --- *)
  let make_merged sub =
    let x = make_x8 sub in
    let kern = make_kern8 sub in
    let%op pr =
      x +* "...| 1*oh<+kh, 1*ow<+kw, ..ic..; |kh, kw, ..ic.. -> ..oc.. => ...| oh, ow, ..oc.." kern
    in
    (* Keep the intermediates materialized, like real convnet activations: a routine-[Local] conv
       output would classify its [Zero_out] as an in-kernel statement and the whole graph would
       (correctly) stay one serial kernel, never exercising the aligned-merged segment. *)
    Ir.Tnode.update_memory_mode pr.Tensor.value Ir.Tnode.On_device 99;
    let%op y2 = relu pr in
    Ir.Tnode.update_memory_mode y2.Tensor.value Ir.Tnode.On_device 99;
    let%op y3 = y2 *. 0.5 in
    (x, kern, y3)
  in
  let want_m =
    let _, _, y = make_merged "cva_r" in
    run_plain "cva_ref" y
  in
  if on_cpu then
    let x, kern, y = make_merged "cva_g" in
    let conv_sched (site : Autotune.conv_site) seg =
      let stage source tile_loops =
        Sched.Stage
          {
            source;
            tile_loops;
            shared = false;
            cooperative = None;
            hoisted = false;
            swizzle = None;
            pad_stride = None;
            pipeline_depth = 1;
            tile_prec = None;
          }
      in
      let tz, _lane =
        Sched.tensorize ~i:site.Autotune.c_row ~j:site.Autotune.c_oc ~k:site.Autotune.c_red
          ~simd_width:1 ()
      in
      Sched.default_cpu ~min_parallel:1 seg
      @ reorder_swaps ~current:site.Autotune.c_loops ~target:(conv_target site)
      @ [
          stage x.Tensor.value [ site.Autotune.c_row; site.Autotune.c_red ];
          stage kern.Tensor.value [ site.Autotune.c_red; site.Autotune.c_oc ];
          tz;
        ]
    in
    let got = run_fiss_sched "cva_gemm" y ~conv_sched in
    p_all2 "cva: aligned-grid conv pipeline on a merged segment matches within tolerance" got want_m
      ~f:(fun a b -> Float.(abs (a - b) < 1e-3))
  else skipped "cva: aligned-grid conv pipeline on a merged segment matches within tolerance";
  clean_cache "autotune_cache_conv_aligned";
  let _, _, y = make_merged "cva_t" in
  let reports = ref [] in
  let ctx = Context.auto () in
  let ctx, routine =
    Autotune.tune ~beam_width:2 ~rounds:0 ~repeats:1 ~cache_dir:"autotune_cache_conv_aligned"
      ~timing_ctx:(Context.auto ())
      ~report:(fun r -> reports := r :: !reports)
      ctx
      (named "cva_tuned" (Train.forward y))
      Ir.Indexing.Empty
  in
  let ctx = Context.run ctx routine in
  let got_m = Context.get_values ctx y.Tensor.value in
  (match !reports with
  | [ r ] ->
      (* cc: the whole routine seeds the serial flavor and its fused twin (the routine carries the
         conv's [Zero_out], so the aligned analysis bails there and no Grid flavor is seeded); per
         fission segment serial + aligned-grid, each with its fused-epilogue twin, all four timed
         (the aligned twin compiles by fuse-before-annotate, gh-ocannl-501). metal: the merged
         segment has more than one companion — GPU seeds are gated. *)
      p "cva: aligned-grid conv seeded and timed on the merged fission segment"
        (if on_cpu then
           r.Autotune.sketch_candidates = 2
           && r.Autotune.epilogue_sketch_candidates = 1
           && r.Autotune.fiss_sketch_candidates = 4
           && r.Autotune.fiss_sketch_timed = 4
         else r.Autotune.sketch_candidates = 0 && r.Autotune.fiss_sketch_candidates = 0)
  | _ -> p "cva: aligned-grid conv seeded and timed on the merged fission segment" false);
  p_all2 "cva: tuned merged conv graph matches the untuned twin within tolerance" got_m want_m
    ~f:(fun a b -> Float.(abs (a - b) < 1e-3))
