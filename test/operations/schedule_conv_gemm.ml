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
   on backends with an mma capability, and on the whole routine, whose zeroed site the pipeline
   expands with its own geometry (gh-ocannl-503): every whole-routine GPU seed executed. - The Grid
   conv flavor on aligned-merged segments: with more than one companion nest the pipeline adopts the
   default preset's whole-segment Grid geometry ([Sched.default_cpu]'s aligned cross-nest analysis),
   value-pinned on the C backends and seeded per fission segment; on a GPU the staged leg seeds the
   same segments (gh-ocannl-1406, executed in conv_merged_gpu_seeds). - Strided rows
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

(* The backend's accumulator residency, which a [Privatize] tile is minted at (gh-ocannl-1116). *)
let accum_prec = Ll_test.backend_accum_prec
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

(* Every leg below asks for a tensorized conv pipeline, and a [Tile_mma] that declines to the scalar
   fallback computes the same values: parity alone cannot say the pipeline ran tensorized, so each
   executed leg also claims its routine's mma census (gh-ocannl-1283). Gated on the seeder's own
   capability judgment for the f32 site on this backend -- the register tiling on the C backends, an
   advertised f32-input tile on a GPU -- never on what a leg seeded. *)
let tensorized_capability =
  lazy
    (Ll_test.tensorized_capability ~is_gpu:(not on_cpu) ~is_cpu:on_cpu
       ~limits:(Context.hardware_limits (Context.auto ()))
       ~a_prec:Ir.Ops.single ~b_prec:Ir.Ops.single ~d_prec:Ir.Ops.single)

(* The seeder's own judgment that a GPU conv leg can seed an f32 site on this backend. *)
let gpu_capable =
  lazy
    ((not on_cpu)
    && match Lazy.force tensorized_capability with `Advertised -> true | `Withheld _ -> false)

let rendered_tensorized (census : Ir.C_syntax.mma_summary) =
  let ok = Ll_test.rendered_tensorized census in
  if not ok then
    Stdio.eprintf "a conv pipeline rendered %s\n" (Ir.C_syntax.mma_summary_string census);
  ok

let claim_tensorized label ok =
  match Lazy.force tensorized_capability with
  | `Advertised -> p label ok
  | `Withheld aggregation -> Verdict.skipped ~aggregation ~backend:backend_name label

(* Whether the tile the device advertises for an f32 site, under the current policy, divides a
   hand-built micro-kernel of these extents. An unpadded pipeline the tile does not divide declines
   by construction (CUDA's 16x16x8 tf32 tile against an 8-row kernel), so its census claim is
   reported skipped there rather than required. *)
let advertised_tile_divides ~m ~n ~k =
  match (Context.hardware_limits (Context.auto ())).Ir.Backend_intf.mma with
  | None -> false
  | Some mma -> (
      match
        Autotune.mma_tile_for_precisions_in_scope mma ~scope:Ir.Backend_intf.Mma_per_statement
          ~a_prec:Ir.Ops.single ~b_prec:Ir.Ops.single ~d_prec:Ir.Ops.single
      with
      | Some (tm, tn, tk) -> m % tm = 0 && n % tn = 0 && k % tk = 0
      | None -> false)

(* Deterministic operands so sibling graphs compute identical values (forward code is consumed by
   compilation, so each leg builds its own graph). *)
(* The device's vector file at a pinned 16-byte width, its register count kept: the C-backend legs
   below seed under it so their seed counts do not follow the host's width. Stated as a whole
   [simd_vector], width and count, so neither is left to a default (gh-ocannl-1399). *)
let narrowed_to_16 (device : Ir.Backend_intf.hardware_limits) =
  { device with simd_vector = { device.simd_vector with bytes = 16 } }

let make_x tag =
  NTDSL.init ~l:(tag ^ "x") ~prec:Ir.Ops.single ~b:[ 2 ] ~o:[ 11; 11; 4 ]
    ~f:(Ll_test.weighted ~weights:[| 1; 1; 2; 3 |] ~modulus:7 ~offset:0. ~stride:1.)
    ()

(* Same values as {!make_x}, but through the padding-aware [reshape] constructor: the buffer is
   created at the node's first compilation with the inferred padding (data copied into the
   interior), so a padded conv's halo demand is satisfiable — [init] commits an unpadded layout at
   creation, and a padded conv on it is rejected. *)
let make_x_fresh tag =
  let ndarray =
    Ir.Ndarray.init_array ~debug:(tag ^ "x") Ir.Ops.single ~dims:[| 2; 11; 11; 4 |] ~padding:None
      ~f:(Ll_test.weighted ~weights:[| 1; 1; 2; 3 |] ~modulus:7 ~offset:0. ~stride:1.)
  in
  NTDSL.reshape ~l:(tag ^ "x") ~b:[ 2 ] ~o:[ 11; 11; 4 ] ndarray ()

let make_kern tag =
  NTDSL.init ~l:(tag ^ "k") ~prec:Ir.Ops.single ~i:[ 3; 3; 4 ] ~o:[ 8 ]
    ~f:(Ll_test.weighted ~weights:[| 2; 1; 1; 3 |] ~modulus:5 ~offset:(-2.) ~stride:1.)
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
    let limits = narrowed_to_16 (Context.hardware_limits ctx) in
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
       two kernel-window loops are no obstacle since the whole-window contraction, gh-ocannl-501),
       less the unfused Grid flavor: the block's inline bias is a materialized tail its Grid loop
       would leave uncovered, so the conv family tree refutes it (gh-ocannl-697; it used to be
       proposed and fail validation). A strided row seeds the same three (gh-ocannl-502): the
       compacting [Stage] packs the strided window densely, so the pipeline applies exactly as on a
       unit-stride row. *)
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
    ~want_seeds:3;
  detect_leg "cvd_s2v" ~stride:2 ~use_padding:false ~want_stride:2 ~want_offset:0 ~want_row:5
    ~want_seeds:3;
  detect_leg "cvd_s1p" ~stride:1 ~use_padding:true ~want_stride:1 ~want_offset:0 ~want_row:11
    ~want_seeds:3;

  (* === Pattern discipline: a matmul is not a conv site === *)
  (let ma =
     NTDSL.init ~l:"cvm_a" ~prec:Ir.Ops.single ~i:[ 16 ] ~o:[ 16 ]
       ~f:(Ll_test.weighted ~weights:[| 1; 1 |] ~modulus:5 ~offset:0. ~stride:1.)
       ()
   in
   let mb =
     NTDSL.init ~l:"cvm_b" ~prec:Ir.Ops.single ~i:[ 16 ] ~o:[ 16 ]
       ~f:(Ll_test.weighted ~weights:[| 1; -1 |] ~modulus:3 ~offset:0. ~stride:1.)
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
      let paths = Ll_test.nest_paths opt.LL.llc in
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
    (Context.get_values ctx y.Tensor.value, routine.Context.mma)
  in
  let pipeline_leg tag make_conv =
    let want =
      let _, _, y = make_conv (tag ^ "_r") in
      run_plain (tag ^ "_ref") y
    in
    let swapped, _ = run_sched (tag ^ "_swap") (make_conv (tag ^ "_s")) ~tensorized:false in
    p_close (tag ^ ": reorder-only conv matches the natural form within tolerance") swapped want
      ~tol:(fun _ -> 1e-3);
    if on_cpu then (
      let full, census = run_sched (tag ^ "_gemm") (make_conv (tag ^ "_g")) ~tensorized:true in
      p_all2
        (tag ^ ": packed+tensorized conv matches the reorder-only twin bitwise")
        full swapped ~f:Float.equal;
      p_close (tag ^ ": packed+tensorized conv matches the natural form within tolerance") full want
        ~tol:(fun _ -> 1e-3);
      claim_tensorized
        (tag ^ ": the packed+tensorized conv's routine is labeled tensorized")
        (rendered_tensorized census);
      Some census)
    else (
      skipped (tag ^ ": packed+tensorized conv matches the reorder-only twin bitwise");
      skipped (tag ^ ": packed+tensorized conv matches the natural form within tolerance");
      skipped (tag ^ ": the packed+tensorized conv's routine is labeled tensorized");
      None)
  in
  let cvg_census = pipeline_leg "cvg" make_conv_s1v in
  (* The padded pipeline leg: the packing Stage's tile loads iterate the padded buffer (its edge
     guards compare against the padded dims, [Tn.dims]), so the halo values participate in the
     packed tile exactly as in the natural form — staging padded convs is sound. *)
  ignore (pipeline_leg "cvp" make_conv_s1p : Ir.C_syntax.mma_summary option);
  (* A strided row used to pack a dilated tile ([Stage] packed by index range), which [Tensorize]'s
     unit-coefficient discipline rejected. The compacting Stage (gh-ocannl-502) packs the strided
     window densely — source reads keep the stride, tile store/read use coefficient 1 — so the
     packed+tensorized stride-2 pipeline now runs, matching its serial twins like the unit-stride
     legs — and the seeding wave proposes it (asserted above, executed below). *)
  let cvg2_census = pipeline_leg "cvg2" make_conv_s2v in
  (match (cvg_census, cvg2_census) with
  | Some cvg_census, Some cvg2_census ->
      (* Every statement rendering the register tiling is the census's to say (gh-ocannl-1316); the
         im2col tiles and the resident fragment are the source's own names. *)
      let struct_pin routine census =
        let src = Generated.read routine in
        let has s = String.is_substring src ~substring:s in
        Ll_test.all_rendered ~what:routine Ir.C_syntax.Mma_register_tiled
          (Ll_test.renderings census)
        && has "fragment_" && has "tile_"
      in
      p "conv pipeline structure: im2col packs, register tiling, resident fragment"
        (struct_pin "cvg_gemm" cvg_census);
      p "cvg2 pipeline structure: compacted im2col packs, register tiling fires on the strided row"
        (struct_pin "cvg2_gemm" cvg2_census)
  | _ ->
      skipped "conv pipeline structure: im2col packs, register tiling, resident fragment";
      skipped
        "cvg2 pipeline structure: compacted im2col packs, register tiling fires on the strided row");
  (* v1 fence: hoisted (host-side) packing does not compact — a strided tile part is rejected loudly
     rather than silently packing a dilated tile. *)
  (let x, _kern, y = make_conv_s2v "cvg2_h" in
   let transform (opt : LL.optimized) =
     let paths = Ll_test.nest_paths opt.LL.llc in
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
    let limits = narrowed_to_16 (Context.hardware_limits (Context.auto ())) in
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
      let transform (opt : LL.optimized) =
        Sched.apply (Autotune.sketch_schedule ~accum_prec ~p:p_ opt) opt
      in
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
      ( (not (Array.is_empty got))
        && Array.for_all2_exn got want2 ~f:(fun a b -> Float.(abs (a - b) < 1e-3)),
        rendered_tensorized routine.Context.mma )
    in
    p_all "cvs2: the stride-2 conv seeds are the serial pipeline and its Grid twin" !seeds
      ~f:(fun s ->
        List.length !seeds = 2
        && s.Autotune.sk_conv && (not s.Autotune.sk_gpu)
        && List.count !seeds ~f:(fun t -> t.Autotune.sk_grid) = 1);
    let results = List.mapi !seeds ~f:run_seed in
    p_all "cvs2: every seeded stride-2 candidate matches the natural form within tolerance" results
      ~f:fst;
    claim_tensorized "cvs2: every seeded stride-2 candidate's routine is labeled tensorized"
      (List.for_all results ~f:snd && not (List.is_empty results)))
  else (
    skipped "cvs2: the stride-2 conv seeds are the serial pipeline and its Grid twin";
    skipped "cvs2: every seeded stride-2 candidate matches the natural form within tolerance";
    skipped "cvs2: every seeded stride-2 candidate's routine is labeled tensorized");

  (* === Autotune seeding on conv+bias+relu: serial + Grid conv pipelines, each with its
     fused-epilogue twin (2-D convs fuse since the whole-window contraction, gh-ocannl-501). The
     unfused Grid flavor would leave the tail write uncovered, so since gh-ocannl-697 the conv family
     tree refutes it at construction (it used to be proposed and fail validation at compile); its
     fused twin, which relocates the tail under the Grid loop, survives. The tuned routine matches
     the untuned twin === *)
  (* A tuning report's seeding counts and the windows that left its dedup set, on stderr (not part of
     the golden): what a failing count claim saw. *)
  let show_counts tag (r : Autotune.report) =
    Stdio.eprintf
      "%s: sketch_candidates=%d epilogue=%d fiss_sketch_candidates=%d fiss_timed=%d contended=%d \
       unbatched=%d unreadable=%d budget_cut=%b hopeless=%d retried=%d\n"
      tag r.Autotune.sketch_candidates r.Autotune.epilogue_sketch_candidates
      r.Autotune.fiss_sketch_candidates r.Autotune.fiss_sketch_timed r.Autotune.timings_contended
      r.Autotune.timings_unbatched r.Autotune.timings_unreadable r.Autotune.budget_cut
      r.Autotune.timings_hopeless r.Autotune.timings_retried
  in
  (* A per-segment GPU candidate that compiles to its whole-routine twin's digest is deduplicated
     only while that twin's digest is still in the search's dedup set. A window refused for any
     cause, one cut as hopeless (gh-ocannl-1527) and one the budget stopped all drop their digest
     from it, so an equivalent later seed is retried, and timed, rather than deduplicated: a loaded
     host refusing a whole-routine window lets its per-segment twin be timed (gh-ocannl-1484). The
     dedup claim is therefore evaluated only on a report where no window left the set, and skipped
     -- as an environment gap, the host's load -- with the departures on stderr otherwise. Off the
     uniform-f32 GPU legs the per-segment seeds have no whole-routine twin, so there is nothing to
     dedup. *)
  let claim_dedup tag (r : Autotune.report) label ok =
    if not has_uniform_f32_mma then skipped label
    else if
      r.Autotune.timings_contended = 0 && r.Autotune.timings_hopeless = 0
      && not r.Autotune.budget_cut
    then p label ok
    else (
      Stdio.eprintf
        "%s: a timing window left the dedup set (refused=%d of which unbatched=%d unreadable=%d, \
         hopeless=%d, budget_cut=%b): per-segment twins could be retried\n"
        tag r.Autotune.timings_contended r.Autotune.timings_unbatched r.Autotune.timings_unreadable
        r.Autotune.timings_hopeless r.Autotune.budget_cut;
      Verdict.skipped ~aggregation:`Environment ~backend:backend_name label)
  in
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
      show_counts "cvt" r;
      (* cc: serial, its fused twin, and the fused Grid flavor (the unfused Grid one refuted on its
         uncovered tail). metal: the whole-routine zeroed site's two GPU staged flavors, fused only
         (gh-ocannl-503) -- their unfused flavor is refuted on companion coverage (gh-ocannl-697),
         the [cvzt] leg below; a GPU leg that cannot seed an f32 site proposes none. *)
      p "conv sketches seeded (serial+grid, with fused-epilogue twins)"
        (if on_cpu then
           r.Autotune.sketch_candidates = 3 && r.Autotune.epilogue_sketch_candidates = 2
         else if has_uniform_f32_mma then
           r.Autotune.sketch_candidates = 2 && r.Autotune.epilogue_sketch_candidates = 2
         else if Lazy.force gpu_capable then r.Autotune.sketch_candidates > 0
         else r.Autotune.sketch_candidates = 0)
  | _ -> p "conv sketches seeded (serial+grid, with fused-epilogue twins)" false);
  p_close "tuned conv+tail matches the untuned twin within tolerance" got_t want_t ~tol:(fun _ ->
      1e-3);

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
        let paths = Ll_test.nest_paths opt.LL.llc in
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
      (Context.get_values ctx y.Tensor.value, !n_real, routine.Context.mma)
    in
    let unfused, n_unfused, census_u =
      run_tail_sched "cvf_unfused" (make_tail "cvf_u") ~fused:false
    in
    let fused, n_fused, census_f = run_tail_sched "cvf_fused" (make_tail "cvf_f") ~fused:true in
    p "cvf: fusion merges the tail into the conv nest" (n_unfused = 2 && n_fused = 1);
    p_all2 "cvf: fused conv twin matches the unfused pipeline bitwise" fused unfused ~f:Float.equal;
    p_close "cvf: fused conv twin matches the natural form within tolerance" fused want_t
      ~tol:(fun _ -> 1e-3);
    claim_tensorized "cvf: the unfused and the fused conv routines are both labeled tensorized"
      (rendered_tensorized census_u && rendered_tensorized census_f))
  else (
    skipped "cvf: fusion merges the tail into the conv nest";
    skipped "cvf: fused conv twin matches the unfused pipeline bitwise";
    skipped "cvf: fused conv twin matches the natural form within tolerance";
    skipped "cvf: the unfused and the fused conv routines are both labeled tensorized");

  (* === The GPU staged leg and the aligned-merged Grid flavor (gh-ocannl-493 follow-ups). Both legs
     use micro-kernel extents divisible by Metal's 8x8x8 intrinsic tile: row = 8, oc = 16, ic = 8.
     The pipelines run through the fission seam ([Sched.fission_scheduled]) because the conv's
     [Zero_out] must land in its own [`Zeros] segment for the annotated conv kernel to validate —
     the same shape the autotune per-segment seeding targets. === *)
  let make_x8 tag =
    NTDSL.init ~l:(tag ^ "x") ~prec:Ir.Ops.single ~b:[ 2 ] ~o:[ 10; 10; 8 ]
      ~f:(Ll_test.weighted ~weights:[| 1; 1; 2; 3 |] ~modulus:7 ~offset:0. ~stride:1.)
      ()
  in
  let make_kern8 tag =
    NTDSL.init ~l:(tag ^ "k") ~prec:Ir.Ops.single ~i:[ 3; 3; 8 ] ~o:[ 16 ]
      ~f:(Ll_test.weighted ~weights:[| 2; 1; 1; 3 |] ~modulus:5 ~offset:(-2.) ~stride:1.)
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
    (Context.get_values ctx y.Tensor.value, routine.Context.mma)
  in
  let want8 =
    let _, _, y = make_conv8 "cvu_r" in
    run_plain "cvu_ref" y
  in
  (* --- The GPU staged pipeline, hand-built: Grid outer loops, cooperative shared staging at the
     kernel-window anchor, Tensorize at the backend lane width (the accumulator fragment stays
     resident across the whole kernel window, gh-ocannl-480/501). Tolerance tier: the reorder and
     the fragment arithmetic reassociate each element's reduction. --- *)
  if has_uniform_f32_mma then (
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
    let got, census = run_fiss_sched "cvu_gpu" y ~conv_sched in
    p_close "cvu: GPU staged conv pipeline matches the natural form within tolerance" got want8
      ~tol:(fun _ -> 1e-3);
    claim_tensorized "cvu: the GPU staged conv pipeline's routine is labeled tensorized"
      (rendered_tensorized census))
  else (
    skipped "cvu: GPU staged conv pipeline matches the natural form within tolerance";
    skipped "cvu: the GPU staged conv pipeline's routine is labeled tensorized");
  (* --- The whole-routine zeroed GPU site (gh-ocannl-503): unfissioned, the conv output's
     whole-node [Zero_out] shares the annotated kernel, so the pipeline expands it into a per-cell
     nest with the site's own geometry (Grid outer loops, the row-block split, the column over the
     [Tensorize] lane width as a [Workgroup] loop) — or [validate_parallel] rejects the candidate.
     Every GPU seed the seeder proposes on the whole-routine lowering is compiled WITHOUT the
     fission seam and executed against the natural form (candidates are timed, not value-checked, so
     an unsound seed would be crowned silently). The zero is load-bearing: the routine runs twice,
     so a zero nest that missed cells would leave the first run's sums in them. Two graphs: the conv
     alone ([cvz]), and the conv with its bias+relu tail ([cvzt]), whose unfused flavor the conv
     family tree refutes on companion coverage (the tail cannot follow the accumulation nest's full
     chain; gh-ocannl-697 -- these seeds used to be proposed and decline at compile), while the
     fused-epilogue twins compile: every seed proposed is a fused one, and every one compiles. Gated
     on the seeder's own capability judgment for an f32 site, never on the seed list. --- *)
  let whole_routine_gpu_leg ~tag ~build ~want =
    let limits = Context.hardware_limits (Context.auto ()) in
    let seeds = ref [] and zeroed = ref false in
    (let y = build (tag ^ "_p") in
     let transform (opt : LL.optimized) =
       seeds := Autotune.sketch_seed_params ~is_gpu:true ~is_cpu:false ~limits opt;
       zeroed :=
         Option.exists (Autotune.detect_conv opt.LL.llc) ~f:(fun site -> site.Autotune.c_zeroed);
       opt
     in
     ignore
       (Context.compile
          ~lowered_transform:(fun o -> [ transform o ])
          (Context.auto ())
          (named (tag ^ "_probe") (Train.forward y))
          Ir.Indexing.Empty));
    Stdio.eprintf "%s: %d GPU seeds on the whole-routine lowering (not part of the golden)\n" tag
      (List.length !seeds);
    let run_seed i p_ =
      let name = Printf.sprintf "%s_%d" tag i in
      let y = build name in
      let transform (opt : LL.optimized) =
        Sched.apply (Autotune.sketch_schedule ~accum_prec ~p:p_ opt) opt
      in
      match
        Context.compile
          ~lowered_transform:(fun o -> [ transform o ])
          (Context.auto ())
          (named name (Train.forward y))
          Ir.Indexing.Empty
      with
      (* [Context.compile] reports the transform's typed decline as its message. *)
      | exception Invalid_argument msg
        when String.is_substring msg ~substring:"companion coverage (gh-521)" ->
          Stdio.eprintf "%s: declined at companion coverage (not part of the golden)\n" name;
          `Declined
      | ctx, routine ->
          let ctx = Context.run ctx routine in
          let ctx = Context.run ctx routine in
          let got = Context.get_values ctx y.Tensor.value in
          `Ran
            ( (not (Array.is_empty got))
              && Array.for_all2_exn got want ~f:(fun a b -> Float.(abs (a - b) < 1e-3)),
              rendered_tensorized routine.Context.mma )
    in
    (!zeroed, !seeds, List.mapi !seeds ~f:run_seed)
  in
  let ran = function `Ran _ -> true | `Declined -> false in
  let ran_ok f = function `Ran r -> f r | `Declined -> true in
  if Lazy.force gpu_capable then (
    let zeroed, seeds, results =
      whole_routine_gpu_leg ~tag:"cvz" ~want:want8 ~build:(fun sub ->
          let _, _, y = make_conv8 sub in
          y)
    in
    p "cvz: the whole-routine conv site is zeroed in its lowering" zeroed;
    p_all "cvz: the whole-routine zeroed site seeds GPU staged conv flavors" seeds ~f:(fun s ->
        s.Autotune.sk_conv && s.Autotune.sk_gpu);
    p_all
      "cvz: every whole-routine zeroed GPU candidate compiles, and matches the natural form run \
       twice"
      results ~f:(function
      | `Ran (ok, _) -> ok
      | `Declined -> false);
    p_all "cvz: every whole-routine zeroed GPU candidate's routine is labeled tensorized" results
      ~f:(ran_ok snd);
    let zeroed, seeds, results =
      whole_routine_gpu_leg ~tag:"cvzt" ~want:want_t ~build:(fun sub ->
          let _, _, _, y = make_tail sub in
          y)
    in
    p "cvzt: the whole-routine conv+tail site is zeroed in its lowering" zeroed;
    p_all "cvzt: the whole-routine zeroed conv+tail site seeds GPU staged conv flavors" seeds
      ~f:(fun s -> s.Autotune.sk_conv && s.Autotune.sk_gpu);
    p_all "cvzt: every whole-routine zeroed conv+tail GPU seed is a fused-epilogue twin" seeds
      ~f:(fun s -> s.Autotune.sk_epilogue);
    p_all "cvzt: every whole-routine zeroed conv+tail GPU candidate compiles" results ~f:ran;
    p_all
      "cvzt: every compiling whole-routine conv+tail GPU candidate matches the natural form, run \
       twice"
      results ~f:(ran_ok fst);
    p_all "cvzt: every compiling whole-routine conv+tail GPU candidate is labeled tensorized"
      results ~f:(ran_ok snd))
  else
    List.iter ~f:skipped
      [
        "cvz: the whole-routine conv site is zeroed in its lowering";
        "cvz: the whole-routine zeroed site seeds GPU staged conv flavors";
        "cvz: every whole-routine zeroed GPU candidate compiles, and matches the natural form run \
         twice";
        "cvz: every whole-routine zeroed GPU candidate's routine is labeled tensorized";
        "cvzt: the whole-routine conv+tail site is zeroed in its lowering";
        "cvzt: the whole-routine zeroed conv+tail site seeds GPU staged conv flavors";
        "cvzt: every whole-routine zeroed conv+tail GPU seed is a fused-epilogue twin";
        "cvzt: every whole-routine zeroed conv+tail GPU candidate compiles";
        "cvzt: every compiling whole-routine conv+tail GPU candidate matches the natural form, run \
         twice";
        "cvzt: every compiling whole-routine conv+tail GPU candidate is labeled tensorized";
      ];
  (* --- The per-segment fused twins on the folded-zero conv+tail segment (gh-ocannl-1453): the
     segmentation the tuner's [F_sketch] candidates use ([fold_zeros], the default GPU mapping
     probe, promoted locals) folds the conv output's per-cell zero into the conv's own segment, and
     the conv output there is NOT materialized: the fused tail is its only consumer. The fused
     twin's [Fuse_epilogue] places it in workgroup-shared storage, so its zero nest needs the site's
     geometry like any companion -- which companion coverage, counting only materialized writes,
     never gave it: every thread stored every cell, and gh-ocannl-959's binding legality refused
     each fused twin at render time. Every seed of the segment is compiled through the fission seam
     and executed twice against the natural form (the second run fails a zero that missed cells);
     the unfused seeds decline at companion coverage, as on the whole routine ([cvzt]). --- *)
  let segment_gpu_leg ~tag ~build ~want =
    let limits = Context.hardware_limits (Context.auto ()) in
    let fission ~preset opt =
      Sched.fission_scheduled ~promote_locals:true ~fold_zeros:true
        ?keep_mapping:(Sched.fission_keep_mapping ~is_gpu:true ~limits)
        ~preset
        ~zero_sched:(fun tns -> Sched.zero_expansion ~limits tns)
        ~static_indices:[] opt
      |> List.map ~f:(fun (_, _, _, post) -> post)
    in
    let other seg = Sched.default_gpu ~min_parallel:1 ~limits seg in
    let seeds = ref [] and folded_local = ref false in
    (let y = build (tag ^ "_p") in
     let preset (seg : LL.optimized) =
       (match Autotune.detect_conv seg.LL.llc with
       | Some site ->
           let d = site.Autotune.c_d in
           seeds := Autotune.sketch_seed_params ~is_gpu:true ~is_cpu:false ~limits seg;
           folded_local :=
             (not (Ir.Tnode.Placements.is_materialized_peek seg.LL.optimize_ctx.LL.placements d))
             && List.exists (LL.flat_lines [ seg.LL.llc ]) ~f:(function
               | LL.Zero_out _ -> false
               | stmt -> Option.exists (LL.zero_initializer_target stmt) ~f:(Ir.Tnode.equal d))
       | None -> ());
       other seg
     in
     ignore
       (Context.compile ~lowered_transform:(fission ~preset) (Context.auto ())
          (named (tag ^ "_probe") (Train.forward y))
          Ir.Indexing.Empty));
    Stdio.eprintf "%s: %d GPU seeds on the folded conv segment (not part of the golden)\n" tag
      (List.length !seeds);
    let run_seed i p_ =
      let name = Printf.sprintf "%s_%d" tag i in
      let y = build name in
      let preset (seg : LL.optimized) =
        match Autotune.detect_conv seg.LL.llc with
        | Some _ -> Autotune.sketch_schedule ~accum_prec ~p:p_ seg
        | None -> other seg
      in
      let result =
        match
          Context.compile ~lowered_transform:(fission ~preset) (Context.auto ())
            (named name (Train.forward y))
            Ir.Indexing.Empty
        with
        | exception Invalid_argument msg
          when String.is_substring msg ~substring:"companion coverage (gh-521)" ->
            `Declined
        | exception Invalid_argument msg -> `Failed msg
        | ctx, routine ->
            let ctx = Context.run ctx routine in
            let ctx = Context.run ctx routine in
            let got = Context.get_values ctx y.Tensor.value in
            `Ran
              ( (not (Array.is_empty got))
                && Array.for_all2_exn got want ~f:(fun a b -> Float.(abs (a - b) < 1e-3)),
                rendered_tensorized routine.Context.mma )
      in
      (match result with
      | `Declined ->
          Stdio.eprintf "%s: declined at companion coverage (not part of the golden)\n" name
      | `Failed msg -> Stdio.eprintf "%s (fused=%b) failed: %s\n" name p_.Autotune.sk_epilogue msg
      | `Ran (ok, _) -> if not ok then Stdio.eprintf "%s differs from the natural form\n" name);
      (p_, result)
    in
    (!folded_local, !seeds, List.mapi !seeds ~f:run_seed)
  in
  if Lazy.force gpu_capable then (
    let folded_local, seeds, results =
      segment_gpu_leg ~tag:"cvft" ~want:want_t ~build:(fun sub ->
          let _, _, _, y = make_tail sub in
          y)
    in
    p "cvft: the conv segment folds the conv output's per-cell zero, the output not materialized"
      folded_local;
    p_exists "cvft: the folded conv+tail segment seeds fused-epilogue GPU twins" seeds ~f:(fun s ->
        s.Autotune.sk_conv && s.Autotune.sk_gpu && s.Autotune.sk_epilogue);
    let fused = List.filter results ~f:(fun (s, _) -> s.Autotune.sk_epilogue) in
    p_all
      "cvft: every fused-epilogue GPU twin of the folded segment compiles, and matches the natural \
       form run twice"
      fused ~f:(function
      | _, `Ran (ok, _) -> ok
      | _, (`Declined | `Failed _) -> false);
    p_all "cvft: every compiling candidate of the folded segment matches the natural form run twice"
      results ~f:(function
      | _, `Ran (ok, _) -> ok
      | _, (`Declined | `Failed _) -> true);
    p_none "cvft: no candidate of the folded segment fails other than at companion coverage" results
      ~f:(function
      | _, `Failed _ -> true
      | _, (`Ran _ | `Declined) -> false);
    p_all "cvft: every compiling candidate of the folded segment is labeled tensorized" results
      ~f:(function
      | _, `Ran (_, t) -> t
      | _, (`Declined | `Failed _) -> true))
  else
    List.iter ~f:skipped
      [
        "cvft: the conv segment folds the conv output's per-cell zero, the output not materialized";
        "cvft: the folded conv+tail segment seeds fused-epilogue GPU twins";
        "cvft: every fused-epilogue GPU twin of the folded segment compiles, and matches the \
         natural form run twice";
        "cvft: every compiling candidate of the folded segment matches the natural form run twice";
        "cvft: no candidate of the folded segment fails other than at companion coverage";
        "cvft: every compiling candidate of the folded segment is labeled tensorized";
      ];
  (* --- Seeding + tuning on the conv-alone graph: the C backends seed serial + Grid per fission
     segment (and whole-routine); metal seeds the staged GPU flavor per fission segment and on the
     whole routine, whose zeroed site the pipeline expands (gh-ocannl-503). --- *)
  clean_cache "autotune_cache_conv_gpu";
  let cvu_dedup_label =
    "cvu: every per-segment GPU candidate dedups against its whole-routine twin (none timed)"
  in
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
      show_counts "cvu" r;
      (* metal: the whole-extent staged flavor plus its pd2 pipelined twin (gh-ocannl-487), on the
         whole routine (gh-ocannl-503) and per fission segment. *)
      p "cvu: conv seeds per fission segment (GPU staged on metal; serial+grid on cc)"
        (if has_uniform_f32_mma then
           r.Autotune.sketch_candidates = 2 && r.Autotune.fiss_sketch_candidates = 2
         else if on_cpu then
           r.Autotune.sketch_candidates = 2
           && r.Autotune.fiss_sketch_candidates = 2
           && r.Autotune.fiss_sketch_timed = 2
         else true);
      (* No per-segment candidate is timed: GPU fission folds the conv's zero into a per-cell
         companion of the one segment, and the whole-routine pipeline expands it into that same nest
         with that same geometry, so each per-segment candidate compiles to its whole-routine twin's
         digest and dedups. *)
      claim_dedup "cvu" r cvu_dedup_label (r.Autotune.fiss_sketch_timed = 0)
  | _ ->
      p "cvu: conv seeds per fission segment (GPU staged on metal; serial+grid on cc)" false;
      p cvu_dedup_label false);
  p_close "cvu: tuned conv matches the untuned twin within tolerance" got_u want8 ~tol:(fun _ ->
      1e-3);

  (* --- The GPU staged pipeline on a stride-2 conv (gh-ocannl-502): the compacting Stage packs the
     strided row densely through the cooperative shared tiles — the loads keep the stride, the tile
     store/read are unit-coefficient — so the tensorized micro-kernel applies unchanged. Input 17
     with k=3, s=2 keeps row = 8, oc = 16, ic = 8 (the 8x8x8 intrinsic tile divides). --- *)
  let make_conv8_s2 sub =
    let x =
      NTDSL.init ~l:(sub ^ "x") ~prec:Ir.Ops.single ~b:[ 2 ] ~o:[ 17; 17; 8 ]
        ~f:(Ll_test.weighted ~weights:[| 1; 1; 2; 3 |] ~modulus:7 ~offset:0. ~stride:1.)
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
  if has_f32_input_mma then (
    let x, kern, y = make_conv8_s2 "cvu2_g" in
    let kernel_extents = ref None in
    let conv_sched (site : Autotune.conv_site) _seg =
      kernel_extents := Some (site.Autotune.c_nrow, site.Autotune.c_noc, site.Autotune.c_nred);
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
    let got, census = run_fiss_sched "cvu2_gpu" y ~conv_sched in
    p_close "cvu2: GPU staged stride-2 conv (compacted) matches the natural form within tolerance"
      got want8s2 ~tol:(fun _ -> 1e-3);
    let rendered_label =
      "cvu2: the GPU staged stride-2 conv pipeline's routine is labeled tensorized"
    in
    match !kernel_extents with
    | Some (m, n, k) when not (advertised_tile_divides ~m ~n ~k) ->
        Stdio.eprintf
          "cvu2: the advertised f32-site tile does not divide the %dx%dx%d micro-kernel\n" m n k;
        skipped rendered_label
    | Some _ -> claim_tensorized rendered_label (rendered_tensorized census)
    | None -> p rendered_label false)
  else (
    skipped "cvu2: GPU staged stride-2 conv (compacted) matches the natural form within tolerance";
    skipped "cvu2: the GPU staged stride-2 conv pipeline's routine is labeled tensorized");

  (* --- The same stride-2 site through the seeding wave (gh-ocannl-502): the per-segment counts
     match the unit-stride [cvu] leg — the stride only changes the packing Stage's load arithmetic,
     not which flavors are proposable — and the tuned routine matches the untuned twin. --- *)
  clean_cache "autotune_cache_conv_s2";
  let cvu2_dedup_label =
    "cvu2: every per-segment GPU stride-2 candidate dedups against its whole-routine twin (none \
     timed)"
  in
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
      show_counts "cvu2" r;
      (* metal: the staged flavor plus its pd2 pipelined twin (gh-ocannl-487), on the whole routine
         and per segment, where each dedups against its whole-routine twin (see [cvu]). *)
      p "cvu2: stride-2 conv seeds per fission segment (GPU staged on metal; serial+grid on cc)"
        (if has_uniform_f32_mma then
           r.Autotune.sketch_candidates = 2 && r.Autotune.fiss_sketch_candidates = 2
         else if on_cpu then
           r.Autotune.sketch_candidates = 2
           && r.Autotune.fiss_sketch_candidates = 2
           && r.Autotune.fiss_sketch_timed = 2
         else true);
      claim_dedup "cvu2" r cvu2_dedup_label (r.Autotune.fiss_sketch_timed = 0)
  | _ ->
      p "cvu2: stride-2 conv seeds per fission segment (GPU staged on metal; serial+grid on cc)"
        false;
      p cvu2_dedup_label false);
  p_close "cvu2: tuned stride-2 conv matches the untuned twin within tolerance" got_u2 want8s2
    ~tol:(fun _ -> 1e-3);

  (* --- Row-block flavors on a strided row (gh-ocannl-502 x gh-ocannl-500 x gh-ocannl-485): an
     unzeroed fission segment proposes the cache-panel flavors too, and a row extent that does not
     divide the panel pads the row on top of the compaction — a stride-2 row of 12 with an 8-row
     panel. Every per-segment seed is executed against the natural form: the padded panel's fringe
     rows read the source past its extent, where [Stage]'s Where-form edge guards must supply the
     accumulation identity. --- *)
  let make_conv_s2_r12 sub =
    let x =
      NTDSL.init ~l:(sub ^ "x") ~prec:Ir.Ops.single ~b:[ 2 ] ~o:[ 25; 25; 4 ]
        ~f:(Ll_test.weighted ~weights:[| 1; 1; 2; 3 |] ~modulus:7 ~offset:0. ~stride:1.)
        ()
    in
    let kern = make_kern sub in
    let%op y =
      x +* "...| 2*oh<+kh, 2*ow<+kw, ..ic..; |kh, kw, ..ic.. -> ..oc.. => ...| oh, ow, ..oc.." kern
    in
    (x, kern, y)
  in
  if on_cpu then (
    let limits = narrowed_to_16 (Context.hardware_limits (Context.auto ())) in
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
      let got, census =
        run_fiss_sched tag y ~conv_sched:(fun site seg ->
            if site.Autotune.c_zeroed then [] else Autotune.sketch_schedule ~accum_prec ~p:p_ seg)
      in
      ( (not (Array.is_empty got))
        && Array.for_all2_exn got want12 ~f:(fun a b -> Float.(abs (a - b) < 1e-3)),
        rendered_tensorized census )
    in
    p_all "cvs2b: the strided non-dividing row seeds a padded row-panel flavor" !seeds ~f:(fun s ->
        List.exists !seeds ~f:(fun t -> t.Autotune.sk_bm = 8)
        && List.length !seeds = 3
        && s.Autotune.sk_conv);
    let results = List.mapi !seeds ~f:run_seed in
    p_all "cvs2b: every per-segment stride-2 candidate matches the natural form within tolerance"
      results ~f:fst;
    claim_tensorized "cvs2b: every per-segment stride-2 candidate's routine is labeled tensorized"
      (List.for_all results ~f:snd && not (List.is_empty results)))
  else (
    skipped "cvs2b: the strided non-dividing row seeds a padded row-panel flavor";
    skipped "cvs2b: every per-segment stride-2 candidate matches the natural form within tolerance";
    skipped "cvs2b: every per-segment stride-2 candidate's routine is labeled tensorized");

  (* === Blocked tile flavors (gh-ocannl-500): the GEMM row is split into [Grid] panels before the
     reorder — cache-blocked pool panels on the C backends, extra threadgroups per row-block on GPU
     — and the in-panel micro-kernel is tensorized. Row = 16 so an 8-row block gives two panels; oc
     = 16 and ic = 8 keep the whole-extent intrinsic-tile gates on the tensorized column/reduction.
     Dividing blocks only: the split stays guard-free, so the reorder's [Swap]s and (GPU)
     cooperative-load barriers are well-formed. === *)
  let make_x16 tag =
    NTDSL.init ~l:(tag ^ "x") ~prec:Ir.Ops.single ~b:[ 2 ] ~o:[ 18; 18; 8 ]
      ~f:(Ll_test.weighted ~weights:[| 1; 1; 2; 3 |] ~modulus:7 ~offset:0. ~stride:1.)
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
  let blocked_rendered_claim = "cvb: the blocked conv pipeline's routine is labeled tensorized" in
  if on_cpu then (
    let x, kern, y = make_conv16 "cvb_c" in
    let got, census =
      run_fiss_sched "cvb_cpu" y ~conv_sched:(blocked_sched ~shared:false ~simd_width:1 (x, kern))
    in
    p_close blocked_claim got want16 ~tol:(fun _ -> 1e-3);
    claim_tensorized blocked_rendered_claim (rendered_tensorized census))
  else if has_uniform_f32_mma then (
    let w =
      match (Context.hardware_limits (Context.auto ())).Ir.Backend_intf.mma with
      | Some m -> m.Ir.Backend_intf.mma_simd_width
      | None -> 32
    in
    let x, kern, y = make_conv16 "cvb_g" in
    let got, census =
      run_fiss_sched "cvb_gpu" y ~conv_sched:(blocked_sched ~shared:true ~simd_width:w (x, kern))
    in
    p_close blocked_claim got want16 ~tol:(fun _ -> 1e-3);
    claim_tensorized blocked_rendered_claim (rendered_tensorized census))
  else (
    skipped blocked_claim;
    skipped blocked_rendered_claim);

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
      show_counts "cvb" r;
      (* cc per-segment: serial + grid + one row-block panel (bm=8, 16/8=2 panels) = 3. metal
         per-segment: whole-extent staged + one threadgroup-block (bm=8), each with its pd2
         pipelined twin (gh-ocannl-487) = 4. *)
      p "cvb: blocked flavors seeded per fission segment"
        (if has_uniform_f32_mma then r.Autotune.fiss_sketch_candidates = 4
         else if on_cpu then r.Autotune.fiss_sketch_candidates = 3
         else true)
  | _ -> p "cvb: blocked flavors seeded per fission segment" false);
  p_close "cvb: tuned blocked conv matches the untuned twin within tolerance" got_b want16
    ~tol:(fun _ -> 1e-3);
  (* The same site whole-routine (gh-ocannl-503, see [cvz]): the dividing 8-row block flavor and the
     pipelined twins, with the expanded zero's row loop split into the same [Grid] blocks. *)
  if Lazy.force gpu_capable then (
    let zeroed, seeds, results =
      whole_routine_gpu_leg ~tag:"cvzb" ~want:want16 ~build:(fun sub ->
          let _, _, y = make_conv16 sub in
          y)
    in
    p "cvzb: the whole-routine blocked-shape conv site is zeroed in its lowering" zeroed;
    (* An 8-row block is a flavor only where it is a whole number of the advertised tile's rows
       (Metal's 8x8x8; CUDA's 16-row tf32 tile cannot split this 16-row site into two blocks). *)
    let label = "cvzb: the whole-routine zeroed site seeds a GPU row-block flavor" in
    if advertised_tile_divides ~m:8 ~n:16 ~k:8 then
      p_exists label seeds ~f:(fun s ->
          s.Autotune.sk_conv && s.Autotune.sk_gpu && s.Autotune.sk_bm > 0)
    else skipped label;
    p_all
      "cvzb: every whole-routine zeroed GPU candidate compiles, and matches the natural form run \
       twice"
      results ~f:(function
      | `Ran (ok, _) -> ok
      | `Declined -> false);
    p_all "cvzb: every whole-routine zeroed GPU candidate's routine is labeled tensorized" results
      ~f:(ran_ok snd))
  else
    List.iter ~f:skipped
      [
        "cvzb: the whole-routine blocked-shape conv site is zeroed in its lowering";
        "cvzb: the whole-routine zeroed site seeds a GPU row-block flavor";
        "cvzb: every whole-routine zeroed GPU candidate compiles, and matches the natural form run \
         twice";
        "cvzb: every whole-routine zeroed GPU candidate's routine is labeled tensorized";
      ];

  (* === Blocked flavor on an aligned-merged segment (gh-ocannl-500): the realistic convnet shape,
     conv + materialized companions. The whole-segment Grid geometry comes from the aligned
     cross-nest analysis ([Sched.default_cpu]) and the row is blocked SERIALLY for cache residency
     within each pool chunk — so the block flavor fires on fused conv+bias/relu/pool segments, not
     only conv-alone. Row = 16 (two 8-row panels). The hand-built pipeline here is the CPU one; the
     GPU staged leg seeds this segment too since its companion-coverage gate (gh-ocannl-1406), and
     conv_merged_gpu_seeds executes every such GPU seed. === *)
  let make_merged16 sub =
    let x = make_x16 sub in
    let kern = make_kern8 sub in
    let%op pr =
      x +* "...| 1*oh<+kh, 1*ow<+kw, ..ic..; |kh, kw, ..ic.. -> ..oc.. => ...| oh, ow, ..oc.." kern
    in
    Ir.Tnode.update_memory_mode pr.Tensor.value Ir.Tnode.On_device (Site "99:test-setup");
    let%op y2 = relu pr in
    Ir.Tnode.update_memory_mode y2.Tensor.value Ir.Tnode.On_device (Site "99:test-setup");
    let%op y3 = y2 *. 0.5 in
    (x, kern, y3)
  in
  let want_mb =
    let _, _, y = make_merged16 "cvmb_r" in
    run_plain "cvmb_ref" y
  in
  if on_cpu then (
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
    let got, census = run_fiss_sched "cvmb_cpu" y ~conv_sched in
    p_close "cvmb: merged-segment blocked conv pipeline matches the natural form within tolerance"
      got want_mb ~tol:(fun _ -> 1e-3);
    claim_tensorized
      "cvmb: the merged-segment blocked conv pipeline's routine is labeled tensorized"
      (rendered_tensorized census))
  else (
    skipped "cvmb: merged-segment blocked conv pipeline matches the natural form within tolerance";
    skipped "cvmb: the merged-segment blocked conv pipeline's routine is labeled tensorized");
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
      show_counts "cvmb" r;
      (* cc merged fission segment: serial + aligned-grid + one serial row-block panel, each with
         its fused-epilogue twin (gh-ocannl-501) = 6, all compiling (the aligned twins omit the
         preset [Retype] on the consumed tail nest — fuse-before-annotate). A GPU with the f32
         capability seeds the staged flavors on the merged segment, its companions covered
         (gh-ocannl-1406; every such seed executed in conv_merged_gpu_seeds). *)
      p "cvmb: blocked flavor seeded on the merged fission segment"
        (if on_cpu then r.Autotune.fiss_sketch_candidates = 6 && r.Autotune.fiss_sketch_timed = 6
         else if Lazy.force gpu_capable then r.Autotune.fiss_sketch_candidates > 0
         else true)
  | _ -> p "cvmb: blocked flavor seeded on the merged fission segment" false);
  p_close "cvmb: tuned merged-blocked conv matches the untuned twin within tolerance" got_mb want_mb
    ~tol:(fun _ -> 1e-3);

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
    Ir.Tnode.update_memory_mode pr.Tensor.value Ir.Tnode.On_device (Site "99:test-setup");
    let%op y2 = relu pr in
    Ir.Tnode.update_memory_mode y2.Tensor.value Ir.Tnode.On_device (Site "99:test-setup");
    let%op y3 = y2 *. 0.5 in
    (x, kern, y3)
  in
  let want_m =
    let _, _, y = make_merged "cva_r" in
    run_plain "cva_ref" y
  in
  if on_cpu then (
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
    let got, census = run_fiss_sched "cva_gemm" y ~conv_sched in
    p_close "cva: aligned-grid conv pipeline on a merged segment matches within tolerance" got
      want_m ~tol:(fun _ -> 1e-3);
    claim_tensorized "cva: the aligned-grid conv pipeline's routine is labeled tensorized"
      (rendered_tensorized census))
  else (
    skipped "cva: aligned-grid conv pipeline on a merged segment matches within tolerance";
    skipped "cva: the aligned-grid conv pipeline's routine is labeled tensorized");
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
      show_counts "cva" r;
      (* cc: the whole routine seeds the serial flavor and its fused twin (the routine carries the
         conv's [Zero_out], so the aligned analysis bails there and no Grid flavor is seeded); per
         fission segment serial + aligned-grid, each with its fused-epilogue twin, all four timed
         (the aligned twin compiles by fuse-before-annotate, gh-ocannl-501). A GPU with the f32
         capability seeds the staged flavors on the whole routine and the merged segment, whose
         companions take the site's geometry (gh-ocannl-1406); one without seeds nothing. *)
      p "cva: aligned-grid conv seeded and timed on the merged fission segment"
        (if on_cpu then
           r.Autotune.sketch_candidates = 2
           && r.Autotune.epilogue_sketch_candidates = 1
           && r.Autotune.fiss_sketch_candidates = 4
           && r.Autotune.fiss_sketch_timed = 4
         else if Lazy.force gpu_capable then
           r.Autotune.sketch_candidates > 0 && r.Autotune.fiss_sketch_candidates > 0
         else r.Autotune.sketch_candidates = 0 && r.Autotune.fiss_sketch_candidates = 0)
  | _ -> p "cva: aligned-grid conv seeded and timed on the merged fission segment" false);
  p_close "cva: tuned merged conv graph matches the untuned twin within tolerance" got_m want_m
    ~tol:(fun _ -> 1e-3)

(* === The CPU conv seeds' register-tile twins (gh-ocannl-697): every CPU conv leaf whose
   micro-kernel (the GEMM row by the out-channel column) admits other C-tiles than the renderer's
   own is twinned with them, as the matmul family's CPU tensorized leaves are (gh-ocannl-619). Each
   twin seeded under the machine's own limits is executed: it renders the requested geometry, named
   in the emitted header with its provenance, and matches its "auto" sibling bitwise -- a C-tile
   geometry reorders no element's reduction. 64 out-channels give the micro-kernel eight to sixteen
   vector columns, where the register file affords alternatives at every width. === *)
let () =
  let tiled_label = "cvr: every register-tile twin renders its geometry from the schedule" in
  let bitwise_label = "cvr: every register-tile twin matches its auto sibling bitwise" in
  if not on_cpu then (
    skipped tiled_label;
    skipped bitwise_label)
  else
    let make tag =
      let x =
        NTDSL.init ~l:(tag ^ "x") ~prec:Ir.Ops.single ~b:[ 2 ] ~o:[ 18; 18; 8 ]
          ~f:(Ll_test.weighted ~weights:[| 1; 1; 2; 3 |] ~modulus:7 ~offset:0. ~stride:1.)
          ()
      in
      let kern =
        NTDSL.init ~l:(tag ^ "k") ~prec:Ir.Ops.single ~i:[ 3; 3; 8 ] ~o:[ 64 ]
          ~f:(Ll_test.weighted ~weights:[| 2; 1; 1; 3 |] ~modulus:5 ~offset:(-2.) ~stride:1.)
          ()
      in
      let%op y =
        x +* "...| oh<+kh, ow<+kw, ..ic..; |kh, kw, ..ic.. -> ..oc.. => ...| oh, ow, ..oc.." kern
      in
      y
    in
    let limits = Context.hardware_limits (Context.auto ()) in
    let seeds = ref [] in
    ignore
      (Context.compile
         ~lowered_transform:(fun o ->
           seeds := Autotune.sketch_seed_params ~is_gpu:false ~is_cpu:true ~limits o;
           [ o ])
         (Context.auto ())
         (named "cvr_probe" (Train.forward (make "cvr_p")))
         Ir.Indexing.Empty);
    let run name p_ =
      let y = make name in
      let ctx, routine =
        Context.compile
          ~lowered_transform:(fun o ->
            [ Sched.apply (Autotune.sketch_schedule ~accum_prec ~p:p_ o) o ])
          (Context.auto ())
          (named name (Train.forward y))
          Ir.Indexing.Empty
      in
      let ctx = Context.run ctx routine in
      Context.get_values ctx y.Tensor.value
    in
    let tiled = List.filter !seeds ~f:(fun s -> Option.is_some s.Autotune.sk_tile) in
    Stdio.eprintf "cvr: %d CPU conv seeds, %d register-tile twins (not part of the golden)\n"
      (List.length !seeds) (List.length tiled);
    if List.is_empty tiled then (
      (* A vector file affording no alternative at this micro-kernel proposes no twin. *)
      skipped tiled_label;
      skipped bitwise_label)
    else
      let results =
        List.mapi tiled ~f:(fun i s ->
            let t = Option.value_exn s.Autotune.sk_tile in
            let name = Printf.sprintf "cvr_t%d" i in
            let got = run name s in
            let src = Generated.read name in
            let header =
              Printf.sprintf "Tile_mma register tiling: %dx%d C-tile of %d-lane"
                t.Ir.Register_tile.rm t.Ir.Register_tile.rn t.Ir.Register_tile.lanes
            in
            let rendered =
              String.is_substring src ~substring:header
              && String.is_substring src ~substring:"geometry from the schedule"
            in
            let auto = run (Printf.sprintf "cvr_a%d" i) { s with Autotune.sk_tile = None } in
            (rendered, Array.equal Float.equal (nonzero name got) auto))
      in
      p_all tiled_label results ~f:fst;
      p_all bitwise_label results ~f:snd
