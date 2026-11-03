(* gh-ocannl-1406: GPU conv seeds on aligned-merged lowerings, executed.

   The GPU staged conv leg used to refuse any lowering with more than two real statements besides
   the site's own zero ([real_stmts <= 2]), although [gpu_conv_sketch_schedule] gives every
   materialized companion the site's Grid and lane geometry through [companion_geometry]. On a
   convnet the conv's materialized companions (bias, relu, pooling) merge into the conv's segment,
   so the cap refused exactly the segments real graphs produce. The leg now asks the builder's own
   coverage verdict ([Sketch_families.conv_coverage_witness]), as the matmul family does.

   Pinned here, on the device's own limits: the conv+relu+scale graph (two materialized companions)
   at two row extents — 8, where the whole-extent flavors seed, and 16, where the row-block flavors
   do too — has three real statements both on its whole-routine lowering and on the tuner's merged
   GPU fission segment, the GPU leg seeds both, and EVERY seed is compiled and executed twice
   against the unscheduled natural form (candidates are timed, not value-checked, so an unsound seed
   would be crowned silently; the second run catches a zero that missed cells). The operands are
   small integers, so every sum is exact in f32 and in tf32 whatever the reassociation: the parity
   is bitwise. Gated on the seeder's own capability judgment for an f32 site, never on the seed
   list.

   gh-ocannl-1598: two more legs at 64 out-channels, wider than the 32-lane workgroup the companions
   are distributed over. Where the row pads to the intrinsic row tile ([0x0x0/<tm>x0]: 8 rows to
   CUDA tf32's 16 at [cm8w], 6 rows to every mma backend's tile at [cm6w], which claims the flavor
   is seeded), the accumulator fragment is workgroup-shared and lane 0 alone stores it back; the
   unfused seed's relu companion then read [pr] from every lane before that store landed — 512 of
   8192 cells wrong on CUDA (rog-nv-linux) while the fused twin, whose tail runs inside lane 0's
   store-back, matched, as on cifar_resnet's stage-3 convs. The [-tf32] rule runs every claim with
   [tf32_matmuls] on, which opens CUDA's tf32 tile to these f32 sites; at the default the CUDA run
   skips them (a no-op elsewhere, where the rule repeats the run). *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
open Verdict.Claims
module LL = Ir.Low_level
module Sched = Ir.Schedule
module Asgns = Ir.Assignments

let accum_prec = Ll_test.backend_accum_prec
let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"")
let on_cpu = Ir.Backend_intf.renders_on_host_pool (Context.backend_facts_of_name backend_name)
let limits = Context.hardware_limits (Context.auto ())

let tensorized_capability =
  lazy
    (Ll_test.tensorized_capability ~is_gpu:(not on_cpu) ~is_cpu:on_cpu ~limits ~a_prec:Ir.Ops.single
       ~b_prec:Ir.Ops.single ~d_prec:Ir.Ops.single)

(* The seeder's own judgment that a GPU conv leg can seed an f32 site on this backend. *)
let gpu_capable =
  lazy
    ((not on_cpu)
    && match Lazy.force tensorized_capability with `Advertised -> true | `Withheld _ -> false)

let named name (comp : Asgns.comp) : Asgns.comp =
  { comp with asgns = Asgns.Block_comment (name, comp.asgns) }

let scratch (o : LL.optimized) =
  {
    o with
    LL.traced_store = Hashtbl.copy o.LL.traced_store;
    LL.optimize_ctx = LL.copy_optimize_ctx o.LL.optimize_ctx;
  }

(* conv (row x row output, 8 in-channels, [oc] out-channels, 3x3 window) + relu + scale, with the
   conv output and the relu materialized like real convnet activations. *)
let make_merged ?(oc = 16) ~rows tag =
  let x =
    NTDSL.init ~l:(tag ^ "x") ~prec:Ir.Ops.single ~b:[ 2 ]
      ~o:[ rows + 2; rows + 2; 8 ]
      ~f:(Ll_test.weighted ~weights:[| 1; 1; 2; 3 |] ~modulus:7 ~offset:0. ~stride:1.)
      ()
  in
  let kern =
    NTDSL.init ~l:(tag ^ "k") ~prec:Ir.Ops.single ~i:[ 3; 3; 8 ] ~o:[ oc ]
      ~f:(Ll_test.weighted ~weights:[| 2; 1; 1; 3 |] ~modulus:5 ~offset:(-2.) ~stride:1.)
      ()
  in
  let%op pr =
    x +* "...| 1*oh<+kh, 1*ow<+kw, ..ic..; |kh, kw, ..ic.. -> ..oc.. => ...| oh, ow, ..oc.." kern
  in
  Ir.Tnode.update_memory_mode pr.Tensor.value Ir.Tnode.On_device (Site "99:test-setup");
  let%op y2 = relu pr in
  Ir.Tnode.update_memory_mode y2.Tensor.value Ir.Tnode.On_device (Site "99:test-setup");
  let%op y3 = y2 *. 0.5 in
  y3

let gpu_seeds ?(limits = limits) seg =
  Autotune.sketch_seed_params ~is_gpu:true ~is_cpu:false ~limits seg

let gpu_real_stmts ?(limits = limits) seg =
  match fst (Autotune.family_seeding ~is_gpu:true ~is_cpu:false ~limits seg) with
  | Autotune.Seeding_conv { real_stmts; _ } -> Some real_stmts
  | Autotune.Seeding_matmul _ | Autotune.Seeding_none -> None

let is_conv_segment (seg : LL.optimized) = Option.is_some (Autotune.detect_conv seg.LL.llc)

(* The routine's whole lowering and its merged conv segment -- the tuner's own segmentation
   ([Ll_test.tuner_segments], gh-ocannl-1483) -- captured on the way through a compile. *)
let lowerings ?target ~name y =
  let captured = ref None in
  ignore
    (Context.compile
       ~lowered_transform:(fun opt ->
         captured := Some (scratch opt);
         [ opt ])
       (Context.auto ())
       (named name (Train.forward y))
       Ir.Indexing.Empty);
  let whole = Option.value_exn !captured in
  let segment =
    List.find_map
      (snd (Ll_test.tuner_segments ?target whole))
      ~f:(fun (_, pre, _, _) -> Option.some_if (is_conv_segment pre) pre)
  in
  (whole, segment)

let run ~name ~transform y =
  let ctx, routine =
    Context.compile ~lowered_transform:transform (Context.auto ())
      (named name (Train.forward y))
      Ir.Indexing.Empty
  in
  let ctx = Context.run ctx routine in
  let ctx = Context.run ctx routine in
  (Context.get_values ctx y.Tensor.value, routine.Context.mma)

(* One seed's executed verdict: [`Ran (matches, tensorized)] or [`Failed message]. [applied] guards
   against a segment transform that never reached the conv segment. *)
let execute_seed ~name ~want ~transform y =
  match run ~name ~transform y with
  | got, census ->
      let matches =
        (not (Array.is_empty got))
        && Array.length got = Array.length want
        && Array.for_all2_exn got want ~f:Float.equal
      in
      if not matches then
        Stdio.eprintf "%s: values differ from the natural form (%s of %d cells)\n" name
          (if Array.length got = Array.length want then
             Int.to_string (Array.counti got ~f:(fun i g -> not (Float.equal g want.(i))))
           else "shape mismatch, all")
          (Array.length want);
      `Ran (matches, Ll_test.rendered_tensorized census)
  | exception exn ->
      Stdio.eprintf "%s: failed (not part of the golden): %s\n" name (Exn.to_string exn);
      `Failed

(* One leg's claims, made in [leg]: open where the GPU conv leg can seed an f32 site, skipped by
   name elsewhere in the order this code makes them (gh-ocannl-1703). Everything they rest on is
   computed lazily, only where [leg] is open. *)
let merged_leg ~leg ?oc ?(row_masked = false) ~tag ~rows () =
  let make_merged ~rows n = make_merged ?oc ~rows n in
  let real_stmts opt = Option.value (gpu_real_stmts opt) ~default:0 in
  let setup =
    lazy
      (let want =
         let y = make_merged ~rows (tag ^ "r") in
         fst (run ~name:(tag ^ "_ref") ~transform:(fun opt -> [ opt ]) y)
       in
       if not (Array.exists want ~f:(fun v -> Float.(v <> 0.))) then
         failwith (tag ^ ": the reference is all zeros — the parity claims would be vacuous");
       let whole, segment = lowerings ~name:(tag ^ "_probe") (make_merged ~rows (tag ^ "p")) in
       let whole_seeds = gpu_seeds whole in
       let seg_seeds = Option.value_map segment ~default:[] ~f:gpu_seeds in
       Stdio.eprintf
         "%s: whole routine %d real statements, %d GPU seeds; merged segment %d real statements, \
          %d GPU seeds (not part of the golden)\n"
         tag (real_stmts whole) (List.length whole_seeds)
         (Option.value_map segment ~default:0 ~f:real_stmts)
         (List.length seg_seeds);
       (want, whole, segment, whole_seeds, seg_seeds))
  in
  in_leg ~leg
    (tag ^ ": the whole routine and its merged conv segment carry three or more real statements")
    (fun label ->
      let _, whole, segment, _, _ = Lazy.force setup in
      p label (real_stmts whole >= 3 && Option.exists segment ~f:(fun s -> real_stmts s >= 3)));
  in_leg ~leg (tag ^ ": the GPU leg seeds the whole routine") (fun label ->
      let _, _, _, whole_seeds, _ = Lazy.force setup in
      p label (not (List.is_empty whole_seeds)));
  in_leg ~leg (tag ^ ": the GPU leg seeds the merged segment") (fun label ->
      let _, _, _, _, seg_seeds = Lazy.force setup in
      p label (not (List.is_empty seg_seeds)));
  (* The pad-masked whole-extent flavor (row padded to the intrinsic row tile, [0x0x0/<tm>x0]): its
     accumulator fragment is workgroup-shared, transferred by lane 0 alone. *)
  if row_masked then
    in_leg ~leg (tag ^ ": a whole-routine GPU seed pads the row to the mma tile") (fun label ->
        let _, _, _, whole_seeds, _ = Lazy.force setup in
        p_exists label whole_seeds ~f:(fun sp -> sp.Autotune.sk_bm = 0 && sp.Autotune.sk_tm > 0));
  let results =
    lazy
      (let want, _, _, whole_seeds, seg_seeds = Lazy.force setup in
       let whole_results =
         List.mapi whole_seeds ~f:(fun i sp ->
             let name = Printf.sprintf "%s_w%d" tag i in
             Stdio.eprintf "%s = %s (not part of the golden)\n%!" name
               (Autotune.sketch_params_label sp);
             execute_seed ~name ~want
               ~transform:(fun opt ->
                 [ Sched.apply (Autotune.sketch_schedule ~accum_prec ~p:sp opt) opt ])
               (make_merged ~rows name))
       in
       let seg_results =
         List.mapi seg_seeds ~f:(fun i sp ->
             let name = Printf.sprintf "%s_s%d" tag i in
             Stdio.eprintf "%s = %s (not part of the golden)\n%!" name
               (Autotune.sketch_params_label sp);
             let applied = ref false in
             (* Compiled as a per-segment sketch candidate: the tuner's segmentation replayed, the
                conv segment sketched, every other one on its preset. *)
             let schedule _ seg =
               if is_conv_segment seg then
                 match List.nth (gpu_seeds seg) i with
                 | Some sp ->
                     applied := true;
                     Some (Autotune.sketch_schedule ~accum_prec ~p:sp seg)
                 | None -> failwith (Printf.sprintf "%s: seed %d is gone from the segment" name i)
               else None
             in
             let transform = Ll_test.tuner_transform ~schedule in
             match execute_seed ~name ~want ~transform (make_merged ~rows name) with
             | `Ran _ when not !applied ->
                 Stdio.eprintf "%s: the seed never reached the conv segment\n" name;
                 `Failed
             | r -> r)
       in
       (whole_results, seg_results))
  in
  let ok = function `Ran (m, _) -> m | `Failed -> false in
  let tensorized = function `Ran (_, t) -> t | `Failed -> false in
  in_leg ~leg
    (tag ^ ": every whole-routine GPU seed compiles and matches the natural form exactly, run twice")
    (fun label ->
      let whole_results, _ = Lazy.force results in
      p_all label whole_results ~f:ok);
  in_leg ~leg
    (tag
   ^ ": every merged-segment GPU seed compiles and matches the natural form exactly, run twice")
    (fun label ->
      let _, seg_results = Lazy.force results in
      p_all label seg_results ~f:ok);
  in_leg ~leg (tag ^ ": every GPU seed's routine is labeled tensorized") (fun label ->
      let whole_results, seg_results = Lazy.force results in
      p_all label (whole_results @ seg_results) ~f:tensorized)

(* The structural leg (gh-ocannl-1626): per-PR CI runs cc, where every leg above is skipped. The
   real-statement counts, the seeds and the merged segment are the seeder's and the tuner's,
   computed on the host, so they are computed here on every backend under a synthetic GPU target
   whose matrix unit is an f32 8x8x8 tile -- the tuner's own pipeline ([Ll_test.tuner_segments]) and
   the GPU seeder at those limits, on the lowering this host captures. The control for the
   row-padded flavor: at 8 rows, which the tile divides, no whole-routine seed pads the row. *)
let () =
  let synthetic_limits = Ll_test.f32_mma_limits in
  let target = Ll_test.synthetic_gpu ~limits:synthetic_limits () in
  let structural ?oc ~tag ~rows () =
    let whole, segment =
      lowerings ~target ~name:(tag ^ "_sprobe") (make_merged ?oc ~rows (tag ^ "s"))
    in
    let tag = tag ^ " (structural)" in
    let real_stmts opt = Option.value (gpu_real_stmts ~limits:synthetic_limits opt) ~default:0 in
    let whole_seeds = gpu_seeds ~limits:synthetic_limits whole in
    p
      (tag ^ ": the whole routine and its merged conv segment carry three or more real statements")
      (real_stmts whole >= 3 && Option.exists segment ~f:(fun s -> real_stmts s >= 3));
    p (tag ^ ": the GPU leg seeds the whole routine") (not (List.is_empty whole_seeds));
    p
      (tag ^ ": the GPU leg seeds the merged segment")
      (Option.exists segment ~f:(fun s ->
           not (List.is_empty (gpu_seeds ~limits:synthetic_limits s))));
    whole_seeds
  in
  let pads (sp : Autotune.sketch_params) = sp.sk_bm = 0 && sp.sk_tm > 0 in
  p_none "cm8 (structural) control: no whole-routine GPU seed pads the row the tile divides"
    (structural ~tag:"cm8" ~rows:8 ())
    ~f:pads;
  ignore (structural ~tag:"cm16" ~rows:16 () : Autotune.sketch_params list);
  ignore (structural ~tag:"cm8w" ~rows:8 ~oc:64 () : Autotune.sketch_params list);
  p_exists "cm6w (structural): a whole-routine GPU seed pads the row to the mma tile"
    (structural ~tag:"cm6w" ~rows:6 ~oc:64 ())
    ~f:pads

let () =
  (* A GPU withholding the capability says which aggregation lifts it (tf32 off on CUDA). *)
  let leg =
    let gpu_capable = Lazy.force gpu_capable in
    let aggregation =
      match Lazy.force tensorized_capability with
      | `Withheld aggregation when not on_cpu -> aggregation
      | `Withheld _ | `Advertised -> `Backend
    in
    Verdict.leg ~aggregation ~when_:gpu_capable ~on:backend_name ()
  in
  merged_leg ~leg ~tag:"cm8" ~rows:8 ();
  merged_leg ~leg ~tag:"cm16" ~rows:16 ();
  merged_leg ~leg ~tag:"cm8w" ~rows:8 ~oc:64 ();
  merged_leg ~leg ~tag:"cm6w" ~rows:6 ~oc:64 ~row_masked:true ()
