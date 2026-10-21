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
   list. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
open Verdict.Claims
module LL = Ir.Low_level
module Sched = Ir.Schedule
module Asgns = Ir.Assignments

let accum_prec = Ll_test.backend_accum_prec
let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")
let on_cpu = Sched.backend_is_cpu backend_name
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

(* conv (row x row output, 8 in-channels, 16 out-channels, 3x3 window) + relu + scale, with the conv
   output and the relu materialized like real convnet activations. *)
let make_merged ~rows tag =
  let x =
    NTDSL.init ~l:(tag ^ "x") ~prec:Ir.Ops.single ~b:[ 2 ]
      ~o:[ rows + 2; rows + 2; 8 ]
      ~f:(Ll_test.weighted ~weights:[| 1; 1; 2; 3 |] ~modulus:7 ~offset:0. ~stride:1.)
      ()
  in
  let kern =
    NTDSL.init ~l:(tag ^ "k") ~prec:Ir.Ops.single ~i:[ 3; 3; 8 ] ~o:[ 16 ]
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

(* The tuner's GPU fission settings ([Autotune.tune]'s segment enumeration). *)
let fission ~preset opt =
  Sched.fission_scheduled ~promote_locals:true ~fold_zeros:true
    ?keep_mapping:(Sched.fission_keep_mapping ~is_gpu:true ~limits)
    ~preset ~zero_sched:(Sched.zero_expansion ~limits) ~static_indices:[] opt

let gpu_seeds seg = Autotune.sketch_seed_params ~is_gpu:true ~is_cpu:false ~limits seg

let gpu_judgment seg =
  match fst (Autotune.family_seeding ~is_gpu:true ~is_cpu:false ~limits seg) with
  | Autotune.Seeding_conv { judgment; _ } -> Some judgment
  | Autotune.Seeding_matmul _ | Autotune.Seeding_none -> None

let is_conv_segment (seg : LL.optimized) = Option.is_some (Autotune.detect_conv seg.LL.llc)

(* The routine's whole lowering and its merged conv segment, captured on the way through a
   compile. *)
let lowerings ~name y =
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
      (fission ~preset:(fun seg -> Sched.default_gpu ~min_parallel:1 ~limits seg) (scratch whole))
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
      if not matches then Stdio.eprintf "%s: values differ from the natural form\n" name;
      `Ran (matches, Ll_test.rendered_tensorized census)
  | exception exn ->
      Stdio.eprintf "%s: failed (not part of the golden): %s\n" name (Exn.to_string exn);
      `Failed

let leg ~tag ~rows =
  let want =
    let y = make_merged ~rows (tag ^ "r") in
    fst (run ~name:(tag ^ "_ref") ~transform:(fun opt -> [ opt ]) y)
  in
  if not (Array.exists want ~f:(fun v -> Float.(v <> 0.))) then
    failwith (tag ^ ": the reference is all zeros — the parity claims would be vacuous");
  let whole, segment = lowerings ~name:(tag ^ "_probe") (make_merged ~rows (tag ^ "p")) in
  let real_stmts opt =
    Option.value_map (gpu_judgment opt) ~default:0 ~f:(fun j -> j.Autotune.cj_real_stmts)
  in
  let whole_seeds = gpu_seeds whole in
  let seg_seeds = Option.value_map segment ~default:[] ~f:gpu_seeds in
  Stdio.eprintf
    "%s: whole routine %d real statements, %d GPU seeds; merged segment %d real statements, %d GPU \
     seeds (not part of the golden)\n"
    tag (real_stmts whole) (List.length whole_seeds)
    (Option.value_map segment ~default:0 ~f:real_stmts)
    (List.length seg_seeds);
  p
    (tag ^ ": the whole routine and its merged conv segment carry three or more real statements")
    (real_stmts whole >= 3 && Option.exists segment ~f:(fun s -> real_stmts s >= 3));
  p (tag ^ ": the GPU leg seeds the whole routine") (not (List.is_empty whole_seeds));
  p (tag ^ ": the GPU leg seeds the merged segment") (not (List.is_empty seg_seeds));
  let whole_results =
    List.mapi whole_seeds ~f:(fun i sp ->
        let name = Printf.sprintf "%s_w%d" tag i in
        execute_seed ~name ~want
          ~transform:(fun opt ->
            [ Sched.apply (Autotune.sketch_schedule ~accum_prec ~p:sp opt) opt ])
          (make_merged ~rows name))
  in
  let seg_results =
    List.mapi seg_seeds ~f:(fun i _ ->
        let name = Printf.sprintf "%s_s%d" tag i in
        let applied = ref false in
        let transform opt =
          let preset seg =
            if is_conv_segment seg then
              match List.nth (gpu_seeds seg) i with
              | Some sp ->
                  applied := true;
                  Autotune.sketch_schedule ~accum_prec ~p:sp seg
              | None -> failwith (Printf.sprintf "%s: seed %d is gone from the segment" name i)
            else Sched.default_gpu ~min_parallel:1 ~limits seg
          in
          List.map (fission ~preset opt) ~f:(fun (_, _, _, post) -> post)
        in
        match execute_seed ~name ~want ~transform (make_merged ~rows name) with
        | `Ran _ when not !applied ->
            Stdio.eprintf "%s: the seed never reached the conv segment\n" name;
            `Failed
        | r -> r)
  in
  let ok = function `Ran (m, _) -> m | `Failed -> false in
  let tensorized = function `Ran (_, t) -> t | `Failed -> false in
  p_all
    (tag ^ ": every whole-routine GPU seed compiles and matches the natural form exactly, run twice")
    whole_results ~f:ok;
  p_all
    (tag
   ^ ": every merged-segment GPU seed compiles and matches the natural form exactly, run twice")
    seg_results ~f:ok;
  p_all
    (tag ^ ": every GPU seed's routine is labeled tensorized")
    (whole_results @ seg_results) ~f:tensorized

let () =
  if Lazy.force gpu_capable then (
    leg ~tag:"cm8" ~rows:8;
    leg ~tag:"cm16" ~rows:16)
  else
    (* A GPU withholding the capability says which aggregation lifts it (tf32 off on CUDA). *)
    let aggregation =
      match Lazy.force tensorized_capability with
      | `Withheld aggregation when not on_cpu -> aggregation
      | `Withheld _ | `Advertised -> `Backend
    in
    List.iter [ "cm8"; "cm16" ] ~f:(fun tag ->
        List.iter
          ~f:(Verdict.skipped ~aggregation ~backend:backend_name)
          [
            tag
            ^ ": the whole routine and its merged conv segment carry three or more real statements";
            tag ^ ": the GPU leg seeds the whole routine";
            tag ^ ": the GPU leg seeds the merged segment";
            tag
            ^ ": every whole-routine GPU seed compiles and matches the natural form exactly, run \
               twice";
            tag
            ^ ": every merged-segment GPU seed compiles and matches the natural form exactly, run \
               twice";
            tag ^ ": every GPU seed's routine is labeled tensorized";
          ])
