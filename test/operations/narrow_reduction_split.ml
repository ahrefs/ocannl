(* gh-ocannl-2505: the GPU preset leaves a 32-channel convolution bias gradient serial at its
   64-cell mapping threshold. The untuned Metal plan splits its long batch and spatial reduction
   into a mapped partials pass and a separate combine. *)

(* cc-fp-contract: same-shape -- cc uses the ordinary pipeline with either split setting *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
open Verdict.Claims
module BI = Ir.Backend_intf
module LL = Ir.Low_level
module Sched = Ir.Schedule

let batch = 64
let channels = 32
let blocks = 32
let f32 x = Int32.float_of_bits (Int32.bits_of_float x)

let fixture ~name ~height ~width ~channels =
  let n = batch * height * width * channels in
  let x = Array.init n ~f:(fun i -> if 11 * i % 37 < 19 then 1. else -1.) in
  let dy = Array.init n ~f:(fun i -> Float.of_int ((389 * i % 1009) - 504) /. 1009.) in
  let input label values =
    NTDSL.ndarray values
      ~label:[ name ^ label ]
      ~batch_dims:[ batch; height; width ] ~output_dims:[ channels ] ()
  in
  let b =
    Operation.init ~l:(name ^ "_bias") ~prec:Ir.Ops.single ~o:[ channels ]
      ~f:(fun _ -> 0.)
      ~grad_spec:Tensor.Require_grad ()
  in
  let x_t = input "_x" x and dy_t = input "_dy" dy in
  let%op loss = (relu (x_t + b) *. dy_t) ++ "... => 0" in
  let grad = (Option.value_exn b.Tensor.diff).Tensor.grad in
  Train.set_materialized grad;
  let want = Array.create ~len:channels 0. in
  Array.iteri dy ~f:(fun i v ->
      if Float.is_positive x.(i) then
        let c = i % channels in
        want.(c) <- want.(c) +. f32 v);
  (grad, Train.grad_update loss, want)

let writes tn (seg : LL.optimized) = Set.mem (Sched.written_nodes seg.llc) tn

let launch_threads (seg : LL.optimized) =
  let dims = LL.launch_dims seg.llc in
  Array.fold dims.grid ~init:(LL.launch_block_threads dims) ~f:( * )

let pipeline opt = Sched.maybe_default_schedules ~facts:BI.metal_facts ~static_indices:[] opt

let has_partials grad segments =
  let partials tn = List.equal String.equal tn.Ir.Tnode.label ("partials" :: grad.Ir.Tnode.label) in
  let rec precedes_combine = function
    | [] -> false
    | (seg : LL.optimized) :: rest ->
        launch_threads seg = channels * blocks
        && Option.exists
             (Set.find (Sched.written_nodes seg.llc) ~f:partials)
             ~f:(fun partials ->
               List.exists rest ~f:(fun seg ->
                   writes grad seg
                   && List.exists (LL.evaluation_reads_of_proc ~blends:false ~self:grad seg.llc)
                        ~f:(fun reads -> Set.mem reads partials)))
        || precedes_combine rest
  in
  precedes_combine segments

let narrow = fixture ~name:"nrs_long" ~height:5 ~width:7 ~channels

let () =
  Utils.with_stated_config [ ("gpu_schedule_min_parallel", "64") ] @@ fun () ->
  let grad, comp, _ = narrow in
  let opt = Ll_test.captured_lowering ~name:"nrs_capture_long" comp in
  let plain = pipeline (Ll_test.hermetic_copy opt) in
  let accumulates seg = writes grad seg && Option.is_none (LL.zero_fill_target seg.LL.llc) in
  p_exists "the ordinary Metal preset leaves the narrow bias accumulation on one thread" plain
    ~f:(fun seg -> accumulates seg && launch_threads seg = 1);
  let lower_threshold =
    Utils.with_stated_config
      [ ("gpu_schedule_min_parallel", "32") ]
      (fun () -> pipeline (Ll_test.hermetic_copy opt))
  in
  p_exists "at the lower mapping threshold the ordinary preset maps the bias channels"
    lower_threshold ~f:(fun seg -> accumulates seg && launch_threads seg = channels);
  let plan =
    Autotune.seeded_plan ~facts:BI.metal_facts ~limits:BI.no_hardware_limits ~accum_prec:Fn.id
      ~static_indices:[] opt
  in
  p "the untuned Metal plan splits the bias gradient into 32 blocks"
    (Option.exists plan ~f:(fun plan ->
         List.exists plan.sp_split ~f:(fun (site, n) ->
             Ir.Tnode.equal site.Autotune.sr_target grad && n = blocks)));
  let scheduled =
    Option.value_map plan ~default:[] ~f:(fun plan ->
        pipeline (Sched.apply (Autotune.split_reduce_prelude plan.sp_split) opt))
  in
  p "the partials pass launches one thread per block and channel before the combine"
    (has_partials grad scheduled);
  let disabled =
    Autotune.seeded_split_sites ~facts:BI.metal_facts ~min_serial:0 ~static_indices:[] opt
  in
  p_empty "the zero split threshold leaves the Metal bias reduction unsplit"
    ~over:(Option.value_map plan ~default:[] ~f:(fun plan -> plan.sp_split))
    disabled

let () =
  Utils.with_stated_config [ ("gpu_schedule_min_parallel", "64") ] @@ fun () ->
  let root = Context.auto () in
  let backend = Context.backend_name root in
  let metal = String.equal backend "metal" in
  let grad, comp, want = narrow in
  let run compile =
    let ctx, routine = compile root comp Ir.Indexing.Empty in
    let ctx = Context.run (Context.run ctx routine) routine in
    Context.sync ctx;
    let got = Context.get_values ctx grad in
    let segments = routine.Context.segments in
    Context.release ctx;
    (got, segments)
  in
  let plain, _ = run (Context.compile ~name:"nrs_plain") in
  let disabled, _ = run (Autotune.seeded_default ~name:"nrs_disabled" ~min_serial:0) in
  p_all2 "disabling the split gives the ordinary bias gradient bitwise" disabled plain
    ~f:(fun a b -> Int64.equal (Int64.bits_of_float a) (Int64.bits_of_float b));
  let got, segments = run (Autotune.untuned_compile ~name:"nrs_untuned") in
  Verdict.gated ~when_:metal ~on:backend
    "the real untuned Metal compile ships the parallel partials pass and its combine"
    (has_partials grad segments);
  let tol _ = 1e-3 in
  p_close "the ordinary bias gradient agrees with the f64 sum" plain want ~tol;
  p_close "the untuned bias gradient agrees with the f64 sum" got want ~tol;
  Context.release root
