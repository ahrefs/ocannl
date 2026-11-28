(* gh-ocannl-2506: LayerNorm gamma/beta parameter-gradient reductions on Metal.

   In untuned bf16 [gpt2_mini_train], backprop lowers [beta.grad] and [gamma.grad] with their serial
   [batch x seq] reduction loops outside the [d_model = 256] output loop. Before gh-ocannl-2507
   (reduction sinking), Metal's default GPU preset mapped [d_model] inside the serial loops, so
   [C_syntax.try_localize_serial_reduce] refused to peel past the hardware loops and both reductions
   remained device-memory RMWs ([2 volatile rmw read(s)]) in a single workgroup ([grid=[1;1;1]
   block=[256;1;1]]). Reduction sinking (gh-ocannl-2507) localized both into registers ([0 volatile
   rmw read(s)]) while leaving both in a single workgroup. Enabling untuned split-reduce on Metal
   (gh-ocannl-2506, matching CUDA and HIP) splits both [batch x seq = 1024] reductions into 32
   workgroups ([grid=[32;1;1] block=[256;1;1]]) with an unrolled 32-way combine and [0 volatile rmw
   read(s)].

   The fixture is an [Nn_blocks.layer_norm] backward pass at GPT-2-mini dimensions ([batch = 8, seq
   = 128, d_model = 256]) with bf16 activations and f32 [gamma]/[beta] masters and gradients.
   Structural claims run on every backend (including [cc]); launch-shape and volatility claims run
   on hardware-grid / volatile-workaround backends via [Verdict.leg]; executed gradient parity
   against an independent host f64 sum and the unsplit sunk reference runs everywhere. *)

(* cc-fp-contract: same-shape -- on cc seeded_default neither sinks nor splits, so every untuned leg
   compiles the default plan *)

open Base
open Ocannl
open Ocannl.Nn_blocks.DSL_modules
module LL = Ir.Low_level
module Sched = Ir.Schedule
module CS = Ir.C_syntax
open Verdict.Claims

let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"")
let facts = Context.backend_facts_of_name backend_name
let splits_here = facts.Ir.Backend_intf.splits_untuned_reductions
let sinks_here = facts.Ir.Backend_intf.sinks_untuned_reductions
let metal_here = String.equal backend_name "metal"

let nb = 8
and ns = 128
and nd = 256

(* Deterministic non-dyadic values so split reassociation moves at least one cell bitwise. *)
let values ~n ~stride =
  Array.init n ~f:(fun c -> (Float.of_int (c * stride % 1009) /. 1009.) -. 0.5)

let x_values = values ~n:(nb * ns * nd) ~stride:389
let dy_values = values ~n:(nb * ns * nd) ~stride:577
let x = NTDSL.ndarray x_values ~label:[ "ln_x" ] ~batch_dims:[ nb; ns ] ~output_dims:[ nd ] ()
let dy = NTDSL.ndarray dy_values ~label:[ "ln_dy" ] ~batch_dims:[ nb; ns ] ~output_dims:[ nd ] ()
let ln = Nn_blocks.layer_norm ~label:[ "ln_test" ] ()
let%op y = ln x
let%op loss = (y *. dy) ++ "... => 0"

let find_param tag =
  Set.find_exn loss.Tensor.params ~f:(fun p ->
      List.mem p.Tensor.value.Ir.Tnode.label tag ~equal:String.equal)

let gamma = find_param "gamma"
let beta = find_param "beta"
let grad_of t = (Option.value_exn ~here:[%here] t.Tensor.diff).Tensor.grad
let gamma_grad = grad_of gamma
let beta_grad = grad_of beta

let () =
  (* Match bench_gpt_model.ml: bf16 activations and activation grads, f32 gamma/beta params and
     grads. *)
  let is_ln tn =
    List.exists tn.Ir.Tnode.label ~f:(fun l -> String.equal l "gamma" || String.equal l "beta")
  in
  Precision_policy.apply ~except:is_ln
    {
      Precision_policy.param_prec = None;
      activation_prec = Some Ir.Ops.bfloat16;
      grad_prec = Some Ir.Ops.bfloat16;
    }
    loss;
  Train.set_materialized gamma_grad;
  Train.set_materialized beta_grad

let comp = Train.grad_update loss
let bitwise a b = Int64.equal (Int64.bits_of_float a) (Int64.bits_of_float b)
let seg_writes tn (seg : LL.optimized) = Set.mem (Sched.written_nodes seg.LL.llc) tn

let dims_of (seg : LL.optimized) =
  let { LL.grid; block } = LL.launch_dims seg.LL.llc in
  (Array.to_list grid, Array.to_list block)

let is_grid_block ~grid ~block seg =
  let g, b = dims_of seg in
  List.equal Int.equal g grid && List.equal Int.equal b block

(* Find the partials segment feeding [target]'s combine segment: a segment writing a distinct node
   that [target]'s segment reads and whose element count is [blocks * nd]. *)
let partials_segment ~blocks ~target (segs : LL.optimized list) =
  Option.bind
    (List.find (List.rev segs) ~f:(seg_writes target))
    ~f:(fun comb ->
      let read_tns = Hashtbl.keys comb.LL.traced_store in
      List.find segs ~f:(fun seg ->
          (not (phys_equal seg comb))
          && List.exists read_tns ~f:(fun tn ->
              (not (Ir.Tnode.equal tn target))
              && Ir.Tnode.num_elems tn = blocks * nd
              && seg_writes tn seg)))

let param_rmw_count (vol : CS.volatility_summary) =
  List.count vol.entries ~f:(fun (_, site) ->
      match site with
      | CS.Volatile_rmw_reads ident ->
          String.is_substring ident ~substring:"gamma"
          || String.is_substring ident ~substring:"beta"
      | CS.Volatile_accumulation_reads _ | CS.Plain_accumulator _ -> false)

(* {2 Structural claims (run on every backend, including cc)} *)

let () =
  let opt = Context.lowered_for_decisions (Context.auto ()) comp Ir.Indexing.Empty in
  let open Ir.Backend_intf in
  let split_sites ~facts ~min_serial =
    Autotune.seeded_split_sites ~facts ~min_serial ~static_indices:[] opt
  in
  let metal_sites = split_sites ~facts:metal_facts ~min_serial:1024 in
  let has_split sites target blocks =
    List.exists sites ~f:(fun (s, b) -> Ir.Tnode.equal s.Autotune.sr_target target && b = blocks)
  in
  p "structural: Metal untuned split splits gamma.grad into 32 blocks at min_serial=1024"
    (has_split metal_sites gamma_grad 32);
  p "structural: Metal untuned split splits beta.grad into 32 blocks at min_serial=1024"
    (has_split metal_sites beta_grad 32);
  let candidates = Autotune.split_reduce_sites opt in
  p_empty "structural control: cc splits none of the LayerNorm reduction sites" ~over:candidates
    (split_sites ~facts:cc_facts ~min_serial:1024);
  p_empty "structural control: min_serial=0 splits none of the LayerNorm reduction sites on Metal"
    ~over:candidates
    (split_sites ~facts:metal_facts ~min_serial:0);
  let copy_opt (o : LL.optimized) =
    {
      o with
      LL.traced_store = Hashtbl.copy o.LL.traced_store;
      optimize_ctx = LL.copy_optimize_ctx o.LL.optimize_ctx;
    }
  in
  let metal_plan =
    Autotune.seeded_plan ~min_serial:1024 ~reduction_sink:true ~facts:metal_facts
      ~limits:no_hardware_limits ~accum_prec:Fn.id ~static_indices:[] opt
  in
  p "structural: Metal seeded_plan records 32-block splits for gamma.grad and beta.grad"
    (Option.exists metal_plan ~f:(fun plan ->
         has_split plan.Autotune.sp_split gamma_grad 32
         && has_split plan.Autotune.sp_split beta_grad 32));
  let metal_split_segs =
    let split_opt =
      Sched.apply_classified ~static_indices:[]
        (Autotune.split_reduce_prelude metal_sites)
        (copy_opt opt)
    in
    Sched.maybe_default_schedules ~facts:metal_facts ~limits:no_hardware_limits ~static_indices:[]
      split_opt
  in
  p "structural: Metal untuned split launches gamma.grad partials as grid=[32;1;1] block=[256;1;1]"
    (Option.exists
       (partials_segment ~blocks:32 ~target:gamma_grad metal_split_segs)
       ~f:(is_grid_block ~grid:[ 32; 1; 1 ] ~block:[ 256; 1; 1 ]));
  p "structural: Metal untuned split launches beta.grad partials as grid=[32;1;1] block=[256;1;1]"
    (Option.exists
       (partials_segment ~blocks:32 ~target:beta_grad metal_split_segs)
       ~f:(is_grid_block ~grid:[ 32; 1; 1 ] ~block:[ 256; 1; 1 ]));
  let metal_unsplit_segs =
    Sched.maybe_default_schedules ~facts:metal_facts ~limits:no_hardware_limits ~static_indices:[]
      (copy_opt opt)
  in
  p "structural control: unsplit Metal schedule keeps gamma.grad and beta.grad in one workgroup"
    (Option.exists
       (List.find metal_unsplit_segs ~f:(seg_writes gamma_grad))
       ~f:(is_grid_block ~grid:[ 1; 1; 1 ] ~block:[ 256; 1; 1 ])
    && Option.exists
         (List.find metal_unsplit_segs ~f:(seg_writes beta_grad))
         ~f:(is_grid_block ~grid:[ 1; 1; 1 ] ~block:[ 256; 1; 1 ]))

(* {2 Executed claims: root cause, sink localization, split launch/volatility, value parity} *)

let compile_and_run compile =
  let ctx = Context.auto () in
  let ctx = Train.init_params ctx Ir.Indexing.Empty loss in
  let ctx, routine = compile ctx comp Ir.Indexing.Empty in
  let ctx = Context.run ctx routine in
  Context.sync ctx;
  let dy_bf16 = Context.get_values ctx dy.Tensor.value in
  ( Context.get_values ctx gamma_grad,
    Context.get_values ctx beta_grad,
    dy_bf16,
    routine.Context.segments,
    routine.Context.volatility )

let () =
  let g_default, b_default, dy_bf16, segs_default, vol_default = compile_and_run Context.compile in
  let r_sunk = ref None in
  let g_sunk, b_sunk, _, segs_sunk, vol_sunk =
    compile_and_run (fun ctx comp bindings ->
        Autotune.seeded_default ~min_serial:0 ~reduction_sink:true
          ~report:(fun r -> r_sunk := Some r)
          ctx comp bindings)
  in
  let r_split = ref None in
  let g_split, b_split, _, segs_split, vol_split =
    compile_and_run (fun ctx comp bindings ->
        Autotune.seeded_default ~min_serial:1024 ~reduction_sink:true
          ~report:(fun r -> r_split := Some r)
          ctx comp bindings)
  in
  let r_split = Option.value_exn ~here:[%here] !r_split in
  (* Host f64 reference for beta.grad over the exact bf16-rounded dy values. *)
  let beta_exact =
    Array.init nd ~f:(fun d ->
        let acc = ref 0. in
        for r = 0 to (nb * ns) - 1 do
          acc := !acc +. dy_bf16.((r * nd) + d)
        done;
        !acc)
  in
  let max_diff a b =
    Array.fold2_exn a b ~init:0. ~f:(fun m x y -> Float.max m (Float.abs (x -. y)))
  in
  Stdio.eprintf
    "backend=%s (not part of the golden): vol default=%d (param_rmw=%d) sunk=%d (param_rmw=%d) \
     split=%d (param_rmw=%d); max_diff beta_vs_f64=%g gamma_split_vs_sunk=%g beta_split_vs_sunk=%g\n\
     %!"
    backend_name vol_default.volatile_rmw_reads (param_rmw_count vol_default)
    vol_sunk.volatile_rmw_reads (param_rmw_count vol_sunk) vol_split.volatile_rmw_reads
    (param_rmw_count vol_split) (max_diff b_split beta_exact) (max_diff g_split g_sunk)
    (max_diff b_split b_sunk);
  let tol _ = 1e-3 in
  p_exists "gamma.grad is not trivially zero" (Array.to_list g_split) ~f:(fun v ->
      Float.(abs v > 1.));
  p_exists "beta.grad is not trivially zero" (Array.to_list b_split) ~f:(fun v ->
      Float.(abs v > 1.));
  p_close "split beta.grad matches the f64 host reference" b_split beta_exact ~tol;
  p_close "split beta.grad matches the sunk unsplit beta.grad" b_split b_sunk ~tol;
  p_close "split gamma.grad matches the sunk unsplit gamma.grad" g_split g_sunk ~tol;
  p_close "sunk unsplit gamma.grad matches the default gamma.grad" g_sunk g_default ~tol;
  p_close "sunk unsplit beta.grad matches the default beta.grad" b_sunk b_default ~tol;
  let vol_leg = Verdict.leg ~when_:metal_here ~on:backend_name () in
  Verdict.in_leg ~leg:vol_leg
    "unsunk default reproduces 2 volatile RMW reads on gamma.grad and beta.grad" (fun label ->
      p label (vol_default.requested && param_rmw_count vol_default = 2));
  Verdict.in_leg ~leg:vol_leg
    "sunk unsplit localizes both gamma.grad and beta.grad to 0 volatile RMW reads" (fun label ->
      p label (vol_sunk.requested && param_rmw_count vol_sunk = 0));
  Verdict.in_leg ~leg:vol_leg
    "split untuned default keeps 0 volatile RMW reads on gamma.grad, beta.grad and partials"
    (fun label -> p label (vol_split.requested && param_rmw_count vol_split = 0));
  let gpu_leg = Verdict.leg ~when_:(splits_here && sinks_here) ~on:backend_name () in
  Verdict.in_leg ~leg:gpu_leg
    "unsunk default launches gamma.grad and beta.grad in one workgroup [1;1;1] x [256;1;1]"
    (fun label ->
      p label
        (Option.exists
           (List.find segs_default ~f:(seg_writes gamma_grad))
           ~f:(is_grid_block ~grid:[ 1; 1; 1 ] ~block:[ 256; 1; 1 ])
        && Option.exists
             (List.find segs_default ~f:(seg_writes beta_grad))
             ~f:(is_grid_block ~grid:[ 1; 1; 1 ] ~block:[ 256; 1; 1 ])));
  Verdict.in_leg ~leg:gpu_leg
    "sunk unsplit still launches gamma.grad and beta.grad in one workgroup [1;1;1] x [256;1;1]"
    (fun label ->
      p label
        (Option.exists
           (List.find segs_sunk ~f:(seg_writes gamma_grad))
           ~f:(is_grid_block ~grid:[ 1; 1; 1 ] ~block:[ 256; 1; 1 ])
        && Option.exists
             (List.find segs_sunk ~f:(seg_writes beta_grad))
             ~f:(is_grid_block ~grid:[ 1; 1; 1 ] ~block:[ 256; 1; 1 ])));
  Verdict.in_leg ~leg:gpu_leg
    "untuned default splits both gamma.grad and beta.grad into 32 blocks without fallback"
    (fun label ->
      p label
        ((not r_split.sd_fell_back)
        && List.exists r_split.sd_split ~f:(fun (tn, b) -> Ir.Tnode.equal tn gamma_grad && b = 32)
        && List.exists r_split.sd_split ~f:(fun (tn, b) -> Ir.Tnode.equal tn beta_grad && b = 32)));
  Verdict.in_leg ~leg:gpu_leg
    "untuned default launches both partials kernels as grid=[32;1;1] block=[256;1;1]" (fun label ->
      p label
        (Option.exists
           (partials_segment ~blocks:32 ~target:gamma_grad segs_split)
           ~f:(is_grid_block ~grid:[ 32; 1; 1 ] ~block:[ 256; 1; 1 ])
        && Option.exists
             (partials_segment ~blocks:32 ~target:beta_grad segs_split)
             ~f:(is_grid_block ~grid:[ 32; 1; 1 ] ~block:[ 256; 1; 1 ])));
  Verdict.in_leg ~leg:gpu_leg
    "split reassociation changes at least one cell bitwise vs sunk unsplit" (fun label ->
      p label
        (Array.exists2_exn g_split g_sunk ~f:(fun a b -> not (bitwise a b))
        && Array.exists2_exn b_split b_sunk ~f:(fun a b -> not (bitwise a b))))
