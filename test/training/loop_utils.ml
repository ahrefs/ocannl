(* Training-loop utilities (gh-ocannl-465), ported from llm.c's loop scaffolding:

   1. Host-side LR schedules (llmc/schedulers.h): a golden table of the four kinds plus claims
   pinning warmup, the decay endpoints, and WSD's stable region. 2. Global-norm gradient clipping
   (llmc/global_norm.cuh): the on-device norm must match a host recomputation from the (untouched!)
   gradient buffers, and the clipped SGD step must match a host oracle folding scale = min(1,
   max_norm/norm) into the update. Also exercises [scheduled_learning_rate]: a second step at a
   different schedule point must move parameters by the NEW rate (the data-backed scalar must not be
   undone by a re-fetch). 3. Gradient accumulation: two micro-steps at batch 2 with [~accum_steps:2]
   must reproduce -- in executed values, per the structural-vs-executable rule -- the gradients and
   the post-SGD parameters of a single batch-4 step. 4. The z-score outlier detector
   (llmc/outlier_detector.h): warmup returns nan, in-family values score small, a spike scores
   large. *)

open Base
open Ocannl
open Stdio
module IDX = Train.IDX
open Ocannl.Operation.DSL_modules

(* ----- 1. Learning-rate schedules ----- *)

let schedules () =
  let show name kind =
    let sched =
      {
        Train.Lr_schedule.kind;
        base_lr = 1.0;
        warmup_steps = 5;
        total_steps = 25;
        final_frac = 0.1;
      }
    in
    let lr step = Train.Lr_schedule.learning_rate sched ~step in
    printf "%-8s" name;
    List.iter [ 0; 2; 4; 5; 10; 15; 19; 20; 24; 30 ] ~f:(fun step ->
        printf " ";
        Test_utils.print_float ~prec:4 (lr step));
    printf "\n";
    sched
  in
  printf "lr at steps [0 2 4 5 10 15 19 20 24 30] (base 1.0, warmup 5, total 25, final_frac 0.1):\n";
  let con_s = show "constant" Train.Lr_schedule.Constant in
  let cos_s = show "cosine" Train.Lr_schedule.Cosine in
  let lin_s = show "linear" Train.Lr_schedule.Linear in
  let wsd_s = show "wsd" (Train.Lr_schedule.Wsd { decay_frac = 0.2 }) in
  let lr sched step = Train.Lr_schedule.learning_rate sched ~step in
  let tol _ = 1e-12 in
  let at sched steps = Array.of_list (List.map steps ~f:(lr sched)) in
  let const c got = Array.create ~len:(Array.length got) c in
  (let got = Array.of_list (List.map [ con_s; cos_s; lin_s; wsd_s ] ~f:(fun s -> lr s 0)) in
   Verdict.p_close "warmup applies to every kind, constant included" got (const 0.2 got) ~tol);
  (let got = at con_s [ 5; 15; 30 ] in
   Verdict.p_close "constant holds base_lr after warmup" got (const 1.0 got) ~tol);
  Verdict.p_close "warmup starts at base_lr/warmup_steps" [| lr cos_s 0 |] [| 0.2 |] ~tol;
  Verdict.p_close "warmup reaches base_lr on its last step" [| lr cos_s 4 |] [| 1.0 |] ~tol;
  Verdict.p_close "cosine reaches final_frac * base_lr past total_steps"
    [| lr cos_s 30 |]
    [| 0.1 |] ~tol;
  Verdict.p_close "linear reaches final_frac * base_lr past total_steps"
    [| lr lin_s 30 |]
    [| 0.1 |] ~tol;
  (let got = at wsd_s [ 5; 12; 19 ] in
   Verdict.p_close "wsd holds base_lr between warmup and the decay point" got (const 1.0 got) ~tol);
  Verdict.p "wsd decays below base_lr after the decay point" Float.(lr wsd_s 21 < 1.0);
  Verdict.p_all "cosine decay is nonincreasing" (List.range 5 30) ~f:(fun s ->
      Float.(lr cos_s Int.(s + 1) <= lr cos_s s +. 1e-12));
  (* Degenerate config: a warmup longer than the horizon must not keep ramping past it. *)
  let over =
    {
      Train.Lr_schedule.kind = Train.Lr_schedule.Cosine;
      base_lr = 1.0;
      warmup_steps = 10;
      total_steps = 5;
      final_frac = 0.1;
    }
  in
  Verdict.p_close "the total_steps clamp outranks a longer warmup"
    [| lr over 6 |]
    [| 0.1 |]
    ~tol:(fun _ -> 1e-12)

(* ----- Shared deterministic model: logits = b + w*x, mean squared error ----- *)

let din = 3
let classes = 2
let x_val b d = (Float.of_int (((b * 13) + (d * 7)) % 11) /. 5.5) -. 1.0
let target_val b c = (Float.of_int (((b * 5) + (c * 3)) % 7) /. 3.5) -. 1.0
let w_val c d = (Float.of_int (((c * 17) + (d * 29)) % 13) /. 6.5) -. 1.0
let b_val c = (Float.of_int (c * 7 % 5) /. 5.0) -. 0.4

(** Builds the model over a [batch]-sized input; [x_row b] gives the dataset row backing batch
    position [b], so accumulation cases can window the same dataset. With [freeze_w] set, the [w]
    branch goes through {!Operation.stop_gradient}: [w] stays in [loss.params] (and keeps its
    gradient node) while the backprop never reaches it. Returns [(x, targets, w, b_p, loss)]; the
    loss is mean-reduced over the batch. *)
let build_model ~freeze_w ~batch ~x_row =
  let x =
    NTDSL.init ~l:"x" ~prec:Ir.Ops.single ~b:[ batch ] ~o:[ din ]
      ~f:(function [| b; d |] -> x_val (x_row b) d | _ -> assert false)
      ()
  in
  let targets =
    NTDSL.init ~l:"targets" ~prec:Ir.Ops.single ~b:[ batch ] ~o:[ classes ]
      ~f:(function [| b; c |] -> target_val (x_row b) c | _ -> assert false)
      ()
  in
  (* Real parameters ([Tensor.param]-registered, deterministic values): {!Train.sgd_update},
     {!Train.grad_l2_norm} and {!Train.zero_params_grads} derive their set from [loss.params] (via
     {!Train.trainable_params}), and tensors built with [Operation.init ~grad_spec:Require_grad] do
     not join it. *)
  let w =
    let nd =
      Ir.Ndarray.init_array ~debug:"w" Ir.Ops.single ~dims:[| classes; din |] ~padding:None
        ~f:(function
        | [| c; d |] -> w_val c d
        | _ -> assert false)
    in
    TDSL.reshape_param ~l:"w" ~i:[ din ] ~o:[ classes ] nd ()
  in
  let b_p = TDSL.param ~values:(Array.init classes ~f:b_val) "b" () in
  Train.set_materialized x.Tensor.value;
  Train.set_materialized targets.Tensor.value;
  let logits = TDSL.O.(b_p + ((if freeze_w then TDSL.O.stop_gradient w else w) * x)) in
  let loss =
    let diff = TDSL.O.(logits - targets) in
    let%op sq_err = (diff *. diff) ++ "...|... => |->0" in
    TDSL.O.(sq_err /. !.(Float.of_int batch))
  in
  Train.set_materialized (Option.value_exn ~here:[%here] w.Tensor.diff).grad;
  Train.set_materialized (Option.value_exn ~here:[%here] b_p.Tensor.diff).grad;
  (x, targets, w, b_p, loss)

let read_params_and_grads ctx w b_p =
  let open Operation.At in
  let wv = Array.init classes ~f:(fun c -> Array.init din ~f:(fun d -> (ctx, w).@{[| c; d |]})) in
  let bv = Array.init classes ~f:(fun c -> (ctx, b_p).@[c]) in
  let wg = Array.init classes ~f:(fun c -> Array.init din ~f:(fun d -> (ctx, w).@%{[| c; d |]})) in
  let bg = Array.init classes ~f:(fun c -> (ctx, b_p).@%[c]) in
  (wv, bv, wg, bg)

(* The frozen-backbone case reads only the trainable gradient: a detached parameter's gradient is
   never written, hence never allocated into the context. *)
let bias_grads ctx b_p =
  let open Operation.At in
  Array.init classes ~f:(fun c -> (ctx, b_p).@%[c])

let global_norm_of wg bg =
  let acc = ref 0. in
  Array.iter wg ~f:(Array.iter ~f:(fun g -> acc := !acc +. (g *. g)));
  Array.iter bg ~f:(fun g -> acc := !acc +. (g *. g));
  Float.sqrt !acc

(* ----- 2. Global-norm clipping + scheduled learning rate ----- *)

let clipping () =
  Tensor.unsafe_reinitialize ();
  let batch = 4 in
  let x_row b = b in
  let _x, _targets, w, b_p, loss = build_model ~freeze_w:false ~batch ~x_row in
  let update = Train.grad_update loss in
  let max_norm = 0.5 in
  Verdict.p "a negative max_norm is rejected (it would reverse gradients)"
    (try
       ignore (Train.clip_by_global_norm ~max_norm:(-1.) loss : Train.grad_clipping);
       false
     with Invalid_argument _ -> true);
  let clip = Train.clip_by_global_norm ~max_norm loss in
  let sched =
    {
      Train.Lr_schedule.kind = Train.Lr_schedule.Linear;
      base_lr = 0.05;
      warmup_steps = 0;
      total_steps = 10;
      final_frac = 0.2;
    }
  in
  let learning_rate, set_lr = Train.scheduled_learning_rate sched in
  let weight_decay = 0.1 in
  let sgd = Train.sgd_update ~learning_rate ~weight_decay ~grad_scale:clip.Train.grad_scale loss in
  let step_comp = Ir.Assignments.sequence [ update; clip.Train.clip_comp; sgd ] in
  let ctx = Context.auto () in
  let ctx = Train.init_params ctx IDX.empty loss in
  let ctx, routine = Context.compile ctx step_comp IDX.empty in
  (* One clipped step: params move by lr * (scale * grad + wd * param). *)
  let open Operation.At in
  let one_step ctx ~step =
    let wv0, bv0, _, _ = read_params_and_grads ctx w b_p in
    let ctx = set_lr ctx ~step in
    let ctx = Context.run ctx routine in
    let wv1, bv1, wg, bg = read_params_and_grads ctx w b_p in
    let host_norm = global_norm_of wg bg in
    let device_norm = (ctx, clip.Train.grad_norm).@[0] in
    let device_scale = (ctx, clip.Train.grad_scale).@[0] in
    let host_scale = if Float.(host_norm > max_norm) then max_norm /. host_norm else 1. in
    let lr = Train.Lr_schedule.learning_rate sched ~step in
    let expected p0 g = p0 -. (lr *. ((host_scale *. g) +. (weight_decay *. p0))) in
    (* The stepped parameters, [w] row-major then [b], beside the host oracle's. *)
    let got = Array.append (Array.concat (Array.to_list wv1)) bv1 in
    let want =
      Array.append
        (Array.concat
           (Array.to_list
              (Array.mapi wv0 ~f:(fun c row ->
                   Array.mapi row ~f:(fun d p0 -> expected p0 wg.(c).(d))))))
        (Array.mapi bv0 ~f:(fun c p0 -> expected p0 bg.(c)))
    in
    (ctx, host_norm, device_norm, device_scale, host_scale, (got, want))
  in
  let ctx, host_norm, device_norm, device_scale, host_scale, (got, want) = one_step ctx ~step:0 in
  Verdict.p "gradient norm exceeds max_norm (clipping engaged)" Float.(host_norm > max_norm);
  Verdict.pass_fail_close "device grad_norm matches host norm over unclipped buffers"
    [| device_norm |] [| host_norm |] ~tol:(fun w -> 1e-4 *. (1. +. Float.abs w));
  Verdict.pass_fail_close "device grad_scale is min(1, max_norm/norm)" [| device_scale |]
    [| host_scale |] ~tol:(fun _ -> 1e-5);
  Verdict.pass_fail_close "clipped sgd step matches host oracle (step 0)" got want ~tol:(fun _ ->
      1e-5);
  (* A later schedule point: a visibly different lr must reach the device (data-backed scalar). *)
  let step = 8 in
  let _ctx, _, _, _, _, (got, want) = one_step ctx ~step in
  Verdict.p "scheduled lr differs between the probed steps"
    Float.(
      abs
        (Train.Lr_schedule.learning_rate sched ~step
        -. Train.Lr_schedule.learning_rate sched ~step:0)
      > 1e-3);
  Verdict.pass_fail_close "clipped sgd step matches host oracle at the rescheduled lr (step 8)" got
    want ~tol:(fun _ -> 1e-5)

(* ----- 3. Gradient accumulation ----- *)

let accumulation () =
  (* Case A: one batch-4 step. *)
  Tensor.unsafe_reinitialize ();
  let _x, _targets, w, b_p, loss = build_model ~freeze_w:false ~batch:4 ~x_row:(fun b -> b) in
  let learning_rate = Train.host_scalar ~l:"lr" 0.05 in
  let update = Train.grad_update loss in
  let sgd = Train.sgd_update ~learning_rate loss in
  let ctx = Context.auto () in
  let ctx = Train.init_params ctx IDX.empty loss in
  let ctx, grad_routine = Context.compile ctx update IDX.empty in
  let ctx, sgd_routine = Context.compile ctx sgd IDX.empty in
  let ctx = Context.run ctx grad_routine in
  let _, _, wg_a, bg_a = read_params_and_grads ctx w b_p in
  let ctx = Context.run ctx sgd_routine in
  let wv_a, bv_a, _, _ = read_params_and_grads ctx w b_p in
  (* Case B: two batch-2 micro-steps with accum_steps = 2 over the same 4 dataset rows. *)
  Tensor.unsafe_reinitialize ();
  let x, targets, w, b_p, loss = build_model ~freeze_w:false ~batch:2 ~x_row:(fun b -> b) in
  let learning_rate = Train.host_scalar ~l:"lr" 0.05 in
  let micro = Train.grad_update ~accum_steps:2 loss in
  let zero = Train.zero_params_grads loss in
  let sgd = Train.sgd_update ~learning_rate loss in
  let ctx = Context.auto () in
  let ctx = Train.init_params ctx IDX.empty loss in
  let ctx, zero_routine = Context.compile ctx zero IDX.empty in
  let ctx, micro_routine = Context.compile ctx micro IDX.empty in
  let ctx, sgd_routine = Context.compile ctx sgd IDX.empty in
  let ctx = Context.run ctx zero_routine in
  let ctx = Context.run ctx micro_routine in
  let _, _, wg_1, bg_1 = read_params_and_grads ctx w b_p in
  (* Second micro-batch: dataset rows 2 and 3. *)
  let x_data = Array.init (2 * din) ~f:(fun i -> x_val (2 + (i / din)) (i % din)) in
  let target_data =
    Array.init (2 * classes) ~f:(fun i -> target_val (2 + (i / classes)) (i % classes))
  in
  let ctx = Context.set_values ctx x.Tensor.value x_data in
  let ctx = Context.set_values ctx targets.Tensor.value target_data in
  let ctx = Context.run ctx micro_routine in
  let _, _, wg_b, bg_b = read_params_and_grads ctx w b_p in
  let max_abs_diff a b =
    Array.fold2_exn a b ~init:0. ~f:(fun acc x y -> Float.max acc (Float.abs (x -. y)))
  in
  (* A model's weights row-major, then its bias, as one array. *)
  let flat w b = Array.append (Array.concat (Array.to_list w)) b in
  Verdict.p "second micro-step changed the accumulated gradients"
    Float.(
      Float.max
        (Array.fold2_exn wg_1 wg_b ~init:0. ~f:(fun acc r1 r2 -> Float.max acc (max_abs_diff r1 r2)))
        (max_abs_diff bg_1 bg_b)
      > 1e-4);
  Verdict.pass_fail_close "accumulated micro-batch gradients match the single big-batch gradients"
    (flat wg_b bg_b) (flat wg_a bg_a) ~tol:(fun _ -> 1e-5);
  let ctx = Context.run ctx sgd_routine in
  let wv_b, bv_b, _, _ = read_params_and_grads ctx w b_p in
  Verdict.pass_fail_close "post-sgd parameters match between accumulation and big batch"
    (flat wv_b bv_b) (flat wv_a bv_a) ~tol:(fun _ -> 1e-5)

(* ----- 3b. Accumulation with a frozen backbone -----

   A parameter detached behind [stop_gradient] stays in [loss.params] and keeps its gradient node,
   but the loss's zero_grads tree does not zero it and the backprop does not accumulate into it.
   [grad_update ~accum_steps] must accept such a model (the documented freezing flow) and leave the
   frozen gradient untouched, while the trainable parameter still accumulates. The optimizer-side
   helpers derive their set from the backprop ([Train.trainable_params], gh-ocannl-673), so a frozen
   parameter takes no step at all -- in particular no weight decay. *)

let frozen_backbone_accumulation () =
  (* Case A: one batch-4 step over the frozen-backbone model. *)
  Tensor.unsafe_reinitialize ();
  let _x, _targets, w, b_p, loss = build_model ~freeze_w:true ~batch:4 ~x_row:(fun b -> b) in
  let trainable = Train.trainable_params loss in
  Verdict.p "trainable_params excludes the frozen parameter and keeps the trained one"
    ((not (Set.mem trainable w)) && Set.mem trainable b_p);
  let update = Train.grad_update loss in
  let ctx = Context.auto () in
  let ctx = Train.init_params ctx IDX.empty loss in
  let ctx, grad_routine = Context.compile ctx update IDX.empty in
  let ctx = Context.run ctx grad_routine in
  let bg_a = bias_grads ctx b_p in
  (* Frozen means frozen: with a decoupled-decay term in the delta, an [sgd_update] over all of
     [loss.params] would decay [w] every step even though no gradient reaches it. The derived set
     emits no step for [w], so its values stay bitwise identical. *)
  let open Operation.At in
  let w_vals ctx =
    Array.init classes ~f:(fun c -> Array.init din ~f:(fun d -> (ctx, w).@{[| c; d |]}))
  in
  let b_vals ctx = Array.init classes ~f:(fun c -> (ctx, b_p).@[c]) in
  let w_before = w_vals ctx and b_before = b_vals ctx in
  let learning_rate = Train.host_scalar ~l:"lr" 0.1 in
  let sgd = Train.sgd_update ~learning_rate ~weight_decay:0.5 loss in
  let ctx, sgd_routine = Context.compile ctx sgd IDX.empty in
  let ctx = Context.run ctx sgd_routine in
  let w_after = w_vals ctx and b_after = b_vals ctx in
  let w_moved =
    Array.exists2_exn w_before w_after ~f:(fun r r' ->
        Array.exists2_exn r r' ~f:(fun v v' -> not (Float.equal v v')))
  in
  let b_moved = Array.exists2_exn b_before b_after ~f:(fun v v' -> not (Float.equal v v')) in
  Verdict.p "a frozen parameter does not move under ~weight_decay" (not w_moved);
  Verdict.p "the trained parameter still takes the optimizer step" b_moved;
  (* Case B: two batch-2 micro-steps with accum_steps = 2 over the same 4 dataset rows. *)
  Tensor.unsafe_reinitialize ();
  let x, targets, _w, b_p, loss = build_model ~freeze_w:true ~batch:2 ~x_row:(fun b -> b) in
  let micro = try Some (Train.grad_update ~accum_steps:2 loss) with Invalid_argument _ -> None in
  Verdict.p "a frozen parameter does not reject the accumulating grad_update" (Option.is_some micro);
  match micro with
  | None -> ()
  | Some micro ->
      let zero = Train.zero_params_grads loss in
      let ctx = Context.auto () in
      let ctx = Train.init_params ctx IDX.empty loss in
      let ctx, zero_routine = Context.compile ctx zero IDX.empty in
      let ctx, micro_routine = Context.compile ctx micro IDX.empty in
      let ctx = Context.run ctx zero_routine in
      let ctx = Context.run ctx micro_routine in
      (* Second micro-batch: dataset rows 2 and 3. *)
      let x_data = Array.init (2 * din) ~f:(fun i -> x_val (2 + (i / din)) (i % din)) in
      let target_data =
        Array.init (2 * classes) ~f:(fun i -> target_val (2 + (i / classes)) (i % classes))
      in
      let ctx = Context.set_values ctx x.Tensor.value x_data in
      let ctx = Context.set_values ctx targets.Tensor.value target_data in
      let ctx = Context.run ctx micro_routine in
      let bg_b = bias_grads ctx b_p in
      Verdict.p "the frozen model's micro-steps move the trainable gradient"
        Float.(Array.fold bg_b ~init:0. ~f:(fun acc g -> Float.max acc (Float.abs g)) > 1e-4);
      Verdict.pass_fail_close "the trainable gradient accumulates the same as the big batch" bg_b
        bg_a ~tol:(fun _ -> 1e-5)

(* ----- 3c. Params-driven helpers over a paramless differentiable loss (gh-ocannl-670) -----

   A leaf built with [Operation.init ~grad_spec:Require_grad] (the deterministic-values test idiom)
   gets a gradient but never joins [t.params], so the params-driven helpers used to compile empty
   routines: the optimizer step was a silent no-op and this file's own executed-parity claims once
   passed vacuously. The optimizer-side ones now raise [Session_error] when the loss trains no
   registered parameters, naming [?params], the way to select such a leaf. [zero_params_grads] is
   the exception: its default is what [grad_update ~accum_steps] accumulates, every leaf gradient
   the backprop writes, the unregistered leaf's included (gh-ocannl-1743). *)

let paramless_guard () =
  Tensor.unsafe_reinitialize ();
  let w =
    Operation.init ~l:"w_unregistered" ~prec:Ir.Ops.single ~o:[ din ]
      ~f:(function [| d |] -> w_val 0 d | _ -> assert false)
      ~grad_spec:Tensor.Require_grad ()
  in
  let%op loss = (w *. w) ++ "...|... => |->0" in
  Verdict.p "an unregistered differentiable leaf does not join loss.params"
    (Set.is_empty loss.Tensor.params);
  Verdict.p "trainable_params is empty for the paramless loss"
    (Set.is_empty (Train.trainable_params loss));
  let learning_rate = Train.host_scalar ~l:"lr" 0.1 in
  let raises f = match f () with () -> false | exception Tensor.Session_error _ -> true in
  Verdict.p "sgd_update rejects a loss that trains no registered parameters"
    (raises (fun () -> ignore (Train.sgd_update ~learning_rate loss : Ir.Assignments.comp)));
  Verdict.p "the refusal names ?params, the way to step such a leaf"
    (match Train.sgd_update ~learning_rate loss with
    | _ -> false
    | exception Tensor.Session_error (msg, _) -> String.is_substring msg ~substring:"[?params]");
  Verdict.p "grad_l2_norm rejects a loss that trains no registered parameters"
    (raises (fun () -> ignore (Train.grad_l2_norm loss : Tensor.t * Ir.Assignments.comp)));
  Verdict.p "grad_checksum rejects a loss that trains no registered parameters"
    (raises (fun () -> ignore (Train.grad_checksum loss : Tensor.t * Ir.Assignments.comp)));
  Verdict.p "zero_params_grads zeroes the unregistered leaf's gradient by default"
    (match Train.zero_params_grads loss with
    | comp ->
        Set.mem
          (Ir.Assignments.collect_written comp.Ir.Assignments.asgns)
          (Option.value_exn ~here:[%here] w.Tensor.diff).Tensor.grad
    | exception Tensor.Session_error _ -> false);
  Verdict.p "an explicit ?params override bypasses the derivation"
    (not
       (raises (fun () ->
            ignore
              (Train.sgd_update ~learning_rate ~params:(Set.singleton (module Tensor) w) loss
                : Ir.Assignments.comp))))

(* ----- 4. Outlier detector ----- *)

let outlier_detector () =
  let det = Train.Outlier_detector.create ~window_size:4 () in
  let zs = List.map [ 1.0; 1.1; 0.9; 1.0 ] ~f:(Train.Outlier_detector.update det) in
  Verdict.p_all "z-score is nan while the window fills" zs ~f:Float.is_nan;
  let z_ordinary = Train.Outlier_detector.update det 1.05 in
  let z_nan = Train.Outlier_detector.update det Float.nan in
  let z_spike = Train.Outlier_detector.update det 10.0 in
  printf "ordinary z: ";
  Test_utils.print_float ~prec:3 z_ordinary;
  printf ", spike z: ";
  Test_utils.print_float ~prec:3 z_spike;
  printf "\n";
  Verdict.p "an in-family value scores |z| < 1" Float.(abs z_ordinary < 1.0);
  (* Scored against the previous window only -- self-inclusion would cap this at sqrt 3. *)
  Verdict.p "a spike scores z > 10 (no self-dilution)" Float.(z_spike > 10.0);
  Verdict.p "a nan sample scores infinite (skip-forcing)" Float.(z_nan = infinity);
  let z_after_nan = Train.Outlier_detector.update det 1.0 in
  Verdict.p "the window survives a nan sample (later scores stay finite)"
    (Float.is_finite z_after_nan);
  (* Centered variance: a large common offset with small real variance must not cancel to a zero std
     (the running E[x^2] - E[x]^2 form scored this window's next deviation as infinity). *)
  let det = Train.Outlier_detector.create ~window_size:4 () in
  List.iter
    [ 1e8; 1e8 +. 1.; 1e8 -. 1.; 1e8 ]
    ~f:(fun v -> ignore (Train.Outlier_detector.update det v : float));
  let z_offset = Train.Outlier_detector.update det (1e8 +. 1.) in
  (let label = "a large-offset window keeps a finite variance (no cancellation)" in
   Verdict.p label
     (Float.is_finite z_offset
     && Verdict.within label [| z_offset |] [| Float.sqrt 2. |] ~tol:(fun _ -> 1e-6)));
  (* Normalized moments: a huge finite spike, once recorded, must not overflow the variance to
     infinity -- which would score the NEXT spike 0 and let it through. Against the post-spike
     window [1e308; 1; 1; 1] the correct z of a second 1e308 is (0.75 / sqrt 0.1875) = sqrt 3. *)
  let det = Train.Outlier_detector.create ~window_size:4 () in
  List.iter [ 1.0; 1.0; 1.0; 1.0 ] ~f:(fun v ->
      ignore (Train.Outlier_detector.update det v : float));
  let z_spike1 = Train.Outlier_detector.update det 1e308 in
  let z_spike2 = Train.Outlier_detector.update det 1e308 in
  Verdict.p "a spike off a constant baseline scores +infinity" Float.(z_spike1 = infinity);
  Verdict.p_close
    "a recorded huge spike does not overflow the variance (second spike scores sqrt 3)"
    [| z_spike2 |]
    [| Float.sqrt 3. |]
    ~tol:(fun _ -> 1e-6);
  (* Mean accumulation: a window saturated at the float maximum must not overflow the x/n partial
     sums into a nan mean -- a sample far below that constant-at-max baseline must score decisively
     negative (nan fails this comparison, so a poisoned mean fails the claim). *)
  let det = Train.Outlier_detector.create ~window_size:3 () in
  List.iter [ Float.max_finite_value; Float.max_finite_value; Float.max_finite_value ] ~f:(fun v ->
      ignore (Train.Outlier_detector.update det v : float));
  let z_below_max = Train.Outlier_detector.update det 1.0 in
  Verdict.p "a maxed-out window still scores (no nan from mean overflow)" Float.(z_below_max < -1e6)

let () =
  schedules ();
  clipping ();
  accumulation ();
  frozen_backbone_accumulation ();
  paramless_guard ();
  outlier_detector ()
