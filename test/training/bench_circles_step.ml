(* Diagnostic bench for the circles_conv sgd step on the configured backend: prints the default
   fission segmentation (kinds, launch geometry, parallel sizes), the autotune report, and
   steady-state per-step wall times with and without the host loss readback. Not a test; run
   manually from test/config. *)

open Base
open Ocannl
open Stdio
module Tn = Ir.Tnode
module IDX = Train.IDX
open Nn_blocks.DSL_modules
module Asgns = Ir.Assignments
module LL = Ir.Low_level
module Sched = Ir.Schedule

let lenet = Nn_blocks.lenet
let softmax = Nn_blocks.softmax

let () =
  let seed = 42 in
  Utils.settings.fixed_state_for_init <- Some seed;
  Tensor.unsafe_reinitialize ();
  TDSL.default_param_init := NTDSL.xavier ~scale_sq:0.06 TDSL.O.uniform1;
  let image_size = 16 in
  let max_circles = 3 in
  let num_classes = max_circles in
  let config =
    Dataprep.Circles.Config.
      { image_size; max_radius = 4; min_radius = 2; max_circles; seed = Some seed }
  in
  let batch_size = 8 in
  let total_samples = batch_size * 20 in
  let n_batches = total_samples / batch_size in
  let images_data, labels_data =
    Dataprep.Circles.generate_single_prec ~config ~len:total_samples ()
  in
  let labels_array = Bigarray.array2_of_genarray labels_data in
  let labels_list =
    List.init total_samples ~f:(fun i -> Int.of_float (Bigarray.Array2.get labels_array i 0) - 1)
  in
  let images_ndarray = Ir.Ndarray.as_array Ir.Ops.Single images_data in
  let labels_one_hot = Nn_blocks.dense_one_hot_of_int_list ~num_classes labels_list in
  let batch_n, bindings = IDX.get_static_symbol ~static_range:n_batches IDX.empty in
  let step_n, bindings = IDX.get_static_symbol bindings in
  let images = TDSL.rebatch ~l:"images" images_ndarray () in
  let%op batch_images = images @| batch_n in
  let%op batch_labels = labels_one_hot @| batch_n in
  let%op logits = lenet () ~train_step:None batch_images in
  let%op probs = softmax ~spec:"...|v" () logits in
  let%op correct_prob = (probs *. batch_labels) ++ "...|... => ...|0" in
  let%op sample_loss = neg (log correct_prob) in
  let%op batch_loss = (sample_loss ++ "...|... => |->0") /. !..batch_size in
  let epochs = 1000 in
  let total_steps = epochs * n_batches in
  Train.every_non_literal_materialized batch_loss;
  let update = Train.grad_update batch_loss in
  let%op learning_rate = 0.01 *. ((1.2 *. !..total_steps) - !@step_n) /. !..total_steps in
  Train.set_materialized learning_rate.value;
  let sgd = Train.sgd_update ~learning_rate batch_loss in
  Train.set_materialized batch_loss.value;
  let ctx = Context.auto () in
  let backend = Context.backend_name ctx in
  let limits = Context.hardware_limits ctx in
  let ctx = Train.init_params ctx bindings batch_loss in
  let step_comp = Asgns.sequence [ update; sgd ] in

  (* --- Segment census: capture the lowered step, apply the default schedule pipeline. --- *)
  let stash = ref None in
  let static_indices = [ batch_n; step_n ] in
  let _c, _r =
    Context.compile
      ~lowered_transform:(fun opt ->
        stash := Some opt;
        [ opt ])
      ctx step_comp bindings
  in
  let opt = Option.value_exn ~here:[%here] !stash in
  let census name segs =
    printf "%s: %d segments\n" name (List.length segs);
    List.iteri segs ~f:(fun i seg ->
        let dims = LL.launch_dims seg.LL.llc in
        let np = Array.fold dims.grid ~init:1 ~f:( * ) * Array.fold dims.block ~init:1 ~f:( * ) in
        let stmts = List.length (LL.flat_lines [ seg.LL.llc ]) in
        printf "  seg%-2d threads=%-6d grid=[%d;%d;%d] block=[%d;%d;%d] stmts=%d\n" i np
          dims.grid.(0) dims.grid.(1) dims.grid.(2) dims.block.(0) dims.block.(1) dims.block.(2)
          stmts)
  in
  printf "backend: %s\n" backend;
  census "default pipeline (min_parallel=1024)"
    (Sched.maybe_default_schedules ~backend_name:backend ~limits ~static_indices opt);
  (* The autotune fissioned preset annotates with min_parallel:1 — what the tuned winner runs. *)
  let preset o = Sched.default_gpu ~min_parallel:1 ~limits o in
  let zero_sched tns = Sched.zero_expansion ~limits tns in
  census "autotune preset (min_parallel=1)"
    (List.map (Sched.fission_scheduled ~preset ~zero_sched ~static_indices opt)
       ~f:(fun (_, _, _, post) -> post));

  (* --- Autotune (same call shape as circles_conv) with the report printed. --- *)
  let scratch = Train.init_params (Context.auto ()) bindings batch_loss in
  let report = ref None in
  let ctx, sgd_routine =
    Autotune.tune ~rounds:0 ~timing_ctx:scratch
      ~report:(fun r -> report := Some r)
      ctx step_comp bindings
  in
  (match !report with
  | Some r ->
      printf "tune: state=%s fissioned=%b candidates=%d baseline_ms=%.3f best_ms=%.3f\n"
        (Autotune.outcome_name r.Autotune.outcome)
        r.Autotune.fissioned r.Autotune.candidates_timed r.Autotune.baseline_ms r.Autotune.best_ms
  | None -> printf "tune: no report\n");

  (* --- Steady-state stepping, as the training loop does it. --- *)
  let step_ref = IDX.find_exn sgd_routine.Context.bindings step_n in
  let batch_ref = IDX.find_exn sgd_routine.Context.bindings batch_n in
  step_ref := 0;
  batch_ref := 0;
  let open Operation.At in
  let time_steps ~readback n =
    (* Warmup. *)
    for _ = 1 to 20 do
      Train.run ctx sgd_routine
    done;
    Context.sync ctx;
    let t0 = Unix.gettimeofday () in
    for _ = 1 to n do
      Train.run ctx sgd_routine;
      if readback then ignore ((ctx, batch_loss).@[0] : float) else Context.sync ctx
    done;
    Context.sync ctx;
    (Unix.gettimeofday () -. t0) /. Float.of_int n *. 1000.
  in
  let with_rb = time_steps ~readback:true 200 in
  let without_rb = time_steps ~readback:false 200 in
  printf "steady-state per step: %.3f ms with loss readback, %.3f ms sync-only\n" with_rb without_rb;

  (* --- Per-segment-boundary cost: a chain of K transposes (every edge misaligned, so fission cuts
     at each), each a trivial 64x64 copy. Slope of routine time over K isolates the launch +
     event-chain overhead per boundary. --- *)
  let bench_k k =
    let n = 64 in
    let xv =
      Array.init (n * n) ~f:(Ll_test.cycle_flat ~dims:[| n; n |] ~modulus:13 ~offset:0. ~stride:1.)
    in
    let x0 = TDSL.ndarray xv ~label:[ "x0" ] ~output_dims:[ n; n ] () in
    let rec chain t i =
      if i = 0 then t
      else
        let t' = TDSL.einsum1 "ij=>ji" t () in
        Train.set_materialized t'.Tensor.value;
        chain t' (i - 1)
    in
    let last = chain x0 k in
    let comp = Train.forward last in
    let ctx = Context.auto () in
    let ctx, routine = Context.compile ctx comp Ir.Indexing.Empty in
    for _ = 1 to 20 do
      ignore (Context.run ctx routine : Context.t)
    done;
    Context.sync ctx;
    let reps = 200 in
    let t0 = Unix.gettimeofday () in
    for _ = 1 to reps do
      let c = Context.run ctx routine in
      Context.sync c
    done;
    let ms = (Unix.gettimeofday () -. t0) /. Float.of_int reps *. 1000. in
    printf "transpose-chain K=%-2d: %.3f ms per run+sync\n" k ms
  in
  List.iter [ 1; 2; 4; 8; 12; 16 ] ~f:bench_k
