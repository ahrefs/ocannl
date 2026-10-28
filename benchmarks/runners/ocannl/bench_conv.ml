(* OCANNL conv runner: the conv workloads (LeNet-5, the cifar-scale two-conv classifiers, the
   reduced ResNet), built from the idiomatic nn_blocks pieces by Bench_conv_model. Fixture weights
   are injected into the block-created inline params via Context.set_values, matched by debug-name
   tokens (see Bench_harness.inject). Layouts documented in gen_fixtures.py build_conv. *)

open Base
open Ocannl
module IDX = Train.IDX
open Nn_blocks.DSL_modules
module Asgns = Ir.Assignments
module St = Safetensors
module H = Bench_harness

let cross_entropy_loss = Nn_blocks.cross_entropy_loss

let () =
  let fixture = Stdlib.Sys.getenv "BENCH_FIXTURE" in
  let tune = H.env_flag "BENCH_TUNE" in
  H.install_timing_trace ();
  H.install_search_cut ();
  let materialize = H.env_flag "BENCH_MATERIALIZE" in
  let debug = H.env_flag "BENCH_DEBUG" in
  let st = St.read fixture in
  let batch_size = H.meta_int st "batch_size" in
  let lr = Float.of_string (H.get_meta st "lr") in
  let x_nd = St.to_ndarray st "x" in
  let y_nd = St.to_ndarray st "y" in
  let total = (Ir.Ndarray.dims x_nd).(0) in
  let n_batches = total / batch_size in
  let xs = TDSL.rebatch ~l:"xs" x_nd () in
  let ys = TDSL.rebatch ~l:"ys" y_nd () in
  let batch_n, bindings = IDX.get_static_symbol ~static_range:n_batches IDX.empty in
  let%op batch_x = xs @| batch_n in
  let%op batch_y = ys @| batch_n in
  let { Bench_conv_model.model; mapping } = Bench_conv_model.build st in
  let logits = model batch_x in
  let%op batch_loss =
    cross_entropy_loss ~spec:"...|v" ~normalize_by:!..batch_size () ~logits ~targets:batch_y
  in
  if materialize then Train.every_non_literal_materialized batch_loss;
  let update = Train.grad_update batch_loss in
  let learning_rate = TDSL.O.( !. ) lr in
  let sgd = Train.sgd_update ~learning_rate batch_loss in
  let step_comp = Asgns.sequence [ update; sgd ] in
  let ctx = Context.auto () in
  let backend = Context.backend_name ctx in
  let ctx = Train.init_params ctx bindings batch_loss in
  let swap_blocks = H.weight_swap_blocks () in
  (* The init draw, kept for the weight-swap diagnostic before the fixture overwrites it. *)
  let init_draw = Option.map swap_blocks ~f:(fun _ -> H.param_values ctx batch_loss) in
  if debug then (
    H.dump_params batch_loss;
    Stdlib.exit 0);
  let ctx = H.inject ctx st batch_loss mapping in
  let t0 = Unix.gettimeofday () in
  (* Both placement arms' crowned candidates go into the emitted result (gh-ocannl-546). *)
  let arms = H.tune_arms () in
  let ctx, routine =
    if tune then
      (* On the fixture's weights, so the crowns are times of the data the steps run on
         (gh-ocannl-1557). *)
      let scratch = H.tuning_scratch st batch_loss mapping bindings in
      (* Placement A/B: tune the default (virtual + promotion) graph and the materialize-all graph,
         keep the measured winner. *)
      (* [flip_report] only counts the gh-555 refinement searches toward the result line's
         [searched] (gh-ocannl-644): they run whenever [tune_inline_flips] is configured, whether
         or not a callback is wired, and a flip search loads this process like an arm search. *)
      (* [~rounds:0]: no beam round whatever [autotune_rounds] says; see bench_gpt.ml for the
         measurement behind it (gh-ocannl-1229) and the open decision (gh-ocannl-1346). *)
      Train.tune_placements ~report:(H.collect_arm arms) ~flip_report:(H.collect_search arms)
        ~on_ship:(H.collect_ship arms) ~rounds:0 ~timing_ctx:scratch ctx batch_loss step_comp
        bindings
    else
      (* The recipes' untuned compile: the model pick under [model_default_schedule=true]
         (gh-ocannl-491), the seeded default otherwise (gh-ocannl-1324). *)
      Autotune.untuned_compile ctx step_comp bindings
  in
  (* What the timed artifact emitted, off the routine itself (gh-ocannl-626): a flip refinement or a
     timing_ctx replay fallback ships something no arm report describes. *)
  H.collect_shipped arms (H.Plain routine);
  H.trace_crowns arms;
  let compile_s = Unix.gettimeofday () -. t0 in
  H.trace_search_done ~compile_s;
  (* Autotune's timing context re-ran param inits on [ctx]; restore fixture weights. *)
  let ctx = if tune then H.inject ctx st batch_loss mapping else ctx in
  let batch_ref = IDX.find_exn routine.Context.bindings batch_n in
  (match (swap_blocks, init_draw) with
  | Some blocks, Some init_draw ->
      (* gh-ocannl-1622: the same routine on the init draw and on the fixture's weights, in one
         process; see [Bench_harness.weight_swap]. *)
      let steps_per_block =
        Option.value_map ~default:20 ~f:Int.of_string
          (Stdlib.Sys.getenv_opt "BENCH_WEIGHT_SWAP_STEPS")
      in
      H.weight_swap ~blocks ~steps_per_block ~ctx ~routine
        ~sets:[ ("init", init_draw); ("fixture", H.param_values ctx batch_loss) ]
        ~prepare_step:(fun ~step -> batch_ref := step % n_batches);
      Stdlib.exit 0
  | _ -> ());
  let step_count = ref 0 in
  let run_step () =
    batch_ref := !step_count % n_batches;
    Train.run ctx routine;
    Int.incr step_count
  in
  let open Operation.At in
  ignore
    (H.measure_and_emit ~routines:(H.Plain routine) ~protocol:(H.protocol_of_st ~fixture st)
       ~backend
       ~variant:(if tune then "tuned" else if materialize then "materialized" else "default")
       ~storage_of:logits ~compile_s ~tune:arms
       ~dominant_kernel:(fun () -> H.dominant_kernel ~ctx ~bindings [ routine ])
       ~run_step
       ~read_loss:(fun () -> (ctx, batch_loss).@[0])
       ~sync:(fun () -> Context.sync ctx)
       ()
      : string)
