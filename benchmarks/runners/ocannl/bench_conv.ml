(* OCANNL conv runner: LeNet-5 with valid convolutions, built from the idiomatic nn_blocks pieces
   (conv2d ~use_padding:false, max_pool2d, mlp_layer). Fixture weights are injected into the
   block-created inline params via Context.set_values, matched by debug-name tokens (see
   Bench_harness.inject). Layouts documented in gen_fixtures.py build_conv. *)

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
  let materialize = H.env_flag "BENCH_MATERIALIZE" in
  let debug = H.env_flag "BENCH_DEBUG" in
  let st = St.read fixture in
  let batch_size = H.meta_int st "batch_size" in
  let lr = Float.of_string (H.get_meta st "lr") in
  let c1 = H.meta_int st "channels1" in
  let c2 = H.meta_int st "channels2" in
  let k = H.meta_int st "kernel_size" in
  let fc1_dim = H.meta_int st "fc1" in
  let fc2_dim = H.meta_int st "fc2" in
  (* Same-padding keeps the spatial extent divisible; valid convs (LeNet) shrink it. The flag
     applies to BOTH convs — gen_fixtures sizes fc1_w as if both preserve extent, and the Python
     runners pad both. Defaults to valid so pre-existing fixtures are unchanged. *)
  let use_padding =
    match List.Assoc.find (St.metadata st) ~equal:String.equal "use_padding" with
    | Some "true" -> true
    | _ -> false
  in
  (* Per-conv strides (cifar_stride's stride-2 stem, gh-ocannl-502); valid-only, see gen_fixtures.
     Absent on pre-existing fixtures. *)
  let s1 = H.meta_int_default st "stride1" ~default:1 in
  let s2 = H.meta_int_default st "stride2" ~default:1 in
  let x_nd = St.to_ndarray st "x" in
  let y_nd = St.to_ndarray st "y" in
  let total = (Ir.Ndarray.dims x_nd).(0) in
  let n_batches = total / batch_size in
  let xs = TDSL.rebatch ~l:"xs" x_nd () in
  let ys = TDSL.rebatch ~l:"ys" y_nd () in
  let batch_n, bindings = IDX.get_static_symbol ~static_range:n_batches IDX.empty in
  let%op batch_x = xs @| batch_n in
  let%op batch_y = ys @| batch_n in
  let conv1 =
    Nn_blocks.conv2d ~label:[ "conv1" ] ~kernel_size:k ~stride:s1 ~use_padding ~out_channels:c1 ()
  in
  let conv2 =
    Nn_blocks.conv2d ~label:[ "conv2" ] ~kernel_size:k ~stride:s2 ~use_padding ~out_channels:c2 ()
  in
  let pool1 = Nn_blocks.max_pool2d ~stride:2 ~window_size:2 () in
  let pool2 = Nn_blocks.max_pool2d ~stride:2 ~window_size:2 () in
  let fc1 = Nn_blocks.mlp_layer ~label:[ "fc1" ] ~hid_dim:fc1_dim () in
  let fc2 = Nn_blocks.mlp_layer ~label:[ "fc2" ] ~hid_dim:fc2_dim () in
  let%op model x =
    ({ w_logits } * fc2 (fc1 (pool2 (relu (conv2 (pool1 (relu (conv1 x)))))))) + { b_logits = 0. }
  in
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
  if debug then (
    H.dump_params batch_loss;
    Stdlib.exit 0);
  let mapping =
    [
      ("conv1_kernel", [ "conv1"; "kernel" ]);
      ("conv1_bias", [ "conv1"; "bias" ]);
      ("conv2_kernel", [ "conv2"; "kernel" ]);
      ("conv2_bias", [ "conv2"; "bias" ]);
      ("fc1_w", [ "fc1"; "w" ]);
      ("fc1_b", [ "fc1"; "b" ]);
      ("fc2_w", [ "fc2"; "w" ]);
      ("fc2_b", [ "fc2"; "b" ]);
      ("w_logits", [ "w"; "logits" ]);
      ("b_logits", [ "b"; "logits" ]);
    ]
  in
  let ctx = H.inject ctx st batch_loss mapping in
  let t0 = Unix.gettimeofday () in
  (* Both placement arms' crowned candidates go into the emitted result (gh-ocannl-546). *)
  let arms = H.tune_arms () in
  let ctx, routine =
    if tune then
      let scratch = Train.init_params (Context.auto ()) bindings batch_loss in
      (* Placement A/B: tune the default (virtual + promotion) graph and the materialize-all graph,
         keep the measured winner. *)
      (* [flip_report] only counts the gh-555 refinement searches toward the result line's
         [searched] (gh-ocannl-644): they run whenever [tune_inline_flips] is configured, whether
         or not a callback is wired, and a flip search loads this process like an arm search. *)
      (* [~rounds:0]: no beam round whatever [autotune_rounds] says; see bench_gpt.ml for the
         measurement behind it (gh-ocannl-1137). *)
      Train.tune_placements ~report:(H.collect_arm arms) ~flip_report:(H.collect_search arms)
        ~on_ship:(H.collect_ship arms) ~rounds:0 ~timing_ctx:scratch ctx batch_loss step_comp
        bindings
    else if Lazy.force Autotune.model_default_enabled then
      (* gh-ocannl-491: the model-picked untuned default (config [model_default_schedule=true]). *)
      Autotune.model_default ctx step_comp bindings
    else Context.compile ctx step_comp bindings
  in
  (* What the timed artifact emitted, off the routine itself (gh-ocannl-626): a flip refinement or a
     timing_ctx replay fallback ships something no arm report describes. *)
  H.collect_shipped arms (H.Plain routine);
  let compile_s = Unix.gettimeofday () -. t0 in
  H.trace_search_done ~compile_s;
  (* Autotune's timing context re-ran param inits on [ctx]; restore fixture weights. *)
  let ctx = if tune then H.inject ctx st batch_loss mapping else ctx in
  let batch_ref = IDX.find_exn routine.Context.bindings batch_n in
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
       ~compile_s ~tune:arms
       ~dominant_kernel:(fun () -> H.dominant_kernel ~ctx ~bindings [ routine ])
       ~run_step
       ~read_loss:(fun () -> (ctx, batch_loss).@[0])
       ~sync:(fun () -> Context.sync ctx)
       ()
      : string)
