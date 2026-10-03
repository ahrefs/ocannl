(* OCANNL GPT-2-style runner: pre-LN decoder blocks built from the idiomatic nn_blocks pieces
   (multi_head_attention, layer_norm — fixture weights injected by name) and a tanh-gelu FFN from
   fixture-backed weight tensors. Token embedding is the logical one-hot gather (gh-343); the
   lm_head is tied to wte via an einsum that reads it transposed. Layouts documented in
   gen_fixtures.py build_gpt.

   The step shape follows the fixture's [mode] metadata, like the Python runners:

   - [mode: infer] (gpt2_mini) — forward-only; the parity metric is softmax-CE of the logits against
   fixture target ids, recorded per batch with no updates. - [mode: train] (gpt2_mini_train,
   gh-ocannl-551) — every weight is a parameter, the step is backprop plus plain SGD, and the whole
   gh-ocannl-492 task-5 gate-cost family (BENCH_PRECISION with BENCH_STATIC_SCALE /
   BENCH_GATE_INTERVAL) applies, since the workload now has an optimizer and hence a loss scale to
   gate. The flags themselves live in Bench_harness, shared with bench_mlp. *)

open Base
open Ocannl
module IDX = Train.IDX
module St = Safetensors
module H = Bench_harness

let () =
  let fixture = Stdlib.Sys.getenv "BENCH_FIXTURE" in
  let tune = H.env_flag "BENCH_TUNE" in
  H.install_timing_trace ();
  let materialize = H.env_flag "BENCH_MATERIALIZE" in
  let debug = H.env_flag "BENCH_DEBUG" in
  let st = St.read fixture in
  let {
    Bench_gpt_model.ctx;
    batch_loss;
    step_shape;
    bindings;
    batch_n;
    n_batches;
    batch_size;
    seq;
    leg;
    mapping;
  } =
    Bench_gpt_model.prepare ~materialize ~debug st
  in
  let backend = Context.backend_name ctx in
  let t0 = Unix.gettimeofday () in
  (* Placement A/B: tune the default (virtual + promotion) graph and the materialize-all graph,
     keep the measured winner. Tuning runs on a scratch lineage, so the repeated candidate
     executions never touch the benchmark's own weights. *)
  (* Both placement arms' crowned candidates go into the emitted result (gh-ocannl-546). *)
  let arms = H.tune_arms () in
  let tuned ctx comp =
    let scratch = Train.init_params (Context.auto ()) bindings batch_loss in
    (* [flip_report] only counts the gh-555 refinement searches toward the result line's [searched]
       (gh-ocannl-644): they run whenever [tune_inline_flips] is configured, whether or not a
       callback is wired, and a flip search loads this process like an arm search. *)
    Train.tune_placements ~report:(H.collect_arm arms) ~flip_report:(H.collect_search arms)
      ~on_ship:(H.collect_ship arms) ~rounds:0 ~timing_ctx:scratch ctx batch_loss comp bindings
  in
  let ctx, routines =
    match step_shape with
    | `Train parts -> H.compile_train_step ~tune ~tuned ctx bindings parts
    | `Forward fwd ->
        let ctx, routine =
          if tune then tuned ctx fwd
          else if Lazy.force Autotune.model_default_enabled then
            (* gh-ocannl-491: the model-picked untuned default (config
               [model_default_schedule=true]). *)
            Autotune.model_default ctx fwd bindings
          else Context.compile ctx fwd bindings
        in
        (ctx, H.Plain routine)
  in
  (* What the timed artifact emitted, off the routines themselves (gh-ocannl-626): a flip refinement
     or a timing_ctx replay fallback ships something no arm report describes. *)
  H.collect_shipped arms routines;
  let compile_s = Unix.gettimeofday () -. t0 in
  H.trace_search_done ~compile_s;
  let ctx = if tune then H.inject ctx st batch_loss mapping else ctx in
  (* The scaled training legs thread the context (Loss_scaler.update overwrites the scale tensors),
     hence the reference. *)
  let ctx_ref = ref ctx in
  let batch_ref = IDX.find_exn (H.train_step_bindings routines) batch_n in
  let step_count = ref 0 in
  let run_step () =
    batch_ref := !step_count % n_batches;
    H.run_train_step routines ctx_ref ~step:!step_count;
    Int.incr step_count
  in
  let open Operation.At in
  ignore
    (H.measure_and_emit ~protocol:(H.protocol_of_st st) ~backend
       ~variant:
         (* Mirror bench_mlp: the scheduling variant alone. orchestrate renders precision as its own
            report column and composes the two axes itself (gh-ocannl-539), so a reduced-precision
            cell is distinguished by the precision field rather than by overloading this one. *)
         (if tune then "tuned" else if materialize then "materialized" else "default")
       ~precision:leg.H.label ~compile_s ~tokens_per_step:(batch_size * seq) ~tune:arms
       ~dominant_kernel:(fun () ->
         H.dominant_kernel ~ctx:!ctx_ref ~bindings (H.step_routines routines))
       ~run_step
       ~read_loss:(fun () -> (!ctx_ref, batch_loss).@[0])
       ~sync:(fun () -> Context.sync !ctx_ref)
       ()
      : string)
