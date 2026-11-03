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
  let tune = H.env_flag "BENCH_TUNE" in
  (* The splice-segments A/B (gh-ocannl-1277): this arm's cache, written from the control's and the
     treatment's before anything else, so the listing needs no fixture. *)
  let splice = Bench_splice.from_env ~tune in
  let fixture = Stdlib.Sys.getenv "BENCH_FIXTURE" in
  H.install_timing_trace ();
  H.install_search_cut ();
  let materialize = H.env_flag "BENCH_MATERIALIZE" in
  let debug = H.env_flag "BENCH_DEBUG" in
  let st = St.read fixture in
  let {
    Bench_gpt_model.ctx;
    batch_loss;
    logits;
    step_shape;
    bindings;
    batch_n;
    n_batches;
    batch_size;
    seq;
    leg;
    mapping;
  } =
    Bench_gpt_model.prepare ~materialize st
  in
  if debug then (
    H.dump_params batch_loss;
    Stdlib.exit 0);
  let ctx = H.inject ctx st batch_loss mapping in
  let backend = Context.backend_name ctx in
  let t0 = Unix.gettimeofday () in
  (* Placement A/B: tune the default (virtual + promotion) graph and the materialize-all graph,
     keep the measured winner. Tuning runs on a scratch lineage, so the repeated candidate
     executions never touch the benchmark's own weights. *)
  (* Both placement arms' crowned candidates go into the emitted result (gh-ocannl-546). *)
  let arms = H.tune_arms () in
  let tuned ctx comp =
    (* On the fixture's weights, so the crowns are times of the data the steps run on
       (gh-ocannl-1557). *)
    let scratch = H.tuning_scratch st batch_loss mapping bindings in
    (* [flip_report] only counts the gh-555 refinement searches toward the result line's [searched]
       (gh-ocannl-644): they run whenever [tune_inline_flips] is configured, whether or not a
       callback is wired, and a flip search loads this process like an arm search. *)
    (* [~rounds:0] overrides [autotune_rounds], the profiles' 4 included; the arms' [rounds_run] on
       the result line records it. Why, from a capped correctness-grade probe on a shared rog box
       (CUDA, approximate profile, gh-ocannl-1229): a round expands the beam's head (the composite)
       in full and its playoff near-copies only on the segments where they differ from it. With
       rounds=4, arm A's round 1 timed 332 candidates (1204 under the old full-expansion rule),
       ~406 s on top of its 413 s of seeds (~+100%), and its best came in 0.35% under the seed
       winner, below the 1% a round needs to be adopted, so the arm shipped its rounds:0 winner.
       Arm B's 624-candidate round was cut off 287 candidates in, at the probe's 2100 s cap.
       Whether rounds should run here -- adopting a sub-threshold win whose replicate spread is
       tight, or not at all -- is open as gh-ocannl-1346. *)
    Train.tune_placements ~report:(H.collect_arm arms) ~flip_report:(H.collect_search arms)
      ~on_ship:(H.collect_ship arms) ~rounds:0 ~timing_ctx:scratch ctx batch_loss comp bindings
  in
  (* gh-ocannl-1675: BENCH_RECOMPILE=K is the probe of what a compile still pays when its kernels
     are in the kernel code cache (each recompile's census says how many were: a few kernels of an
     unchanged step can still miss), and of what a process's first compile pays. It times this
     process's first compile of the step, then compiles the step K more times from the context the
     step was compiled from -- fresh code values whose kernels the first compile published -- prints
     each compile's census to stderr, emits the first compile's link and pipeline seconds and the
     recompiles' medians as [bench-result] lines, and exits without measuring a step. *)
  let recompile =
    Option.map (Stdlib.Sys.getenv_opt "BENCH_RECOMPILE") ~f:(fun k ->
        (* The recompiles are untuned, so a tuned first compile would have published another
           schedule's kernels than the ones they look up. *)
        if tune then failwith "BENCH_RECOMPILE recompiles the untuned step: unset BENCH_TUNE";
        match Int.of_string_opt (String.strip k) with
        | Some k when k > 0 -> k
        | _ -> failwith ("BENCH_RECOMPILE must be a positive integer; found: " ^ k))
  in
  let pre_step_ctx = ctx in
  let (ctx, routines), first_cost =
    let compile () =
      Bench_splice.replay_guarded splice
        ~counts:(fun () ->
          (arms.H.searches, arms.H.replays, arms.H.no_searches, List.length arms.H.arm_reports))
        (fun () -> H.compile_step ~tune ~tuned ctx bindings step_shape)
    in
    if Option.is_some recompile then Ir.Compile_cost.with_census compile
    else (compile (), Ir.Compile_cost.empty)
  in
  Option.iter recompile ~f:(fun k ->
      let report label wall_s (cost : Ir.Compile_cost.t) =
        Stdio.eprintf
          "bench: %s: %.3f s (lower %.3f, transform %.3f, codegen %.3f, link %.3f, of which delta \
           %.3f and pipelines %.3f), %d of %d kernels compiled, %d reused, %d pipelines created\n\
           %!"
          label wall_s cost.lower_s cost.transform_s (Ir.Compile_cost.codegen_s cost) cost.link_s
          cost.link_delta_s cost.pipeline_s cost.kernels_compiled cost.kernels cost.kernels_reused
          cost.pipelines_created
      in
      report "first compile" (Ir.Compile_cost.total_s first_cost) first_cost;
      let costs =
        List.init k ~f:(fun i ->
            let c = Mtime_clock.counter () in
            let (recompiled, recompiled_routines), cost =
              Ir.Compile_cost.with_census (fun () ->
                  H.compile_step ~tune:false ~tuned pre_step_ctx bindings step_shape)
            in
            let wall_s = Mtime.Span.to_float_ns (Mtime_clock.count c) /. 1e9 in
            report ("recompile " ^ Int.to_string (i + 1)) wall_s cost;
            (* Leaf first: a host-gated step compiles its optimizer from the gradient routine's
               context, whose pools releasing the leaf does not free. *)
            Context.release recompiled;
            List.iter
              (List.rev (H.compiled_step_routines recompiled_routines))
              ~f:(fun (r : Context.routine) ->
                if not (phys_equal r.context pre_step_ctx) then Context.release r.context);
            (wall_s, cost))
      in
      let median xs =
        let a = List.sort xs ~compare:Float.compare |> Array.of_list in
        a.(Array.length a / 2)
      in
      let emit variant seconds =
        if Float.(seconds > 0.) then
          Bench_out.result ~variant
            ~check:(Bench_out.Unchecked "compile-cost probe: no output to check") ~seconds
      in
      emit "first_link" first_cost.link_s;
      emit "first_pipelines" first_cost.pipeline_s;
      emit "recompile_compile" (median (List.map costs ~f:fst));
      emit "recompile_link" (median (List.map costs ~f:(fun (_, c) -> c.link_s)));
      emit "recompile_link_delta" (median (List.map costs ~f:(fun (_, c) -> c.link_delta_s)));
      emit "recompile_codegen"
        (median (List.map costs ~f:(fun (_, c) -> Ir.Compile_cost.codegen_s c)));
      Bench_out.flush ();
      Stdlib.exit 0);
  (* What the timed artifact emitted, off the routines themselves (gh-ocannl-626): a flip refinement
     or a timing_ctx replay fallback ships something no arm report describes. *)
  H.collect_shipped arms routines;
  H.trace_crowns arms;
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
    (if H.env_flag "BENCH_GATE_TRACE" then
       match routines with
       | H.Host_gate (scaler, _, _, _) ->
           Stdio.eprintf "bench: gate step %d optimizer_runs=%d scale=%h\n%!" !step_count
             !H.host_gated_optimizer_runs
             (Mixed_prec.Loss_scaler.scale_value scaler)
       | _ -> ());
    Int.incr step_count
  in
  let open Operation.At in
  ignore
    (H.measure_and_emit ~routines ~protocol:(H.protocol_of_st ~fixture st) ~backend
       ~variant:
         (* Mirror bench_mlp: the scheduling variant alone. orchestrate renders precision as its own
            report column and composes the two axes itself (gh-ocannl-539), so a reduced-precision
            cell is distinguished by the precision field rather than by overloading this one. *)
         (if tune then "tuned" else if materialize then "materialized" else "default")
       ~precision:leg.H.label ~storage_of:logits ~compile_s ~tokens_per_step:(batch_size * seq)
       ~tune:arms
       ~dominant_kernel:(fun () ->
         H.dominant_kernel ~ctx:!ctx_ref ~bindings (H.step_routines routines))
       ~run_step
       ~read_loss:(fun () -> (!ctx_ref, batch_loss).@[0])
       ~sync:(fun () -> Context.sync !ctx_ref)
       ()
      : string)
