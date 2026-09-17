(* OCANNL runner for the cross-framework benchmark suite (see benchmarks/README.md).

   Reads a self-describing safetensors fixture (initial weights, dataset, and hyperparameters in the
   metadata map), builds an n-layer relu MLP with softmax cross-entropy loss, and trains with plain
   SGD. Prints a single JSON result line on stdout: parity losses for the first [parity_steps]
   steps, then steady-state per-step wall times (per-step synced percentiles and queued mean).

   The backend is selected the usual OCANNL way (e.g. [--ocannl_backend=metal]). Environment:
   BENCH_FIXTURE is the fixture path; BENCH_TUNE=1 enables the autotuned variant
   ([Train.tune_placements]: placement A/B — the default virtual-plus-promotion graph and the
   materialize-all graph are both tuned and the measured winner kept).

   BENCH_PRECISION=bf16|f16 (gh-ocannl-492 task 4) trains under the mixed-precision recipe: fixture
   weights stay f32 masters with reduced-precision cast twins feeding the graph
   (Mixed_prec.with_master_weights), activations and gradients storage-assigned to the reduced
   precision over the logits subtree (Precision_policy.apply — the softmax-CE loss head stays f32),
   and — f16 only — dynamic loss scaling: the step becomes a gradient routine plus an optimizer
   routine gated on the host-read gradient checksum, so the reported step times include the per-step
   device sync that the inf/nan gate costs. Composing BENCH_PRECISION with BENCH_NO_SGD is not
   supported.

   BENCH_PRECISION composes with BENCH_TUNE (gh-ocannl-529): under a reduced precision it is the
   operand storage precision that decides whether a tensorized candidate is seeded at all, and on
   RDNA3/3.5 — whose WMMA has no f32-input shape — bf16 is the only tensor-core route there is. On
   the dynamic-loss-scaling legs only the gradient (or fused) routine is tuned; the step shape is
   what those legs measure, so the small optimizer routine keeps its plain compile.

   The gh-ocannl-492 task-5 gate-cost experiment legs (f16 only): BENCH_STATIC_SCALE=1 uses a fixed
   loss scale with no gate and no host read (the discriminating experiment — if f16's penalty
   collapses toward bf16's under it, the dynamic gate is confirmed as the GPU-side cost);
   BENCH_GATE_INTERVAL=N uses the fused on-device gate (Mixed_prec.gated_scaled_update) with the
   host sampling the sticky window checksum every N steps. The reported "precision" field becomes
   f16-static / f16-gatedN respectively. Since gh-ocannl-551 the flag parsing and the step shapes
   they select live in Bench_harness, shared with every other training runner. *)

open Base
open Ocannl
module IDX = Train.IDX
open Nn_blocks.DSL_modules
module St = Safetensors
module H = Bench_harness

let cross_entropy_loss = Nn_blocks.cross_entropy_loss

(* `bench_mlp --self-test` (gh-ocannl-702): the fixture-free smoke run of the measurement path. It
   fabricates a tiny model in memory, so it needs neither benchmarks/fixtures/ — empty in a fresh
   checkout — nor the Python environment that fills it, and it prints a real result line for a
   workload named [selftest-tiny], which is deliberately NOT a comparable cell. The mode lives in
   Bench_harness, so #720's legs can offer it from any runner; it is wired up here alone for now,
   and test/operations/bench_self_test is what runs it in CI. *)
let self_test_requested () = Array.exists Stdlib.Sys.argv ~f:(String.equal "--self-test")

let () =
  if self_test_requested () then (
    ignore (H.run_self_test () : string);
    Stdlib.exit 0);
  let fixture = Stdlib.Sys.getenv "BENCH_FIXTURE" in
  let tune = H.env_flag "BENCH_TUNE" in
  let st = St.read fixture in
  let leg = H.precision_leg ~runner:"bench_mlp" ~training:(H.is_training st) ~st () in
  let mp_prec = leg.H.prec in
  let meta = St.metadata st in
  let get k = List.Assoc.find_exn meta ~equal:String.equal k in
  let n_layers = Int.of_string (get "n_layers") in
  let batch_size = Int.of_string (get "batch_size") in
  let lr = Float.of_string (get "lr") in
  let x_nd = St.to_ndarray st "x" in
  let y_nd = St.to_ndarray st "y" in
  let total = (Ir.Ndarray.dims x_nd).(0) in
  let n_batches = total / batch_size in
  let no_slice =
    match Stdlib.Sys.getenv_opt "BENCH_NO_SLICE" with Some "1" -> true | _ -> false
  in
  let xs = TDSL.rebatch ~l:"xs" x_nd () in
  let ys = TDSL.rebatch ~l:"ys" y_nd () in
  let batch_x, batch_y, batch_n, bindings =
    if no_slice then (
      assert (n_batches = 1);
      (xs, ys, None, IDX.empty))
    else
      let batch_n, bindings = IDX.get_static_symbol ~static_range:n_batches IDX.empty in
      let%op batch_x = xs @| batch_n in
      let%op batch_y = ys @| batch_n in
      (batch_x, batch_y, Some batch_n, bindings)
  in
  let make_params () =
    List.init n_layers ~f:(fun idx ->
        let i = idx + 1 in
        let w_name = Printf.sprintf "w%d" i in
        let b_name = Printf.sprintf "b%d" i in
        let info = Option.value_exn ~here:[%here] (St.info st w_name) in
        let din, dout =
          match info.St.shape with
          | [ o; i_ ] -> (i_, o)
          | _ -> failwith ("bench_mlp: weight is not a matrix: " ^ w_name)
        in
        let w = TDSL.wrap_param ~l:w_name ~i:[ din ] ~o:[ dout ] (St.to_ndarray st w_name) () in
        let b = TDSL.wrap_param ~l:b_name ~o:[ dout ] (St.to_ndarray st b_name) () in
        (w, b))
  in
  (* Under a reduced precision the fixture weights stay f32 masters (the sgd targets); the graph
     reads their cast twins. wrap_param routes through Tensor.param, so the hook applies. *)
  let params =
    match mp_prec with
    | Some prec ->
        let placement =
          match Stdlib.Sys.getenv_opt "BENCH_TWIN_PLACEMENT" with
          | Some "materialized" -> Mixed_prec.Twin_materialized
          | Some "virtual" -> Mixed_prec.Twin_virtual
          | _ -> Mixed_prec.Twin_auto
        in
        Mixed_prec.with_master_weights ~placement ~prec make_params
    | None -> make_params ()
  in
  (* Under the mixed-precision recipe the postprocess hook has replaced each parameter with its cast
     twin, so these ARE the twin nodes. BENCH_PRESEED_TWINS=1 hands them to
     [Context.decide_materialized] before tuning (gh-ocannl-558's other route to site-targeted
     materialization): a context-level decision, unlike BENCH_TWIN_PLACEMENT=materialized, which
     declares tnode-level intent and so is invisible to the placement A/B and to the gh-555 flip
     chain. Empty without a reduced precision, where there are no twins. *)
  let twin_tns =
    match mp_prec with
    | None -> []
    | Some _ -> List.concat_map params ~f:(fun (w, b) -> [ w.Tensor.value; b.Tensor.value ])
  in
  let materialize =
    match Stdlib.Sys.getenv_opt "BENCH_MATERIALIZE" with Some "1" -> true | _ -> false
  in
  let logits =
    List.foldi params ~init:batch_x ~f:(fun idx acc (w, b) ->
        let last = Int.equal idx (n_layers - 1) in
        let open TDSL.O in
        let z = b + (w * acc) in
        (* Explicit variant: store pre-activations for the backward pass (what the other frameworks
           do) instead of the default fully-Virtual recompute-in-backward. *)
        if materialize then Train.set_materialized z.Tensor.value;
        if last then z else relu z)
  in
  let%op batch_loss =
    cross_entropy_loss ~spec:"...|v" ~normalize_by:!..batch_size () ~logits ~targets:batch_y
  in
  (* Storage-precision policy: activations and gradients of the MLP body (the logits subtree) go
     reduced; the softmax-CE loss head and its gradients are PINNED at f32 via [except] — the
     "softmax and losses stay f32" AMP default. Merely not assigning the head is not enough:
     bottom-up precision inference would promote it to the reduced precision from the logits operand
     (e.g. the stable-softmax max_logits, whose -inf init the f16 constant guard rejects). Masters
     and twins already carry Specified precisions and are untouched. *)
  Option.iter mp_prec ~f:(fun prec ->
      let body = Hash_set.create (module Int) in
      let rec walk t =
        if not (Hash_set.mem body t.Tensor.value.Ir.Tnode.id) then (
          Hash_set.add body t.Tensor.value.Ir.Tnode.id;
          Option.iter t.Tensor.diff ~f:(fun d -> Hash_set.add body d.Tensor.grad.Ir.Tnode.id);
          List.iter t.Tensor.children ~f:(fun c -> walk c.Tensor.subtensor))
      in
      walk logits;
      let except tn = not (Hash_set.mem body tn.Ir.Tnode.id) in
      Precision_policy.apply ~except
        { Precision_policy.param_prec = None; activation_prec = Some prec; grad_prec = Some prec }
        batch_loss);
  let debug = H.env_flag "BENCH_DEBUG" in
  let learning_rate = TDSL.O.( !. ) lr in
  let no_sgd = H.env_flag "BENCH_NO_SGD" in
  if no_sgd && Option.is_some mp_prec then
    failwith "bench_mlp: BENCH_NO_SGD with BENCH_PRECISION is not supported";
  (* The step shape is the leg's business (Bench_harness.train_step_parts): the f16 default is a
     gradient routine plus an optimizer routine gated on the host-read gradient checksum (dynamic
     loss scaling), the other legs are a single fused routine. Note: grad_update consumes the loss's
     forward code, so exactly one shape may be built. *)
  let parts = H.train_step_parts ~setup_for_parallel:debug ~no_sgd ~leg ~learning_rate batch_loss in
  let ctx = Context.auto () in
  let backend = Context.backend_name ctx in
  let ctx = Train.init_params ctx bindings batch_loss in
  let t0 = Unix.gettimeofday () in
  (* Both placement arms' outcomes go into the emitted result (gh-ocannl-546); BENCH_TUNE_REPORT=1
     additionally prints each arm's full search report on stderr. *)
  let arms = H.tune_arms () in
  let verbose_report =
    match Stdlib.Sys.getenv_opt "BENCH_TUNE_REPORT" with Some "1" -> true | _ -> false
  in
  (* The same field dump for both callbacks. [tune_placements]'s [report] is positional (arm A then
     arm B) and its [flip_report] carries the gh-555 refinement searches, so a flip arm must NOT go
     into [arms] — [tune_json]'s attribution rule names arms by arrival order. Printed under a
     distinct tag instead, which is all the flip chain needs to be readable (gh-ocannl-558). *)
  let print_report tag (r : Autotune.report) =
    if verbose_report then
      let declines =
        String.concat ~sep:","
          (List.map r.declines ~f:(fun d ->
               Sexp.to_string (Ir.Schedule_outcome.sexp_of_rejection_key d.key)
               ^ "=" ^ Int.to_string d.count))
      in
      let terminal =
        Option.value_map (Autotune.terminal_failure r) ~default:"none" ~f:(fun failure ->
            Sexp.to_string (Ir.Schedule_outcome.sexp_of_phase failure.phase)
            ^ Option.value_map failure.candidate ~default:"" ~f:(fun candidate -> ":" ^ candidate))
      in
      Stdlib.Printf.eprintf
        "%s: state=%s timed=%d failed=%d declines=[%s] terminal=%s rounds=%d sketch=%d \
         mma_candidates=%d mma_timed=%d model_pruned=%d bound_pruned=%d fissioned=%b timing=%s \
         baseline_ms=%.4f default_ms=%s best_ms=%.4f best=%s tensorized=%b tensorization=%s \
         mma_statements=%d mma_scalar_fallbacks=%d mma_best_ms=%s\n\
         %!"
        tag (Autotune.outcome_name r.outcome) r.candidates_timed r.candidates_failed declines
        terminal r.rounds_run r.sketch_candidates r.mma_candidates r.mma_timed r.model_pruned
        r.bound_pruned r.fissioned (Autotune.timing_string r.timing) r.baseline_ms
        (Option.value_map r.default_ms ~default:"none" ~f:(Printf.sprintf "%.4f"))
        r.best_ms
        (if String.is_empty r.best_label then "none" else r.best_label)
        r.best_tensorized
        (Option.value_map r.best_tensorization ~default:"none" ~f:Ir.C_syntax.tensorization_name)
        r.best_mma_statements r.best_mma_scalar_fallbacks
        (if Float.is_inf r.mma_best_ms then "none" else Printf.sprintf "%.4f" r.mma_best_ms)
  in
  let report =
    Some
      (fun (r : Autotune.report) ->
        H.collect_arm arms r;
        print_report "tune arm" r)
  in
  let flip_report =
    Some
      (fun (r : Autotune.report) ->
        (* Not an arm — but a flip refinement that actually searched leaves this process just as
           loaded as an arm search, so it counts toward the result line's [searched]
           (gh-ocannl-644). *)
        H.collect_search arms r;
        print_report "tune flip" r)
  in
  (* BENCH_TUNE composes with every precision leg (gh-ocannl-529). It used to be rejected outright
     with BENCH_PRECISION, which made bf16 unmeasurable under autotuning — and bf16 is the ONLY
     tensor-core route on RDNA3/3.5, whose WMMA has no f32-input shape, so whether HIP even seeds a
     tensorized candidate could not be asked. (Which candidates are seeded depends on the operand
     AND accumulator storage precisions, via mma_input_formats_of_prec / mma_acc_format_of_prec
     against the backend's mma_format_tiles — gh-ocannl-545, where keying on the operands alone had
     CUDA seeding and timing 20 bf16 candidates that every emission rendered as scalar.)

     Each leg tunes the routine that carries the work. The dynamic-loss-scaling legs keep their
     step SHAPE — the gate is what is being measured — so only the gradient/fused routine is tuned
     and the tiny optimizer routine is compiled plainly, from the tuned context so the lineage's
     compile order is unchanged. Tuning runs on a scratch lineage ([timing_ctx]), so the repeated
     candidate executions never touch the benchmark's own weights or the host-side scaler state. *)
  (* BENCH_FLIP_DUMP=1 prints the whole gh-555 inlining decision surface of the default-placement
     compile — every candidate, not just the [tune_inline_flips] prefix the chain can afford to
     try. Asking "is this node on the surface at all" is otherwise indistinguishable from "it
     ranked below the budget" (gh-ocannl-558 part 2.1). Its own capture compile, so it observes
     the same lowering the chain reads and stays out of the searches' way. *)
  let dump_flip_candidates ctx comp =
    if H.env_flag "BENCH_FLIP_DUMP" then
      match
        Context.compile
          ~lowered_transform:(fun o ->
            List.iteri o.Ir.Low_level.flip_candidates ~f:(fun i fc ->
                Stdlib.Printf.eprintf "flip candidate %d: %s %s uid=%d prec=%s cost=%d\n%!" i
                  (match fc.Ir.Low_level.fc_flip with
                  | `Inline -> "inline"
                  | `Materialize -> "materialize"
                  | `Footprint -> "footprint")
                  (Ir.Tnode.debug_name fc.Ir.Low_level.fc_tn)
                  fc.Ir.Low_level.fc_tn.Ir.Tnode.uid
                  (Ir.Ops.prec_string
                     (Stdlib.Lazy.force fc.Ir.Low_level.fc_tn.Ir.Tnode.storage_prec))
                  fc.Ir.Low_level.fc_recompute_cost);
            [ o ])
          ctx comp bindings
      with
      | (_ : Context.t), (_ : Context.routine) -> ()
      | exception exn ->
          Stdlib.Printf.eprintf "flip candidate dump failed: %s\n%!" (Exn.to_string exn)
  in
  let preseed_twins = H.env_flag "BENCH_PRESEED_TWINS" in
  if preseed_twins && List.is_empty twin_tns then
    failwith "bench_mlp: BENCH_PRESEED_TWINS needs BENCH_PRECISION (no twins without it)";
  (* The two flags do not compose, and neither failure mode is visible in the result line, so refuse
     rather than measure something other than what was asked for. With BENCH_TWIN_PLACEMENT=virtual
     the twins carry declared [Virtual] intent, which [Context.decide_materialized] skips by
     contract — the pre-seed would be a silent no-op and the "control" would be the default arm
     again. With =materialized the intent already pins the twins in BOTH arms, so the run is no
     longer the context-level-decision-only experiment the pre-seed exists to be. *)
  if preseed_twins && Option.is_some (Stdlib.Sys.getenv_opt "BENCH_TWIN_PLACEMENT") then
    failwith
      "bench_mlp: BENCH_PRESEED_TWINS conflicts with BENCH_TWIN_PLACEMENT. The pre-seed is a \
       context-level placement decision over twins whose declared intent is left open; setting \
       intent as well either nullifies it (virtual) or applies it to both arms (materialized). \
       Unset one of the two.";
  let tuned ctx comp =
    (* Both lineages, so the timing context's placements match the compiled one's. *)
    let ctx = if preseed_twins then Context.decide_materialized ctx twin_tns else ctx in
    dump_flip_candidates ctx comp;
    let scratch = Train.init_params (Context.auto ()) bindings batch_loss in
    let scratch = if preseed_twins then Context.decide_materialized scratch twin_tns else scratch in
    Train.tune_placements ?report ?flip_report ~on_ship:(H.collect_ship arms) ~rounds:0
      ~timing_ctx:scratch ctx batch_loss comp bindings
  in
  let ctx, routines = H.compile_train_step ~tune ~tuned ctx bindings parts in
  (* What the timed artifact emitted, off the routines themselves (gh-ocannl-626): a flip refinement
     or a timing_ctx replay fallback ships something no arm report describes. *)
  H.collect_shipped arms routines;
  let compile_s = Unix.gettimeofday () -. t0 in
  (* The scaled step threads the context (Loss_scaler.update overwrites the scale tensors). *)
  let ctx_ref = ref ctx in
  let batch_ref =
    Option.map batch_n ~f:(fun bn -> IDX.find_exn (H.train_step_bindings routines) bn)
  in
  let step_count = ref 0 in
  let run_step () =
    Option.iter batch_ref ~f:(fun r -> r := !step_count % n_batches);
    H.run_train_step routines ctx_ref ~step:!step_count;
    Int.incr step_count
  in
  let open Operation.At in
  if debug then (
    run_step ();
    List.iteri params ~f:(fun idx (_w, b) ->
        let dout = (Ir.Ndarray.dims (St.to_ndarray st (Printf.sprintf "b%d" (idx + 1)))).(0) in
        let k = min dout 4 in
        Stdio.printf "b%d grad:" (idx + 1);
        for j = 0 to k - 1 do
          Stdio.printf " %.9g" (!ctx_ref, b).@%[j]
        done;
        Stdio.printf "  value after 1 step:";
        for j = 0 to k - 1 do
          Stdio.printf " %.9g" (!ctx_ref, b).@[j]
        done;
        Stdio.printf "\n");
    Stdlib.exit 0);
  ignore
    (H.measure_and_emit ~protocol:(H.protocol_of_st st) ~backend
       ~variant:
         (* Scheduling variant only: the storage precision is the "precision" field's business
            (gh-ocannl-539). They are independent axes, and folding a reduced precision into the
            variant made a tuned bf16 cell unnameable. *)
         (if tune then "tuned" else if materialize then "materialized" else "default")
       ~precision:leg.H.label ~compile_s ~tune:arms ~run_step
       ~read_loss:(fun () -> (!ctx_ref, batch_loss).@[0])
       ~sync:(fun () -> Context.sync !ctx_ref)
       ()
      : string)
